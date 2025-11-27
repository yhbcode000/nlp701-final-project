from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from openai import OpenAI
from tqdm import tqdm


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip()


def normalize_grid_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip() and "|" in ln]


def to_cube(lines: Sequence[str], size: int, fill: str = "air") -> np.ndarray:
    target_lines = size * size
    filtered = list(lines)[:target_lines]
    filler_line = "|" + "|".join([fill] * size) + "|"
    while len(filtered) < target_lines:
        filtered.append(filler_line)

    cube = np.full((size, size, size), fill, dtype=object)
    for line_idx, line in enumerate(filtered[:target_lines]):
        tokens = [tok for tok in line.split("|") if tok]
        if len(tokens) < size:
            tokens += [fill] * (size - len(tokens))
        tokens = tokens[:size]
        x = line_idx % size
        y = line_idx // size
        for z in range(size):
            cube[x, y, z] = tokens[z]
    return cube


def cube_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    return float(np.mean(a == b))


def masked_similarity(a: np.ndarray, b: np.ndarray, mask: np.ndarray, mask_label: str = "mask") -> float:
    mask_region = mask == mask_label
    if not mask_region.any():
        return 0.0
    return float(np.mean(a[mask_region] == b[mask_region]))


def bleu_no_newlines(reference: str, prediction: str) -> float:
    ref_flat = reference.replace("\n", "")
    pred_flat = prediction.replace("\n", "")
    ref_tokens = [tok for tok in ref_flat.split("|") if tok]
    pred_tokens = [tok for tok in pred_flat.split("|") if tok]
    if not ref_tokens or not pred_tokens:
        return 0.0
    smoothie = SmoothingFunction().method1
    return float(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie))


def clean_prediction(raw_text: str, client: OpenAI, model: str, size: int, mask_size: int) -> str:
    lines_needed = size * size
    system_msg = (
        "You clean Minecraft masked-frame predictions. "
        f"Return exactly {lines_needed} lines ({size}x{size}x{size} flattened), each like '|block|...|block|'. "
        f"The masked sub-cube size is {mask_size}x{mask_size}x{mask_size}. "
        "Remove any prose or labels."
    )
    user_msg = f"{raw_text}\n\nTask: return only the cleaned grid with {lines_needed} lines."
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=2048,
    )
    return strip_code_fences(resp.choices[0].message.content or "")


def plot_distributions(scores: List[float], title: str, xlabel: str, path: Path) -> None:
    if not scores:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(scores, bins=20, color="#4C72B0", alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mask reconstruction raw outputs.")
    parser.add_argument("--input", type=Path, default=Path("results/mask_reconstruction_raw.json"))
    parser.add_argument("--size", type=int, default=7, help="Cube size of the masked input.")
    parser.add_argument("--mask-size", type=int, default=3, help="Mask cube size (for label/pred parsing).")
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--output-json", type=Path, help="Optional path to save per-record metrics.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="all",
        help="Model key to evaluate ('all' evaluates every list entry in the raw JSON).",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    data = json.loads(args.input.read_text())

    # Collect model entries
    available = {k: v for k, v in data.items() if isinstance(v, list)}
    if not available:
        raise ValueError("No model entries found in input JSON.")

    if args.model_name.lower() == "all":
        selected_keys = list(available.keys())
    else:
        requested = [m.strip() for m in args.model_name.split(",") if m.strip()]
        missing = [m for m in requested if m not in available]
        if missing:
            raise ValueError(f"Model(s) {missing} not found in input. Available: {list(available.keys())}")
        selected_keys = requested

    client = OpenAI()

    all_rows: List[Dict[str, object]] = []
    for model_key in selected_keys:
        entries = available[model_key]
        bleu_scores: List[float] = []
        sim_scores: List[float] = []
        mask_sim_scores: List[float] = []

        for entry in tqdm(entries, desc=f"Evaluating ({model_key})"):
            masked_text = entry["masked_input"]
            label_text = entry["z_label"]
            mask_text = entry.get("mask_cube", "")
            coords = entry.get("coords") or [0, 0, 0]
            try:
                sx, sy, sz = [int(c) for c in coords]
            except Exception:
                sx = sy = sz = 0
            cleaned_pred = clean_prediction(entry["z_prediction"], client, args.openai_model, args.mask_size, args.mask_size)

            label_cube = to_cube(normalize_grid_lines(label_text), args.mask_size)
            pred_cube = to_cube(normalize_grid_lines(cleaned_pred), args.mask_size)

            if mask_text:
                mask_full = to_cube(normalize_grid_lines(mask_text), args.size, fill="")
                if (
                    sx + args.mask_size <= mask_full.shape[0]
                    and sy + args.mask_size <= mask_full.shape[1]
                    and sz + args.mask_size <= mask_full.shape[2]
                ):
                    mask_cube = mask_full[sx : sx + args.mask_size, sy : sy + args.mask_size, sz : sz + args.mask_size]
                else:
                    mask_cube = np.zeros_like(label_cube, dtype=object)
            else:
                mask_cube = np.zeros_like(label_cube, dtype=object)

            bleu = bleu_no_newlines(label_text, cleaned_pred)
            sim = cube_similarity(label_cube, pred_cube)
            mask_sim = masked_similarity(label_cube, pred_cube, mask_cube, mask_label="mask")

            bleu_scores.append(bleu)
            sim_scores.append(sim)
            mask_sim_scores.append(mask_sim)

            all_rows.append(
                {
                    "model": model_key,
                    "index": entry.get("index"),
                    "episode": entry.get("episode"),
                    "cube_size": entry.get("cube_size"),
                    "mask_size": entry.get("mask_size"),
                    "bleu": bleu,
                    "similarity_full": sim,
                    "similarity_masked": mask_sim,
                    "z_label": label_text,
                    "z_clean_prediction": cleaned_pred,
                    "mask_cube": mask_text,
                    "masked_input": masked_text,
                }
            )

        mean_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0
        mean_mask_sim = float(np.mean(mask_sim_scores)) if mask_sim_scores else 0.0
        print(f"[{model_key}] Mean BLEU: {mean_bleu:.4f}")
        print(f"[{model_key}] Mean similarity (masked region): {mean_mask_sim:.4f}")

        plots_dir = Path("plots")
        plot_distributions(
            bleu_scores,
            f"Mask Reconstruction BLEU — {model_key} (size={args.size}, mask={args.mask_size})",
            "BLEU",
            plots_dir / f"{model_key}_size{args.size}_mask{args.mask_size}_mask_reconstruction_bleu.png",
        )
        plot_distributions(
            mask_sim_scores,
            f"Mask Reconstruction Similarity (masked) — {model_key} (size={args.size}, mask={args.mask_size})",
            "Similarity",
            plots_dir / f"{model_key}_size{args.size}_mask{args.mask_size}_mask_reconstruction_similarity_masked.png",
        )

        # Save CSV per model
        csv_path = Path("evals") / f"{model_key}_size{args.size}_mask{args.mask_size}_mask_reconstruction_eval.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["model", "index", "episode", "cube_size", "mask_size", "bleu", "similarity_masked", "z_label", "z_clean_prediction", "mask_cube", "masked_input"]
            )
            for r in [row for row in all_rows if row["model"] == model_key]:
                writer.writerow(
                    [
                        r["model"],
                        r["index"],
                        r["episode"],
                        r["cube_size"],
                        r["mask_size"],
                        f"{r['bleu']:.6f}",
                        f"{r['similarity_masked']:.6f}",
                        (r.get("z_label", "") or "").replace("\n", "\\n"),
                        (r.get("z_clean_prediction", "") or "").replace("\n", "\\n"),
                        (r.get("mask_cube", "") or "").replace("\n", "\\n"),
                        (r.get("masked_input", "") or "").replace("\n", "\\n"),
                    ]
                )
        print(f"Wrote CSV to {csv_path}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False))
        print(f"Wrote per-record metrics to {args.output_json}")


if __name__ == "__main__":
    main()
