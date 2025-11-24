from __future__ import annotations

import argparse
import json
import os
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from openai import OpenAI
from tqdm import tqdm


def load_raw(path: Path) -> Dict[str, List[Dict]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip()


def extract_grid_with_openai(raw_text: str, client: OpenAI, model: str, size: int) -> str:
    lines_needed = size * size
    system_msg = (
        "You are a strict cleaner/normalizer for Minecraft FRAME RECONSTRUCTION outputs.\n"
        "The model was asked to predict the next frame given the history. Your ONLY job is to extract that predicted grid.\n"
        "Requirements:\n"
        f"- Output exactly {lines_needed} lines ({size}x{size}x{size} flattened), each row formatted like "
        f"'|block|block|...|block|' with exactly {size} entries.\n"
        "- Preserve the order and count; no labels, no narration, no reasoning, no metadata.\n"
        "- Ignore or remove any extra text, explanations, or action notes in the input.\n"
        f"- If the grid is incomplete, fill or infer sensibly but still return {lines_needed} rows in the exact format.\n"
        "Return ONLY the cleaned grid lines and nothing else."
    )
    user_msg = (
        f"{raw_text}\n\n"
        "Task: Return ONLY the grid lines separated by newlines—no extra text. "
        f"If unsure, output your best attempt at a {size}x{size} grid."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=2048,
    )
    content = resp.choices[0].message.content or ""
    return strip_code_fences(content)


def normalize_grid_lines(text: str) -> List[str]:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" not in line:
            continue
        lines.append(line)
    return lines


def to_cube(lines: Sequence[str], size: int = 5, fill: str = "air") -> Optional[np.ndarray]:
    target_lines = size * size
    filtered = lines[:target_lines]
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


def cube_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None or a.shape != b.shape:
        return None
    total = a.size
    return float(np.sum(a == b) / total)


def bleu_no_newlines(reference: str, prediction: str) -> float:
    ref_flat = reference.replace("\n", "")
    pred_flat = prediction.replace("\n", "")
    ref_tokens = [tok for tok in ref_flat.split("|") if tok]
    pred_tokens = [tok for tok in pred_flat.split("|") if tok]
    if not ref_tokens or not pred_tokens:
        return 0.0
    smoothie = SmoothingFunction().method1
    return float(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie))


def summarize_results(rows: List[Dict[str, object]], model_name: str) -> None:
    header = f"Results for {model_name}"
    print("\n" + header)
    print("-" * len(header))
    print(f"{'idx':>5} {'BLEU':>8} {'3D Sim':>8}")
    for row in rows:
        bleu = row["bleu"]
        sim = row["similarity"]
        sim_str = f"{sim:.3f}" if sim is not None else "n/a"
        print(f"{row['index']:>5} {bleu:>8.3f} {sim_str:>8}")
    bleu_scores = [r["bleu"] for r in rows]
    sim_scores = [r["similarity"] for r in rows if r["similarity"] is not None]
    if bleu_scores:
        print(f"Mean BLEU: {np.mean(bleu_scores):.4f}")
    if sim_scores:
        print(f"Mean 3D similarity: {np.mean(sim_scores):.4f}")


def plot_distributions(
    bleu_scores: List[float],
    sim_scores: List[float],
    out_dir: Path,
    model_name: str,
    size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.hist(bleu_scores, bins=20, color="#4C72B0", alpha=0.85)
    plt.title(f"Frame Reconstruction BLEU — {model_name}")
    plt.xlabel("BLEU score")
    plt.ylabel("Count")
    bleu_path = out_dir / f"{model_name}_size{size}_frame_reconstruction_bleu_distribution.png"
    plt.tight_layout()
    plt.savefig(bleu_path, dpi=200)
    plt.close()

    if sim_scores:
        plt.figure(figsize=(6, 4))
        plt.hist(sim_scores, bins=20, color="#55A868", alpha=0.85)
        plt.title(f"Frame Reconstruction 3D Similarity — {model_name}")
        plt.xlabel("Element-wise match fraction")
        plt.ylabel("Count")
        sim_path = out_dir / f"{model_name}_size{size}_frame_reconstruction_3d_similarity_distribution.png"
        plt.tight_layout()
        plt.savefig(sim_path, dpi=200)
        plt.close()

    print(f"Saved plots to {out_dir}")


def evaluate_model(
    data: Dict[str, List[Dict]],
    *,
    model_name: str,
    client: OpenAI,
    openai_model: str,
    size: int,
    ) -> List[Dict[str, object]]:
    if model_name not in data:
        raise ValueError(f"Model '{model_name}' not found in raw file")

    rows: List[Dict[str, object]] = []
    for entry in tqdm(data[model_name], desc=f"Cleaning {model_name}", unit="record"):
        cleaned_pred = extract_grid_with_openai(entry["z_prediction"], client, openai_model, size)
        label_text = entry["z_label"]
        bleu_score = bleu_no_newlines(label_text, cleaned_pred)

        label_cube = to_cube(normalize_grid_lines(label_text), size=size)
        pred_cube = to_cube(normalize_grid_lines(cleaned_pred), size=size)
        sim_score = cube_similarity(label_cube, pred_cube)

        rows.append(
            {
                "index": entry.get("index"),
                "episode": entry.get("episode"),
                "bleu": bleu_score,
                "similarity": sim_score,
                "z_label": label_text,
                "z_clean_prediction": cleaned_pred,
            }
        )

    summarize_results(rows, model_name)
    bleu_scores = [r["bleu"] for r in rows]
    sim_scores = [r["similarity"] for r in rows if r["similarity"] is not None]
    plot_distributions(bleu_scores, sim_scores, Path("plots"), model_name, size)
    return rows


def save_metrics_xml(
    model_name: str,
    rows: List[Dict[str, object]],
    out_dir: Path,
    size: int,
) -> None:
    """Deprecated XML saver (kept for backward compatibility)."""
    save_metrics_csv(model_name, rows, out_dir, size)


def save_metrics_csv(
    model_name: str,
    rows: List[Dict[str, object]],
    out_dir: Path,
    size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_bleu = float(np.mean([r["bleu"] for r in rows])) if rows else 0.0
    sim_scores = [r["similarity"] for r in rows if r["similarity"] is not None]
    mean_sim = float(np.mean(sim_scores)) if sim_scores else 0.0

    csv_path = out_dir / f"{model_name}_size{size}_frame_reconstruction_eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", model_name])
        writer.writerow(["size", size])
        writer.writerow(["mean_bleu", f"{mean_bleu:.6f}"])
        writer.writerow(["mean_similarity", f"{mean_sim:.6f}"])
        writer.writerow([])
        writer.writerow(["index", "episode", "bleu", "similarity", "z_label", "z_clean_prediction"])
        for r in rows:
            sim_val = "" if r.get("similarity") is None else f"{r.get('similarity', 0.0):.6f}"
            writer.writerow([
                r.get("index"),
                r.get("episode"),
                f"{r.get('bleu', 0.0):.6f}",
                sim_val,
                (r.get("z_label", "") or "").replace("\n", "\\n"),
                (r.get("z_clean_prediction", "") or "").replace("\n", "\\n"),
            ])
    print(f"Saved CSV summary to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frame reconstruction raw outputs with OpenAI cleaning + BLEU/3D similarity."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=5,
        help="Voxel cube side length (used for cube parsing and output naming).",
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated model names to evaluate. Defaults to all keys in the raw JSON.",
    )
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini", help="OpenAI model for cleaning.")
    parser.add_argument("--output-json", type=Path, help="Optional path to save per-record scores.")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Please export it before running.")

    default_input = Path("results") / f"frame_reconstruction_raw_size{args.size}.json"
    raw_data = load_raw(default_input)
    selected_models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else list(raw_data.keys())
    )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    all_rows: Dict[str, List[Dict[str, object]]] = {}

    for model_name in tqdm(selected_models, desc="Models"):
        print(f"\n=== Evaluating {model_name} ===")
        rows = evaluate_model(
            raw_data,
            model_name=model_name,
            client=client,
            openai_model=args.openai_model,
            size=args.size,
        )
        all_rows[model_name] = rows

        save_metrics_csv(model_name, rows, Path("evals"), args.size)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2, ensure_ascii=False)
        print(f"\nWrote aggregated rows to {args.output_json}")


if __name__ == "__main__":
    main()
