from __future__ import annotations

import argparse
import json
import os
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


ACTION_MAP = {
    "straight": {"forward": 1, "backward": -1, "noop": 0},
    "pan": {"right": 1, "left": -1, "noop": 0},
    "jump": {"jump": 1, "noop": 0},
}


def load_raw(path: Path) -> Dict[str, List[Dict]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip()


def extract_action_block_with_openai(raw_text: str, client: OpenAI, model: str) -> str:
    system_msg = (
        "You clean Minecraft action predictions. "
        "Return EXACTLY three lines:\n"
        "straight: [noop|forward|backward]\n"
        "pan: [noop|left|right]\n"
        "jump: [noop|jump]\n"
        "Do not add explanations or extra text."
    )
    user_msg = f"Extract and normalize the action block from this noisy prediction:\n{raw_text}"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=128,
    )
    content = resp.choices[0].message.content or ""
    return strip_code_fences(content)


def parse_action_block(text: str) -> Optional[Tuple[int, int, int]]:
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    values: Dict[str, int] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, val = [part.strip() for part in line.split(":", 1)]
        if key in ACTION_MAP and val in ACTION_MAP[key]:
            values[key] = ACTION_MAP[key][val]
            # Stop early if we have all fields; remaining noise (e.g., "Frame 2 Next: ...") is ignored.
            if all(k in values for k in ("straight", "pan", "jump")):
                break
    if "straight" in values and "pan" in values and "jump" in values:
        return values["straight"], values["pan"], values["jump"]
    return None


def compute_dimension_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels: List[int],
) -> Dict[str, object]:
    acc = float(np.mean([int(a == b) for a, b in zip(y_true, y_pred)])) if y_true else 0.0
    conf = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": acc,
        "confusion_matrix": conf.tolist(),
        "labels": labels,
    }


def plot_confusion(conf: np.ndarray, labels: List[int], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(j, i, conf[i, j], ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def evaluate_model(
    data: Dict[str, List[Dict]],
    *,
    model_name: str,
    client: OpenAI,
    openai_model: str,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if model_name not in data:
        raise ValueError(f"Model '{model_name}' not found in raw file")

    rows: List[Dict[str, object]] = []
    straight_true: List[int] = []
    straight_pred: List[int] = []
    pan_true: List[int] = []
    pan_pred: List[int] = []
    jump_true: List[int] = []
    jump_pred: List[int] = []

    for entry in tqdm(data[model_name], desc=f"Cleaning {model_name}", unit="record"):
        cleaned = extract_action_block_with_openai(entry["y_prediction"], client, openai_model)
        parsed_pred = parse_action_block(cleaned)
        parsed_label = parse_action_block(entry["y_label"])

        if parsed_label is None:
            continue  # skip malformed label entries

        if parsed_pred is None:
            parsed_pred = (0, 0, 0)  # fall back to noop if unparseable

        s_l, p_l, j_l = parsed_label
        s_p, p_p, j_p = parsed_pred

        straight_true.append(s_l)
        straight_pred.append(s_p)
        pan_true.append(p_l)
        pan_pred.append(p_p)
        jump_true.append(j_l)
        jump_pred.append(j_p)

        rows.append(
            {
                "index": entry.get("index"),
                "episode": entry.get("episode"),
                "y_label": entry["y_label"],
                "y_prediction": entry["y_prediction"],
                "y_clean_prediction": cleaned,
                "parsed_label": parsed_label,
                "parsed_prediction": parsed_pred,
            }
        )

    metrics = {
        "straight": compute_dimension_metrics(straight_true, straight_pred, labels=[-1, 0, 1]),
        "pan": compute_dimension_metrics(pan_true, pan_pred, labels=[-1, 0, 1]),
        "jump": compute_dimension_metrics(jump_true, jump_pred, labels=[0, 1]),
    }
    return rows, metrics


def save_metrics_csv(
    model_name: str,
    rows: List[Dict[str, object]],
    metrics: Dict[str, object],
    out_dir: Path,
    size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{model_name}_size{size}_action_recognition_eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", model_name])
        writer.writerow(["size", size])
        writer.writerow([])
        writer.writerow(["dimension", "accuracy"])
        for dim in ["straight", "pan", "jump"]:
            writer.writerow([dim, f"{metrics[dim]['accuracy']:.6f}"])
        writer.writerow([])
        writer.writerow([
            "index",
            "episode",
            "y_label",
            "y_clean_prediction",
            "parsed_label",
            "parsed_prediction",
            "straight_true",
            "straight_pred",
            "pan_true",
            "pan_pred",
            "jump_true",
            "jump_pred",
        ])
        for r in rows:
            parsed_label = r.get("parsed_label") or (None, None, None)
            parsed_pred = r.get("parsed_prediction") or (None, None, None)
            writer.writerow([
                r.get("index"),
                r.get("episode"),
                (r.get("y_label", "") or "").replace("\n", "\\n"),
                (r.get("y_clean_prediction", "") or "").replace("\n", "\\n"),
                str(parsed_label),
                str(parsed_pred),
                parsed_label[0],
                parsed_pred[0],
                parsed_label[1],
                parsed_pred[1],
                parsed_label[2],
                parsed_pred[2],
            ])
    print(f"Saved CSV summary to {csv_path}")


def print_metrics(metrics: Dict[str, object], model_name: str) -> None:
    print(f"\n=== Metrics for {model_name} ===")
    for dim in ["straight", "pan", "jump"]:
        dim_metrics = metrics[dim]
        acc = dim_metrics["accuracy"]
        print(f"{dim.capitalize()} accuracy: {acc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate action recognition raw outputs with OpenAI cleaning + per-dimension metrics."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=5,
        help="Voxel cube side length (used for default input naming).",
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated model names to evaluate. Defaults to all keys in the raw JSON.",
    )
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini", help="OpenAI model for cleaning.")
    parser.add_argument("--output-json", type=Path, help="Optional path to save per-record scores and metrics.")
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to write confusion matrix plots.",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Please export it before running.")

    default_input = Path("results") / f"action_recognition_raw_size{args.size}.json"
    raw_data = load_raw(default_input)
    selected_models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else list(raw_data.keys())
    )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    all_rows: Dict[str, List[Dict[str, object]]] = {}
    all_metrics: Dict[str, Dict[str, object]] = {}

    for model_name in tqdm(selected_models, desc="Models"):
        rows, metrics = evaluate_model(
            raw_data,
            model_name=model_name,
            client=client,
            openai_model=args.openai_model,
        )
        all_rows[model_name] = rows
        all_metrics[model_name] = metrics
        print_metrics(metrics, model_name)

        save_metrics_csv(model_name, rows, metrics, Path("evals"), args.size)

        # plots per dimension
        for dim, dim_metrics in metrics.items():
            conf = np.array(dim_metrics["confusion_matrix"])
            labels = dim_metrics["labels"]
            plot_confusion(
                conf,
                labels,
                title=f"{model_name} {dim} confusion",
                path=args.plots_dir / f"{model_name}_{dim}_size{args.size}_confusion.png",
            )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(
                {"rows": all_rows, "metrics": all_metrics},
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nWrote aggregated rows/metrics to {args.output_json}")


if __name__ == "__main__":
    main()
