from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_action_csv(path: Path) -> Tuple[str, int, Dict[str, float], List[Dict[str, int]]]:
    """Return model, size, per-dim accuracy, and per-record parsed labels/preds."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    model = rows[0][1]
    size = int(rows[1][1])

    accuracies: Dict[str, float] = {}
    for row in rows:
        if len(row) >= 2 and row[0] in {"straight", "pan", "jump"}:
            accuracies[row[0]] = float(row[1])

    # Find header for record rows
    header_idx = None
    for idx, row in enumerate(rows):
        if row and row[0] == "index":
            header_idx = idx
            break

    record_rows: List[Dict[str, int]] = []
    if header_idx is not None:
        for row in rows[header_idx + 1 :]:
            if not row or not row[0].strip():
                continue
            try:
                record_rows.append(
                    {
                        "index": int(row[0]),
                        "straight_true": int(row[6]),
                        "straight_pred": int(row[7]),
                        "pan_true": int(row[8]),
                        "pan_pred": int(row[9]),
                        "jump_true": int(row[10]),
                        "jump_pred": int(row[11]),
                    }
                )
            except Exception:
                continue

    return model, size, accuracies, record_rows


def plot_metrics_vs_size(data: Dict[str, Dict[int, Dict[str, float]]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for dim in ["straight", "pan", "jump"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for model, size_map in data.items():
            sizes = sorted(size_map.keys())
            ys = [size_map[s].get(dim, 0.0) for s in sizes]
            ax.plot(sizes, ys, marker="o", label=model)
        ax.set_xlabel("Voxel size")
        ax.set_ylabel(f"{dim} accuracy")
        ax.set_title(f"{dim.capitalize()} accuracy vs size")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"action_recognition_{dim}_accuracy_vs_size.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved {path}")


def main() -> None:
    eval_dir = Path("evals")
    csv_paths = sorted(eval_dir.glob("*_action_recognition_eval.csv"))
    if not csv_paths:
        raise FileNotFoundError("No action recognition eval CSVs found in evals/")

    metrics_map: Dict[str, Dict[int, Dict[str, float]]] = {}
    per_record: Dict[str, Dict[int, List[Dict[str, int]]]] = {}

    for path in csv_paths:
        model, size, accuracies, records = load_action_csv(path)
        metrics_map.setdefault(model, {})[size] = accuracies
        per_record.setdefault(model, {})[size] = records

    plot_dir = Path("plots")
    plot_metrics_vs_size(metrics_map, plot_dir)


if __name__ == "__main__":
    main()
