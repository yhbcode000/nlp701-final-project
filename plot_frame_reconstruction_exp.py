from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def gaussian_kde(values: np.ndarray, num_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Simple Gaussian KDE implementation (no external deps)."""
    if values.size < 2:
        return np.array([]), np.array([])

    std = np.std(values, ddof=1)
    if std == 0:
        std = 1e-6

    bandwidth = 1.06 * std * values.size ** (-1 / 5)
    grid = np.linspace(values.min() - 1.5 * std, values.max() + 1.5 * std, num_points)

    diffs = grid[:, None] - values[None, :]
    kernel = np.exp(-0.5 * (diffs / bandwidth) ** 2) / (np.sqrt(2 * np.pi) * bandwidth)
    density = kernel.mean(axis=1)
    return grid, density


def load_recon_csv(path: Path) -> Tuple[str, int, float, float, List[Dict[str, float]]]:
    """Return model, size, mean_bleu, mean_sim, and per-record rows."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    model = rows[0][1]
    size = int(rows[1][1])
    mean_bleu = float(rows[2][1])
    mean_sim = float(rows[3][1])

    # Find header for record rows
    header_idx = None
    for idx, row in enumerate(rows):
        if row and row[0] == "index":
            header_idx = idx
            break
    record_rows: List[Dict[str, float]] = []
    if header_idx is not None:
        for row in rows[header_idx + 1 :]:
            if not row or not row[0].strip():
                continue
            try:
                record_rows.append(
                    {
                        "index": int(row[0]),
                        "bleu": float(row[2]),
                        "similarity": float(row[3]) if row[3] else np.nan,
                    }
                )
            except Exception:
                continue

    return model, size, mean_bleu, mean_sim, record_rows


def plot_metrics_vs_size(data: Dict[str, Dict[int, Dict[str, float]]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for metric, ax in zip(["mean_bleu", "mean_similarity"], axes):
        for model, size_map in data.items():
            sizes = sorted(size_map.keys())
            ys = [size_map[s][metric] for s in sizes]
            ax.plot(sizes, ys, marker="o", label=model)
        ax.set_xlabel("Voxel size")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "frame_reconstruction_metrics_vs_size.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_distributions(
    per_record: Dict[str, Dict[int, List[Dict[str, float]]]],
    out_dir: Path,
    metric: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for model, size_map in per_record.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for size, rows in sorted(size_map.items()):
            values = np.array([r[metric] for r in rows if not np.isnan(r[metric])])
            grid, density = gaussian_kde(values)
            if grid.size == 0:
                continue
            ax.plot(grid, density, label=f"size {size}")
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_ylabel("Density")
        ax.set_title(f"{metric.replace('_', ' ').title()} KDE — {model}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"{model}_{metric}_kde_by_size.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved {path}")


def main() -> None:
    eval_dir = Path("evals")
    csv_paths = sorted(eval_dir.glob("*_frame_reconstruction_eval.csv"))
    if not csv_paths:
        raise FileNotFoundError("No frame reconstruction eval CSVs found in evals/")

    metrics_map: Dict[str, Dict[int, Dict[str, float]]] = {}
    per_record: Dict[str, Dict[int, List[Dict[str, float]]]] = {}

    for path in csv_paths:
        model, size, mean_bleu, mean_sim, records = load_recon_csv(path)
        metrics_map.setdefault(model, {})[size] = {
            "mean_bleu": mean_bleu,
            "mean_similarity": mean_sim,
        }
        per_record.setdefault(model, {})[size] = records

    plot_dir = Path("plots")
    plot_metrics_vs_size(metrics_map, plot_dir)
    plot_distributions(per_record, plot_dir, metric="bleu")
    plot_distributions(per_record, plot_dir, metric="similarity")


if __name__ == "__main__":
    main()
