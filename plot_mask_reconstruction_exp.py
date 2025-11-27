from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse

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


def compute_ci(values: np.ndarray) -> Tuple[float, float]:
    """Return (lower, upper) 95% CI for the mean."""
    if values.size == 0:
        return (np.nan, np.nan)
    mean = float(np.mean(values))
    if values.size < 2:
        return (mean, mean)
    std = float(np.std(values, ddof=1))
    margin = 1.96 * std / np.sqrt(values.size)
    return (mean - margin, mean + margin)


def load_mask_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _is_air_only(text: str) -> bool:
    tokens = [tok.strip() for tok in text.replace("\\n", "\n").replace("\r", "\n").split("|") if tok.strip()]
    return bool(tokens) and all(tok == "air" for tok in tokens)


def collect_metrics(eval_dir: Path, drop_air: bool) -> Tuple[Dict[str, Dict[int, Dict[str, float]]], Dict[str, Dict[int, List[Dict[str, float]]]]]:
    """Return (metrics_map, per_record) keyed by model -> mask_size."""
    csv_paths = sorted(eval_dir.glob("*_mask_reconstruction_eval.csv"))
    if not csv_paths:
        raise FileNotFoundError("No mask reconstruction eval CSVs found in evals/")

    metrics_map: Dict[str, Dict[int, Dict[str, float]]] = {}
    per_record: Dict[str, Dict[int, List[Dict[str, float]]]] = {}

    for path in csv_paths:
        rows = load_mask_csv(path)
        if not rows:
            continue
        model = rows[0]["model"]
        mask_size = int(rows[0]["mask_size"])

        filtered_rows = rows
        if drop_air:
            filtered_rows = [
                r
                for r in rows
                if not (_is_air_only(r.get("z_label", "")) and _is_air_only(r.get("z_clean_prediction", "")))
            ]
            if not filtered_rows:
                continue

        bleu_vals = [float(r["bleu"]) for r in filtered_rows if r.get("bleu")]
        sim_vals = [float(r["similarity_masked"]) for r in filtered_rows if r.get("similarity_masked")]
        mean_bleu = float(np.mean(bleu_vals)) if bleu_vals else 0.0
        mean_sim = float(np.mean(sim_vals)) if sim_vals else 0.0

        metrics_map.setdefault(model, {})[mask_size] = {
            "mean_bleu": mean_bleu,
            "mean_similarity": mean_sim,
        }
        per_record.setdefault(model, {})[mask_size] = [
            {
                "index": int(r["index"]),
                "bleu": float(r["bleu"]),
                "similarity": float(r["similarity_masked"]),
            }
            for r in filtered_rows
            if r.get("bleu") and r.get("similarity_masked")
        ]
    return metrics_map, per_record


def plot_metrics_vs_mask(metrics_map: Dict[str, Dict[int, Dict[str, float]]], per_record: Dict[str, Dict[int, List[Dict[str, float]]]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for metric, ax in zip(["mean_bleu", "mean_similarity"], axes):
        for model, mask_map in metrics_map.items():
            masks = sorted(mask_map.keys())
            means = []
            lowers = []
            uppers = []
            for m in masks:
                mean_val = mask_map[m][metric]
                values = np.array([r["bleu"] if metric == "mean_bleu" else r["similarity"] for r in per_record.get(model, {}).get(m, []) if not np.isnan(r["bleu"] if metric == "mean_bleu" else r["similarity"])])
                if values.size > 0:
                    lo, hi = compute_ci(values)
                else:
                    lo = hi = mean_val
                means.append(mean_val)
                lowers.append(lo)
                uppers.append(hi)
            ax.errorbar(
                masks,
                means,
                yerr=[np.array(means) - np.array(lowers), np.array(uppers) - np.array(means)],
                marker="o",
                label=model,
                capsize=4,
            )
        ax.set_xlabel("Mask size")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "mask_reconstruction_metrics_vs_mask.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_distributions(per_record: Dict[str, Dict[int, List[Dict[str, float]]]], out_dir: Path, metric: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for model, mask_map in per_record.items():
        # KDE
        fig, ax = plt.subplots(figsize=(6, 4))
        for mask_size, rows in sorted(mask_map.items()):
            values = np.array([r[metric] for r in rows if not np.isnan(r[metric])])
            grid, density = gaussian_kde(values)
            if grid.size == 0:
                continue
            ax.plot(grid, density, label=f"mask {mask_size}")
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_ylabel("Density")
        ax.set_title(f"{metric.replace('_', ' ').title()} KDE — {model}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"{model}_{metric}_kde_by_mask.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved {path}")

        # Box plot
        fig, ax = plt.subplots(figsize=(6, 4))
        masks = sorted(mask_map.keys())
        data = [np.array([r[metric] for r in mask_map[m] if not np.isnan(r[metric])]) for m in masks]
        ax.boxplot(
            data,
            labels=[f"mask {m}" for m in masks],
            showmeans=True,
            meanline=True,
            flierprops={"marker": "x", "markersize": 5},
        )
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} Box Plot — {model}")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        box_path = out_dir / f"{model}_{metric}_box_by_mask.png"
        fig.savefig(box_path, dpi=200)
        plt.close(fig)
        print(f"Saved {box_path}")


def plot_seaborn_kde(per_record: Dict[str, Dict[int, List[Dict[str, float]]]], out_dir: Path, metric: str) -> None:
    """Facet KDE by mask size, hue=model, using seaborn."""
    records = []
    for model, mask_map in per_record.items():
        for mask, rows in mask_map.items():
            for r in rows:
                val = r[metric]
                if np.isnan(val):
                    continue
                records.append({"model": model, "mask": mask, metric: val})
    if not records:
        return
    import pandas as pd  # local import

    df = pd.DataFrame(records)
    g = sns.FacetGrid(df, col="mask", hue="model", sharex=False, sharey=False, col_order=sorted(df["mask"].unique()))
    g.map_dataframe(sns.kdeplot, x=metric, fill=True, alpha=0.3)
    g.add_legend(title="Model")
    g.set_axis_labels(metric.replace("_", " ").title(), "Density")
    g.fig.suptitle(f"{metric.replace('_', ' ').title()} KDE by mask (seaborn)", y=1.02)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mask_reconstruction_{metric}_kde_seaborn.png"
    g.fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    print(f"Saved {out_path}")


def plot_seaborn_scatter(per_record: Dict[str, Dict[int, List[Dict[str, float]]]], out_dir: Path) -> None:
    """Scatter BLEU vs similarity with linear fit, hue=model, style=mask."""
    records = []
    for model, mask_map in per_record.items():
        for mask, rows in mask_map.items():
            for r in rows:
                if np.isnan(r["bleu"]) or np.isnan(r["similarity"]):
                    continue
                records.append({"model": model, "mask": mask, "bleu": r["bleu"], "similarity": r["similarity"]})
    if not records:
        return
    import pandas as pd

    df = pd.DataFrame(records)
    sns.set_style("whitegrid")
    g = sns.lmplot(
        data=df,
        x="bleu",
        y="similarity",
        hue="model",
        markers="o",
        scatter_kws={"alpha": 0.6, "s": 30},
        line_kws={"linewidth": 2},
        height=4,
        aspect=1.2,
    )
    g.fig.suptitle("BLEU vs Similarity (linear fit by model)", y=1.02)
    g.set_axis_labels("BLEU", "Similarity")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mask_reconstruction_bleu_vs_similarity_scatter.png"
    g.fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    print(f"Saved {out_path}")


def main() -> None:
    eval_dir = Path("evals")
    parser = argparse.ArgumentParser(description="Plot mask reconstruction evaluation distributions.")
    parser.add_argument("--drop-air", action="store_true", help="Exclude entries where both label and prediction are all 'air'.")
    args = parser.parse_args()

    metrics_map, per_record = collect_metrics(eval_dir, drop_air=args.drop_air)
    plot_dir = Path("plots")
    plot_metrics_vs_mask(metrics_map, per_record, plot_dir)
    plot_distributions(per_record, plot_dir, metric="bleu")
    plot_distributions(per_record, plot_dir, metric="similarity")
    plot_seaborn_kde(per_record, plot_dir, metric="bleu")
    plot_seaborn_kde(per_record, plot_dir, metric="similarity")
    plot_seaborn_scatter(per_record, plot_dir)


if __name__ == "__main__":
    main()
