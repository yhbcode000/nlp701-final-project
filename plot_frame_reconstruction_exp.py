from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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


def compute_ci(values: np.ndarray) -> Tuple[float, float]:
    """Return (lower, upper) 95% CI for the mean; falls back to std if too few points."""
    if values.size == 0:
        return (np.nan, np.nan)
    mean = float(np.mean(values))
    if values.size < 2:
        return (mean, mean)
    std = float(np.std(values, ddof=1))
    margin = 1.96 * std / np.sqrt(values.size)
    return (mean - margin, mean + margin)


MODEL_SIZE_ORDER = [0.6, 1.7, 4, 8, 14, 32]
MODEL_COLORS = sns.color_palette(None, len(MODEL_SIZE_ORDER))
MODEL_COLOR_MAP = {size: color for size, color in zip(MODEL_SIZE_ORDER, MODEL_COLORS)}


def _parse_model_size(model: str) -> float:
    """Extract numeric size (e.g., 0.6 from 'qwen3-0.6b'); inf if unknown."""
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)b", model)
    return float(match.group(1)) if match else float("inf")


def model_sort_key(model: str) -> tuple:
    size = _parse_model_size(model)
    try:
        idx = MODEL_SIZE_ORDER.index(size)
    except ValueError:
        idx = len(MODEL_SIZE_ORDER)
    return (idx, size, model)


def model_color(model: str) -> str:
    size = _parse_model_size(model)
    return MODEL_COLOR_MAP.get(size, "#4C72B0")


def model_palette(models: List[str]) -> Dict[str, str]:
    return {m: model_color(m) for m in models}


def plot_metrics_vs_size(
    data: Dict[str, Dict[int, Dict[str, float]]],
    per_record: Dict[str, Dict[int, List[Dict[str, float]]]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for metric, ax in zip(["mean_bleu", "mean_similarity"], axes):
        for model in sorted(data.keys(), key=model_sort_key):
            size_map = data[model]
            sizes = sorted(size_map.keys())
            ys = [size_map[s][metric] for s in sizes]
            # compute 95% CI from per-record
            ci_lower = []
            ci_upper = []
            for s in sizes:
                vals = np.array(
                    [
                        r["bleu"] if metric == "mean_bleu" else r["similarity"]
                        for r in per_record.get(model, {}).get(s, [])
                        if not np.isnan(r["bleu"] if metric == "mean_bleu" else r["similarity"])
                    ]
                )
                low, high = compute_ci(vals)
                ci_lower.append(low)
                ci_upper.append(high)
            ax.errorbar(
                sizes,
                ys,
                yerr=[
                    np.array(ys) - np.array(ci_lower),
                    np.array(ci_upper) - np.array(ys),
                ],
                marker="o",
                markersize=4,
                label=model,
                color=model_color(model),
                linewidth=1,
                capsize=4,
            )
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
        # KDE plot
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

        # Box plot
        fig, ax = plt.subplots(figsize=(6, 4))
        sizes = sorted(size_map.keys())
        data = [np.array([r[metric] for r in size_map[s] if not np.isnan(r[metric])]) for s in sizes]
        ax.boxplot(
            data,
            labels=[f"size {s}" for s in sizes],
            showmeans=True,
            meanline=True,
            flierprops={"marker": "x", "markersize": 5},
        )
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} Box Plot — {model}")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        box_path = out_dir / f"{model}_{metric}_box_by_size.png"
        fig.savefig(box_path, dpi=200)
        plt.close(fig)
        print(f"Saved {box_path}")


def plot_combined_box(per_record: Dict[str, Dict[int, List[Dict[str, float]]]], out_dir: Path, metric: str) -> None:
    """Side-by-side box plots per size (3/5/7) with model legend."""
    target_sizes = [3, 5, 7]
    models = sorted(per_record.keys(), key=model_sort_key)
    if not models:
        return

    # gather data by size then model
    grouped: Dict[int, Dict[str, np.ndarray]] = {s: {} for s in target_sizes}
    for model in models:
        for size in target_sizes:
            rows = per_record.get(model, {}).get(size, [])
            vals = np.array([r[metric] for r in rows if not np.isnan(r[metric])])
            if vals.size > 0:
                grouped[size][model] = vals

    fig, ax = plt.subplots(figsize=(8, 4))
    legend_handles = []
    positions = []
    labels = []
    box_data = []

    width = 0.35  # spacing between models within a size
    for si, size in enumerate(target_sizes):
        model_vals = grouped.get(size, {})
        if not model_vals:
            continue
        base_pos = si * (len(models) * width + 0.4)
        for mi, model in enumerate(models):
            vals = model_vals.get(model)
            if vals is None or vals.size == 0:
                continue
            pos = base_pos + mi * width
            positions.append(pos)
            labels.append(f"{size}")
            box_data.append(vals)
            color = model_color(model)
            bp = ax.boxplot(
                vals,
                positions=[pos],
                widths=width * 0.8,
                showmeans=True,
                meanline=True,
                patch_artist=True,
                flierprops={"marker": "x", "markersize": 5},
            )
            bp["boxes"][0].set_facecolor(color)
            for median in bp["medians"]:
                median.set_color(color)
            legend_handles.append((color, model))

    if not box_data:
        plt.close(fig)
        return

    # Build legend without duplicates
    unique_leg = []
    seen = set()
    for color, model in legend_handles:
        if model in seen:
            continue
        seen.add(model)
        unique_leg.append((color, model))
    for color, model in unique_leg:
        ax.plot([], [], color=color, label=model)

    ax.set_xticks([np.mean([p for p, lbl in zip(positions, labels) if lbl == str(size)]) for size in target_sizes if any(lbl == str(size) for lbl in labels)])
    ax.set_xticklabels([str(size) for size in target_sizes if any(lbl == str(size) for lbl in labels)])
    ax.set_xlabel("Voxel size")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} Box Plot — models vs size")
    ax.legend(title="Model")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"frame_reconstruction_{metric}_box_sizes_3_5_7.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_seaborn_kde(per_record: Dict[str, Dict[int, List[Dict[str, float]]]], out_dir: Path, metric: str) -> None:
    """Facet KDE by size, hue=model, using seaborn."""
    records = []
    for model, size_map in per_record.items():
        for size, rows in size_map.items():
            for r in rows:
                val = r[metric]
                if np.isnan(val):
                    continue
                records.append({"model": model, "size": size, metric: val})

    if not records:
        return

    import pandas as pd  # local import to avoid dependency at module import time

    df = pd.DataFrame(records)
    model_order = sorted(df["model"].unique(), key=model_sort_key)
    g = sns.FacetGrid(
        df,
        col="size",
        hue="model",
        hue_order=model_order,
        palette=model_palette(model_order),
        sharex=False,
        sharey=False,
        col_order=sorted(df["size"].unique()),
    )
    g.map_dataframe(sns.kdeplot, x=metric, fill=True, alpha=1.0)
    g.add_legend(title="Model")
    g.set_axis_labels(metric.replace("_", " ").title(), "Density")
    g.fig.suptitle(f"{metric.replace('_', ' ').title()} KDE by size (seaborn)", y=1.02)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"frame_reconstruction_{metric}_kde_seaborn.png"
    g.fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    print(f"Saved {out_path}")


def plot_seaborn_scatter(per_record: Dict[str, Dict[int, List[Dict[str, float]]]], out_dir: Path) -> None:
    """Scatter BLEU vs similarity with linear fit, hue=model, style=size."""
    records = []
    for model, size_map in per_record.items():
        for size, rows in size_map.items():
            for r in rows:
                if np.isnan(r["bleu"]) or np.isnan(r["similarity"]):
                    continue
                records.append({"model": model, "size": size, "bleu": r["bleu"], "similarity": r["similarity"]})

    if not records:
        return

    import pandas as pd  # local import

    df = pd.DataFrame(records)
    sns.set_style("whitegrid")
    model_order = sorted(df["model"].unique(), key=model_sort_key)
    g = sns.lmplot(
        data=df,
        x="bleu",
        y="similarity",
        hue="model",
        hue_order=model_order,
        palette=model_palette(model_order),
        markers="o",
        scatter_kws={"alpha": 1.0, "s": 15},
        line_kws={"linewidth": 1},
        height=4,
        aspect=1.2,
    )
    g.fig.suptitle("BLEU vs Similarity (linear fit by model)", y=1.02)
    g.set_axis_labels("BLEU", "Similarity")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "frame_reconstruction_bleu_vs_similarity_scatter.png"
    g.fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    print(f"Saved {out_path}")


def plot_seaborn_scatter_by_size(per_record: Dict[str, Dict[int, List[Dict[str, float]]]], out_dir: Path) -> None:
    """Scatter BLEU vs similarity faceted by size (3/5/7), hue=model."""
    records = []
    for model, size_map in per_record.items():
        for size, rows in size_map.items():
            for r in rows:
                if np.isnan(r["bleu"]) or np.isnan(r["similarity"]):
                    continue
                records.append({"model": model, "size": size, "bleu": r["bleu"], "similarity": r["similarity"]})

    if not records:
        return

    import pandas as pd  # local import

    df = pd.DataFrame(records)
    target_sizes = sorted({3, 5, 7} & set(df["size"].unique()))
    if not target_sizes:
        target_sizes = sorted(df["size"].unique())

    sns.set_style("whitegrid")
    model_order = sorted(df["model"].unique(), key=model_sort_key)
    g = sns.lmplot(
        data=df[df["size"].isin(target_sizes)],
        x="bleu",
        y="similarity",
        hue="model",
        hue_order=model_order,
        palette=model_palette(model_order),
        col="size",
        col_wrap=3,
        markers="o",
        scatter_kws={"alpha": 1.0, "s": 15},
        line_kws={"linewidth": 1},
        height=4,
        aspect=0.9,
    )
    g.fig.suptitle("BLEU vs Similarity by size (linear fit per model)", y=1.05)
    g.set_axis_labels("BLEU", "Similarity")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "frame_reconstruction_bleu_vs_similarity_scatter_by_size.png"
    g.fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    print(f"Saved {out_path}")


def plot_metric_vs_size_scatter(metrics_map: Dict[str, Dict[int, Dict[str, float]]], out_dir: Path) -> None:
    """Scatter+linear fit of size vs metric (BLEU, similarity) with model legend."""
    import pandas as pd  # local import

    rows = []
    for model, size_map in metrics_map.items():
        for size, vals in size_map.items():
            rows.append({"model": model, "size": size, "bleu": vals["mean_bleu"], "similarity": vals["mean_similarity"]})
    if not rows:
        return

    df = pd.DataFrame(rows)
    sns.set_style("whitegrid")
    model_order = sorted(df["model"].unique(), key=model_sort_key)
    for metric in ["similarity", "bleu"]:
        g = sns.lmplot(
            data=df,
            x="size",
            y=metric,
            hue="model",
            hue_order=model_order,
            palette=model_palette(model_order),
            markers="o",
            scatter_kws={"s": 35, "alpha": 1.0},
            line_kws={"linewidth": 1},
            height=4,
            aspect=1.2,
        )
        g.fig.suptitle(f"{metric.title()} vs size (scatter + linear fit)", y=1.02)
        g.set_axis_labels("Voxel size", metric.title())
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"frame_reconstruction_{metric}_vs_size_scatter.png"
        g.fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(g.fig)
        print(f"Saved {out_path}")


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
    plot_metrics_vs_size(metrics_map, per_record, plot_dir)
    plot_distributions(per_record, plot_dir, metric="bleu")
    plot_distributions(per_record, plot_dir, metric="similarity")
    plot_combined_box(per_record, plot_dir, metric="bleu")
    plot_combined_box(per_record, plot_dir, metric="similarity")
    plot_seaborn_kde(per_record, plot_dir, metric="bleu")
    plot_seaborn_kde(per_record, plot_dir, metric="similarity")
    plot_seaborn_scatter(per_record, plot_dir)
    plot_seaborn_scatter_by_size(per_record, plot_dir)
    plot_metric_vs_size_scatter(metrics_map, plot_dir)


if __name__ == "__main__":
    main()
