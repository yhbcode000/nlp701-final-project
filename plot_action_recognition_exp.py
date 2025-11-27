from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns


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


def compute_ci(values: List[float]) -> Tuple[float, float]:
    """95% CI for mean; returns (mean, mean) if insufficient data."""
    if not values:
        return float("nan"), float("nan")
    vals = np.array(values, dtype=float)
    mean = float(np.mean(vals))
    if len(vals) < 2:
        return mean, mean
    std = float(np.std(vals, ddof=1))
    margin = 1.96 * std / np.sqrt(len(vals))
    return mean - margin, mean + margin


def is_correct_action(true_val: int, pred_val: int) -> bool:
    """Correct if non-zero truth gets any non-zero prediction; zero truth expects zero prediction."""
    if true_val == 0:
        return pred_val == 0
    return pred_val != 0


def plot_combined_with_error(
    metrics_map: Dict[str, Dict[int, Dict[str, float]]],
    per_record: Dict[str, Dict[int, List[Dict[str, int]]]],
    out_dir: Path,
) -> None:
    """Single plot per model with three lines (straight/pan/jump) and 95% CI error bars."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dims = ["straight", "pan", "jump"]
    colors = {"straight": "#4C72B0", "pan": "#C44E52", "jump": "#55A868"}

    for model, size_map in metrics_map.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        sizes = sorted(size_map.keys())
        for dim in dims:
            means = []
            lower = []
            upper = []
            for s in sizes:
                recs = per_record.get(model, {}).get(s, [])
                accs = [
                    1.0 if is_correct_action(r[f"{dim}_true"], r[f"{dim}_pred"]) else 0.0
                    for r in recs
                ]
                means.append(float(np.mean(accs)) if accs else 0.0)
                lo, hi = compute_ci(accs)
                lower.append(lo)
                upper.append(hi)
            ax.errorbar(
                sizes,
                means,
                yerr=[np.array(means) - np.array(lower), np.array(upper) - np.array(means)],
                label=dim,
                color=colors.get(dim),
                marker="o",
                capsize=4,
            )
        ax.set_xlabel("Voxel size")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Action accuracy vs size — {model}")
        ax.legend(title="Dimension")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"action_recognition_accuracy_vs_size_{model}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved {path}")


def plot_per_dim_with_error(
    metrics_map: Dict[str, Dict[int, Dict[str, float]]],
    per_record: Dict[str, Dict[int, List[Dict[str, int]]]],
    out_dir: Path,
) -> None:
    """Three separate plots (straight/pan/jump), models as legend, with 95% CI error bars."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dims = ["straight", "pan", "jump"]
    colors = sns.color_palette("tab10")
    models = sorted(metrics_map.keys())

    for dim in dims:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for mi, model in enumerate(models):
            size_map = metrics_map[model]
            sizes = sorted(size_map.keys())
            means = []
            lower = []
            upper = []
            for s in sizes:
                recs = per_record.get(model, {}).get(s, [])
                accs = [
                    1.0 if is_correct_action(r[f"{dim}_true"], r[f"{dim}_pred"]) else 0.0
                    for r in recs
                ]
                means.append(float(np.mean(accs)) if accs else 0.0)
                lo, hi = compute_ci(accs)
                lower.append(lo)
                upper.append(hi)
            ax.errorbar(
                sizes,
                means,
                yerr=[np.array(means) - np.array(lower), np.array(upper) - np.array(means)],
                label=model,
                color=colors[mi % len(colors)],
                marker="o",
                capsize=4,
            )
        ax.set_xlabel("Voxel size")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{dim.capitalize()} accuracy vs size (95% CI)")
        ax.legend(title="Model")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"action_recognition_{dim}_accuracy_vs_size.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved {path}")


def _overall_accuracy_curve(per_record, exclude_all_zero: bool, min_correct: int = 3):
    colors = sns.color_palette("tab10")
    curves = []
    for mi, (model, size_map) in enumerate(sorted(per_record.items())):
        sizes = sorted(size_map.keys())
        means = []
        lower = []
        upper = []
        for s in sizes:
            recs = size_map[s]
            accs = []
            for r in recs:
                if exclude_all_zero:
                    label_all_zero = r["straight_true"] == 0 and r["pan_true"] == 0 and r["jump_true"] == 0
                    pred_all_zero = r["straight_pred"] == 0 and r["pan_pred"] == 0 and r["jump_pred"] == 0
                    if label_all_zero and pred_all_zero:
                        continue
                correct_count = sum(
                    1 if is_correct_action(r[f"{dim}_true"], r[f"{dim}_pred"]) else 0
                    for dim in ["straight", "pan", "jump"]
                )
                accs.append(1.0 if correct_count >= min_correct else 0.0)
            lo, hi = compute_ci(accs)
            lower.append(lo)
            upper.append(hi)
            means.append(float(np.mean(accs)) if accs else float("nan"))
        curves.append((model, sizes, means, lower, upper, colors[mi % len(colors)]))
    return curves


def plot_overall_accuracy(per_record: Dict[str, Dict[int, List[Dict[str, int]]]], out_dir: Path) -> None:
    """Plot overall action accuracy including all records."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for model, sizes, means, lower, upper, color in _overall_accuracy_curve(per_record, exclude_all_zero=False, min_correct=3):
        ax.errorbar(
            sizes,
            means,
            yerr=[np.array(means) - np.array(lower), np.array(upper) - np.array(means)],
            label=model,
            color=color,
            marker="o",
            capsize=4,
        )
    ax.set_xlabel("Voxel size")
    ax.set_ylabel("Overall accuracy")
    ax.set_title("Overall action accuracy vs size (all records)")
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "action_recognition_overall_accuracy_vs_size.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_overall_accuracy_no_zero(per_record: Dict[str, Dict[int, List[Dict[str, int]]]], out_dir: Path) -> None:
    """Plot overall action accuracy excluding all-zero labels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for model, sizes, means, lower, upper, color in _overall_accuracy_curve(per_record, exclude_all_zero=True, min_correct=3):
        ax.errorbar(
            sizes,
            means,
            yerr=[np.array(means) - np.array(lower), np.array(upper) - np.array(means)],
            label=model,
            color=color,
            marker="o",
            capsize=4,
        )
    ax.set_xlabel("Voxel size")
    ax.set_ylabel("Overall accuracy")
    ax.set_title("Overall action accuracy vs size (exclude all-zero labels)")
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "action_recognition_overall_accuracy_vs_size_no_zero.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_at_least_one_accuracy(per_record: Dict[str, Dict[int, List[Dict[str, int]]]], out_dir: Path, exclude_all_zero: bool, suffix: str) -> None:
    """Plot accuracy for 'at least one correct' rule, with option to exclude all-zero labels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for model, sizes, means, lower, upper, color in _overall_accuracy_curve(per_record, exclude_all_zero=exclude_all_zero, min_correct=1):
        ax.errorbar(
            sizes,
            means,
            yerr=[np.array(means) - np.array(lower), np.array(upper) - np.array(means)],
            label=model,
            color=color,
            marker="o",
            capsize=4,
        )
    ax.set_xlabel("Voxel size")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"At least one action correct vs size ({'exclude zeros' if exclude_all_zero else 'all records'})")
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"action_recognition_at_least_one_accuracy_{suffix}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_at_least_two_accuracy(per_record: Dict[str, Dict[int, List[Dict[str, int]]]], out_dir: Path, exclude_all_zero: bool, suffix: str) -> None:
    """Plot accuracy for 'at least two correct' rule, with option to exclude all-zero labels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for model, sizes, means, lower, upper, color in _overall_accuracy_curve(per_record, exclude_all_zero=exclude_all_zero, min_correct=2):
        ax.errorbar(
            sizes,
            means,
            yerr=[np.array(means) - np.array(lower), np.array(upper) - np.array(means)],
            label=model,
            color=color,
            marker="o",
            capsize=4,
        )
    ax.set_xlabel("Voxel size")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"At least two actions correct vs size ({'exclude zeros' if exclude_all_zero else 'all records'})")
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"action_recognition_at_least_two_accuracy_{suffix}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_at_least_three_accuracy(per_record: Dict[str, Dict[int, List[Dict[str, int]]]], out_dir: Path, exclude_all_zero: bool, suffix: str) -> None:
    """Plot accuracy for 'at least three correct' rule, with option to exclude all-zero labels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for model, sizes, means, lower, upper, color in _overall_accuracy_curve(per_record, exclude_all_zero=exclude_all_zero, min_correct=3):
        ax.errorbar(
            sizes,
            means,
            yerr=[np.array(means) - np.array(lower), np.array(upper) - np.array(means)],
            label=model,
            color=color,
            marker="o",
            capsize=4,
        )
    ax.set_xlabel("Voxel size")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"At least three actions correct vs size ({'exclude zeros' if exclude_all_zero else 'all records'})")
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"action_recognition_at_least_three_accuracy_{suffix}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_confusion_matrices(per_record: Dict[str, Dict[int, List[Dict[str, int]]]], out_dir: Path) -> None:
    """Plot confusion matrices for straight/pan/jump per model (aggregated over sizes).

    Value sets:
    - straight: {-1, 0, 1} (back/left/down, noop, forward/right/up)
    - pan: {-1, 0, 1}
    - jump: {0, 1} (noop, up)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dim_configs = {
        "straight": [-1, 0, 1],
        "pan": [-1, 0, 1],
        "jump": [0, 1],
    }
    label_map = {-1: "back/left/down", 0: "noop", 1: "forward/right/up"}

    for model, size_map in sorted(per_record.items()):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for di, (dim, values) in enumerate(dim_configs.items()):
            n = len(values)
            conf = np.zeros((n, n), dtype=int)
            for recs in size_map.values():
                for r in recs:
                    true_val = r[f"{dim}_true"]
                    pred_val = r[f"{dim}_pred"]
                    if true_val not in values or pred_val not in values:
                        continue
                    ti = values.index(true_val)
                    pi = values.index(pred_val)
                    conf[ti, pi] += 1

            ax = axes[di]
            sns.heatmap(
                conf,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=[label_map.get(v, str(v)) for v in values],
                yticklabels=[label_map.get(v, str(v)) for v in values],
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{dim.capitalize()} confusion — {model}")

        fig.tight_layout()
        path = out_dir / f"action_recognition_confusion_{model}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved {path}")


def plot_action_type_confusion(per_record: Dict[str, Dict[int, List[Dict[str, int]]]], out_dir: Path) -> None:
    """Confusion between action types (which action was taken) aggregated per model, shown as a 3x3 per model (2 models -> 3x3x2)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dim_order = ["straight", "pan", "jump"]

    model_items = sorted(per_record.items())
    if not model_items:
        return

    fig, axes = plt.subplots(1, len(model_items), figsize=(6 * len(model_items), 4), sharex=True, sharey=True)
    if len(model_items) == 1:
        axes = [axes]

    for ax, (model, size_map) in zip(axes, model_items):
        conf = np.zeros((3, 3), dtype=int)
        for recs in size_map.values():
            for r in recs:
                true_dim = next((di for di, dim in enumerate(dim_order) if r[f"{dim}_true"] != 0), None)
                pred_dim = next((di for di, dim in enumerate(dim_order) if r[f"{dim}_pred"] != 0), None)
                if true_dim is None or pred_dim is None:
                    continue
                conf[true_dim, pred_dim] += 1

        sns.heatmap(
            conf,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=[d.capitalize() for d in dim_order],
            yticklabels=[d.capitalize() for d in dim_order],
            ax=ax,
        )
        ax.set_xlabel("Predicted action type")
        ax.set_ylabel("True action type")
        ax.set_title(f"{model}")

    fig.suptitle("Action-type confusion (3x3 per model)", y=1.02)
    fig.tight_layout()
    path = out_dir / "action_recognition_action_type_confusion_models.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_joint_confusion(per_record: Dict[str, Dict[int, List[Dict[str, int]]]], out_dir: Path) -> None:
    """Joint confusion over all action combinations (3*3*2 = 18 classes) per model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    straight_vals = [-1, 0, 1]
    pan_vals = [-1, 0, 1]
    jump_vals = [0, 1]

    combos = []
    for s in straight_vals:
        for p in pan_vals:
            for j in jump_vals:
                combos.append((s, p, j))
    labels = [f"s{s}_p{p}_j{j}" for s, p, j in combos]
    combo_index = {c: i for i, c in enumerate(combos)}

    for model, size_map in sorted(per_record.items()):
        conf = np.zeros((len(combos), len(combos)), dtype=int)
        for recs in size_map.values():
            for r in recs:
                true_combo = (r["straight_true"], r["pan_true"], r["jump_true"])
                pred_combo = (r["straight_pred"], r["pan_pred"], r["jump_pred"])
                if true_combo not in combo_index or pred_combo not in combo_index:
                    continue
                ti = combo_index[true_combo]
                pi = combo_index[pred_combo]
                conf[ti, pi] += 1

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            conf,
            annot=False,
            cmap="Blues",
            cbar=True,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted (s,p,j)")
        ax.set_ylabel("True (s,p,j)")
        ax.set_title(f"Joint action confusion (18x18) — {model}")
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
        fig.tight_layout()
        path = out_dir / f"action_recognition_joint_confusion_{model}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


def plot_correctness_stacked(per_record: Dict[str, Dict[int, List[Dict[str, int]]]], out_dir: Path) -> None:
    """Stacked bars per size showing counts of 0/1/2/3 correctly predicted actions, separated by model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    categories = ["0 correct", "1 correct", "2 correct", "3 correct"]
    colors = ["#C44E52", "#DD8452", "#55A868", "#4C72B0"]

    for model, size_map in sorted(per_record.items()):
        size_counts: Dict[int, List[int]] = {}
        for size, recs in size_map.items():
            bucket = size_counts.setdefault(size, [0, 0, 0, 0])  # idx=correct count
            for r in recs:
                correct = 0
                correct += 1 if is_correct_action(r["straight_true"], r["straight_pred"]) else 0
                correct += 1 if is_correct_action(r["pan_true"], r["pan_pred"]) else 0
                correct += 1 if is_correct_action(r["jump_true"], r["jump_pred"]) else 0
                bucket[correct] += 1

        if not size_counts:
            continue

        sizes = sorted(size_counts.keys())
        bottoms = np.zeros(len(sizes))
        fig, ax = plt.subplots(figsize=(7, 4))
        bar_containers = []
        for idx, label in enumerate(categories):
            values = [size_counts[s][idx] for s in sizes]
            container = ax.bar(sizes, values, bottom=bottoms, label=label, color=colors[idx], alpha=0.85)
            bar_containers.append((container, values, bottoms.copy()))
            bottoms += np.array(values)

        # add value labels inside bars
        for container, values, bottom_vals in bar_containers:
            for rect, val, bottom in zip(container, values, bottom_vals):
                if val <= 0:
                    continue
                height = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    bottom + height / 2,
                    f"{int(val)}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )
        ax.set_xlabel("Voxel size")
        ax.set_ylabel("Count")
        ax.set_title(f"Correctness distribution by size — {model}")
        ax.legend(title="Correct actions")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"action_recognition_correctness_stacked_by_size_{model}.png"
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
    # plot_combined_with_error(metrics_map, per_record, plot_dir)
    plot_per_dim_with_error(metrics_map, per_record, plot_dir)
    plot_overall_accuracy(per_record, plot_dir)
    plot_overall_accuracy_no_zero(per_record, plot_dir)
    plot_at_least_one_accuracy(per_record, plot_dir, exclude_all_zero=False, suffix="all")
    plot_at_least_one_accuracy(per_record, plot_dir, exclude_all_zero=True, suffix="no_zero")
    plot_at_least_two_accuracy(per_record, plot_dir, exclude_all_zero=False, suffix="all")
    plot_at_least_two_accuracy(per_record, plot_dir, exclude_all_zero=True, suffix="no_zero")
    plot_at_least_three_accuracy(per_record, plot_dir, exclude_all_zero=False, suffix="all")
    plot_at_least_three_accuracy(per_record, plot_dir, exclude_all_zero=True, suffix="no_zero")
    plot_correctness_stacked(per_record, plot_dir)
    plot_confusion_matrices(per_record, plot_dir)
    plot_action_type_confusion(per_record, plot_dir)
    plot_joint_confusion(per_record, plot_dir)


if __name__ == "__main__":
    main()
