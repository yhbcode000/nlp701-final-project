from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class PlotUtils:
    """Utility class for building publication-quality visualizations."""

    def __init__(self, style: str = "seaborn-v0_8-paper"):
        """Initialize plotting defaults and ensure output directory exists."""
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("seaborn-v0_8")

        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 10)
        os.makedirs("plots", exist_ok=True)

    @staticmethod
    def _resolve_scales(metric_keys: Sequence[str], scales: Optional[Sequence[float]], default: float = 100.0) -> List[float]:
        """Resolve scaling factors for a list of metrics."""
        if scales is None:
            return [default] * len(metric_keys)
        scale_values = list(scales)
        if len(scale_values) != len(metric_keys):
            raise ValueError("Number of scales must match number of metric keys")
        return [float(value) for value in scale_values]

    @staticmethod
    def plot_multi_metric_bar(
        results: Dict[str, Dict],
        metric_keys: Sequence[str],
        metric_labels: Sequence[str],
        title: str,
        save_path: str,
        scales: Optional[Sequence[float]] = None,
        ylabel: Optional[str] = None,
        ylim: Optional[Sequence[float]] = None,
    ):
        """Render grouped bar charts comparing multiple metrics per model."""
        os.makedirs("plots", exist_ok=True)
        models = list(results.keys())
        if not models or not metric_keys:
            print("No data to plot multi-metric bar chart.")
            return

        scale_values = PlotUtils._resolve_scales(metric_keys, scales)
        if ylabel is None:
            ylabel = "Score (%)" if all(abs(scale - 100.0) < 1e-6 for scale in scale_values) else "Score"

        x_positions = np.arange(len(models))
        width = 0.8 / max(1, len(metric_keys))
        figure, axes = plt.subplots(figsize=(12, 6))

        for index, (metric, label, scale) in enumerate(zip(metric_keys, metric_labels, scale_values)):
            values = [results[model].get(metric, 0.0) * scale for model in models]
            offset = (index - (len(metric_keys) - 1) / 2) * width
            bars = axes.bar(x_positions + offset, values, width, label=label)
            for bar, value in zip(bars, values):
                axes.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

        axes.set_xticks(x_positions)
        axes.set_xticklabels(models)
        axes.set_ylabel(ylabel)
        axes.set_title(title, fontsize=16, fontweight="bold", pad=20)
        axes.legend()
        axes.grid(axis="y", alpha=0.3, linestyle="--")
        if ylim is not None:
            axes.set_ylim(ylim)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {save_path}")

    @staticmethod
    def plot_metrics_heatmap(
        results: Dict[str, Dict],
        title: str,
        save_path: str,
        metrics: Optional[Sequence[str]] = None,
        metric_labels: Optional[Sequence[str]] = None,
        scales: Optional[Sequence[float]] = None,
    ):
        """Plot a heatmap for metric comparison across models."""
        os.makedirs("plots", exist_ok=True)
        models = list(results.keys())
        if not models:
            print("No data to plot heatmap.")
            return

        metrics = list(metrics) if metrics is not None else ["accuracy", "precision", "recall", "f1"]
        metric_labels = list(metric_labels) if metric_labels is not None else [metric.replace("_", " ").title() for metric in metrics]
        if len(metric_labels) != len(metrics):
            raise ValueError("Metric labels length must match metrics length")

        scale_values = PlotUtils._resolve_scales(metrics, scales)
        data = []
        for model in models:
            row = []
            for metric, scale in zip(metrics, scale_values):
                value = results.get(model, {}).get(metric, 0.0)
                row.append(value * scale)
            data.append(row)
        heatmap = np.array(data)

        figure, axes = plt.subplots(figsize=(10, 6))
        image = axes.imshow(heatmap, cmap="YlGnBu", aspect="auto")

        axes.set_xticks(np.arange(len(metrics)))
        axes.set_yticks(np.arange(len(models)))
        axes.set_xticklabels(metric_labels, fontsize=12)
        axes.set_yticklabels(models, fontsize=12)

        for i, model in enumerate(models):
            for j, metric_label in enumerate(metric_labels):
                axes.text(j, i, f"{heatmap[i, j]:.1f}", ha="center", va="center", color="black", fontsize=11, fontweight="bold")

        colorbar = plt.colorbar(image, ax=axes)
        colorbar.set_label("Score (%)" if all(abs(scale - 100.0) < 1e-6 for scale in scale_values) else "Score", rotation=270, labelpad=20)

        axes.set_title(title, fontsize=16, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {save_path}")

    @staticmethod
    def plot_training_curves(train_losses: Sequence[float], val_losses: Sequence[float], title: str, save_path: str):
        """Plot training and validation loss curves."""
        os.makedirs("plots", exist_ok=True)
        epochs = np.arange(1, len(train_losses) + 1)

        figure, axes = plt.subplots(figsize=(12, 6))
        axes.plot(epochs, train_losses, marker="o", label="Training Loss", linewidth=2)
        axes.plot(epochs, val_losses, marker="s", label="Validation Loss", linewidth=2)

        axes.set_xlabel("Epoch", fontsize=14, fontweight="bold")
        axes.set_ylabel("Loss", fontsize=14, fontweight="bold")
        axes.set_title(title, fontsize=16, fontweight="bold", pad=20)
        axes.legend(fontsize=12, loc="best")
        axes.grid(alpha=0.3, linestyle="--")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {save_path}")

    @staticmethod
    def plot_method_metric_bar(
        method_results: Dict[str, Dict[str, Dict]],
        metric_key: str,
        title: str,
        save_path: str,
        scale: float = 100.0,
        ylabel: Optional[str] = None,
        method_labels: Optional[Dict[str, str]] = None,
        metric_label: Optional[str] = None,
        ylim: Optional[Sequence[float]] = None,
    ):
        """Compare a single metric across multiple methods and models."""
        os.makedirs("plots", exist_ok=True)
        if not method_results:
            print("No method data to plot.")
            return

        methods = list(method_results.keys())
        model_names = sorted({model for result in method_results.values() for model in result.keys()})
        if not model_names:
            print("No model data to plot.")
            return

        x_positions = np.arange(len(model_names))
        width = 0.8 / max(1, len(methods))
        ylabel = ylabel or ("Score (%)" if abs(scale - 100.0) < 1e-6 else "Score")
        metric_label = metric_label or metric_key.replace("_", " ").title()

        figure, axes = plt.subplots(figsize=(12, 6))
        for index, method in enumerate(methods):
            display_name = method_labels.get(method, method.replace("_", " ").title()) if method_labels else method.replace("_", " ").title()
            values = [method_results[method].get(model, {}).get(metric_key, 0.0) * scale for model in model_names]
            offset = (index - (len(methods) - 1) / 2) * width
            bars = axes.bar(x_positions + offset, values, width, label=display_name)
            for bar, value in zip(bars, values):
                axes.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

        axes.set_xticks(x_positions)
        axes.set_xticklabels(model_names)
        axes.set_ylabel(ylabel)
        axes.set_title(title, fontsize=16, fontweight="bold", pad=20)
        axes.legend()
        axes.grid(axis="y", alpha=0.3, linestyle="--")
        if ylim is not None:
            axes.set_ylim(ylim)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {save_path}")

    @staticmethod
    def plot_method_comparison(
        results_dict: Dict[str, Dict[str, Dict]],
        title: str,
        save_path: str,
        methods: List[str],
        metric: str = "accuracy",
    ):
        """Compatibility wrapper to compare named methods using a shared metric."""
        filtered = {method: results_dict[method] for method in methods if method in results_dict}
        if not filtered:
            print("No matching methods to plot.")
            return
        PlotUtils.plot_method_metric_bar(
            filtered,
            metric_key=metric,
            title=title,
            save_path=save_path,
            scale=100.0,
            ylabel=f"{metric.replace('_', ' ').title()} (%)",
        )

    @staticmethod
    def plot_confusion_matrix(confusion_matrix: np.ndarray, class_names: List[str], title: str, save_path: str):
        """Visualize a confusion matrix with annotations."""
        os.makedirs("plots", exist_ok=True)

        figure, axes = plt.subplots(figsize=(10, 8))
        image = axes.imshow(confusion_matrix, cmap="Blues", aspect="auto")

        axes.set_xticks(np.arange(len(class_names)))
        axes.set_yticks(np.arange(len(class_names)))
        axes.set_xticklabels(class_names, rotation=45, ha="right")
        axes.set_yticklabels(class_names)

        colorbar = plt.colorbar(image, ax=axes)
        colorbar.set_label("Count", rotation=270, labelpad=20)

        for i, _ in enumerate(class_names):
            for j, _ in enumerate(class_names):
                value = int(confusion_matrix[i, j])
                axes.text(
                    j,
                    i,
                    value,
                    ha="center",
                    va="center",
                    color="white" if value > confusion_matrix.max() / 2 else "black",
                )

        axes.set_xlabel("Predicted", fontsize=14, fontweight="bold")
        axes.set_ylabel("True", fontsize=14, fontweight="bold")
        axes.set_title(title, fontsize=16, fontweight="bold", pad=20)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {save_path}")


def main() -> None:
    """Basic plotting smoke test that writes a sample chart."""
    plotter = PlotUtils()
    sample_results = {"model_a": {"accuracy": 0.85, "f1": 0.8}, "model_b": {"accuracy": 0.9, "f1": 0.82}}
    plotter.plot_multi_metric_bar(
        sample_results,
        metric_keys=["accuracy", "f1"],
        metric_labels=["Accuracy", "F1"],
        title="Sample Metrics",
        save_path="plots/sample_metrics.png",
    )
    print("Sample plot created at plots/sample_metrics.png")


if __name__ == "__main__":
    main()
