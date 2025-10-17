from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence, Tuple

from minecraft_dataset import MinecraftDataset

warnings.filterwarnings("ignore", message="CUDA initialization: Unexpected error from cudaGetDeviceCount().*")


def load_full_dataset(config, data_dir: str | Path) -> Tuple[MinecraftDataset, int, Sequence[str]]:
    """Load the Minecraft dataset using the configured maximum length."""
    resolved_dir = Path(data_dir)
    dataset = MinecraftDataset(
        data_dir=str(resolved_dir),
        max_length=config.get_config()["max_length"],
    )

    total_pairs = len(dataset)
    unique_actions = sorted({pair["y"] for pair in dataset.data_pairs})

    print(f"Loaded {total_pairs} sequential frame/action pairs from {resolved_dir}.")
    print(f"Unique actions: {unique_actions}")

    return dataset, total_pairs, unique_actions


def preview_dataset_example(dataset: MinecraftDataset, idx: int = 0) -> None:
    """Print a dataset example for quick inspection."""
    example = dataset[idx]
    print("=" * 80)
    print(f"Example {idx}")
    print("- Current Frame (x):")
    print(example["x"])
    print("- Action (y):")
    print(example["y"])
    print("- Next Frame (z):")
    print(example["z"])
    if example.get("history_reconstruction"):
        print("- History (frame→action chain):")
        print(example["history_reconstruction"])
    if example.get("history_action"):
        print("- History (frame/frame/action pattern):")
        print(example["history_action"])
    print("=" * 80)


def main() -> None:
    """Quick smoke test for dataset loading."""
    from hyperparameter_config import HyperparameterConfig

    config = HyperparameterConfig()
    dataset, total, actions = load_full_dataset(config, "datasets/minecraft/data")
    print(f"Total pairs: {total}, sample actions: {actions[:5]}")
    preview_dataset_example(dataset, 0)


if __name__ == "__main__":
    main()
