from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence, Tuple

from minecraft_dataset import MinecraftDataset

warnings.filterwarnings("ignore", message="CUDA initialization: Unexpected error from cudaGetDeviceCount().*")


def load_full_dataset(
    config,
    data_dir: str | Path,
    *,
    voxel_size: int | None = None,
) -> Tuple[MinecraftDataset, int, Sequence[str]]:
    """Load the Minecraft dataset using the configured maximum length.

    The `voxel_size` argument lets you override the cube window passed into
    `voxel2word` (defaults to 5 if not provided).
    """
    resolved_dir = Path(data_dir)
    config_dict = config.get_config()
    resolved_voxel_size = voxel_size if voxel_size is not None else config_dict.get("voxel_size", 5)
    dataset = MinecraftDataset(
        data_dir=str(resolved_dir),
        max_length=config_dict["max_length"],
        voxel_size=resolved_voxel_size,
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
