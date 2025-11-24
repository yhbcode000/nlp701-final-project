"""
Filter Minecraft dataset by collapsing consecutive identical (0,0,0) actions.

For each creative episode:
- Walk frames in order.
- If multiple adjacent frames have action == [0,0,0], keep the first and drop
  subsequent duplicates.
- Reindex remaining frames starting from 000000.npy to preserve order.

Usage:
    python3 filter_data.py --data-dir data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_frame(path: Path) -> dict:
    return np.load(path, allow_pickle=True).item()


def save_frame(path: Path, frame: dict) -> None:
    np.save(path, frame)


def is_zero_action(frame: dict) -> bool:
    action = frame.get("action")
    return isinstance(action, (list, tuple, np.ndarray)) and np.all(np.array(action) == 0)


def filter_episode(episode_dir: Path) -> None:
    npy_files = sorted(episode_dir.glob("*.npy"))
    if len(npy_files) < 2:
        return

    frames = [load_frame(p) for p in npy_files]

    filtered = []
    prev_zero = False
    for frame in frames:
        zero = is_zero_action(frame)
        if zero and prev_zero:
            continue
        filtered.append(frame)
        prev_zero = zero

    # Rewrite with new indices.
    for idx, frame in enumerate(filtered):
        new_name = f"{idx:06d}.npy"
        save_frame(episode_dir / new_name, frame)

    # Remove any leftover files whose names are not part of the new sequence.
    expected = {f"{idx:06d}.npy" for idx in range(len(filtered))}
    for old in episode_dir.glob("*.npy"):
        if old.name not in expected:
            try:
                old.unlink()
            except FileNotFoundError:
                pass

    print(f"{episode_dir}: kept {len(filtered)} of {len(frames)} frames")


def filter_dataset(root: Path) -> None:
    creative_dirs = sorted(p for p in root.rglob("*") if p.is_dir() and p.name.startswith("creative:"))
    if not creative_dirs:
        raise ValueError(f"No creative:* directories found under {root}")
    for creative_dir in creative_dirs:
        filter_episode(creative_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collapse consecutive zero-action frames.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset root containing creative episodes")
    args = parser.parse_args()
    filter_dataset(args.data_dir)


if __name__ == "__main__":
    main()
