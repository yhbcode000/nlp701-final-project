"""
Relabel Minecraft dataset actions by inferring movement between consecutive frames.

The new action encoding is inferred purely from voxel deltas:
- straight (forward/backward): 0 still, 1 forward (+x), 2 backward (-x)
- pan (left/right):          0 still, 1 left (+z), 2 right (-z)
- jump:                      0 no jump, 1 jump (agent moved up)

This script walks every `creative:*` episode under the dataset root, compares
frame `t` and `t+1` to find the best 1-block shift that aligns the voxel
grids, and writes the resulting action vector back into frame `t`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def _extract_block_names(voxel: np.ndarray | dict) -> np.ndarray:
    """Return a raw ndarray of block names regardless of MineDojo dict wrapper."""
    if isinstance(voxel, dict) and "block_name" in voxel:
        return voxel["block_name"]
    return voxel


def _slices(size: int, shift: int) -> Tuple[slice, slice]:
    """Return aligned slices for source and target given an integer shift."""
    if shift >= 0:
        return slice(shift, size), slice(0, size - shift)
    return slice(0, size + shift), slice(-shift, size)


def _best_shift_along_axis(voxel_a: np.ndarray, voxel_b: np.ndarray, axis: int, candidates) -> int:
    """
    Independently find the shift along a single axis that maximizes overlap,
    holding the other axes fixed.
    """
    best_shift = 0
    best_score = -1
    size = voxel_a.shape[axis]

    for shift in candidates:
        slices_a = [slice(None), slice(None), slice(None)]
        slices_b = [slice(None), slice(None), slice(None)]
        sa, sb = _slices(size, shift)
        slices_a[axis] = sa
        slices_b[axis] = sb

        overlap_a = voxel_a[tuple(slices_a)]
        overlap_b = voxel_b[tuple(slices_b)]
        score = np.sum(overlap_a == overlap_b)

        if score > best_score or (score == best_score and shift == 0):
            best_score = score
            best_shift = shift

    return best_shift


def infer_shift(voxel_a: np.ndarray, voxel_b: np.ndarray) -> Tuple[int, int, int]:
    """
    Infer shifts independently per axis:
    - dx in {-1,0,1} (forward/backward)
    - dy in {0,1} (jump)
    - dz in {-1,0,1} (left/right)
    """
    dx = _best_shift_along_axis(voxel_a, voxel_b, axis=0, candidates=(-1, 0, 1))
    dy = _best_shift_along_axis(voxel_a, voxel_b, axis=1, candidates=(0, 1))
    dz = _best_shift_along_axis(voxel_a, voxel_b, axis=2, candidates=(-1, 0, 1))
    return dx, dy, dz


def shift_to_action(shift: Tuple[int, int, int]) -> np.ndarray:
    """Map (dx, dy, dz) shift to the 3-element action vector."""
    dx, dy, dz = shift

    # straight axis (forward/backward)
    straight = 0
    if dx > 0:
        straight = 1  # forward
    elif dx < 0:
        straight = 2  # backward

    # pan axis (left/right)
    pan = 0
    if dz > 0:
        pan = 1  # left
    elif dz < 0:
        pan = 2  # right

    # jump flag
    jump = 1 if dy > 0 else 0

    return np.array([straight, pan, jump], dtype=np.int64)


def relabel_episode(episode_dir: Path) -> None:
    """Relabel all frames in a single creative episode."""
    npy_files = sorted(episode_dir.glob("*.npy"))
    if len(npy_files) < 2:
        return

    for idx in range(len(npy_files) - 1):
        current_path = npy_files[idx]
        next_path = npy_files[idx + 1]

        current = np.load(current_path, allow_pickle=True).item()
        next_frame = np.load(next_path, allow_pickle=True).item()

        voxel_a = _extract_block_names(current["voxel"])
        voxel_b = _extract_block_names(next_frame["voxel"])

        shift = infer_shift(voxel_a, voxel_b)
        current["action"] = shift_to_action(shift)

        np.save(current_path, current)

    # For the final frame we cannot infer the next action; leave as-is but ensure shape.
    last = np.load(npy_files[-1], allow_pickle=True).item()
    if "action" not in last or np.shape(last["action"]) != (3,):
        last["action"] = np.zeros(3, dtype=np.int64)
        np.save(npy_files[-1], last)


def relabel_dataset(root: Path) -> None:
    """Relabel every creative episode under the given dataset root."""
    creative_dirs: Iterable[Path] = sorted(
        p for p in root.rglob("*") if p.is_dir() and p.name.startswith("creative:")
    )
    if not creative_dirs:
        raise ValueError(f"No creative:* directories found under {root}")

    for creative_dir in creative_dirs:
        print(f"Relabeling {creative_dir}")
        relabel_episode(creative_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel Minecraft dataset actions from voxel motion.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to dataset root containing creative:* episodes (default: data)",
    )
    args = parser.parse_args()
    relabel_dataset(args.data_dir)


if __name__ == "__main__":
    main()
