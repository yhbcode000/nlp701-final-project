from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def center_crop(voxel: np.ndarray, cube_size: int) -> np.ndarray:
    """Crop an odd-sized cube around the center of the voxel grid."""
    if cube_size % 2 == 0:
        raise ValueError("cube_size must be odd")
    if cube_size > voxel.shape[0]:
        raise ValueError(f"cube_size {cube_size} exceeds voxel dimension {voxel.shape[0]}")

    half = cube_size // 2
    center = voxel.shape[0] // 2
    return voxel[
        center - half : center + half + 1,
        center - half : center + half + 1,
        center - half : center + half + 1,
    ]


def apply_random_mask(
    voxel: np.ndarray,
    *,
    cube_size: int = 7,
    mask_size: int = 3,
    mask_label: str = "mask",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """
    Crop a cube from the voxel grid and apply a random mask sub-cube filled with `mask_label`.

    Args:
        voxel: Input voxel array (assumed cubic on the first dimension).
        cube_size: Side length of the cropped cube (must be odd).
        mask_size: Side length of the masked sub-cube.
        mask_label: Label to fill the masked region with.
        rng: Optional NumPy Generator for reproducibility.

    Returns:
        A new voxel cube with the masked region applied.
    """
    if mask_size > cube_size:
        raise ValueError("mask_size must be <= cube_size")

    rng = rng or np.random.default_rng()

    cropped = center_crop(voxel, cube_size).copy()
    max_start = cube_size - mask_size
    start_x = rng.integers(0, max_start + 1)
    start_y = rng.integers(0, max_start + 1)
    start_z = rng.integers(0, max_start + 1)

    mask = np.zeros_like(cropped, dtype=object)
    mask[
        start_x : start_x + mask_size,
        start_y : start_y + mask_size,
        start_z : start_z + mask_size,
    ] = mask_label

    cropped[
        start_x : start_x + mask_size,
        start_y : start_y + mask_size,
        start_z : start_z + mask_size,
    ] = mask_label
    return cropped, mask, (start_x, start_y, start_z)


def voxel_to_word(voxel: np.ndarray) -> str:
    """Convert a voxel cube to the pipe-delimited word format (matches read_data.voxel2word)."""
    word = ""
    shape = voxel.shape
    for y in range(shape[1]):
        for x in range(shape[0]):
            for z in range(shape[2]):
                word += f"|{voxel[x, y, z]}"
            word += "|\n"
    return word


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a random 3D mask to a voxel grid.")
    parser.add_argument("--cube-size", type=int, default=7, help="Side length of the cropped cube (odd).")
    parser.add_argument("--mask-size", type=int, default=3, help="Side length of the masked sub-cube.")
    args = parser.parse_args()

    # Demo with a synthetic cube of block labels
    size = 11
    voxel = np.full((size, size, size), "air", dtype=object)
    voxel[size // 2, size // 2, size // 2] = "stone"

    masked, mask = apply_random_mask(voxel, cube_size=args.cube_size, mask_size=args.mask_size)
    print("Masked voxel as text:\n")
    print(voxel_to_word(masked))
    print("Mask cube as text (masked cells marked with label, others empty):\n")
    print(voxel_to_word(mask))


if __name__ == "__main__":
    main()
