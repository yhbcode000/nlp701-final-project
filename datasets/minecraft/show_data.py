"""
Visualize Minecraft voxel frames as a 3D scatter plot with colors per block type.

Usage:
    # Single frame
    python3 show_data.py --path data/creative:0/000000.npy --dim 5

    # Animate all frames in an episode directory
    python3 show_data.py --path data/creative:0 --dim 5

Arguments:
    --path / -p : Path to a single frame .npy or an episode directory
    --dim  / -d : Odd cube size to crop around the center (default 5)
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)

_ANIMATIONS: list[animation.FuncAnimation] = []
_NON_INTERACTIVE_BACKENDS = {"agg", "cairoagg", "svg", "pdf", "ps"}


def _is_interactive_backend() -> bool:
    backend = plt.get_backend().lower()
    return not any(tag in backend for tag in _NON_INTERACTIVE_BACKENDS)

def load_frame(path: Path) -> dict:
    """Load full frame dict from a .npy file."""
    data = np.load(path, allow_pickle=True).item()
    return data


def extract_voxel(frame: dict) -> np.ndarray:
    """Extract voxel array, handling MineDojo dict structure."""
    voxel = frame["voxel"]
    if isinstance(voxel, dict) and "block_name" in voxel:
        voxel = voxel["block_name"]
    return voxel


def format_action(action) -> str:
    """Format action vector as text."""
    if action is None:
        return "action: (none)"
    arr = np.array(action).tolist()
    return f"action: straight={arr[0]} pan={arr[1]} jump={arr[2]}"


def center_crop(voxel: np.ndarray, dim: int) -> np.ndarray:
    """Crop an odd-sized cube around the center of the voxel grid."""
    if dim % 2 == 0:
        raise ValueError("--dim must be odd")
    size = voxel.shape[0]
    if dim > size:
        raise ValueError(f"--dim {dim} exceeds voxel size {size}")
    half = dim // 2
    center = size // 2
    return voxel[
        center - half : center + half + 1,
        center - half : center + half + 1,
        center - half : center + half + 1,
    ]


def build_color_map(blocks: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
    """Assign a unique color to each block type."""
    unique_blocks = sorted(b for b in set(blocks.flatten().tolist()) if b != "air")
    cmap = plt.get_cmap("tab20")
    colors = {}
    for idx, block in enumerate(unique_blocks):
        colors[block] = cmap(idx % cmap.N)
    return colors


def render_static(voxel: np.ndarray, dim: int, action_text: str | None = None, output: Path | None = None) -> None:
    """Render a single frame."""
    color_map = build_color_map(np.array(list(set(voxel.flatten().tolist()))))
    voxel, xs, ys, zs, labels, colors = frame_to_points(voxel, dim, color_map)

    if not xs:
        print("No non-air blocks to plot.")
        return

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Map X/Z to the ground plane, Y to vertical; use squares.
    scatter = ax.scatter(xs, zs, ys, c=colors, s=60, depthshade=True, marker="s")

    ax.set_xlabel("X (forward/back)")
    ax.set_ylabel("Z (left/right)")
    ax.set_zlabel("Y (height)")
    title = "Minecraft Voxel Frame"
    if action_text:
        title = f"{title}\n{action_text}"
    ax.set_title(title)
    ax.set_xlim(0, voxel.shape[0])
    ax.set_ylim(0, voxel.shape[2])
    ax.set_zlim(0, voxel.shape[1])
    ax.set_box_aspect((1, 1, 1))

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=block, markerfacecolor=color_map[block], markersize=10)
        for block in color_map
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=200)
        print(f"Saved frame to {output}")
    else:
        if _is_interactive_backend():
            plt.show()
        else:
            fallback = Path("frame.png")
            plt.savefig(fallback, dpi=200)
            print(f"Backend is non-interactive; saved frame to {fallback}")


def frame_to_points(voxel: np.ndarray, dim: int, color_map: Dict[str, Tuple[float, float, float]]):
    """Helper to crop voxel and build scatter inputs."""
    voxel = center_crop(voxel, dim)
    xs, ys, zs, labels = [], [], [], []
    for x in range(voxel.shape[0]):
        for y in range(voxel.shape[1]):
            for z in range(voxel.shape[2]):
                block = voxel[x, y, z]
                if block == "air":
                    continue
                xs.append(x)
                ys.append(y)
                zs.append(z)
                labels.append(block)
    colors = [color_map[b] for b in labels] if labels else []
    return voxel, xs, ys, zs, labels, colors


def animate_frames(voxels: list[np.ndarray], actions: list[str], dim: int, output: Path | None = None) -> None:
    """Animate a sequence of frames."""
    # Build a stable color map across all frames.
    all_blocks = {b for v in voxels for b in v.flatten().tolist() if b != "air"}
    color_map = build_color_map(np.array(list(all_blocks)))

    voxel0, xs0, ys0, zs0, labels0, colors0 = frame_to_points(voxels[0], dim, color_map)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(xs0, zs0, ys0, c=colors0, s=60, depthshade=True, marker="s")

    ax.set_xlabel("X (forward/back)")
    ax.set_ylabel("Z (left/right)")
    ax.set_zlabel("Y (height)")
    ax.set_xlim(0, voxel0.shape[0])
    ax.set_ylim(0, voxel0.shape[2])
    ax.set_zlim(0, voxel0.shape[1])
    ax.set_box_aspect((1, 1, 1))

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=block, markerfacecolor=color_map[block], markersize=10)
        for block in color_map
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    def update(frame_idx: int):
        voxel, xs, ys, zs, labels, colors = frame_to_points(voxels[frame_idx], dim, color_map)
        if xs:
            scatter._offsets3d = (xs, zs, ys)
            scatter.set_color(colors)
        else:
            scatter._offsets3d = ([], [], [])
            scatter.set_color([])
        action_text = actions[frame_idx] if frame_idx < len(actions) else ""
        ax.set_title(f"Frame {frame_idx}\n{action_text}")
        return scatter,

    anim = animation.FuncAnimation(fig, update, frames=len(voxels), interval=500, blit=False, repeat=True)
    # Keep references to avoid garbage collection before rendering.
    fig._anim = anim  # type: ignore[attr-defined]
    _ANIMATIONS.append(anim)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        anim.save(output, writer="pillow", dpi=120)
        print(f"Saved animation to {output}")
    else:
        if _is_interactive_backend():
            plt.show()
        else:
            fallback = Path("animation.gif")
            anim.save(fallback, writer="pillow", dpi=120)
            print(f"Backend is non-interactive; saved animation to {fallback}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D plot of Minecraft voxel frame.")
    parser.add_argument("--path", "-p", type=Path, required=True, help="Path to frame .npy file or creative directory")
    parser.add_argument("--dim", "-d", type=int, default=5, help="Odd cube size to visualize (default 5)")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional output file. For a single frame: PNG path. For animation: GIF path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.path
    if target.is_dir():
        npy_files = sorted(target.glob("*.npy"))
        if not npy_files:
            raise ValueError(f"No .npy files found in directory {target}")
        frames = [load_frame(p) for p in npy_files]
        voxels = [extract_voxel(f) for f in frames]
        actions = [format_action(f.get("action")) for f in frames]
        animate_frames(voxels, actions, args.dim, args.output)
    else:
        frame = load_frame(target)
        voxel = extract_voxel(frame)
        action_text = format_action(frame.get("action"))
        render_static(voxel, args.dim, action_text, args.output)


if __name__ == "__main__":
    main()
