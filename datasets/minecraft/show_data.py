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
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)

_ANIMATIONS: list[animation.FuncAnimation] = []
_NON_INTERACTIVE_BACKENDS = {"agg", "cairoagg", "svg", "pdf", "ps"}
DEFAULT_COLORS_FILE = Path("color.json")
KNOWN_COLORS = {
    "grass": "#4caf50",
    "grass block": "#4caf50",
    "dirt": "#8b5a2b",
    "stone": "#7d7d7d",
    "wood": "#a67c52",
    "log": "#a67c52",
    "leaves": "#2e8b57",
    "flower": "#ff69b4",
    "water": "#1e90ff",
    "sand": "#d2b48c",
}


def _is_interactive_backend() -> bool:
    backend = plt.get_backend().lower()
    return not any(tag in backend for tag in _NON_INTERACTIVE_BACKENDS)


def _hex_to_rgb_tuple(hex_str: str) -> Tuple[float, float, float]:
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join([c * 2 for c in hex_str])
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return (r, g, b)


def load_color_map(colors_path: Path) -> Dict[str, str]:
    if colors_path.exists():
        try:
            with colors_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {k: v for k, v in data.items() if isinstance(v, str)}
        except Exception:
            pass
    return {}


def save_color_map(colors_path: Path, color_map: Dict[str, str]) -> None:
    colors_path.parent.mkdir(parents=True, exist_ok=True)
    with colors_path.open("w", encoding="utf-8") as f:
        json.dump(color_map, f, indent=2, ensure_ascii=False)

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


def build_color_map(blocks: np.ndarray, colors_path: Path = DEFAULT_COLORS_FILE) -> Dict[str, Tuple[float, float, float]]:
    """Assign a persistent color to each block type using color.json."""
    unique_blocks = sorted(b for b in set(blocks.flatten().tolist()) if b != "air")
    stored = load_color_map(colors_path)
    updated = False

    cmap = plt.get_cmap("tab20")
    color_hex: Dict[str, str] = dict(stored)

    for idx, block in enumerate(unique_blocks):
        if block in color_hex:
            continue
        if block.lower() in KNOWN_COLORS:
            color_hex[block] = KNOWN_COLORS[block.lower()]
        else:
            r, g, b, _ = cmap(idx % cmap.N)
            color_hex[block] = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        updated = True

    if updated:
        save_color_map(colors_path, color_hex)

    return {blk: _hex_to_rgb_tuple(hex_color) for blk, hex_color in color_hex.items()}


def render_static(voxel: np.ndarray, dim: int, action_text: str | None = None, output: Path | None = None) -> None:
    """Render a single frame."""
    color_map = build_color_map(np.array(list(set(voxel.flatten().tolist()))))
    voxel, xs, ys, zs, labels, colors = frame_to_points(voxel, dim, color_map)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    if xs:
        for x, y, z, label in zip(xs, ys, zs, labels):
            r, g, b = color_map.get(label, (0.5, 0.5, 0.5))
            ax.bar3d(
                x,
                z,
                y,
                1,
                1,
                1,
                color=(r, g, b, 0.5),
                edgecolor=(r, g, b, 0.5),
                shade=True,
            )
    else:
        print("No non-air blocks to plot.")

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
    bars = []
    if xs0:
        for x, y, z, label in zip(xs0, ys0, zs0, labels0):
            r, g, b = color_map.get(label, (0.5, 0.5, 0.5))
            bar = ax.bar3d(
                x,
                z,
                y,
                1,
                1,
                1,
                color=(r, g, b, 0.5),
                edgecolor=(r, g, b, 0.5),
                shade=True,
            )
            bars.append(bar)

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
        # Remove previous bars
        while bars:
            bar = bars.pop()
            try:
                bar.remove()
            except Exception:
                pass
        voxel, xs, ys, zs, labels, colors = frame_to_points(voxels[frame_idx], dim, color_map)
        if xs:
            for x, y, z, label in zip(xs, ys, zs, labels):
                r, g, b = color_map.get(label, (0.5, 0.5, 0.5))
                bar = ax.bar3d(
                    x,
                    z,
                    y,
                    1,
                    1,
                    1,
                    color=(r, g, b, 0.5),
                    edgecolor=(r, g, b, 0.5),
                    shade=True,
                )
                bars.append(bar)
        action_text = actions[frame_idx] if frame_idx < len(actions) else ""
        ax.set_title(f"Frame {frame_idx}\n{action_text}")
        return bars

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
