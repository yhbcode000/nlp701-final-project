from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Find header
    header_idx = None
    header = []
    for idx, row in enumerate(rows):
        if row and row[0] == "index":
            header_idx = idx
            header = row
            break
    if header_idx is None:
        raise ValueError("No data header found in CSV (expected a row starting with 'index').")

    data_rows = []
    for row in rows[header_idx + 1 :]:
        if not row or not row[0].strip():
            continue
        entry = {header[i]: row[i] if i < len(row) else "" for i in range(len(header))}
        data_rows.append(entry)
    return data_rows


def unescape_text(text: str) -> str:
    return (text or "").replace("\\n", "\n").strip()


def text_to_cube(text: str) -> np.ndarray:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and "|" in ln]
    if not lines:
        raise ValueError("No grid lines found in text.")

    # Infer size from the maximum token count per line (more robust) and line count.
    token_counts = [len([tok for tok in ln.split("|") if tok]) for ln in lines]
    max_tokens = max(token_counts) if token_counts else 0
    line_count_guess = int(round(math.sqrt(len(lines)))) or 1
    size = max(1, max_tokens, line_count_guess)

    target_lines = size * size
    lines = lines[:target_lines]
    filler_line = "|" + "|".join(["air"] * size) + "|"
    while len(lines) < target_lines:
        lines.append(filler_line)

    cube = np.full((size, size, size), "air", dtype=object)
    for line_idx, line in enumerate(lines):
        tokens = [tok for tok in line.split("|") if tok]
        if len(tokens) < size:
            tokens += ["air"] * (size - len(tokens))
        tokens = tokens[:size]
        x = line_idx % size
        y = line_idx // size
        for z in range(size):
            cube[x, y, z] = tokens[z]
    return cube


def sync_views(ax_left, ax_right):
    syncing = {"flag": False}

    def on_motion(event):
        if event.inaxes not in {ax_left, ax_right} or syncing["flag"]:
            return
        source = event.inaxes
        target = ax_right if source is ax_left else ax_left
        syncing["flag"] = True
        target.view_init(elev=source.elev, azim=source.azim)
        target.set_xlim3d(source.get_xlim3d())
        target.set_ylim3d(source.get_ylim3d())
        target.set_zlim3d(source.get_zlim3d())
        target.figure.canvas.draw_idle()
        syncing["flag"] = False

    ax_left.figure.canvas.mpl_connect("motion_notify_event", on_motion)
    ax_right.figure.canvas.mpl_connect("motion_notify_event", on_motion)


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


def build_color_map_from_blocks(blocks: List[str], colors_path: Path) -> Dict[str, Tuple[float, float, float]]:
    stored = load_color_map(colors_path)
    updated = False

    # Start palette from matplotlib tab20 for unknown blocks
    tab20 = plt.get_cmap("tab20")
    palette = [tab20(i % tab20.N) for i in range(len(blocks) + 10)]

    color_hex: Dict[str, str] = dict(stored)

    for idx, block in enumerate(blocks):
        if block in color_hex:
            continue
        if block.lower() in KNOWN_COLORS:
            color_hex[block] = KNOWN_COLORS[block.lower()]
        else:
            r, g, b, _ = palette[idx]
            color_hex[block] = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        updated = True

    if updated:
        save_color_map(colors_path, color_hex)

    # Convert to rgb tuples
    return {blk: _hex_to_rgb_tuple(hex_color) for blk, hex_color in color_hex.items()}


def cube_to_points(cube: np.ndarray, color_map: Dict[str, Tuple[float, float, float]]):
    xs, ys, zs, labels = [], [], [], []
    for x in range(cube.shape[0]):
        for y in range(cube.shape[1]):
            for z in range(cube.shape[2]):
                block = cube[x, y, z]
                if block == "air":
                    continue
                xs.append(x)
                ys.append(y)
                zs.append(z)
                labels.append(block)
    colors = [color_map[b] for b in labels] if labels else []
    return xs, ys, zs, labels, colors


def draw_cubes(ax, cube: np.ndarray, color_map: Dict[str, Tuple[float, float, float]], alpha: float = 0.5):
    """Render each non-air block as a translucent cube of size 1."""
    xs, ys, zs, labels, _ = cube_to_points(cube, color_map)
    if not xs:
        return False

    for x, y, z, label in zip(xs, ys, zs, labels):
        r, g, b = color_map.get(label, (0.5, 0.5, 0.5))
        ax.bar3d(
            x,
            z,
            y,
            1,
            1,
            1,
            color=(r, g, b, alpha),
            edgecolor=(r, g, b, alpha),
            shade=True,
        )
    return True


def draw_wireframe(ax, cube: np.ndarray, color: str = "#cccccc"):
    """Draw a light wireframe box to show volume when no blocks are present."""
    s = cube.shape[0]
    # corners of a cube
    corners = [
        (0, 0, 0),
        (s, 0, 0),
        (s, s, 0),
        (0, s, 0),
        (0, 0, s),
        (s, 0, s),
        (s, s, s),
        (0, s, s),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i, j in edges:
        x0, y0, z0 = corners[i]
        x1, y1, z1 = corners[j]
        ax.plot([x0, x1], [z0, z1], [y0, y1], color=color, linewidth=1, alpha=0.6)


def plot_comparison(
    label_cube: np.ndarray,
    pred_cube: np.ndarray,
    title_prefix: str = "",
    output: Path | None = None,
    colors_path: Path = DEFAULT_COLORS_FILE,
) -> None:
    # Build a shared color map across both cubes (sorted for stability)
    blocks = sorted({b for b in label_cube.flatten().tolist() if b != "air"}.union(
        {b for b in pred_cube.flatten().tolist() if b != "air"}
    ))
    color_map = build_color_map_from_blocks(blocks, colors_path)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    for ax, cube, subtitle in [(ax1, label_cube, "Ground Truth"), (ax2, pred_cube, "Prediction")]:
        has_blocks = draw_cubes(ax, cube, color_map, alpha=0.5)
        if not has_blocks:
            draw_wireframe(ax, cube)
            ax.text(0.5, 0.5, 0.5, "All air / no blocks", ha="center", va="center")
        ax.set_xlabel("X (forward/back)")
        ax.set_ylabel("Z (left/right)")
        ax.set_zlabel("Y (height)")
        ax.set_xlim(0, cube.shape[0])
        ax.set_ylim(0, cube.shape[2])
        ax.set_zlim(0, cube.shape[1])
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"{title_prefix}{subtitle}")

    # Legend similar to show_data
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=block,
            markerfacecolor=color_map[block],
            markersize=10,
        )
        for block in color_map
    ]
    if handles:
        ax2.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")

    sync_views(ax1, ax2)
    plt.tight_layout()

    # Decide how to render/output
    target_path: Path | None = output
    if not _is_interactive_backend() and target_path is None:
        target_path = Path("comparison.png")

    if target_path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(target_path, dpi=200)
        print(f"Saved comparison plot to {target_path}")
    else:
        plt.show()


def resolve_csv_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    candidate = Path("evals") / name_or_path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"CSV file not found: {name_or_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare reconstruction prediction vs label with synchronized 3D views."
    )
    parser.add_argument(
        "--csv",
        help="Path or filename of the reconstruction eval CSV (if filename, looked up in evals/).",
    )
    parser.add_argument(
        "--model",
        help="Model key to locate CSV under evals/ (e.g., qwen3-0.6b). Requires --size.",
    )
    parser.add_argument(
        "--size",
        type=int,
        help="Voxel size when resolving CSV by model.",
    )
    parser.add_argument(
        "--idx",
        type=int,
        help="Global index value to select a row (matches CSV 'index' column).",
    )
    parser.add_argument(
        "--episode",
        help="Episode name to disambiguate rows with the same index (matches CSV 'episode' column).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the comparison plot (PNG). If omitted, tries to show the plot.",
    )
    parser.add_argument(
        "--colors",
        type=Path,
        default=DEFAULT_COLORS_FILE,
        help="Path to JSON file storing block→color mappings (auto-updated).",
    )
    args = parser.parse_args()

    if args.csv:
        csv_path = resolve_csv_path(args.csv)
    else:
        if not (args.model and args.size):
            raise ValueError("Provide either --csv, or both --model and --size.")
        inferred = Path("evals") / f"{args.model}_size{args.size}_frame_reconstruction_eval.csv"
        if not inferred.exists():
            raise FileNotFoundError(f"Inferred CSV not found: {inferred}")
        csv_path = inferred
    data_rows = parse_csv(csv_path)
    if not data_rows:
        raise ValueError(f"No data rows found in {csv_path}")

    if args.idx is not None:
        candidates = [r for r in data_rows if str(r.get("index")) == str(args.idx)]
        if args.episode:
            ep_arg = str(args.episode)
            candidates = [
                r for r in candidates
                if str(r.get("episode")) == ep_arg or str(r.get("episode")).endswith(ep_arg)
            ]
        if not candidates:
            raise ValueError(f"No row found with index={args.idx} and episode={args.episode or 'any'}")
        row = candidates[0]
    else:
        row = data_rows[0]
    z_label_text = unescape_text(row.get("z_label", ""))
    z_pred_text = unescape_text(row.get("z_clean_prediction", ""))

    label_cube = text_to_cube(z_label_text)
    pred_cube = text_to_cube(z_pred_text)

    title_prefix = ""
    if "episode" in row and "index" in row:
        title_prefix = f"ep {row.get('episode', '')} idx {row.get('index', '')}\n"

    out_path = args.output
    if out_path is None and args.model and args.size:
        ep = row.get("episode", "")
        idx = row.get("index", "")
        out_path = Path("plots") / f"{args.model}_size{args.size}_ep{ep}_idx{idx}_frame_reconstruction_comparison.png"

    plot_comparison(
        label_cube,
        pred_cube,
        title_prefix=title_prefix,
        output=out_path,
        colors_path=args.colors,
    )


if __name__ == "__main__":
    main()
