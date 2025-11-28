from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()

RESET = "\033[0m"
GREEN = "\033[92m"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

MODEL_SIZE_ORDER = [0.6, 1.7, 4, 8, 14, 32]


def model_sort_key(model: str) -> tuple:
    size = parse_model_size(model)
    try:
        idx = MODEL_SIZE_ORDER.index(size)
    except ValueError:
        idx = len(MODEL_SIZE_ORDER)
    return (idx, size, model)


def display_len(text: str) -> int:
    return len(ANSI_RE.sub("", text))


def pad_ansi(text: str, width: int) -> str:
    """Left align accounting for ANSI sequences."""
    extra = len(text) - display_len(text)
    return f"{text:<{width + extra}}"


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [max(display_len(h), *(display_len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    lines = []
    header_line = "  ".join(pad_ansi(h, widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    for row in rows:
        lines.append("  ".join(pad_ansi(val, widths[i]) for i, val in enumerate(row)))
    return "\n".join(lines)


def load_recon_csv(path: Path) -> Tuple[str, int, float, float]:
    """Return model, size, mean_bleu, mean_similarity."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    model = rows[0][1]
    size = int(rows[1][1])
    mean_bleu = float(rows[2][1])
    mean_sim = float(rows[3][1])
    return model, size, mean_bleu, mean_sim


def parse_model_size(model: str) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)b", model)
    return float(match.group(1)) if match else math.inf


def linear_regress(xs: List[float], ys: List[float]):
    xs_arr = np.array(xs, dtype=float)
    ys_arr = np.array(ys, dtype=float)
    mask = ~np.isnan(xs_arr) & ~np.isnan(ys_arr)
    xs_arr = xs_arr[mask]
    ys_arr = ys_arr[mask]
    if xs_arr.size < 2:
        return None
    try:
        from scipy.stats import linregress

        res = linregress(xs_arr, ys_arr)
        return res.slope, res.intercept, res.rvalue**2, res.pvalue
    except Exception:
        slope, intercept = np.polyfit(xs_arr, ys_arr, 1)
        preds = slope * xs_arr + intercept
        ss_res = float(np.sum((ys_arr - preds) ** 2))
        ss_tot = float(np.sum((ys_arr - np.mean(ys_arr)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return slope, intercept, r2, float("nan")


def fmt(val: float, best: float) -> str:
    if val is None or math.isnan(val):
        return "--"
    text = f"{val:.2f}"
    return f"{GREEN}{text}{RESET}" if math.isclose(val, best, rel_tol=1e-9, abs_tol=1e-12) else text


def styled_num(val: float, best: float) -> Text:
    if val is None or math.isnan(val):
        return Text("--")
    style = "green" if math.isclose(val, best, rel_tol=1e-9, abs_tol=1e-12) else ""
    return Text(f"{val:.2f}", style=style)


def delta_text(val: float, baseline: float | None) -> Text:
    if baseline is None or val is None or math.isnan(val) or math.isnan(baseline):
        return Text("--")
    diff = val - baseline
    sign = "+" if diff > 0 else ("-" if diff < 0 else " ")
    style = "green" if diff > 0 else ("red" if diff < 0 else "")
    return Text(f"{sign}{abs(diff):.2f}", style=style)


def print_table(rows: List[Dict[str, float]]) -> None:
    if not rows:
        print("No frame reconstruction evals found.")
        return

    # Baseline: qwen3-0.6b at smallest available voxel size
    baseline_size = min(r["size"] for r in rows if r["model"] == "qwen3-0.6b")
    baseline_rows = [r for r in rows if r["model"] == "qwen3-0.6b" and r["size"] == baseline_size]
    baseline_bleu = baseline_rows[0]["bleu"] if baseline_rows else None
    baseline_sim = baseline_rows[0]["sim"] if baseline_rows else None

    best_bleu = max(r["bleu"] for r in rows)
    best_sim = max(r["sim"] for r in rows)

    table = Table(title="Frame Reconstruction Metrics (baseline: qwen3-0.6b @ min size)", show_header=True, header_style="bold")
    table.add_column("Model", justify="left")
    table.add_column("Voxel size", justify="right")
    table.add_column("BLEU", justify="right")
    table.add_column("Δ BLEU", justify="right")
    table.add_column("Similarity", justify="right")
    table.add_column("Δ Sim", justify="right")
    for r in sorted(rows, key=lambda x: (model_sort_key(x["model"]), x["size"])):
        table.add_row(
            r["model"],
            str(r["size"]),
            styled_num(r["bleu"], best_bleu),
            delta_text(r["bleu"], baseline_bleu),
            styled_num(r["sim"], best_sim),
            delta_text(r["sim"], baseline_sim),
        )
    console.print(table)


def print_coverage(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    expected_sizes = [3, 5, 7, 9, 11, 13]
    models = sorted({r["model"] for r in rows}, key=model_sort_key)
    print("Coverage by model (present/expected voxel sizes):")
    for model in models:
        have = sorted({r["size"] for r in rows if r["model"] == model})
        missing = [s for s in expected_sizes if s not in have]
        print(f"  {model}: {len(have)}/{len(expected_sizes)} sizes present; missing {missing if missing else 'none'}")


def print_aggregates(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    header = ["Model", "Avg BLEU", "Avg Similarity", "N sizes"]
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for r in rows:
        grouped.setdefault(r["model"], []).append(r)

    best_bleu = max(np.mean([x["bleu"] for x in v]) for v in grouped.values())
    best_sim = max(np.mean([x["sim"] for x in v]) for v in grouped.values())
    baseline_bleu = np.mean([x["bleu"] for x in grouped.get("qwen3-0.6b", [])]) if grouped.get("qwen3-0.6b") else None
    baseline_sim = np.mean([x["sim"] for x in grouped.get("qwen3-0.6b", [])]) if grouped.get("qwen3-0.6b") else None
    table_rows = []
    for model in sorted(grouped.keys(), key=model_sort_key):
        vals = grouped[model]
        avg_bleu = float(np.mean([x["bleu"] for x in vals]))
        avg_sim = float(np.mean([x["sim"] for x in vals]))
        table_rows.append(
            [
                model,
                avg_bleu,
                avg_sim,
                str(len(vals)),
            ]
        )
    table = Table(title="Per-model averages (Δ vs qwen3-0.6b avg)", show_header=True, header_style="bold")
    table.add_column("Model", justify="left")
    table.add_column("Avg BLEU", justify="right")
    table.add_column("Δ BLEU", justify="right")
    table.add_column("Avg Similarity", justify="right")
    table.add_column("Δ Sim", justify="right")
    table.add_column("N sizes", justify="right")
    for row in table_rows:
        table.add_row(
            row[0],
            styled_num(float(row[1]), best_bleu),
            delta_text(float(row[1]), baseline_bleu),
            styled_num(float(row[2]), best_sim),
            delta_text(float(row[2]), baseline_sim),
            row[3],
        )
    console.print(table)


def print_volatility(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    table = Table(title="Per-model volatility (std across sizes, Δ vs qwen3-0.6b)", show_header=True, header_style="bold")
    table.add_column("Model", justify="left")
    table.add_column("BLEU std", justify="right")
    table.add_column("Δ BLEU std", justify="right")
    table.add_column("Similarity std", justify="right")
    table.add_column("Δ Sim std", justify="right")
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for r in rows:
        grouped.setdefault(r["model"], []).append(r)
    base_vals = grouped.get("qwen3-0.6b", [])
    base_bleu_std = float(np.std([x["bleu"] for x in base_vals])) if base_vals else None
    base_sim_std = float(np.std([x["sim"] for x in base_vals])) if base_vals else None
    for model in sorted(grouped.keys(), key=model_sort_key):
        vals = grouped[model]
        bleu_std = float(np.std([x["bleu"] for x in vals])) if vals else float("nan")
        sim_std = float(np.std([x["sim"] for x in vals])) if vals else float("nan")
        table.add_row(
            model,
            f"{bleu_std:.2f}",
            delta_text(bleu_std, base_bleu_std),
            f"{sim_std:.2f}",
            delta_text(sim_std, base_sim_std),
        )
    console.print(table)


def print_leaderboards(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    sizes = sorted({r["size"] for r in rows})
    table = Table(title="Best model per voxel size", show_header=True, header_style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Top BLEU", justify="left")
    table.add_column("Top Similarity", justify="left")
    for s in sizes:
        subset = [r for r in rows if r["size"] == s]
        if not subset:
            continue
        best_bleu = max(subset, key=lambda x: x["bleu"])
        best_sim = max(subset, key=lambda x: x["sim"])
        table.add_row(
            str(s),
            f"{best_bleu['model']} ({best_bleu['bleu']:.2f})",
            f"{best_sim['model']} ({best_sim['sim']:.2f})",
        )
    console.print(table)

    table2 = Table(title="Best voxel size per model", show_header=True, header_style="bold")
    table2.add_column("Model", justify="left")
    table2.add_column("Top BLEU @size", justify="left")
    table2.add_column("Top Sim @size", justify="left")
    models = sorted({r["model"] for r in rows}, key=model_sort_key)
    for m in models:
        subset = [r for r in rows if r["model"] == m]
        if not subset:
            continue
        best_bleu = max(subset, key=lambda x: x["bleu"])
        best_sim = max(subset, key=lambda x: x["sim"])
        table2.add_row(
            m,
            f"{best_bleu['bleu']:.2f} @ {best_bleu['size']}",
            f"{best_sim['sim']:.2f} @ {best_sim['size']}",
        )
    console.print(table2)


def print_joint_leader(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    table = Table(
        title="Joint leader per voxel size (mean of BLEU and Similarity)",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Size", justify="right")
    table.add_column("Top (BLEU+Sim)/2", justify="left")
    sizes = sorted({r["size"] for r in rows})
    for s in sizes:
        subset = [r for r in rows if r["size"] == s]
        if not subset:
            continue
        best = max(subset, key=lambda x: (x["bleu"] + x["sim"]) / 2.0)
        score = (best["bleu"] + best["sim"]) / 2.0
        table.add_row(str(s), f"{best['model']} ({score:.2f})")
    console.print(table)


def print_regressions(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    metrics = ["bleu", "sim"]
    labels = {"bleu": "BLEU", "sim": "Similarity"}
    voxel_sizes = [r["size"] for r in rows]
    model_sizes = [parse_model_size(r["model"]) for r in rows]
    for metric in metrics:
        vals = [r[metric] for r in rows]
        reg_voxel = linear_regress(voxel_sizes, vals)
        reg_model = linear_regress(model_sizes, vals)
        if reg_voxel:
            s, i, r2, p = reg_voxel
            print(
                f"{labels[metric]} vs voxel size: slope {s:.2f}, intercept {i:.2f}, R^2 {r2:.2f}, p {p:.2f}"
            )
        if reg_model:
            s, i, r2, p = reg_model
            print(
                f"{labels[metric]} vs model size: slope {s:.2f}, intercept {i:.2f}, R^2 {r2:.2f}, p {p:.2f}"
            )


def main() -> None:
    eval_dir = Path("evals")
    csv_paths = sorted(eval_dir.glob("*_frame_reconstruction_eval.csv"))
    rows = []
    for path in csv_paths:
        try:
            model, size, bleu, sim = load_recon_csv(path)
            rows.append({"model": model, "size": size, "bleu": bleu, "sim": sim})
        except Exception:
            continue

    print("=== Frame Reconstruction Report ===")
    print_table(rows)
    print()
    print_coverage(rows)
    print()
    print_aggregates(rows)
    print()
    print_leaderboards(rows)
    print()
    print_joint_leader(rows)
    print()
    print_volatility(rows)
    print()
    print_regressions(rows)


if __name__ == "__main__":
    main()
