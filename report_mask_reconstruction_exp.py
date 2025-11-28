from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

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
    extra = len(text) - display_len(text)
    return f"{text:<{width + extra}}"


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [max(display_len(h), *(display_len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    lines = []
    lines.append("  ".join(pad_ansi(h, widths[i]) for i, h in enumerate(headers)))
    for row in rows:
        lines.append("  ".join(pad_ansi(val, widths[i]) for i, val in enumerate(row)))
    return "\n".join(lines)


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


def collect_metrics() -> List[Dict[str, float]]:
    eval_dir = Path("evals")
    csv_paths = sorted(eval_dir.glob("*_mask_reconstruction_eval.csv"))
    grouped: Dict[str, Dict[int, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for path in csv_paths:
        try:
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    model = row["model"]
                    mask_size = int(row["mask_size"])
                    bleu = float(row["bleu"])
                    sim = float(row["similarity_masked"])
                    grouped[model][mask_size].append({"bleu": bleu, "sim": sim})
        except Exception:
            continue

    rows = []
    for model, mask_map in grouped.items():
        for mask_size, vals in mask_map.items():
            if not vals:
                continue
            bleu_vals = [v["bleu"] for v in vals]
            sim_vals = [v["sim"] for v in vals]
            rows.append(
                {
                    "model": model,
                    "mask_size": mask_size,
                    "bleu": float(np.mean(bleu_vals)),
                    "sim": float(np.mean(sim_vals)),
                }
            )
    return rows


def print_table(rows: List[Dict[str, float]]) -> None:
    if not rows:
        print("No mask reconstruction evals found.")
        return

    best_bleu = max(r["bleu"] for r in rows)
    best_sim = max(r["sim"] for r in rows)
    baseline_mask = min(r["mask_size"] for r in rows if r["model"] == "qwen3-0.6b") if any(r["model"] == "qwen3-0.6b" for r in rows) else None
    baseline_row = next((r for r in rows if r["model"] == "qwen3-0.6b" and r["mask_size"] == baseline_mask), None)
    base_bleu = baseline_row["bleu"] if baseline_row else None
    base_sim = baseline_row["sim"] if baseline_row else None

    table = Table(title="Mask Reconstruction Metrics (baseline: qwen3-0.6b @ min mask)", show_header=True, header_style="bold")
    table.add_column("Model", justify="left")
    table.add_column("Mask size", justify="right")
    table.add_column("BLEU", justify="right")
    table.add_column("Δ BLEU", justify="right")
    table.add_column("Similarity", justify="right")
    table.add_column("Δ Sim", justify="right")
    for r in sorted(rows, key=lambda x: (model_sort_key(x["model"]), x["mask_size"])):
        table.add_row(
            r["model"],
            str(r["mask_size"]),
            styled_num(r["bleu"], best_bleu),
            delta_text(r["bleu"], base_bleu),
            styled_num(r["sim"], best_sim),
            delta_text(r["sim"], base_sim),
        )
    console.print(table)


def print_coverage(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    expected_masks = [1, 2, 3, 4, 5, 6]
    models = sorted({r["model"] for r in rows}, key=model_sort_key)
    table = Table(title="Coverage by model (present/expected mask sizes)", show_header=True, header_style="bold")
    table.add_column("Model", justify="left")
    table.add_column("Present/Expected", justify="right")
    table.add_column("Missing", justify="left")
    for model in models:
        have = sorted({r["mask_size"] for r in rows if r["model"] == model})
        missing = [m for m in expected_masks if m not in have]
        table.add_row(model, f"{len(have)}/{len(expected_masks)}", str(missing if missing else "none"))
    console.print(table)


def print_regressions(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    metrics = ["bleu", "sim"]
    labels = {"bleu": "BLEU", "sim": "Similarity"}
    mask_sizes = [r["mask_size"] for r in rows]
    model_sizes = [parse_model_size(r["model"]) for r in rows]
    for metric in metrics:
        vals = [r[metric] for r in rows]
        reg_mask = linear_regress(mask_sizes, vals)
        reg_model = linear_regress(model_sizes, vals)
        if reg_mask:
            s, i, r2, p = reg_mask
            print(
                f"{labels[metric]} vs mask size: slope {s:.2f}, intercept {i:.2f}, R^2 {r2:.2f}, p {p:.2f}"
            )
        if reg_model:
            s, i, r2, p = reg_model
            print(
                f"{labels[metric]} vs model size: slope {s:.2f}, intercept {i:.2f}, R^2 {r2:.2f}, p {p:.2f}"
        )


def print_aggregates(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    header = ["Model", "Avg BLEU", "Avg Similarity", "N masks"]
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for r in rows:
        grouped.setdefault(r["model"], []).append(r)

    best_bleu = max(np.mean([x["bleu"] for x in v]) for v in grouped.values())
    best_sim = max(np.mean([x["sim"] for x in v]) for v in grouped.values())
    base_bleu = np.mean([x["bleu"] for x in grouped.get("qwen3-0.6b", [])]) if grouped.get("qwen3-0.6b") else None
    base_sim = np.mean([x["sim"] for x in grouped.get("qwen3-0.6b", [])]) if grouped.get("qwen3-0.6b") else None
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
    table.add_column("N masks", justify="right")
    for row in table_rows:
        table.add_row(
            row[0],
            styled_num(float(row[1]), best_bleu),
            delta_text(float(row[1]), base_bleu),
            styled_num(float(row[2]), best_sim),
            delta_text(float(row[2]), base_sim),
            row[3],
        )
    console.print(table)


def print_volatility(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    table = Table(title="Per-model volatility (std across masks, Δ vs qwen3-0.6b)", show_header=True, header_style="bold")
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
    masks = sorted({r["mask_size"] for r in rows})
    table = Table(title="Best model per mask size", show_header=True, header_style="bold")
    table.add_column("Mask", justify="right")
    table.add_column("Top BLEU", justify="left")
    table.add_column("Top Similarity", justify="left")
    for m in masks:
        subset = [r for r in rows if r["mask_size"] == m]
        if not subset:
            continue
        best_bleu = max(subset, key=lambda x: x["bleu"])
        best_sim = max(subset, key=lambda x: x["sim"])
        table.add_row(
            str(m),
            f"{best_bleu['model']} ({best_bleu['bleu']:.2f})",
            f"{best_sim['model']} ({best_sim['sim']:.2f})",
        )
    console.print(table)

    table2 = Table(title="Best mask size per model", show_header=True, header_style="bold")
    table2.add_column("Model", justify="left")
    table2.add_column("Top BLEU @mask", justify="left")
    table2.add_column("Top Sim @mask", justify="left")
    models = sorted({r["model"] for r in rows}, key=model_sort_key)
    for model in models:
        subset = [r for r in rows if r["model"] == model]
        if not subset:
            continue
        best_bleu = max(subset, key=lambda x: x["bleu"])
        best_sim = max(subset, key=lambda x: x["sim"])
        table2.add_row(
            model,
            f"{best_bleu['bleu']:.2f} @ {best_bleu['mask_size']}",
            f"{best_sim['sim']:.2f} @ {best_sim['mask_size']}",
        )
    console.print(table2)


def main() -> None:
    print("=== Mask Reconstruction Report ===")
    rows = collect_metrics()
    print_table(rows)
    print()
    print_coverage(rows)
    print()
    print_aggregates(rows)
    print()
    print_volatility(rows)
    print()
    print_leaderboards(rows)
    print()
    print_regressions(rows)


if __name__ == "__main__":
    main()
