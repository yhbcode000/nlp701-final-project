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
    extra = len(text) - display_len(text)
    return f"{text:<{width + extra}}"


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [max(display_len(h), *(display_len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    lines = []
    lines.append("  ".join(pad_ansi(h, widths[i]) for i, h in enumerate(headers)))
    for row in rows:
        lines.append("  ".join(pad_ansi(val, widths[i]) for i, val in enumerate(row)))
    return "\n".join(lines)


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


def load_action_csv(path: Path) -> Tuple[str, int, Dict[str, float]]:
    """Return model, size, per-dim accuracy."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    model = rows[0][1]
    size = int(rows[1][1])

    accuracies: Dict[str, float] = {}
    for row in rows:
        if len(row) >= 2 and row[0] in {"straight", "pan", "jump"}:
            accuracies[row[0]] = float(row[1])
    return model, size, accuracies


def read_prediction_stats(path: Path) -> Tuple[int, int, float]:
    """Return (unique_predictions, total_rows, dominant_fraction)."""
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    header_idx = None
    header = None
    for i, row in enumerate(rows):
        if row and row[0] == "index":
            header_idx = i
            header = row
            break
    if header_idx is None:
        return 0, 0, float("nan")

    preds = []
    for row in rows[header_idx + 1 :]:
        if not row or not row[0].strip():
            continue
        rec = dict(zip(header, row))
        preds.append(rec.get("y_clean_prediction"))
    total = len(preds)
    if total == 0:
        return 0, 0, float("nan")
    counts = {}
    for p in preds:
        counts[p] = counts.get(p, 0) + 1
    dominant = max(counts.values())
    return len(counts), total, dominant / total


def collect_metrics() -> Tuple[List[Dict[str, float]], Dict[Tuple[str, int], Tuple[int, int, float]]]:
    eval_dir = Path("evals")
    csv_paths = sorted(eval_dir.glob("*_action_recognition_eval.csv"))
    rows = []
    pred_stats: Dict[Tuple[str, int], Tuple[int, int, float]] = {}
    for path in csv_paths:
        try:
            model, size, accs = load_action_csv(path)
            straight = accs.get("straight", float("nan"))
            pan = accs.get("pan", float("nan"))
            jump = accs.get("jump", float("nan"))
            avg = np.nanmean([straight, pan, jump])
            rows.append(
                {
                    "model": model,
                    "size": size,
                    "straight": straight,
                    "pan": pan,
                    "jump": jump,
                    "avg": float(avg),
                }
            )
            pred_stats[(model, size)] = read_prediction_stats(path)
        except Exception:
            continue
    return rows, pred_stats


def print_table(rows: List[Dict[str, float]]) -> None:
    if not rows:
        print("No action recognition evals found.")
        return

    metrics = ["straight", "pan", "jump", "avg"]
    bests = {m: max(r[m] for r in rows) for m in metrics}
    # Baseline: qwen3-0.6b at smallest size
    base_size = min(r["size"] for r in rows if r["model"] == "qwen3-0.6b") if any(r["model"] == "qwen3-0.6b" for r in rows) else None
    base_row = next((r for r in rows if r["model"] == "qwen3-0.6b" and r["size"] == base_size), None)
    base_vals = {
        "straight": base_row["straight"] if base_row else None,
        "pan": base_row["pan"] if base_row else None,
        "jump": base_row["jump"] if base_row else None,
        "avg": base_row["avg"] if base_row else None,
    }
    table = Table(title="Action Recognition Metrics (baseline: qwen3-0.6b @ min size)", show_header=True, header_style="bold")
    table.add_column("Model", justify="left")
    table.add_column("Voxel size", justify="right")
    table.add_column("Straight", justify="right")
    table.add_column("Δ Straight", justify="right")
    table.add_column("Pan", justify="right")
    table.add_column("Δ Pan", justify="right")
    table.add_column("Jump", justify="right")
    table.add_column("Δ Jump", justify="right")
    table.add_column("Avg", justify="right")
    table.add_column("Δ Avg", justify="right")
    for r in sorted(rows, key=lambda x: (model_sort_key(x["model"]), x["size"])):
        table.add_row(
            r["model"],
            str(r["size"]),
            styled_num(r["straight"], bests["straight"]),
            delta_text(r["straight"], base_vals["straight"]),
            styled_num(r["pan"], bests["pan"]),
            delta_text(r["pan"], base_vals["pan"]),
            styled_num(r["jump"], bests["jump"]),
            delta_text(r["jump"], base_vals["jump"]),
            styled_num(r["avg"], bests["avg"]),
            delta_text(r["avg"], base_vals["avg"]),
        )
    console.print(table)


def print_coverage(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    expected_sizes = [3, 5, 7, 9, 11, 13]
    models = sorted({r["model"] for r in rows}, key=model_sort_key)
    table = Table(title="Coverage by model (present/expected voxel sizes)", show_header=True, header_style="bold")
    table.add_column("Model", justify="left")
    table.add_column("Present/Expected", justify="right")
    table.add_column("Missing", justify="left")
    for model in models:
        have = sorted({r["size"] for r in rows if r["model"] == model})
        missing = [s for s in expected_sizes if s not in have]
        table.add_row(model, f"{len(have)}/{len(expected_sizes)}", str(missing if missing else "none"))
    console.print(table)


def print_aggregates(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for r in rows:
        grouped.setdefault(r["model"], []).append(r)

    metrics = ["straight", "pan", "jump", "avg"]
    bests = {m: max(np.nanmean([x[m] for x in v]) for v in grouped.values()) for m in metrics}
    header = ["Model", "Avg Straight", "Avg Pan", "Avg Jump", "Avg (overall)", "N sizes"]
    table_rows = []
    for model in sorted(grouped.keys(), key=model_sort_key):
        vals = grouped[model]
        avg_straight = float(np.nanmean([x["straight"] for x in vals]))
        avg_pan = float(np.nanmean([x["pan"] for x in vals]))
        avg_jump = float(np.nanmean([x["jump"] for x in vals]))
        avg_overall = float(np.nanmean([x["avg"] for x in vals]))
        table_rows.append(
            [
                model,
                avg_straight,
                avg_pan,
                avg_jump,
                avg_overall,
                str(len(vals)),
            ]
        )
    base_vals = grouped.get("qwen3-0.6b", [])
    base_s = float(np.nanmean([x["straight"] for x in base_vals])) if base_vals else None
    base_p = float(np.nanmean([x["pan"] for x in base_vals])) if base_vals else None
    base_j = float(np.nanmean([x["jump"] for x in base_vals])) if base_vals else None
    base_a = float(np.nanmean([x["avg"] for x in base_vals])) if base_vals else None

    table = Table(title="Per-model averages (Δ vs qwen3-0.6b avg)", show_header=True, header_style="bold")
    table.add_column("Model", justify="left")
    table.add_column("Avg Straight", justify="right")
    table.add_column("Δ Straight", justify="right")
    table.add_column("Avg Pan", justify="right")
    table.add_column("Δ Pan", justify="right")
    table.add_column("Avg Jump", justify="right")
    table.add_column("Δ Jump", justify="right")
    table.add_column("Avg (overall)", justify="right")
    table.add_column("Δ Avg", justify="right")
    table.add_column("N sizes", justify="right")
    for row in table_rows:
        table.add_row(
            row[0],
            styled_num(float(row[1]), bests["straight"]),
            delta_text(float(row[1]), base_s),
            styled_num(float(row[2]), bests["pan"]),
            delta_text(float(row[2]), base_p),
            styled_num(float(row[3]), bests["jump"]),
            delta_text(float(row[3]), base_j),
            styled_num(float(row[4]), bests["avg"]),
            delta_text(float(row[4]), base_a),
            row[5],
        )
    console.print(table)


def print_volatility(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    table = Table(title="Per-model volatility (std across voxel sizes, Δ vs qwen3-0.6b)", show_header=True, header_style="bold")
    table.add_column("Model", justify="left")
    table.add_column("Straight std", justify="right")
    table.add_column("Δ Straight std", justify="right")
    table.add_column("Pan std", justify="right")
    table.add_column("Δ Pan std", justify="right")
    table.add_column("Jump std", justify="right")
    table.add_column("Δ Jump std", justify="right")
    table.add_column("Avg std", justify="right")
    table.add_column("Δ Avg std", justify="right")
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for r in rows:
        grouped.setdefault(r["model"], []).append(r)
    base_vals = grouped.get("qwen3-0.6b", [])
    base_s = float(np.std([x["straight"] for x in base_vals])) if base_vals else None
    base_p = float(np.std([x["pan"] for x in base_vals])) if base_vals else None
    base_j = float(np.std([x["jump"] for x in base_vals])) if base_vals else None
    base_a = float(np.std([x["avg"] for x in base_vals])) if base_vals else None
    for model in sorted(grouped.keys(), key=model_sort_key):
        vals = grouped[model]
        s_std = float(np.std([x["straight"] for x in vals])) if vals else float("nan")
        p_std = float(np.std([x["pan"] for x in vals])) if vals else float("nan")
        j_std = float(np.std([x["jump"] for x in vals])) if vals else float("nan")
        a_std = float(np.std([x["avg"] for x in vals])) if vals else float("nan")
        table.add_row(
            model,
            f"{s_std:.2f}",
            delta_text(s_std, base_s),
            f"{p_std:.2f}",
            delta_text(p_std, base_p),
            f"{j_std:.2f}",
            delta_text(j_std, base_j),
            f"{a_std:.2f}",
            delta_text(a_std, base_a),
        )
    console.print(table)


def print_leaderboards(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    sizes = sorted({r["size"] for r in rows})
    table = Table(title="Best model per voxel size (avg accuracy)", show_header=True, header_style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Top Avg", justify="left")
    for s in sizes:
        subset = [r for r in rows if r["size"] == s]
        if not subset:
            continue
        best_avg = max(subset, key=lambda x: x["avg"])
        table.add_row(str(s), f"{best_avg['model']} ({best_avg['avg']:.2f})")
    console.print(table)

    table2 = Table(title="Best voxel size per model (avg accuracy)", show_header=True, header_style="bold")
    table2.add_column("Model", justify="left")
    table2.add_column("Top Avg @size", justify="left")
    models = sorted({r["model"] for r in rows}, key=model_sort_key)
    for m in models:
        subset = [r for r in rows if r["model"] == m]
        if not subset:
            continue
        best_avg = max(subset, key=lambda x: x["avg"])
        table2.add_row(m, f"{best_avg['avg']:.2f} @ {best_avg['size']}")
    console.print(table2)


def print_joint_leader(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    table = Table(title="Joint leader per voxel size (mean of straight/pan/jump)", show_header=True, header_style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Top Avg", justify="left")
    sizes = sorted({r["size"] for r in rows})
    for s in sizes:
        subset = [r for r in rows if r["size"] == s]
        if not subset:
            continue
        best = max(subset, key=lambda x: x["avg"])
        table.add_row(str(s), f"{best['model']} ({best['avg']:.2f})")
    console.print(table)


def print_regressions(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    metrics = ["straight", "pan", "jump", "avg"]
    labels = {"straight": "Straight", "pan": "Pan", "jump": "Jump", "avg": "Avg"}
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


def print_constant_prediction_flags(pred_stats: Dict[Tuple[str, int], Tuple[int, int, float]]) -> None:
    if not pred_stats:
        return
    flags = []
    model_totals: Dict[str, Dict[str, int]] = {}
    for (model, size), (uniq, total, dom_frac) in pred_stats.items():
        model_totals.setdefault(model, {"total": 0, "flagged": 0})
        model_totals[model]["total"] += 1
        if total > 0 and uniq <= 2 and dom_frac >= 0.9:
            flags.append((model, size, uniq, dom_frac, total))
            model_totals[model]["flagged"] += 1

    print("\nConstant-prediction check (>=90% same prediction, <=2 unique predictions):")
    if not flags:
        print("  None detected across all action-recognition evals.")
    else:
        print("These cases likely yield identical accuracies across sizes because the model outputs nearly the same prediction every time.")
        print("\t".join(["Model", "Voxel size", "Unique preds", "Top frac", "Total rows"]))
        for model, size, uniq, dom_frac, total in sorted(flags, key=lambda x: (model_sort_key(x[0]), x[1])):
            print("\t".join([model, str(size), str(uniq), f"{dom_frac:.2f}", str(total)]))

    # Per-model summary coverage
    print("\nPer-model constant-prediction summary (flagged/total sizes):")
    for model in sorted(model_totals.keys(), key=model_sort_key):
        flagged = model_totals[model]["flagged"]
        total = model_totals[model]["total"]
        print(f"  {model}: {flagged}/{total} sizes flagged")


def main() -> None:
    print("=== Action Recognition Report ===")
    rows, pred_stats = collect_metrics()
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
    print_joint_leader(rows)
    print()
    print_regressions(rows)
    print_constant_prediction_flags(pred_stats)


if __name__ == "__main__":
    main()
