from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

RESET = "\033[0m"
GREEN = "\033[92m"


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
    header = ["Model", "Mask size", "BLEU", "Similarity"]
    print("\t".join(header))
    for r in sorted(rows, key=lambda x: (x["model"], x["mask_size"])):
        print(
            "\t".join(
                [
                    r["model"],
                    str(r["mask_size"]),
                    fmt(r["bleu"], best_bleu),
                    fmt(r["sim"], best_sim),
                ]
            )
        )


def print_coverage(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    expected_masks = [1, 2, 3, 4, 5, 6]
    models = sorted({r["model"] for r in rows})
    print("Coverage by model (present/expected mask sizes):")
    for model in models:
        have = sorted({r["mask_size"] for r in rows if r["model"] == model})
        missing = [m for m in expected_masks if m not in have]
        print(f"  {model}: {len(have)}/{len(expected_masks)} masks present; missing {missing if missing else 'none'}")


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
    print("Per-model averages:")
    print("\t".join(header))
    for model in sorted(grouped.keys()):
        vals = grouped[model]
        avg_bleu = float(np.mean([x["bleu"] for x in vals]))
        avg_sim = float(np.mean([x["sim"] for x in vals]))
        print(
            "\t".join(
                [
                    model,
                    fmt(avg_bleu, best_bleu),
                    fmt(avg_sim, best_sim),
                    str(len(vals)),
                ]
            )
        )


def print_volatility(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    print("Per-model volatility (std across masks):")
    print("\t".join(["Model", "BLEU std", "Similarity std"]))
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for r in rows:
        grouped.setdefault(r["model"], []).append(r)
    for model in sorted(grouped.keys()):
        vals = grouped[model]
        bleu_std = float(np.std([x["bleu"] for x in vals])) if vals else float("nan")
        sim_std = float(np.std([x["sim"] for x in vals])) if vals else float("nan")
        print("\t".join([model, f"{bleu_std:.2f}", f"{sim_std:.2f}"]))


def print_leaderboards(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    masks = sorted({r["mask_size"] for r in rows})
    print("Best model per mask size:")
    print("\t".join(["Mask", "Top BLEU", "Top Similarity"]))
    for m in masks:
        subset = [r for r in rows if r["mask_size"] == m]
        if not subset:
            continue
        best_bleu = max(subset, key=lambda x: x["bleu"])
        best_sim = max(subset, key=lambda x: x["sim"])
        print(
            "\t".join(
                [
                    str(m),
                    f"{best_bleu['model']} ({best_bleu['bleu']:.2f})",
                    f"{best_sim['model']} ({best_sim['sim']:.2f})",
                ]
            )
        )

    print()
    print("Best mask size per model:")
    print("\t".join(["Model", "Top BLEU @mask", "Top Sim @mask"]))
    models = sorted({r["model"] for r in rows})
    for model in models:
        subset = [r for r in rows if r["model"] == model]
        if not subset:
            continue
        best_bleu = max(subset, key=lambda x: x["bleu"])
        best_sim = max(subset, key=lambda x: x["sim"])
        print(
            "\t".join(
                [
                    model,
                    f"{best_bleu['bleu']:.2f} @ {best_bleu['mask_size']}",
                    f"{best_sim['sim']:.2f} @ {best_sim['mask_size']}",
                ]
            )
        )


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
