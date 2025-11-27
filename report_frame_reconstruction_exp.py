from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

RESET = "\033[0m"
GREEN = "\033[92m"


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


def print_table(rows: List[Dict[str, float]]) -> None:
    if not rows:
        print("No frame reconstruction evals found.")
        return

    best_bleu = max(r["bleu"] for r in rows)
    best_sim = max(r["sim"] for r in rows)

    header = ["Model", "Voxel size", "BLEU", "Similarity"]
    print("\t".join(header))
    for r in sorted(rows, key=lambda x: (x["model"], x["size"])):
        print(
            "\t".join(
                [
                    r["model"],
                    str(r["size"]),
                    fmt(r["bleu"], best_bleu),
                    fmt(r["sim"], best_sim),
                ]
            )
        )


def print_coverage(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    expected_sizes = [3, 5, 7, 9, 11, 13]
    models = sorted({r["model"] for r in rows})
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
    print("Per-model volatility (std across sizes):")
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
    sizes = sorted({r["size"] for r in rows})
    print("Best model per voxel size:")
    print("\t".join(["Size", "Top BLEU", "Top Similarity"]))
    for s in sizes:
        subset = [r for r in rows if r["size"] == s]
        if not subset:
            continue
        best_bleu = max(subset, key=lambda x: x["bleu"])
        best_sim = max(subset, key=lambda x: x["sim"])
        print(
            "\t".join(
                [
                    str(s),
                    f"{best_bleu['model']} ({best_bleu['bleu']:.2f})",
                    f"{best_sim['model']} ({best_sim['sim']:.2f})",
                ]
            )
            )

    print()
    print("Best voxel size per model:")
    print("\t".join(["Model", "Top BLEU @size", "Top Sim @size"]))
    models = sorted({r["model"] for r in rows})
    for m in models:
        subset = [r for r in rows if r["model"] == m]
        if not subset:
            continue
        best_bleu = max(subset, key=lambda x: x["bleu"])
        best_sim = max(subset, key=lambda x: x["sim"])
        print(
            "\t".join(
                [
                    m,
                    f"{best_bleu['bleu']:.2f} @ {best_bleu['size']}",
                    f"{best_sim['sim']:.2f} @ {best_sim['size']}",
                ]
            )
        )


def print_joint_leader(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    print("\nJoint leader per voxel size (mean of BLEU and Similarity):")
    print("\t".join(["Size", "Top (BLEU+Sim)/2"]))
    sizes = sorted({r["size"] for r in rows})
    for s in sizes:
        subset = [r for r in rows if r["size"] == s]
        if not subset:
            continue
        best = max(subset, key=lambda x: (x["bleu"] + x["sim"]) / 2.0)
        score = (best["bleu"] + best["sim"]) / 2.0
        print("\t".join([str(s), f"{best['model']} ({score:.2f})"]))


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
