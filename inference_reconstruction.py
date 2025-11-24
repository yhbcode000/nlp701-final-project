from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from dataset_loader import load_full_dataset
from dataset_utils import compute_dataset_splits
from hyperparameter_config import HyperparameterConfig
from model_registry import (
    MODEL_PATHS,
    get_model_wrapper,
    release_all_models,
    release_model,
)
from utils_module import Utils


def run_inference(
    *,
    data_dir: Path,
    models: List[str],
    voxel_size: int,
    subset_fraction: float,
    batch_size: int,
    max_new_tokens: int,
    ckpt_path: Path | None,
    result_path: Path,
    raw_path: Path,
) -> None:
    """Evaluate frame reconstruction for one or more models and save results/raw outputs."""
    config = HyperparameterConfig()
    config_dict = config.get_config()
    dataset, total_pairs, unique_actions = load_full_dataset(
        config, data_dir, voxel_size=voxel_size
    )
    print(f"Dataset loaded: {total_pairs} pairs, {len(unique_actions)} unique actions")

    splits, _ = compute_dataset_splits(dataset, subset_fraction=subset_fraction)
    test_indices = splits["test"]
    print(f"Using test split of size {len(test_indices)} (subset_fraction={subset_fraction})")

    results: Dict[str, dict] = {}
    raw_outputs: Dict[str, list] = {}

    for model_key in models:
        print(f"\n=== Evaluating model: {model_key} ===")
        wrapper = get_model_wrapper(model_key)

        if ckpt_path:
            print(f"Loading checkpoint from {ckpt_path}")
            wrapper.load_checkpoint(str(ckpt_path))

        metrics = wrapper.evaluate_task(
            dataset,
            test_indices,
            task_type="frame_reconstruction",
            model_key=model_key,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )

        predictions = metrics.pop("predictions", [])
        targets = metrics.pop("targets", [])

        raw_records = []
        if len(predictions) != len(test_indices):
            print(
                f"Warning: prediction count {len(predictions)} does not match test indices {len(test_indices)} for {model_key}."
            )

        for idx, pred, target in zip(test_indices, predictions, targets):
            pair = dataset.data_pairs[int(idx)]
            history = pair.get("history_reconstruction") or pair.get("history_action") or pair.get("x")
            raw_records.append(
                {
                    "index": int(idx),
                    "episode": pair.get("episode"),
                    "history": history,
                    "z_label": target,
                    "z_prediction": pred,
                }
            )

        raw_outputs[model_key] = raw_records
        results[model_key] = metrics

        release_model(model_key)

    Utils.save_json(results, result_path)
    Utils.save_json(raw_outputs, raw_path)
    print(f"\nSaved reconstruction summary to {result_path}")
    print(f"Saved reconstruction raw outputs to {raw_path}")
    release_all_models()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frame reconstruction inference and save metrics/raw outputs."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/minecraft/data"), help="Dataset root.")
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated model keys to evaluate (defaults to all registered models).",
    )
    parser.add_argument("--size", type=int, default=5, help="Voxel cube side length for voxel2word.")
    parser.add_argument("--subset-fraction", type=float, default=0.05, help="Fraction of data to evaluate.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens to generate per sample.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        help="Optional checkpoint to load before inference (defaults to base weights).",
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default=None,
        help="Where to save metrics (defaults to results/frame_reconstruction_result_size{size}.json).",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=None,
        help="Where to save raw outputs (defaults to results/frame_reconstruction_raw_size{size}.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else list(MODEL_PATHS.keys())
    )

    default_dir = Path("results")
    default_dir.mkdir(parents=True, exist_ok=True)

    result_path = args.result_path or default_dir / f"frame_reconstruction_result_size{args.size}.json"
    raw_path = args.raw_path or default_dir / f"frame_reconstruction_raw_size{args.size}.json"

    run_inference(
        data_dir=args.data_dir,
        models=selected_models,
        voxel_size=args.size,
        subset_fraction=args.subset_fraction,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        ckpt_path=args.ckpt,
        result_path=result_path,
        raw_path=raw_path,
    )


if __name__ == "__main__":
    main()
