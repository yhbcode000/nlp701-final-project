from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from dataset_loader import load_full_dataset
from hyperparameter_config import HyperparameterConfig
from mask_apply import apply_random_mask, center_crop, voxel_to_word
from model_registry import MODEL_PATHS, get_model_wrapper, release_all_models, release_model
from utils_module import Utils


def enforce_grid_format(text: str, size: int) -> str:
    """Keep only pipe-delimited lines and truncate/pad to size*size rows."""
    lines = [ln.strip() for ln in text.splitlines() if "|" in ln]
    lines = lines[: size * size]
    filler = "|" + "|".join(["air"] * size) + "|"
    while len(lines) < size * size:
        lines.append(filler)
    return "\n".join(lines)


def mask_and_prompt(
    voxel: np.ndarray, cube_size: int, mask_size: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, str, Tuple[int, int, int]]:
    cropped = center_crop(voxel, cube_size)
    masked, mask, coords = apply_random_mask(
        cropped, cube_size=cube_size, mask_size=mask_size, rng=rng
    )
    masked_text = voxel_to_word(masked)
    return masked, mask, masked_text, coords


def build_few_shot_prompt(
    masked_text: str,
    cube_size: int,
    mask_size: int,
    history_text: str,
    examples: List[Tuple[str, str, str]],
) -> List[Dict[str, str]]:
    lines_needed = cube_size * cube_size
    instruction = (
        "Task: predict the masked blocks in the final frame based on the history.\n"
        f"- The frame is a {cube_size}x{cube_size}x{cube_size} grid, flattened into {lines_needed} lines.\n"
        f"- The masked region is {mask_size}x{mask_size}x{mask_size} and is marked with the token 'mask'.\n"
        "- Replace every 'mask' token with the most likely Minecraft block name.\n"
        "- Output only the completed grid with exactly the required number of lines, each formatted like '|block|block|...|'."
    )

    prompt = "\n\n".join(
        [
            instruction,
            "History:\n" + history_text,
            "Masked grid:\n" + masked_text,
            "Completed grid:",
        ]
    )
    return prompt


def generate_with_model(wrapper, prompt: str, max_new_tokens: int = 2048) -> str:
    tokenized = wrapper.tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=wrapper.tokenizer.model_max_length
    )
    input_ids = tokenized["input_ids"].to(wrapper.device)
    attention_mask = tokenized["attention_mask"].to(wrapper.device)
    outputs = wrapper.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=wrapper.tokenizer.pad_token_id,
    )
    generated = outputs[0][input_ids.shape[1] :]
    return wrapper.tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mask a voxel cube and ask a local model to reconstruct the masked area.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/minecraft/data"))
    parser.add_argument("--size", type=int, default=7, help="Cube side length to crop (odd).")
    parser.add_argument("--mask-size", type=int, default=3, help="Mask side length.")
    parser.add_argument("--subset-fraction", type=float, default=0.1, help="Fraction of dataset to run.")
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated model keys to evaluate (defaults to all registered models).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max tokens to generate (generation capped to 1024).")
    parser.add_argument("--batch-size", type=int, default=1, help="Generation batch size (set to 1 to avoid OOM).")
    parser.add_argument(
        "--output-raw",
        type=Path,
        help="Where to save raw outputs (defaults to results/mask_reconstruction_raw_size{size}_mask{mask}.json).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for mask placement.")
    parser.add_argument(
        "--variants",
        type=int,
        default=2,
        help="Number of random mask variants to generate per sample (default: 2).",
    )
    args = parser.parse_args()

    selected_models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else list(MODEL_PATHS.keys())
    )
    config = HyperparameterConfig()
    dataset, total_pairs, _ = load_full_dataset(config, args.data_dir, voxel_size=args.size)
    total = len(dataset)
    take = total if args.subset_fraction >= 1 else max(1, int(total * args.subset_fraction))
    indices = list(range(take))
    print(
        f"Using {len(indices)} samples (subset_fraction={args.subset_fraction}), "
        f"{args.variants} variants each, for mask reconstruction."
    )

    # Precompute masked variants so every model sees the same masked inputs.
    variant_records = []
    for idx in indices:
        pair = dataset.data_pairs[idx]
        target_voxel = pair["z_raw"]
        history_text = pair.get("history_reconstruction") or ""

        for variant in range(args.variants):
            seed = args.seed + idx * args.variants + variant
            rng = np.random.default_rng(seed)
            masked_cube, mask_cube, masked_text, (sx, sy, sz) = mask_and_prompt(
                target_voxel, args.size, args.mask_size, rng
            )
            cropped = center_crop(target_voxel, args.size)
            label_mask_cube = cropped[
                sx : sx + args.mask_size, sy : sy + args.mask_size, sz : sz + args.mask_size
            ]
            label_text = voxel_to_word(label_mask_cube)
            variant_records.append(
                {
                    "index": int(idx),
                    "episode": pair.get("episode"),
                    "history": history_text,
                    "mask_size": int(args.mask_size),
                    "cube_size": int(args.size),
                    "variant": int(variant),
                    "seed": int(seed),
                    "coords": [int(sx), int(sy), int(sz)],
                    "masked_input": masked_text,
                    "mask_cube": voxel_to_word(mask_cube),
                    "x_cropped": voxel_to_word(cropped),
                    "z_label": label_text,
                }
            )

    default_output = Path(f"results/mask_reconstruction_raw_size{args.size}_mask{args.mask_size}.json")
    output_path = args.output_raw or default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_outputs: Dict[str, List[Dict]] = {}

    for model_key in selected_models:
        print(f"\n=== Evaluating model: {model_key} ===")
        wrapper = get_model_wrapper(model_key)
        records: List[Dict] = []

        batch_size = max(1, args.batch_size)
        for start in tqdm(range(0, len(variant_records), batch_size), desc=f"Mask reconstructing ({model_key})"):
            batch = variant_records[start : start + batch_size]
            prompts = [
                build_few_shot_prompt(
                    rec["masked_input"],
                    rec["cube_size"],
                    rec["mask_size"],
                    rec["history"],
                    [],
                )
                for rec in batch
            ]

            for prompt, rec in zip(prompts, batch):
                # Generate one by one to minimize VRAM (batch_size default=1)
                prediction_text = generate_with_model(
                    wrapper,
                    prompt,
                    max_new_tokens=min(args.max_new_tokens, 1024),
                )
                prediction_text_clean = enforce_grid_format(prediction_text, rec["mask_size"])

                out_row = {
                    **rec,
                    "model": model_key,
                    "z_prediction": prediction_text_clean,
                    "z_prediction_raw": prediction_text,
                }
                records.append(out_row)

        raw_outputs[model_key] = records
        release_model(model_key)

    Utils.save_json(raw_outputs, output_path)
    print(f"\nSaved mask reconstruction raw outputs to {output_path}")
    release_all_models()


if __name__ == "__main__":
    main()
