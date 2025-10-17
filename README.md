# Minecraft Voxel LLM Experiments

This project investigates two sequential reasoning tasks on Minecraft gameplay data using Qwen 3 language models. We evaluate both a lightweight 0.6B parameter model and a larger 4B parameter model under zero-shot (in-context) and LoRA fine-tuned regimes.

## Dataset & Splits

- Source: `datasets/minecraft/data/` organised as MineDojo-style `creative:{episode}` directories with `.npy` frame dumps.
- Loader: `minecraft_dataset.MinecraftDataset` constructs paired samples containing the current frame (`x`), the next frame (`z`), the action (`y`), and history strings.
- Splits:
  - Experimental subset: `compute_dataset_splits(..., subset_fraction=0.05)` builds train/val/test indices for training and quick experimentation.
  - Full test set: `INFERENCE_TEST_INDICES` (computed with `subset_fraction=1.0`) is used for all official evaluations to ensure consistency.

## Task Definitions

### D1 — Frame Reconstruction

- **Input**: History `h` containing alternating frame/action text (`history_reconstruction`) ending at time *t*.
- **Output**: Predicted next frame `ẑ`, matching the pipe-delimited grid format of the ground-truth frame `z`.
- **Targets stored in dataset**: `pair["z"]` and `pair["history_reconstruction"]`.

### D2 — Action Recognition

- **Input**: History `h` capturing frame–frame–action triplets (`history_action`) up to time *t+1*.
- **Output**: Predicted action block `ŷ` describing straight/pan/jump decisions in a three-line format.
- **Targets stored in dataset**: `pair["y"]` and `pair["history_action"]`.

## Models

- **Qwen3-0.6B** (`MODEL_PATHS["qwen3-0.6b"]`)
- **Qwen3-4B** (`MODEL_PATHS["qwen3-4b"]`)
- Models are wrapped via `model_wrapper.ModelWrapper`, which supports zero-shot inference and optional LoRA adapters for fine-tuning.

## Evaluation Workflow

### Zero-Shot (In-Context) Evaluation

1. Load models and dataset (Section 1 of `main.ipynb`).
2. Execute **Cell 2.1** for D1 and **Cell 2.3** for D2:
   - Iterates through each model, calling `ModelWrapper.evaluate_task(...)` on `INFERENCE_TEST_INDICES`.
   - Saves summaries:
     - D1: `2.1-result.json` with aggregate metrics (`strict_match_accuracy`, `reconstruction_accuracy`, latency stats).
     - D2: `2.3-result.json` with metrics (`strict_match_accuracy`, `word2vec_cosine`, `precision`, `recall`, `f1`, etc.).
   - Saves raw outputs:
     - D1: `2.1-raw.json` storing, per index, `history`, ground-truth `z_label`, and model `z_prediction`.
     - D2: `2.3-raw.json` storing `history`, `y_label`, and `y_prediction`.
   - When both files already exist, the cells skip inference and load the cached JSON artefacts directly.
3. Visualisation cells (2.2 & 2.4) consume these summaries to produce bar charts, heatmaps, and confusion matrices.

### LoRA Fine-Tuning

1. Sections 3 & 4 in the notebook handle LoRA training for D1 and D2 respectively (configurable via `hyperparameter_config.HyperparameterConfig`).
2. Fine-tuned checkpoints are evaluated on the same `INFERENCE_TEST_INDICES`, producing:
   - D1 results: `3.2-result.json` and detailed outputs in `3.2-raw.json`.
   - D2 results: `4.2-result.json` and detailed outputs in `4.2-raw.json`.
   Cached result/raw pairs are reused automatically on subsequent runs.
3. Comparative plots in Sections 3.3 and 4.3 contrast zero-shot vs. fine-tuned metrics for each model.

### Raw Output Structure

Each raw JSON file (`2.1-raw.json`, `2.3-raw.json`, `3.2-raw.json`, `4.2-raw.json`) is keyed by model identifier:

```json
{
  "qwen3-0.6b": [
    {
      "index": 12,
      "episode": "creative:42",
      "history": "...",
      "z_label": "...",
      "z_prediction": "..."
    }
  ],
  "qwen3-4b": [...]
}
```

Frame reconstruction entries use `z_label` / `z_prediction`, while action recognition entries use `y_label` / `y_prediction`. This layout simplifies downstream error analysis, enabling side-by-side inspection of history strings, labels, and predictions.

## Reproducing Experiments

1. Install dependencies (see `.python-version` and project environment setup).
2. Open `main.ipynb` and execute cells sequentially:
   - Section 1: setup and dataset preview.
   - Section 2: zero-shot evaluation & plotting.
   - Sections 3 & 4: optional LoRA fine-tuning and evaluation.
3. Generated artefacts (created if missing, otherwise reused):
   - Metrics: `2.1-result.json`, `2.3-result.json`, `3.2-result.json`, `4.2-result.json`.
   - Raw outputs: `2.1-raw.json`, `2.3-raw.json`, `3.2-raw.json`, `4.2-raw.json`.
   - Plots: stored under `plots/`.

## Directory Guide

- `hyperparameter_config.py` – consolidated hyperparameter defaults and grid search utilities.
- `model_wrapper.py` – handles generation, metric computation, and checkpoint loading.
- `dataset_utils.py` – split computations, dataloader builders, embeddings for action similarity.
- `plot_utils.py` – heatmaps, metric bars, confusion matrices for reporting.
- `notebook/` – auxiliary notebooks or exported figures (if present).

## Notes

- Ensure the dataset tree follows the documented layout in `datasets/minecraft/README.md`.
- For reproducible evaluations, the notebook saves results deterministically; rerunning 2.1/2.3 overwrites the corresponding JSON files.
