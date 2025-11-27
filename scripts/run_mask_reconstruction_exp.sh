#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_mask_reconstruction_exp.sh
# Runs mask reconstruction inference/eval for fixed cube size 7 and mask sizes 2,3,4.

CUBE_SIZE=7
MASK_SIZES=(2 4 6)
SUBSET=0.1

for MASK in "${MASK_SIZES[@]}"; do
  echo "Running mask reconstruction inference (cube=${CUBE_SIZE}, mask=${MASK}, subset=${SUBSET})..."
  uv run inference_mask_reconstruction.py \
    --size "${CUBE_SIZE}" \
    --mask-size "${MASK}" \
    --subset-fraction "${SUBSET}" \
    --batch-size 1 \
    --max-new-tokens 1024 \
    --output-raw "results/mask_reconstruction_raw_size${CUBE_SIZE}_mask${MASK}.json"

  echo "Evaluating mask reconstruction (cube=${CUBE_SIZE}, mask=${MASK})..."
  uv run eval_mask_reconstruction_result.py \
    --input "results/mask_reconstruction_raw_size${CUBE_SIZE}_mask${MASK}.json" \
    --size "${CUBE_SIZE}" \
    --mask-size "${MASK}" \
    --model-name all
done

echo "Plotting mask reconstruction distributions..."
uv run plot_mask_reconstruction_exp.py

echo "Done. Outputs in results/, evals/, plots/."
