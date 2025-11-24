#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_frame_reconstruction_exp.sh
# Iterates sizes 3,5,7; subset-fraction is fixed at 0.1 as requested.

SIZES=(3 5 7)

for SIZE in "${SIZES[@]}"; do
  echo "Running frame reconstruction inference (size=${SIZE}, subset-fraction=0.1)..."
  uv run inference_reconstruction.py \
    --size "${SIZE}" \
    --subset-fraction 0.1 \
    --batch-size 1

  echo "Running frame reconstruction eval (size=${SIZE})..."
  uv run eval_reconstruction_result.py \
    --size "${SIZE}"
done

echo "Plotting frame reconstruction metrics vs size..."
uv run plot_frame_reconstruction_exp.py

echo "Done. Outputs in results/ (JSON) and evals/ (CSV), plots in plots/."
