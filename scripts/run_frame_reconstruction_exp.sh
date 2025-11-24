#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_frame_reconstruction_exp.sh
# Iterates sizes 3,5,7,9,11; subset-fraction is fixed at 1 as requested.

SIZES=(3 5 7 9 11)

for SIZE in "${SIZES[@]}"; do
  echo "Running frame reconstruction inference (size=${SIZE}, subset-fraction=1)..."
  uv run inference_reconstruction.py \
    --size "${SIZE}" \
    --subset-fraction 1

  echo "Running frame reconstruction eval (size=${SIZE})..."
  uv run eval_reconstruction_result.py \
    --size "${SIZE}"
done

echo "Plotting frame reconstruction metrics vs size..."
uv run plot_frame_reconstruction_exp.py

echo "Done. Outputs in results/ (JSON) and evals/ (CSV), plots in plots/."
