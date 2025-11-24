#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_action_recognition_exp.sh
# Iterates sizes 3,5,7; subset-fraction is fixed at 0.1 as requested.

SIZES=(3 5 7)

for SIZE in "${SIZES[@]}"; do
  echo "Running action recognition inference (size=${SIZE}, subset-fraction=0.1)..."
  uv run inference_action_recognition.py \
    --size "${SIZE}" \
    --subset-fraction 0.1

  echo "Running action recognition eval (size=${SIZE})..."
  uv run eval_action_recognition_result.py \
    --size "${SIZE}"
done

echo "Plotting action recognition metrics vs size..."
uv run plot_action_recognition_exp.py

echo "Done. Outputs in results/ (JSON) and evals/ (CSV), plots in plots/."
