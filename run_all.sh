#!/usr/bin/env bash
# Train every experiment on the train split, then evaluate all of them on the test split.
# Full run takes ~2 h on a 4-core laptop CPU (no GPU needed).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
mkdir -p logs

for run in "mlp none" "mlp affine" "mlp full" \
           "cnn none" "cnn affine" "cnn full" \
           "vit none" "vit full"; do
  set -- $run
  $PY src/train.py --model "$1" --aug "$2" 2>&1 | tee "logs/train_$1_$2.log"
done

$PY src/test.py 2>&1 | tee logs/test.log
