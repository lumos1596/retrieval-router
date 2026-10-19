#!/usr/bin/env bash
# Full replay pipeline for one dataset / one side:
#   dense encode (full corpus) -> dense replay top100 -> multi-vector encode
#   (union of candidates) -> late-interaction rerank replay
# Usage: pipeline.sh <PHYS_GPU> <dataset> <mm|text>
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="$1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DS="$2"; SIDE="$3"
PY=/home/luwa/Documents/miniforge3/envs/rr-verify/bin/python

if [ "$SIDE" = "mm" ]; then DENSE=MD; RR=MR; BS=1; else DENSE=TD; RR=TR; BS=16; fi

echo "=== [$DS $SIDE] $DENSE dense encode ==="
$PY encode_dense.py --arm "$DENSE" --dataset "$DS" --gpu 0 --batch-size "$BS"
echo "=== [$DS $SIDE] $DENSE dense replay ==="
$PY replay_dense.py --arm "$DENSE" --datasets "$DS" --gpu 0
echo "=== [$DS $SIDE] $RR multi-vector encode ==="
$PY encode_multi.py --arm "$RR" --dataset "$DS" --gpu 0
echo "=== [$DS $SIDE] $RR rerank replay ==="
$PY replay_rerank.py --arm "$RR" --datasets "$DS" --gpu 0
echo "=== [$DS $SIDE] DONE ==="
