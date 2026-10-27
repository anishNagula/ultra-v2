#!/usr/bin/env bash
# Smoke test (validation only, --skip-test). Seed 20261211 is a registered C-time seed: the local arm
# must reproduce its registered validation curve exactly. Seed 20261309 is used only here.
# It ran with the source checkpoint rule first planned (rule a, DESIGN.md), pinned below.
set -euo pipefail
STUDY="$(cd "$(dirname "$0")" && pwd)"
DATA="$STUDY/../enterprise_split/data/v1/C-time"
OUT="$STUDY/results/smoke"
mkdir -p "$OUT" "$STUDY/logs"
cd "$STUDY/src"
rm -rf "$OUT/source_cache"  # the smoke cache only
py() { PYTHONDONTWRITEBYTECODE=1 python run_method.py "$@" --skip-test; }
py --arm local --target "$DATA/target_f0.10" --seeds 20261211 --output "$OUT/local_f0.10.json" > "$STUDY/logs/smoke_local.log" 2>&1 &
py --arm sparse_source --target "$DATA/target_f0.10" --seeds 20261309 --rounds 2 --source-rounds 2 --source-checkpoint source_valid \
   --source-cache "$OUT/source_cache" --output "$OUT/sparse_source_f0.10.json" > "$STUDY/logs/smoke_sparse.log" 2>&1 &
py --arm dense_source --target "$DATA/target_f0.25" --seeds 20261309 --rounds 2 --source-rounds 2 --source-checkpoint source_valid \
   --source-cache "$OUT/source_cache" --output "$OUT/dense_source_f0.25.json" > "$STUDY/logs/smoke_dense.log" 2>&1
py --arm dense_source --target "$DATA/target_f1.00" --seeds 20261309 --rounds 2 --source-rounds 2 --source-checkpoint source_valid \
   --source-cache "$OUT/source_cache" --output "$OUT/dense_source_f1.00.json" >> "$STUDY/logs/smoke_dense.log" 2>&1
wait
echo "smoke runs finished"
