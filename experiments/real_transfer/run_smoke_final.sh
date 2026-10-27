#!/usr/bin/env bash
# Smoke test of the registered configuration (validation only, --skip-test, seed 20261309, a short budget).
# It checks the final-round source checkpoint, --source-only caching, and a cache hit from a target run.
set -euo pipefail
STUDY="$(cd "$(dirname "$0")" && pwd)"
DATA="$STUDY/../enterprise_split/data/v1/C-time"
OUT="$STUDY/results/smoke_final"
rm -rf "$OUT"  # smoke outputs only
mkdir -p "$OUT" "$STUDY/logs"
cd "$STUDY/src"
py() { PYTHONDONTWRITEBYTECODE=1 python run_method.py --seeds 20261309 --rounds 2 --source-rounds 3 \
         --source-cache "$OUT/source_cache" --skip-test "$@"; }
{
  py --arm sparse_source --target "$DATA/target_f0.10" --output "$OUT/sparse_source_f0.10.json"
  py --arm dense_source --target "$DATA/target_f1.00" --source-only
  py --arm dense_source --target "$DATA/target_f0.10" --output "$OUT/dense_source_f0.10.json"
} > "$STUDY/logs/smoke_final.log" 2>&1
echo "final-configuration smoke finished"
