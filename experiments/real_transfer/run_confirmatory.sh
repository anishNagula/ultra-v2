#!/usr/bin/env bash
# The preregistered confirmatory grid (PREREGISTRATION.md): seeds 20261501-20261510, every arm at every
# registered fraction. Test metrics go to the run files only; logs print "test=scored".
# Stage 1 trains and caches the dense source stage (seed halves in parallel) alongside the sparse arms
# and two local arms; stage 2 runs the arms that reuse the cached dense source weights.
set -euo pipefail
STUDY="$(cd "$(dirname "$0")" && pwd)"
DATA="$STUDY/../enterprise_split/data/v1/C-time"
OUT="$STUDY/results/confirmatory"
CACHE="$OUT/source_cache"
SEEDS=20261501,20261502,20261503,20261504,20261505,20261506,20261507,20261508,20261509,20261510
mkdir -p "$OUT" "$STUDY/logs"
cd "$STUDY/src"
arm() {  # arm f [extra args]
  local a="$1" f="$2"; shift 2
  PYTHONDONTWRITEBYTECODE=1 python run_method.py --arm "$a" --target "$DATA/target_f$f" --seeds "$SEEDS" \
    --source-cache "$CACHE" --output "$OUT/${a}_f$f.json" "$@" \
    >> "$STUDY/logs/confirmatory_${a}_f$f.log" 2>&1 && echo "done $a f$f" || echo "FAILED $a f$f"
}
warm() {  # name seeds: train and cache the dense source stage only
  PYTHONDONTWRITEBYTECODE=1 python run_method.py --arm dense_source --target "$DATA/target_f1.00" --seeds "$2" \
    --source-cache "$CACHE" --source-only >> "$STUDY/logs/confirmatory_dense_source_cache_$1.log" 2>&1 \
    && echo "done dense source cache $1" || echo "FAILED dense source cache $1"
}
echo "stage 1 started $(date -u +%FT%TZ)"
( warm a 20261501,20261502,20261503,20261504,20261505; arm local 0.10 ) &
( warm b 20261506,20261507,20261508,20261509,20261510; arm local 0.25 ) &
( arm sparse_source 0.10 --source-fraction 0.10 ) &
( arm sparse_source 0.25 --source-fraction 0.25 ) &
wait
echo "stage 2 started $(date -u +%FT%TZ)"
arm dense_source 0.10 &
arm dense_source 0.25 &
arm dense_source 1.00 &
arm local 1.00 &
wait
echo "confirmatory grid finished $(date -u +%FT%TZ)"
