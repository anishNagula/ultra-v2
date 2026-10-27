#!/usr/bin/env bash
# Pilot (validation only, --skip-test; seeds 20261401-20261403, never used for confirmation).
# The source fraction of sparse_source matches the target fraction. dense_source's source stage is
# computed once per seed (lane A, f = 0.10) and reused from the cache at f = 0.25 and f = 1.00.
# It ran with the source checkpoint rule first planned (rule a, DESIGN.md), pinned below.
set -euo pipefail
STUDY="$(cd "$(dirname "$0")" && pwd)"
DATA="$STUDY/../enterprise_split/data/v1/C-time"
OUT="$STUDY/results/pilot"
SEEDS=20261401,20261402,20261403
mkdir -p "$OUT" "$STUDY/logs"
cd "$STUDY/src"
arm() {  # arm f [extra args]
  local a="$1" f="$2"; shift 2
  PYTHONDONTWRITEBYTECODE=1 python run_method.py --arm "$a" --target "$DATA/target_f$f" --seeds "$SEEDS" \
    --source-cache "$OUT/source_cache" --source-checkpoint source_valid --skip-test --output "$OUT/${a}_f$f.json" "$@" \
    > "$STUDY/logs/pilot_${a}_f$f.log" 2>&1 && echo "done $a f$f" || echo "FAILED $a f$f"
}
( arm dense_source 0.10; arm dense_source 0.25; arm dense_source 1.00 ) &
( arm sparse_source 0.10 --source-fraction 0.10; arm local 0.10; arm local 0.25 ) &
( arm sparse_source 0.25 --source-fraction 0.25; arm local 1.00 ) &
wait
echo "pilot finished"
