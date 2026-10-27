#!/usr/bin/env bash
# Phase-2 diagnostics (validation only; see src/diagnose.py). Pilot seeds, never used for confirmation.
set -euo pipefail
STUDY="$(cd "$(dirname "$0")" && pwd)"
SEEDS=20261401,20261402
mkdir -p "$STUDY/results/diagnostics" "$STUDY/logs"
cd "$STUDY/src"
run() {  # name rounds
  PYTHONDONTWRITEBYTECODE=1 python diagnose.py --diagnostic "$1" --seeds "$SEEDS" --rounds "$2" \
    --output "$STUDY/results/diagnostics/$1.json" > "$STUDY/logs/diagnostic_$1.log" 2>&1 \
    && echo "done $1" || echo "FAILED $1 (see logs/diagnostic_$1.log)"
}
run oracle_labels 40 &
run dense_weights 20 &
run source_sparse 30 &
run source_dense 30 &
wait
echo "all diagnostics finished"
