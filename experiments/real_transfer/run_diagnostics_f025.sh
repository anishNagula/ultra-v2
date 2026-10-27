#!/usr/bin/env bash
# Phase-2 diagnostics at the secondary fraction f = 0.25 (validation only; see src/diagnose.py).
set -euo pipefail
STUDY="$(cd "$(dirname "$0")" && pwd)"
SEEDS=20261401,20261402
mkdir -p "$STUDY/results/diagnostics" "$STUDY/logs"
cd "$STUDY/src"
run() {  # name rounds
  PYTHONDONTWRITEBYTECODE=1 python diagnose.py --diagnostic "$1" --fraction 0.25 --seeds "$SEEDS" --rounds "$2" \
    --output "$STUDY/results/diagnostics/$1_f0.25.json" > "$STUDY/logs/diagnostic_$1_f0.25.log" 2>&1 \
    && echo "done $1 f0.25" || echo "FAILED $1 f0.25 (see logs/diagnostic_$1_f0.25.log)"
}
run oracle_labels 40 &
run source_sparse 30 &
wait
echo "f0.25 diagnostics finished"
