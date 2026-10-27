#!/usr/bin/env bash
# Phase-2/pilot probe of the source-stage round choice (validation only; see src/diagnose.py source_select).
# Pilot seeds: the source trajectory is identical to the pilot's sparse_source arm of the same seed.
set -euo pipefail
STUDY="$(cd "$(dirname "$0")" && pwd)"
SEEDS=20261401,20261402,20261403
mkdir -p "$STUDY/results/diagnostics" "$STUDY/logs"
cd "$STUDY/src"
run() {  # fraction
  PYTHONDONTWRITEBYTECODE=1 python diagnose.py --diagnostic source_select --fraction "$1" --seeds "$SEEDS" --rounds 30 \
    --output "$STUDY/results/diagnostics/source_select_f$1.json" > "$STUDY/logs/diagnostic_source_select_f$1.log" 2>&1 \
    && echo "done source_select f$1" || echo "FAILED source_select f$1"
}
run 0.10 &
run 0.25 &
wait
echo "source_select diagnostics finished"
