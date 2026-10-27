#!/usr/bin/env bash
# Phase-2 mechanism probe for the registered FedAvg result (validation only; see src/diagnose.py).
# Seeds are registered C-time confirmatory seeds, so curves pair with the registered local/FedAvg
# validation histories. No test metric is computed.
set -euo pipefail
STUDY="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$STUDY/results/diagnostics" "$STUDY/logs"
cd "$STUDY/src"
run() {  # name seeds rounds
  PYTHONDONTWRITEBYTECODE=1 python diagnose.py --diagnostic "$1" --seeds "$2" --rounds "$3" \
    --output "$STUDY/results/diagnostics/$1.json" > "$STUDY/logs/diagnostic_$1.log" 2>&1 \
    && echo "done $1" || echo "FAILED $1 (see logs/diagnostic_$1.log)"
}
run fedavg_replay 20261211 3          # must equal the registered curve's first rounds
run fedavg_sparse 20261211,20261212,20261213 20
echo "fedavg diagnostics finished"
