#!/usr/bin/env bash
# Runs every registered job (PREREGISTRATION.md), 4 at a time, longest first.
# Re-running is safe: conditions_runner.py skips seeds already saved in an output file.
set -euo pipefail
STUDY="$(cd "$(dirname "$0")" && pwd)"
SYN="$STUDY/data/synthetic"
REAL="$STUDY/data/real"
RES="$STUDY/results"
S_SEEDS=20261111,20261112,20261113,20261114,20261115
M_SEEDS=20261121,20261122,20261123,20261124,20261125,20261126,20261127,20261128,20261129,20261130
I_SEEDS=20261141,20261142,20261143
mkdir -p "$RES/synthetic" "$RES/real" "$STUDY/logs"

jobs() {
  for f in 0.25 0.10 1.00; do
    echo "--arm fedavg --target $REAL/mobile_f$f --source enterprise --seeds $M_SEEDS --output $RES/real/fedavg_mobile_f$f.json"
  done
  for g in toolkit campaign; do for n in 13 41 136 408; do
    echo "--arm fedavg --target $SYN/target_n$n --source $SYN/source_$g --seeds $S_SEEDS --output $RES/synthetic/fedavg_${g}_n$n.json"
  done; done
  for f in 0.10 0.25 1.00; do
    echo "--arm fedavg --target $REAL/ics_f$f --source enterprise --seeds $I_SEEDS --skip-test --output $RES/real/dev_fedavg_ics_f$f.json"
  done
  for n in 13 41 136 408; do
    echo "--arm local --target $SYN/target_n$n --seeds $S_SEEDS --output $RES/synthetic/local_n$n.json"
  done
  for f in 0.10 0.25 1.00; do
    echo "--arm local --target $REAL/mobile_f$f --seeds $M_SEEDS --output $RES/real/local_mobile_f$f.json"
    echo "--arm local --target $REAL/ics_f$f --seeds $I_SEEDS --skip-test --output $RES/real/dev_local_ics_f$f.json"
  done
}

cd "$STUDY/src"
# Each job line becomes the arguments of run_job (paths contain no spaces).
run_job() {
  local out="${@: -1}"  # every job line ends with --output <path>
  local log="$STUDY/logs/$(basename "$out" .json).log"
  if PYTHONDONTWRITEBYTECODE=1 python conditions_runner.py "$@" > "$log" 2>&1; then
    echo "done $(basename "$out")"
  else
    echo "FAILED $(basename "$out") (see $log)"
  fi
}
export -f run_job
export STUDY
if [[ "${1:-}" == "--dry-run" ]]; then
  jobs | while read -r line; do set -- $line; echo "${@: -1} <- $*"; done
  exit 0
fi
jobs | xargs -P 4 -L 1 bash -c 'run_job "$@"' _
echo "all jobs finished"
