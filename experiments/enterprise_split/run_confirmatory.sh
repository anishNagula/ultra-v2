#!/usr/bin/env bash
# Runs the registered confirmatory grid (PREREGISTRATION.md), 4 jobs at a time, primary cells first.
# Re-running is safe: run_arm.py skips seeds already saved in an output file.
#   bash run_confirmatory.sh            # run
#   bash run_confirmatory.sh --dry-run  # list the jobs
#   bash run_confirmatory.sh --smoke    # the validation-only smoke test (seed 20261209)
set -euo pipefail
STUDY="$(cd "$(dirname "$0")" && pwd)"
DATA="$STUDY/data/v1/C-time"
SEEDS=20261211,20261212,20261213,20261214,20261215,20261216,20261217,20261218,20261219,20261220
mkdir -p "$STUDY/results/confirmatory" "$STUDY/results/smoke" "$STUDY/logs"

jobs() {
  if [[ "${1:-}" == "--smoke" ]]; then
    for f in 0.10 1.00; do
      echo "--arm fedavg --target $DATA/target_f$f --source $DATA/source --seeds 20261209 --skip-test --output $STUDY/results/smoke/fedavg_f$f.json"
      echo "--arm local --target $DATA/target_f$f --seeds 20261209 --skip-test --output $STUDY/results/smoke/local_f$f.json"
    done
    return
  fi
  for f in 0.10 1.00 0.25; do
    echo "--arm fedavg --target $DATA/target_f$f --source $DATA/source --seeds $SEEDS --output $STUDY/results/confirmatory/fedavg_f$f.json"
    echo "--arm local --target $DATA/target_f$f --seeds $SEEDS --output $STUDY/results/confirmatory/local_f$f.json"
  done
}

cd "$STUDY/src"
# Each job line becomes the arguments of run_job (paths contain no spaces).
run_job() {
  local out="${@: -1}"  # every job line ends with --output <path>
  local log="$STUDY/logs/$(basename "$(dirname "$out")")_$(basename "$out" .json).log"
  if PYTHONDONTWRITEBYTECODE=1 python run_arm.py "$@" > "$log" 2>&1; then
    echo "done $(basename "$out")"
  else
    echo "FAILED $(basename "$out") (see $log)"
  fi
}
export -f run_job
export STUDY
if [[ "${1:-}" == "--dry-run" ]]; then
  jobs "${2:-}" | while read -r line; do set -- $line; echo "${@: -1} <- $*"; done
  exit 0
fi
jobs "${1:-}" | xargs -P 4 -L 1 bash -c 'run_job "$@"' _
echo "all jobs finished"
