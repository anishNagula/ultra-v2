#!/usr/bin/env bash
# The registered analysis (PREREGISTRATION.md). Run once, after every confirmatory run file holds all
# 10 seeds. Every comparison listed here is reported, whatever its outcome.
set -euo pipefail
STUDY="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$STUDY/../.." && pwd)"
R="$STUDY/results/confirmatory"
A="$R/analysis"
# Every registered file must still match its hash (PREREGISTRATION.md, Registration).
(cd "$ROOT" && shasum -a 256 -c --quiet experiments/real_transfer/results/preregistration_hashes.txt) \
  || { echo "a registered file changed; see results/README.md" >&2; exit 1; }
echo "registered hashes verified"
mkdir -p "$A"
cd "$STUDY/src"
cmp() {  # name baseline-file baseline-score candidate-file candidate-score
  PYTHONDONTWRITEBYTECODE=1 python analyze_transfer.py --baseline "$R/$2.json" --baseline-score "$3" \
    --candidate "$R/$4.json" --candidate-score "$5" --output "$A/$1.json" 2>&1 | grep -v -E "UserWarning|warnings.warn"
}
echo "== Primary"
cmp P1_f0.10 local_f0.10 test sparse_source_f0.10 test
echo "== Secondary: the same comparison at the other fractions"
cmp P2_f0.25 local_f0.25 test sparse_source_f0.25 test
cmp P3_f1.00 local_f1.00 test dense_source_f1.00 test
echo "== Secondary: zero-shot transfer against the pretrained model (no target training on either side)"
cmp Z1_f0.10 local_f0.10 zero_shot sparse_source_f0.10 zero_shot
cmp Z2_f0.25 local_f0.25 zero_shot sparse_source_f0.25 zero_shot
cmp Z3_f1.00 local_f1.00 zero_shot dense_source_f1.00 zero_shot
echo "== Mechanism (DIAGNOSIS.md E2): directional predictions, no +0.03 threshold"
cmp M1_f0.10 local_f0.10 zero_shot dense_source_f0.10 zero_shot
cmp M2_f0.10 dense_source_f0.10 zero_shot sparse_source_f0.10 zero_shot
cmp M3_f0.10 local_f0.10 test dense_source_f0.10 test
cmp M4_f0.25 dense_source_f0.25 zero_shot sparse_source_f0.25 zero_shot
echo "== Reference: zero-shot transfer against fine-tuned local"
cmp R1_f0.10 local_f0.10 test sparse_source_f0.10 zero_shot
cmp R2_f0.25 local_f0.25 test sparse_source_f0.25 zero_shot
echo "analysis finished"
