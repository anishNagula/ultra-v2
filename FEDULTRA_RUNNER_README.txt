FedULTRA local-vs-FedAvg runner

This script runs local fine-tuning baselines and FedAvg from the same pretrained
ULTRA checkpoint. Each client gets the same number of optimizer updates in local
and federated conditions. Model selection uses validation MRR only. The test set
is evaluated after selection; the model's message-passing graph contains training
triples only. Validation filtering uses train+valid; test filtering uses all facts.

Apply by merging this ZIP into ~/code/ultra-v2, never replace the experiments folder:
  cd ~/code/ultra-v2
  unzip -o ~/Downloads/fedultra_runner_patch.zip -d .
  source .venv/bin/activate

First run a tiny diagnostic only:
  python experiments/fedultra_fresh/src/fedavg_runner.py \
    --seeds 20261023 --rounds 1 --local-steps 1 --num-negative 8 \
    --eval-limit 4 --output experiments/fedultra_fresh/results/pilot.json

A diagnostic subset is only a plumbing check, not a performance result.

If the pilot passes, run the planned full experiment:
  python experiments/fedultra_fresh/src/fedavg_runner.py \
    --seeds 20261023,20261024,20261025 --rounds 5 --local-steps 10 \
    --num-negative 64 --eval-limit 0 \
    --output experiments/fedultra_fresh/results/fedavg_comparison.json

The full experiment may take substantial time on CPU because validation/test rank
all candidate entities. Do not use --eval-limit for reported benchmark results.
The runner saves progress after each completed seed. Use a new output path for
each new run; --overwrite intentionally replaces the target JSON.

Reported acceptance checks: mean FedAvg MRR gain >= +0.03, seed-level 95% paired
t interval excludes zero, positive mean gain, and neither client's mean MRR
regression is below -0.02. With only three seeds, confidence intervals can be wide;
report the actual interval and do not claim success if criteria fail.
