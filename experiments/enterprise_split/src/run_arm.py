#!/usr/bin/env python3
"""Run one arm (local or FedAvg) of the Enterprise-split study.

This is the transfer-conditions arm runner (`transfer_conditions/src/conditions_runner.py`),
unchanged except for the protocol label it records. Both arms share:
- the pretrained checkpoint;
- AdamW;
- the 20 rounds x 10 steps budget;
- the target sampling stream;
- validation-based round selection;
- a single test scoring.

FedAvg weights the two clients equally. Each client trains only on its own graph, and only
model weights are averaged.

  python run_arm.py --arm fedavg --target ../data/v1/C-time/target_f0.10 --source ../data/v1/C-time/source \
      --seeds 20261211 --output ../results/confirmatory/fedavg_f0.10.json
"""
import env  # noqa: F401  (puts the reused code on sys.path)
import conditions_runner as runner

runner.PROTOCOL_VERSION = "enterprise-split v1 (PREREGISTRATION.md)"

if __name__ == "__main__":
    runner.main()
