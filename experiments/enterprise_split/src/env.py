"""Paths shared by the enterprise-split scripts. Importing this puts the reused code on sys.path.

Reused, unmodified: experiments/fedultra_fresh/src (builder, evaluation protocol, FedAvg
helpers, bootstrap) and experiments/transfer_conditions/src (hidden-aware loader, arm runner,
arm pairing, path-rule test).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "experiments" / "enterprise_split"
FRESH_SRC = ROOT / "experiments" / "fedultra_fresh" / "src"
TC = ROOT / "experiments" / "transfer_conditions"
for path in (ROOT, FRESH_SRC, TC / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

BUNDLE = ROOT / "datasets/cybersecurity/mitre/enterprise/v19.2/enterprise-attack-19.2.json"
FROZEN_ENTERPRISE = ROOT / "experiments/fedultra_fresh/data_processed/enterprise"
