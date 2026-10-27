"""Paths shared by the real-transfer scripts. Importing this puts the reused code on sys.path.

Reused, unmodified: experiments/fedultra_fresh/src (model helpers, local training step, scorer,
bootstrap) and experiments/transfer_conditions/src (hidden-aware loader, local arm runner).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "experiments" / "real_transfer"
FRESH_SRC = ROOT / "experiments" / "fedultra_fresh" / "src"
TC_SRC = ROOT / "experiments" / "transfer_conditions" / "src"
for path in (ROOT, FRESH_SRC, TC_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# The registered C-time benchmark of experiments/enterprise_split (data are read, never written).
DATA = ROOT / "experiments" / "enterprise_split" / "data" / "v1" / "C-time"
SOURCE = DATA / "source"
TARGET = {f: DATA / f"target_f{f}" for f in ("0.10", "0.25", "1.00")}
ACTOR_PREFIXES = ("intrusion-set--", "campaign--")


def is_actor_use(triple):
    """The target's evaluated task: a threat actor (group or campaign) `uses` a technique or software."""
    head, relation, _ = triple
    return relation == "uses" and head.startswith(ACTOR_PREFIXES)
