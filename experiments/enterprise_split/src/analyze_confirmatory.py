#!/usr/bin/env python3
"""Registered analysis for the Enterprise-split study (PREREGISTRATION.md).

For each target fraction, pairs the local and FedAvg runs by seed and applies
bootstrap_cyber_transfer.py's full success criterion. H1 is the f = 0.10 cell; f = 0.25 and
f = 1.00 are secondary. With --smoke, it instead checks the validation-only smoke-test gates.

  python analyze_confirmatory.py            # after run_confirmatory.sh
  python analyze_confirmatory.py --smoke    # after the smoke test
"""
import argparse
import json

from env import STUDY
import analyze as tc  # transfer_conditions/src/analyze.py: compare_and_bootstrap, cell_summary, fmt

FRACTIONS = ("0.10", "0.25", "1.00")
PRIMARY = "0.10"


def smoke():
    d = STUDY / "results/smoke"
    load = lambda name: json.loads((d / f"{name}.json").read_text())["results"][0]
    runs = {(arm, f): load(f"{arm}_f{f}") for arm in ("local", "fedavg") for f in ("0.10", "1.00")}
    out = {"seconds_per_seed": {f"{a}_f{f}": r["seconds"] for (a, f), r in runs.items()},
           "best_validation_mrr": {f"{a}_f{f}": r["selected_validation_mrr"] for (a, f), r in runs.items()},
           "selected_round": {f"{a}_f{f}": r["selected_round"] for (a, f), r in runs.items()}}
    out["gate_b_round0_parity"] = {f: runs[("local", f)]["validation_mrr_history"][0] == runs[("fedavg", f)]["validation_mrr_history"][0]
                                   for f in ("0.10", "1.00")}
    gap = runs[("local", "1.00")]["selected_validation_mrr"] - runs[("local", "0.10")]["selected_validation_mrr"]
    out["gate_c_local_gap_f1.00_minus_f0.10"] = gap
    out["gate_c_passes"] = gap >= 0.03
    out["note"] = "validation only; FedAvg numbers are descriptive and enter no decision"
    (d / "smoke_summary.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    if p.parse_args().smoke:
        return smoke()
    res = STUDY / "results/confirmatory"
    out = {"primary_fraction": PRIMARY, "cells": {}}
    for f in FRACTIONS:
        local, fed = res / f"local_f{f}.json", res / f"fedavg_f{f}.json"
        if local.exists() and fed.exists():
            out["cells"][f"f{f}"] = tc.cell_summary(*tc.compare_and_bootstrap(local, fed, res, f"f{f}"))
    primary = out["cells"].get(f"f{PRIMARY}")
    out["H1_supported"] = bool(primary and primary["meets_full_success_criterion"])
    (res / "analysis_summary.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("== target test MRR, FedAvg - local ==")
    for k, c in out["cells"].items():
        print(f"  {k}{' (H1)' if k == f'f{PRIMARY}' else ' (secondary)'}: {tc.fmt(c)}")
    print(f"H1 supported: {out['H1_supported']}")


if __name__ == "__main__":
    main()
