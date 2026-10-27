#!/usr/bin/env python3
"""Calibrate the transfer-opportunity proxy against earlier FedAvg - local results (descriptive).

Proxy = target validation rule MRR with pooled (target + source) path-rule statistics minus
target-only statistics: the k <= 3 rule test of transfer_conditions/src/explore_enrichment.py.
It is computed for every earlier cell whose FedAvg - local result is known (Study S and Mobile:
test delta; ICS: validation delta). Reads train, valid and hidden facts only; no test triple.
"""
import json
import time

import numpy as np
import torch

from env import FROZEN_ENTERPRISE, STUDY, TC
from evaluation_protocol import read_triples
import explore_enrichment as rules


def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])


def domain(folder):
    optional = lambda name: read_triples(folder / name) if (folder / name).is_file() else []
    return rules.Domain(folder.name, read_triples(folder / "train.txt"), optional("valid.txt"), optional("hidden.txt"))


def main():
    torch.set_num_threads(8)
    known = json.loads((TC / "results/analysis_summary.json").read_text())
    syn, real = TC / "data/synthetic", TC / "data/real"
    sources = {"toolkit": syn / "source_toolkit", "campaign": syn / "source_campaign", "enterprise": FROZEN_ENTERPRISE}
    targets = {f"target_n{n}": (syn / f"target_n{n}", {
        "toolkit": ("Study S shared", known["study_S"][f"shared_n{n}"]["mean_delta"], "test"),
        "campaign": ("Study S disjoint", known["study_S"][f"disjoint_n{n}"]["mean_delta"], "test")})
        for n in (13, 41, 136, 408)}
    for f in ("0.10", "0.25", "1.00"):
        targets[f"mobile_f{f}"] = (real / f"mobile_f{f}", {
            "enterprise": ("Mobile", known["study_R_mobile"][f"f{f}"]["mean_delta"], "test")})
        targets[f"ics_f{f}"] = (real / f"ics_f{f}", {
            "enterprise": ("ICS", known["study_R_ics_validation_dev"][f"f{f}"]["mean_validation_delta"], "validation")})
    t0 = time.time()
    source_stats = {name: rules.rule_stats(domain(folder)) for name, folder in sources.items()}
    print(f"source rule statistics in {time.time() - t0:.0f}s", flush=True)
    rows = []
    for tname, (folder, cells) in targets.items():
        dom = domain(folder)
        own = rules.rule_stats(dom)
        tabs = {}
        for sname in cells:
            t = rules.tables(dom, own, source_stats[sname])
            tabs["own"], tabs[f"pooled:{sname}"], tabs[f"source:{sname}"] = t["own"], t["pooled"], t["source"]
        mrr = rules.evaluate(dom, tabs)
        for sname, (study, delta, split) in cells.items():
            row = {"cell": f"{study}, {tname}", "own": mrr["own"], "source": mrr[f"source:{sname}"], "pooled": mrr[f"pooled:{sname}"],
                   "proxy": mrr[f"pooled:{sname}"] - mrr["own"], "fedavg_minus_local": delta, "delta_split": split}
            rows.append(row)
            print(f"  {row['cell']:<34} own {row['own']:.4f} source {row['source']:.4f} pooled {row['pooled']:.4f} proxy {row['proxy']:+.4f} | "
                  f"FedAvg-local ({split}) {delta:+.4f}  [{time.time() - t0:.0f}s]", flush=True)
    rho = spearman([r["proxy"] for r in rows], [r["fedavg_minus_local"] for r in rows])
    out = {"note": __doc__.strip(), "cells": rows, "spearman_rho": rho}
    (STUDY / "results").mkdir(exist_ok=True)
    (STUDY / "results/proxy_calibration.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"Spearman rho(proxy, FedAvg - local) = {rho:.3f} over {len(rows)} cells")


if __name__ == "__main__":
    main()
