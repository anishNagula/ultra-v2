#!/usr/bin/env python3
"""Registered analysis for the transfer-conditions study (PREREGISTRATION.md).

Pairs local and FedAvg runs, applies bootstrap_cyber_transfer.py's full success
criterion to every registered cell, and bootstraps the registered contrasts
(H2: n=13 vs n=408; H3: shared vs disjoint source) with the same resampling scheme:
seeds and test triples are resampled jointly, triples shared across chosen seeds.
ICS development runs are validation-only and summarised descriptively.
"""
import json
import random
import subprocess
import sys
from pathlib import Path

from common import FRESH_SRC, STUDY
import transfer_runner as transfer

RES = STUDY / "results"
SAMPLES, BOOT_SEED = 20_000, 20261026
LEVELS = (13, 41, 136, 408)
FRACTIONS = ("0.10", "0.25", "1.00")


def compare_and_bootstrap(local, fedavg, out_dir, stem):
    comparison, boot = out_dir / f"{stem}_comparison.json", out_dir / f"{stem}_bootstrap.json"
    subprocess.run([sys.executable, str(STUDY / "src" / "compare_arms.py"), "--local", str(local),
                    "--fedavg", str(fedavg), "--output", str(comparison)], check=True, capture_output=True)
    subprocess.run([sys.executable, str(FRESH_SRC / "bootstrap_cyber_transfer.py"), "--input", str(comparison),
                    "--output", str(boot), "--bootstrap-samples", str(SAMPLES), "--seed", str(BOOT_SEED)],
                   check=True, capture_output=True)
    return json.loads(comparison.read_text()), json.loads(boot.read_text())


def contrast(cmp_a, cmp_b):
    """Bootstrap mean(delta_a - delta_b) over the same seeds and the same test triples."""
    a = {r["seed"]: r["per_query_test_comparison"] for r in cmp_a["results"]}
    b = {r["seed"]: r["per_query_test_comparison"] for r in cmp_b["results"]}
    if set(a) != set(b):
        raise ValueError("contrast needs the same seeds")
    seeds = sorted(a)
    diffs = []
    for s in seeds:
        if [q["triple"] for q in a[s]] != [q["triple"] for q in b[s]]:
            raise ValueError("contrast needs the same test triples in the same order")
        diffs.append([x["delta_mrr"] - y["delta_mrr"] for x, y in zip(a[s], b[s])])
    seed_means = [sum(d) / len(d) for d in diffs]
    rng = random.Random(BOOT_SEED)
    n_s, n_q = len(diffs), len(diffs[0])
    boot = []
    for _ in range(SAMPLES):
        si = [rng.randrange(n_s) for _ in range(n_s)]
        qi = [rng.randrange(n_q) for _ in range(n_q)]
        boot.append(sum(diffs[s][q] for s in si for q in qi) / (n_s * n_q))
    boot.sort()
    return {"mean": sum(seed_means) / n_s, "per_seed": seed_means,
            "seed_level_95pct_t_ci": transfer.ci95(seed_means),
            "hierarchical_paired_bootstrap_95pct_ci": [boot[int(0.025 * SAMPLES)], boot[int(0.975 * SAMPLES) - 1]]}


def cell_summary(comparison, boot):
    s = comparison["summary"]
    return {"mean_local_test_mrr": s["mean_local_test_mrr"], "mean_fedavg_test_mrr": s["mean_fedavg_test_mrr"],
            "mean_delta": boot["mean_delta_mrr"], "per_seed_delta": boot["per_seed_mean_delta_mrr"],
            "seed_level_95pct_t_ci": boot["seed_level_95pct_t_ci"],
            "bootstrap_95pct_ci": boot["hierarchical_paired_bootstrap_95pct_ci"],
            "meets_full_success_criterion": boot["acceptance_criteria"]["meets_full_success_criterion"],
            "selected_rounds_local": [r["local"]["selected_round"] for r in comparison["results"]],
            "selected_rounds_fedavg": [r["fedavg"]["selected_round"] for r in comparison["results"]],
            "mean_delta_by_relation": s["mean_delta_by_relation"]}


def fmt(c):
    t, b = c["seed_level_95pct_t_ci"], c["bootstrap_95pct_ci"]
    return (f"local {c['mean_local_test_mrr']:.4f} | FedAvg {c['mean_fedavg_test_mrr']:.4f} | "
            f"Δ {c['mean_delta']:+.4f} | t-CI [{t[0]:+.4f}, {t[1]:+.4f}] | boot [{b[0]:+.4f}, {b[1]:+.4f}] | "
            f"{'PASS' if c['meets_full_success_criterion'] else 'fail'}")


def main():
    out = {"study_S": {}, "study_R_mobile": {}, "study_R_ics_validation_dev": {}, "contrasts": {}}
    syn, real = RES / "synthetic", RES / "real"
    comparisons = {}
    for g in ("toolkit", "campaign"):
        for n in LEVELS:
            local, fed = syn / f"local_n{n}.json", syn / f"fedavg_{g}_n{n}.json"
            if not (local.exists() and fed.exists()):
                continue
            cmp_, boot = compare_and_bootstrap(local, fed, syn, f"{g}_n{n}")
            comparisons[(g, n)] = cmp_
            out["study_S"][f"{'shared' if g == 'toolkit' else 'disjoint'}_n{n}"] = cell_summary(cmp_, boot)
    if ("toolkit", 13) in comparisons and ("toolkit", 408) in comparisons:
        out["contrasts"]["H2_shared_delta_n13_minus_n408"] = contrast(comparisons[("toolkit", 13)], comparisons[("toolkit", 408)])
    for n in LEVELS:
        if ("toolkit", n) in comparisons and ("campaign", n) in comparisons:
            out["contrasts"][f"H3_delta_shared_minus_disjoint_n{n}"] = contrast(comparisons[("toolkit", n)], comparisons[("campaign", n)])
    for f in FRACTIONS:
        local, fed = real / f"local_mobile_f{f}.json", real / f"fedavg_mobile_f{f}.json"
        if local.exists() and fed.exists():
            out["study_R_mobile"][f"f{f}"] = cell_summary(*compare_and_bootstrap(local, fed, real, f"mobile_f{f}"))
        local, fed = real / f"dev_local_ics_f{f}.json", real / f"dev_fedavg_ics_f{f}.json"
        if local.exists() and fed.exists():
            l = {r["seed"]: r for r in json.loads(local.read_text())["results"]}
            fe = {r["seed"]: r for r in json.loads(fed.read_text())["results"]}
            seeds = sorted(set(l) & set(fe))
            d = [fe[s]["selected_validation_mrr"] - l[s]["selected_validation_mrr"] for s in seeds]
            out["study_R_ics_validation_dev"][f"f{f}"] = {
                "seeds": seeds, "zero_shot_validation_mrr": l[seeds[0]]["validation_mrr_history"][0] if seeds else None,
                "mean_local_selected_validation_mrr": sum(l[s]["selected_validation_mrr"] for s in seeds) / len(seeds),
                "mean_fedavg_selected_validation_mrr": sum(fe[s]["selected_validation_mrr"] for s in seeds) / len(seeds),
                "mean_validation_delta": sum(d) / len(d), "per_seed_validation_delta": d,
                "note": "development only: validation MRR at the validation-selected round (optimistic for both arms)"}
    (RES / "analysis_summary.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for section in ("study_S", "study_R_mobile"):
        print(f"\n== {section} (target test MRR, FedAvg - local) ==")
        for k, c in out[section].items():
            print(f"  {k:<14} {fmt(c)}")
    print("\n== registered contrasts ==")
    for k, c in out["contrasts"].items():
        t, b = c["seed_level_95pct_t_ci"], c["hierarchical_paired_bootstrap_95pct_ci"]
        print(f"  {k:<36} {c['mean']:+.4f}  t-CI [{t[0]:+.4f}, {t[1]:+.4f}]  boot [{b[0]:+.4f}, {b[1]:+.4f}]")
    print("\n== ICS development (validation only, descriptive) ==")
    for k, c in out["study_R_ics_validation_dev"].items():
        print(f"  {k}: zero-shot {c['zero_shot_validation_mrr']:.4f} local {c['mean_local_selected_validation_mrr']:.4f} "
              f"FedAvg {c['mean_fedavg_selected_validation_mrr']:.4f} Δ {c['mean_validation_delta']:+.4f}")


if __name__ == "__main__":
    main()
