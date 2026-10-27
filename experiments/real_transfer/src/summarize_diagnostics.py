#!/usr/bin/env python3
"""Summarise the Phase-2 diagnostics (validation only) into results/diagnostics/summary.json.

For every diagnostic curve: round 0, best value and round, the mean of the last 10 rounds (a plateau
level less inflated by selection than the best value), and the range after round 1. Also the registered
C-time validation results and the paired FedAvg comparison. DIAGNOSIS.md cites this file.

  python summarize_diagnostics.py
"""
import json
from pathlib import Path

from env import STUDY, ROOT

DIAG = STUDY / "results" / "diagnostics"
REGISTERED = ROOT / "experiments" / "enterprise_split" / "results" / "confirmatory"


def ranks(xs):
    order = sorted(range(len(xs)), key=xs.__getitem__)
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):  # average ranks over ties
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5


def checkpoint_rules(path, fraction):
    """Target zero-shot validation under each source-checkpoint rule, on the same source trajectory."""
    rows = []
    for r in json.loads(path.read_text())["results"]:
        source, target = r["curves"]["source_valid"], r["curves"][f"target_f{fraction}_valid"]
        best_source = max(range(len(source)), key=source.__getitem__)
        best_target = max(range(len(target)), key=target.__getitem__)
        rows.append({"seed": r["seed"],
                     "source_valid_argmax": {"round": best_source, "target_valid": target[best_source]},
                     "final_round": {"round": len(target) - 1, "target_valid": target[-1]},
                     "target_valid_argmax": {"round": best_target, "target_valid": target[best_target]},
                     "spearman_source_vs_target_rounds_1_on": spearman(source[1:], target[1:])})
    mean = {rule: sum(x[rule]["target_valid"] for x in rows) / len(rows)
            for rule in ("source_valid_argmax", "final_round", "target_valid_argmax")}
    return {"per_seed": rows, "mean_target_valid": mean,
            "mean_spearman": sum(x["spearman_source_vs_target_rounds_1_on"] for x in rows) / len(rows)}


def describe(curve):
    best = max(range(len(curve)), key=curve.__getitem__)
    return {"round0": curve[0], "best": curve[best], "best_round": best,
            "mean_last10": sum(curve[-10:]) / 10, "min_after_round1": min(curve[1:]), "max_after_round1": max(curve[1:]),
            "rounds": len(curve) - 1}


def main():
    out = {"note": "validation only; no test metric is read except the registered test summary", "curves": {}}
    for path in sorted(DIAG.glob("*.json")):
        if path.stem in ("summary", "reachability", "popularity"):
            continue
        data = json.loads(path.read_text())
        out["curves"][path.stem] = {f"{r['seed']}:{name}": describe(curve)
                                    for r in data["results"] for name, curve in r["curves"].items()}
    registered = {}
    for f in ("0.10", "0.25", "1.00"):
        for arm in ("local", "fedavg"):
            runs = json.loads((REGISTERED / f"{arm}_f{f}.json").read_text())["results"]
            best = [r["selected_validation_mrr"] for r in runs]
            registered[f"{arm}_f{f}"] = {"mean_best_validation": sum(best) / len(best), "min": min(best), "max": max(best),
                                         "mean_selected_round": sum(r["selected_round"] for r in runs) / len(runs),
                                         "by_seed": {r["seed"]: r for r in runs}}
    out["registered_validation"] = {k: {x: y for x, y in v.items() if x != "by_seed"} for k, v in registered.items()}
    sparse_runs = json.loads((DIAG / "fedavg_sparse.json").read_text())["results"]
    paired = []
    for r in sparse_runs:
        curve = r["curves"]["target_f0.10_valid"]
        local, fedavg = (registered[f"{a}_f0.10"]["by_seed"][r["seed"]] for a in ("local", "fedavg"))
        paired.append({"seed": r["seed"], "local_best": local["selected_validation_mrr"],
                       "fedavg_best": fedavg["selected_validation_mrr"], "fedavg_sparse_best": max(curve),
                       "local_mean_rounds_1_20": sum(local["validation_mrr_history"][1:]) / 20,
                       "fedavg_mean_rounds_1_20": sum(fedavg["validation_mrr_history"][1:]) / 20,
                       "fedavg_sparse_mean_rounds_1_20": sum(curve[1:]) / 20})
    out["fedavg_sparse_vs_registered_f0.10"] = {
        "per_seed": paired,
        "mean": {k: sum(p[k] for p in paired) / len(paired) for k in paired[0] if k != "seed"}}
    replay = json.loads((DIAG / "fedavg_replay.json").read_text())["results"][0]
    reg_curve = registered["fedavg_f0.10"]["by_seed"][replay["seed"]]["validation_mrr_history"]
    rep_curve = replay["curves"]["target_f0.10_valid"]
    out["fedavg_replay_matches_registered"] = rep_curve == reg_curve[:len(rep_curve)]
    for name in ("reachability", "popularity"):
        out[name] = json.loads((DIAG / f"{name}.json").read_text())
    out["source_checkpoint_rules"] = {f"f{f}": checkpoint_rules(DIAG / f"source_select_f{f}.json", f)
                                      for f in ("0.10", "0.25") if (DIAG / f"source_select_f{f}.json").exists()}
    (DIAG / "summary.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for name, curves in out["curves"].items():
        for key, d in curves.items():
            print(f"{name:<22} {key:<32} r0 {d['round0']:.4f}  best {d['best']:.4f}@{d['best_round']:<2}  "
                  f"last10 {d['mean_last10']:.4f}  after-r1 [{d['min_after_round1']:.4f}, {d['max_after_round1']:.4f}]")
    for k, v in out["registered_validation"].items():
        print(f"registered {k:<12} mean best {v['mean_best_validation']:.4f} [{v['min']:.4f}, {v['max']:.4f}] "
              f"mean selected round {v['mean_selected_round']:.1f}")
    print("fedavg_sparse vs registered (f0.10, paired):", {k: round(v, 4) for k, v in out["fedavg_sparse_vs_registered_f0.10"]["mean"].items()})
    print("fedavg_replay matches registered:", out["fedavg_replay_matches_registered"])
    for f, rules in out["source_checkpoint_rules"].items():
        for row in rules["per_seed"]:
            print(f"source checkpoint {f} seed {row['seed']}: " + "  ".join(
                f"{rule} r{row[rule]['round']} {row[rule]['target_valid']:.4f}"
                for rule in ("source_valid_argmax", "final_round", "target_valid_argmax"))
                  + f"  spearman {row['spearman_source_vs_target_rounds_1_on']:+.2f}")
        print(f"source checkpoint {f} mean: " + "  ".join(f"{k} {v:.4f}" for k, v in rules["mean_target_valid"].items())
              + f"  mean spearman {rules['mean_spearman']:+.2f}")


if __name__ == "__main__":
    main()
