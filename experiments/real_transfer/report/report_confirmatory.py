#!/usr/bin/env python3
"""Checks and tables for the confirmatory results. Added after registration and before the analysis ran;
it is not one of the registered files. It reads the run files and the registered analysis outputs,
computes descriptive means only, and selects nothing.

  python report_confirmatory.py --check   # before run_analysis.sh: structure and hashes only, no metric is read
  python report_confirmatory.py           # after run_analysis.sh: writes results/confirmatory/report.md
"""
import argparse
import json
import statistics
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
R = STUDY / "results" / "confirmatory"
A = R / "analysis"
DATA = "experiments/enterprise_split/data/v1/C-time"
SEEDS = list(range(20261501, 20261511))
RUNS = [("local", "0.10"), ("local", "0.25"), ("local", "1.00"), ("sparse_source", "0.10"),
        ("sparse_source", "0.25"), ("dense_source", "0.10"), ("dense_source", "0.25"), ("dense_source", "1.00")]
TARGET_STAGE = {"rounds": 20, "local_steps": 10, "batch_size": 8, "num_negative": 64, "eval_limit": 0,
                "lr": 5e-4, "weight_decay": 0.0, "skip_test": False, "threads": 2}
# Name, role, what is compared: run_analysis.sh and PREREGISTRATION.md, in that order.
COMPARISONS = [
    ("P1_f0.10", "primary", "sparse_source[test] − local[test]"),
    ("P2_f0.25", "secondary", "sparse_source[test] − local[test]"),
    ("P3_f1.00", "secondary", "dense_source[test] − local[test]"),
    ("Z1_f0.10", "secondary (zero-shot question)", "sparse_source[zero_shot] − pretrained"),
    ("Z2_f0.25", "secondary", "sparse_source[zero_shot] − pretrained"),
    ("Z3_f1.00", "secondary", "dense_source[zero_shot] − pretrained"),
    ("M1_f0.10", "mechanism", "dense_source[zero_shot] − pretrained"),
    ("M2_f0.10", "mechanism", "sparse_source[zero_shot] − dense_source[zero_shot]"),
    ("M3_f0.10", "mechanism", "dense_source[test] − local[test]"),
    ("M4_f0.25", "mechanism", "sparse_source[zero_shot] − dense_source[zero_shot]"),
    ("R1_f0.10", "reference", "sparse_source[zero_shot] − local[test]"),
    ("R2_f0.25", "reference", "sparse_source[zero_shot] − local[test]"),
]
# The registered directional predictions (PREREGISTRATION.md, Hypotheses), as tests on the analysis output.
PREDICTIONS = {
    "P1_f0.10": ("FAIL; upper end of the t-CI below +0.03", lambda o: not o["PASS"] and o["seed_level_95pct_t_ci"][1] < 0.03),
    "M1_f0.10": ("mean below +0.01", lambda o: o["mean_delta"] < 0.01),
    "M2_f0.10": ("t-CI above 0", lambda o: o["seed_level_95pct_t_ci"][0] > 0),
    "M3_f0.10": ("mean within ±0.01", lambda o: abs(o["mean_delta"]) <= 0.01),
    "M4_f0.25": ("t-CI above 0", lambda o: o["seed_level_95pct_t_ci"][0] > 0),
}


def registered_hashes():
    lines = (STUDY / "results" / "preregistration_hashes.txt").read_text().splitlines()
    return {path: digest for digest, path in (line.split(None, 1) for line in lines)}


def load(arm, f):
    return json.loads((R / f"{arm}_f{f}.json").read_text())


def check():
    """Structure, configuration and hashes of every run file. No metric value is read or printed."""
    hashes, problems, triples, dense_states = registered_hashes(), [], {}, {}
    for arm, f in RUNS:
        name, run = f"{arm}_f{f}", load(arm, f)
        folder = f"{DATA}/target_f{f}"
        expect = {"protocol": "real-transfer v1", "arm": arm, "target": str(ROOT / folder), "diagnostic_only": False,
                  "checkpoint_sha256": hashes["ckpts/ultra_4g.pth"],
                  "target_sha256": {p.rsplit("/", 1)[1]: d for p, d in hashes.items() if p.startswith(folder + "/")},
                  "source_sha256": None if arm == "local" else hashes[f"{DATA}/source/train.txt"]}
        problems += [f"{name}: {k}" for k, v in expect.items() if run[k] != v]
        config = dict(TARGET_STAGE)
        if arm != "local":
            config |= {"source_rounds": 30, "source_task": "held-out actor `uses` facts", "source_checkpoint": "final",
                       "source_fraction": float(f) if arm == "sparse_source" else 1.0}
        if run["config"] != config:
            problems.append(f"{name}: config {run['config']}")
        if [r["seed"] for r in run["results"]] != SEEDS:
            problems.append(f"{name}: seeds {[r['seed'] for r in run['results']]}")
        for r in run["results"]:
            for label, queries in (("test", r["per_query_test"]), ("zero_shot", r["zero_shot"]["per_query_test"])):
                order = [tuple(q["triple"]) for q in queries]
                if len(order) != 314 or triples.setdefault(f, order) != order:
                    problems.append(f"{name} seed {r['seed']}: {label} triples")
            if len(r["validation_mrr_history"]) != 21:
                problems.append(f"{name} seed {r['seed']}: validation curve")
            w = r["starting_weights"]
            if arm != "local" and (w["checkpoint_rule"] != "final" or w["selected_round"] != 30):
                problems.append(f"{name} seed {r['seed']}: source checkpoint")
            if arm == "dense_source" and dense_states.setdefault(r["seed"], w["state_sha256"]) != w["state_sha256"]:
                problems.append(f"{name} seed {r['seed']}: dense source weights differ across fractions")
    if problems:
        raise SystemExit("check failed:\n  " + "\n  ".join(problems))
    print(f"check passed: {len(RUNS)} run files x {len(SEEDS)} registered seeds; 314 aligned test triples per "
          "fraction, fine-tuned and zero-shot; registered protocol, configuration, checkpoint and data hashes; "
          "final-round source weights (round 30); identical dense source weights across fractions per seed")


def f4(x):
    return f"{x:+.4f}"


def ci(c):
    return f"[{f4(c[0])}, {f4(c[1])}]"


def report():
    out = {name: json.loads((A / f"{name}.json").read_text()) for name, _, _ in COMPARISONS}
    lines = ["# Confirmatory results (generated by `report/report_confirmatory.py`)", "",
             "## Registered comparisons", "",
             "| Comparison | Role | Δ | Baseline MRR | Candidate MRR | Mean Δ | Seed-level t-CI 95% | "
             "Hierarchical bootstrap CI 95% | Seeds with Δ > 0 | Full criterion |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, role, what in COMPARISONS:
        o = out[name]
        wins = sum(s["delta"] > 0 for s in o["per_seed"])
        lines.append(f"| {name} | {role} | {what} | {o['mean_baseline_test_mrr']:.4f} | {o['mean_candidate_test_mrr']:.4f} | "
                     f"**{f4(o['mean_delta'])}** | {ci(o['seed_level_95pct_t_ci'])} | {ci(o['bootstrap_95pct_ci'])} | "
                     f"{wins}/{len(o['per_seed'])} | {'PASS' if o['PASS'] else 'fail'} |")
    lines += ["", "## Registered directional predictions", "", "| Comparison | Prediction | Holds |", "|---|---|---|"]
    lines += [f"| {name} | {text} | {'yes' if test(out[name]) else 'no'} |" for name, (text, test) in PREDICTIONS.items()]
    lines += ["", "## Per seed", "",
              "Δ, then the selected target round, baseline / candidate.", "",
              "| Seed | " + " | ".join(n for n, _, _ in COMPARISONS[:6]) + " |", "|---|" + "---|" * 6]
    for i, seed in enumerate(SEEDS):
        cells = []
        for name, _, _ in COMPARISONS[:6]:
            s = out[name]["per_seed"][i]
            assert s["seed"] == seed
            rounds = f"{s['selected_round_baseline']} / {s['selected_round_candidate']}" if name[0] == "P" else "0 / 0"
            cells.append(f"{f4(s['delta'])} ({rounds})")
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")
    lines += ["", "## Per arm (descriptive, mean over seeds)", "",
              "| f | Arm | Test MRR | Hits@1 | Hits@3 | Hits@10 | Zero-shot test MRR | Best validation MRR | "
              "Mean selected round | Seconds per seed, job | Seconds per seed, source stage |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    mean = statistics.mean
    for arm, f in sorted(RUNS, key=lambda x: (x[1], x[0] != "local", x[0])):
        res = load(arm, f)["results"]
        m = {k: mean(r["test"][k] for r in res) for k in ("mrr", "hits@1", "hits@3", "hits@10")}
        source = "–" if arm == "local" else f"{mean(r['starting_weights']['source_seconds'] for r in res):.0f}"
        lines.append(f"| {f} | {arm} | {m['mrr']:.4f} | {m['hits@1']:.4f} | {m['hits@3']:.4f} | {m['hits@10']:.4f} | "
                     f"{mean(r['zero_shot']['test']['mrr'] for r in res):.4f} | "
                     f"{mean(r['selected_validation_mrr'] for r in res):.4f} | "
                     f"{mean(r['selected_round'] for r in res):.1f} | {mean(r['seconds'] for r in res):.0f} | {source} |")
    text = "\n".join(lines) + "\n"
    (R / "report.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    check() if p.parse_args().check else report()


if __name__ == "__main__":
    main()
