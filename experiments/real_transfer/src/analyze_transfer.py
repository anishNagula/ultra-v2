#!/usr/bin/env python3
"""Pair a baseline and a candidate run by seed and apply the registered success criterion.

delta = candidate - baseline target test MRR, paired by seed and by test triple. Each side is scored as
  test       the target stage's validation-selected round (fine-tuned), or
  zero_shot  the weights the target stage started from, with no target training.
The statistics are the earlier studies': the seed-level 95% t-CI (transfer_runner.summarize_direction) and
the hierarchical paired bootstrap (fedultra_fresh/src/bootstrap_cyber_transfer.py: 20,000 samples,
seed 20261026, resampling seeds and test triples jointly). Full success needs mean delta >= +0.03 and
both CIs entirely above 0.

  python analyze_transfer.py --baseline ../results/confirmatory/local_f0.10.json \
      --candidate ../results/confirmatory/sparse_source_f0.10.json --output ../results/confirmatory/analysis/P1_f0.10.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from env import FRESH_SRC
import transfer_runner as transfer

SAMPLES, BOOT_SEED = 20_000, 20261026
SCORES = ("test", "zero_shot")
# The target stage every arm shares; source settings may differ between the two sides.
TARGET_STAGE = ("rounds", "local_steps", "batch_size", "num_negative", "eval_limit", "lr", "weight_decay", "threads")


def view(run, score):
    """(metrics, per-query records) of one seed's run under `score`."""
    if score == "test":
        return run["test"], run["per_query_test"]
    return run["zero_shot"]["test"], run["zero_shot"]["per_query_test"]


def pair(baseline, candidate, baseline_score="test", candidate_score="test"):
    for key in ("protocol", "target", "target_sha256", "checkpoint_sha256"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"baseline and candidate differ in {key}")
    if any(baseline["config"].get(k) != candidate["config"].get(k) for k in TARGET_STAGE):
        raise ValueError("baseline and candidate differ in the shared target-stage configuration")
    for run in (baseline, candidate):
        if run["config"]["skip_test"] or run["diagnostic_only"]:
            raise ValueError("refusing to compare validation-only or diagnostic runs")
    base_by_seed = {r["seed"]: r for r in baseline["results"]}
    if set(base_by_seed) != {r["seed"] for r in candidate["results"]}:
        raise ValueError("baseline and candidate cover different seeds")
    results = []
    for c in sorted(candidate["results"], key=lambda r: r["seed"]):
        b = base_by_seed[c["seed"]]
        (b_test, b_queries), (c_test, c_queries) = view(b, baseline_score), view(c, candidate_score)
        if [q["triple"] for q in b_queries] != [q["triple"] for q in c_queries]:
            raise AssertionError("test triples are not aligned")
        results.append({
            "seed": c["seed"],
            "baseline": {"score": baseline_score, "selected_round": b["selected_round"],
                         "validation_mrr": b["selected_validation_mrr"], "test": b_test},
            "candidate": {"score": candidate_score, "selected_round": c["selected_round"],
                          "validation_mrr": c["selected_validation_mrr"], "test": c_test,
                          "starting_validation_mrr": c["validation_mrr_history"][0],
                          "source_selected_round": c["starting_weights"].get("selected_round")},
            "transfer_delta_mrr": c_test["mrr"] - b_test["mrr"],
            "per_query_test_comparison": [
                {"triple": x["triple"], "baseline_mrr": x["mrr"], "candidate_mrr": y["mrr"], "delta_mrr": y["mrr"] - x["mrr"]}
                for x, y in zip(b_queries, c_queries)],
            "diagnostic_only": False,
        })
    summary = transfer.summarize_direction(results)
    summary["mean_baseline_test_mrr"] = sum(r["baseline"]["test"]["mrr"] for r in results) / len(results)
    summary["mean_candidate_test_mrr"] = sum(r["candidate"]["test"]["mrr"] for r in results) / len(results)
    return results, summary


def analyze(baseline_path, candidate_path, output, baseline_score="test", candidate_score="test"):
    baseline, candidate = (json.loads(Path(x).read_text()) for x in (baseline_path, candidate_path))
    results, summary = pair(baseline, candidate, baseline_score, candidate_score)
    label = f"{candidate['arm']}[{candidate_score}] - {baseline['arm']}[{baseline_score}]"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison = output.with_name(output.stem + "_comparison.json")
    boot_path = output.with_name(output.stem + "_bootstrap.json")
    comparison.write_text(json.dumps({"comparison": label, "baseline_output": str(baseline_path),
                                      "candidate_output": str(candidate_path), "results": results,
                                      "summary": summary, "diagnostic_only": False}, indent=1) + "\n",
                          encoding="utf-8")
    subprocess.run([sys.executable, str(FRESH_SRC / "bootstrap_cyber_transfer.py"), "--input", str(comparison),
                    "--output", str(boot_path), "--bootstrap-samples", str(SAMPLES), "--seed", str(BOOT_SEED)],
                   check=True, capture_output=True)
    boot = json.loads(boot_path.read_text())
    out = {
        "comparison": label, "target": baseline["target"], "seeds": [r["seed"] for r in results],
        "per_seed": [{"seed": r["seed"], "baseline_test_mrr": r["baseline"]["test"]["mrr"],
                      "candidate_test_mrr": r["candidate"]["test"]["mrr"], "delta": r["transfer_delta_mrr"],
                      "selected_round_baseline": r["baseline"]["selected_round"],
                      "selected_round_candidate": r["candidate"]["selected_round"],
                      "source_selected_round_candidate": r["candidate"]["source_selected_round"]} for r in results],
        "mean_baseline_test_mrr": summary["mean_baseline_test_mrr"],
        "mean_candidate_test_mrr": summary["mean_candidate_test_mrr"],
        "mean_delta": boot["mean_delta_mrr"],
        "seed_level_95pct_t_ci": boot["seed_level_95pct_t_ci"],
        "bootstrap_95pct_ci": boot["hierarchical_paired_bootstrap_95pct_ci"],
        "acceptance": boot["acceptance_criteria"],
        "PASS": boot["acceptance_criteria"]["meets_full_success_criterion"],
    }
    output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--baseline-score", choices=SCORES, default="test")
    p.add_argument("--candidate-score", choices=SCORES, default="test")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = analyze(args.baseline, args.candidate, args.output, args.baseline_score, args.candidate_score)
    t, b = out["seed_level_95pct_t_ci"], out["bootstrap_95pct_ci"]
    print(f"{out['comparison']}: baseline {out['mean_baseline_test_mrr']:.4f} | candidate {out['mean_candidate_test_mrr']:.4f} | "
          f"delta {out['mean_delta']:+.4f} | t-CI [{t[0]:+.4f}, {t[1]:+.4f}] | boot [{b[0]:+.4f}, {b[1]:+.4f}] | "
          f"{'PASS' if out['PASS'] else 'FAIL'}")


if __name__ == "__main__":
    main()
