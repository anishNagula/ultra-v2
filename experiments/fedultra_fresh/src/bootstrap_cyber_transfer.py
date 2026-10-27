#!/usr/bin/env python3
"""Hierarchical paired bootstrap for the unbridged cyber-transfer experiment.

Requires the completed, full-evaluation output from cyber_transfer_runner.py.
The bootstrap resamples training seeds and, within each selected seed, a shared
set of paired target test triples. It never trains or reads target validation.
"""
import argparse
import json
import random
from pathlib import Path


def percentile(values, p):
    xs = sorted(values)
    if not xs:
        raise ValueError("No bootstrap samples")
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    if args.bootstrap_samples < 1000:
        raise ValueError("Use at least 1,000 bootstrap samples for the confirmatory interval")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("diagnostic_only"):
        raise ValueError("Refusing bootstrap inference from a diagnostic-only run")
    runs = payload.get("results")
    if not isinstance(runs, list) or len(runs) < 3:
        raise ValueError("Input must contain at least three completed seed runs in results[]")
    if any(run.get("diagnostic_only") for run in runs):
        raise ValueError("One or more seed runs are diagnostic-only")

    expected_triples = None
    deltas_by_seed = []
    for run in runs:
        queries = run.get("per_query_test_comparison")
        if not queries:
            raise ValueError("Missing per_query_test_comparison in a seed run")
        triples = [tuple(q["triple"]) for q in queries]
        if expected_triples is None:
            expected_triples = triples
        elif triples != expected_triples:
            raise ValueError("Target test triples/order differ across seeds")
        values = [float(q["delta_mrr"]) for q in queries]
        if not all(map(lambda x: x == x and abs(x) != float("inf"), values)):
            raise ValueError("Non-finite per-query MRR delta detected")
        deltas_by_seed.append(values)

    n_seeds = len(deltas_by_seed)
    n_queries = len(expected_triples)
    if n_queries < 2:
        raise ValueError("Need at least two held-out target test triples")
    seed_means = [sum(values) / len(values) for values in deltas_by_seed]
    observed = sum(seed_means) / n_seeds
    rng = random.Random(args.seed)
    boot = []
    for _ in range(args.bootstrap_samples):
        seed_indices = [rng.randrange(n_seeds) for _ in range(n_seeds)]
        # Paired query indices are shared across selected seeds.
        query_indices = [rng.randrange(n_queries) for _ in range(n_queries)]
        sampled = [deltas_by_seed[si][qi]
                   for si in seed_indices for qi in query_indices]
        boot.append(sum(sampled) / len(sampled))

    ci = [percentile(boot, 0.025), percentile(boot, 0.975)]
    seed_ci = payload.get("summary", {}).get("paired_seed_level_95pct_t_ci")
    seed_ci_positive = bool(seed_ci and seed_ci[0] > 0)
    bootstrap_ci_positive = ci[0] > 0
    mean_pass = observed >= 0.03
    diagnostic_only = bool(payload.get("diagnostic_only"))
    summary = {
        "source_results": str(args.input.resolve()),
        "bootstrap_samples": args.bootstrap_samples,
        "random_seed": args.seed,
        "num_training_seeds": n_seeds,
        "target_test_triples": n_queries,
        "per_seed_mean_delta_mrr": seed_means,
        "mean_delta_mrr": observed,
        "seed_level_95pct_t_ci": seed_ci,
        "hierarchical_paired_bootstrap_95pct_ci": ci,
        "acceptance_criteria": {
            "minimum_mean_delta_mrr": 0.03,
            "mean_gain_meets_threshold": mean_pass,
            "seed_level_ci_entirely_above_zero": seed_ci_positive,
            "hierarchical_bootstrap_ci_entirely_above_zero": bootstrap_ci_positive,
            "not_diagnostic_only": not diagnostic_only,
            "meets_full_success_criterion": bool(
                mean_pass and seed_ci_positive and bootstrap_ci_positive
                and n_seeds >= 3 and not diagnostic_only
            ),
        },
        "interpretation_limit": (
            "The interval resamples training seeds and paired triples from the fixed target test sample; "
            "it does not quantify uncertainty across alternate graph snapshots, sampling rules, or domains."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
