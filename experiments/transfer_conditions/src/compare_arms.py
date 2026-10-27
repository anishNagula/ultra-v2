#!/usr/bin/env python3
"""Pair a local and a FedAvg run of conditions_runner.py by seed (delta = FedAvg - local).

Writes the input format of fedultra_fresh/src/bootstrap_cyber_transfer.py, so the
same hierarchical paired bootstrap and +0.03 acceptance rule are applied.
"""
import argparse
import json
from pathlib import Path

from common import FRESH_SRC  # noqa: F401  (puts fedultra_fresh/src on sys.path)
import transfer_runner as transfer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local", type=Path, required=True)
    p.add_argument("--fedavg", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    local, fed = (json.loads(path.read_text()) for path in (args.local, args.fedavg))
    if (local["arm"], fed["arm"]) != ("local", "fedavg"):
        raise ValueError("expected one local and one fedavg output")
    for key in ("target", "target_sha256", "config", "protocol"):
        if local[key] != fed[key]:
            raise ValueError(f"local and fedavg runs differ in {key}")
    if local["config"]["skip_test"] or local["diagnostic_only"] or fed["diagnostic_only"]:
        raise ValueError("refusing to compare validation-only or diagnostic runs")
    by_seed = {r["seed"]: r for r in local["results"]}
    if set(by_seed) != {r["seed"] for r in fed["results"]}:
        raise ValueError("local and fedavg runs cover different seeds")

    results = []
    for f in sorted(fed["results"], key=lambda r: r["seed"]):
        l = by_seed[f["seed"]]
        if [q["triple"] for q in l["per_query_test"]] != [q["triple"] for q in f["per_query_test"]]:
            raise AssertionError("test queries are not aligned")
        results.append({
            "seed": f["seed"],
            "local": {"selected_round": l["selected_round"], "validation_mrr": l["selected_validation_mrr"], "test": l["test"]},
            "fedavg": {"selected_round": f["selected_round"], "validation_mrr": f["selected_validation_mrr"], "test": f["test"]},
            "transfer_delta_mrr": f["test"]["mrr"] - l["test"]["mrr"],
            "delta_by_relation": {rel: f["test_by_relation"][rel]["mrr"] - l["test_by_relation"][rel]["mrr"]
                                  for rel in sorted(l["test_by_relation"])},
            "per_query_test_comparison": [
                {"triple": a["triple"], "local_mrr": a["mrr"], "fedavg_mrr": b["mrr"], "delta_mrr": b["mrr"] - a["mrr"]}
                for a, b in zip(l["per_query_test"], f["per_query_test"])],
            "diagnostic_only": False,
        })
    summary = transfer.summarize_direction(results)
    summary["mean_local_test_mrr"] = sum(r["local"]["test"]["mrr"] for r in results) / len(results)
    summary["mean_fedavg_test_mrr"] = sum(r["fedavg"]["test"]["mrr"] for r in results) / len(results)
    rels = results[0]["delta_by_relation"]
    summary["mean_delta_by_relation"] = {rel: sum(r["delta_by_relation"][rel] for r in results) / len(results) for rel in rels}
    payload = {"protocol": local["protocol"], "target": local["target"], "source": fed["source"],
               "config": local["config"], "local_output": str(args.local), "fedavg_output": str(args.fedavg),
               "results": results, "summary": summary, "diagnostic_only": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
