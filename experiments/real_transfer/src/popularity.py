#!/usr/bin/env python3
"""Validation-only probe: how much entity-level knowledge the source holds about the target's answers.

ULTRA has no entity embeddings, so its weights can carry relational patterns but not facts about
particular entities. The clients share 344 techniques and 79 software as entities (not as facts).
This ranks each target validation query by popularity counted from

  target   the target's own training graph (reachability.py's ranker),
  source   the source's actor `uses` facts, and
  pooled   both.

Ranks are pessimistic and filtered with the registered filters. Target actors never occur in the
source, so source counts only inform tail queries; head queries fall back to target counts. This is a
diagnostic: a method using source counts would share per-entity source statistics, which is
outside this study's weights-only design.

  python popularity.py --output ../results/diagnostics/popularity.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from env import SOURCE, TARGET, is_actor_use
from common import load_bundle


def mrr(bundle, tail_pop, head_pop, split="valid"):
    """Mean over queries of the mean of the tail and head reciprocal ranks (ties ranked last)."""
    known = set(bundle["raw_triples"]["train"]) | set(bundle["hidden"]) | set(bundle["raw_triples"]["valid"])
    entities = list(bundle["entity_to_id"])
    total = 0.0
    for h, r, t in bundle["raw_triples"][split]:
        def rank(true, score_of, is_known):
            s = score_of(true)
            return 1 + sum(1 for e in entities if e != true and not is_known(e) and score_of(e) >= s)
        rt = rank(t, lambda e: tail_pop[(r, e)], lambda e: (h, r, e) in known)
        rh = rank(h, lambda e: head_pop[(r, e)], lambda e: (e, r, t) in known)
        total += (1 / rt + 1 / rh) / 2
    return total / len(bundle["raw_triples"][split])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = [x for x in load_bundle(SOURCE)["raw_triples"]["train"] if is_actor_use(x)]
    source_tails = Counter((r, t) for _, r, t in source)
    out = {"note": "validation only; per-entity popularity rankers, not methods", "source_actor_uses": len(source),
           "fractions": {}}
    for f, folder in TARGET.items():
        bundle = load_bundle(folder)
        train = bundle["raw_triples"]["train"]
        tails, heads = Counter((r, t) for _, r, t in train), Counter((r, h) for h, r, _ in train)
        answers = {t for _, _, t in bundle["raw_triples"]["valid"]}
        out["fractions"][f"f{f}"] = row = {
            "target": mrr(bundle, tails, heads), "source": mrr(bundle, source_tails, heads),
            "pooled": mrr(bundle, tails + source_tails, heads),
            "validation_answers_seen_in_target_train": sum(tails[("uses", t)] > 0 for t in answers) / len(answers),
            "validation_answers_seen_in_source": sum(source_tails[("uses", t)] > 0 for t in answers) / len(answers)}
        print(f"f{f}: " + "  ".join(f"{k} {v:.4f}" for k, v in row.items()), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
