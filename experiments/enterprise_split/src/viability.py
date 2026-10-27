#!/usr/bin/env python3
"""Phase 2 viability checks for the Enterprise-split candidates (DESIGN.md).

For each candidate and target fraction it reports:
- graph sizes;
- shared entities;
- exact, inverse-equivalent and entity-pair leakage (L1-L4);
- multi-hop relation compositions;
- source support for target validation queries;
- the transfer-opportunity proxy;
- evaluation coverage.

It then applies the hard constraints and the selection and go/no-go rules of DESIGN.md (Amendment 1).
Test facts are read only for the structural leakage and coverage checks; no model is trained and
no test fact is ranked.

  python viability.py --root ../data/candidates
"""
import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import torch

from env import BUNDLE, STUDY
import build_dataset as builder
from evaluation_protocol import read_triples
import explore_enrichment as rules
from build_split import ACTOR_TYPES, CANDIDATES, FRACTIONS

SCARCE = ("0.10", "0.25")
PRIMARY = "0.10"  # DESIGN.md Amendment 1


def read(folder, split):
    path = folder / f"{split}.txt"
    return read_triples(path) if path.is_file() else []


def pair(x):
    return frozenset((x[0], x[2]))


def leakage(source, target, kind):
    """L1-L4 counts (all must be zero) plus shared-entity counts by type."""
    tgt_all = target["train"] + target["hidden"] + target["valid"] + target["test"]
    src = set(source["train"])
    reverse = lambda facts: {(t, h) for h, _, t in facts}
    src_pairs = {pair(x) for x in src}
    tr_hidden = target["train"] + target["hidden"]
    ev = target["valid"] + target["test"]
    src_ents = {e for h, _, t in src for e in (h, t)}
    tgt_ents = {e for h, _, t in tgt_all for e in (h, t)}
    return {
        "L1_exact": len(src & set(tgt_all)),
        "L1_inverse_equivalent": len({(h, t) for h, _, t in src} & reverse(tgt_all)),
        "L2_exact": (len(set(target["valid"]) & set(target["test"])) + len(set(ev) & set(tr_hidden))),
        "L2_inverse_equivalent": len({(h, t) for h, _, t in ev} & reverse(tr_hidden)) + len(
            {(h, t) for h, _, t in target["valid"]} & reverse(target["test"])),
        "L3_entity_pair": sum(pair(x) in src_pairs for x in ev),
        "L4_target_actors_in_source": len({e for e in tgt_ents if kind[e] in ACTOR_TYPES} & src_ents),
        "L4_source_actors_in_target": len({e for e in src_ents if kind[e] in ACTOR_TYPES} & tgt_ents),
        "shared_entities_by_type": dict(Counter(kind[e] for e in src_ents & tgt_ents)),
        "source_entities": len(src_ents), "target_entities": len(tgt_ents),
    }


def semantic_overlap(source, target, nodes):
    """Overlap the L-checks cannot see: shared software and techniques, and plain-text actor mentions."""
    kind = {k: v["type"] for k, v in nodes.items()}
    actors = lambda facts: {h for h, _, _ in facts if kind[h] in ACTOR_TYPES}
    tgt_facts = target["train"] + target["hidden"] + target["valid"] + target["test"]
    src_act, tgt_act = actors(source["train"]), actors(tgt_facts)
    used = lambda facts, a: {t for h, _, t in facts if h in a}
    su, tu = used(source["train"], src_act), used(tgt_facts, tgt_act)
    names = lambda a: {n for x in a for n in [nodes[x]["name"], *(nodes[x].get("aliases") or [])] if len(n) >= 4}
    mentions = []
    for side_a, side_b, label in ((src_act, tgt_act, "source describes target"), (tgt_act, src_act, "target describes source")):
        for b in sorted(side_b):
            if not names({b}):  # every name shorter than 4 characters: too ambiguous to search for
                continue
            pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(names({b}))) + r")\b")
            for a in sorted(side_a):
                if pattern.search(nodes[a].get("description", "")):
                    mentions.append(f"{label}: {nodes[a]['name']} mentions {nodes[b]['name']}")
    return {"software_used_by_both_clients_actors": len({e for e in su & tu if kind[e] in ("malware", "tool")}),
            "techniques_used_by_both_clients_actors": len({e for e in su & tu if kind[e] == "attack-pattern"}),
            "plain_text_actor_mentions_across_clients": mentions}


def compositions(stats):
    body_count, support = stats
    by_len = Counter(len(b) for b, c in body_count.items() if c > 0)
    uses = {b: s for (r, b), s in support.items() if r == "uses" and s >= rules.MIN_SUPPORT}
    total = sum(uses.values())
    entropy = -sum(s / total * math.log(s / total) for s in uses.values()) if total else 0.0
    return {"relation_paths_with_instances": {f"length_{k}": by_len[k] for k in (1, 2, 3)},
            "uses_rules_support_ge_2": {f"length_{k}": sum(len(b) == k for b in uses) for k in (1, 2, 3)},
            "uses_rule_support_entropy_nats": entropy, "uses_rule_effective_number": math.exp(entropy)}


def query_support(dom, src_support):
    """Share of validation facts connected by a k <= 3 training path, and by one that is a source rule."""
    q = [(dom.e2i[h], r, dom.e2i[t]) for h, r, t in dom.valid]
    hs, ts = torch.tensor([x[0] for x in q]), torch.tensor([x[2] for x in q])
    connected = torch.zeros(len(q), dtype=torch.bool)
    supported = torch.zeros(len(q), dtype=torch.bool)
    for body, B in dom.bodies():
        hit = B[hs, ts]
        if hit.any():
            connected |= hit
            ok = torch.tensor([src_support.get((r, body), 0) >= rules.MIN_SUPPORT for _, r, _ in q])
            supported |= hit & ok
    return {"validation_connected_by_target_path": float(connected.float().mean()),
            "validation_connected_by_source_supported_rule": float(supported.float().mean())}


def coverage(target):
    deg = Counter(e for h, _, t in target["train"] for e in (h, t))
    share = lambda facts: sum(deg[h] > 0 and deg[t] > 0 for h, _, t in facts) / len(facts)
    return {"valid": share(target["valid"]), "test": share(target["test"])}


def relation_counts(facts):
    return dict(sorted(Counter(r for _, r, _ in facts).items()))


def decide(report):
    """DESIGN.md Amendment 1: choose by availability on the full target graph; primary f = 0.10."""
    for o in report["candidates"].values():
        m = o["fractions"]["1.00"]["validation_rule_mrr"]
        o["availability_source_over_own_at_f1.00"] = m["source"] / m["own"]
    passing = sorted(c for c, o in report["candidates"].items() if o["passes_hard_constraints"])
    decision = {"rule": "DESIGN.md Amendment 1", "primary_fraction": PRIMARY, "selected_candidate": None}
    if passing:
        cand = max(passing, key=lambda c: (report["candidates"][c]["availability_source_over_own_at_f1.00"], c == "C-rand"))
        a = report["candidates"][cand]["availability_source_over_own_at_f1.00"]
        decision.update(selected_candidate=cand, availability=a, go_i_hard_constraints=True,
                        go_ii_availability_ge_0_9=a >= 0.9,
                        go_iii_ultra_need="validation-only smoke test: local best valid MRR at f=0.10 <= f=1.00 - 0.03",
                        go_iv_automated_tests="test_pipeline.py on the written files")
    return decision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help="Folder holding one subfolder per candidate")
    p.add_argument("--candidates", nargs="+", default=list(CANDIDATES))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path, default=STUDY / "results/viability.json")
    p.add_argument("--decide-only", action="store_true", help="Re-apply the decision rules to an existing --output")
    args = p.parse_args()
    if args.decide_only:
        report = json.loads(args.output.read_text())
        report["decision"] = decide(report)
        args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(report["decision"], indent=1))
        return
    torch.set_num_threads(args.threads)
    nodes, _, _ = builder.extract(BUNDLE)
    kind = {k: v["type"] for k, v in nodes.items()}
    report = {"design": "DESIGN.md (hash in results/design_hash.txt)", "candidates": {}}
    for cand in args.candidates:
        t0 = time.time()
        root = args.root.expanduser().resolve() / cand
        source = {"train": read(root / "source", "train")}
        src_dom = rules.Domain("source", source["train"], [])
        src_stats = rules.rule_stats(src_dom)
        out = {"source": {"triples": len(source["train"]), "entities": src_dom.n,
                          "relations": relation_counts(source["train"]), **compositions(src_stats)},
               "fractions": {}}
        for f in FRACTIONS:
            folder = root / f"target_f{f}"
            target = {s: read(folder, s) for s in ("train", "valid", "test", "hidden")}
            dom = rules.Domain(f"target_f{f}", target["train"], target["valid"], target["hidden"])
            own = rules.rule_stats(dom)
            tabs = rules.tables(dom, own, src_stats)
            mrr = rules.evaluate(dom, tabs)
            cell = {"triples": {s: len(v) for s, v in target.items()}, "train_relations": relation_counts(target["train"]),
                    "leakage": leakage(source, target, kind), "evaluation_coverage": coverage(target),
                    **compositions(own), **query_support(dom, src_stats[1]),
                    "validation_rule_mrr": {"own": mrr["own"], "source": mrr["source"], "pooled": mrr["pooled"]},
                    "proxy_pooled_minus_own": mrr["pooled"] - mrr["own"]}
            if f == "1.00":
                cell["semantic_overlap"] = semantic_overlap(source, target, nodes)
            out["fractions"][f] = cell
            print(f"{cand} f={f}: train {len(target['train'])} | own {mrr['own']:.4f} source {mrr['source']:.4f} "
                  f"pooled {mrr['pooled']:.4f} proxy {cell['proxy_pooled_minus_own']:+.4f} | "
                  f"coverage valid {cell['evaluation_coverage']['valid']:.3f} test {cell['evaluation_coverage']['test']:.3f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        fr = out["fractions"]
        leak = [v for c in fr.values() for k, v in c["leakage"].items() if k[:2] in ("L1", "L2", "L3", "L4")]
        src_rel = set(out["source"]["relations"])
        out["hard_constraints"] = {
            "V1_no_leakage": all(v == 0 for v in leak),
            "V2_relation_coverage": all(set(c["train_relations"]) == src_rel for c in fr.values()),
            "V3_evaluation_size": min(fr["1.00"]["triples"]["valid"], fr["1.00"]["triples"]["test"]) >= 250,
            "V4_evaluation_coverage": all(min(c["evaluation_coverage"].values()) >= 0.80 for c in fr.values()),
        }
        out["passes_hard_constraints"] = all(out["hard_constraints"].values())
        out["mean_scarce_proxy"] = sum(fr[f]["proxy_pooled_minus_own"] for f in SCARCE) / len(SCARCE)
        report["candidates"][cand] = out
    report["decision"] = decide(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({c: {"hard_constraints": o["hard_constraints"], "mean_scarce_proxy": o["mean_scarce_proxy"]}
                      for c, o in report["candidates"].items()}, indent=1))
    print(json.dumps(report["decision"], indent=1))


if __name__ == "__main__":
    main()
