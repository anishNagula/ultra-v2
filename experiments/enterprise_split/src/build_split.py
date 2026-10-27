#!/usr/bin/env python3
"""Build the Enterprise-split clients (DESIGN.md, partition algorithm v1).

Two clients from one real domain:
- Each threat-actor unit goes wholly to the source or the target. A unit is the groups and
  campaigns linked by attribution, description cross-links or shared aliases.
- Software and technique reference facts go to the client whose actors use them most, counting
  training facts only.
- Every fact lands in exactly one client, and none is invented.

  python build_split.py --candidates C-rand C-time --output-root ../data/candidates
  python build_split.py --candidates C-rand --output-root ../data/v1
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from env import BUNDLE, ROOT
import build_dataset as builder
from common import sha256, write_triples

ALGORITHM = "enterprise-split partition v1 (DESIGN.md)"
SEEDS = {"units": 20261201, "ownership": 20261202, "eval": 20261203, "scarcity": 20261204}
FRACTIONS = ("0.10", "0.25", "1.00")
CANDIDATES = ("C-rand", "C-time")
ACTOR_TYPES = {"intrusion-set", "campaign"}
SOFTWARE_TYPES = {"malware", "tool"}
LINK = re.compile(r"attack\.mitre\.org/(?:groups|campaigns)/([GC]\d{4})\b")


def coin(seed, key):
    """Fair coin keyed by an entity, independent of iteration order (str seeds hash with sha512)."""
    return random.Random(f"{seed}:{key}").random() < 0.5


def actor_units(nodes, triples):
    """Connected components of actors linked by attribution, description cross-links or a shared alias."""
    kind = {k: v["type"] for k, v in nodes.items()}
    actors = sorted({e for h, _, t in triples for e in (h, t) if kind[e] in ACTOR_TYPES})
    parent = {a: a for a in actors}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    links = defaultdict(set)
    for h, r, t in triples:
        if r == "attributed-to":
            links["attributed-to"].add(tuple(sorted((h, t))))
    by_attack_id = {ref["external_id"]: k for k, o in nodes.items() for ref in o.get("external_references", [])
                    if ref.get("source_name") == "mitre-attack" and "external_id" in ref}
    for a in actors:
        for xid in LINK.findall(nodes[a].get("description", "")):
            b = by_attack_id.get(xid)
            if b in parent and b != a:
                links["description"].add(tuple(sorted((a, b))))
    alias = defaultdict(set)
    for a in actors:
        for name in nodes[a].get("aliases") or []:
            alias[name.lower()].add(a)
    for members in alias.values():
        members = sorted(members)
        links["alias"].update((members[0], b) for b in members[1:])
    for pairs in links.values():
        for a, b in pairs:
            parent[find(a)] = find(b)
    components = defaultdict(list)
    for a in actors:
        components[find(a)].append(a)
    return sorted(sorted(m) for m in components.values()), {k: len(v) for k, v in sorted(links.items())}


def assign_units(units, sizes, nodes, candidate):
    """Indices of the units that go to the target."""
    target = set()
    if candidate == "C-rand":
        order = sorted(range(len(units)), key=lambda i: (-sizes[i], units[i][0]))
        rng = random.Random(SEEDS["units"])
        for k in range(0, len(order), 2):
            pair = order[k:k + 2]
            if len(pair) == 2:
                target.add(pair[0] if rng.random() < 0.5 else pair[1])
            elif rng.random() < 0.5:
                target.add(pair[0])
    elif candidate == "C-time":
        created = lambda i: min((nodes[a]["created"][:19], nodes[a]["created"]) for a in units[i])
        order = sorted(range(len(units)), key=lambda i: (created(i), units[i][0]))
        total, held = sum(sizes), 0
        for i in order:
            if held >= 0.5 * total:
                target.add(i)
            else:
                held += sizes[i]
    else:
        raise ValueError(candidate)
    return target


def evaluation_split(actor_facts):
    """10% validation and 10% test from the target's actor `uses` facts; endpoints stay covered by training."""
    uses = sorted(x for x in actor_facts if x[1] == "uses")
    random.Random(SEEDS["eval"]).shuffle(uses)
    n = round(0.10 * len(uses))
    held = {x: "test" for x in uses[:n]} | {x: "valid" for x in uses[n:2 * n]}
    train = uses[2 * n:] + sorted(x for x in actor_facts if x[1] != "uses")
    covered = {e for h, _, t in train for e in (h, t)}
    moved = 0
    for x in sorted(held):
        if x[0] not in covered or x[2] not in covered:
            del held[x]
            train.append(x)
            covered.update((x[0], x[2]))
            moved += 1
    return (sorted(train), sorted(x for x, s in held.items() if s == "valid"),
            sorted(x for x, s in held.items() if s == "test"), moved)


def anchor(fact):
    """The entity whose owner receives a non-actor fact."""
    h, r, t = fact
    return t if r in ("mitigates", "detects") else h  # subtechnique-of: the sub-technique; software uses: the software


def scarce(train, f):
    """Per head entity, keep max(1, round(f * n_h)) facts; one shuffle per head makes fractions nested."""
    by_head = defaultdict(list)
    for x in sorted(train):
        by_head[x[0]].append(x)
    keep, hide = [], []
    for h, facts in sorted(by_head.items()):
        random.Random(f"{SEEDS['scarcity']}:{h}").shuffle(facts)
        k = max(1, round(float(f) * len(facts)))
        keep += facts[:k]
        hide += facts[k:]
    return sorted(keep), sorted(hide)


def build(nodes, triples, candidate):
    kind = {k: v["type"] for k, v in nodes.items()}
    units, links = actor_units(nodes, triples)
    unit_of = {a: i for i, members in enumerate(units) for a in members}
    actor_facts = defaultdict(list)
    for x in sorted(triples):
        if kind[x[0]] in ACTOR_TYPES:
            actor_facts[unit_of[x[0]]].append(x)
    sizes = [len(actor_facts[i]) for i in range(len(units))]
    to_target = assign_units(units, sizes, nodes, candidate)
    side = {a: ("target" if unit_of[a] in to_target else "source") for a in unit_of}
    src_actor = [x for i in range(len(units)) if i not in to_target for x in actor_facts[i]]
    tgt_actor = [x for i in sorted(to_target) for x in actor_facts[i]]
    assert all(side[x[2]] == side[x[0]] for x in src_actor + tgt_actor if x[1] == "attributed-to")

    tgt_train_actor, valid, test, moved = evaluation_split(tgt_actor)
    usage = {"source": Counter(t for _, _, t in src_actor), "target": Counter(t for _, _, t in tgt_train_actor)}
    other = sorted(x for x in triples if kind[x[0]] not in ACTOR_TYPES)
    assert all(kind[x[0]] in SOFTWARE_TYPES for x in other if x[1] == "uses")
    owner, ties = {}, Counter()
    for e in sorted({anchor(x) for x in other}):
        s, t = usage["source"][e], usage["target"][e]
        if s == t:
            ties[kind[e]] += 1
            owner[e] = "target" if coin(SEEDS["ownership"], e) else "source"
        else:
            owner[e] = "target" if t > s else "source"
    source_train = sorted(src_actor + [x for x in other if owner[anchor(x)] == "source"])
    target_train = sorted(tgt_train_actor + [x for x in other if owner[anchor(x)] == "target"])

    clients = {"source": {"train": source_train}}
    for f in FRACTIONS:
        keep, hide = scarce(target_train, f)
        clients[f"target_f{f}"] = {"train": keep, "valid": valid, "test": test, "hidden": hide}
    info = {
        "units": len(units), "association_links": links,
        "target_units": len(to_target), "source_units": len(units) - len(to_target),
        "actor_facts": {"source": len(src_actor), "target": len(tgt_actor)},
        "evaluation_facts_moved_to_train": moved,
        "owned_anchor_entities": {c: dict(Counter(kind[e] for e, o in owner.items() if o == c)) for c in ("source", "target")},
        "ownership_ties_broken_by_coin": dict(ties),
        "actors": {c: sorted(f"{nodes[a]['name']} ({a})" for a, s in side.items() if s == c) for c in ("source", "target")},
    }
    return clients, info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    nodes, triples, rejected = builder.extract(BUNDLE)
    for candidate in args.candidates:
        out = args.output_root.expanduser().resolve() / candidate
        clients, info = build(nodes, triples, candidate)
        files = {}
        for name, splits in clients.items():
            for split, facts in splits.items():
                if facts:  # read_triples rejects empty files; f = 1.00 has no hidden facts
                    path = out / name / f"{split}.txt"
                    write_triples(path, facts)
                    files[f"{name}/{split}.txt"] = {"triples": len(facts), "sha256": sha256(path)}
        manifest = {"algorithm": ALGORITHM, "candidate": candidate, "seeds": SEEDS, "fractions": FRACTIONS,
                    "bundle": {"path": str(BUNDLE.relative_to(ROOT)), "sha256": sha256(BUNDLE)},
                    "extracted": {"triples": len(triples), "rejected": dict(rejected)},
                    "files": files, **info}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        print(f"{candidate}: {info['units']} units ({info['target_units']} target), actor facts {info['actor_facts']}, "
              f"moved to train {info['evaluation_facts_moved_to_train']}, ties {info['ownership_ties_broken_by_coin']}")
        for k, v in files.items():
            print(f"   {k:<26} {v['triples']:>6}")


if __name__ == "__main__":
    main()
