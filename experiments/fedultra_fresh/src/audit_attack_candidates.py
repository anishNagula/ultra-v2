#!/usr/bin/env python3
"""Audit Mobile ATT&CK and Enterprise platform subgraph candidates with the project's extraction rules.

Run from the ULTRA repo:
  python experiments/fedultra_fresh/src/audit_attack_candidates.py \
    --enterprise-json datasets/cybersecurity/mitre/enterprise/v19.2/enterprise-attack-19.2.json \
    --mobile-json datasets/cybersecurity/mitre/mobile/v19.2/mobile-attack-19.2.json

The script downloads Mobile v19.2 from MITRE's official attack-stix-data repository
if --mobile-json does not exist. It does not train a model or alter existing splits.
"""
import argparse
import hashlib
import json
import random
import urllib.request
from collections import Counter, defaultdict, deque
from pathlib import Path

MOBILE_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
    "master/mobile-attack/mobile-attack-19.2.json"
)
NODE_TYPES = {
    "attack-pattern", "malware", "tool", "intrusion-set",
    "campaign", "course-of-action", "x-mitre-asset",
    "x-mitre-data-source", "x-mitre-data-component",
    "x-mitre-analytic", "x-mitre-detection-strategy", "x-mitre-tactic",
}
RELATION_TYPES = {
    "uses", "mitigates", "detects", "subtechnique-of",
    "attributed-to", "targets",
}
SEED = 20261023


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def active(obj):
    return not obj.get("revoked", False) and not obj.get("x_mitre_deprecated", False)


def extract(path):
    bundle = json.loads(path.read_text(encoding="utf-8"))
    objects = bundle["objects"]
    nodes = {
        o["id"]: o for o in objects
        if o.get("type") in NODE_TYPES and isinstance(o.get("id"), str) and active(o)
    }
    triples, rejected = set(), Counter()
    for o in objects:
        if o.get("type") != "relationship":
            continue
        if not active(o):
            rejected["inactive_relationship"] += 1
            continue
        rel = o.get("relationship_type")
        if rel not in RELATION_TYPES:
            rejected["unsupported_relation"] += 1
            continue
        h, t = o.get("source_ref"), o.get("target_ref")
        if h not in nodes or t not in nodes:
            rejected["excluded_endpoint"] += 1
            continue
        triples.add((h, rel, t))
    return nodes, triples, rejected


def split_counts(triples, seed=SEED):
    rng = random.Random(seed)
    by_rel = defaultdict(list)
    for triple in sorted(triples):
        by_rel[triple[1]].append(triple)
    train, valid, test = set(), set(), set()
    for rel, items0 in sorted(by_rel.items()):
        items = list(items0)
        rng.shuffle(items)
        n = len(items)
        nt = max(1, round(n * .10)) if n >= 3 else 0
        nv = max(1, round(n * .10)) if n >= 3 else 0
        while nt + nv >= n:
            if nv >= nt and nv:
                nv -= 1
            elif nt:
                nt -= 1
            else:
                break
        test.update(items[:nt])
        valid.update(items[nt:nt + nv])
        train.update(items[nt + nv:])
    all_entities = {e for h, _, t in triples for e in (h, t)}
    train_entities = {e for h, _, t in train for e in (h, t)}
    for e in sorted(all_entities - train_entities):
        candidates = sorted(x for x in valid | test if e in (x[0], x[2]))
        if candidates:
            chosen = candidates[0]
            valid.discard(chosen)
            test.discard(chosen)
            train.add(chosen)
    return {"train": len(train), "valid": len(valid), "test": len(test)}


def components(triples):
    adj = defaultdict(set)
    for h, _, t in triples:
        adj[h].add(t)
        adj[t].add(h)
    unseen = set(adj)
    sizes = []
    while unseen:
        root = unseen.pop()
        q = [root]
        size = 0
        while q:
            u = q.pop()
            size += 1
            for v in adj[u]:
                if v in unseen:
                    unseen.remove(v)
                    q.append(v)
        sizes.append(size)
    sizes.sort(reverse=True)
    n = len(adj)
    return {
        "connected_components": len(sizes),
        "largest_component_nodes": sizes[0] if sizes else 0,
        "largest_component_fraction": (sizes[0] / n) if n else 0.0,
        "entities_with_degree_1": sum(1 for x in adj if len(adj[x]) == 1),
        "entities_with_degree_0_in_extracted_graph": 0,
    }


def graph_summary(label, nodes, triples, rejected=None):
    entities = {e for h, _, t in triples for e in (h, t)}
    relation_counts = Counter(r for _, r, _ in triples)
    type_counts = Counter(nodes[e].get("type", "unknown") for e in entities)
    degree = Counter()
    for h, _, t in triples:
        degree[h] += 1
        degree[t] += 1
    return {
        "name": label,
        "active_selected_node_inventory": len(nodes),
        "triples": len(triples),
        "entities_in_triples": len(entities),
        "relations": dict(sorted(relation_counts.items())),
        "entity_types_in_triples": dict(sorted(type_counts.items())),
        "split_counts_estimate": split_counts(triples),
        "degree": {
            "mean": (sum(degree.values()) / len(degree)) if degree else 0.0,
            "median": sorted(degree.values())[len(degree)//2] if degree else 0,
            "degree_1_entity_count": sum(1 for d in degree.values() if d == 1),
            "degree_1_entity_fraction": (
                sum(1 for d in degree.values() if d == 1) / len(degree)
            ) if degree else 0.0,
        },
        "topology": components(triples),
        "rejected": dict(rejected or {}),
    }


def platform_candidates(nodes, triples, top_n=15):
    """Induced one-hop platform ego-subgraphs around platform-tagged attack-patterns.

    Each candidate contains techniques tagged for that platform plus every entity
    directly adjacent to one of those techniques. It includes extracted triples
    whose endpoints both lie in that candidate. These candidates overlap by design;
    overlap is measured, not hidden.
    """
    tech_by_platform = defaultdict(set)
    for sid, obj in nodes.items():
        if obj.get("type") != "attack-pattern":
            continue
        for p in obj.get("x_mitre_platforms", []) or []:
            tech_by_platform[p].add(sid)
    neighbors = defaultdict(set)
    for h, _, t in triples:
        neighbors[h].add(t)
        neighbors[t].add(h)

    candidates = []
    for platform, techs in tech_by_platform.items():
        selected = set(techs)
        for tid in techs:
            selected.update(neighbors[tid])
        sub = {tr for tr in triples if tr[0] in selected and tr[2] in selected}
        entities = {e for h, _, t in sub for e in (h, t)}
        candidates.append({
            "platform": platform,
            "tagged_techniques": len(techs),
            "selected_entities_before_induction": len(selected),
            "triples": len(sub),
            "entities_in_triples": len(entities),
            "relation_counts": dict(sorted(Counter(r for _, r, _ in sub).items())),
            "split_counts_estimate": split_counts(sub),
            "overlap_entity_ids_with_enterprise_other_platforms": None,
            "_entities": entities,
            "_triples": sub,
        })
    candidates.sort(key=lambda x: x["triples"], reverse=True)
    # Preserve entity sets separately because private fields are removed below.
    entity_sets = [candidate["_entities"] for candidate in candidates]
    for i, c in enumerate(candidates):
        other_entities = set()
        for j, other in enumerate(candidates):
            if i != j:
                other_entities.update(entity_sets[j])
        c["entities_shared_with_any_other_platform_candidate"] = len(c["_entities"] & other_entities)
        c["shared_entity_fraction"] = (
            len(c["_entities"] & other_entities) / len(c["_entities"])
            if c["_entities"] else 0.0
        )
        c.pop("_entities")
        c.pop("_triples")
    return candidates[:top_n]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--enterprise-json", type=Path, required=True)
    p.add_argument("--mobile-json", type=Path, default=Path(
        "datasets/cybersecurity/mitre/mobile/v19.2/mobile-attack-19.2.json"
    ))
    p.add_argument("--output", type=Path, default=Path(
        "experiments/fedultra_fresh/results/attack_candidate_audit.json"
    ))
    args = p.parse_args()
    enterprise_path = args.enterprise_json.expanduser().resolve()
    mobile_path = args.mobile_json.expanduser().resolve()
    if not enterprise_path.is_file():
        raise FileNotFoundError(f"Enterprise JSON not found: {enterprise_path}")
    if not mobile_path.exists():
        mobile_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading official Mobile ATT&CK v19.2 to {mobile_path}")
        req = urllib.request.Request(MOBILE_URL, headers={"User-Agent": "ULTRA-ATTACK-audit/1.0"})
        with urllib.request.urlopen(req, timeout=120) as response, mobile_path.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

    en_nodes, en_triples, en_rejected = extract(enterprise_path)
    mo_nodes, mo_triples, mo_rejected = extract(mobile_path)
    shared_triples = en_triples & mo_triples
    mo_unique = mo_triples - shared_triples
    en_entities = {e for h, _, t in en_triples for e in (h, t)}
    mo_entities = {e for h, _, t in mo_triples for e in (h, t)}
    en_summary = graph_summary("Enterprise v19.2", en_nodes, en_triples, en_rejected)
    mobile_summary = graph_summary("Mobile v19.2 (raw extracted)", mo_nodes, mo_triples, mo_rejected)
    mobile_unique_summary = graph_summary("Mobile v19.2 after removing exact Enterprise triple overlap",
                                          mo_nodes, mo_unique, mo_rejected)
    output = {
        "source_files": {
            "enterprise": {"path": str(enterprise_path), "sha256": sha256(enterprise_path)},
            "mobile": {"path": str(mobile_path), "sha256": sha256(mobile_path), "official_url": MOBILE_URL},
        },
        "extraction_rules": {
            "node_types": sorted(NODE_TYPES),
            "relation_types": sorted(RELATION_TYPES),
            "active_rule": "not revoked and not x_mitre_deprecated",
            "split_rule": "relation-aware 80/10/10 target with all entities/relations kept in train",
        },
        "enterprise": en_summary,
        "mobile": mobile_summary,
        "mobile_after_overlap_removal": mobile_unique_summary,
        "cross_domain_overlap": {
            "shared_entity_ids": len(en_entities & mo_entities),
            "shared_exact_triples": len(shared_triples),
            "shared_entity_fraction_enterprise": len(en_entities & mo_entities) / len(en_entities) if en_entities else 0.0,
            "shared_entity_fraction_mobile": len(en_entities & mo_entities) / len(mo_entities) if mo_entities else 0.0,
        },
        "enterprise_platform_candidates": platform_candidates(en_nodes, en_triples),
        "decision_guidance": [
            "Prefer Mobile if it is sufficiently large after the exact same extraction and has a useful relation vocabulary; it is a distinct ATT&CK domain.",
            "Treat platform ego-subgraphs as a controlled same-domain experiment, not independent domains; quantify their overlap before use.",
            "Do not select a candidate based on ULTRA MRR. Select based on predeclared structural/data criteria, then run the model.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "enterprise": en_summary,
        "mobile": mobile_summary,
        "mobile_after_overlap_removal": mobile_unique_summary,
        "cross_domain_overlap": output["cross_domain_overlap"],
        "enterprise_platform_candidates": output["enterprise_platform_candidates"],
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()

