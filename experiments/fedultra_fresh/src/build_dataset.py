#!/usr/bin/env python3
"""Build fresh MITRE Enterprise/ICS splits with configurable source paths.

Run from any working directory:
  python experiments/fedultra_fresh/src/build_dataset.py
Optional overrides:
  --enterprise-json /path/to/enterprise-attack.json
  --ics-json /path/to/ics-attack.json
  --output-root experiments/fedultra_fresh/data_processed
  --manifest experiments/fedultra_fresh/data_manifest.json
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "experiments/fedultra_fresh/data_processed"
DEFAULT_MANIFEST = ROOT / "experiments/fedultra_fresh/data_manifest.json"
SPLIT_SEED = 20261023

DEFAULT_SOURCES = {
    "enterprise": Path.home() / (
        "code/ULTRA/datasets/cybersecurity/mitre/"
        "enterprise/v19.2/enterprise-attack-19.2.json"
    ),
    "ics": Path.home() / (
        "code/ULTRA/datasets/cybersecurity/mitre/"
        "ics/v19.2/ics-attack-19.2.json"
    ),
}

NODE_TYPES = {
    "attack-pattern", "malware", "tool", "intrusion-set",
    "campaign", "course-of-action", "x-mitre-asset",
    "x-mitre-data-source", "x-mitre-data-component",
    "x-mitre-analytic", "x-mitre-detection-strategy",
    "x-mitre-tactic",
}
RELATION_TYPES = {
    "uses", "mitigates", "detects", "subtechnique-of",
    "attributed-to", "targets",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active(obj):
    return not obj.get("revoked", False) and not obj.get("x_mitre_deprecated", False)


def extract(path):
    bundle = json.loads(path.read_text(encoding="utf-8"))
    nodes = {
        obj["id"]: obj for obj in bundle["objects"]
        if obj.get("type") in NODE_TYPES
        and isinstance(obj.get("id"), str)
        and active(obj)
    }
    triples, rejected = set(), Counter()
    for obj in bundle["objects"]:
        if obj.get("type") != "relationship":
            continue
        if not active(obj):
            rejected["inactive_relationship"] += 1
            continue
        relation = obj.get("relationship_type")
        if relation not in RELATION_TYPES:
            rejected["unsupported_relation"] += 1
            continue
        head, tail = obj.get("source_ref"), obj.get("target_ref")
        if head not in nodes or tail not in nodes:
            rejected["excluded_endpoint"] += 1
            continue
        triples.add((head, relation, tail))
    return nodes, triples, rejected


def split_triples(triples, seed):
    """Relation-aware split; keep every entity and relation represented in train."""
    rng = random.Random(seed)
    by_relation = {}
    for triple in sorted(triples):
        by_relation.setdefault(triple[1], []).append(triple)
    train, valid, test = set(), set(), set()
    for relation, items in sorted(by_relation.items()):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * 0.10)) if n >= 3 else 0
        n_valid = max(1, round(n * 0.10)) if n >= 3 else 0
        while n_test + n_valid >= n:
            if n_valid >= n_test and n_valid:
                n_valid -= 1
            elif n_test:
                n_test -= 1
            else:
                break
        test.update(items[:n_test])
        valid.update(items[n_test:n_test + n_valid])
        train.update(items[n_test + n_valid:])

    all_entities = {e for h, _, t in triples for e in (h, t)}
    train_entities = {e for h, _, t in train for e in (h, t)}
    for entity in sorted(all_entities - train_entities):
        candidates = sorted(
            triple for triple in valid | test if entity in (triple[0], triple[2])
        )
        if candidates:
            chosen = candidates[0]
            valid.discard(chosen)
            test.discard(chosen)
            train.add(chosen)

    assert not (train & valid or train & test or valid & test)
    train_entities = {e for h, _, t in train for e in (h, t)}
    assert all_entities <= train_entities
    assert {r for _, r, _ in triples} <= {r for _, r, _ in train}
    return {"train": train, "valid": valid, "test": test}


def write_client(directory, splits):
    directory.mkdir(parents=True, exist_ok=True)
    for split_name, triples in splits.items():
        with (directory / f"{split_name}.txt").open("w", encoding="utf-8") as handle:
            for head, relation, tail in sorted(triples):
                handle.write(f"{head}\t{relation}\t{tail}\n")
    return {name: len(items) for name, items in splits.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enterprise-json", type=Path, default=DEFAULT_SOURCES["enterprise"])
    parser.add_argument("--ics-json", type=Path, default=DEFAULT_SOURCES["ics"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    sources = {"enterprise": args.enterprise_json.expanduser(),
               "ics": args.ics_json.expanduser()}
    output_root = args.output_root.expanduser()
    manifest_path = args.manifest.expanduser()
    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name} MITRE source: {path}\n"
                "Pass the correct location with --enterprise-json or --ics-json."
            )

    extracted, manifest = {}, {
        "split_seed": SPLIT_SEED, "sources": {}, "clients": {},
    }
    for name, path in sources.items():
        nodes, triples, rejected = extract(path)
        extracted[name] = (nodes, triples)
        manifest["sources"][name] = {
            "path": str(path.resolve()), "sha256": sha256(path),
            "active_selected_entities": len(nodes),
            "unique_extracted_triples": len(triples),
            "rejected": dict(rejected),
        }

    enterprise_triples = extracted["enterprise"][1]
    ics_nodes, ics_triples = extracted["ics"]
    shared_triples = enterprise_triples & ics_triples
    ics_triples = ics_triples - shared_triples
    final_triples = {"enterprise": enterprise_triples, "ics": ics_triples}

    print(f"Split seed: {SPLIT_SEED}")
    print(f"Exact shared triples removed from ICS: {len(shared_triples)}")
    for name, triples in final_triples.items():
        splits = split_triples(triples, SPLIT_SEED)
        counts = write_client(output_root / name, splits)
        entities = {e for h, _, t in triples for e in (h, t)}
        relations = {r for _, r, _ in triples}
        manifest["clients"][name] = {
            "triples": len(triples), "entities": len(entities),
            "relations": sorted(relations), "split_counts": counts,
        }
        print(f"\n{name.upper()}\nEntities: {len(entities)}")
        print(f"Relations: {sorted(relations)}\nTriples: {len(triples)}\nSplits: {counts}")

    manifest["cross_domain"] = {
        "shared_entity_ids": len(set(extracted["enterprise"][0]) & set(ics_nodes)),
        "shared_triples_removed_from_ics": len(shared_triples),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nManifest: {manifest_path.resolve()}")
    print(f"Dataset output: {output_root.resolve()}")


if __name__ == "__main__":
    main()
