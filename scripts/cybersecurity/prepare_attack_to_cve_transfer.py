#!/usr/bin/env python3
"""Prepare a frozen, entity-disjoint ULTRA transfer target from CAPEC/CWE/NVD.

This protocol intentionally does NOT create ATT&CK--CAPEC identity edges. The
source ATT&CK and target vulnerability graph have different entity namespaces;
no explicit bridge is required to test whether ULTRA's learned relational
reasoning transfers to a previously unseen graph.

To keep full-entity filtered ranking computationally tractable, the default
protocol deterministically samples at most 20,000 CVEs by SHA-256 of their IDs.
The sample is chosen before link splitting and without using model scores.
Only CVE--has_weakness-->CWE triples are prediction targets. Every other target
triple is retained as training-only message-passing context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

DEFAULT_TRIPLES = Path("data/processed/cyber_cti/triples.tsv")
DEFAULT_SOURCE_SPLITS = Path("experiments/fedultra_fresh/data_processed/enterprise")
DEFAULT_OUT = Path("data/processed/cyber_transfer")
SEED = 20261026
TARGET_RELATION = "has_weakness"
CONTEXT_RELATIONS = {"affects_product_configuration"}
DEFAULT_MAX_CVES = 20_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_triples(path: Path) -> set[tuple[str, str, str]]:
    triples: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            fields = line.rstrip("\r\n").split("\t")
            if line_no == 1 and fields == ["head", "relation", "tail"]:
                continue
            if len(fields) != 3 or not all(fields):
                raise ValueError(f"Expected 3 non-empty TSV fields at {path}:{line_no}")
            triple = tuple(fields)
            if triple in triples:
                raise ValueError(f"Duplicate triple at {path}:{line_no}: {triple}")
            triples.add(triple)
    if not triples:
        raise ValueError(f"No triples read from {path}")
    return triples


def read_source_entities(source_splits: Path) -> set[str]:
    entities: set[str] = set()
    for split in ("train.txt", "valid.txt", "test.txt"):
        path = source_splits / split
        if not path.is_file():
            raise FileNotFoundError(f"Missing source split: {path}")
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != 3 or not all(fields):
                    raise ValueError(f"Expected 3 non-empty TSV fields at {path}:{line_no}")
                entities.update((fields[0], fields[2]))
    return entities


def stable_cve_sample(cves: set[str], max_cves: int, seed: int) -> set[str]:
    if max_cves < 0:
        raise ValueError("--max-cves must be 0 (all CVEs) or a positive integer")
    if max_cves == 0 or len(cves) <= max_cves:
        return set(cves)
    ranked = sorted(cves, key=lambda c: hashlib.sha256(f"{seed}\0{c}".encode()).hexdigest())
    return set(ranked[:max_cves])


def select_target_graph(base_triples: set[tuple[str, str, str]], max_cves: int, seed: int):
    cves = {e for h, _, t in base_triples for e in (h, t) if e.startswith("CVE:")}
    selected = stable_cve_sample(cves, max_cves, seed)
    graph = {
        triple for triple in base_triples
        if all(not entity.startswith("CVE:") or entity in selected for entity in (triple[0], triple[2]))
    }
    if not graph:
        raise ValueError("CVE selection produced an empty target graph")
    if not any(r == TARGET_RELATION for _, r, _ in graph):
        raise ValueError(f"No {TARGET_RELATION} edges remain after CVE selection")
    return graph, cves, selected


def entity_set(triples):
    return {e for h, _, t in triples for e in (h, t)}


def split_target(triples: set[tuple[str, str, str]], seed: int,
                 valid_fraction: float, test_fraction: float):
    if not (0 < valid_fraction < 0.5 and 0 < test_fraction < 0.5
            and valid_fraction + test_fraction < 0.5):
        raise ValueError("valid/test fractions must be >0, <0.5, and sum to <0.5")

    target_edges = sorted(t for t in triples if t[1] == TARGET_RELATION)
    context_edges = {t for t in triples if t[1] != TARGET_RELATION}
    if len(target_edges) < 3:
        raise ValueError(f"Need at least 3 {TARGET_RELATION} triples; found {len(target_edges)}")

    context_entities = entity_set(context_edges)
    remaining_degree = Counter(e for h, _, t in target_edges for e in (h, t))
    shuffled = list(target_edges)
    random.Random(seed).shuffle(shuffled)
    requested_test = max(1, round(len(target_edges) * test_fraction))
    requested_valid = max(1, round(len(target_edges) * valid_fraction))
    test: set[tuple[str, str, str]] = set()
    valid: set[tuple[str, str, str]] = set()

    def can_holdout(triple):
        h, _, t = triple
        return all(e in context_entities or remaining_degree[e] > 1 for e in (h, t))

    for destination, requested in ((test, requested_test), (valid, requested_valid)):
        for triple in shuffled:
            if len(destination) >= requested:
                break
            if triple in test or triple in valid:
                continue
            if not can_holdout(triple):
                continue
            destination.add(triple)
            h, _, t = triple
            remaining_degree[h] -= 1
            remaining_degree[t] -= 1

    train = context_edges | (set(target_edges) - test - valid)
    train_entities = entity_set(train)
    # A final defensive filter; the greedy selection should already guarantee this.
    valid = {t for t in valid if t[0] in train_entities and t[2] in train_entities}
    test = {t for t in test if t[0] in train_entities and t[2] in train_entities}
    train = context_edges | (set(target_edges) - test - valid)
    train_entities = entity_set(train)
    all_entities = entity_set(triples)

    if not valid or not test:
        raise ValueError("Could not create non-empty validation and test splits")
    if train & valid or train & test or valid & test:
        raise AssertionError("Split overlap detected")
    if not all_entities <= train_entities:
        missing = sorted(all_entities - train_entities)[:10]
        raise AssertionError(f"Entities absent from training graph, e.g. {missing}")
    if any(r != TARGET_RELATION for _, r, _ in valid | test):
        raise AssertionError("Only has_weakness triples may be held out")
    if not all(h in train_entities and t in train_entities for h, _, t in valid | test):
        raise AssertionError("A held-out endpoint is absent from training")

    return {"train": train, "valid": valid, "test": test}, {
        "target_relation_total": len(target_edges),
        "requested_valid_count": requested_valid,
        "requested_test_count": requested_test,
        "actual_valid_count": len(valid),
        "actual_test_count": len(test),
        "train_has_weakness_count": sum(r == TARGET_RELATION for _, r, _ in train),
        "all_non_target_context_triples_kept_in_train": True,
        "all_held_out_edges_are_has_weakness": True,
        "all_valid_test_endpoints_present_in_train": True,
        "all_target_entities_present_in_train": True,
    }


def write_split(path: Path, triples: set[tuple[str, str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for h, r, t in sorted(triples):
            f.write(f"{h}\t{r}\t{t}\n")


def type_counts(entities: set[str]):
    prefixes = ("CVE:", "CPE:", "CAPEC:", "CWE:", "ATTACK:")
    counts = Counter()
    for entity in entities:
        matched = next((prefix[:-1] for prefix in prefixes if entity.startswith(prefix)), "OTHER")
        counts[matched] += 1
    return dict(sorted(counts.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--triples", type=Path, default=DEFAULT_TRIPLES)
    p.add_argument("--source-splits", type=Path, default=DEFAULT_SOURCE_SPLITS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=SEED,
                   help="Frozen seed for deterministic CVE sampling and link splitting")
    p.add_argument("--max-cves", type=int, default=DEFAULT_MAX_CVES,
                   help="Maximum CVEs in target graph; 20000 is the frozen default; 0 uses all")
    p.add_argument("--valid-fraction", type=float, default=0.10)
    p.add_argument("--test-fraction", type=float, default=0.10)
    args = p.parse_args()

    if not args.triples.is_file():
        raise FileNotFoundError(args.triples)
    base_triples = read_triples(args.triples)
    graph, all_cves, selected_cves = select_target_graph(base_triples, args.max_cves, args.seed)
    source_entities = read_source_entities(args.source_splits)
    target_entities = entity_set(graph)
    overlap = source_entities & target_entities
    if overlap:
        raise ValueError(
            "Expected an unbridged entity-disjoint source/target protocol, but found "
            f"{len(overlap)} shared IDs (examples: {sorted(overlap)[:10]})."
        )

    source_hashes = {name: sha256(args.source_splits / name)
                     for name in ("train.txt", "valid.txt", "test.txt")}
    splits, split_audit = split_target(graph, args.seed, args.valid_fraction, args.test_fraction)
    out_dir = args.out_dir
    target_dir = out_dir / "target"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        write_split(target_dir / f"{name}.txt", split)

    graph_path = out_dir / "target_graph.tsv"
    with graph_path.open("w", encoding="utf-8") as f:
        f.write("head\trelation\ttail\n")
        for h, r, t in sorted(graph):
            f.write(f"{h}\t{r}\t{t}\n")

    split_counts = {name: len(values) for name, values in splits.items()}
    relation_counts = {name: dict(sorted(Counter(r for _, r, _ in values).items()))
                       for name, values in splits.items()}
    train_entities = entity_set(splits["train"])
    audit = {
        "protocol": "pretrained ULTRA -> source-finetuned ULTRA on an unbridged, entity-disjoint CAPEC/CWE/CVE target KG",
        "seed": args.seed,
        "split_config": {
            "method": "seeded greedy edge holdout preserving every entity in train",
            "valid_fraction_requested": args.valid_fraction,
            "test_fraction_requested": args.test_fraction,
            "prediction_relation": TARGET_RELATION,
        },
        "cve_sampling": {
            "method": "sort CVE IDs by SHA-256(f'{seed}\\0{CVE_ID}') and take first max_cves",
            "seed": args.seed,
            "max_cves": args.max_cves,
            "available_cves_in_base_graph": len(all_cves),
            "selected_cves": len(selected_cves),
            "sampling_uses_model_or_test_metrics": False,
        },
        "source": {
            "split_dir": str(args.source_splits.resolve()),
            "entity_count_across_train_valid_test": len(source_entities),
            "split_sha256": source_hashes,
        },
        "inputs": {
            "base_triples_path": str(args.triples.resolve()),
            "base_triples_sha256": sha256(args.triples),
            "base_triples_count": len(base_triples),
        },
        "cross_domain": {
            "explicit_bridge_edges_added": 0,
            "source_target_entity_overlap_count": len(overlap),
            "source_target_entity_overlap_examples": [],
            "claim": "Unbridged cross-KG relational transfer; no identity or mapping links are asserted between source and target entities.",
        },
        "graph": {
            "target_triples": len(graph),
            "entity_count": len(target_entities),
            "entity_type_counts": type_counts(target_entities),
            "relation_counts": dict(sorted(Counter(r for _, r, _ in graph).items())),
            "has_weakness_edges_total": sum(r == TARGET_RELATION for _, r, _ in graph),
            "product_configuration_context_edges": sum(r == "affects_product_configuration" for _, r, _ in graph),
            "train_entity_coverage_fraction": len(train_entities) / len(target_entities) if target_entities else 0,
        },
        "split": {
            **split_audit,
            "counts": split_counts,
            "relation_counts": relation_counts,
            "train_entities": len(train_entities),
            "target_entities": len(target_entities),
        },
        "metrics_and_scope": {
            "primary_metric": "filtered target test MRR over head and tail corruption, all target entities as candidates",
            "comparison": "pretrained zero-shot ULTRA vs Enterprise-source-finetuned ULTRA on identical target queries",
            "acceptance_threshold_mean_delta_mrr": 0.03,
            "required_95pct_ci": "entirely above zero",
            "source_fine_tuning_uses_target_gradients": False,
            "target_message_passing_uses_target_train_triples_only": True,
            "target_validation_test_used_for_source_checkpoint_selection": False,
            "only_has_weakness_edges_are_held_out": True,
            "all_other_target_relations_are_training_context": True,
            "target_graph_is_cve_sampled_for_computational_feasibility": True,
            "interpretation_limit": "This tests transfer to a separate target KG without shared entity IDs or explicit semantic bridge edges; it does not directly test attack detection or prove broad transfer to all cybersecurity KGs.",
        },
        "outputs": {
            "target_graph_tsv": str(graph_path.resolve()),
            "target_train": str((target_dir / "train.txt").resolve()),
            "target_valid": str((target_dir / "valid.txt").resolve()),
            "target_test": str((target_dir / "test.txt").resolve()),
            "sha256": {
                "target_graph_tsv": sha256(graph_path),
                "target_train": sha256(target_dir / "train.txt"),
                "target_valid": sha256(target_dir / "valid.txt"),
                "target_test": sha256(target_dir / "test.txt"),
            },
        },
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "protocol": audit["protocol"],
        "cve_sampling": audit["cve_sampling"],
        "cross_domain": audit["cross_domain"],
        "graph": audit["graph"],
        "split": audit["split"],
        "outputs": audit["outputs"],
    }, indent=2))
    print(f"\nWrote target graph and splits under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
