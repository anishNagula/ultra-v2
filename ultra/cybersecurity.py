"""Cybersecurity-specific path support utilities for ULTRA.

This module deliberately keeps the pretrained ULTRA architecture/checkpoint intact.
It defines an auditable task adapter: attack-pattern-related triples are selected
using STIX type metadata, and a query is path-supported only when its endpoints are
connected by an alternate 2..max_hops path in the training graph after the queried
fact is removed. Entity types are used for task selection/analysis, never fed to ULTRA.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import json


def load_stix_type_map(path: str | Path) -> dict[str, str]:
    """Return STIX object ID -> type from a MITRE ATT&CK bundle JSON."""
    path = Path(path)
    bundle = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for obj in bundle.get("objects", []):
        object_id, object_type = obj.get("id"), obj.get("type")
        if isinstance(object_id, str) and isinstance(object_type, str):
            result[object_id] = object_type
    if not result:
        raise ValueError(f"No STIX id/type entries found in {path}")
    return result


def is_attack_pattern_triple(triple, stix_types: dict[str, str]) -> bool:
    """Whether either endpoint of an original STIX triple is an attack-pattern."""
    head, _relation, tail = triple
    return (
        stix_types.get(head) == "attack-pattern"
        or stix_types.get(tail) == "attack-pattern"
    )


def _adjacency(context_triples):
    """Build adjacency once; retain edge identity so each query can hide its own fact."""
    adjacency = {}
    for triple in context_triples:
        head, relation, tail = triple
        edge_key = (head, relation, tail)
        adjacency.setdefault(head, []).append((tail, edge_key))
        adjacency.setdefault(tail, []).append((head, edge_key))
    return adjacency


def _has_path_in_adjacency(adjacency, query_triple, max_hops: int = 4) -> bool:
    """True if query endpoints have an alternate path of length 2..max_hops.

    The exact queried triple is omitted. Because ULTRA adds inverse edges, path
    support is checked on an undirected adjacency representation. A direct edge is
    never sufficient; a path must have at least two hops.
    """
    if max_hops < 2:
        raise ValueError("max_hops must be >= 2")
    head, _relation, tail = query_triple
    if head == tail:
        return False
    skip = tuple(query_triple)
    queue = deque([(head, 0)])
    best_depth = {head: 0}
    while queue:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbor, edge_key in adjacency.get(node, ()):
            if edge_key == skip:
                continue
            next_depth = depth + 1
            if neighbor == tail and next_depth >= 2:
                return True
            if next_depth < max_hops and next_depth < best_depth.get(neighbor, max_hops + 1):
                best_depth[neighbor] = next_depth
                queue.append((neighbor, next_depth))
    return False


def has_alternate_path(context_triples, query_triple, max_hops: int = 4) -> bool:
    """True if query endpoints have an alternate path of length 2..max_hops."""
    if max_hops < 2:
        raise ValueError("max_hops must be >= 2")
    return _has_path_in_adjacency(_adjacency(context_triples), query_triple, max_hops)


def path_supported_indices(context_triples, query_triples, stix_types,
                           max_hops: int = 4, attack_pattern_only: bool = True):
    """Return indices of queries with an alternate path in ``context_triples``.

    Adjacency is built once per split, rather than once per query, to keep the audit
    tractable on the Enterprise graph.
    """
    if max_hops < 2:
        raise ValueError("max_hops must be >= 2")
    adjacency = _adjacency(context_triples)
    selected = []
    for index, triple in enumerate(query_triples):
        if attack_pattern_only and not is_attack_pattern_triple(triple, stix_types):
            continue
        if _has_path_in_adjacency(adjacency, triple, max_hops):
            selected.append(index)
    return selected


def attack_pattern_indices(triples, stix_types):
    """Return indices for triples incident to at least one attack-pattern node."""
    return [i for i, triple in enumerate(triples)
            if is_attack_pattern_triple(triple, stix_types)]
