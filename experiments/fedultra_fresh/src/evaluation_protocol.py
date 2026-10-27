"""Leakage-safe ULTRA graph construction for the fresh FedULTRA experiment.

Message-passing graphs contain training triples only. Validation filtering uses
train + validation facts; test filtering uses train + validation + test facts.
All relations/entities are mapped deterministically from the training split.
"""
from pathlib import Path
import sys

import torch
from torch_geometric.data import Data

from ultra.tasks import build_relation_graph

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_ROOT = ROOT / "experiments" / "fedultra_fresh" / "data_processed"
CLIENTS = ("enterprise", "ics")
SPLITS = ("train", "valid", "test")


def read_triples(path):
    triples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3 or not all(parts):
                raise ValueError(f"Expected 3 non-empty tab-separated fields at {path}:{line_number}")
            triples.append(tuple(parts))  # (head, relation, tail)
    if not triples:
        raise ValueError(f"Empty split: {path}")
    if len(triples) != len(set(triples)):
        raise ValueError(f"Duplicate triples found in {path}")
    return triples


def _tensor_edges(triples, entity_to_id, relation_to_id):
    heads, tails, relations = [], [], []
    for head, relation, tail in triples:
        heads.append(entity_to_id[head])
        tails.append(entity_to_id[tail])
        relations.append(relation_to_id[relation])
    edge_index = torch.tensor([heads, tails], dtype=torch.long)
    edge_type = torch.tensor(relations, dtype=torch.long)
    return edge_index, edge_type


def _message_graph(train_edges, train_types, num_nodes, num_direct_relations):
    inverse_edges = train_edges.flip(0)
    inverse_types = train_types + num_direct_relations
    graph = Data(
        edge_index=torch.cat((train_edges, inverse_edges), dim=1),
        edge_type=torch.cat((train_types, inverse_types), dim=0),
        num_nodes=num_nodes,
        num_relations=2 * num_direct_relations,
        target_edge_index=train_edges,
        target_edge_type=train_types,
    )
    return build_relation_graph(graph)


def _filter_graph(triples, entity_to_id, relation_to_id, num_nodes, num_direct_relations):
    edge_index, edge_type = _tensor_edges(triples, entity_to_id, relation_to_id)
    # Deliberately use canonical (non-inverse) facts here. ULTRA's filtered
    # ranking mask matches direct relation IDs for both head and tail queries.
    return Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_nodes,
        num_relations=2 * num_direct_relations,
    )


def load_client(client, data_root=DATA_ROOT):
    """Return ULTRA Data objects with leakage-safe graph/filter separation."""
    # The standard experiment is restricted to the named MITRE clients. A caller
    # may explicitly supply a separate root for an independent target graph.
    resolved_root = Path(data_root).resolve()
    if client not in CLIENTS and resolved_root == DATA_ROOT.resolve():
        raise ValueError(f"Unknown client {client!r}; expected one of {CLIENTS} or an explicit data_root")
    folder = resolved_root / client
    raw = {split: read_triples(folder / f"{split}.txt") for split in SPLITS}

    train = raw["train"]
    train_entities = {entity for head, _, tail in train for entity in (head, tail)}
    train_relations = {relation for _, relation, _ in train}
    for split in ("valid", "test"):
        split_entities = {entity for head, _, tail in raw[split] for entity in (head, tail)}
        split_relations = {relation for _, relation, _ in raw[split]}
        if not split_entities <= train_entities:
            raise ValueError(f"{client}/{split} contains entities absent from train")
        if not split_relations <= train_relations:
            raise ValueError(f"{client}/{split} contains relations absent from train")

    # Sorting makes ID assignments deterministic across runs.
    entity_to_id = {value: index for index, value in enumerate(sorted(train_entities))}
    relation_to_id = {value: index for index, value in enumerate(sorted(train_relations))}
    num_nodes = len(entity_to_id)
    num_direct_relations = len(relation_to_id)

    encoded = {
        split: _tensor_edges(raw[split], entity_to_id, relation_to_id)
        for split in SPLITS
    }
    train_edges, train_types = encoded["train"]

    train_data = _message_graph(train_edges, train_types, num_nodes, num_direct_relations)

    def target_data(split):
        edges, types = encoded[split]
        graph = Data(
            edge_index=train_data.edge_index,
            edge_type=train_data.edge_type,
            num_nodes=num_nodes,
            num_relations=2 * num_direct_relations,
            target_edge_index=edges,
            target_edge_type=types,
        )
        return build_relation_graph(graph)

    valid_data = target_data("valid")
    test_data = target_data("test")

    valid_filter = _filter_graph(
        raw["train"] + raw["valid"], entity_to_id, relation_to_id,
        num_nodes, num_direct_relations,
    )
    test_filter = _filter_graph(
        raw["train"] + raw["valid"] + raw["test"], entity_to_id, relation_to_id,
        num_nodes, num_direct_relations,
    )

    return {
        "client": client,
        "raw_triples": raw,
        "entity_to_id": entity_to_id,
        "relation_to_id": relation_to_id,
        "train_data": train_data,
        "valid_data": valid_data,
        "test_data": test_data,
        "valid_filter": valid_filter,
        "test_filter": test_filter,
    }
