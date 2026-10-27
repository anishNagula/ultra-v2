"""Shared loading, training and evaluation helpers for the transfer-conditions study.

Reuses the fresh FedULTRA protocol: message passing uses training triples only,
validation filters train+valid facts and test filters train+valid+test facts.
Unlike load_client, a client folder may also contain hidden.txt: true facts the
client does not observe (removed by the scarcity protocol). Hidden facts never
enter message passing or training, but they are filtered at evaluation so a model
is not penalised for ranking a true fact above the held-out answer.
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FRESH_SRC = ROOT / "experiments" / "fedultra_fresh" / "src"
for path in (ROOT, FRESH_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from torch_geometric.data import Data  # noqa: E402
from ultra.tasks import build_relation_graph  # noqa: E402
from evaluation_protocol import _filter_graph, _message_graph, _tensor_edges, read_triples  # noqa: E402

STUDY = ROOT / "experiments" / "transfer_conditions"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_triples(path, triples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for head, relation, tail in sorted(triples):
            handle.write(f"{head}\t{relation}\t{tail}\n")


def load_bundle(folder):
    """Load a client folder. valid/test/hidden are optional (a source client may have train only)."""
    folder = Path(folder)
    optional = lambda name: read_triples(folder / name) if (folder / name).is_file() else []
    raw = {"train": read_triples(folder / "train.txt"),
           "valid": optional("valid.txt"), "test": optional("test.txt")}
    hidden = optional("hidden.txt")
    splits = [raw["train"], hidden, raw["valid"], raw["test"]]
    seen = set()
    for triples in splits:
        if seen & set(triples):
            raise ValueError(f"{folder}: a fact appears in more than one of train/hidden/valid/test")
        seen |= set(triples)

    # Entities known to the client include ones whose facts were all hidden: the
    # catalogue (techniques, groups, software) is public even when relations are not.
    entities = sorted({e for triples in splits for h, _, t in triples for e in (h, t)})
    train_relations = sorted({r for _, r, _ in raw["train"]})
    for split in ("valid", "test"):
        missing = {r for _, r, _ in raw[split]} - set(train_relations)
        if missing:
            raise ValueError(f"{folder}/{split} has relations absent from train: {sorted(missing)}")
    entity_to_id = {e: i for i, e in enumerate(entities)}
    relation_to_id = {r: i for i, r in enumerate(train_relations)}
    num_nodes, num_rel = len(entity_to_id), len(relation_to_id)

    train_edges, train_types = _tensor_edges(raw["train"], entity_to_id, relation_to_id)
    train_data = _message_graph(train_edges, train_types, num_nodes, num_rel)
    bundle = {"folder": str(folder), "raw_triples": raw, "hidden": hidden,
              "entity_to_id": entity_to_id, "relation_to_id": relation_to_id,
              "train_data": train_data}

    def target_data(split):
        edges, types = _tensor_edges(raw[split], entity_to_id, relation_to_id)
        graph = Data(edge_index=train_data.edge_index, edge_type=train_data.edge_type,
                     num_nodes=num_nodes, num_relations=2 * num_rel,
                     target_edge_index=edges, target_edge_type=types)
        return build_relation_graph(graph)

    if raw["valid"]:
        bundle["valid_data"] = target_data("valid")
        bundle["valid_filter"] = _filter_graph(raw["train"] + hidden + raw["valid"],
                                               entity_to_id, relation_to_id, num_nodes, num_rel)
    if raw["test"]:
        bundle["test_data"] = target_data("test")
        bundle["test_filter"] = _filter_graph(raw["train"] + hidden + raw["valid"] + raw["test"],
                                              entity_to_id, relation_to_id, num_nodes, num_rel)
    return bundle
