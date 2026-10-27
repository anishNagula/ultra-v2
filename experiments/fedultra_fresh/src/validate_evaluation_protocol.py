"""Run structural and filtered-ranking checks for evaluation_protocol.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from ultra import tasks
from evaluation_protocol import CLIENTS, load_client



def validate(client):
    bundle = load_client(client)
    raw = bundle["raw_triples"]
    entity_to_id = bundle["entity_to_id"]
    relation_to_id = bundle["relation_to_id"]
    train_data = bundle["train_data"]
    valid_data = bundle["valid_data"]
    test_data = bundle["test_data"]
    valid_filter = bundle["valid_filter"]
    test_filter = bundle["test_filter"]

    # Message-passing edges must be precisely train facts plus their inverses.
    train_edge_count = len(raw["train"])
    assert train_data.edge_index.shape[1] == 2 * train_edge_count
    assert train_data.edge_type.shape[0] == 2 * train_edge_count
    assert torch.equal(valid_data.edge_index, train_data.edge_index)
    assert torch.equal(test_data.edge_index, train_data.edge_index)
    assert torch.equal(valid_data.edge_type, train_data.edge_type)
    assert torch.equal(test_data.edge_type, train_data.edge_type)

    # Validation filtering excludes test-only facts; test filtering includes all facts.
    assert valid_filter.edge_index.shape[1] == len(raw["train"]) + len(raw["valid"])
    assert test_filter.edge_index.shape[1] == sum(len(raw[s]) for s in ("train", "valid", "test"))

    # Every validation/test positive must be masked as a known positive for both directions.
    for split, data, filter_graph in (
        ("valid", valid_data, valid_filter),
        ("test", test_data, test_filter),
    ):
        # ULTRA strict_negative_mask accepts (head, tail, relation).
        batch = torch.stack(
            [data.target_edge_index[0], data.target_edge_index[1], data.target_edge_type],
            dim=1,
        )
        t_mask, h_mask = tasks.strict_negative_mask(filter_graph, batch)
        row = torch.arange(len(batch))
        assert not t_mask[row, batch[:, 1]].any(), f"{client}/{split}: positive tail not filtered"
        assert not h_mask[row, batch[:, 0]].any(), f"{client}/{split}: positive head not filtered"

        # Verify all mapped IDs are in range and the direct relation set is stable.
        assert int(data.target_edge_index.max()) < len(entity_to_id)
        assert int(data.target_edge_type.max()) < len(relation_to_id)
        print(f"  {split}: positives={len(batch)}, filter_facts={filter_graph.edge_index.shape[1]}, masks=PASS")

    print(
        f"PASS {client}: entities={len(entity_to_id)}, relations={len(relation_to_id)}, "
        f"train={len(raw['train'])}, valid={len(raw['valid'])}, test={len(raw['test'])}"
    )


if __name__ == "__main__":
    for name in CLIENTS:
        validate(name)
