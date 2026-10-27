#!/usr/bin/env python3
"""One-update ULTRA smoke test on each client; not a benchmark.

Checks checkpoint loading, one training update, finite loss/gradients, and
small-sample filtered ranking. Uses CPU by default for portable sparse ops.
"""
import copy
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import optim
from torch.nn import functional as F
from ultra import tasks
from ultra.models import Ultra
from evaluation_protocol import CLIENTS, load_client

CHECKPOINT = ROOT / "ckpts" / "ultra_4g.pth"
MODEL_CONFIG = {
    "relation_model": {
        "class": "RelNBFNet", "input_dim": 64,
        "hidden_dims": [64, 64, 64, 64, 64, 64],
        "message_func": "distmult", "aggregate_func": "sum",
        "short_cut": True, "layer_norm": True,
    },
    "entity_model": {
        "class": "EntityNBFNet", "input_dim": 64,
        "hidden_dims": [64, 64, 64, 64, 64, 64],
        "message_func": "distmult", "aggregate_func": "sum",
        "short_cut": True, "layer_norm": True,
    },
}


def new_model():
    model = Ultra(
        rel_model_cfg=copy.deepcopy(MODEL_CONFIG["relation_model"]),
        entity_model_cfg=copy.deepcopy(MODEL_CONFIG["entity_model"]),
    )
    state = torch.load(CHECKPOINT, map_location="cpu")
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(f"Unexpected checkpoint format: {CHECKPOINT}")
    model.load_state_dict(state["model"], strict=True)
    return model


def to_batch(data, limit=4):
    positives = torch.stack(
        [data.target_edge_index[0], data.target_edge_index[1], data.target_edge_type],
        dim=1,
    )
    if len(positives) < limit:
        limit = len(positives)
    return positives[:limit].clone()


def smoke(client):
    bundle = load_client(client)
    train_data = bundle["train_data"]
    model = new_model()
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)

    positives = to_batch(train_data, limit=4)
    batch = tasks.negative_sampling(train_data, positives, num_negative=8, strict=True)
    pred = model(train_data, batch)
    target = torch.zeros_like(pred)
    target[:, 0] = 1
    loss_each = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    weights = torch.ones_like(pred)
    with torch.no_grad():
        weights[:, 1:] = F.softmax(pred[:, 1:], dim=-1)
    loss = ((loss_each * weights).sum(dim=-1) / weights.sum(dim=-1)).mean()
    if not torch.isfinite(loss):
        raise AssertionError(f"{client}: non-finite training loss")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all().item() for g in grads):
        raise AssertionError(f"{client}: missing or non-finite gradients")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Smoke-only validation metric on a small deterministic subset. This is NOT
    # a reported benchmark result and does not select or save a checkpoint.
    model.eval()
    valid_data = bundle["valid_data"]
    val_batch = to_batch(valid_data, limit=4)
    with torch.no_grad():
        tail_candidates, head_candidates = tasks.all_negative(train_data, val_batch)
        tail_scores = model(train_data, tail_candidates)
        head_scores = model(train_data, head_candidates)
        tail_mask, head_mask = tasks.strict_negative_mask(bundle["valid_filter"], val_batch)
        tail_rank = tasks.compute_ranking(tail_scores, val_batch[:, 1], tail_mask)
        head_rank = tasks.compute_ranking(head_scores, val_batch[:, 0], head_mask)
        ranks = torch.cat([tail_rank, head_rank]).float()
        mrr = (1.0 / ranks).mean().item()
    if not math.isfinite(mrr):
        raise AssertionError(f"{client}: non-finite smoke MRR")
    print(
        f"PASS {client}: checkpoint=loaded, update=1, "
        f"loss={loss.item():.6f}, finite_gradients={len(grads)}, "
        f"smoke_valid_mrr={mrr:.6f} (n={len(val_batch)} triples; diagnostic only)"
    )


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")
    torch.manual_seed(20261023)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    print(f"Device: CPU\nCheckpoint: {CHECKPOINT}")
    for client in CLIENTS:
        smoke(client)
    print("SMOKE TEST PASS: no full training run was performed.")


if __name__ == "__main__":
    main()
