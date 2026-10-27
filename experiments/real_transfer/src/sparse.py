"""Training on sparse views of a client's own graph.

A view keeps max(1, round(f * n_h)) of each head entity's facts. The kept facts are chosen by a shuffle keyed by
(seed, round, head). This is the per-head rule that produced the target's scarce training graph
(`enterprise_split/src/build_split.py::scarce`), redrawn every round. The model passes messages over the
kept facts only and learns to predict held-out facts. Negatives are filtered against the client's complete
graph, so no fact the client knows is ever used as a negative. Everything here runs inside one client,
on that client's own facts.
"""
import random
from collections import defaultdict

import torch
from torch.nn import functional as F
from torch_geometric.data import Data

import env  # noqa: F401  (puts the reused code on sys.path)
import fedavg_runner as base
from evaluation_protocol import _filter_graph, _message_graph, _tensor_edges
from ultra import tasks


def per_head_view(triples, fraction, key):
    """Split facts into (kept, held out): per head, keep max(1, round(fraction * n_h)) after a keyed shuffle."""
    by_head = defaultdict(list)
    for x in sorted(triples):
        by_head[x[0]].append(x)
    keep, hide = [], []
    for head, facts in sorted(by_head.items()):
        random.Random(f"{key}:{head}").shuffle(facts)
        k = max(1, round(float(fraction) * len(facts)))
        keep += facts[:k]
        hide += facts[k:]
    return sorted(keep), sorted(hide)


def fact_tensor(triples, bundle):
    """(head, tail, relation) ids, the layout of fedavg_runner.triples_tensor."""
    edges, types = _tensor_edges(triples, bundle["entity_to_id"], bundle["relation_to_id"])
    return torch.stack([edges[0], edges[1], types], dim=1)


def message_graph(triples, bundle):
    """Facts plus inverse edges over the client's full entity set, with ULTRA's relation graph."""
    edges, types = _tensor_edges(triples, bundle["entity_to_id"], bundle["relation_to_id"])
    return _message_graph(edges, types, len(bundle["entity_to_id"]), len(bundle["relation_to_id"]))


def filter_graph(triples, bundle):
    """Known facts (with inverses) for strict negative sampling; no relation graph is needed."""
    edges, types = _tensor_edges(triples, bundle["entity_to_id"], bundle["relation_to_id"])
    num_rel = len(bundle["relation_to_id"])
    return Data(edge_index=torch.cat((edges, edges.flip(0)), dim=1),
                edge_type=torch.cat((types, types + num_rel)),
                num_nodes=len(bundle["entity_to_id"]), num_relations=2 * num_rel)


def train_steps(model, optimizer, graph, negatives_from, positives, steps, batch_size, num_negative, stream_seed):
    """fedavg_runner.train_updates with its three inputs made explicit.

    The inputs are the message-passing graph, the graph that filters negatives, and the positives.
    With graph = negatives_from = train_data and positives = triples_tensor(train_data), this is
    the same computation as train_updates (test_method.py checks this).
    """
    if batch_size % 2:
        raise ValueError("batch size must be even: ULTRA splits each batch into tail and head negatives")
    torch.manual_seed(stream_seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stream_seed)
    model.train()
    losses = []
    for _ in range(steps):
        ids = torch.randint(len(positives), (batch_size,), generator=generator)
        batch = tasks.negative_sampling(negatives_from, positives[ids], num_negative=num_negative, strict=True)
        pred = model(graph, batch)
        target = torch.zeros_like(pred)
        target[:, 0] = 1
        loss_each = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        weights = torch.ones_like(pred)
        if num_negative > 0:
            with torch.no_grad():
                weights[:, 1:] = F.softmax(pred[:, 1:], dim=-1)
        loss = ((loss_each * weights).sum(dim=-1) / weights.sum(dim=-1)).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all().item() for g in grads):
            raise FloatingPointError("missing or non-finite gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
    return sum(losses) / len(losses)


def source_stream(seed, round_index):
    """Sampling stream of the source stage, distinct from the target's (seed*1000003 + 7919 + round*97)."""
    return seed * 1000003 + 104729 + round_index * 97


SOURCE_VALID_KEY = "real-transfer source validation v1"


def source_holdout(source, task, share=0.10):
    """Fixed split (identical for every seed) of the source's task facts into training and source validation."""
    facts = source["raw_triples"]["train"]
    task_facts = sorted(x for x in facts if task(x))
    random.Random(SOURCE_VALID_KEY).shuffle(task_facts)
    valid = sorted(task_facts[:round(share * len(task_facts))])
    held = set(valid)
    return sorted(x for x in facts if x not in held), valid


def source_validation(source, train_facts, valid_facts, fraction):
    """Score source-validation facts in the regime being trained for.

    The message graph is one fixed per-head view (fraction < 1) or all the source's training facts
    (fraction = 1). The filter holds every source fact.
    """
    keep = per_head_view(train_facts, fraction, key=SOURCE_VALID_KEY)[0] if fraction < 1 else train_facts
    graph = message_graph(keep, source)
    edges, types = _tensor_edges(valid_facts, source["entity_to_id"], source["relation_to_id"])
    valid_data = Data(edge_index=graph.edge_index, edge_type=graph.edge_type, num_nodes=graph.num_nodes,
                      num_relations=graph.num_relations, target_edge_index=edges, target_edge_type=types)
    valid_data.relation_graph = graph.relation_graph
    valid_filter = _filter_graph(source["raw_triples"]["train"], source["entity_to_id"], source["relation_to_id"],
                                 len(source["entity_to_id"]), len(source["relation_to_id"]))
    return {"train_data": graph, "valid_data": valid_data, "valid_filter": valid_filter}


def train_source(model, source, *, fraction, rounds, steps, batch_size, num_negative, seed, lr, weight_decay,
                 task=None, select=False, checkpoint="source_valid", on_round=None):
    """Source-only training on the source's own graph.

    fraction < 1: each round draws a fresh sparse view and trains on the held-out facts of `task`.
    fraction = 1: messages pass over the complete graph and the positives are all facts of `task`
    (ULTRA's remove_easy_edges removes the training edge from the graph).
    `task` filters raw triples (default: every fact).
    select=True: a fixed 10% of the task facts is set aside as source validation, never trained on, and
    scored after every round in the same regime. checkpoint="source_valid" leaves the model at the best
    round (0 = start); checkpoint="final" leaves it at the last round, and the scores are only recorded.
    Target data are never read. Returns a record with per-round losses (and validation MRRs).
    """
    if checkpoint not in ("source_valid", "final"):
        raise ValueError(checkpoint)
    task = task or (lambda triple: True)
    negatives_from = source["train_data"]  # every source fact, including source validation
    facts, valid = source_holdout(source, task) if select else (source["raw_triples"]["train"], [])
    scorer = source_validation(source, facts, valid, fraction) if select else None
    optimizer = base.make_optimizer(model, lr, weight_decay)
    record = {"losses": [], "source_valid_mrr": [], "selected_round": rounds, "source_valid_facts": len(valid)}
    if select:
        record["source_valid_mrr"].append(base.evaluate(model, scorer, "valid", 4, 0)["mrr"])
        best_state, record["selected_round"] = base.clone_state(model), 0
    if fraction >= 1:
        dense = (message_graph(facts, source), [x for x in facts if task(x)])
    for r in range(1, rounds + 1):
        if fraction < 1:
            keep, held = per_head_view(facts, fraction, key=f"{seed}:{r}")
            graph, positive_facts = message_graph(keep, source), [x for x in held if task(x)]
        else:
            graph, positive_facts = dense
        loss = train_steps(model, optimizer, graph, negatives_from, fact_tensor(positive_facts, source),
                           steps, batch_size, num_negative, source_stream(seed, r))
        record["losses"].append(loss)
        if select:
            record["source_valid_mrr"].append(base.evaluate(model, scorer, "valid", 4, 0)["mrr"])
            if record["source_valid_mrr"][-1] > record["source_valid_mrr"][record["selected_round"]]:
                best_state, record["selected_round"] = base.clone_state(model), r
        if on_round:
            on_round(r, loss)
    if select:
        record["source_valid_best_round"] = record["selected_round"]
        if checkpoint == "final":
            record["selected_round"] = rounds
        else:
            model.load_state_dict(best_state, strict=True)
    record["checkpoint_rule"] = checkpoint if select else "final"
    return record
