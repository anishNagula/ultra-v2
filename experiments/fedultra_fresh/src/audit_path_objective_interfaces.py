#!/usr/bin/env python3
"""Read-only ULTRA interface audit before integrating a path-composition loss.

This script does not train, modify source files, or load/write checkpoints.
Run from the ULTRA repository root with the project virtualenv active.
"""
from __future__ import annotations
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path.cwd()
CANDIDATES = [
    "ultra/models.py",
    "ultra/model.py",
    "ultra/tasks.py",
    "ultra/training.py",
    "ultra/evaluation.py",
    "ultra/model_state.py",
    "script/run.py",
    "experiments/fedultra_fresh/src/attack_pattern_transfer_runner.py",
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def describe_python(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {"path": str(path), "sha256": sha256(path), "syntax_error": str(e)}
    items = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if isinstance(node, ast.ClassDef):
                bases = [ast.unparse(b) for b in node.bases]
                items.append({
                    "kind": "class", "name": node.name, "line": node.lineno,
                    "bases": bases,
                    "methods": [
                        {"name": n.name, "line": n.lineno,
                         "signature": ast.unparse(n.args)}
                        for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ],
                })
            else:
                # Focus on score, forward, predict, loss, train, evaluate, batch and path APIs.
                if any(k in node.name.lower() for k in (
                    "forward", "predict", "score", "loss", "train", "evaluate",
                    "batch", "negative", "metric", "path", "relation"
                )):
                    items.append({
                        "kind": "function", "name": node.name, "line": node.lineno,
                        "signature": ast.unparse(node.args),
                    })
    return {"path": str(path), "sha256": sha256(path), "definitions": sorted(items, key=lambda x: (x["line"], x["name"]))}

def main():
    report = {
        "purpose": "Read-only source interface audit; no training or checkpoint changes.",
        "cwd": str(ROOT),
        "python": sys.version,
        "files": [],
        "missing_candidates": [],
    }
    for rel in CANDIDATES:
        path = ROOT / rel
        if path.is_file():
            report["files"].append(describe_python(path))
        else:
            report["missing_candidates"].append(rel)

    # Runtime signatures can reveal the actual model/task API without running a training step.
    runtime = {}
    try:
        import torch
        runtime["torch_version"] = torch.__version__
    except Exception as e:
        runtime["torch_import_error"] = repr(e)
    for module_name in ("ultra.models", "ultra.model", "ultra.tasks", "ultra.training", "ultra.evaluation"):
        try:
            module = __import__(module_name, fromlist=["*"])
        except Exception as e:
            runtime[module_name] = {"import_error": repr(e)}
            continue
        found = {}
        for name in dir(module):
            obj = getattr(module, name)
            if inspect.isclass(obj) and any(k in name.lower() for k in ("ultra", "model", "task")):
                try:
                    found[name] = {"class_signature": str(inspect.signature(obj))}
                    methods = {}
                    for method_name in ("forward", "predict", "score", "forward_and_loss", "train_one_epoch", "evaluate"):
                        method = getattr(obj, method_name, None)
                        if callable(method):
                            try:
                                methods[method_name] = str(inspect.signature(method))
                            except Exception:
                                methods[method_name] = "signature unavailable"
                    found[name]["methods"] = methods
                except Exception:
                    pass
            elif inspect.isfunction(obj) and any(k in name.lower() for k in ("train", "evaluate", "negative", "metric")):
                try:
                    found[name] = str(inspect.signature(obj))
                except Exception:
                    found[name] = "signature unavailable"
        runtime[module_name] = found
    report["runtime_api"] = runtime

    out = ROOT / "experiments/fedultra_fresh/results/path_objective_interface_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(out),
        "files_inspected": len(report["files"]),
        "missing_candidates": report["missing_candidates"],
        "runtime_modules": list(runtime),
        "training_performed": False,
        "checkpoints_modified": False,
    }, indent=2))

if __name__ == "__main__":
    main()
