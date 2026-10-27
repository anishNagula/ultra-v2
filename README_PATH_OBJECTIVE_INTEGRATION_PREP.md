# ULTRA path-objective integration prep

This is a read-only interface audit used to inspect the exact ULTRA scoring/training API before wiring a path-composition objective into it. It intentionally does not guess model method names, launch training, alter the checkpoint, or change existing experiment outputs.

## Run

From the `ultra-v2` repository root with its virtual environment active:

```bash
python experiments/fedultra_fresh/src/audit_path_objective_interfaces.py
```

It writes:
`experiments/fedultra_fresh/results/path_objective_interface_audit.json`

Send that JSON back. The next patch can then be written against the actual scorer and training loss signatures rather than an assumed API.
