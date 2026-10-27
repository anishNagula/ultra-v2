FedULTRA relation coverage audit patch

Merge/extract this ZIP into the root of ~/code/ultra-v2 (do not replace the experiments folder):
  cd ~/code/ultra-v2
  unzip -o ~/Downloads/fedultra_relation_coverage_audit.zip -d .

Run:
  source .venv/bin/activate
  python experiments/fedultra_fresh/src/audit_relation_coverage.py

This is a read-only audit of the restored split files. It does not modify any dataset or code.
