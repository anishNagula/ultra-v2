FedULTRA remaining-fixes patch

Merge this ZIP into the ROOT of ~/code/ultra-v2. Do not replace the experiments folder.
It updates build_dataset.py to use paths relative to its own file and adds CLI overrides,
and adds a one-update smoke test. It does not modify ULTRA core files or run full training.

Commands:
  cd ~/code/ultra-v2
  unzip -o ~/Downloads/fedultra_remaining_fixes_patch.zip -d .
  source .venv/bin/activate

First run the smoke test using the restored splits:
  python experiments/fedultra_fresh/src/smoke_test.py

To verify the configurable source paths without rewriting the dataset, run:
  python experiments/fedultra_fresh/src/build_dataset.py --help

Only run build_dataset.py without --help if you intentionally want to regenerate the processed splits.
The builder defaults to the MITRE files under ~/code/ULTRA/... but accepts explicit source paths.
