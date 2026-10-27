# Files kept out of git

These files exist in the working copy but are not committed. All of them are public
downloads or are regenerated from public downloads. Together they would add about 235 MB to
the repository. They are listed in `.gitignore`.

Every experiment can be re-run from the committed, derived datasets. The raw files are needed
only to rebuild those datasets from scratch.

| Path | Size | SHA-256 | What it is | Needed for |
|---|---|---|---|---|
| `datasets/cybersecurity/mitre/enterprise/v19.2/enterprise-attack-19.2.json` | 53.8 MB | `dc1639caa5501d720e280cf1cbd8fbe009884a0c9b3e6e9ed9d0c25166c3d8f4` | MITRE ATT&CK Enterprise v19.2 STIX bundle | Rebuilding `experiments/enterprise_split/data/` (`build_split.py`, the rebuild check in `test_pipeline.py`) |
| `datasets/cybersecurity/mitre/mobile/v19.2/mobile-attack-19.2.json` | 5.8 MB | `acfa5ca2d93484476f79bf38590e2b55bb675fc0ce85e76bffa0af2c82dada64` | MITRE ATT&CK Mobile v19.2 STIX bundle | Rebuilding `experiments/transfer_conditions/data/real/mobile_*` |
| `data/raw/nvd/nvdcve-2.0-{2020..2025}.json.gz` | 120 MB in total | in `data/raw/source_sha256.txt` | NVD CVE JSON 2.0 yearly feeds | `scripts/cybersecurity/build_cyber_cti_graph.py` |
| `data/raw/cwe/cwec_latest.xml.zip` | 2.0 MB | in `data/raw/source_sha256.txt` | MITRE CWE catalogue | same |
| `data/raw/capec/capec_latest.zip` | 0.6 MB | in `data/raw/source_sha256.txt` | MITRE CAPEC catalogue | same |
| `data/processed/cyber_cti/triples.tsv` | 53.0 MB | `079cf94d67813ab20ed9054959048cc1ff9e5a0e486d931a295d903221bebbeb` | CTI graph built from the raw files above | `scripts/cybersecurity/prepare_attack_to_cve_transfer.py` |

## Recovering them

- **ATT&CK bundles.** Download the versioned v19.2 files from MITRE's ATT&CK STIX data
  repository (`mitre-attack/attack-stix-data` on GitHub: `enterprise-attack/enterprise-attack-19.2.json`
  and `mobile-attack/mobile-attack-19.2.json`). Place them at the paths above, and check the SHA-256
  before rebuilding anything.
- **NVD, CWE and CAPEC.** These are "latest" or yearly feeds that their publishers keep updating. A fresh
  download will therefore not match the recorded hashes. The hashes identify the exact snapshot
  used here. Download from NVD (`nvdcve-2.0-YYYY.json.gz`), MITRE CWE (`cwec_latest.xml.zip`)
  and MITRE CAPEC.
- **`triples.tsv`.** Run `python scripts/cybersecurity/build_cyber_cti_graph.py` (defaults read the raw paths
  above and write to `data/processed/cyber_cti/`). The processed transfer split that the
  CVE-transfer runner reads, `data/processed/cyber_transfer/`, is committed, so that runner does not
  need this file.

Not committed for other reasons: `.venv/` (local virtual environment, 431 MB; recreate from
`requirements.txt` and `INSTALL_AND_RUN.txt`), `__pycache__/` and `.DS_Store` files.

## Regenerable model weights

| Path | Size | Integrity | What it is | How to recover |
|---|---|---|---|---|
| `experiments/real_transfer/results/*/source_cache/source_*.pt` | about 0.7 MB each | each run file records its SHA-256 as `starting_weights.state_sha256` | Weights produced by the real-transfer source stage, cached so several target fractions can reuse them | Rerun the stage: `python experiments/real_transfer/src/run_method.py --arm <arm> --seeds <seeds> --source-only --source-cache <dir> ...` (see `experiments/real_transfer/README.md`). The stage is deterministic (`test_method.py` checks this), so a rerun reproduces the recorded hash. The runner refuses a cached file whose weights do not match its own recorded hash. |

These are binary, about 28 MB in total, and fully determined by committed code, data and seeds.
