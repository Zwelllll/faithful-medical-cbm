# Faithful Medical CBM

Research on faithful and intervenable concept bottleneck models for dermoscopic
melanoma classification. The frozen design is in [PROJECT_SPEC](docs/PROJECT_SPEC.md);
engineering constraints are in [AGENTS.md](AGENTS.md).

## Status

Stage 2A: raw Derm7pt audited; the user-approved primary cohort and seven binary
concept mappings are frozen. No development/test split, folds, model or training exists.
The starter lives in this directory, one level below the supplied workspace root.

## Setup (PowerShell, Python 3.11+)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/report_environment.py
```

Use a mutually compatible PyTorch/torchvision installation appropriate to the
machine's CUDA runtime, or a CPU build. Dependency lower bounds are not a tested
lockfile. Record the resolved environment before experiments with
`python -m pip freeze > artifacts/requirements-resolved.txt` and
`python scripts/report_environment.py > artifacts/environment.json`.
Torchvision is the planned augmentation library; no transforms are implemented yet.
Tests use Python's standard-library unittest framework. RNG tests skip explicitly
when NumPy or PyTorch is absent. No lint tool is configured.

## Configuration and reproducibility

`configs/default.toml` centralizes seeds, paths, image size, batch size, folds,
model name and ordered concept names. Seed 42, image size 224 and batch size 32
are provisional engineering defaults, not tuned experimental choices.
Paths resolve relative to the parent of the configuration directory, independent
of the shell's working directory. Keep configs directly in `configs/`.

```python
from faithful_medical_cbm.config import load_config
from faithful_medical_cbm.reproducibility import seed_everything

config = load_config("configs/default.toml")
seed_everything(**config["reproducibility"])
```

Call seeding before CUDA initialization. Strict deterministic algorithms may
reject unsupported operations. Reproducibility across library versions/hardware
is not guaranteed. DataLoader worker seeding will be added at the DataLoader stage.

## Layout

- `src/faithful_medical_cbm/`: importable package; data, models, training,
  interventions and evaluation subpackages reserved for later stages.
- `src/data/audit.py`: standalone read-only release audit, run from the repository root.
- `configs/`: central configuration.
- `scripts/`: environment reporting.
- `tests/`: initialization and synthetic dataset-integrity checks.
- `data/raw/release_v0/`: user-provided raw release, excluded from Git.
- `data/processed/stage2a/`: cohort and excluded metadata; `data/splits/` remains empty.
- `artifacts/audit/`: audit evidence and report; checkpoints and notebooks remain empty.
- `docs/`: authoritative specification and append-only decision record.

## Dataset audit

```powershell
python -m src.data.audit --raw data/raw/release_v0 --output artifacts/audit
python -m unittest discover -s tests -v
```

The audit uses the standard library and Pillow, retains raw categorical strings,
checks every image by decoding it, and hashes the complete release before and after
inspection to verify that its contents remain unchanged. Supplied split-index files
are inventoried and hashed, never parsed or adopted. Outputs cannot overlap the raw tree.

See [audit report](artifacts/audit/report.md) and [machine-readable summary](artifacts/audit/summary.json).
Detailed CSV files record raw file hashes, image dimensions/formats, image references,
case-image pairs and all raw value counts. Duplicate reports compare filenames,
metadata rows, references, exact file bytes and decoded RGB pixels; near-duplicates
are not excluded by these checks. Metadata and notes determine modality, without
visual clinical adjudication. Provenance records Python/Pillow and audit source hashes.

The audit finds no patient identifier. `case_num` supports a case-level fallback;
the sparse `case_id` does not establish complete patient or lesion grouping.
The Stage 1 report is historical; its clinical inclusion and concept-conversion
questions were resolved by the user's Stage 2A instructions and recorded in DECISIONS.md.

## Stage 2A cohort construction

```powershell
python -m src.data.cohort
python -m unittest discover -s tests -v
```

`configs/cohort_mapping.json` stores every included/excluded diagnosis and every
audited raw concept category's explicit binary mapping, in the frozen target order.
`src/data/cohort.py` fails on unknown/missing labels, duplicate case numbers, output
column collisions, unapproved path corrections, or raw inputs differing from Stage 1.
Original Derm7pt split-index files are never opened by this stage.

Outputs:

- `data/processed/stage2a/cohort.csv`: 823 cases, including 248 positive and 575 negative.
  All 19 raw columns remain intact. Added columns are `diagnosis_binary`, the seven
  named binary concept targets, and `derm_path` relative to `data/raw/release_v0/images/`.
- `data/processed/stage2a/excluded.csv`: all 188 excluded rows with raw fields and an
  exclusion reason. Excluded diagnoses are never encoded as negative targets.
- `artifacts/stage2a/summary.json`: class counts, excluded counts by raw diagnosis,
  concept counts/prevalence, path corrections and SHA-256 provenance/output hashes.

Generated processed metadata remains excluded from Git under the existing data policy;
it is reproducible with the command above. No images are copied or changed. The only
path correction is case 816's `FCl/Fcl068.jpg` to `FCL/Fcl068.jpg` in the added derm_path;
the raw derm column remains unchanged. All metadata/image hashes are checked against
Stage 1 and checked again after construction. Raw diagnosis, seven_point_score and
other retained metadata are audit fields, not model inputs.

Before Stage 2B, acknowledge the case-level grouping limitation or obtain verified
patient/lesion linkage. The architecture and 80/20 protocol remain unchanged.
