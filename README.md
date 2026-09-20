# Faithful Medical CBM

Research on faithful and intervenable concept bottleneck models for dermoscopic
melanoma classification. The frozen design is in [PROJECT_SPEC](docs/PROJECT_SPEC.md);
engineering constraints are in [AGENTS.md](AGENTS.md).

## Status

Stage 2B: the primary cohort, seven concept mappings, 80/20 case-level split and four
development folds are frozen. No image datasets/loaders, models or training exist.
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
model name and ordered concept names. Seed 42 is frozen for splitting. Image size 224
and batch size 32 remain provisional engineering defaults, not tuned choices.
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
- `data/processed/stage2a/`: cohort and excluded metadata.
- `data/splits/`: permanent development/test IDs, development validation-fold assignments and metadata.
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

The Stage 2B instruction accepts case-level stratification with its documented
patient-independence limitation. The architecture and 80/20 protocol remain unchanged.

## Stage 2B frozen splitting

```powershell
python -m src.data.splitting
python -m unittest discover -s tests -v
```

The split has already been created. Rerunning the command verifies its hashes, settings,
counts and identity invariants without generating another candidate or rewriting files.
There is no overwrite/force option. Changed inputs/settings, corrupted files or partial
outputs fail validation. Keep the four permanent files under version control; do not
delete and regenerate them. No alternative balance or seed was selected.

Input is exclusively the processed Stage 2A cohort and its summary. Generation uses
case_num string IDs sorted lexicographically, train_test_split(test_size=0.20,
shuffle=True, stratify=diagnosis_binary, random_state=42), then StratifiedKFold with
four shuffled folds and random_state=42 on sorted development IDs only.
Parameters come from configs/default.toml; the 20% fraction rounds up to 165 test cases.
Saved IDs are original case_num values, never dataframe row positions.

| Subset | Total | Positive | Negative |
|---|---:|---:|---:|
| Development | 658 | 198 | 460 |
| Locked test | 165 | 50 | 115 |
| Fold 0 validation | 165 | 50 | 115 |
| Fold 1 validation | 165 | 50 | 115 |
| Fold 2 validation | 164 | 49 | 115 |
| Fold 3 validation | 164 | 49 | 115 |

`development_ids.csv` and `test_ids.csv` contain case_num only.
`development_folds.csv` contains case_num and validation_fold (0-based); for fold k,
validation uses k and training uses the remaining development folds.
`split_metadata.json` records both training/validation class counts, the seed, methods,
input/output hashes, runtime versions and the methodological limitation.

This is CASE-LEVEL separation. case_num is not a patient identifier, and patient-level
independence cannot be verified. The locked test is reserved for final frozen evaluation
and may not be used during development, tuning, calibration, early stopping or debugging.
The splitter never opens raw metadata, images, or original Derm7pt split-index files.
