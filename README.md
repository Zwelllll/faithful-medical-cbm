# Faithful Medical CBM

Research on faithful and intervenable concept bottleneck models for dermoscopic
melanoma classification. The frozen design is in [PROJECT_SPEC](docs/PROJECT_SPEC.md);
engineering constraints are in [AGENTS.md](AGENTS.md).

## Status

Stage 4: the black-box EfficientNet-B0 baseline and development-fold training
infrastructure are implemented. No real training experiment has been run.
The cohort, concept mappings and splits remain frozen.
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
Torchvision supplies the implemented image transforms and EfficientNet-B0 backbone.
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
is not guaranteed. DataLoader worker seeding is implemented in the loader utilities.

## Layout

- `src/faithful_medical_cbm/`: importable package with configuration, reproducibility,
  Dataset/DataLoader utilities, and the black-box baseline model/training modules.
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

## Stage 3 Dataset and DataLoader infrastructure

```python
from faithful_medical_cbm.config import load_config
from faithful_medical_cbm.reproducibility import seed_everything
from faithful_medical_cbm.data.loaders import LoaderFactory

# When num_workers > 0, run this inside if __name__ == "__main__": on Windows.
config = load_config("configs/default.toml")
seed_everything(**config["reproducibility"])
factory = LoaderFactory("configs/default.toml")
development_train = factory.development()
development_eval = factory.development(training=False)
fold_zero = factory.fold(0)  # keys: "train", "validation"
test_loader = factory.locked_test()  # construction only; sample access is blocked
```

Use `pip install -e .` as described above to import the package. Each Dataset item is:

| Key | Type/shape | Meaning |
|---|---|---|
| image | float32 tensor `[3, H, W]` | normalized dermoscopic RGB image |
| diagnosis | float32 scalar tensor | processed binary diagnosis |
| concepts | float32 tensor `[7]` | frozen Stage 2A target order |
| case_num | string | unchanged permanent case identifier |
| split | string | development or test |
| role | string | train, validation, or evaluation |
| validation_fold | integer | case's own frozen validation fold; -1 for test |

Default batches have image shape `[B, 3, 224, 224]`, diagnosis `[B]`, and concepts
`[B, 7]`. Bookkeeping is for provenance, not model input. Concept order comes from
the hash-verified Stage 2A summary; raw labels are never remapped in the Dataset.
Only selected processed rows are converted into targets/paths once during construction.
`__getitem__` opens only derm_path, converts to RGB, transforms the image and returns
precomputed targets. It never parses CSV or categorical metadata. Clinical photographs,
sex, location, seven_point_score and other audit metadata are not returned or used.

Transforms are defined in `data/transforms.py` and configured in `configs/default.toml`:

- Training: random resized crop (area 0.85–1.0, aspect ratio 0.9–1.1), horizontal and
  vertical flips (each probability 0.5), rotation within ±15 degrees, and brightness,
  contrast and saturation jitter of 0.1. Crop resizing is bicubic; rotation is bilinear
  with torchvision's default black fill. These are provisional engineering settings,
  not choices based on locked-test performance.
- Evaluation: deterministic bicubic resize of the full image to 224×224 without cropping;
  this resizes aspect ratio to square. No random operations or color jitter.
- Both: convert pixel values to float32 in [0, 1], then ImageNet RGB normalization with
  mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. No weights are loaded.

Batch size 32, image size 224, workers 0 and pin_memory=false are centralized in config.
Training shuffles with a seeded torch.Generator. Evaluation keeps frozen ID order.
No loader drops the last batch. Fold training uses the other three development folds;
validation uses the requested fold. Factories never generate or rewrite split files.
Input hashes and ID disjointness/coverage are checked before constructing datasets.

For reproducible training augmentations with zero workers, seed the main process with
seed_everything before iteration. With workers enabled, spawned workers receive seeded
PyTorch state plus Python/NumPy seeds derived from torch.initial_seed(). Worker counts
may change augmentation streams; reproducibility assumes the same configuration/runtime.

The locked-test factory is separate and uses evaluation transforms only. Its default
Dataset blocks sample access before opening an image. The explicit
allow_locked_test_iteration=True option is reserved for the final frozen evaluation;
do not enable it for development or debugging. Stage 3 tests use synthetic test fixtures,
and the real locked-test loader is constructed but not indexed or iterated.

Patient-level independence still cannot be verified. GPU pinning/transfer remains
unverified on the current CPU-only runtime; this does not block CPU infrastructure use.

## Stage 4 black-box baseline

Stage 4B first-run preparation: open [the Fold 0 Colab notebook](notebooks/colab_fold0.ipynb).
It uses a pinned Git checkout and Google Drive for development data and persistent
outputs, verifies frozen hashes and fold membership, and runs only Fold 0 when its
training cell is explicitly executed. No real GPU experiment has been run locally.

See [GPU/Colab execution instructions](docs/BASELINE_GPU.md) for prerequisites,
the exact training schedule, checkpoint loading and limitations. The real-training
entry point requires CUDA and operates on one frozen development fold at a time:

```bash
python -m faithful_medical_cbm.training.train_black_box --config configs/default.toml --fold 0 --run-name baseline-v1
```

The direct image-to-logit EfficientNet-B0 uses ImageNet weights for actual runs.
Offline tests construct random weights, exercise one forward pass and one tiny
synthetic optimizer step, and verify checkpoint loading and fold isolation. No
real locked-test images are accessed. Training history and raw validation predictions
will be saved under artifacts/baseline/; best/last checkpoints under checkpoints/baseline/.

## Stage 5 one-concept sanity infrastructure

See [the concept GPU run instructions](docs/CONCEPT_SANITY_GPU.md). The separate
configs/concept_sanity.toml trains only atypical_pigment_network on Fold 0, with
training-derived weighted BCE, fixed threshold 0.5, validation AUROC and Macro-F1.
The user reports all four baseline GPU development runs completed externally; no
concept GPU run has been performed locally. Cohort, mappings and splits are unchanged.

## Stage 6 seven-concept infrastructure

The user reports the external Stage 5 sanity run PASSED (AUROC 0.8449, Macro-F1
0.7471, best epoch 15; early stopped after 22 epochs; locked test unused).
See [seven-concept GPU instructions](docs/SEVEN_CONCEPT_GPU.md) for the new
configs/seven_concept.toml and train_seven_concept entry point. One B0 predicts
seven ordered concept logits; training-only weights and validation macro AUROC
selection use the unchanged transfer schedule. Only offline tests ran locally.

## Stage 7 OOF assembly

[Saved best-epoch OOF assembly](docs/OOF_ASSEMBLY.md) is complete: 658 development
cases exactly once, no locked-test overlap. Pooled seven-concept macro AUROC is
0.8146 and macro F1 is 0.6971 at threshold 0.5. Outputs and source provenance are in
artifacts/oof/. No inference, training or CBM diagnosis head was required.

## Stage 8 sequential soft CBM

[Sequential soft CBM results and protocol](docs/SEQUENTIAL_SOFT_CBM.md) use only the
seven OOF probability columns and fixed L2 logistic regression. Four cross-fitted
LR heads produce 658 held-out diagnosis predictions (pooled AUROC 0.8635,
Macro-F1 0.7720, Brier 0.1279, ECE 0.0249). The full-development head is saved
separately and excluded from these metrics. Artifacts are under
artifacts/cbm/sequential_soft/. This is LR-level cross-fitting on the existing
OOF features, not fully nested validation of the entire pipeline. Test images
remain untouched; no hard/oracle model or interventions have been implemented.

Stage 9 hard/oracle CBM results, reproduction and limitations: [docs/HARD_ORACLE_CBM.md](docs/HARD_ORACLE_CBM.md).

Stage 10 cross-fitted correction and forced-value engine: [docs/INTERVENTION_ENGINE.md](docs/INTERVENTION_ENGINE.md). No intervention-policy experiments run.

Stage 11 frozen intervention-policy experiments: [docs/INTERVENTION_POLICIES.md](docs/INTERVENTION_POLICIES.md). Oracle policies are non-deployable; development-only results.
