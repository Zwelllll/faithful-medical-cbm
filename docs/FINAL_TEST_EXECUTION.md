# Stage 15A: final evaluator preparation and checkpoint verification

**Preparation only. No locked-test access or execution has occurred.**
Stage 14 decisions and its manifest are unchanged. This document does not authorize
running the evaluation command below.

## Source state and prerequisites

Current repository commit at preparation: `f38895f5271028d82ccacb6f085c41588cfc0f8c`.
Stage 14 freeze base commit: `4bfe95d05a1347dfe1bac24948f43104e7a08a10`.
The freeze includes the latter commit plus the exact source/config/docs snapshot
recorded in its manifest. Every recorded hash matches at Stage 15A entry; no
documentation-only exceptions or hash substitutions are needed. All existing
frozen files remain unchanged. Stage 15A adds new files only.

These new evaluator files are initially uncommitted and are not present in either
commit above. Before Colab use, review and commit/push Stage 15A yourself, then pin
the resulting exact commit. No commit/push is performed by this task. Do not clone
the older freeze base and expect the new evaluator to be present. Its attestation
records the actual execution commit and hashes every `evaluation/final_*.py` file.

Use a clean checkout of that reviewed implementation commit in Colab. Mount Drive
with `from google.colab import drive; drive.mount('/content/drive')`, install the
project with `python -m pip install -e .`, and retain a compatible CUDA-enabled
torch/torchvision environment. Record resolved package versions. No pretrained
weights are downloaded: the verifier constructs models with pretrained=False and
loads the frozen state dictionaries strictly, without forward passes.

The verification utility checks exact frozen bytes, not just Git commit ancestry.
If cloning changes line endings, stop on the hash mismatch and restore the exact
matching frozen files from the local snapshot. Never update the Stage 14 hashes to
make a different checkout pass. No source/config exception is silently allowed.

## Expected checkpoint inventory

Let `OUTPUTS=/content/drive/MyDrive/Explanable_AI/faithful-cbm/outputs`.
The joint summaries report this Drive root. Other families use the same expected
workflow root; their physical presence has not been verified. A different physical
root is permitted via `--drive-outputs`, without changing any relative selection.

All 16 expected files, relative to OUTPUTS:

| Path | Epoch |
|---|---:|
| checkpoints/baseline/baseline-v1/fold_0/best.pt | 15 |
| checkpoints/baseline/baseline-v1/fold_1/best.pt | 9 |
| checkpoints/baseline/baseline-v1/fold_2/best.pt | 21 |
| checkpoints/baseline/baseline-v1/fold_3/best.pt | 10 |
| checkpoints/seven_concept/seven-concept-v1/fold_0/best.pt | 19 |
| checkpoints/seven_concept/seven-concept-v1/fold_1/best.pt | 18 |
| checkpoints/seven_concept/seven-concept-v1/fold_2/best.pt | 16 |
| checkpoints/seven_concept/seven-concept-v1/fold_3/best.pt | 30 |
| checkpoints/joint_soft/joint-soft-v1/fold_0/best.pt | 13 |
| checkpoints/joint_soft/joint-soft-v1/fold_1/best.pt | 15 |
| checkpoints/joint_soft/joint-soft-v1/fold_2/best.pt | 16 |
| checkpoints/joint_soft/joint-soft-v1/fold_3/best.pt | 14 |
| checkpoints/joint_hard/joint-hard-v1/fold_0/best.pt | 6 |
| checkpoints/joint_hard/joint-hard-v1/fold_1/best.pt | 18 |
| checkpoints/joint_hard/joint-hard-v1/fold_2/best.pt | 14 |
| checkpoints/joint_hard/joint-hard-v1/fold_3/best.pt | 16 |

## Verify before test authorization

This command is safe to run without linking the raw dataset or processed cohort;
it needs frozen development fold IDs, metadata, model definitions and checkpoints:

```bash
python -m faithful_medical_cbm.evaluation.final_verification \
  --root . \
  --drive-outputs /content/drive/MyDrive/Explanable_AI/faithful-cbm/outputs \
  --report /content/drive/MyDrive/Explanable_AI/faithful-cbm/outputs/verification/stage15a_checkpoints.json
```

The JSON records every expected file, existence/failure reason, verified size/SHA256,
family/fold/epoch, architecture, concept order, embedded provenance, false test-use
flags and strict state-dict compatibility. Training/validation IDs must match the
frozen development fold exactly; cohort/split hashes and seed must match. Available
Stage 6/joint run sidecars must agree with checkpoint provenance. Nonfinite parameters
are rejected. No image, cohort row, test ID/label, loader or inference is involved.
It returns exit code 2 unless all 16 pass. Existing reports are never overwritten.
If a failed attempt needs repeating after resolving missing files, use a new report
filename and retain the failed report. Do not replace selected epochs/checkpoints.

Local status is **0/16 verified**, because no real checkpoint binary exists under
the local checkpoint directory. The machine-readable local report is
`docs/stage15a_checkpoint_verification_local.json`. It is blocked and cannot authorize
execution. Temporary random-weight checkpoints used by offline tests are synthetic,
not evidence of verification of any research model.

### Baseline metadata finding

No local baseline `run.json`, `summary.json`, history or `.pt` is available. This
is an unavailable-artifact issue, not proof that provenance was never saved.
The frozen baseline writer saves `epoch`, `model_config.architecture=efficientnet_b0`,
pretraining/unfreeze metadata, `model_state`, complete config and provenance including
`validation_fold`, training/validation IDs, cohort/split/config/source hashes, seed,
Git commit and `locked_test_images_accessed=false`.

Baseline checkpoints do not contain a separate literal `model_type` field. The
recorded architecture tag plus strict loading into the frozen one-logit,
concept-free `BlackBoxEfficientNet` establishes the family; the verifier records
this evidence explicitly. It does not infer fold/epoch from filenames. These fields
are sufficient **if the actual Drive binaries contain the saved schema**. Missing
required fields, incompatible weights or conflicting provenance produce a failed
record with the exact missing field. Do not invent missing provenance or retrain.

## Data layout for the separately authorized session

Only after all 16 checkpoints pass and final execution is separately authorized,
expose the existing frozen files via read-only-intended Drive links:

```text
repository/
  configs/default.toml                         # unchanged
  data/raw/release_v0/ -> DRIVE_DATA/release_v0/
    images/...                                # original dermoscopic paths
  data/processed/stage2a/ -> DRIVE_DATA/stage2a/
    cohort.csv                                # exact frozen bytes, never rebuilt
  data/splits/                                # frozen tracked CSV/JSON files
  artifacts/stage2a/summary.json               # frozen tracked aggregate metadata
  artifacts/cbm/.../full_development_model.json
```

Set DRIVE_DATA to the actual externally stored dataset root. For each link, require
that its target exists and the repository destination does not already exist;
create the symlink using `Path(destination).symlink_to(target, target_is_directory=True)`.
If a destination exists, verify and reuse it rather than delete/replace it blindly.
Do not regenerate the cohort or splits, rename raw paths or put raw images in Git.
The processed path-case normalization is already frozen. Do not manually browse
test images or annotations to debug the setup.

## Explicit final execution command — DO NOT RUN DURING STAGE 15A

After separate authorization, and only with the passed 16-checkpoint attestation:

```bash
python -m faithful_medical_cbm.evaluation.final_evaluator \
  --root . \
  --drive-outputs /content/drive/MyDrive/Explanable_AI/faithful-cbm/outputs \
  --attestation /content/drive/MyDrive/Explanable_AI/faithful-cbm/outputs/verification/stage15a_checkpoints.json \
  --authorize-locked-test --device cuda
```

Without `--authorize-locked-test`, refusal occurs before any file read, loader
import/construction or output write. With authorization, source/head/attestation
hashes and checkpoint bytes are rechecked, every model is loaded before test data,
then the existing frozen LoaderFactory is used with deterministic evaluation
transforms. An execution-start marker reserves the output namespace before exposure.
No fit/optimizer/training calls exist in this evaluator. Existing or partial results
block reruns; investigate a failed execution under a separately documented recovery,
without tuning models or deleting evidence.

## Workflow, outputs and integrity

Outputs persist under `OUTPUTS/artifacts/final_test/`. Stage 15A leaves the actual
final-test namespace absent. `docs/final_output_schemas.json` specifies CSV columns.

Seven prediction tables: black_box, sequential_concept, sequential_soft,
sequential_hard, oracle, joint_soft and joint_hard, each ending `_predictions.csv`.
They retain case_num, targets, probabilities and applicable fold logits/probabilities.
Sequential heads also record exact head inputs. All reference integrity.json;
sequential diagnosis tables join by case_num to the concept table for fold evidence.
Oracle rows are explicitly non-deployable. Integrity links each family/fold to its
epoch, checkpoint hash and run metadata and links each saved full-development head.

Black box/joint diagnosis average four end-to-end probabilities. Sequential heads
receive mean concepts, soft directly and hard at >=0.5. Oracle uses true concepts.
Joint concept averaging never replaces its diagnosis route. Save all seven raw
prediction tables before aggregate metrics. Require 165 unique frozen test IDs,
zero development overlap, 50/115 diagnosis counts, exact concept order, finite
logits and valid probabilities/targets. The loader verifies frozen input hashes.

Then save diagnosis_metrics.csv, concept_metrics.csv, bootstrap_intervals.csv and
paired_bootstrap_differences.csv. Preserve bootstrap_indices.npy and its case-order
JSON before intervals. Use the exact frozen metrics, thresholds and ECE. Undefined
single-class concept AUROC is null with a reason; macro AUROC is null if any of its
seven inputs is undefined. There is no silent omission or post-test choice.

Bootstrap uses PCG64(42), 2,000 replicates, sorted positive/negative pools, 50 positive
then 115 negative draws per replicate with replacement. Identical indices apply to
all six models. Percentile 2.5/97.5 intervals use linear interpolation for the six
frozen metrics. All 15 paired differences use earlier-minus-later frozen model order.
No refitting, ECE interval, patient-independent claim or ranking is introduced.

`interventions/` contains six `soft_..._trajectories.csv` / `hard_..._trajectories.csv`
tables, budget_metrics.csv, intervention_curves.csv, auc_summary.json and definitions.json.
The array-backed FullHeadEngine adapter reuses Stage 10 forced-value/correction code
and the unchanged Stage 11 trajectory/selectors/RNG/aggregation. It uses saved
full-development heads (`validation_fold=-1` means no held-out head), original mean
q, cumulative corrections, detached truth-free active views and post-selection reveal.
Random orders remain paired across soft/hard, 100 repetitions, seed42; budgets0..7.
Raw trajectories are written before their budget metrics. Intervention AUROC ranks
final probabilities as Stage 14 requires; denominators and sample SD/curve areas are
unchanged. Zero-denominator definitions are explicitly recorded.

summary.json plus integrity.json mark completion; integrity records output hashes,
checkpoint attestation, source/head hashes, coverage, package versions and limitations.
An execution_started.json alone is not a completed experiment. Patient independence,
OOF-to-ensemble shift, frozen joint calibration behavior and oracle limitations remain.

## Offline verification and remaining gate

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m unittest tests.test_final_evaluator -v
```

Tests use synthetic arrays, synthetic temporary checkpoint weights, frozen source
and development metadata only. They never invoke the authorization flag, construct
a test loader, open an image or perform a CNN forward pass. New source and preparation
hashes are separately recorded; the Stage 14 manifest is never rewritten.

Before the first test: commit/review this implementation, restore exact frozen bytes
in Colab, obtain a passed 16/16 Drive attestation with sufficient baseline fields,
expose exact existing data paths, and obtain separate final-execution authorization.
**locked_test_accessed=false; final_test_executed=false. Stop after Stage 15A.**
