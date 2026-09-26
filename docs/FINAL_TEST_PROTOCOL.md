# Stage 14: frozen final-test protocol

Development decisions are complete. This stage records the protocol only; the
locked test has not been opened. The machine-readable record is
`artifacts/final_protocol/development_freeze_manifest.json`, with centralized
rules in `configs/final_protocol.json`. D019 records the authorization.

## Frozen inputs and limitations

The cohort contains 823 cases; development contains 658 (198 melanoma, 460 benign);
locked test contains 165 (50 melanoma, 115 benign). These are previously frozen
split-metadata counts, not newly inspected test labels. Splitting is case-level;
patient-level independence remains unverifiable. Cohort, split and concept mappings
are immutable. The manifest records their existing hashes without opening the
cohort or test IDs/annotations. It separately verifies development ID bytes.

Concept order is exactly:

1. atypical_pigment_network
2. regression_structures_present
3. irregular_pigmentation
4. blue_whitish_veil_present
5. atypical_vascular_structures
6. irregular_dots_and_globules
7. irregular_streaks

Every concept and diagnosis decision uses probability **>= 0.5**. No threshold
search, recalibration, loss change, model reselection, refitting, full-development
neural retraining or test-time augmentation is permitted. Use existing deterministic
evaluation preprocessing (224 pixels, ImageNet normalization), model eval mode and
inference without gradients. Only dermoscopic images enter neural models.
Specifically, convert to RGB, resize the full frame to 224x224 with bicubic
interpolation and antialiasing, convert to a 0..1 tensor, then normalize with
means [0.485,0.456,0.406] and standard deviations [0.229,0.224,0.225]. All neural
backbones are the saved ImageNet-initialized torchvision EfficientNet-B0; sequential
and joint diagnosis heads remain linear/logistic without image-feature bypasses.

Stage 13 joint-soft/hard diagnosis predictions were positive for 90.43%/93.16% of
development cases, with specificity 12.83%/8.48%. This severe fixed-threshold and
calibration behavior is accepted unchanged. It must not motivate test-driven tuning.
Selected development results and non-nested sequential evaluation are not unbiased
final-test estimates. No model is designated a winner.

## Selected checkpoints and inference

| Neural family | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Drive outputs-relative namespace |
|---|---:|---:|---:|---:|---|
| Black box | 15 | 9 | 21 | 10 | checkpoints/baseline/baseline-v1 |
| Seven concepts | 19 | 18 | 16 | 30 | checkpoints/seven_concept/seven-concept-v1 |
| Joint soft | 13 | 15 | 16 | 14 | checkpoints/joint_soft/joint-soft-v1 |
| Joint hard STE | 6 | 18 | 14 | 16 | checkpoints/joint_hard/joint-hard-v1 |

Each namespace ends in `fold_N/best.pt`. The manifest records paths reported by
available development summaries, plus the expected logical path for all 16 files.
Drive is authoritative for binaries; no checkpoint is opened at this stage.
Baseline epochs are user-frozen and lack local run metadata. All binary SHA256s
remain explicitly null/unverified. Before any test exposure, verify the exact
16 files, embedded epoch/fold/model/concept order, run/config/training provenance,
and save their hashes in a separate pre-execution attestation. Do not overwrite
this manifest or substitute last checkpoints. A physical Drive-root relocation
does not authorize changing selected weights. Any mismatch blocks execution.

The six final diagnosis models have these exact rules:

| Model | Frozen computation |
|---|---|
| Black-box ensemble | Mean of four fold melanoma probabilities, then >=0.5; no label vote |
| Sequential Soft CBM | Mean of four concept probability vectors, then saved Stage 8 full-development soft LR head |
| Sequential Hard CBM | Same mean concept vector, threshold each element >=0.5, then saved Stage 9 full-development hard LR head |
| Joint Soft CBM | Mean of four end-to-end fold diagnosis probabilities; each fold uses its own soft bottleneck |
| Joint Hard STE CBM | Mean of four end-to-end fold diagnosis probabilities; each fold uses its own exact binary bottleneck internally |
| Oracle concept model | Frozen seven test concept truths into saved Stage 9 full-development oracle LR head; no image-derived concepts |

Full-development heads are `artifacts/cbm/{sequential_soft,sequential_hard,oracle}/full_development_model.json`.
Their exact bytes, labels, ordered features and 658-development-ID membership are
verified and hashed. Fold-specific LR heads must not be used for final test.
The oracle is a **non-deployable oracle upper-bound analysis**, an analytical
reference rather than a guaranteed empirical ceiling.

Sequential LR heads were trained on one held-out model's OOF probabilities per
development case. Final test instead uses four-model averaged probabilities. This
OOF-to-ensemble distribution shift is an accepted limitation and must accompany
results. Joint diagnosis is never recomputed by feeding averaged concepts into
another head. For joint concept evaluation only, average four **soft** concept
probabilities; use them for AUROC and their >=0.5 values for F1.

## Metrics and saved evidence

For all six diagnosis models report AUROC (final probability ranking), Macro-F1
(unweighted mean of class 0/1 F1, zero_division=0), accuracy, sensitivity TP/(TP+FN),
specificity TN/(TN+FP), Brier mean((p-y)^2), and ECE. AUROC is threshold-independent;
F1/accuracy/sensitivity/specificity are threshold-dependent; Brier/ECE describe
calibration. Report factual differences without qualitative rankings.

ECE uses ten equal-width positive-probability bins [0,.1), ...,[.9,1], left-inclusive
and right-exclusive except 1 is included in the last bin. Empty bins contribute
zero. ECE = sum(n_bin/N * abs(mean_probability - observed_melanoma_frequency)).

For sequential, joint-soft and joint-hard concept ensembles, report each concept's
AUROC from mean probabilities and binary Macro-F1 at >=0.5, plus arithmetic means
over all seven concepts. Use frozen test annotations only in the separately
authorized final evaluation. Undefined metrics receive null and a reason; do not
silently omit a concept from a macro mean. Missing/invalid predictions block reporting.

Before aggregate metrics in that future stage, save raw case IDs, all four fold
diagnosis/concept probabilities where applicable, final averaged probabilities,
frozen targets, fixed-threshold predictions, and full provenance. Require exactly
165 unique frozen test IDs for every model. Preserve raw intervention trajectories
or replayable orders and bootstrap indices before reporting summaries. Final output
namespace will be `artifacts/final_test/`; Stage 14 must leave it absent.

## Frozen interventions

Only sequential soft and hard receive intervention experiments. Use their saved
full-development LR heads and original q = four-model mean concept probabilities.
Budgets are k=0..7. Unqueried soft inputs remain q; unqueried hard inputs remain
I(q>=.5); queried inputs become exact revealed truth 0/1. Do not alter oracle or
joint heads or create new joint intervention semantics.

Reuse Stage 11/D016 policies:

- **Random-error ORACLE / NON-DEPLOYABLE:** uniformly permute original threshold
  errors without replacement, seed 42, 100 repetitions (0..99), NumPy PCG64 with
  SeedSequence([42, repetition]). Process lexicographically sorted case_num strings;
  pair random orders across soft/hard. Stop when no errors remain and pad budgets
  with unchanged state and blank selection fields.
- **Confidently-wrong ORACLE / NON-DEPLOYABLE:** select the remaining original
  threshold error maximizing abs(original q-.5), ties by frozen concept order.
  Stop/pad when errors are exhausted.
- **Active uncertainty x downstream impact:** among unqueried concepts maximize
  `(1 - 2*abs(original q-.5)) * abs(p_force_1_current - p_force_0_current)`.
  Recompute impact with the current cumulative state and appropriate head; retain
  original q for uncertainty. Ties use frozen concept order. Selection receives
  no truth or diagnosis target. Reveal only the selected concept afterward. Query
  all seven distinct concepts, including threshold-correct concepts.

This user-frozen D016 formula supersedes the historical entropy/expected-impact
formula in PROJECT_SPEC section 15. Simulated corrections assume perfect truth.
Oracle knowledge does not guarantee better diagnosis. Soft oracle endpoints can
differ from all-query active endpoints because already-correct probabilities remain
uncorrected in oracle policies.

At each budget report all seven diagnosis metrics, cumulative query precision
(errors corrected / actual queries), mean errors corrected (/165), diagnosis change
from k=0 (/165), initially wrong now correct (/initially wrong), and initially
correct now wrong (/initially correct). Denominators come from each model's frozen
k=0 predictions; zero denominators, including zero queries, produce null with reason.
Random summaries are mean +/- sample SD (ddof=1) of 100 per-repetition metrics, not
metrics of averaged predictions. Compute unnormalized trapezoidal area across
0..7 for AUROC/F1/accuracy (unit spacing); summarize per-repetition areas by mean
and sample SD. These are descriptive summaries, not causal effects or deployment utility.

## Prespecified uncertainty

Include 2,000 stratified case-bootstrap replicates, seed 42, an independent
NumPy PCG64(42) stream. Sort positive and negative pools lexicographically by case_num.
Each replicate draws 50 positives followed by 115 negatives, separately with
replacement; concatenate and use identical indices for all six models. Do not refit.
For AUROC, Macro-F1, accuracy, sensitivity, specificity and Brier, use percentile
95% intervals: 2.5th/97.5th percentiles with linear interpolation.

Compute paired differences for all 15 unordered model pairs in the six-model order
above: earlier model minus later model on identical resampled cases, with the same
percentile intervals. These are descriptive, conditional on fitted models, without
multiplicity-adjusted significance or patient-level claims. No ECE, concept or
intervention bootstrap intervals are added in this protocol. Bootstrap outcomes
cannot change any model, threshold, calibration or evaluation choice.

## Safety, source freeze and execution gate

`scripts/freeze_development.py` is a standard-library, metadata-only writer. Its
read allowlist rejects raw images, processed cohort, test IDs, checkpoint binaries
and test outputs. It imports no training, dataset or loader code. It refuses to
overwrite the freeze and refuses known final-output namespaces even if empty.
Tests use metadata and temporary synthetic files only; no CNN forward pass occurs.
The absence check covers recorded namespaces, not arbitrary files on Drive.

The manifest records UTC time, base Git commit/status, exact-byte SHA256 snapshots
of source/config/docs and model metadata, split-metadata hashes and head identities.
The manifest excludes itself from hashing. No commit or push is performed. Preserve
these exact bytes; if a cross-platform checkout converts line endings, verify against
the frozen snapshot before execution rather than silently updating hashes.

Reproduce the checks from repository root:

```powershell
python -m unittest tests.test_final_protocol -v
```

The one-time creation command is `python scripts/freeze_development.py`; rerunning
it after creation intentionally fails. It is not an evaluation entry point.

Before final execution: verify/attest the Drive binaries and missing baseline run
metadata, implement and offline-test an executor against this manifest, and obtain
separate final-stage authorization. Stage 14 does not implement or run that executor.
Freeze decisions remain binding even if later test results are disappointing.
An integrity failure stops execution and must be documented; it cannot authorize
selecting alternative models based on test performance.

**locked_test_accessed=false; final_test_executed=false. Stop after Stage 14.**
