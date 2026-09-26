# Research Decisions

Use this file as an audit trail. Do not rewrite history; append changes as new entries.

## D001 — Primary clinical cohort
Decision: Restrict the primary experiment to melanoma vs benign melanocytic lesions.
Reason: The seven-point checklist is intended for melanocytic lesion assessment and this avoids diagnosis leakage through concept applicability.
Status: Frozen
Test set consulted: No

## D002 — Imaging modality
Decision: Use dermoscopic images only in the primary study.
Reason: Preserve a clean image -> concepts -> diagnosis pathway.
Status: Frozen
Test set consulted: No

## D003 — Backbone
Decision: Use ImageNet-pretrained EfficientNet-B0.
Reason: Appropriate capacity for a small dataset with lower overfitting risk than much larger architectures.
Status: Frozen
Test set consulted: No

## D004 — Diagnosis head
Decision: Use a linear/logistic downstream diagnosis model.
Reason: Preserve direct mathematical intervenability and prevent nonlinear shortcutting.
Status: Frozen
Test set consulted: No

## D005 — Evaluation design
Decision: Use an 80% development / 20% locked test protocol, with OOF concept predictions inside the development pool.
Reason: Avoid downstream training on in-sample concept predictions and preserve a final unbiased test set.
Status: Frozen
Test set consulted: No

## D006 — Explicit primary cohort mapping (Stage 2A, 2026-09-20)
Decision: Apply the user's exact diagnosis mapping in `configs/cohort_mapping.json`.
Positive (`diagnosis_binary=1`): melanoma; melanoma (in situ); melanoma (less than 0.76 mm);
melanoma (0.76 to 1.5 mm); melanoma (more than 1.5 mm).
Negative (`diagnosis_binary=0`): blue nevus; clark nevus; combined nevus; congenital nevus;
dermal nevus; recurrent nevus; reed or spitz nevus.
Excluded: melanoma metastasis; basal cell carcinoma; dermatofibroma; lentigo; melanosis;
miscellaneous; seborrheic keratosis; vascular lesion. Excluded cases are not negative cases.
Reason: Explicit user authorization resolves Stage 1's cohort inclusion questions.
Result: 823 included cases (248 positive, 575 negative), with 188 excluded cases.
Preserve all raw columns and case_num strings. Unrecognized/missing labels raise an error;
no case folding, whitespace stripping, imputation or silent reassignment is permitted.
Status: Frozen by user
Test set consulted: No; original Derm7pt index files were not read or used. No project split exists.

## D007 — Seven binary concept mappings (Stage 2A, 2026-09-20)
Decision: Freeze the following ordered targets. Each target is 1 for the specified raw
values, and 0 for all other audited values. The config explicitly enumerates every audited
raw category, including zeros; unfamiliar/missing values require a new audit/decision.

| Binary target | Raw column | Values mapped to 1 |
|---|---|---|
| atypical_pigment_network | pigment_network | atypical |
| regression_structures_present | regression_structures | every audited value except absent |
| irregular_pigmentation | pigmentation | diffuse irregular; localized irregular |
| blue_whitish_veil_present | blue_whitish_veil | present |
| atypical_vascular_structures | vascular_structures | dotted; linear irregular |
| irregular_dots_and_globules | dots_and_globules | irregular |
| irregular_streaks | streaks | irregular |

Reason: Exact user-approved operational definitions; raw categorical annotations remain
unchanged alongside separate binary targets. No validity masks or target proxies are created.
The original diagnosis, seven_point_score, clinical image reference and other metadata are
retained for audit purposes only, not approved as model inputs. Primary input remains dermoscopy.
Status: Frozen by user
Test set consulted: No

## D008 — Processed image paths and grouping limitation (Stage 2A, 2026-09-20)
Decision: Preserve the raw derm column and store a separate derm_path relative to the release's
images directory. For case_num 816 only, record FCl/Fcl068.jpg as FCL/Fcl068.jpg in derm_path.
No raw file or raw field is changed. Other unexpected path mismatches fail validation.
Reason: Stage 1 verified the exact filename casing; the user explicitly authorized processed-only normalization.
Grouping remains unresolved at patient/lesion level: case_num is a complete case key, while
case_id is sparse and no patient linkage exists. Before Stage 2B, accept/document the
specification's case-level fallback or obtain authoritative linkage. No grouping IDs are invented.
Status: Path normalization frozen; patient/lesion grouping limitation remains open
Test set consulted: No

## D009 — Frozen case-level split and development folds (Stage 2B, 2026-09-20)
Decision: Use deterministic CASE-LEVEL diagnosis-stratified splitting because no
authoritative patient identifier or reliable lesion linkage is available. The user
explicitly accepts this methodological limitation for the study. This resolves D008's
choice of splitting strategy, not the absence of patient/lesion linkage.
case_num is the permanent sample identifier only; grouping by this unique case number
does not provide patient-level protection. Patient-level independence cannot be verified.

Method: Read only the processed Stage 2A cohort, verify its Stage 2A hash and class counts,
and sort case_num strings lexicographically. Use scikit-learn train_test_split with
test_size=0.20, shuffle=True, stratify=diagnosis_binary, random_state=42. The configured
seed is frozen at 42. Round the test size up to 165 cases, leaving 658 development cases.
The first generated split is permanent; no alternative seeds or class balances were tried.
Original Derm7pt train/valid/test index files and raw metadata/images are not read.

Within the sorted development IDs only, use StratifiedKFold(n_splits=4, shuffle=True,
random_state=42), stratified by diagnosis_binary. Fold numbers are 0-based. Each
development ID occurs in exactly one validation fold; its training folds are the other
three validation folds. No locked-test ID may appear in any development fold.

| Subset | Total | Positive | Negative |
|---|---:|---:|---:|
| Development | 658 | 198 | 460 |
| Locked test | 165 | 50 | 115 |
| Fold 0 validation | 165 | 50 | 115 |
| Fold 1 validation | 165 | 50 | 115 |
| Fold 2 validation | 164 | 49 | 115 |
| Fold 3 validation | 164 | 49 | 115 |

Permanent files: data/splits/development_ids.csv, test_ids.csv,
development_folds.csv and split_metadata.json. Metadata includes class counts for
both training and validation parts of each fold, input/output hashes, versions,
seed, exact methods and the patient-independence limitation. Reruns verify existing
files without regenerating or rewriting them; partial, altered or incompatible
frozen outputs fail rather than being replaced.

Status: Frozen. The locked test set must remain unused during development, including
model selection, tuning, calibration, early stopping, threshold selection and debugging.
It is reserved for the final evaluation after the development freeze.
Test set access: IDs and binary diagnosis were used only to create and verify this
authorized split and report its counts. No test images, model predictions or performance
were accessed. No models, datasets/loaders or training were implemented.

## D010 — Stage 5 one-concept development sanity infrastructure (2026-09-23)
Decision: Implement only atypical_pigment_network on development Fold 0, using its
unchanged Stage 2A target. Use training-only negative/positive weighting (364/129),
a fixed 0.5 probability threshold (>= is positive), and validation AUROC for early
stopping/checkpoint selection. Report validation Macro-F1 as well as AUROC and losses.
The independent concept config retains the baseline's original transfer schedule;
external baseline scores did not influence configuration choices. No seven-head,
CBM diagnosis, OOF or intervention work is authorized in this stage.
Status: Infrastructure implemented; real concept GPU run pending.
Context: User reports completed external baseline development AUROCs 0.8833, 0.8708,
0.9207, 0.8673 (mean 0.8855, sample SD 0.0244). External run artifacts were not supplied
locally for verification; these are development results only.
Test set access: No images or performance accessed. Only frozen exclusion IDs are
read by existing loader integrity checks. Patient-level independence remains unverifiable.

## D011 — Stage 6 seven-concept development infrastructure (2026-09-24)
Context: The user considers Stage 5 PASSED after external GPU training: Fold 0
atypical_pigment_network best validation AUROC 0.8449, best Macro-F1 0.7471,
best epoch 15, 22 epochs completed, early stopped, locked_test_used=false.
These are user-reported development results; external run artifacts were not supplied
locally for verification. This resolves D010's pending external sanity run.
Decision: Extend the concept output to all seven D007 targets in exactly that order,
sharing one EfficientNet-B0. Per-concept negative/positive weights use only the selected
fold's training labels. Keep D010's transfer schedule unchanged. Use the arithmetic
mean of all seven validation AUROCs for selection and early stopping, report each
concept's AUROC/Macro-F1 and the seven-concept mean F1, and fix probability >=0.5
as positive. Fail on any single-class concept subset instead of omitting concepts
from the selection metric. Save raw per-epoch validation outputs before metrics.
Status: Infrastructure only; one selected fold per GPU command, folds 0-3 supported.
No full local training, OOF assembly, CBM diagnosis, interventions or threshold tuning.
Test set access: No images or performance accessed; existing manifest exclusion IDs
are used only for integrity checks. Cohort, mappings and split files are unchanged.

## D012 — Stage 7 saved best-epoch OOF assembly (2026-09-24)
Context: Four external Stage 6 development runs are complete. Uploaded run/summary
and validation CSVs verify best epochs 19,18,16,30 and their reported fold scores.
Decision: Assemble exactly one held-out prediction per development case from each
fold's summary-selected saved CSV. Verify frozen IDs, labels, hashes, concept order,
run training/validation provenance and best-epoch metrics before writing outputs.
Use processed diagnosis_binary only, fixed probability >=0.5, and no model inference,
fold averaging or threshold optimization. Record source hashes and Git commit.
Result: 658 unique development rows (165/165/164/164), zero test overlap or missing
cases. Pooled macro OOF AUROC 0.8146319458412441 and Macro-F1 0.6970823357130504.
These pooled metrics differ from the mean of fold metrics. Source commit:
57406ca22157773fb568ed462e57113d8db6bbb4, run seven-concept-v1.
Status: Stage 7 complete. Saved run provenance checked; checkpoint weights not loaded.
No sequential CBM head, interventions or locked-test evaluation was implemented.
Test set access: Exclusion IDs only; no test images, targets or performance inspected.

## D013 — Stage 8 cross-fitted sequential soft CBM (2026-09-24)
Decision: Use exactly the seven frozen-order OOF probability columns as the only
inputs to scikit-learn LogisticRegression. Fixed L2, C=1, lbfgs, intercept, no class
weighting/scaling, max_iter=1000, tol=1e-8, random_state=42. No search or tuning.
Cross-fit four LR heads on the existing validation_fold assignments; save each
case's held-out score/probability and report AUROC, binary Macro-F1, accuracy,
sensitivity, specificity, Brier and positive-probability ECE (10 equal-width bins).
Threshold is fixed at probability >=0.5. Coefficients are descriptive, not causal
concept importance. After CV evaluation, fit/save a separately labelled
FULL-DEVELOPMENT SOFT CBM HEAD; never use it for reported development predictions.
Result: 658 unique cross-fitted diagnosis predictions; pooled AUROC
0.8635375494071147, Macro-F1 0.772002772002772, Brier 0.12792545989565157,
ECE 0.024853814067755744. Configuration was chosen before these results.
Limitation: This is LR-head cross-fitting on fixed OOF features, not fully nested
CNN-plus-LR validation. Concept early stopping used held-out concept labels, and
CNNs producing LR-training features can include cases in the LR-validation fold.
Report these as development diagnostics, not an independent test estimate. Existing
patient-independence and OOF-to-ensemble shift limitations also remain.
Status: Stage 8 complete; no hard/oracle CBM, interventions or CNN training/inference.
Test set access: Exclusion IDs only, no test labels/images/performance. Cohort,
splits, concept mappings and OOF predictions were not modified.

## D014 — Stage 9 hard and oracle concept heads (2026-09-24)
Decision: Reuse D013 logistic-regression settings and folds without tuning. Hard
features are the seven OOF probabilities thresholded at >=0.5; oracle features are
exactly the frozen true concept targets, checked against the processed cohort.
Cross-fit four diagnosis heads per model, then save separate full-development
heads that never contribute to reported metrics. Thresholds remain 0.5; ECE uses
the same ten bins as Stage 8. Coefficients are descriptive, not causal importance.
Result: Each model covers 658 unique development cases, with zero test overlap.
Hard pooled AUROC 0.8401570048309178, Macro-F1 0.7503843466107617.
Oracle pooled AUROC 0.9217226613965744, Macro-F1 0.80236080115654.
Stage 8 metrics are read unchanged for the comparison; source hashes verified.
Limitations: D013 non-nested development evaluation applies to hard/soft; patient
independence remains unverifiable. Oracle is a true-concept analytical reference,
not a deployable predictor or guaranteed empirical ceiling. Gaps are descriptive.
Status: Stage 9 complete. See HARD_ORACLE_CBM.md for fold metrics and coefficients.
No interventions, CNN training/inference, or test labels/images/performance access.
Cohort, splits, mappings, OOF and Stage 8 artifacts remain unchanged.

## D015 — Stage 10 cross-fitted intervention engine (2026-09-24)
Decision: Reuse each development case's saved soft/hard fold-specific LR head,
verifying held-out membership and baseline score/probability reproduction with
absolute tolerance 1e-12, relative tolerance 0. No model retraining or inference.
Soft corrections replace selected OOF probabilities with frozen binary targets;
hard corrections replace selected thresholded inputs. All other inputs stay fixed.
Forced binary values are separately labelled counterfactuals, not clinician corrections.
Separate a detached truth-free selection view from the target-reading executor.
Only selected targets are fetched after selection; no policies are implemented.
Concept error means probability >=0.5 disagrees with the frozen target. Soft
corrections may also change threshold-correct probabilities by snapping to 0/1.
Result: Per model, 4,606 correction rows and 9,212 forced-value rows. Both have
967 wrong concepts. Wrong-concept correction mean absolute probability effects
are 0.1503743994 (soft) and 0.1401182101 (hard); flips 197/967 and 168/967.
These are descriptive model-input effects, not causal importance or necessarily
clinical improvements. D013/D014 evaluation limitations remain unchanged.
Status: Stage 10 complete; no policy experiments or intervention-aware training.
Full-development heads never used. Test exclusion IDs only; no test targets/images.
Stages 7-9 artifacts, cohort, splits and mappings unchanged. See INTERVENTION_ENGINE.md.

## D016 — Stage 11 frozen intervention policies (2026-09-24)
User-authorized Stage 11 definition supersedes PROJECT_SPEC section 15's earlier
entropy/expected-impact formula for this experiment: active uncertainty is
1 - 2*abs(original q - 0.5); impact is abs(p_force_1 - p_force_0) evaluated against
the current cumulative state using the case's cross-fitted soft/hard head. Select
maximum uncertainty*impact, ties by frozen concept order, then reveal target.
The detached active-selection API receives no ground truth or diagnosis labels.
Random-error and confidently-wrong are explicitly ORACLE / NON-DEPLOYABLE.
Both select only original threshold errors; confidence is abs(original q - 0.5).
Random policy uses seed 42, 100 independent PCG64 SeedSequence([42,rep]) streams;
random orders paired across soft/hard. Stop/pad oracles when no errors remain.
Evaluate budgets 0..7, fixed diagnosis/concept thresholds 0.5, Stage 8 ECE bins.
Report cumulative query precision, improvement and harm with separate denominators.
Use sample SD across random repetitions; raw trapezoidal areas over 0..7 are
solely descriptive. Oracle error knowledge does not guarantee optimal diagnosis.
Results: active k=7 AUROC soft 0.9046277997 / hard 0.9028491436; accuracy soft
0.8267477204 / hard 0.8191489362. Soft active improves 71/118 initially wrong
and harms 67/540 initially right diagnoses; hard improves 78/133, harms 64/525.
Soft oracles leave already-correct probabilities untouched, unlike all-query active;
thus their endpoints differ. No winner or causal importance interpretation.
Storage: exact random selection orders plus frozen source hashes permit complete
trajectory replay; deterministic policies store all steps. Full curves and AUCs
are in artifacts/interventions/policies/; see INTERVENTION_POLICIES.md.
Status: Stage 11 complete. No model retraining, intervention-aware heads, joint
CBMs, threshold tuning, test images/labels, or changes to Stage 7-10 artifacts.
Existing non-nested development and patient-independence limitations remain.

## D017 — Stage 12 joint soft/hard infrastructure (2026-09-24)
Decision: Reuse ImageNet EfficientNet-B0 and seven frozen ordered concept logits.
Joint soft feeds sigmoid probabilities to a single linear 7-to-1 diagnosis layer
with intercept. Joint hard feeds exact binary >=0.5 concepts; backward uses a
sigmoid straight-through surrogate, explicitly JOINT HARD CBM WITH STE. Threshold
itself is not differentiable. No hidden diagnosis layers or image-feature bypass.
Loss is training-fold-weighted seven-concept BCE + unweighted diagnosis BCE,
weights exactly 1.0/1.0. Keep image 224, batch 32, seed 42, AdamW decay 0.0001,
3 head-only epochs LR 0.001 then last 3 feature children LR 0.0001, max 30,
patience 7. Both heads train in both phases; frozen feature blocks remain eval.
Select/stop only on validation diagnosis AUROC, not concept or composite scores.
Report seven concept AUROCs/F1s, concept macro means, diagnosis metrics including
frozen ECE, and all three losses. Persist raw validation rows before metrics.
Separate joint_soft/joint_hard namespaces with overwrite guards and full provenance.
Checkpoints reject incompatible model types/orders. Full entry point requires CUDA.
Status: Infrastructure complete; eight offline tests pass. Only synthetic forward,
optimizer and checkpoint tests ran; no actual GPU training or weight download.
Fold 0 weights derive only from 493 training cases; see joint_preflight.json and
JOINT_CBM_GPU.md for exact values, Drive setup and first-run commands.
No locked-test loaders/images/labels, cohort/split/mapping changes, sequential
retraining, joint OOF generation, or changes to Stage 7-11 artifacts. Existing
case-level patient-independence limitation remains. GPU execution is pending.

## D018 — Stage 13 saved joint OOF assembly (2026-09-26)
Decision: Assemble saved best-epoch validation CSVs only: joint soft 13/15/16/14,
joint hard STE 6/18/14/16. Verify source hashes, run/summary type/order/test flags,
exact frozen folds, training exclusions and processed/Stage 7 truth. Each joint
OOF table covers all 658 development cases once, with zero test overlap.
Joint concept AUROC uses saved soft probabilities; F1 uses >=0.5. Diagnosis AUROC
uses logits; all classification metrics keep >=0.5 and the frozen ten-bin ECE.
Copy frozen sequential metrics unchanged; distinguish pooled values from fold means.
Result: joint soft pooled diagnosis AUROC 0.8695103206, Macro-F1 0.3574513128;
joint hard STE AUROC 0.8365612648, Macro-F1 0.3139724823. Joint concept macro AUROC/F1:
soft 0.8094709563/0.6871424813, hard 0.7944875641/0.6852945588.
Diagnosis predictions >=0.5: 90.4255% soft, 93.1611% hard, compared with 30.0912%
melanoma prevalence. Fixed-threshold classification/calibration remain unresolved;
no threshold optimization or recalibration was performed or authorized by this stage.
Status: Stage 13 complete. No training/inference/checkpoint access, joint interventions
or locked-test image/label use. Previous Stage 7-12 artifacts remain unchanged.
Validation-selected epochs, non-nested sequential evaluation and unverified patient
independence limit interpretation. See JOINT_OOF_COMPARISON.md and artifacts/joint_oof/.

## D019 — Stage 14 development freeze and final-test protocol (2026-09-26)
User freezes all development choices before test access. Cohort 823, development
658, locked test 165 (50 melanoma/115 benign); patient independence unverifiable.
Concept order/mappings/splits unchanged; every concept/diagnosis threshold >=0.5.
Freeze neural best epochs by fold: black box 15/9/21/10; seven concepts 19/18/16/30;
joint soft 13/15/16/14; joint hard STE 6/18/14/16. Average four diagnosis probabilities
for black box and each joint family. Sequential models use four-model mean concept
probabilities, soft directly and hard thresholded, through saved full-development
LR heads. Oracle uses true concepts and its saved full-development head, non-deployable.
Joint concept metrics average soft probabilities; never rebuild joint diagnosis from
averaged concepts. Accept OOF-to-ensemble shift and severe joint fixed-threshold
behavior unchanged: no recalibration, new loss, retraining or threshold optimization.
Final metrics: AUROC, Macro-F1, accuracy, sensitivity, specificity, Brier and frozen
10-bin ECE; concept per-target AUROC/F1 and arithmetic macro means. No winner labels.
Only sequential soft/hard interventions, D016 policies unchanged, original ensemble
q and full-development heads; random seed 42/100 repetitions, budgets 0..7.
Prespecify case-stratified bootstrap 2000/seed42, 50 positive and 115 negative draws,
paired indices for all models, percentile 95% intervals and all 15 paired differences;
no refitting. Exact definitions/denominators are in FINAL_TEST_PROTOCOL.md and config.
Manifest captures base Git/source/config/model-metadata/head hashes. Cohort/test hash
values come from split metadata without opening test IDs or annotations. Neural
binaries remain on Drive, unopened and unverified; binary attestation and baseline
run verification are mandatory before a separately authorized final execution.
Status: protocol only. locked_test_accessed=false; final_test_executed=false.
No CNN inference/training, test metrics, model/head/split edits, commit or push.
