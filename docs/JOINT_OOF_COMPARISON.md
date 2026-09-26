# Stage 13: joint OOF assembly and development comparison

Completed 2026-09-26 from saved validation CSVs only. No training, inference, checkpoint access, threshold optimization, recalibration or locked-test evaluation.

## Exact sources

| Model | Fold | Best epoch | Validation rows |
|---|---:|---:|---:|
| joint_soft | 0 | 13 | 165 |
| joint_soft | 1 | 15 | 165 |
| joint_soft | 2 | 16 | 164 |
| joint_soft | 3 | 14 | 164 |
| joint_hard_ste | 0 | 6 | 165 |
| joint_hard_ste | 1 | 18 | 165 |
| joint_hard_ste | 2 | 14 | 164 |
| joint_hard_ste | 3 | 16 | 164 |

Soft sources: artifacts/joint_soft/joint-soft-v1/fold_N/. Hard sources: artifacts/joint_hard/joint-hard-v1/fold_N/. Each summary-selected validation_epoch_NNN.csv was checked against the organization inventory hash, run/summary model type, concept order, frozen fold membership, training exclusion and locked_test_used=false. Source run, summary, history and prediction hashes are recorded. The saved best-epoch diagnosis/concept metrics were reproduced for validation.

## Pooled diagnosis results

Each row below is the pooled 658-case development result. Sequential and oracle values were copied from their existing hash-verified metrics, not recomputed or replaced.

| Model | AUROC | Macro-F1 | Accuracy | Sensitivity | Specificity | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| sequential_soft | 0.863538 | 0.772003 | 0.820669 | 0.595960 | 0.917391 | 0.127925 | 0.024854 |
| sequential_hard | 0.840157 | 0.750384 | 0.797872 | 0.601010 | 0.882609 | 0.133517 | 0.045250 |
| joint_soft | 0.869510 | 0.357451 | 0.384498 | 0.979798 | 0.128261 | 0.244310 | 0.302290 |
| joint_hard_ste | 0.836561 | 0.313972 | 0.351064 | 0.969697 | 0.084783 | 0.239970 | 0.297078 |
| oracle | 0.921723 | 0.802361 | 0.837386 | 0.691919 | 0.900000 | 0.104316 | 0.037424 |

| Descriptive difference | AUROC | Macro-F1 |
|---|---:|---:|
| joint_soft_minus_sequential_soft | +0.005973 | -0.414551 |
| joint_hard_ste_minus_sequential_hard | -0.003596 | -0.436412 |
| oracle_minus_joint_soft | +0.052212 | +0.444909 |
| oracle_minus_joint_hard_ste | +0.085161 | +0.488388 |

AUROC measures ranking independently of the 0.5 decision threshold. Macro-F1, accuracy, sensitivity and specificity use that fixed threshold. The joint models retain discrimination but classify most benign cases as positive at 0.5. No threshold was adjusted to repair this behavior. Brier and ECE describe probability quality and were not fitted or optimized.

## Fold means versus pooled results

| Model | Mean fold diagnosis AUROC | Pooled diagnosis AUROC | Mean fold diagnosis F1 | Pooled diagnosis F1 | Mean fold concept macro AUROC | Pooled concept macro AUROC | Mean fold concept macro F1 | Pooled concept macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| joint_soft | 0.870798 | 0.869510 | 0.356905 | 0.357451 | 0.811899 | 0.809471 | 0.686183 | 0.687142 |
| joint_hard_ste | 0.841257 | 0.836561 | 0.313842 | 0.313972 | 0.806977 | 0.794488 | 0.684243 | 0.685295 |

Fold means are unweighted arithmetic means of four best-fold metrics. Pooled metrics are recomputed across all 658 saved predictions. They are not interchangeable; fold-specific score scales and cross-fold ranking can change pooled AUROC. No locked-test results exist. Per-fold results are retained in each joint_*_metrics.json.

## Pooled concept comparison

Each cell is AUROC / binary Macro-F1. Joint concept AUROC uses saved soft probabilities; F1 uses probability >=0.5 for both models. Stage 7 frozen metrics are copied as recorded (its AUROC used logit ranking); they were not changed. Source fold summaries used logit ranking and are verified separately.

| Concept | Stage 7 sequential | Joint soft | Joint hard STE |
|---|---:|---:|---:|
| atypical_pigment_network | 0.797773 / 0.637191 | 0.796610 / 0.630507 | 0.757908 / 0.629132 |
| regression_structures_present | 0.772845 / 0.683512 | 0.758888 / 0.667351 | 0.743198 / 0.664799 |
| irregular_pigmentation | 0.798309 / 0.691121 | 0.801163 / 0.694448 | 0.783772 / 0.701043 |
| blue_whitish_veil_present | 0.925411 / 0.817781 | 0.922629 / 0.815210 | 0.904465 / 0.793657 |
| atypical_vascular_structures | 0.809242 / 0.672637 | 0.786294 / 0.642193 | 0.810217 / 0.618779 |
| irregular_dots_and_globules | 0.793529 / 0.702262 | 0.799033 / 0.646171 | 0.779863 / 0.673060 |
| irregular_streaks | 0.805314 / 0.675072 | 0.801680 / 0.714118 | 0.781990 / 0.716592 |
| macro_mean | 0.814632 / 0.697082 | 0.809471 / 0.687142 | 0.794488 / 0.685295 |

## Probability and threshold diagnostics

True melanoma prevalence is 198/658 = 30.0912%. These are descriptive probabilities, without recalibration.

| Model | Mean probability | Mean: melanoma | Mean: benign | All >=0.5 | Melanoma >=0.5 | Benign >=0.5 | Probability range |
|---|---:|---:|---:|---:|---:|---:|---|
| joint_soft | 0.537955 | 0.577621 | 0.520881 | 90.4255% | 97.9798% | 87.1739% | 0.469446–0.654237 |
| joint_hard_ste | 0.547158 | 0.605527 | 0.522034 | 93.1611% | 96.9697% | 91.5217% | 0.448620–0.704081 |

Both joint models produce probabilities in a narrow range centered above the fixed cutoff. Consequently, high sensitivity coexists with low specificity. This describes the observed outputs, not the cause of the training behavior. No new loss, threshold, calibration or model-selection decision was made.

Calibration-bin data: calibration_bins.csv and each metrics JSON. ECE uses ten equal-width positive-probability bins, left-inclusive/right-exclusive except the final bin includes 1. Empty bins contribute zero. ECE is sum(n_bin/N * abs(mean_probability - fraction_positive)).

## Integrity, schema and artifacts

Both tables contain 658 unique case_num strings, exact frozen development coverage and fold counts 165/165/164/164, with zero test overlap. Each case comes from its own held-out fold and is excluded from that run's training IDs. Diagnosis labels match the processed frozen development cohort; concepts match both that cohort and Stage 7 truth. All scores/probabilities are finite, probabilities in [0,1], and sigmoid consistency is checked. Hard inputs are binary and exactly probability >=0.5.

OOF schema: case_num, validation_fold, diagnosis_binary, diagnosis_logit, diagnosis_probability, then seven concept_target, seven concept_logit and seven concept_probability columns, in frozen order. Hard additionally has seven concept_hard_input columns. Soft has 26 columns; hard 33. IDs use frozen lexicographic order.

New namespace artifacts/joint_oof/:

- joint_soft_oof.csv and joint_hard_oof.csv
- joint_soft_metrics.json and joint_hard_metrics.json (pooled, per-fold, fold means and calibration bins)
- joint_concept_metrics.csv and concept_model_comparison.csv
- development_model_comparison.csv
- calibration_diagnostics.csv and calibration_bins.csv
- summary.json and integrity.json

New implementation: src/faithful_medical_cbm/evaluation/joint_oof.py; config: configs/joint_oof.json; tests: tests/test_joint_oof.py. Documentation updates include this report, RESULTS_INDEX, README and decision D018. The namespace .gitattributes preserves CSV/JSON LF bytes; .gitignore permits only this new lightweight result namespace.

Source and output hashes, source Git commits and local package versions are recorded. Previous Stage 7-12 outputs remain unchanged. No checkpoint, image or training module is needed by the assembler. The existing processed-cohort reader filters to development records before examining targets. Test exclusion IDs only are used.

## Reproduction and tests

```powershell
$env:PYTHONPATH = 'src'
python -m faithful_medical_cbm.evaluation.joint_oof --config configs/joint_oof.json
python -m unittest tests.test_joint_oof -v
```

Existing outputs are never overwritten. For a separate verification directory, use assemble(config_path, output_dir) from Python. Five tests cover exact epoch selection, type/order/flags, fold isolation, source tampering, duplicate/missing/wrong-fold/test IDs, truth and numerical validation, hard threshold consistency, both schemas, pooled metrics, ECE/diagnostics, comparison arithmetic and unchanged sequential values/hashes. Tests use saved tabular files only.

## Before freezing development decisions

No assembly blocker remains. Fixed-0.5 diagnosis behavior and calibration of the joint models are unresolved scientific issues; this stage only documents them. Any later threshold/calibration or objective change needs a separately specified development protocol and must not use the locked test.

Joint best epochs were selected using these held-out validation diagnosis labels, so these are selected development results rather than unbiased final-test estimates. Sequential heads use the previously documented non-nested CNN/LR protocol. Patient-level independence cannot be verified. These limits should accompany all comparisons; no winner/rank or deployment claim is assigned. No joint-model intervention experiments or next-stage work was performed.
