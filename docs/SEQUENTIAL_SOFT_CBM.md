# Stage 8: sequential soft CBM diagnosis head

Completed using the frozen 658-case Stage 7 OOF table. No CNNs were trained,
concept inference rerun, or images opened. No locked-test labels or images were
inspected; exclusion IDs are used only for integrity checks. Earlier cohort,
split, mapping, concept predictions and model configurations remain unchanged.

## Fixed inputs and model

`configs/sequential_soft.json` specifies seven probability features, in this order:

1. atypical_pigment_network_probability
2. regression_structures_present_probability
3. irregular_pigmentation_probability
4. blue_whitish_veil_present_probability
5. atypical_vascular_structures_probability
6. irregular_dots_and_globules_probability
7. irregular_streaks_probability

`diagnosis_binary` is the target. An explicit allowlist extracts only these seven
columns into a float64 matrix. Logits, ground-truth concepts, images, case/fold IDs
and patient metadata cannot enter the LR feature matrix. No scaling, clipping,
label-derived feature modification, feature selection or interactions are used.

scikit-learn LogisticRegression: L2, C=1.0, solver=lbfgs, fit_intercept=true,
class_weight=null, max_iter=1000, tol=1e-8, random_state=42. This was specified
before evaluation, with no hyperparameter search. On sklearn 1.8+ the constructor
uses l1_ratio=0 instead of the deprecated penalty='l2'; the objective is identical.
Actual constructor arguments and package versions are saved. Convergence warnings
are fatal rather than prompting silent retries or configuration changes. All five
heads converged (24/27/24/26 iterations for folds, 23 for the full-development head).

The model is `score = intercept + sum(coefficient[j] * probability[j])`, with
`P(melanoma) = sigmoid(score)`. Coefficients are conditional descriptive model
parameters, **not causal concept importance**, particularly with correlated predictors.

## Evaluation and calibration

Use the existing validation_fold values, never a new split. For each diagnosis
fold, fit LR only on the other three folds' OOF rows and predict the held-out fold.
All 658 reported diagnosis predictions come from heads that excluded their cases.
Save raw scores/probabilities before computing metrics. After all four fold and
pooled evaluations finish, fit one additional head on all 658 OOF rows. It is labelled
FULL-DEVELOPMENT SOFT CBM HEAD and is never called to predict development outcomes
in this workflow. It is reserved for later use after development decisions freeze.

Threshold is fixed: probability >=0.5 predicts melanoma. AUROC uses LR decision
scores. Macro-F1 averages class-0 and class-1 F1 with zero_division=0; sensitivity
is TP/(TP+FN); specificity is TN/(TN+FP); Brier is mean((p-y)^2).

ECE is positive-class probability calibration with 10 equal-width bins. Bin j is
[j/10,(j+1)/10), except the last includes 1. Boundaries enter the bin to their right,
and empty bins contribute zero. ECE = sum_b(n_b/N)*abs(mean(p_b)-mean(y_b)).
This is not a max-class-confidence ECE. Full per-bin counts, mean probabilities,
observed melanoma fractions and contributions are in calibration_bins.json.
No calibration model or diagnosis threshold was fitted/optimized.

| Subset | AUROC | Macro-F1 | Accuracy | Sensitivity | Specificity | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fold 0 | 0.8628 | 0.7424 | 0.8000 | 0.5400 | 0.9130 | 0.1328 | 0.0705 |
| Fold 1 | 0.8464 | 0.7800 | 0.8242 | 0.6200 | 0.9130 | 0.1336 | 0.0694 |
| Fold 2 | 0.9070 | 0.8000 | 0.8354 | 0.6939 | 0.8957 | 0.1124 | 0.0871 |
| Fold 3 | 0.8676 | 0.7623 | 0.8232 | 0.5306 | 0.9478 | 0.1328 | 0.0737 |
| Pooled cross-fitted | 0.8635 | 0.7720 | 0.8207 | 0.5960 | 0.9174 | 0.1279 | 0.0249 |

Pooled confusion counts: TP=118, TN=422, FP=38, FN=80. Rows per fold:
165/165/164/164. Diagnosis counts are 198 positive and 460 negative. Pooled ECE
is recomputed on all predictions, not averaged from fold ECEs; opposite calibration
gaps across folds can cancel when pooling. Interpret the per-fold results too.

## Saved model parameters

Rounded below; JSON files store full precision in both ordered arrays and named mappings.
Each fold JSON includes exact training/validation IDs, source OOF hash, configuration,
Git commit, working-tree-dirty flag and hashes of the source used for the run.

| Parameter | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Full development |
|---|---:|---:|---:|---:|---:|
| Intercept | -3.385663 | -3.438576 | -3.201643 | -3.522269 | -3.443926 |
| atypical_pigment_network | 1.363369 | 1.030123 | 1.124458 | 0.569099 | 1.106676 |
| regression_structures_present | 1.395001 | 1.015108 | 1.191636 | 1.152950 | 1.220063 |
| irregular_pigmentation | 1.443310 | 1.675644 | 1.700838 | 1.407226 | 1.670254 |
| blue_whitish_veil_present | 1.304011 | 0.947688 | 0.878391 | 1.190392 | 1.089406 |
| atypical_vascular_structures | 0.178912 | 0.516663 | 0.356909 | 0.411279 | 0.345055 |
| irregular_dots_and_globules | 1.765052 | 2.078498 | 1.752367 | 2.412968 | 2.031766 |
| irregular_streaks | 0.199147 | 0.477878 | 0.322875 | 0.314024 | 0.217334 |

## Reproduction and artifacts

Files added: configs/sequential_soft.json; src/faithful_medical_cbm/models/soft_cbm.py;
src/faithful_medical_cbm/evaluation/diagnosis_metrics.py;
src/faithful_medical_cbm/training/train_sequential_soft.py;
tests/test_sequential_soft.py; this document; the artifacts listed below and their
local .gitattributes. Files updated: README.md, docs/DECISIONS.md, .gitignore.

Validation: 7 Stage 8 tests and 6 OOF regression tests passed; source compilation
and git diff --check passed. All 658 saved predictions were independently reconstructed
from their saved fold coefficients, and all recorded input/output hashes matched.

From the repository root after editable package installation:

```bash
python -m faithful_medical_cbm.training.train_sequential_soft --config configs/sequential_soft.json
```

Outputs default to artifacts/cbm/sequential_soft/. Existing run contents are rejected,
never overwritten. Use --output-dir with a fresh directory for an explicit reproduction.
No GPU, weights or raw images are required. The processed cohort and frozen split
manifests are needed to check IDs and diagnosis truth, not as model features.

- cross_fitted_predictions.csv: case_num, validation_fold, diagnosis_binary,
  decision_score, melanoma_probability, predicted_diagnosis.
- fold_metrics.csv and pooled_metrics.json: held-out development metrics only.
- fold_0_model.json through fold_3_model.json: cross-fitted evaluation heads.
- full_development_model.json: separately labelled full-development head with no
  validation IDs and used_for_development_metrics=false.
- calibration_bins.json: per-fold and pooled reliability-bin statistics.
- run_config.json: the fixed Stage 8 configuration.
- integrity.json: input/output hashes, seven-feature allowlist, exact coverage,
  fold isolation, unchanged probabilities, test exclusion and run/source provenance.

The Stage 7 CSV hash is checked against its integrity report. The 658 unique IDs,
four assignments and diagnosis labels are checked against frozen metadata. Binary
labels and finite [0,1] probability features are mandatory. No feature imputation
is performed. Input hashes are rechecked after fitting; output hashes are recorded.

## Interpretation limits and stage boundary

This follows the requested **LR-head cross-fitting on the existing OOF table**.
It is not fully nested end-to-end CNN-plus-LR validation: concept-model early stopping
used held-out concept labels, and CNNs producing LR-training features may have trained
on cases belonging to the LR-validation fold. The LR fitting excludes its validation
cases, but these scores are development diagnostics, not a final independent test
estimate. No CNN retraining or resplitting was introduced to change that protocol.

Patient-level independence remains unverifiable. The full-development LR is trained
on individual-fold OOF features; later four-model averaged test features can have a
different distribution, as already accepted in PROJECT_SPEC. The locked test remains
reserved until development is frozen. No data-integrity blocker was found; these
methodological limits should remain in reporting before any hard/oracle comparison.
No hard CBM, oracle model, interventions or locked-test evaluation were implemented.
