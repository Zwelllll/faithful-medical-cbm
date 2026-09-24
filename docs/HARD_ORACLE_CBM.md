# Stage 9: hard and oracle concept diagnosis models

Completed on 2026-09-24. Development diagnostics only; locked-test exclusion IDs were checked, but no test labels, images or performance were accessed. No CNN training/inference or threshold tuning.

Both models reuse the saved Stage 8 configuration exactly: L2 logistic regression, C=1, lbfgs, intercept, no class weighting/scaling, max_iter=1000, tol=1e-8, seed=42. Hard inputs are OOF probability >=0.5; oracle inputs are the seven frozen target columns. Diagnosis threshold >=0.5. ECE uses the Stage 8 ten equal-width positive-probability bins.

Cross-fit heads on the other three development folds and evaluate the held-out fold. Full-development heads are fitted afterward and never used for these metrics.

## Metrics

| Model / fold | AUROC | Macro-F1 | Accuracy | Sensitivity | Specificity | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| sequential_hard / 0 | 0.836087 | 0.741321 | 0.787879 | 0.600000 | 0.869565 | 0.137719 | 0.049730 |
| sequential_hard / 1 | 0.821391 | 0.722689 | 0.787879 | 0.500000 | 0.913043 | 0.142721 | 0.059066 |
| sequential_hard / 2 | 0.884827 | 0.792559 | 0.823171 | 0.734694 | 0.860870 | 0.115883 | 0.087785 |
| sequential_hard / 3 | 0.860870 | 0.739683 | 0.792683 | 0.571429 | 0.886957 | 0.137662 | 0.071243 |
| sequential_hard / pooled | 0.840157 | 0.750384 | 0.797872 | 0.601010 | 0.882609 | 0.133517 | 0.045250 |
| oracle / 0 | 0.943826 | 0.840341 | 0.866667 | 0.760000 | 0.913043 | 0.089133 | 0.050048 |
| oracle / 1 | 0.925913 | 0.770833 | 0.818182 | 0.600000 | 0.913043 | 0.109016 | 0.072501 |
| oracle / 2 | 0.897604 | 0.784211 | 0.817073 | 0.714286 | 0.860870 | 0.118097 | 0.044267 |
| oracle / 3 | 0.915084 | 0.812400 | 0.847561 | 0.693878 | 0.913043 | 0.101081 | 0.082180 |
| oracle / pooled | 0.921723 | 0.802361 | 0.837386 | 0.691919 | 0.900000 | 0.104316 | 0.037424 |

## Frozen soft comparison

| Model | AUROC | Macro-F1 | Accuracy | Sensitivity | Specificity | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| soft | 0.863538 | 0.772003 | 0.820669 | 0.595960 | 0.917391 | 0.127925 | 0.024854 |
| hard | 0.840157 | 0.750384 | 0.797872 | 0.601010 | 0.882609 | 0.133517 | 0.045250 |
| oracle | 0.921723 | 0.802361 | 0.837386 | 0.691919 | 0.900000 | 0.104316 | 0.037424 |

| Descriptive gap | AUROC | Macro-F1 |
|---|---:|---:|
| soft_minus_hard | 0.023381 | 0.021618 |
| oracle_minus_soft | 0.058185 | 0.030358 |
| oracle_minus_hard | 0.081566 | 0.051976 |

## Coefficients

Descriptive log-odds coefficients, not causal importance. Columns c1-c7 use the frozen order: atypical_pigment_network, regression_structures_present, irregular_pigmentation, blue_whitish_veil_present, atypical_vascular_structures, irregular_dots_and_globules, irregular_streaks. JSON artifacts retain full precision.

| Model / head | Intercept | c1 | c2 | c3 | c4 | c5 | c6 | c7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sequential_hard / fold_0 | -2.440680 | 0.711008 | 1.458810 | 0.783401 | 1.266994 | 0.156374 | 1.012731 | 0.490777 |
| sequential_hard / fold_1 | -2.516444 | 0.695898 | 1.181141 | 0.887442 | 1.134061 | 0.246362 | 1.229041 | 0.528468 |
| sequential_hard / fold_2 | -2.282020 | 0.457872 | 1.376049 | 0.883258 | 1.001419 | -0.017596 | 0.939383 | 0.673064 |
| sequential_hard / fold_3 | -2.437846 | 0.083994 | 1.264539 | 0.554798 | 1.309556 | -0.048925 | 1.327173 | 0.561843 |
| sequential_hard / full_development | -2.429778 | 0.515932 | 1.342694 | 0.790731 | 1.193628 | 0.066453 | 1.116724 | 0.554125 |
| oracle / fold_0 | -3.820249 | 1.704292 | 1.198761 | 1.053717 | 1.625642 | 0.837765 | 1.388851 | 0.897622 |
| oracle / fold_1 | -3.985429 | 1.851078 | 1.133350 | 1.211543 | 1.222287 | 1.229149 | 1.551021 | 0.760637 |
| oracle / fold_2 | -4.055801 | 1.935409 | 1.258156 | 1.328831 | 1.481274 | 0.729116 | 1.547237 | 0.839151 |
| oracle / fold_3 | -3.965164 | 1.566062 | 0.945789 | 0.928862 | 1.336138 | 0.836417 | 1.827796 | 1.002576 |
| oracle / full_development | -4.028712 | 1.798953 | 1.171074 | 1.155530 | 1.458154 | 0.953374 | 1.608690 | 0.876409 |

## Outputs and integrity

Each of `artifacts/cbm/sequential_hard/` and `artifacts/cbm/oracle/` contains cross_fitted_predictions.csv, fold_metrics.csv, pooled_metrics.json, calibration_bins.json, fold_0_model.json through fold_3_model.json, full_development_model.json, run_config.json and integrity.json. The comparison is artifacts/cbm/comparison/comparison.json.

Prediction schema: case_num, validation_fold, diagnosis_binary, seven frozen-order inputs (hard: concept_hard; oracle: concept_target), decision_score, melanoma_probability, predicted_diagnosis.

Both tables cover exactly 658 unique development cases (folds 165/165/164/164), with zero test overlap. All LR validation IDs are excluded from that head's training. Oracle labels match the processed frozen cohort. Separate feature allowlists prevent target substitution in hard inputs and predicted-feature use in oracle inputs. Source and output hashes, coefficient order, environment and Git provenance are saved. Stage 8 output hashes remained unchanged.

## Reproduction and tests

PowerShell from repository root:

```powershell
$env:PYTHONPATH = 'src'
python -m faithful_medical_cbm.training.train_binary_cbm --config configs/binary_cbm.json
```

Outputs are frozen after creation; the command refuses overwrites. The Python run(config_path, output_root=...) API permits an empty separate directory for verification. No package installation or internet is required in the existing environment.

Tests: 11 passed, comprising tests.test_binary_cbm, tests.test_oof, and Stage 8 test_frozen_coverage_and_malformed_rows / test_metrics_and_calibration_calculation. Only tabular LR was fitted; no full CNN test suite was run.

## Limits before intervention work

Patient-level independence cannot be verified. Hard/soft results inherit the Stage 8 non-nested protocol: concept early stopping used held-out concept labels, and CNNs producing LR-training features may have trained on cases in the LR-validation fold. These are development diagnostics, not independent test estimates. Oracle uses true concepts and is an analytical reference, not a deployable predictor or guaranteed empirical ceiling. Differences are descriptive, not causal effects. OOF-to-ensemble shift remains a future evaluation limitation.

No implementation blocker remains for Stage 9. Intervention policies and evaluation protocol need explicit authorization in the next stage; none were implemented here.
