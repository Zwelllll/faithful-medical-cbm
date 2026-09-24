# Stage 10: cross-fitted intervention engine

Completed 2026-09-24. No CNN/LR fitting, concept inference, policy experiments, threshold optimization or locked-test image/label access. Frozen exclusion IDs are used only to verify development membership.

## API and semantics

The reusable API lives in src/faithful_medical_cbm/interventions/engine.py. Construct InterventionEngine(Path("configs/intervention_engine.json")). Initialization verifies frozen OOF/cohort/split hashes, saved head hashes, exact training/validation partitions, concept order and baseline numerical reproduction for every development case. Only fold_0 through fold_3 heads are deserialized for scoring. Full-development files are hashed as part of source preservation, never used as heads.

- intervene(case_num, "soft" or "hard", concepts, values): force selected concepts to supplied binary values; no target lookup.
- correct(case_num, model_type, concepts): validate the selected names, then fetch only those frozen targets and replace their inputs.
- trajectory(case_num, model_type, concept_order): return k=0 through k=len(order), each recomputed cumulatively from the original inputs. Effects in each entry are relative to k=0.
- selection_view(case_num, model_type): immutable detached view containing concept names/probabilities, original inputs/prediction and forced-zero/one probabilities; no targets or diagnosis label.
- select_then_correct(case_num, model_type, selector): call the supplied callback with only that view, validate its selections, then fetch targets. No selection policy is implemented.

Each result contains original/intervened seven-element vectors in frozen order, score, probability, predicted diagnosis, signed/absolute probability change, fold, selected concepts/values, and operation type. Names must be unique and valid; values must be finite binary numbers. Case identifiers are preserved strings; unknown/test IDs and full-development heads are rejected.

Soft: untouched values remain original OOF probabilities. Hard: untouched values remain probability >=0.5. Corrected values are frozen 0/1 targets in either model. Scores use the case's matching held-out LR head: intercept + coefficients dot inputs; probability is sigmoid(score), diagnosis is probability >=0.5. Coefficients are model parameters, not causal importance.

A concept is marked wrong when its original thresholded probability disagrees with its target. A soft correction can change a threshold-correct probability by snapping it to 0/1. A correct hard concept has exactly zero correction effect. Forced values are model-input counterfactuals, always labelled separately from clinician corrections.

The selection interface prevents accidental label access, not malicious Python code accessing global files or closures. Retrospective target storage is confined to the executor; tests prove the callback completes selection before any chosen-target lookup and that no unselected target is read. Future policy callers must pass only this view.

## Baseline reproduction

Tolerance is absolute 1e-12, relative 0; failure prevents output generation. All 658 cases per model passed.

| Model | Maximum score error | Maximum probability error |
|---|---:|---:|
| soft | 8.8818e-16 | 3.3307e-16 |
| hard | 8.8818e-16 | 2.2204e-16 |

## Descriptive effects

Each model has 4,606 case-concept correction pairs and 9,212 forced-value rows (658 x 7 x 2). There are 967 incorrect concept predictions (20.9944%) for each model. Probability changes are on the 0-1 scale, not percentages.

| Model | Mean absolute effect, wrong concepts | Median | Q25 | Q75 | Q95 | Max | Wrong corrections flipping diagnosis |
|---|---:|---:|---:|---:|---:|---:|---:|
| soft | 0.150374 | 0.140488 | 0.062111 | 0.223547 | 0.320677 | 0.514031 | 197/967 (20.3723%) |
| hard | 0.140118 | 0.131516 | 0.074111 | 0.192398 | 0.297856 | 0.349337 | 168/967 (17.3733%) |

Per-concept summaries below include all 658 corrections, including originally threshold-correct concepts. The JSON also reports wrong-concept-only distributions and flip fractions separately for every concept.

| Concept | Incorrect | Soft mean absolute effect | Soft median | Hard mean absolute effect | Hard median |
|---|---:|---:|---:|---:|---:|
| atypical_pigment_network | 157 | 0.045348 | 0.016082 | 0.018422 | 0.000000 |
| regression_structures_present | 151 | 0.050999 | 0.019380 | 0.048917 | 0.000000 |
| irregular_pigmentation | 168 | 0.076940 | 0.040750 | 0.031858 | 0.000000 |
| blue_whitish_veil_present | 85 | 0.032881 | 0.007647 | 0.026597 | 0.000000 |
| atypical_vascular_structures | 66 | 0.008646 | 0.002431 | 0.002091 | 0.000000 |
| irregular_dots_and_globules | 186 | 0.115950 | 0.057374 | 0.055460 | 0.000000 |
| irregular_streaks | 154 | 0.014434 | 0.005585 | 0.022573 | 0.000000 |

## Files and schemas

New configuration: configs/intervention_engine.json. New code: interventions/engine.py and interventions/build_engine_artifacts.py within src/faithful_medical_cbm/. New tests: tests/test_intervention_engine.py. Documentation: this report, README and appended decision D015; .gitignore allows only the engine outputs.

Artifacts under artifacts/interventions/engine/:

- soft_single_concept_ground_truth_corrections.csv
- hard_single_concept_ground_truth_corrections.csv
- soft_forced_value_counterfactuals.csv
- hard_forced_value_counterfactuals.csv
- summary.json
- integrity.json
- .gitattributes (preserves CSV/JSON LF bytes)

Correction columns: case_num, validation_fold, diagnosis_binary, concept_name, original_concept_value, ground_truth_concept_value, concept_was_wrong, original_diagnosis_score, intervened_diagnosis_score, original_melanoma_probability, intervened_melanoma_probability, signed_probability_change, absolute_probability_change, original_predicted_diagnosis, intervened_predicted_diagnosis, diagnosis_prediction_changed.

Forced-value schema replaces ground_truth_concept_value/concept_was_wrong with forced_value, and retains the remaining columns. It does not expose concept truth. Rows are ordered by frozen development ID order, then frozen concept order; forced values are 0 then 1. Raw CSVs precede summaries.

Integrity records source/output SHA256, Git and package provenance, exact schemas, fold-exclusion assertions and numerical reproduction errors. Existing Stage 7-9 artifacts are never rewritten. Output creation refuses overwrites and protected source paths.

## Reproduction and validation

From repository root, PowerShell:

```powershell
$env:PYTHONPATH = 'src'
python -m faithful_medical_cbm.interventions.build_engine_artifacts --config configs/intervention_engine.json
python -m unittest tests.test_intervention_engine -v
```

Seven tests passed: all-case k=0 reproduction, replacement and untouched inputs, forced 0/1, sequential updates, fold head validation, full-head rejection, malformed names/values/test IDs, selection/target separation, fatal reproduction mismatch, deterministic CSVs and summaries, output hashes and overwrite guards. The end-to-end test forbids LogisticRegression.fit. No training regression tests or image-loading tests were run.

## Before policy experiments

No Stage 10 implementation blocker remains. Diagnosis flips and model-input effects are not necessarily clinical improvements or causal effects. Case-level independence remains unverifiable; the Stage 8/9 non-nested development evaluation limitations remain. Full policy evaluation and any intervention-aware fitting require the next explicitly authorized stage. No policy rankings or experiments were performed.
