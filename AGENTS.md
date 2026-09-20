# AGENTS.md

## Project
This repository contains a research project investigating faithful and intervenable Concept Bottleneck Models (CBMs) for dermoscopic melanoma classification.

The authoritative research specification is:
- docs/PROJECT_SPEC.md

Read that document before making architectural or experimental decisions.

## Core research question
We are investigating whether clinically meaningful concept bottlenecks actually control the diagnostic decisions of a medical image classifier.

Primary comparison:
- Soft vs Hard concept bottlenecks
- Sequential vs Joint training

Primary task:
- melanoma vs benign melanocytic lesions
- dermoscopic images from Derm7pt

## Scientific constraints
1. Do not use the locked test set for model selection, hyperparameter tuning, threshold selection, calibration fitting, early stopping, or debugging.
2. The test set must remain untouched until the final frozen evaluation.
3. Generate downstream training features using out-of-fold concept predictions. Never train the diagnosis classifier on in-sample predictions from the concept model.
4. Prevent patient/lesion leakage across folds whenever reliable grouping IDs are available.
5. Never feed diagnosis-derived metadata, validity masks, or target proxies into the model.
6. Active intervention policies may not inspect concept ground truth before choosing which concept to query.
7. Oracle intervention policies must always be explicitly labeled as oracle.
8. The primary diagnosis head must remain linear/logistic unless explicitly changed by the user.
9. Do not add residual/complement pathways, diffusion models, large vision transformers, or other architectural extensions without explicit approval.
10. Do not silently change the clinical cohort or concept definitions.

## Reproducibility
- Centralize random seeds.
- Save split IDs and fold assignments.
- Use config files instead of hard-coded experiment parameters.
- Record package versions.
- Save model configuration with checkpoints.
- Save raw prediction tables before aggregate metrics.
- Prefer deterministic behavior when practical.

## Engineering expectations
Use Python and PyTorch.

Prefer:
- src/data/
- src/models/
- src/training/
- src/interventions/
- src/evaluation/

Reusable research logic belongs in src/, not only in notebooks.
Use type hints for important public functions.
Add assertions for assumptions that could cause leakage.
Write unit tests for critical research logic.

## Critical tests
Eventually verify:
- train/test IDs do not overlap
- group IDs do not cross splits when grouping is available
- each OOF sample was predicted by a model that did not train on it
- each development sample appears exactly once in OOF predictions
- active intervention selection does not access ground-truth concepts
- concept replacement produces mathematically correct LR outputs
- the locked test set is not loaded by development training scripts

## Working style
Work one project stage at a time.

Before implementing a stage:
1. Read the relevant specification.
2. Inspect the existing repository.
3. State assumptions.
4. Make the smallest coherent implementation.

After implementing:
1. Run relevant tests.
2. Run lint/static checks if configured.
3. Report files changed.
4. Report commands executed.
5. Report test results.
6. Mention unresolved risks or assumptions.

Do not continue into the next research stage unless explicitly requested.

## Scientific integrity
If a requested implementation would introduce data leakage, invalidate the evaluation protocol, or conflict with the frozen research design, stop and explain the conflict rather than silently implementing it.
