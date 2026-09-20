# Project Specification

**Status: FROZEN CORE DESIGN**

Changes to the following require explicit user approval:
- clinical cohort
- primary task
- seven concept definitions
- 80/20 evaluation protocol
- OOF training strategy
- EfficientNet-B0 concept architecture
- logistic diagnosis head
- primary evaluation metrics
- intervention definitions

## 1. Project in One Sentence
Build an explainable deep-learning system that diagnoses melanoma versus benign melanocytic lesions from dermoscopic images by first predicting seven clinically meaningful dermoscopic concepts, then test whether those concepts genuinely control the model's diagnosis when a human corrects them.

## 2. Core Research Question
When a medical AI claims that its decision is based on human-understandable clinical concepts, can we trust that claim?

More specifically:
How do soft versus hard concept bottlenecks and sequential versus joint training affect diagnostic performance, concept faithfulness, calibration, and human intervention effectiveness?

## 3. Dataset and Clinical Scope
Use Derm7pt.

Primary experiment:
- Positive class: melanoma
- Negative class: benign melanocytic nevi included by explicit config mapping
- Dermoscopic images only

Do not initially use clinical photographs, age, sex, anatomical location, or other patient metadata.

Do not silently reassign ambiguous diagnoses.

## 4. Seven Binary Clinical Concepts
Primary concept representation:
1. atypical pigment network
2. regression structures
3. irregular pigmentation
4. blue-whitish veil
5. atypical vascular structures
6. irregular dots/globules
7. irregular streaks

Preserve original categorical annotations in processed metadata for auditing.

## 5. Architecture
Dermoscopic image
-> ImageNet-pretrained EfficientNet-B0
-> shared visual embedding
-> seven binary concept heads
-> concept vector
-> logistic regression
-> melanoma probability

The downstream diagnosis model must remain linear/logistic in the primary study.

## 6. Why EfficientNet-B0
Derm7pt is small, so use transfer learning rather than training from scratch.

Initial strategy:
1. ImageNet-pretrained EfficientNet-B0
2. freeze most backbone layers
3. train concept heads
4. unfreeze upper blocks
5. low-learning-rate fine-tuning

Do not replace this with a large ViT or larger backbone without explicit approval.

## 7. Model Variants
### Model 0 — Black-box baseline
image -> EfficientNet-B0 -> melanoma probability

### Model 1 — Sequential Soft CBM
image -> concept probabilities -> frozen logistic regression -> diagnosis

### Model 2 — Sequential Hard CBM
image -> concept probabilities -> hard concepts -> logistic regression -> diagnosis

### Model 3 — Joint Soft CBM
Train concept and diagnosis objectives jointly:
L = L_concept + lambda_y * L_diagnosis

### Model 4 — Joint Hard CBM
Forward uses hard/discrete concepts with an explicitly documented differentiable approximation such as a straight-through estimator.

### Model 5 — Oracle Concept Classifier
ground-truth concepts -> logistic regression -> diagnosis
This is an analytical ceiling, not a deployable model.

## 8. Primary Experimental Matrix
- CBM-S-SEQ
- CBM-H-SEQ
- CBM-S-JOINT
- CBM-H-JOINT
Plus:
- black-box baseline
- oracle concept classifier

## 9. Hypotheses
H1: Hard bottlenecks may sacrifice some diagnostic performance but improve semantic faithfulness relative to soft bottlenecks.
H2: Correcting incorrectly predicted concepts should improve downstream diagnostic performance.
H3: Intervention-aware diagnosis training should improve robustness to mixed AI/human concept inputs.
H4: Active querying based on uncertainty and counterfactual diagnostic impact should require fewer interventions than uninformed querying for similar diagnostic benefit.
H5: Joint optimization may improve diagnosis while reducing semantic purity or intervenability relative to sequential training.

## 10. Data Integrity and Splitting
Before any model training:
1. audit dataset identity fields
2. determine whether reliable patient IDs exist
3. determine whether lesion/case IDs exist
4. determine whether duplicates or paired samples could cross folds

Do NOT assume patient_id exists.

Grouping hierarchy:
- use patient ID if verified
- else lesion ID if verified
- else deterministic case-level stratification and explicitly state the limitation

Create once:
- 80% development
- 20% locked test

Locked test must never be used for:
- architecture choice
- augmentation tuning
- learning-rate tuning
- epoch selection
- threshold tuning
- regularization tuning
- calibration fitting
- intervention-policy design
- debugging

Save permanent sample IDs.

## 11. Development Cross-Validation
Use 4-fold cross-validation inside the 80% development pool.

For each fold:
- train concept model on the other 3 folds
- predict the held-out fold
- store predictions and provenance

Every OOF sample must be predicted by a model that did not train on it.

## 12. OOF Artifact
Save a machine-readable table containing:
- sample_id
- grouping ID if available
- fold
- diagnosis
- seven concept truths
- seven concept probabilities
- model checkpoint identifier
- seed/config provenance

Every development sample appears exactly once.

## 13. Sequential CBM Training
Train downstream logistic regression only on OOF concept predictions.

Soft:
[p1, ..., p7]

Hard:
[I(p1>t1), ..., I(p7>t7)]

Thresholds selected only from development data.

## 14. Intervention-Aware Training
Expose the diagnosis layer during training to mixed predicted and human-corrected concepts.

Sample an intervention budget:
k ~ Uniform(0, K)

Replace k predicted concepts with their ground-truth values.

Do mixing in the training loop/on-device rather than inside the Dataset class.

## 15. Intervention Policies
### A. Random-error oracle
Randomly correct a concept known to be wrong using ground truth.
Oracle-only.

### B. Confidently-wrong oracle
Correct the most confident known error.
Oracle-only.

### C. Active uncertainty x impact
For concept i:
H_i = -[p_i log p_i + (1-p_i) log(1-p_i)]

Impact_i =
sum over k in {0,1} of
P(c_i=k|x) * |P(Y|C with c_i=k) - P(Y|C)|

Score_i = Uncertainty_i * Impact_i

Query the highest-scoring concept.
Ground truth must not be inspected before selection.

## 16. Why Raw Logistic Weights Are Not Importance
Concepts may be correlated, making individual LR coefficients unstable as importance measures.

Do not use |w_i| as the primary intervention importance score.
Use explicit counterfactual downstream effect.

## 17. Primary Evaluation Metrics
Diagnostic:
- AUROC

Concept quality:
- Macro-F1 per concept
- mean concept Macro-F1

Calibration:
- ECE
- reliability curve

Intervenability:
- diagnostic performance as a function of number of concept interventions

## 18. Statistical Reporting
For final locked test metrics:
- bootstrap confidence intervals
- paired bootstrap comparisons when comparing models on the same cases
- patient-level resampling if reliable patient grouping exists

Avoid unnecessary p-value-heavy analysis.

## 19. Final Test Inference
Keep all four fold concept models.

For each locked test image:
1. get concept probabilities from all four
2. average probabilities
3. pass averaged representation through frozen diagnosis model

Soft CBM:
use mean probabilities

Hard CBM:
threshold mean probabilities using development-selected thresholds

Do not select the best fold based on test performance.

## 20. OOF-to-Ensemble Distribution Shift
Explicitly acknowledge:
P(C_hat_OOF) != P(C_bar_ensemble)

The diagnosis layer is trained on individual-model OOF predictions while test inference uses an ensemble average.

This is accepted and must be documented.

## 21. Black-box Baseline Protocol
Use the same split, preprocessing family, augmentation family, backbone family, and general fine-tuning strategy.

## 22. Data Augmentation
Candidate train-time augmentations:
- horizontal/vertical flips
- small rotations
- random resized crop
- mild brightness/contrast/saturation changes

Avoid biologically implausible transformations.

Validation/test preprocessing is deterministic.

## 23. Class Imbalance
Possible development-side strategies:
- weighted BCE
- balanced sampling
- threshold tuning on development data

Do not use plain accuracy as the main selection metric.

## 24. Dataset Engineering
Dataset items should be simple:
- image
- diagnosis
- concepts
- sample_id

Metadata cleaning, diagnosis mapping, concept conversion, and cohort logic happen offline.

Do not do heavy pandas/string parsing in __getitem__.

## 25. Validity Masks and Target Leakage
Never feed diagnosis-derived validity/applicability masks into the downstream classifier.

The primary study avoids this by restricting the cohort to clinically appropriate melanocytic lesions.

## 26. Reproducibility
Record:
- Python
- PyTorch
- CUDA
- cuDNN
- timm
- scikit-learn
- augmentation library
- seeds
- split IDs
- fold assignments
- configs
- checkpoint provenance
- Git commit hash when available

Set seeds for Python, NumPy, PyTorch, and CUDA where available.

## 27. Preferred Repository Structure
faithful-medical-cbm/
├── AGENTS.md
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── artifacts/
├── checkpoints/
├── docs/
│   ├── PROJECT_SPEC.md
│   └── DECISIONS.md
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── interventions/
│   └── evaluation/
├── tests/
├── README.md
└── requirements.txt or environment.yml

## 28. Required Build Stages
00. Repository initialization
01. Dataset audit
02. Cohort construction
03. Frozen split
04. PyTorch Dataset/DataLoader
05. Black-box EfficientNet baseline
06. One-concept / one-fold sanity check
07. Seven-head concept predictor
08. Four-fold concept training
09. OOF generation
10. OOF integrity audit
11. Sequential Soft CBM
12. Sequential Hard CBM
13. Oracle concept classifier
14. Basic intervention engine
15. Intervention-aware diagnosis training
16. Random-error intervention
17. Confidently-wrong intervention
18. Active intervention
19. Joint Soft CBM
20. Joint Hard CBM
21. DEVELOPMENT FREEZE
22. Locked test evaluation
23. Error analysis
24. Interactive demo
25. Final report / README / presentation

Do not jump ahead without explicit user instruction.

## 29. Stage Exit Criteria
Examples:

Dataset audit:
- verified schema
- diagnosis counts
- concept value counts
- missing-data report
- grouping recommendation
- duplicate audit

Frozen split:
- permanent IDs saved
- zero sample overlap
- zero group overlap when grouping exists
- class balance reported
- four development folds saved

OOF:
- every development ID appears exactly once
- no duplicates
- no missing IDs
- each OOF prediction came from a model that did not train on that sample

Intervention engine:
- changing a concept produces mathematically expected logistic output
- active selector never reads ground truth before selection

## 30. Deliberately Out of Scope
Do not add to the primary project:
- residual/complement branch
- diffusion counterfactual generation
- giant vision transformers
- nonlinear diagnosis MLP
- full Derm7pt disease spectrum
- patient metadata
- Grad-CAM as the primary explanation mechanism
- large collections of unrelated XAI metrics

## 31. Final Demo
The final prototype should:
1. accept a dermoscopic image
2. display seven concept probabilities
3. display melanoma probability
4. recommend a concept for review using active uncertainty x impact
5. allow correction of the concept
6. immediately recompute diagnosis
7. show intervention history

Clearly label it as a research prototype, not a clinical diagnostic device.

## 32. Final Scientific Definition
A controlled investigation of the faithfulness and intervenability of Concept Bottleneck Models for dermoscopic melanoma classification, comparing soft and hard semantic bottlenecks under sequential and joint optimization, using out-of-fold training, a locked test set, and simulated clinician interventions to determine whether human-correctable clinical concepts genuinely control diagnostic predictions.
