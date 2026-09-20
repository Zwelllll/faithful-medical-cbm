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
