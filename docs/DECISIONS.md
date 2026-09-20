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
