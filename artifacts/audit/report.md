# Stage 1: Derm7pt dataset audit

Scope: raw-file inspection only. No cohort, concept conversion, split or model was created.

## Schema

`meta/meta.csv`: 1011 case records; 19 columns.

```text
case_num, diagnosis, seven_point_score, pigment_network, streaks, pigmentation, regression_structures, dots_and_globules, blue_whitish_veil, vascular_structures, level_of_diagnostic_difficulty, elevation, location, sex, management, clinic, derm, case_id, notes
```

Metadata files: `meta/meta.csv`, `meta/test_indexes.csv`, `meta/train_indexes.csv`, `meta/valid_indexes.csv`.
The supplied split-index files are inventoried and hashed only; their assignments are not used.
`README.txt` describes the release; HTML gallery checks are in summary.json.

## Raw diagnosis distribution

| Raw label | Cases |
|---|---:|
| basal cell carcinoma | 42 |
| blue nevus | 28 |
| clark nevus | 399 |
| combined nevus | 13 |
| congenital nevus | 17 |
| dermal nevus | 33 |
| dermatofibroma | 20 |
| lentigo | 24 |
| melanoma | 1 |
| melanoma (0.76 to 1.5 mm) | 53 |
| melanoma (in situ) | 64 |
| melanoma (less than 0.76 mm) | 102 |
| melanoma (more than 1.5 mm) | 28 |
| melanoma metastasis | 4 |
| melanosis | 16 |
| miscellaneous | 8 |
| recurrent nevus | 6 |
| reed or spitz nevus | 79 |
| seborrheic keratosis | 45 |
| vascular lesion | 29 |

## Raw concept distributions

| Column | Values and counts |
|---|---|
| `pigment_network` | absent: 400; atypical: 230; typical: 381 |
| `streaks` | absent: 653; irregular: 251; regular: 107 |
| `pigmentation` | absent: 588; diffuse irregular: 265; diffuse regular: 115; localized irregular: 40; localized regular: 3 |
| `regression_structures` | absent: 758; blue areas: 116; combinations: 99; white areas: 38 |
| `dots_and_globules` | absent: 229; irregular: 448; regular: 334 |
| `blue_whitish_veil` | absent: 816; present: 195 |
| `vascular_structures` | absent: 823; arborizing: 31; comma: 23; dotted: 53; hairpin: 15; linear irregular: 18; within regression: 46; wreath: 2 |

## Images and missing data

2013 physical image files; 0 unreferenced; 0 broken.
1011 dermoscopic references and 1011 clinical references. 9 files are shared across the two reference columns, leaving 1002 files referenced only as clinical.
Sizes below are width x height, counting unique files within each metadata modality.

- derm: 512x512: 1; 626x474: 1; 692x482: 1; 754x512: 1; 756x513: 1; 759x510: 1; 767x508: 3; 768x502: 1; 768x509: 2; 768x511: 57; 768x512: 941; 768x532: 1
  Formats: {'JPEG': 1011}; reference status counts: {'exact': 1010, 'case_mismatch': 1}.
- clinic: 512x512: 1; 640x480: 1; 724x512: 1; 752x512: 1; 753x512: 1; 756x504: 1; 759x510: 1; 767x508: 2; 768x491: 1; 768x504: 1; 768x509: 1; 768x511: 58; 768x512: 939; 768x513: 1; 768x532: 1
  Formats: {'JPEG': 1011}; reference status counts: {'exact': 1011}.

Missing values (blank and lexical null token counts):

- `case_num`: {'blank': 0, 'null_token': 0}.
- `diagnosis`: {'blank': 0, 'null_token': 0}.
- `seven_point_score`: {'blank': 0, 'null_token': 0}.
- `pigment_network`: {'blank': 0, 'null_token': 0}.
- `streaks`: {'blank': 0, 'null_token': 0}.
- `pigmentation`: {'blank': 0, 'null_token': 0}.
- `regression_structures`: {'blank': 0, 'null_token': 0}.
- `dots_and_globules`: {'blank': 0, 'null_token': 0}.
- `blue_whitish_veil`: {'blank': 0, 'null_token': 0}.
- `vascular_structures`: {'blank': 0, 'null_token': 0}.
- `level_of_diagnostic_difficulty`: {'blank': 0, 'null_token': 0}.
- `elevation`: {'blank': 0, 'null_token': 0}.
- `location`: {'blank': 0, 'null_token': 0}.
- `sex`: {'blank': 0, 'null_token': 0}.
- `management`: {'blank': 0, 'null_token': 0}.
- `clinic`: {'blank': 0, 'null_token': 0}.
- `derm`: {'blank': 0, 'null_token': 0}.
- `case_id`: {'blank': 984, 'null_token': 0}.
- `notes`: {'blank': 1002, 'null_token': 0}.

Nonempty notes are recorded for cases: 565, 800, 824, 827, 833, 834, 839, 844, 851.

- Case 816 (derm): `FCl/Fcl068.jpg` -> `FCL/Fcl068.jpg` (case_mismatch). Raw metadata was not changed.

## Duplicates

- metadata_rows: 0 groups.
- case_num: 0 groups.
- nonempty_case_id: 0 groups.
- basenames: 0 groups.
- case_insensitive_paths: 0 groups.
- case_insensitive_basenames: 0 groups.
- repeated_references: 9 groups.
- byte_identical_images: 0 groups.
- pixel_identical_images: 0 groups.
- cross_case_shared_paths: 0 groups.
- cross_case_pixel_duplicates: 0 groups.

See duplicates.json for within-case versus cross-case reuse and exact-content matches.
Exact bytes and decoded RGB pixels were compared. Near-duplicates and repeated lesions with different views remain unverified.

## Grouping

Identifier completeness: {"case_num": {"nonmissing": 1011, "unique_nonmissing": 1011}, "case_id": {"nonmissing": 27, "unique_nonmissing": 27}}.
The semantics of case_id are undocumented locally.
No patient identifier or independently verified lesion identifier exists in the supplied metadata.
Whether a patient has multiple cases/lesions/images cannot be determined. Distinct available image files per case: {"2": 1002, "1": 9}.
Recommend the specification's fallback: use case_num as the case key and keep paired images together in a future split. This is case-level grouping, not verified patient-level or lesion-level independence.
Do not infer identity from filename prefixes or sparse case_id values. External linkage would be needed to verify patient or repeated-lesion grouping.

## Discrepancies and decisions before Stage 2

1. The release includes 20 diagnosis labels. Approve an explicit inclusion/exclusion mapping for benign melanocytic diagnoses and melanoma variants, particularly melanoma metastasis, the unsuffixed melanoma label, and potentially ambiguous labels. No mapping was applied.
2. The seven concepts are categorical, not seven ready-made binary targets. Approve explicit conversion rules for every raw value, especially vascular structures, regression structures and pigmentation. No conversion was applied.
3. Accept the case-level grouping limitation or obtain authoritative patient/lesion linkage. Do not interpret case_id as a complete lesion or patient identifier.
4. Approve case-sensitive path resolution in future processed metadata for reported filename-case mismatches, preserving raw values and files unchanged.
5. Acknowledge the clinical substitutions documented in notes; clinical images remain outside the dermoscopy-only primary experiment.
6. Supplied split indexes do not define this project's frozen 80/20 protocol. They were not adopted. The eventual split is a separate stage.
7. Image dimensions vary; resizing belongs to a later preprocessing stage, not this audit.

## Reproduction and integrity

Run `python -m src.data.audit` from the repository root. Outputs are deterministic for identical inputs and runtime.
`raw_manifest.csv` records every input file's SHA-256. All raw files were rehashed after the audit; the file list and contents were unchanged.
`provenance.json` records the audit source hash, manifest hash, Python and Pillow versions.
`image_inventory.csv`, `image_references.csv`, `case_images.csv`, `value_counts.csv`, `duplicates.json` and `summary.json` contain the detailed machine-readable evidence.
