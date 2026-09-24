# Stage 7: saved validation prediction assembly

Completed from the uploaded `seven-concept-v1` run, source Git commit
`57406ca22157773fb568ed462e57113d8db6bbb4`. Fold summaries select epochs
19, 18, 16 and 30. No inference, model loading, training or image access occurs.

From the repository root after installing the local package:

```bash
python -m faithful_medical_cbm.evaluation.assemble_oof --config configs/seven_concept.toml --run-dir artifacts/seven-concept-v1-20260924T041739Z-1-001/seven-concept-v1
```

For another storage location, `--run-dir` is the parent containing fold_0 through
fold_3, each with run.json, summary.json and its selected validation_epoch_NNN.csv.
No checkpoints are needed or deserialized. The saved summaries and permanent run
training/validation IDs establish provenance; this does not independently prove
checkpoint contents. Missing files fail without any inference fallback.

Existing OOF outputs are never overwritten. Use `--output-dir` with a new directory
to reproduce and compare the CSV hash. No split regeneration or label remapping is
part of this command. Only processed development labels are inspected; test IDs
are read solely as an exclusion set.

## Exact schema and checks

The CSV has 24 columns in this sequence:

1. case_num (unchanged string identifier)
2. validation_fold (0–3)
3. diagnosis_binary (frozen processed melanoma target)
4. Seven `<concept>_target` columns
5. Seven `<concept>_logit` columns
6. Seven `<concept>_probability` columns

Each seven-column block uses this exact order:
atypical_pigment_network, regression_structures_present, irregular_pigmentation,
blue_whitish_veil_present, atypical_vascular_structures, irregular_dots_and_globules,
irregular_streaks. The complete ordered header is recorded in the integrity JSON.
Rows follow the frozen development ID file order. Original numerical prediction
strings are retained; no fold averaging or prediction recalculation occurs.

Before any output is written the assembler checks:

- Frozen split, cohort and Stage 2A summary hashes; 658 development and 165 exclusion
  IDs, disjoint and exactly covering the cohort; expected fold assignments/counts.
- Source run cohort/split hashes, exact concept order, fold identifier, training IDs
  exactly equal to the other folds and validation IDs exactly equal to the held-out fold.
- Selected epoch validity, threshold 0.5, macro AUROC selection, test-exclusion flags.
- Strict ordered CSV schema, no malformed rows, blank/duplicate/unexpected IDs,
  missing development cases or locked-test overlap.
- Saved binary concept labels equal frozen processed labels; diagnosis joined from
  processed development rows only. No raw diagnosis strings are consumed.
- Finite logits; finite probabilities in [0,1]; sigmoid/logit consistency within
  1e-6 absolute/relative tolerance for float32 output roundoff.
- Recomputed selected-file per-concept and macro scores match summary.json's
  best-epoch metrics; source bytes are unchanged throughout assembly.

Both classes are required for every metric. AUROC uses saved logits; binary
Macro-F1 averages labels [0,1], using probability >=0.5 and zero_division=0.
The macro OOF values are arithmetic means over the seven pooled concept metrics,
not arithmetic means over folds. No thresholds were selected or optimized.

## Results

658 unique cases; rows per fold 165/165/164/164; 198 diagnosis-positive and 460
negative (30.0912% positive). No missing cases, duplicates, or test overlap.

| Concept | Positive | Prevalence | OOF AUROC | OOF Macro-F1 |
|---|---:|---:|---:|---:|
| atypical_pigment_network | 177 | 26.90% | 0.7978 | 0.6372 |
| regression_structures_present | 176 | 26.75% | 0.7728 | 0.6835 |
| irregular_pigmentation | 205 | 31.16% | 0.7983 | 0.6911 |
| blue_whitish_veil_present | 148 | 22.49% | 0.9254 | 0.8178 |
| atypical_vascular_structures | 47 | 7.14% | 0.8092 | 0.6726 |
| irregular_dots_and_globules | 303 | 46.05% | 0.7935 | 0.7023 |
| irregular_streaks | 185 | 28.12% | 0.8053 | 0.6751 |

Macro OOF AUROC: 0.8146319458412441. Macro OOF F1: 0.6970823357130504.

Outputs: artifacts/oof/seven_concept_oof.csv,
seven_concept_oof_summary.json and seven_concept_oof_integrity.json. Summary includes
prevalences, metrics, source run/commit/epochs and hashes of selected CSVs, summaries
and run manifests. Integrity includes all input hashes and the final CSV hash.
The uploaded source directories remain untouched and ignored by Git; archive them
with the original GPU artifacts for reproducibility. The three final OOF files are
eligible for version control with LF line endings.

Patient-level independence remains unverifiable. No data-integrity blocker remains
for the next authorized stage; no sequential CBM diagnosis model is implemented here.
