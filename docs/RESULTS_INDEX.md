# Results index and storage map

All model results below are DEVELOPMENT-only. The locked test has not been evaluated.
The initial organization audit is retained below. Stage 13 joint OOF assembly is now complete; see the added entry.

| Stage | Local location and important files | Purpose / checkpoint status |
|---|---|---|
| 1 Audit | artifacts/audit/: report.md, summary.json, provenance.json, inventory/reference/value-count CSVs | Historical dataset audit; no checkpoints |
| 2A Cohort | artifacts/stage2a/summary.json; data/processed/stage2a/cohort.csv; configs/cohort_mapping.json | Frozen cohort/concepts; processed data ignored; no checkpoints |
| 2B Split | data/splits/: development_ids.csv, test_ids.csv, development_folds.csv, split_metadata.json | Frozen IDs; test IDs are an exclusion manifest, not test evaluation |
| 3 Pipeline | src/faithful_medical_cbm/data/; tests/test_data_loading.py | Dataset/loading infrastructure; no separate result or checkpoint directory |
| 4 Baseline | docs/BASELINE_GPU.md; docs/colab_fold0_manifest.json | External GPU scores recorded in DECISIONS D010; no local baseline run folder found. Expected checkpoints on Drive, not independently verified here |
| 5 Sanity | docs/CONCEPT_SANITY_GPU.md; docs/concept_sanity_preflight.json | External result recorded in D011; no local GPU run folder found. Expected checkpoints on Drive, not independently verified |
| 6 Seven concepts | artifacts/seven-concept-v1-20260924T041739Z-1-001/seven-concept-v1/fold_0..3 | run/summary/history/epoch predictions. Legacy download location retained because Stage 7 provenance references it; Stage 1-11 must not move. Expected checkpoint namespace on Drive: seven_concept/seven-concept-v1 |
| 7 OOF | artifacts/oof/: seven_concept_oof.csv, seven_concept_oof_summary.json, seven_concept_oof_integrity.json | Frozen 658-case seven-concept development OOF table |
| 8 Soft CBM | artifacts/cbm/sequential_soft/ | Cross-fitted predictions, fold/pooled metrics, JSON LR parameters, full-development head and integrity; lightweight heads local/Git |
| 9 Hard/oracle | artifacts/cbm/sequential_hard/, oracle/, comparison/ | Analogous LR results and frozen comparison; lightweight heads local/Git |
| 10 Engine | artifacts/interventions/engine/ | Correction and forced-value CSVs, summary/integrity; no new checkpoints |
| 11 Policies | artifacts/interventions/policies/ | Deterministic trajectories, compact random orders, budget_metrics.csv, intervention_curves.csv, auc_summary.json, summary/integrity |
| 12 Joint soft | artifacts/joint_soft/joint-soft-v1/fold_0..3/ | Completed GPU validation files; large checkpoints remain on Drive |
| 12 Joint hard STE | artifacts/joint_hard/joint-hard-v1/fold_0..3/ | Completed GPU validation files; large checkpoints remain on Drive |

## Completed Stage 12 GPU results

Verified from local summaries and matching history rows. Each fold contains run.json,
summary.json, history.csv, validation_epoch_*.csv and the selected best-epoch CSV.
Best-epoch CSV IDs match the run's saved validation IDs. No checkpoints were opened.

| Model | Fold | Best epoch | Diagnosis AUROC | Concept macro AUROC |
|---|---:|---:|---:|---:|
| Joint soft | 0 | 13 | 0.864000 | 0.796619 |
| Joint soft | 1 | 15 | 0.859652 | 0.808978 |
| Joint soft | 2 | 16 | 0.889973 | 0.827058 |
| Joint soft | 3 | 14 | 0.869565 | 0.814941 |
| soft mean | — | — | 0.870798 | 0.811899 |
| Joint hard STE | 0 | 6 | 0.844957 | 0.765971 |
| Joint hard STE | 1 | 18 | 0.825130 | 0.820837 |
| Joint hard STE | 2 | 14 | 0.847294 | 0.821348 |
| Joint hard STE | 3 | 16 | 0.847649 | 0.819753 |
| hard mean | — | — | 0.841257 | 0.806977 |

These are mean best-fold validation metrics, NOT pooled joint OOF metrics and NOT
locked-test results. They match the supplied six-decimal values. No pooled joint
OOF table was constructed at the time of that organization audit. Stage 13 pooled results are indexed below.

## Organization verification

The requested source wrappers were present but already empty at first inspection:
- artifacts/joint-soft-v1-20260924T140609Z-1-001/
- artifacts/joint-hard-v1-20260924T140622Z-1-001/

The inner runs were already at their canonical destinations. No move or merge was
needed. Therefore a historical before/after move comparison cannot be claimed.
Current contents were inventoried and rechecked unchanged during this task:
98 soft files (4,931,661 bytes), 94 hard files (4,920,063 bytes).
See artifact_organization_inventory.json for each file's SHA256 and size.
No second populated joint run was found. The soft run also contains an empty
artifacts/ subdirectory; it was left untouched. Empty wrappers remain because
automatic approval review blocked the removal command without a specific reason.
No files or directories were deleted or moved by this task.

## Storage convention

Git/local lightweight material: source, configs, documentation, tests, summaries,
reasonably sized prediction CSVs and provenance/integrity JSONs. Stage 12 JSON/CSV
files are now explicitly eligible for Git; namespace attributes disable text
conversion to preserve downloaded file bytes. They have NOT been staged.

Drive is the authoritative storage for large .pt/.pth/.ckpt weights, large/raw
training data, raw Derm7pt images and bulky outputs. Existing ignored local dataset
copies remain in place for authorized workflows; they are not Git content.
Expected Stage 12 checkpoint destinations from the documented Drive workflow:
outputs/checkpoints/joint_soft/joint-soft-v1/fold_N/{best.pt,last.pt} and
outputs/checkpoints/joint_hard/joint-hard-v1/fold_N/{best.pt,last.pt}.
Drive was not mounted/queried here; presence and hash equality of those remote
checkpoints cannot be independently confirmed. No Drive checkpoint was modified.

Local checkpoints/ contains only tracked .gitkeep (0 bytes). Keep it because configs
and training code expect the checkpoint root. Global binary ignore rules also
protect checkpoints accidentally placed outside checkpoints/.

## Root housekeeping and remaining items

Canonical source/research directories and project-control files remain at root.
.cache/, .pytest_cache/ and .venv/ are untracked and ignored. Keep .venv locally.
Caches were deliberately retained: .cache includes historical Stage 3/4 hash
snapshots and a source scratch file; deleting them is unnecessary for reproducibility.
.pytest_cache contains only regenerable pytest data, but need not be removed.

CODEX_KICKOFF_PROMPT.txt contains the historical initialization request. No current
repository text reference was found (excluding Git internals, environment and raw
data). It is retained as a historical project-control file, not automatically deleted.

No tracked raw images, neural checkpoint binaries, virtualenv/cache files were found.
No file larger than 10 MB was found in artifacts/checkpoints/docs/notebooks/src/tests.
This size scan deliberately excludes raw data, .venv and .git; those are not clutter
candidates. No suspicious large file outside those expected local stores was found.
The legacy Stage 6 wrapper is deliberately retained, not duplicated/reorganized.

Recommended review/commit candidates are this index, the SHA inventory, README and
.gitignore changes, namespace .gitattributes and the 192 Stage 12 result JSON/CSVs.
Nothing was staged, committed or pushed. Remaining optional housekeeping is removing
the two empty wrappers and the empty soft-run artifacts/ directory; no research
blocker or duplicate populated joint run was found.

## Stage 13 — pooled joint OOF and development comparison

Canonical directory: artifacts/joint_oof/. Contains both 658-case joint OOF CSVs,
pooled/per-fold metric JSONs, diagnosis/concept comparison CSVs, calibration bins
and diagnostics, summary and integrity hashes. No checkpoints or new inference.
See [JOINT_OOF_COMPARISON.md](JOINT_OOF_COMPARISON.md). These are pooled development
metrics, distinct from the Stage 12 fold means; no locked-test results exist.
