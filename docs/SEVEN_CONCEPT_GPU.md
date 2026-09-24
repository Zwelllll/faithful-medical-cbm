# Stage 6: seven-concept predictor infrastructure

Stage 5 is PASSED per the user's external GPU report: Fold 0 atypical pigment
network best AUROC 0.8449, best Macro-F1 0.7471, best epoch 15, 22 epochs completed,
early stopped, locked_test_used=false. External run files were not supplied locally
for independent verification. The reported results did not tune this stage.

## Model, order and weights

One ImageNet-pretrained torchvision EfficientNet-B0 backbone, adaptive average pool,
dropout 0.2 and linear 1280-to-7 layer returns `[B,7]` logits (including B=1).
The seven outputs are independent binary heads, not a softmax. There is no diagnosis
head, concept-to-diagnosis path or patient-metadata input. Existing Dataset/loaders,
image transforms and transfer-learning mechanics are reused.

This exact order is checked against config, the hash-verified Stage 2A summary,
Dataset columns and checkpoint metadata. Fold 0 has 493 training cases:

| Concept | Positive | Negative | Prevalence | Training pos_weight |
|---|---:|---:|---:|---:|
| atypical_pigment_network | 129 | 364 | 26.1663% | 2.821705 |
| regression_structures_present | 133 | 360 | 26.9777% | 2.706767 |
| irregular_pigmentation | 155 | 338 | 31.4402% | 2.180645 |
| blue_whitish_veil_present | 107 | 386 | 21.7039% | 3.607477 |
| atypical_vascular_structures | 33 | 460 | 6.6937% | 13.939394 |
| irregular_dots_and_globules | 229 | 264 | 46.4503% | 1.152838 |
| irregular_streaks | 134 | 359 | 27.1805% | 2.679104 |

`seven_concept_preflight.json` records exact counts/weights and input hashes. No
images were opened for this report. Weights are computed again at runtime as
training negatives / training positives, separately for each selected fold and
concept. No validation/test labels enter weighting. Validation uses the same
training-derived vector. BCE reduction is a mean over all B*7 entries, with epoch
loss averaged by sample count. There is no class resampling or weight clipping.

Both classes must be present for every training and validation concept; otherwise
fail explicitly before constructing pretrained weights. Never silently omit a
concept from macro AUROC or substitute a score. Fold 0 passes this check.

## Unchanged schedule and metrics

`configs/seven_concept.toml` is separate from baseline and single-concept configs:
seed 42; image 224; batch 32; workers 0; pin_memory=false; Stage 3 augmentation and
ImageNet normalization. AdamW, weight decay 0.0001, max 30 epochs. First 3 epochs
head-only at LR 0.001, then last 3 EfficientNet feature children at LR 0.0001.
Frozen blocks stay in eval mode; optimizer state resets at the phase transition.

Each epoch reports weighted train/validation losses, seven validation AUROCs and
seven binary Macro-F1s. F1 averages labels [0,1], zero_division=0, and probability
>=0.5 predicts positive. AUROC uses logits to preserve ranking at sigmoid saturation.
The arithmetic mean of all seven AUROCs alone selects best.pt and controls early
stopping (patience 7, min_delta 0.0). The arithmetic mean of seven Macro-F1s is also
reported. No thresholds are optimized and no Stage 5 score affected these settings.

## First GPU run (Fold 0 only)

Clone/checkout a pinned commit containing Stage 6 in a CUDA Colab runtime. Mount
Drive and keep locked-test images offline. Reuse the Stage 4 notebook's dependency
setup (compatible CUDA torch/torchvision, requirements, editable local package),
but do not rerun its baseline training/preflight or overwrite completed runs.

Expected project-relative data paths remain:

```text
data/processed/stage2a/cohort.csv
data/raw/release_v0/images/<development derm_path>
data/splits/{development_ids.csv,test_ids.csv,development_folds.csv,split_metadata.json}
artifacts/stage2a/summary.json
```

Transfer processed metadata byte-for-byte, preserve filename case, and retain the
tracked frozen files. Raw/processed data stay out of Git. Whole data/raw and
data/processed directories can be linked to the existing Drive dataset roots.

From the repository root, link separate output parents before training:

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path('docs').resolve()))
from colab_fold0 import link_directory
outputs = Path('/content/drive/MyDrive/faithful-cbm/outputs')
for name in ('artifacts', 'checkpoints'):
    target = outputs / name / 'seven_concept'
    target.mkdir(parents=True, exist_ok=True)
    link_directory(Path(name) / 'seven_concept', target)
```

Check CUDA, then run the new entry point (prefix shell commands with `!` in Colab):

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m faithful_medical_cbm.training.train_seven_concept --config configs/seven_concept.toml --fold 0 --run-name seven-concept-v1
```

The CLI supports explicit fold numbers 0 through 3, one run at a time; there is no
automatic sweep. This stage has only prepared infrastructure. No full local or GPU
training was performed. First ImageNet weight download requires internet or cache.
The frozen cohort/split hashes and selected development memberships are verified
before model construction. Locked-test loader construction is never called.

## Persistent outputs

`artifacts/seven_concept/seven-concept-v1/fold_0/` contains:

- run.json: resolved configuration and provenance, concept order, named training
  weights/counts, permanent train/validation IDs, cohort/split/config/source hashes,
  Git commit, package versions, CUDA/cuDNN and GPU model.
- validation_epoch_NNN.csv: case_num plus all seven `_target`, `_logit` and
  `_probability` columns in frozen order (22 columns total), written BEFORE metrics.
- history.csv: losses, all per-concept scores, macro scores and training phase.
- summary.json: best epoch/macro AUROC, all best-epoch and final-epoch metrics,
  epochs completed, early stopping, concept order, weights, threshold and test policy.

`checkpoints/seven_concept/seven-concept-v1/fold_0/` contains best.pt and last.pt.
The Drive links above persist both locations separately from prior runs. Existing
run directories are rejected. Checkpoints include model/optimizer state, exact
concept names/order, config, metrics, early-stopping state and provenance. Use
`SevenConceptEfficientNet(pretrained=False)` with
`faithful_medical_cbm.training.seven_concept.load_checkpoint` for offline restoration.
Stage 5 and reordered-concept checkpoints are rejected. Exact interrupted-run
resume remains unsupported; preserve partial outputs and decide recovery separately.

No OOF table, diagnosis head, interventions, test inference or threshold tuning is
implemented. Patient-level independence remains unverifiable with case-level splits.
