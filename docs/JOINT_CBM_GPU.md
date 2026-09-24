# Stage 12: joint soft and joint hard CBM infrastructure

Infrastructure only. No full local training, pretrained weight download, locked-test
images/labels, sequential retraining or changes to Stage 7-11 artifacts occurred.

## Architecture and objective

Both variants reuse the Stage 6 torchvision ImageNet-pretrained EfficientNet-B0,
adaptive pooling, dropout 0.2 and linear 1280-to-7 concept head. Frozen concept
order is checked against config, processed metadata and checkpoint metadata.

- **joint_soft:** sigmoid of the seven logits feeds one linear 7-to-1 diagnosis
  layer with an intercept. There are no hidden diagnosis layers or residual routes.
- **joint_hard_ste — JOINT HARD CBM WITH STRAIGHT-THROUGH ESTIMATOR:** forward
  diagnosis inputs are exactly sigmoid(logits) >=0.5 converted to 0/1. Backward
  gradients use the sigmoid surrogate. Implementation is
  `hard + (soft - soft.detach())`, equivalent in gradient semantics to
  `soft + (hard - soft).detach()`, with exactly binary forward values.
  Thresholding itself is not differentiable. Validation also uses the binary path.

The model returns concept_logits [B,7], concept_probabilities [B,7],
diagnosis_inputs [B,7] and diagnosis_logits [B]. Diagnosis receives only the
seven concept representations, never image features, raw metadata or labels.

Loss is exactly weighted concept BCEWithLogitsLoss + unweighted diagnosis
BCEWithLogitsLoss, coefficients 1.0 and 1.0. Concept BCE averages across B*7;
diagnosis BCE averages across B. Epoch losses are sample-weighted means. Concept
positive weights are training negatives / positives for each selected fold,
with no clipping, scaling or validation-derived weighting. Validation loss uses
the same training weights. Both classes are required for every concept.

## Fixed schedule and selection

Separate configs: configs/joint_soft.toml and configs/joint_hard.toml.
ImageNet normalization and Stage 3 augmentation/loaders are reused. Image 224,
batch 32, seed 42, AdamW, weight decay 0.0001. First three epochs train concept
and diagnosis heads at LR 0.001 with frozen backbone. Thereafter both heads and
the last three torchvision feature children train at LR 0.0001. Frozen blocks
remain in evaluation mode (BatchNorm buffers and stochastic depth frozen).
Optimizer state resets at the phase transition, as in the previous schedule.

Maximum 30 epochs; patience 7; strict validation **diagnosis AUROC** improvement
(min_delta 0) selects best.pt and controls stopping. Ties retain the earlier best.
Concept quality is measured separately and never enters checkpoint selection.
No schedule or loss tradeoff was tuned from previous results.

Fold 0: 493 training and 165 validation cases. Metadata-only preflight:

| Concept | Positive | Negative | Prevalence | pos_weight |
|---|---:|---:|---:|---:|
| atypical_pigment_network | 129 | 364 | 26.1663% | 2.821705 |
| regression_structures_present | 133 | 360 | 26.9777% | 2.706767 |
| irregular_pigmentation | 155 | 338 | 31.4402% | 2.180645 |
| blue_whitish_veil_present | 107 | 386 | 21.7039% | 3.607477 |
| atypical_vascular_structures | 33 | 460 | 6.6937% | 13.939394 |
| irregular_dots_and_globules | 229 | 264 | 46.4503% | 1.152838 |
| irregular_streaks | 134 | 359 | 27.1805% | 2.679104 |

Exact weights/counts and frozen file hashes: joint_preflight.json. Fold weights
are recomputed from that fold's training metadata, never reused across folds.

## Validation and saved outputs

Each epoch records train/validation concept_loss, diagnosis_loss and total_loss.
Diagnosis metrics: AUROC (logit ranking), binary Macro-F1, accuracy, sensitivity,
specificity, Brier and Stage 8 positive-probability ECE with ten equal-width bins.
Concept metrics: seven AUROCs, seven binary Macro-F1 values, arithmetic mean
concept AUROC and mean concept F1. All thresholds are fixed at >=0.5.

Raw validation_epoch_NNN.csv is written BEFORE aggregate metrics. Columns:
case_num, diagnosis_binary, diagnosis_logit, diagnosis_probability, seven ordered
concept_target columns, seven concept_logit columns, seven concept_probability
columns; hard additionally records seven concept_hard_input columns. All concept
blocks preserve the frozen order. Soft probabilities are retained for both models.

Output structure:

```text
artifacts/joint_soft/joint-soft-v1/fold_0/{run.json,history.csv,summary.json,validation_epoch_NNN.csv}
checkpoints/joint_soft/joint-soft-v1/fold_0/{best.pt,last.pt}
artifacts/joint_hard/joint-hard-v1/fold_0/{run.json,history.csv,summary.json,validation_epoch_NNN.csv}
checkpoints/joint_hard/joint-hard-v1/fold_0/{best.pt,last.pt}
```

run.json and checkpoint provenance record config, model type/STE label, concept
order, training weights/prevalence, exact train/validation IDs, fold, seed,
cohort/split/config hashes, Git commit, source hashes, package/Python/CUDA/cuDNN/GPU
versions, pretrained weight identifier and locked_test_used=false. Checkpoints
add model/optimizer state, epoch, phase, loss weights and early-stopping state.
Best and last save atomically. Existing run directories are rejected.

Restore offline using JointCBM(model_type, pretrained=False) and
training.joint_cbm.load_checkpoint(path, model). Incompatible model type, order,
architecture, input dimension or loss protocol is rejected. For optimizer restore,
first configure the saved trainable blocks and matching optimizer. This is not
an exact interrupted-run resume API: RNG/loader states are not restored.

## Colab setup contract

1. Select a GPU runtime. Clone/pull the repository and checkout a commit containing
   Stage 12; record that commit. Work from the inner repository root containing
   pyproject.toml. No new notebook is required.
2. Mount Google Drive. Retain a compatible CUDA torch/torchvision pair, install
   dependencies/local package with `python -m pip install -e .`, then check CUDA.
3. Expose the existing processed cohort byte-for-byte at
   data/processed/stage2a/cohort.csv and development dermoscopic images at
   data/raw/release_v0/images/<derm_path>. Keep locked-test images offline.
   Keep tracked data/splits files and artifacts/stage2a/summary.json unchanged.
   Preserve FCL path casing. Raw/processed data must remain outside Git.
4. Link ONLY the new output namespaces to Drive, using the existing safe helper:

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import sys
sys.path.insert(0, str(Path('docs').resolve()))
from colab_fold0 import link_directory

# Adjust this external dataset root to the existing Drive location.
dataset = Path('/content/drive/MyDrive/faithful-cbm/dataset')
link_directory(Path('data/raw'), dataset / 'raw')
link_directory(Path('data/processed'), dataset / 'processed')

outputs = Path('/content/drive/MyDrive/faithful-cbm/outputs')
for namespace in ('joint_soft', 'joint_hard'):
    for category in ('artifacts', 'checkpoints'):
        target = outputs / category / namespace
        target.mkdir(parents=True, exist_ok=True)
        link_directory(Path(category) / namespace, target)
```

The helper accepts an already-matching link and refuses to replace nonempty
directories or differently targeted links. Do not delete previous run outputs.
Do not invoke the old notebook's baseline training cells.

Verify the frozen setup without iterating images:

```python
import hashlib, json, torch, torchvision, timm
from faithful_medical_cbm.data.loaders import LoaderFactory
from faithful_medical_cbm.training.baseline import development_fold_loaders
assert torch.cuda.is_available(), 'Select a CUDA GPU runtime'
print(torch.cuda.get_device_name(0), torch.__version__, torchvision.__version__,
      timm.__version__, torch.version.cuda, torch.backends.cudnn.version())
manifest = json.loads(Path('docs/joint_preflight.json').read_text())
for name, spec in manifest['source_files'].items():
    content = Path(name).read_bytes()
    if spec['mode'] == 'lf':
        content = content.replace(b'\r\n', b'\n')
    assert hashlib.sha256(content).hexdigest() == spec['sha256'], name
for name in ('soft', 'hard'):
    factory = LoaderFactory(f'configs/joint_{name}.toml')
    loaders = development_fold_loaders(factory, 0)
    assert all(p.is_file() for loader in loaders.values() for p in loader.dataset.paths)
```

The entry point itself checks CUDA before constructing any loaders, validates
frozen hashes/fold isolation and all development file paths before downloading
weights, and never constructs a locked-test loader. The first GPU run needs
internet or a populated torchvision cache for ImageNet weights.

Exact first-run shell commands (prefix with `!` in a Colab cell):

```bash
python -m faithful_medical_cbm.training.train_joint --config configs/joint_soft.toml --fold 0 --run-name joint-soft-v1
python -m faithful_medical_cbm.training.train_joint --config configs/joint_hard.toml --fold 0 --run-name joint-hard-v1
```

Run separately. Folds 1-3 are supported explicitly; there is no automatic sweep.

## Local verification and limits

`python -m unittest tests.test_joint_cbm -v`: eight tests passed. Tests use
pretrained=False and reject weight downloads/image opens. They cover both actual
EfficientNet shapes, one synthetic optimizer step per variant, soft/hard head
inputs, absence of residual routes, STE gradients, phase freezing, checkpoint
round trips/type-order rejection, all-four-fold isolation, training-only weights,
joint loss, diagnosis/concept metrics, CSV serialization, CPU/test-loader guards,
synthetic validation and diagnosis-only selection with mocked epochs. Mocked
epoch-selection checks are not training experiments.

Created: models/joint_cbm.py, training/joint_cbm.py, training/train_joint.py,
configs/joint_soft.toml, configs/joint_hard.toml, tests/test_joint_cbm.py,
docs/joint_preflight.json and this guide. Updated README and appended D017.

No implementation blocker remains; GPU execution and pretrained-weight download
are untested locally. Case-level independence remains unverifiable. Joint loss
can change concept semantics, so concept quality remains separately measured.
No OOF assembly, joint intervention experiment or locked-test evaluation is part
of this stage.
