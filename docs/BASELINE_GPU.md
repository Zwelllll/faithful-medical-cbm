# Stage 4: black-box baseline execution

For the first Fold 0 run, use [the Stage 4B Colab notebook](../notebooks/colab_fold0.ipynb).
It clones a pinned Git commit, mounts Google Drive, installs dependencies while preserving
CUDA PyTorch, verifies the frozen artifact/config/source manifest, and connects outputs
to persistent storage. Fill in your Git URL, commit SHA and Drive paths. No Git remote
was configured locally when preparing this workflow; publish the code to your own
repository first, excluding raw/processed data. The notebook contains the dataset layout,
preflight checks, one explicit training cell, and a post-run summary. The companion
`colab_fold0.py` checks paths without opening images or iterating loaders.
The Stage 2A summary was originally hashed with CRLF line endings. `.gitattributes`
preserves those checkout bytes on Linux as well; frozen split files remain LF.
The cohort must be transferred byte-for-byte from local storage, not resaved in an editor.

Infrastructure is implemented; no real training run has been performed. CPU checks
use random weights and synthetic tensors. Their outputs are not research results.

## Cloud/Colab setup

1. Select a CUDA GPU runtime. Copy or clone this repository to the runtime and enter
   the repository directory containing AGENTS.md. In Colab, use `%cd` to set that directory.
2. Transfer the existing `data/processed/stage2a/cohort.csv`,
   `artifacts/stage2a/summary.json`, and all four frozen `data/splits/` files unchanged.
   Processed data is ignored by Git, so cloning alone is insufficient. Transfer files
   as binary copies to preserve hashes and line endings. Do not rerun earlier stages.
3. Place development dermoscopic images at `data/raw/release_v0/images/`, preserving
   the subdirectories referenced by processed derm_path. Only development images are
   needed. Keep locked-test images offline. Raw metadata and original Derm7pt index
   files are not needed by this trainer.
4. Install the package using the runtime's compatible CUDA PyTorch/torchvision pair:

   ```bash
   python -m pip install -e .
   python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
   ```

   In Colab, prefix shell commands with `!`. The first actual model construction may
   download torchvision's `EfficientNet_B0_Weights.IMAGENET1K_V1` to the torch cache.
   Internet access or an already populated weight cache is needed. Offline tests never
   download weights. Do not install the local CPU-only environment onto the GPU runtime.
5. Preserve the project/run output directories on persistent storage or copy artifacts
   and checkpoints off the runtime before it is reclaimed. Relative paths resolve from
   the parent of configs/, not from the terminal's working directory.

## Exact first-run command

From the repository root after installation:

```bash
python -m faithful_medical_cbm.training.train_black_box --config configs/default.toml --fold 0 --run-name baseline-v1
```

This trains on 493 development cases and validates on the 165 cases of validation
fold 0. It never builds the locked-test loader. The GPU-only entry point rejects CPU
execution before creating a model, downloading weights or opening images.
For later requested runs, choose folds 1, 2 or 3 explicitly; there is no automatic
multi-fold sweep. Do not choose a fold or seed based on locked-test performance.

## Model and initial configuration

ImageNet-pretrained EfficientNet-B0 features -> adaptive average pooling -> dropout
(torchvision B0 default 0.2) -> linear 1280-to-1 classifier. Forward returns `[B]`
logits. BCEWithLogits is unweighted; sigmoid is applied only for prediction reporting.
The batch's concepts and other metadata never enter the model or loss.

Existing Stage 3 transforms, ImageNet normalization, image size 224, batch size 32,
workers 0 and pin_memory=false are reused without change. The initial training
settings live in `[baseline]` of configs/default.toml:

| Setting | Initial value |
|---|---|
| optimizer | AdamW (Adam and SGD also supported) |
| head learning rate | 0.001 |
| fine-tuning learning rate | 0.0001 |
| weight decay | 0.0001 |
| maximum epochs | 30 |
| head-only epochs | 3 |
| final feature children unfrozen afterward | 3 (indices 6, 7, 8, including the final convolution) |
| early-stopping patience | 7 validation epochs without improvement |
| minimum AUROC improvement | 0.0; ties are not improvements |
| seed | 42 |

Frozen feature blocks stay in evaluation mode so BatchNorm statistics and stochastic
depth do not drift. At the fine-tuning transition the optimizer is rebuilt and its
momentum/adaptive state is intentionally reset. Early stopping continues across phases.
These are starting engineering settings, not validated hyperparameters. Full training
uses float32; AMP and a learning-rate scheduler are not implemented.

## Metrics and output files

Every epoch reports training BCE, validation BCE and validation AUROC. Losses are
weighted by sample count when aggregating batches. AUROC is computed over all cases
in the chosen validation fold, using raw logits as ranking scores to avoid sigmoid
saturation; sigmoid probabilities are also saved. Both classes must be present.
There is no threshold selection or calibration fitting.

Before calculating AUROC, save raw case_num, diagnosis, logit and probability to
`artifacts/baseline/baseline-v1/fold_0/validation_epoch_NNN.csv`.
The same directory contains `history.csv`, `run.json` (configuration, permanent
training/validation IDs, seed, input/source hashes, Git commit and runtime versions),
and the final development-only `summary.json`.

`checkpoints/baseline/baseline-v1/fold_0/best.pt` is selected only by validation AUROC;
`last.pt` records the most recent completed epoch. Checkpoint files are replaced
atomically within a new run directory. A preexisting run directory is rejected,
preventing accidental overwrite of earlier experiments.

Each checkpoint contains model and optimizer states, model configuration/trainable
blocks, complete configuration, epoch, early-stopping state, metrics and provenance.
Restore without downloading weights:

```python
from pathlib import Path
from faithful_medical_cbm.models.black_box import BlackBoxEfficientNet
from faithful_medical_cbm.training.baseline import load_checkpoint

model = BlackBoxEfficientNet(pretrained=False)
record = load_checkpoint(Path("checkpoints/baseline/baseline-v1/fold_0/best.pt"), model)
model.eval()
```

Optional optimizer restoration is supported when configured for the saved trainable
blocks first. Exact interrupted-run resume is not implemented: RNG and loader iterator
positions are not restored, and there is no resume CLI. Loading a checkpoint does not
authorize locked-test evaluation.

## Remaining first-run checks

The pretrained download and CUDA execution have not been exercised locally. Ensure
the GPU runtime imports a compatible torch/torchvision pair, can obtain the weights,
and has the unchanged processed metadata, manifests and development images. The
trainer fails clearly if these conditions are unmet. No local research AUROC exists.
The study remains case-level; patient-level independence cannot be verified.
