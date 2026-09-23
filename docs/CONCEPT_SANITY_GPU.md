# Stage 5: one-concept, Fold 0 sanity run

Only `atypical_pigment_network` is predicted, using its unchanged processed Stage 2A
target (raw pigment_network == atypical). No diagnosis prediction, other concept
heads, OOF assembly, threshold search, or test evaluation is implemented here.

The user reported external development baseline AUROCs of 0.8833, 0.8708, 0.9207,
0.8673 (mean 0.8855, sample SD 0.0244). Those run artifacts are not available locally
for verification. These values were not used to tune Stage 5. The user reports that
the locked test remains untouched.

## Data and weighting

The existing LoaderFactory verifies the frozen cohort/summary/split hashes and
development ID isolation before constructing Fold 0 loaders. Only processed concept
targets are selected; categorical labels are never parsed in training. The configured
seven-column order must match the hash-verified Stage 2A order. No image is needed
to calculate the following counts (also in concept_sanity_preflight.json):

| Fold 0 subset | Total | Positive | Negative | Prevalence |
|---|---:|---:|---:|---:|
| Training | 493 | 129 | 364 | 26.1663% |
| Validation | 165 | 48 | 117 | 29.0909% |

Training `pos_weight = 364 / 129 = 2.8217054263565893`, computed again at runtime
from the training Dataset's preloaded concept tensor only. Validation/test labels
never contribute. Both training and validation losses use this same training-derived
weight; validation loss is therefore weighted BCE, not unweighted BCE. Epoch losses
are sample-weighted means. Both classes must exist; otherwise the run fails.

## Model and configuration

ImageNet EfficientNet-B0 -> adaptive average pooling -> dropout 0.2 -> linear
1280-to-1 concept head, returning `[B]` logits. The model reuses the baseline's B0
construction/freezing mechanics but has its own task identity, named concept output,
training loop and checkpoint schema. There is no diagnostic head or concept input
to the model. A future multi-head implementation can extend the explicit ordered
output schema; seven-head training is not implemented.

`configs/concept_sanity.toml` is independent of `configs/default.toml`. Its initial
settings preserve the baseline's existing schedule, without using baseline performance:
seed 42; 224px; batch 32; workers 0; pin_memory=false; existing augmentations/ImageNet
normalization; AdamW; weight decay 0.0001; 30 maximum epochs. Train only the head for
3 epochs at LR 0.001, then unfreeze the last 3 feature children at LR 0.0001. Frozen
blocks remain in eval mode. The optimizer is rebuilt at the phase transition.

Early stopping and best checkpoint selection use validation AUROC, patience 7,
minimum improvement 0.0. Also report binary Macro-F1 over labels [0,1], with
probability >= 0.5 positive and zero_division=0. No threshold optimization occurs.
AUROC uses logits to preserve ranking when sigmoid saturates. Raw validation
probabilities, logits and ground-truth concept labels are saved before either metric.

## Colab/GPU execution

1. Push these changes and clone/checkout the new pinned commit in a GPU runtime.
   Install a compatible CUDA torch/torchvision pair and `python -m pip install -e .`.
   Confirm `torch.cuda.is_available()` and print the GPU model. The CLI refuses CPU
   execution before loader/model construction. Weights may download on the first run.
2. Mount Drive. Expose the unchanged processed cohort at
   `data/processed/stage2a/cohort.csv` and development dermoscopic files at
   `data/raw/release_v0/images/<derm_path>`. Keep test images offline. Retain tracked
   Stage 2A summary and frozen split files. Do not rerun earlier stages.
3. Reuse the Stage 4 notebook's dependency and Drive setup only. Do not execute its
   baseline training cell or baseline-run preflight (completed baseline directories
   intentionally fail that preflight). If needed, use `docs/colab_fold0.py`'s
   `link_directory` to connect data/raw and data/processed as whole directory roots.
4. Link separate concept output parents to Drive BEFORE running, from the repository:

   ```python
   from pathlib import Path
   import sys
   sys.path.insert(0, str(Path('docs').resolve()))
   from colab_fold0 import link_directory
   outputs = Path('/content/drive/MyDrive/faithful-cbm/outputs')
   for name in ('artifacts', 'checkpoints'):
       target = outputs / name / 'concept_sanity'
       target.mkdir(parents=True, exist_ok=True)
       link_directory(Path(name) / 'concept_sanity', target)
   ```

   Do not create the final run directory. Existing runs are rejected, never overwritten.
5. From the repository root run exactly:

   ```bash
   python -m faithful_medical_cbm.training.train_concept --config configs/concept_sanity.toml --fold 0 --run-name concept-sanity-v1
   ```

Only Fold 0 is accepted. Use the fresh Stage 5 config; do not change the baseline
config, split files or mappings. This is the first real concept run, not a four-fold
sweep. No local full training or GPU/weight download check has been performed.

## Outputs and restore

Under `artifacts/concept_sanity/concept-sanity-v1/fold_0/`: `run.json` (resolved config,
target/order, train-only weight, subset counts, permanent IDs, hashes, package/GPU
versions and Git commit), `validation_epoch_NNN.csv`, `history.csv`, `summary.json`.
Under `checkpoints/concept_sanity/concept-sanity-v1/fold_0/`: `best.pt`, `last.pt`.
With the links above these persist on Drive separately from baseline outputs.

Checkpoints include model/optimizer states, task/output schema, trainable blocks,
config, metrics, early-stopping state and provenance. Instantiate
`ConceptEfficientNet(pretrained=False)` and use `training.concept.load_checkpoint`
to restore offline; an incompatible task or concept name is rejected. Exact resumed
RNG/loader state is not implemented; preserve interrupted artifacts and decide recovery
separately rather than overwriting them.

Review development training/validation curves and saved predictions after this run.
Do not proceed to seven heads or OOF generation without the next stage instruction.
Patient-level independence still cannot be verified for the case-level split.
