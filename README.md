# Faithful Medical CBM

Research on faithful and intervenable concept bottleneck models for dermoscopic
melanoma classification. The frozen design is in [PROJECT_SPEC](docs/PROJECT_SPEC.md);
engineering constraints are in [AGENTS.md](AGENTS.md).

## Status

Stage 00 only: repository initialization. No dataset has been downloaded, loaded,
preprocessed or split; no models or training pipeline have been implemented.
The starter lives in this directory, one level below the supplied workspace root.

## Setup (PowerShell, Python 3.11+)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/report_environment.py
```

Use a mutually compatible PyTorch/torchvision installation appropriate to the
machine's CUDA runtime, or a CPU build. Dependency lower bounds are not a tested
lockfile. Record the resolved environment before experiments with
`python -m pip freeze > artifacts/requirements-resolved.txt` and
`python scripts/report_environment.py > artifacts/environment.json`.
Torchvision is the planned augmentation library; no transforms are implemented yet.
Tests use Python's standard-library unittest framework. RNG tests skip explicitly
when NumPy or PyTorch is absent. No lint tool is configured.

## Configuration and reproducibility

`configs/default.toml` centralizes seeds, paths, image size, batch size, folds,
model name and ordered concept names. Seed 42, image size 224 and batch size 32
are provisional engineering defaults, not tuned experimental choices.
Paths resolve relative to the parent of the configuration directory, independent
of the shell's working directory. Keep configs directly in `configs/`.

```python
from faithful_medical_cbm.config import load_config
from faithful_medical_cbm.reproducibility import seed_everything

config = load_config("configs/default.toml")
seed_everything(**config["reproducibility"])
```

Call seeding before CUDA initialization. Strict deterministic algorithms may
reject unsupported operations. Reproducibility across library versions/hardware
is not guaranteed. DataLoader worker seeding will be added at the DataLoader stage.

## Layout

- `src/faithful_medical_cbm/`: importable package; data, models, training,
  interventions and evaluation subpackages reserved for later stages.
- `configs/`: central configuration.
- `scripts/`: environment reporting.
- `tests/`: initialization checks.
- `data/raw/`, `data/processed/`, `data/splits/`: empty data placeholders.
- `artifacts/`, `checkpoints/`, `notebooks/`: empty output/work placeholders.
- `docs/`: authoritative specification and append-only decision record.

No metadata schema, diagnosis mapping or grouping identifier is assumed.
Dataset audit is the next stage and requires a separate instruction.
