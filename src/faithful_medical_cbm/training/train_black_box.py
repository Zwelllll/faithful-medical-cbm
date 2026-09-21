"""GPU-only entry point for one frozen development validation fold."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess

import torch

from ..config import load_config
from ..data.loaders import LoaderFactory
from ..models.black_box import BlackBoxEfficientNet
from ..reproducibility import seed_everything
from .baseline import development_fold_loaders, fit_fold, validate_settings


def run_training(config_path: Path, fold: int, run_name: str) -> dict:
    config_path = config_path.resolve()
    config = load_config(config_path)
    validate_settings(config["baseline"])
    if config["experiment"]["model_name"] != "efficientnet_b0":
        raise ValueError("This baseline implements EfficientNet-B0 only")
    if type(fold) is not int or fold not in range(config["experiment"]["num_folds"]):
        raise ValueError("Invalid validation fold")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", run_name):
        raise ValueError("run_name must be 1-64 letters, digits, underscores or hyphens")
    if not torch.cuda.is_available():
        raise RuntimeError("Full baseline training requires CUDA; use offline unit smoke tests on CPU")
    seed_everything(**config["reproducibility"])
    factory = LoaderFactory(config_path)
    loaders = development_fold_loaders(factory, fold)
    for loader in loaders.values():
        if set(loader.dataset.diagnoses.tolist()) != {0.0, 1.0}:
            raise ValueError("Each training/validation subset must contain both classes")
    artifact_dir = config["paths"]["artifacts"] / "baseline" / run_name / f"fold_{fold}"
    checkpoint_dir = config["paths"]["checkpoints"] / "baseline" / run_name / f"fold_{fold}"
    for output in (artifact_dir, checkpoint_dir):
        for key in ("raw", "processed", "splits"):
            protected = config["paths"][key]
            if output == protected or protected in output.parents or output in protected.parents:
                raise ValueError("Run outputs may not overlap protected data")
        if output.exists():
            raise FileExistsError(f"Run already exists; refusing to overwrite: {output}")
    if artifact_dir == checkpoint_dir:
        raise ValueError("Artifacts and checkpoints require separate directories")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    packages = {}
    for name in ("torch", "torchvision", "numpy", "scikit-learn", "Pillow", "timm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config_path.parent.parent,
                             capture_output=True, text=True, check=False)
        commit = git.stdout.strip() if git.returncode == 0 else None
    except OSError:
        commit = None
    provenance = {
        "validation_fold": fold, "seed": config["reproducibility"]["seed"],
        "training_ids": list(loaders["train"].dataset.case_nums),
        "validation_ids": list(loaders["validation"].dataset.case_nums),
        "cohort_sha256": sha(factory.cohort_path),
        "split_metadata_sha256": sha(config["paths"]["splits"] / "split_metadata.json"),
        "config_sha256": sha(config_path), "git_commit": commit,
        "source_sha256": {str(path.relative_to(Path(__file__).parents[1])): sha(path)
                          for path in sorted(Path(__file__).parents[1].rglob("*.py"))},
        "python": platform.python_version(), "packages": packages,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0), "weights": "EfficientNet_B0_Weights.IMAGENET1K_V1",
        "patient_independence": "Unverifiable: frozen split is case-level only",
        "locked_test_images_accessed": False,
    }
    # This is the only download-capable step; tests never execute it with real weights.
    model = BlackBoxEfficientNet(pretrained=True).to(torch.device("cuda"))
    artifact_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "run.json").write_text(json.dumps({"config": config, "provenance": provenance},
                                                     default=str, indent=2) + "\n", encoding="utf-8")
    return fit_fold(model, loaders, torch.device("cuda"), config, artifact_dir, checkpoint_dir, provenance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument("--fold", type=int, required=True, choices=range(4))
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    run_training(args.config, args.fold, args.run_name)


if __name__ == "__main__":
    main()
