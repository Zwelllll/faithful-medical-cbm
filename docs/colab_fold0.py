"""Stage 4B setup checks only; never iterate loaders or construct a model."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def digest(path: Path, mode: str) -> str:
    data = path.read_bytes()
    if mode == "lf":
        data = data.replace(b"\r\n", b"\n")
    elif mode != "bytes":
        raise ValueError("Unknown hash mode")
    return hashlib.sha256(data).hexdigest()


def verify_manifest(repo: Path) -> dict:
    manifest = json.loads((repo / "docs/colab_fold0_manifest.json").read_text())
    for name, record in manifest["files"].items():
        if digest(repo / name, record["mode"]) != record["sha256"]:
            raise ValueError(f"Frozen artifact/config/source hash mismatch: {name}")
    return manifest


def link_directory(link: Path, target: Path) -> None:
    """Replace only an empty Git placeholder, never an existing data directory."""
    target = target.resolve(strict=True)
    if not target.is_dir():
        raise ValueError(f"Not a directory: {target}")
    if link.is_symlink():
        if link.resolve() != target:
            raise ValueError(f"Existing link has a different destination: {link}")
        return
    if link.exists():
        if not link.is_dir() or any(p.name != ".gitkeep" for p in link.iterdir()):
            raise ValueError(f"Refusing to replace existing content: {link}")
        (link / ".gitkeep").unlink(missing_ok=True)
        link.rmdir()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def wire_storage(repo: Path, dataset: Path, outputs: Path) -> None:
    dataset, outputs = dataset.resolve(strict=True), outputs.resolve()
    if dataset == outputs or dataset in outputs.parents or outputs in dataset.parents:
        raise ValueError("Dataset and output roots must be separate")
    link_directory(repo / "data/raw", dataset / "raw")
    link_directory(repo / "data/processed", dataset / "processed")
    for name in ("artifacts", "checkpoints"):
        target = outputs / name / "baseline"
        target.mkdir(parents=True, exist_ok=True)
        link_directory(repo / name / "baseline", target)


def verify_development_paths(loaders: dict, image_root: Path) -> dict:
    """Check selected development paths only, without opening any image."""
    root = image_root.resolve()
    forbidden = {"test", "locked_test", "locked-test"}
    if forbidden.intersection(p.lower() for p in root.parts):
        raise ValueError("Locked-test image root is forbidden")
    result = {}
    for role, loader in loaders.items():
        ds = loader.dataset
        if ds.split != "development" or ds.role != role:
            raise ValueError("Only development loaders are allowed")
        missing = []
        for path in ds.paths:
            resolved = path.resolve()
            if root not in resolved.parents or forbidden.intersection(p.lower() for p in resolved.parts):
                raise ValueError("Development image points outside its root or into a test directory")
            if not resolved.is_file():
                missing.append(str(path))
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing {role} images: {missing[:5]}")
        positive = int(ds.diagnoses.sum().item())
        result[role] = {"total": len(ds.case_nums), "positive": positive,
                        "negative": len(ds.case_nums) - positive}
    return result


def preflight(repo: Path, outputs: Path) -> dict:
    verify_manifest(repo)
    from faithful_medical_cbm.data.loaders import LoaderFactory
    from faithful_medical_cbm.training.baseline import development_fold_loaders

    factory = LoaderFactory(repo / "configs/default.toml")
    loaders = development_fold_loaders(factory, 0)
    counts = verify_development_paths(loaders, factory.image_root)
    expected = {"train": {"total": 493, "positive": 148, "negative": 345},
                "validation": {"total": 165, "positive": 50, "negative": 115}}
    if counts != expected:
        raise ValueError(f"Unexpected frozen Fold 0 counts: {counts}")
    for name in ("artifacts", "checkpoints"):
        parent = repo / name / "baseline"
        if parent.resolve() != (outputs / name / "baseline").resolve():
            raise ValueError("Outputs are not linked to the requested persistent location")
        if (parent / "baseline-v1/fold_0").exists():
            raise FileExistsError("Run already exists; do not overwrite or automatically restart it")
    return {"fold": 0, "counts": counts, "hashes_verified": True,
            "locked_test_images_accessed": False, "loaders_iterated": False,
            "patient_level_independence": "cannot be verified"}


def summarize(outputs: Path) -> dict:
    artifacts = outputs / "artifacts/baseline/baseline-v1/fold_0"
    checkpoints = outputs / "checkpoints/baseline/baseline-v1/fold_0"
    if not (artifacts / "summary.json").is_file():
        raise RuntimeError("Run is incomplete or not started; no final summary. Preserve partial outputs; automatic resume is unsupported.")
    summary = json.loads((artifacts / "summary.json").read_text())
    with (artifacts / "history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    if not history:
        raise ValueError("Missing epoch history")
    paths = {name: str(checkpoints / f"{name}.pt") for name in ("best", "last")}
    if not all(Path(p).is_file() for p in paths.values()):
        raise FileNotFoundError("Run summary exists but a checkpoint is missing")
    return {**summary, "final_train_loss": float(history[-1]["train_loss"]),
            "final_validation_loss": float(history[-1]["validation_loss"]),
            "checkpoint_paths": paths}
