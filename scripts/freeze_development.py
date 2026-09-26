"""Stage 14: metadata-only freeze; deliberately no ML/data-loader dependencies."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "artifacts/final_protocol/development_freeze_manifest.json"
FINAL_OUTPUTS = ("artifacts/final_test", "artifacts/locked_test", "artifacts/test_evaluation")


def read_bytes(root: Path, relative: str) -> bytes:
    """Narrow allowlist prevents accidental raw/cohort/test/checkpoint access."""
    path = (root / relative).resolve()
    rel = path.relative_to(root.resolve()).as_posix()
    allowed = (
        rel in {"data/splits/split_metadata.json", "data/splits/development_ids.csv",
                "README.md", "AGENTS.md", ".gitignore", ".gitattributes",
                "pyproject.toml", "requirements.txt"}
        or (rel.startswith(("configs/", "src/", "scripts/", "tests/", "docs/"))
            and path.suffix in {".py", ".toml", ".json", ".md"})
        or (rel.startswith(("artifacts/cbm/", "artifacts/joint_soft/",
                            "artifacts/joint_hard/", "artifacts/seven-concept-v1-"))
            and path.name in {"run.json", "summary.json", "full_development_model.json"})
    )
    if not allowed:
        raise ValueError(f"Stage 14 input not allowed: {rel}")
    return path.read_bytes()


def sha(root: Path, path: str) -> str:
    return hashlib.sha256(read_bytes(root, path)).hexdigest()


def load(root: Path, path: str) -> dict:
    return json.loads(read_bytes(root, path))


def ensure_no_final_outputs(root: Path) -> None:
    # Inspect directory existence only; never inspect any test result contents.
    for relative in FINAL_OUTPUTS:
        if (root / relative).exists():
            raise ValueError(f"Final-test output namespace already exists: {relative}")


def build_manifest(root: Path = ROOT) -> dict:
    ensure_no_final_outputs(root)
    config = load(root, "configs/final_protocol.json")
    split = load(root, "data/splits/split_metadata.json")
    if split["counts"]["cohort"]["total"] != 823 or split["counts"]["development"]["total"] != 658 or split["counts"]["test"] != {"total": 165, "positive": 50, "negative": 115}:
        raise ValueError("Frozen split counts changed")
    dev_bytes = read_bytes(root, "data/splits/development_ids.csv")
    if hashlib.sha256(dev_bytes).hexdigest() != split["file_sha256"]["development_ids.csv"]:
        raise ValueError("Development IDs hash mismatch")
    dev_ids = [row["case_num"] for row in csv.DictReader(dev_bytes.decode().splitlines())]
    if len(dev_ids) != 658 or len(set(dev_ids)) != 658:
        raise ValueError("Invalid development ID coverage")
    metadata_hashes = {}
    heads = {}
    for model, path in config["full_development_heads"].items():
        head = load(root, path)
        suffix = {"sequential_soft": "probability", "sequential_hard": "hard", "oracle": "target"}[model]
        label = {"sequential_soft": "SOFT CBM", "sequential_hard": "HARD CBM", "oracle": "ORACLE CONCEPT"}[model]
        if (head["label"] != f"FULL-DEVELOPMENT {label} HEAD"
            or head["validation_fold"] is not None or head["validation_ids"]
            or head["used_for_development_metrics"] is not False
            or len(head["training_ids"]) != 658 or set(head["training_ids"]) != set(dev_ids)
            or head["feature_columns"] != [f"{c}_{suffix}" for c in config["concept_order"]]
            or head["classes"] != [0, 1] or head["threshold"] != 0.5):
            raise ValueError(f"Not the frozen full-development head: {model}")
        metadata_hashes[path] = sha(root, path)
        heads[model] = {"path": path, "sha256": metadata_hashes[path], "label": head["label"], "training_cases": 658, "feature_columns": head["feature_columns"]}
    checkpoints = {}
    for model, epochs in config["selected_epochs"].items():
        checkpoints[model] = []
        for fold, epoch in enumerate(epochs):
            relative = f"checkpoints/{config['checkpoint_namespaces'][model]}/fold_{fold}/best.pt"
            record = {"fold": fold, "epoch": epoch, "expected_path_relative_to_drive_outputs": relative,
                      "binary_sha256": None, "binary_verified": False,
                      "selection_evidence": "user-frozen epochs; local baseline run unavailable"}
            run_dir = config["source_run_directories"][model]
            if run_dir:
                summary_path = f"{run_dir}/fold_{fold}/summary.json"
                summary = load(root, summary_path)
                if summary["best_epoch"] != epoch or summary["concept_order"] != config["concept_order"]:
                    raise ValueError(f"Frozen epoch/order mismatch: {summary_path}")
                record["reported_drive_path"] = summary.get("best_checkpoint")
                record["selection_evidence"] = summary_path
                for name in ("summary.json", "run.json"):
                    path = f"{run_dir}/fold_{fold}/{name}"
                    metadata_hashes[path] = sha(root, path)
            checkpoints[model].append(record)
    # Snapshot code/config/docs, without importing or executing any project pipeline.
    source_paths = {"README.md", "AGENTS.md", ".gitignore", ".gitattributes", "pyproject.toml", "requirements.txt"}
    for directory in ("src", "scripts", "tests", "configs", "docs"):
        source_paths.update(p.relative_to(root).as_posix() for p in (root / directory).rglob("*")
                            if p.is_file() and p.suffix in {".py", ".json", ".toml", ".md"})
    source_hashes = {path: sha(root, path) for path in sorted(source_paths)}
    git = lambda *args: subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    return {
        "schema_version": 1, "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "decisions_frozen_checkpoint_binary_verification_pending",
        "protocol": config,
        "locked_test_accessed": False, "final_test_executed": False,
        "git": {"commit": git("rev-parse", "HEAD"), "status_porcelain_at_freeze": git("status", "--short"),
                "source_state": "base commit plus SHA256 snapshot of current source/config/docs; no commit made"},
        "split": {"counts": split["counts"], "limitation": split["limitation"],
                  "metadata_path": "data/splits/split_metadata.json", "metadata_sha256": sha(root, "data/splits/split_metadata.json"),
                  "cohort_sha256": split["inputs"]["cohort_sha256"], "file_sha256": split["file_sha256"],
                  "hash_evidence": "cohort, test IDs and fold hashes copied from frozen split metadata; development IDs verified; cohort/test IDs not opened"},
        "checkpoints": checkpoints, "full_development_heads": heads,
        "model_metadata_sha256": metadata_hashes, "source_config_docs_sha256": source_hashes,
        "hash_algorithm": "SHA256 of exact file bytes; manifest itself excluded",
        "safety": {"final_output_namespaces_absent": list(FINAL_OUTPUTS), "scope": "metadata-only; no images, cohort rows, test IDs, checkpoints, loaders, inference or metrics read/executed"},
        "blockers": ["Verify all 16 Drive checkpoint binaries against frozen epochs, fold/run/model/order/config and training provenance; record binary hashes in a separate pre-execution attestation before test exposure.",
                     "Baseline source run metadata not locally available; verify it on Drive with epochs 15/9/21/10.",
                     "Final-test executor is not implemented or authorized by Stage 14; validate it offline against this frozen protocol before separately authorized execution."]
    }


def write_manifest(root: Path = ROOT) -> Path:
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError("Freeze manifest already exists; refusing overwrite")
    manifest = build_manifest(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return output


if __name__ == "__main__":
    print(write_manifest())
