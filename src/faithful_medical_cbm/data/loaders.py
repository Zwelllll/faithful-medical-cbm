"""Read-only loader factories for frozen IDs; never call split-generation code."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import load_config
from .dataset import DermoscopyDataset
from .transforms import build_transform


def seed_worker(worker_id: int) -> None:
    """Top-level callable for Windows spawn. PyTorch already sets each worker's torch seed."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class LoaderFactory:
    """Validate frozen manifests once; construct development/test loaders separately.

    Only IDs are read from the test manifest to enforce disjointness. Development
    datasets never parse test targets or resolve/open test image paths.
    """

    def __init__(self, config_path: str | Path = "configs/default.toml") -> None:
        self.config = load_config(config_path)
        cfg = self.config
        workers = cfg["data_loading"]["num_workers"]
        if type(workers) is not int or workers < 0 or type(cfg["data_loading"]["pin_memory"]) is not bool:
            raise ValueError("num_workers must be a nonnegative integer and pin_memory a boolean")
        self.cohort_path = (cfg["paths"]["processed"] / cfg["splitting"]["cohort"]).resolve()
        summary_path = (cfg["paths"]["artifacts"] / cfg["splitting"]["cohort_summary"]).resolve()
        self.image_root = (cfg["paths"]["raw"] / cfg["data_loading"]["image_root"]).resolve()
        for parent, child in ((cfg["paths"]["processed"], self.cohort_path),
                              (cfg["paths"]["artifacts"], summary_path),
                              (cfg["paths"]["raw"], self.image_root)):
            if parent not in child.parents:
                raise ValueError("Configured data path escapes its root")
        split_dir = cfg["paths"]["splits"]
        metadata = json.loads((split_dir / "split_metadata.json").read_text(encoding="utf-8"))
        if metadata["status"] != "frozen":
            raise ValueError("A frozen split is required")

        def checked_bytes(path: Path, expected: str) -> bytes:
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Frozen input hash mismatch: {path.name}")
            return data

        summary_bytes = checked_bytes(summary_path, metadata["inputs"]["stage2a_summary_sha256"])
        summary = json.loads(summary_bytes)
        cohort_bytes = checked_bytes(self.cohort_path, metadata["inputs"]["cohort_sha256"])
        if hashlib.sha256(cohort_bytes).hexdigest() != summary["output_sha256"]["cohort.csv"]:
            raise ValueError("Cohort differs from Stage 2A")
        self.concept_columns = tuple(summary["concept_target_order"])

        def read_manifest(name: str, header: list[str]) -> list[list[str]]:
            data = checked_bytes(split_dir / name, metadata["file_sha256"][name])
            reader = csv.reader(io.StringIO(data.decode("utf-8")), strict=True)
            if next(reader, []) != header:
                raise ValueError(f"Invalid manifest schema: {name}")
            rows = list(reader)
            if any(len(row) != len(header) for row in rows):
                raise ValueError(f"Malformed manifest: {name}")
            return rows

        self.development_ids = tuple(row[0] for row in read_manifest("development_ids.csv", ["case_num"]))
        self.test_ids = tuple(row[0] for row in read_manifest("test_ids.csv", ["case_num"]))
        fold_rows = read_manifest("development_folds.csv", ["case_num", "validation_fold"])
        all_ids = self.development_ids + self.test_ids
        if len(all_ids) != len(set(all_ids)) or any(not identifier.strip() for identifier in all_ids):
            raise ValueError("Duplicate, blank or overlapping development/test IDs")
        cohort_ids = [row["case_num"] for row in csv.DictReader(io.StringIO(cohort_bytes.decode("utf-8-sig")))]
        if len(cohort_ids) != len(set(cohort_ids)) or set(cohort_ids) != set(all_ids):
            raise ValueError("Frozen IDs must exactly cover the cohort")
        self.validation_folds = {row[0]: int(row[1]) for row in fold_rows}
        self.num_folds = metadata["settings"]["num_folds"]
        if self.num_folds != cfg["experiment"]["num_folds"]:
            raise ValueError("Configured folds differ from the frozen protocol")
        if (len(self.validation_folds) != len(fold_rows)
                or set(self.validation_folds) != set(self.development_ids)
                or set(self.validation_folds.values()) != set(range(self.num_folds))):
            raise ValueError("Invalid development-only fold assignments")
        if len(self.development_ids) != metadata["counts"]["development"]["total"] or len(
                self.test_ids) != metadata["counts"]["test"]["total"]:
            raise ValueError("Frozen split size mismatch")

    def _loader(self, ids: tuple[str, ...], *, training: bool, role: str,
                split: str = "development", allow_locked_test_iteration: bool = False) -> DataLoader:
        dataset = DermoscopyDataset(
            self.cohort_path, self.image_root, ids, self.concept_columns,
            build_transform(self.config, training=training), split=split, role=role,
            validation_folds=self.validation_folds if split == "development" else None,
            allow_locked_test_iteration=allow_locked_test_iteration,
        )
        settings = self.config["data_loading"]
        generator = torch.Generator().manual_seed(self.config["reproducibility"]["seed"])
        options = {"multiprocessing_context": "spawn"} if settings["num_workers"] else {}
        return DataLoader(dataset, batch_size=self.config["experiment"]["batch_size"],
                          shuffle=training, drop_last=False, num_workers=settings["num_workers"],
                          pin_memory=settings["pin_memory"], generator=generator,
                          worker_init_fn=seed_worker, **options)

    def development(self, *, training: bool = True) -> DataLoader:
        """All development cases; choose training=False for deterministic OOF-free inspection."""
        return self._loader(self.development_ids, training=training, role="train" if training else "evaluation")

    def fold(self, fold: int) -> dict[str, DataLoader]:
        """Train on the other folds; validate on the selected frozen fold only."""
        if type(fold) is not int or fold not in range(self.num_folds):
            raise ValueError("Invalid development fold")
        training_ids = tuple(i for i in self.development_ids if self.validation_folds[i] != fold)
        validation_ids = tuple(i for i in self.development_ids if self.validation_folds[i] == fold)
        return {"train": self._loader(training_ids, training=True, role="train"),
                "validation": self._loader(validation_ids, training=False, role="validation")}

    def locked_test(self, *, allow_locked_test_iteration: bool = False) -> DataLoader:
        """Construct lazily. Explicit opt-in is reserved for final frozen evaluation."""
        return self._loader(self.test_ids, training=False, role="evaluation", split="test",
                            allow_locked_test_iteration=allow_locked_test_iteration)
