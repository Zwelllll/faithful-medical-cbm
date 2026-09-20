"""Selected processed cases only, with pre-parsed targets and dermoscopic paths."""
from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset


class DermoscopyDataset(Dataset):
    """No images are opened at construction; raw categorical fields are never mapped.

    Use the loader factories for frozen split/hash validation. Direct construction
    supports synthetic fixtures and explicit subsets. Test sample access is guarded
    independently of transforms and worker count.
    """

    def __init__(self, cohort_path: Path, image_root: Path, case_nums: Sequence[str],
                 concept_columns: Sequence[str], transform: Callable, *,
                 split: str = "development", role: str = "evaluation",
                 validation_folds: dict[str, int] | None = None,
                 allow_locked_test_iteration: bool = False) -> None:
        if split not in ("development", "test") or role not in ("train", "validation", "evaluation"):
            raise ValueError("Invalid split/role")
        if split == "test" and role != "evaluation":
            raise ValueError("Locked test supports evaluation only")
        self.case_nums = tuple(case_nums)
        if not self.case_nums or len(set(self.case_nums)) != len(self.case_nums) or any(
                not isinstance(i, str) or not i.strip() for i in self.case_nums):
            raise ValueError("Selected case_num values must be nonempty unique strings")
        self.concept_columns = tuple(concept_columns)
        if len(self.concept_columns) != 7 or len(set(self.concept_columns)) != 7:
            raise ValueError("Exactly seven distinct processed concept columns are required")
        root = Path(image_root).resolve()
        selected = set(self.case_nums)
        records = {}
        with Path(cohort_path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            columns = reader.fieldnames or []
            required = {"case_num", "diagnosis_binary", "derm_path", *self.concept_columns}
            if not required <= set(columns) or len(columns) != len(set(columns)):
                raise ValueError("Missing or duplicate processed metadata columns")
            for row in reader:
                identifier = row["case_num"]
                if identifier not in selected:
                    continue  # Do not inspect targets/paths for unselected (e.g. test) rows.
                if identifier in records:
                    raise ValueError(f"Duplicate selected case: {identifier}")
                values = [row["diagnosis_binary"], *(row[c] for c in self.concept_columns)]
                if any(value not in ("0", "1") for value in values):
                    raise ValueError(f"Nonbinary processed targets for case {identifier}")
                reference = row["derm_path"]
                if not reference or "\\" in reference or ":" in reference:
                    raise ValueError("derm_path must be a relative POSIX path")
                relative = PurePosixPath(reference)
                path = (root / reference).resolve()
                if relative.is_absolute() or ".." in relative.parts or root not in path.parents:
                    raise ValueError("derm_path escapes image root")
                records[identifier] = (path, [int(v) for v in values])
        if set(records) != selected:
            raise ValueError(f"Selected IDs missing from processed cohort: {sorted(selected - set(records))}")
        self.paths = tuple(records[i][0] for i in self.case_nums)
        targets = torch.tensor([records[i][1] for i in self.case_nums], dtype=torch.float32)
        self.diagnoses, self.concepts = targets[:, 0], targets[:, 1:]
        self.split, self.role, self.transform = split, role, transform
        self.allow_locked_test_iteration = allow_locked_test_iteration
        if validation_folds is not None and not selected <= set(validation_folds):
            raise ValueError("Missing selected validation-fold assignments")
        self.validation_folds = tuple(validation_folds[i] if validation_folds is not None else -1
                                      for i in self.case_nums)

    def __len__(self) -> int:
        return len(self.case_nums)

    def __getitem__(self, index: int) -> dict:
        if self.split == "test" and not self.allow_locked_test_iteration:
            raise RuntimeError("Locked-test iteration is disabled; enable only for final frozen evaluation")
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return {"image": tensor, "diagnosis": self.diagnoses[index].clone(),
                "concepts": self.concepts[index].clone(), "case_num": self.case_nums[index],
                "split": self.split, "role": self.role, "validation_fold": self.validation_folds[index]}
