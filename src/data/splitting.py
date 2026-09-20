"""Stage 2B: create once, then verify frozen case-level IDs without rewriting.

Run from the repository root: python -m src.data.splitting
This module reads processed Stage 2A metadata only. No raw data or images are opened.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.faithful_medical_cbm.config import load_config

FILES = ("development_ids.csv", "test_ids.csv", "development_folds.csv", "split_metadata.json")
LIMITATION = (
    "Case-level stratification only. Authoritative patient identifiers and reliable lesion linkage "
    "are unavailable; patient-level independence cannot be verified. case_num identifies a case, "
    "not a patient, and does not provide patient-level protection."
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_cohort(data: bytes) -> dict[str, int]:
    """Keep only permanent string IDs and binary diagnosis; reject malformed records."""
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")), strict=True)
    columns = reader.fieldnames or []
    if len(columns) != len(set(columns)) or not {"case_num", "diagnosis_binary"} <= set(columns):
        raise ValueError("Processed cohort requires unique headers, case_num and diagnosis_binary")
    labels = {}
    for row in reader:
        if None in row or None in row.values():
            raise ValueError("Malformed processed cohort row")
        identifier, label = row["case_num"], row["diagnosis_binary"]
        if not identifier.strip() or identifier in labels:
            raise ValueError("Cohort IDs must be nonmissing and unique")
        if label not in ("0", "1"):
            raise ValueError("Diagnosis must be exactly 0 or 1")
        labels[identifier] = int(label)
    if set(labels.values()) != {0, 1}:
        raise ValueError("Both diagnosis classes are required")
    return labels


def counts(ids: list[str], labels: dict[str, int]) -> dict[str, int]:
    positive = sum(labels[i] for i in ids)
    return {"total": len(ids), "positive": positive, "negative": len(ids) - positive}


def validate_assignments(labels: dict[str, int], development: list[str], test: list[str],
                         folds: list[tuple[str, int]], num_folds: int) -> dict:
    """Check all identity/coverage invariants and report training/validation counts."""
    dev_set, test_set = set(development), set(test)
    if len(dev_set) != len(development) or len(test_set) != len(test):
        raise ValueError("Duplicate IDs in development/test")
    if dev_set & test_set:
        raise ValueError("Development/test overlap")
    if dev_set | test_set != set(labels):
        raise ValueError("Development/test do not exactly cover the cohort")
    fold_ids = [i for i, _ in folds]
    if len(fold_ids) != len(set(fold_ids)):
        raise ValueError("A development case has multiple validation assignments")
    if set(fold_ids) & test_set:
        raise ValueError("Test IDs appear in development folds")
    if set(fold_ids) != dev_set:
        raise ValueError("Validation assignments must exactly cover development")
    if {f for _, f in folds} != set(range(num_folds)):
        raise ValueError("Invalid or missing validation fold")
    result = {"cohort": counts(sorted(labels), labels), "development": counts(development, labels),
              "test": counts(test, labels), "folds": []}
    for fold in range(num_folds):
        validation = [i for i, f in folds if f == fold]
        training = sorted(dev_set - set(validation))
        result["folds"].append({"fold": fold, "training": counts(training, labels),
                                 "validation": counts(validation, labels)})
    for subset in [result["development"], result["test"],
                   *(f[role] for f in result["folds"] for role in ("training", "validation"))]:
        if not subset["positive"] or not subset["negative"]:
            raise ValueError("Every split/fold must contain both classes")
    return result


def generate_assignments(labels: dict[str, int], seed: int, test_size: float,
                         num_folds: int) -> tuple[list[str], list[str], list[tuple[str, int]]]:
    """One deterministic split; no balance search. Canonical string sorting preserves IDs."""
    ids = sorted(labels)
    development, test = train_test_split(
        ids, test_size=test_size, random_state=seed, shuffle=True,
        stratify=[labels[i] for i in ids],
    )
    development, test = sorted(development), sorted(test)
    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    assignments = []
    for fold, (_, validation_positions) in enumerate(splitter.split(
            development, [labels[i] for i in development])):
        assignments.extend((development[int(position)], fold) for position in validation_positions)
    assignments.sort()
    validate_assignments(labels, development, test, assignments, num_folds)
    return development, test, assignments


def csv_bytes(header: list[str], rows: list[tuple]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def read_assignments(directory: Path) -> tuple[list[str], list[str], list[tuple[str, int]]]:
    def read(name, header):
        with (directory / name).open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            if next(reader, []) != header:
                raise ValueError(f"Unexpected frozen schema: {name}")
            rows = list(reader)
            if any(len(row) != len(header) for row in rows):
                raise ValueError(f"Malformed frozen rows: {name}")
            return rows
    development = [row[0] for row in read(FILES[0], ["case_num"])]
    test = [row[0] for row in read(FILES[1], ["case_num"])]
    folds = [(row[0], int(row[1])) for row in read(FILES[2], ["case_num", "validation_fold"])]
    return development, test, folds


def freeze_splits(config_path: Path) -> tuple[dict, str]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    seed = config["reproducibility"]["seed"]
    num_folds = config["experiment"]["num_folds"]
    test_size = config["splitting"]["test_size"]
    if num_folds != 4 or test_size != 0.20:
        raise ValueError("Frozen research design requires four folds and a 20% test fraction")
    processed = config["paths"]["processed"]
    artifacts = config["paths"]["artifacts"]
    cohort_path = (processed / config["splitting"]["cohort"]).resolve()
    summary_path = (artifacts / config["splitting"]["cohort_summary"]).resolve()
    output = config["paths"]["splits"]
    if processed not in cohort_path.parents or artifacts not in summary_path.parents:
        raise ValueError("Inputs must stay inside processed metadata and artifact directories")
    for protected in (processed, artifacts, config["paths"]["raw"]):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Split outputs must be separate from raw/processed data and artifacts")
    cohort_bytes = cohort_path.read_bytes()
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    if sha256(cohort_bytes) != summary["output_sha256"]["cohort.csv"]:
        raise ValueError("Processed cohort does not match its Stage 2A hash")
    labels = read_cohort(cohort_bytes)
    if counts(sorted(labels), labels) != {"total": summary["cohort_size"],
            "positive": summary["positive_cases"], "negative": summary["negative_cases"]}:
        raise ValueError("Processed cohort counts differ from Stage 2A")
    settings = {"seed": seed, "test_size": test_size, "num_folds": num_folds,
                "identifier": "case_num", "stratification_target": "diagnosis_binary",
                "input_order": "lexicographic case_num string order",
                "split_method": "sklearn.model_selection.train_test_split",
                "fold_method": "sklearn.model_selection.StratifiedKFold",
                "shuffle": True, "fold_numbering": "0-based"}
    inputs = {"cohort_sha256": sha256(cohort_bytes), "stage2a_summary_sha256": sha256(summary_bytes)}
    present = [(output / name).exists() for name in FILES]
    if any(present):
        if not all(present):
            raise ValueError("Partial frozen split exists; refusing to regenerate or overwrite")
        metadata = json.loads((output / FILES[3]).read_text(encoding="utf-8"))
        if metadata["settings"] != settings or metadata["inputs"] != inputs:
            raise ValueError("Frozen inputs/settings differ; refusing to alter the split")
        for name in FILES[:3]:
            if sha256((output / name).read_bytes()) != metadata["file_sha256"][name]:
                raise ValueError(f"Frozen file hash mismatch: {name}; refusing to overwrite")
        actual = validate_assignments(labels, *read_assignments(output), num_folds)
        if actual != metadata["counts"]:
            raise ValueError("Frozen split/fold count mismatch")
        return metadata, "verified_existing_no_writes"

    development, test, folds = generate_assignments(labels, seed, test_size, num_folds)
    reports = validate_assignments(labels, development, test, folds, num_folds)
    payloads = {FILES[0]: csv_bytes(["case_num"], [(i,) for i in development]),
                FILES[1]: csv_bytes(["case_num"], [(i,) for i in test]),
                FILES[2]: csv_bytes(["case_num", "validation_fold"], folds)}
    metadata = {
        "stage": "2B", "status": "frozen", "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": settings, "inputs": inputs, "counts": reports, "limitation": LIMITATION,
        "test_policy": "Locked test is reserved for final frozen evaluation; never use it during development, tuning, calibration, early stopping or debugging.",
        "original_derm7pt_indexes_used": False, "raw_metadata_or_images_read": False,
        "split_candidates_generated": 1,
        "test_rounding": "ceil(cohort_size * test_size); development receives remaining cases",
        "file_sha256": {name: sha256(data) for name, data in payloads.items()},
        "provenance": {"python": platform.python_version(), "numpy": np.__version__,
                       "scikit_learn": sklearn.__version__, "source_sha256": sha256(Path(__file__).read_bytes()),
                       "config_sha256": sha256(config_path.read_bytes())},
    }
    payloads[FILES[3]] = (json.dumps(metadata, indent=2) + "\n").encode("utf-8")
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: any existing file aborts; there is intentionally no overwrite/force mode.
    # An interrupted partial write must be reviewed manually, never silently regenerated.
    for name, data in payloads.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    return metadata, "created_and_frozen"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    args = parser.parse_args()
    metadata, status = freeze_splits(args.config)
    print(json.dumps({"status": status, "settings": metadata["settings"],
                      "counts": metadata["counts"], "limitation": metadata["limitation"]}, indent=2))


if __name__ == "__main__":
    main()
