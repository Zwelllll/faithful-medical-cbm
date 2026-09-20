"""Stage 2A only: exact configured cohort selection and categorical mapping.

Run: python -m src.data.cohort
No split indexes, randomization, folds, model imports or image transforms.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
from typing import Any

from .audit import digest, read_metadata, resolve_reference, write_csv


def load_mapping(path: Path) -> dict[str, Any]:
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate configuration key: {key}")
            result[key] = value
        return result

    config = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
    if set(config["diagnoses"].values()) != {"positive", "negative", "exclude"}:
        raise ValueError("Diagnosis mappings require positive, negative and exclude actions")
    concepts = config["concepts"]
    if len(concepts) != 7 or len({c["target"] for c in concepts}) != 7:
        raise ValueError("Exactly seven distinct concept targets are required")
    if len({c["source"] for c in concepts}) != 7:
        raise ValueError("Exactly seven distinct raw concept sources are required")
    for concept in concepts:
        if not concept["values"] or any(type(v) is not int or v not in (0, 1)
                                        for v in concept["values"].values()):
            raise ValueError("Concept mappings must contain explicit binary integers")
    corrections = config["derm_path_corrections"]
    if len({c["case_num"] for c in corrections}) != len(corrections):
        raise ValueError("Duplicate path correction case numbers")
    return config


def map_row(row: dict[str, str], config: dict[str, Any]) -> tuple[int | None, dict[str, int]]:
    """Map exact raw strings; excluded diagnoses return None, never zero."""
    label = row[config["diagnosis_column"]]
    if label not in config["diagnoses"]:
        raise ValueError(f"Unmapped diagnosis: {label!r}")
    action = config["diagnoses"][label]
    diagnosis = {"positive": 1, "negative": 0, "exclude": None}[action]
    targets = {}
    for concept in config["concepts"]:
        value = row[concept["source"]]
        if value not in concept["values"]:
            raise ValueError(f"Unmapped {concept['source']}: {value!r}")
        targets[concept["target"]] = concept["values"][value]
    return diagnosis, targets


def construct_rows(columns: list[str], rows: list[dict[str, str]], config: dict[str, Any],
                   available_images: set[str]) -> tuple[list[dict], list[dict], list[dict]]:
    """Preserve raw columns verbatim and normalize only explicitly configured paths."""
    required = {"case_num", "derm", config["diagnosis_column"],
                *(c["source"] for c in config["concepts"])}
    if missing := required - set(columns):
        raise ValueError(f"Missing required raw columns: {sorted(missing)}")
    added = [config["diagnosis_target"], *(c["target"] for c in config["concepts"]),
             "derm_path", "exclusion_reason"]
    if set(added) & set(columns) or len(added) != len(set(added)):
        raise ValueError("Output columns must not overwrite raw columns or one another")
    ids = [r["case_num"] for r in rows]
    if not rows or any(not i.strip() for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Case numbers must be nonmissing and unique")
    corrections = {c["case_num"]: c for c in config["derm_path_corrections"]}
    included, excluded, changes = [], [], []
    for row in rows:
        diagnosis, targets = map_row(row, config)
        normalized = row["derm"]
        if correction := corrections.get(row["case_num"]):
            if normalized != correction["raw"]:
                raise ValueError("Configured path correction does not match the raw reference")
            normalized = correction["normalized"]
            changes.append({**correction, "included": diagnosis is not None})
        status, _ = resolve_reference(normalized, available_images)
        if status != "exact":
            raise ValueError(f"Unresolved dermoscopic path for case {row['case_num']}: {normalized!r} ({status})")
        if diagnosis is None:
            excluded.append({**row, "exclusion_reason": "excluded_by_frozen_primary_cohort"})
        else:
            included.append({**row, config["diagnosis_target"]: diagnosis, **targets, "derm_path": normalized})
    return included, excluded, changes


def build_cohort(raw: Path, mapping: Path, audit_dir: Path, processed: Path,
                 artifacts: Path) -> dict[str, Any]:
    raw, mapping, audit_dir, processed, artifacts = (
        p.resolve() for p in (raw, mapping, audit_dir, processed, artifacts))
    for output in (processed, artifacts):
        for protected in (raw, audit_dir):
            if output == protected or protected in output.parents or output in protected.parents:
                raise ValueError("Outputs must be separate from raw data and Stage 1 evidence")
    if processed == artifacts or processed in artifacts.parents or artifacts in processed.parents:
        raise ValueError("Processed and artifact directories must be separate")
    config = load_mapping(mapping)
    _, manifest = read_metadata(audit_dir / "raw_manifest.csv")
    expected = {m["path"]: m["sha256"] for m in manifest
                if m["path"] == "meta/meta.csv" or m["path"].startswith("images/")}
    # Only metadata and images are read; original train/valid/test index files are never opened.
    def fingerprint() -> dict[str, str]:
        paths = [raw / "meta/meta.csv", *sorted((raw / "images").rglob("*"))]
        return {p.relative_to(raw).as_posix(): digest(p) for p in paths if p.is_file()}

    before = fingerprint()
    if before != expected:
        raise ValueError("Raw metadata/images differ from the Stage 1 audit; re-audit before mapping")
    columns, rows = read_metadata(raw / "meta/meta.csv")
    stage1 = json.loads((audit_dir / "summary.json").read_text(encoding="utf-8"))
    if set(config["diagnoses"]) != set(stage1["diagnosis_counts"]):
        raise ValueError("Diagnosis configuration must cover exactly the audited label vocabulary")
    for concept in config["concepts"]:
        if set(concept["values"]) != set(stage1["concept_counts"][concept["source"]]):
            raise ValueError(f"Mapping must cover exactly the audited vocabulary for {concept['source']}")
    available = {p.removeprefix("images/") for p in before if p.startswith("images/")}
    included, excluded, changes = construct_rows(columns, rows, config, available)
    if not included:
        raise ValueError("The configured cohort is empty")
    if before != fingerprint():
        raise RuntimeError("Raw metadata/images changed during cohort construction")
    positives = sum(r[config["diagnosis_target"]] for r in included)
    concept_counts = {
        c["target"]: {"positive": sum(r[c["target"]] for r in included),
                      "negative": sum(1 - r[c["target"]] for r in included),
                      "prevalence": sum(r[c["target"]] for r in included) / len(included)}
        for c in config["concepts"]
    }
    summary = {
        "stage": "2A", "input_cases": len(rows), "cohort_size": len(included),
        "positive_cases": positives, "negative_cases": len(included) - positives,
        "excluded_cases": len(excluded),
        "included_counts_by_raw_diagnosis": dict(sorted(Counter(r[config["diagnosis_column"]] for r in included).items())),
        "excluded_counts_by_raw_diagnosis": dict(sorted(Counter(r[config["diagnosis_column"]] for r in excluded).items())),
        "diagnosis_target": config["diagnosis_target"], "concept_target_order": [c["target"] for c in config["concepts"]],
        "concept_counts": concept_counts, "raw_columns_preserved": columns,
        "path_corrections": changes, "derm_path_base": "raw release images/ directory",
        "raw_metadata_and_images_unchanged": True, "original_split_indexes_read": False,
        "splits_created": False, "folds_created": False,
        "grouping_limitation": "No reliable patient/lesion linkage; case_num is the specification's fallback case key. Confirm this limitation before splitting.",
        "provenance": {"python": platform.python_version(), "config_sha256": digest(mapping),
                       "source_sha256": digest(Path(__file__)), "raw_metadata_sha256": before["meta/meta.csv"],
                       "stage1_manifest_sha256": digest(audit_dir / "raw_manifest.csv"),
                       "stage1_summary_sha256": digest(audit_dir / "summary.json")},
    }
    processed.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    write_csv(processed / "cohort.csv", columns + [config["diagnosis_target"],
              *(c["target"] for c in config["concepts"]), "derm_path"], included)
    write_csv(processed / "excluded.csv", columns + ["exclusion_reason"], excluded)
    summary["output_sha256"] = {name: digest(processed / name) for name in ("cohort.csv", "excluded.csv")}
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("raw", "data/raw/release_v0"), ("mapping", "configs/cohort_mapping.json"),
                          ("audit", "artifacts/audit"), ("processed", "data/processed/stage2a"),
                          ("artifacts", "artifacts/stage2a")):
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    args = parser.parse_args()
    result = build_cohort(args.raw, args.mapping, args.audit, args.processed, args.artifacts)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
