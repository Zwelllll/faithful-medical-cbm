"""Read-only Derm7pt release audit. No cohort, label conversion or split logic.

Run from the repository root: python -m src.data.audit
Only Pillow and the standard library are required. Raw strings are preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import platform
from typing import Any

import PIL
from PIL import Image

CONCEPTS = (
    "pigment_network", "streaks", "pigmentation", "regression_structures",
    "dots_and_globules", "blue_whitish_veil", "vascular_structures",
)
# Report tokens without interpreting clinical words such as "absent" as missing.
NULL_TOKENS = {"na", "n/a", "nan", "null", "none"}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_metadata(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Reject ambiguous headers and ragged rows; preserve values and leading zeros."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        columns = next(reader, [])
        if not columns or any(not c for c in columns) or len(set(columns)) != len(columns):
            raise ValueError("Metadata must have nonempty, unique column names")
        rows = []
        for line, values in enumerate(reader, 2):
            if len(values) != len(columns):
                raise ValueError(f"Malformed metadata record at line {line}: expected {len(columns)} fields")
            rows.append(dict(zip(columns, values)))
    return columns, rows


def missing_kind(value: str) -> str | None:
    if not value.strip():
        return "blank"
    if value.strip().casefold() in NULL_TOKENS:
        return "null_token"
    return None


def duplicate_groups(pairs: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for key, value in pairs:
        groups[key].append(value)
    return [{"key": key, "members": members} for key, members in sorted(groups.items())
            if len(members) > 1]


def resolve_reference(value: str, available: set[str]) -> tuple[str, str | None]:
    """Resolve against an exact inventory, never silently hide Windows case mismatches."""
    if missing_kind(value):
        return "missing_value", None
    name = value.replace("\\", "/")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or ":" in name:
        return "unsafe_path", None
    if name in available:
        return "exact", name
    matches = sorted(item for item in available if item.casefold() == name.casefold())
    if len(matches) == 1:
        return "case_mismatch", matches[0]
    return ("ambiguous_case" if matches else "missing_file"), None


def inspect_image(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"format": None, "width": None, "height": None,
                              "mode": None, "pixel_sha256": None, "error": None}
    try:
        with Image.open(path) as image:
            result.update(format=image.format, width=image.width, height=image.height, mode=image.mode)
            image.verify()
        with Image.open(path) as image:
            image.load()  # Decode pixels, not just the header.
            rgb = image.convert("RGB")
            result["pixel_sha256"] = hashlib.sha256(
                f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes()
            ).hexdigest()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


class ImageLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "img":
            self.sources.extend(value for key, value in attrs if key == "src" and value is not None)


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary: dict[str, Any], references: list[dict[str, Any]]) -> str:
    """Produce a human-readable companion from the same audited facts."""
    modality_paths = {m: {r["resolved_path"] for r in references
                          if r["modality"] == m and r["resolved_path"]} for m in ("clinic", "derm")}
    shared = len(modality_paths["clinic"] & modality_paths["derm"])
    distinct_clinic = len(modality_paths["clinic"] - modality_paths["derm"])
    lines = ["# Stage 1: Derm7pt dataset audit", "",
             "Scope: raw-file inspection only. No cohort, concept conversion, split or model was created.",
             "", "## Schema", "",
             f"`meta/meta.csv`: {summary['rows']} case records; {len(summary['columns'])} columns.",
             "", "```text", ", ".join(summary["columns"]), "```", "",
             "Metadata files: " + ", ".join(f"`{m['path']}`" for m in summary["metadata_files"]) + ".",
             "The supplied split-index files are inventoried and hashed only; their assignments are not used.",
             "`README.txt` describes the release; HTML gallery checks are in summary.json.",
             "", "## Raw diagnosis distribution", "", "| Raw label | Cases |", "|---|---:|"]
    lines.extend(f"| {label} | {count} |" for label, count in summary["diagnosis_counts"].items())
    lines.extend(["", "## Raw concept distributions", "", "| Column | Values and counts |", "|---|---|"])
    for column, values in summary["concept_counts"].items():
        lines.append(f"| `{column}` | " + "; ".join(f"{v}: {n}" for v, n in values.items()) + " |")
    lines.extend(["", "## Images and missing data", "",
                  f"{summary['image_files']} physical image files; {len(summary['unreferenced_images'])} unreferenced; {len(summary['broken_images'])} broken.",
                  f"{summary['modalities']['derm']['references']} dermoscopic references and {summary['modalities']['clinic']['references']} clinical references. {shared} files are shared across the two reference columns, leaving {distinct_clinic} files referenced only as clinical.",
                  "Sizes below are width x height, counting unique files within each metadata modality.", ""])
    for modality, info in summary["modalities"].items():
        lines.append(f"- {modality}: " + "; ".join(f"{size}: {n}" for size, n in info["dimensions"].items()))
        lines.append(f"  Formats: {info['formats']}; reference status counts: {info['status_counts']}.")
    lines.extend(["", "Missing values (blank and lexical null token counts):", ""])
    lines.extend(f"- `{column}`: {values}." for column, values in summary["missing_counts"].items())
    lines.extend(["", "Nonempty notes are recorded for cases: " + ", ".join(
                      item["case_num"] for item in summary["nonempty_notes"]) + ".", ""])
    for ref in references:
        if ref["status"] != "exact":
            lines.append(f"- Case {ref['case_num']} ({ref['modality']}): `{ref['reference']}` -> `{ref['resolved_path']}` ({ref['status']}). Raw metadata was not changed.")
    lines.extend(["", "## Duplicates", ""])
    lines.extend(f"- {key}: {count} groups." for key, count in summary["duplicate_group_counts"].items())
    lines.extend(["", "See duplicates.json for within-case versus cross-case reuse and exact-content matches.",
                  "Exact bytes and decoded RGB pixels were compared. Near-duplicates and repeated lesions with different views remain unverified.",
                  "", "## Grouping", "",
                  "Identifier completeness: " + json.dumps(summary["identifier_completeness"]) + ".",
                  "The semantics of case_id are undocumented locally.",
                  "No patient identifier or independently verified lesion identifier exists in the supplied metadata.",
                  "Whether a patient has multiple cases/lesions/images cannot be determined. Distinct available image files per case: " + json.dumps(summary["case_image_path_counts"]) + ".",
                  "Recommend the specification's fallback: use case_num as the case key and keep paired images together in a future split. This is case-level grouping, not verified patient-level or lesion-level independence.",
                  "Do not infer identity from filename prefixes or sparse case_id values. External linkage would be needed to verify patient or repeated-lesion grouping.",
                  "", "## Discrepancies and decisions before Stage 2", "",
                  f"1. The release includes {len(summary['diagnosis_counts'])} diagnosis labels. Approve an explicit inclusion/exclusion mapping for benign melanocytic diagnoses and melanoma variants, particularly melanoma metastasis, the unsuffixed melanoma label, and potentially ambiguous labels. No mapping was applied.",
                  "2. The seven concepts are categorical, not seven ready-made binary targets. Approve explicit conversion rules for every raw value, especially vascular structures, regression structures and pigmentation. No conversion was applied.",
                  "3. Accept the case-level grouping limitation or obtain authoritative patient/lesion linkage. Do not interpret case_id as a complete lesion or patient identifier.",
                  "4. Approve case-sensitive path resolution in future processed metadata for reported filename-case mismatches, preserving raw values and files unchanged.",
                  "5. Acknowledge the clinical substitutions documented in notes; clinical images remain outside the dermoscopy-only primary experiment.",
                  "6. Supplied split indexes do not define this project's frozen 80/20 protocol. They were not adopted. The eventual split is a separate stage.",
                  "7. Image dimensions vary; resizing belongs to a later preprocessing stage, not this audit.",
                  "", "## Reproduction and integrity", "",
                  "Run `python -m src.data.audit` from the repository root. Outputs are deterministic for identical inputs and runtime.",
                  "`raw_manifest.csv` records every input file's SHA-256. All raw files were rehashed after the audit; the file list and contents were unchanged.",
                  "`provenance.json` records the audit source hash, manifest hash, Python and Pillow versions.",
                  "`image_inventory.csv`, `image_references.csv`, `case_images.csv`, `value_counts.csv`, `duplicates.json` and `summary.json` contain the detailed machine-readable evidence.", ""])
    return "\n".join(lines)


def audit(raw: Path, output: Path) -> dict[str, Any]:
    raw, output = raw.resolve(), output.resolve()
    if output == raw or raw in output.parents or output in raw.parents:
        raise ValueError("Audit output must be separate from the raw dataset tree")
    columns, rows = read_metadata(raw / "meta/meta.csv")
    required = {"case_num", "case_id", "diagnosis", "clinic", "derm", "notes", *CONCEPTS}
    if missing := required - set(columns):
        raise ValueError(f"Release schema is missing expected observed columns: {sorted(missing)}")
    if not rows:
        raise ValueError("Metadata contains no cases")

    # Inventory and hashes include supplied index files, but their assignments are never parsed.
    files = sorted(p for p in raw.rglob("*") if p.is_file())
    manifest = [{"path": p.relative_to(raw).as_posix(), "bytes": p.stat().st_size,
                 "sha256": digest(p)} for p in files]
    hashes = {item["path"]: item["sha256"] for item in manifest}
    image_paths = [p for p in files if (raw / "images") in p.parents]
    images = []
    for p in image_paths:
        images.append({"path": p.relative_to(raw / "images").as_posix(),
                       "bytes": p.stat().st_size, "sha256": hashes[p.relative_to(raw).as_posix()],
                       **inspect_image(p)})
    by_path = {item["path"]: item for item in images}
    available = set(by_path)
    references = []
    for row_number, row in enumerate(rows, 2):
        for modality in ("derm", "clinic"):
            status, resolved = resolve_reference(row[modality], available)
            references.append({"csv_line": row_number, "case_num": row["case_num"],
                               "modality": modality, "reference": row[modality], "status": status,
                               "resolved_path": resolved,
                               "decode_error": by_path[resolved]["error"] if resolved else None})

    counts = {c: dict(sorted(Counter(row[c] for row in rows).items())) for c in columns}
    missing_counts = {c: {kind: sum(missing_kind(row[c]) == kind for row in rows)
                          for kind in ("blank", "null_token")} for c in columns}
    duplicates = {
        "metadata_rows": duplicate_groups([(json.dumps([row[c] for c in columns]), i)
                                             for i, row in enumerate(rows, 2)]),
        "case_num": duplicate_groups([(r["case_num"], i) for i, r in enumerate(rows, 2)
                                       if not missing_kind(r["case_num"])]),
        "nonempty_case_id": duplicate_groups([(r["case_id"], i) for i, r in enumerate(rows, 2)
                                               if not missing_kind(r["case_id"])]),
        "basenames": duplicate_groups([(PurePosixPath(p).name, p) for p in sorted(available)]),
        "case_insensitive_paths": duplicate_groups([(p.casefold(), p) for p in sorted(available)]),
        "case_insensitive_basenames": duplicate_groups([(PurePosixPath(p).name.casefold(), p)
                                                        for p in sorted(available)]),
        "repeated_references": duplicate_groups([(r["resolved_path"],
            {"case_num": r["case_num"], "modality": r["modality"]})
            for r in references if r["resolved_path"]]),
        "byte_identical_images": duplicate_groups([(im["sha256"], im["path"]) for im in images]),
        "pixel_identical_images": duplicate_groups([(im["pixel_sha256"], im["path"])
                                                     for im in images if im["pixel_sha256"]]),
    }
    html = {}
    for modality in ("clinic", "derm"):
        parser = ImageLinks()
        parser.feed((raw / f"{modality}.html").read_text(encoding="utf-8"))
        expected = Counter("images/" + row[modality] for row in rows)
        actual = Counter(parser.sources)
        html[modality] = {"image_count": len(parser.sources),
                          "matches_metadata_multiset": actual == expected,
                          "html_only": dict(actual - expected), "metadata_only": dict(expected - actual)}
    modality_summary = {}
    for modality in ("derm", "clinic"):
        refs = [r for r in references if r["modality"] == modality]
        paths = sorted({r["resolved_path"] for r in refs if r["resolved_path"]})
        modality_summary[modality] = {
            "references": len(refs), "unique_available_files": len(paths),
            "status_counts": dict(Counter(r["status"] for r in refs)),
            "broken_references": sum(bool(r["decode_error"]) for r in refs),
            "formats": dict(Counter(by_path[p]["format"] for p in paths)),
            "dimensions": dict(sorted(Counter(f'{by_path[p]["width"]}x{by_path[p]["height"]}'
                                              for p in paths).items())),
        }
    case_images = []
    for i, row in enumerate(rows):
        pair = references[2*i:2*i+2]
        paths = {r["resolved_path"] for r in pair if r["resolved_path"]}
        pixels = {by_path[p]["pixel_sha256"] for p in paths if by_path[p]["pixel_sha256"]}
        case_images.append({"case_num": row["case_num"], "case_id": row["case_id"],
                            "unique_available_paths": len(paths), "unique_decoded_rgb_images": len(pixels),
                            "derm": row["derm"], "clinic": row["clinic"], "notes": row["notes"]})
    used = {r["resolved_path"] for r in references if r["resolved_path"]}
    path_cases: dict[str, set[str]] = defaultdict(set)
    for r in references:
        if r["resolved_path"]:
            path_cases[r["resolved_path"]].add(r["case_num"])
    duplicates["cross_case_shared_paths"] = [
        {"path": path, "case_nums": sorted(cases)} for path, cases in sorted(path_cases.items())
        if len(cases) > 1
    ]
    cross_case = []
    for group in duplicates["pixel_identical_images"]:
        cases = sorted(set().union(*(path_cases[p] for p in group["members"])))
        if len(cases) > 1:
            cross_case.append({**group, "case_nums": cases})
    duplicates["cross_case_pixel_duplicates"] = cross_case
    identifiers = {c: {"nonmissing": sum(not missing_kind(r[c]) for r in rows),
                       "unique_nonmissing": len({r[c] for r in rows if not missing_kind(r[c])})}
                   for c in columns if "id" in c.casefold() or "case" in c.casefold() or "patient" in c.casefold()}
    summary = {
        "schema_version": 1, "metadata_file": "meta/meta.csv", "rows": len(rows),
        "columns": columns, "column_storage": "Raw CSV strings, unmodified; no clinical mappings",
        "metadata_files": [m for m in manifest if m["path"].startswith("meta/")],
        "supplied_split_indexes": "Inventoried and hashed only; not parsed or adopted; no project split exists",
        "concept_columns": list(CONCEPTS), "diagnosis_counts": counts["diagnosis"],
        "concept_counts": {c: counts[c] for c in CONCEPTS}, "all_column_counts": counts,
        "missing_counts": missing_counts, "null_token_policy": sorted(NULL_TOKENS),
        "image_files": len(images), "modalities": modality_summary,
        "unreferenced_images": sorted(available - used),
        "broken_images": [im["path"] for im in images if im["error"]],
        "html_galleries": html,
        "duplicate_group_counts": {k: len(v) for k, v in duplicates.items()},
        "identifier_completeness": identifiers,
        "case_image_path_counts": dict(Counter(str(r["unique_available_paths"]) for r in case_images)),
        "case_image_content_counts": dict(Counter(str(r["unique_decoded_rgb_images"]) for r in case_images)),
        "nonempty_notes": [{"case_num": r["case_num"], "notes": r["notes"]} for r in rows if r["notes"]],
        "grouping_assessment": {
            "patient_identifier": None,
            "patient_multiplicity": "Unknown: no patient identifier or patient linkage is supplied",
            "lesion_identifier": "No independently verified lesion identifier; case_num identifies metadata records",
            "paired_images": "clinic and derm associate two image references with each case_num; notes may flag substitutions",
            "recommendation": "Use case_num as the fallback case key; keep paired images together. Resolve cross-case duplicate components before any future split. Do not infer patient IDs from filenames or case_id.",
            "limitation": "Patient independence and repeated lesions across distinct cases cannot be established from these files",
        },
        "limitations": ["Exact byte and decoded RGB equality only; visually near-duplicate images are not excluded",
                        "Image modality follows metadata and notes, not clinical visual adjudication",
                        "Missing token detection reports lexical candidates and does not impute values"],
    }
    # Verify full raw-tree bytes and file list remain unchanged during this run.
    after = {p.relative_to(raw).as_posix(): digest(p) for p in sorted(raw.rglob("*")) if p.is_file()}
    if hashes != after:
        raise RuntimeError("Raw dataset changed during audit; outputs were not written")
    summary["raw_unchanged_verified"] = True
    output.mkdir(parents=True, exist_ok=True)
    for name, obj in (("summary.json", summary), ("duplicates.json", duplicates),
                      ("provenance.json", {"python": platform.python_version(), "pillow": PIL.__version__,
                       "audit_source_sha256": digest(Path(__file__)), "raw_manifest_sha256": hashlib.sha256(
                           json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                       "raw_unchanged_verified": True})):
        (output / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, records in (("raw_manifest.csv", manifest), ("image_inventory.csv", images),
                           ("image_references.csv", references), ("case_images.csv", case_images)):
        if records:
            write_csv(output / name, list(records[0]), records)
    distributions = [{"column": c, "raw_value": value, "count": count,
                      "missing_kind": missing_kind(value)}
                     for c in columns for value, count in counts[c].items()]
    write_csv(output / "value_counts.csv", ["column", "raw_value", "count", "missing_kind"], distributions)
    (output / "report.md").write_text(render_report(summary, references), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/release_v0"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/audit"))
    args = parser.parse_args()
    result = audit(args.raw, args.output)
    print(json.dumps({k: result[k] for k in ("rows", "image_files", "duplicate_group_counts",
                                            "raw_unchanged_verified")}, indent=2))


if __name__ == "__main__":
    main()
