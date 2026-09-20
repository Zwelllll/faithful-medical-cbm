"""Synthetic fixtures only: tests never read the real release or any split."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.audit import (
    CONCEPTS, audit, digest, inspect_image, missing_kind, read_metadata, resolve_reference,
)


class AuditTests(unittest.TestCase):
    def setUp(self):
        cache = ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=cache)
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def test_csv_preserves_quoted_values_and_leading_zero_ids(self):
        path = self.folder / "meta.csv"
        path.write_text('\ufeffcase_num,notes,concept\n001,"a,b",absent\n', encoding="utf-8")
        columns, rows = read_metadata(path)
        self.assertEqual(columns, ["case_num", "notes", "concept"])
        self.assertEqual(rows[0], {"case_num": "001", "notes": "a,b", "concept": "absent"})

    def test_ragged_records_and_duplicate_headers_fail(self):
        path = self.folder / "bad.csv"
        for text in ("id,id\n1,2\n", "id,value\n1\n", "id,value\n1,2,3\n"):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                read_metadata(path)

    def test_missing_values_do_not_treat_absence_as_null(self):
        for value in ("absent", "0", "typical", "within regression"):
            self.assertIsNone(missing_kind(value))
        self.assertEqual(missing_kind("  "), "blank")
        self.assertEqual(missing_kind("NULL"), "null_token")

    def test_paths_missing_casing_collision_and_traversal(self):
        paths = {"X/a.JPG", "X/b.jpg", "X/B.JPG"}
        self.assertEqual(resolve_reference("X/a.JPG", paths), ("exact", "X/a.JPG"))
        self.assertEqual(resolve_reference("x/A.jpg", paths), ("case_mismatch", "X/a.JPG"))
        self.assertEqual(resolve_reference("x/b.jpg", paths), ("ambiguous_case", None))
        self.assertEqual(resolve_reference("missing.jpg", paths), ("missing_file", None))
        for value in ("../outside.jpg", "/outside.jpg", "C:/outside.jpg"):
            self.assertEqual(resolve_reference(value, paths), ("unsafe_path", None))

    def test_decode_checks_format_and_broken_pixels(self):
        path = self.folder / "image.jpg"
        Image.new("RGB", (8, 6), "red").save(path, format="PNG")
        info = inspect_image(path)
        self.assertEqual((info["width"], info["height"], info["format"]), (8, 6, "PNG"))
        path.write_bytes(b"not an image")
        self.assertIsNotNone(inspect_image(path)["error"])

    def make_release(self):
        raw = self.folder / "release"
        (raw / "meta").mkdir(parents=True)
        (raw / "images").mkdir()
        columns = ["case_num", "diagnosis", *CONCEPTS, "clinic", "derm", "case_id", "notes"]
        row = dict.fromkeys(columns, "absent")
        row.update(case_num="001", diagnosis="raw label", clinic="a.png", derm="b.png", case_id="", notes="")
        other = dict(row, case_num="002", clinic="missing.png", derm="c.png")
        with (raw / "meta/meta.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows([row, other, other])
        for name in ("a.png", "b.png", "c.png", "orphan.png"):
            Image.new("RGB", (4, 5), "blue").save(raw / "images" / name)
        for modality in ("clinic", "derm"):
            (raw / f"{modality}.html").write_text("".join(
                f'<img src="images/{r[modality]}">' for r in (row, other, other)))
        # Not valid CSV: successful auditing proves assignments are never parsed.
        (raw / "meta/test_indexes.csv").write_bytes(b"\xff\xfe\x00")
        return raw

    def test_end_to_end_integrity_duplicates_and_determinism(self):
        raw = self.make_release()
        before = {str(p): digest(p) for p in raw.rglob("*") if p.is_file()}
        output = self.folder / "audit"
        result = audit(raw, output)
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["modalities"]["clinic"]["status_counts"]["missing_file"], 2)
        self.assertEqual(result["duplicate_group_counts"]["metadata_rows"], 1)
        self.assertEqual(result["duplicate_group_counts"]["case_num"], 1)
        self.assertEqual(result["duplicate_group_counts"]["cross_case_pixel_duplicates"], 1)
        self.assertEqual(result["unreferenced_images"], ["orphan.png"])
        self.assertEqual(result["missing_counts"]["case_id"]["blank"], 3)
        self.assertEqual(result["missing_counts"]["pigment_network"]["blank"], 0)
        self.assertEqual(before, {str(p): digest(p) for p in raw.rglob("*") if p.is_file()})
        first = {p.name: p.read_bytes() for p in output.iterdir()}
        audit(raw, output)
        self.assertEqual(first, {p.name: p.read_bytes() for p in output.iterdir()})
        self.assertTrue(json.loads((output / "summary.json").read_text())["raw_unchanged_verified"])

    def test_output_cannot_overlap_raw_tree(self):
        raw = self.folder / "raw"
        for output in (raw, raw / "audit", raw.parent):
            with self.subTest(output=output), self.assertRaises(ValueError):
                audit(raw, output)

    def test_same_file_referenced_by_different_cases_is_flagged(self):
        raw = self.make_release()
        path = raw / "meta/meta.csv"
        path.write_text(path.read_text().replace("c.png", "b.png"))
        result = audit(raw, self.folder / "audit")
        self.assertEqual(result["duplicate_group_counts"]["cross_case_shared_paths"], 1)

    def test_input_change_aborts_before_writing_outputs(self):
        raw = self.make_release()
        calls = {}

        def changed_digest(path):
            key = str(path)
            calls[key] = calls.get(key, 0) + 1
            original = digest(path)
            return "changed" if calls[key] > 1 and path.name == "meta.csv" else original

        output = self.folder / "audit"
        with patch("src.data.audit.digest", side_effect=changed_digest):
            with self.assertRaisesRegex(RuntimeError, "Raw dataset changed"):
                audit(raw, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
