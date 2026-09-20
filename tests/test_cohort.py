"""Frozen user-approved mappings, checked independently of implementation/config."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.audit import digest, read_metadata, write_csv
from src.data.cohort import build_cohort, construct_rows, load_mapping, map_row

DIAGNOSES = {
    "melanoma": 1, "melanoma (in situ)": 1, "melanoma (less than 0.76 mm)": 1,
    "melanoma (0.76 to 1.5 mm)": 1, "melanoma (more than 1.5 mm)": 1,
    "blue nevus": 0, "clark nevus": 0, "combined nevus": 0, "congenital nevus": 0,
    "dermal nevus": 0, "recurrent nevus": 0, "reed or spitz nevus": 0,
    "melanoma metastasis": None, "basal cell carcinoma": None, "dermatofibroma": None,
    "lentigo": None, "melanosis": None, "miscellaneous": None,
    "seborrheic keratosis": None, "vascular lesion": None,
}
CONCEPTS = {
    "atypical_pigment_network": ("pigment_network", {"absent": 0, "typical": 0, "atypical": 1}),
    "regression_structures_present": ("regression_structures", {"absent": 0, "blue areas": 1, "white areas": 1, "combinations": 1}),
    "irregular_pigmentation": ("pigmentation", {"absent": 0, "diffuse regular": 0, "localized regular": 0, "diffuse irregular": 1, "localized irregular": 1}),
    "blue_whitish_veil_present": ("blue_whitish_veil", {"absent": 0, "present": 1}),
    "atypical_vascular_structures": ("vascular_structures", {"absent": 0, "arborizing": 0, "within regression": 0, "hairpin": 0, "dotted": 1, "comma": 0, "linear irregular": 1, "wreath": 0}),
    "irregular_dots_and_globules": ("dots_and_globules", {"absent": 0, "regular": 0, "irregular": 1}),
    "irregular_streaks": ("streaks", {"absent": 0, "regular": 0, "irregular": 1}),
}


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.config = load_mapping(ROOT / "configs/cohort_mapping.json")
        self.row = {source: "absent" for source, _ in CONCEPTS.values()}
        self.row.update(case_num="001", diagnosis="melanoma", derm="image.jpg", notes="preserve me")

    def test_every_diagnosis(self):
        self.assertEqual(set(self.config["diagnoses"]), set(DIAGNOSES))
        for label, expected in DIAGNOSES.items():
            with self.subTest(label=label):
                self.assertEqual(map_row(dict(self.row, diagnosis=label), self.config)[0], expected)

    def test_every_raw_concept_value_and_order(self):
        self.assertEqual([c["target"] for c in self.config["concepts"]], list(CONCEPTS))
        for target, (source, values) in CONCEPTS.items():
            rule = next(c for c in self.config["concepts"] if c["target"] == target)
            self.assertEqual(rule["source"], source)
            self.assertEqual(set(rule["values"]), set(values))
            for value, expected in values.items():
                with self.subTest(target=target, value=value):
                    self.assertEqual(map_row({**self.row, source: value}, self.config)[1][target], expected)

    def test_unknown_or_missing_labels_fail_including_excluded_rows(self):
        for label in ("", "Melanoma", "melanoma ", "unknown"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                map_row(dict(self.row, diagnosis=label), self.config)
        for source, _ in CONCEPTS.values():
            for value in ("", "null", "unexpected"):
                with self.subTest(source=source, value=value), self.assertRaises(ValueError):
                    map_row({**self.row, "diagnosis": "melanoma metastasis", source: value}, self.config)

    def test_preserves_raw_fields_ids_and_inputs(self):
        rows = [self.row, dict(self.row, case_num="002", diagnosis="melanoma metastasis")]
        original = copy.deepcopy(rows)
        included, excluded, changes = construct_rows(list(self.row), rows, self.config, {"image.jpg"})
        self.assertEqual(rows, original)
        self.assertEqual({k: included[0][k] for k in self.row}, self.row)
        self.assertEqual({k: excluded[0][k] for k in self.row}, rows[1])
        self.assertEqual(included[0]["case_num"], "001")
        self.assertNotIn("diagnosis_binary", excluded[0])
        self.assertEqual(changes, [])
        self.assertFalse({"split", "fold", "patient_id"} & set(included[0]))

    def test_known_path_correction_only_in_added_column(self):
        row = dict(self.row, case_num="816", derm="FCl/Fcl068.jpg")
        included, _, changes = construct_rows(list(row), [row], self.config, {"FCL/Fcl068.jpg"})
        self.assertEqual(included[0]["derm"], "FCl/Fcl068.jpg")
        self.assertEqual(included[0]["derm_path"], "FCL/Fcl068.jpg")
        self.assertEqual(len(changes), 1)
        with self.assertRaises(ValueError):
            construct_rows(list(row), [dict(row, case_num="817")], self.config, {"FCL/Fcl068.jpg"})
        with self.assertRaises(ValueError):
            construct_rows(list(row), [dict(row, derm="other.jpg")], self.config, {"other.jpg"})

    def test_invalid_identifiers_paths_or_column_collisions_fail(self):
        for rows in ([self.row, self.row], [dict(self.row, case_num="")],
                     [dict(self.row, derm="../image.jpg")], [dict(self.row, derm="missing.jpg")],
                     [dict(self.row, diagnosis_binary="raw collision")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                construct_rows(list(rows[0]), rows, self.config, {"image.jpg"})

    def test_output_cannot_overlap_raw_or_audit(self):
        with self.assertRaises(ValueError):
            build_cohort(Path("raw"), Path("unused"), Path("audit"), Path("raw/output"), Path("out"))

    def test_end_to_end_determinism_integrity_and_no_index_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw, audit_dir = root / "raw", root / "audit"
            (raw / "meta").mkdir(parents=True)
            (raw / "images").mkdir()
            audit_dir.mkdir()
            rows = [dict(self.row, case_num=str(i), diagnosis=label) for i, label in enumerate(DIAGNOSES, 1)]
            write_csv(raw / "meta/meta.csv", list(self.row), rows)
            (raw / "images/image.jpg").write_bytes(b"fixture already verified at audit stage")
            for name in ("train_indexes.csv", "valid_indexes.csv", "test_indexes.csv"):
                (raw / "meta" / name).write_bytes(b"must never open")
            records = [{"path": name, "sha256": digest(raw / name)}
                       for name in ("meta/meta.csv", "images/image.jpg")]
            write_csv(audit_dir / "raw_manifest.csv", ["path", "sha256"], records)
            (audit_dir / "summary.json").write_text(json.dumps({
                "diagnosis_counts": dict.fromkeys(DIAGNOSES, 1),
                "concept_counts": {source: dict.fromkeys(values, 0) for source, values in CONCEPTS.values()}}))
            original_open = Path.open

            def guarded_open(path, *args, **kwargs):
                if path.name.endswith("_indexes.csv"):
                    raise AssertionError("Original split index was accessed")
                return original_open(path, *args, **kwargs)

            args = (raw, ROOT / "configs/cohort_mapping.json", audit_dir, root / "processed", root / "out")
            with patch.object(Path, "open", guarded_open):
                first = build_cohort(*args)
                second = build_cohort(*args)
            self.assertEqual(first, second)
            self.assertEqual((first["cohort_size"], first["positive_cases"], first["negative_cases"]), (12, 5, 7))
            self.assertEqual(first["excluded_cases"], 8)
            self.assertEqual([r["case_num"] for r in read_metadata(root / "processed/cohort.csv")[1]],
                             [str(i) for i in range(1, 13)])
            self.assertTrue(first["raw_metadata_and_images_unchanged"])
            (raw / "images/image.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "differ from the Stage 1 audit"):
                build_cohort(*args)


if __name__ == "__main__":
    unittest.main()
