"""Synthetic-only tests: never load the actual locked test cases or raw release."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import splitting


class SplittingTests(unittest.TestCase):
    def setUp(self):
        # Same class totals as the cohort, but explicitly synthetic permanent IDs.
        self.labels = {f"synthetic-{i:04}": int(i < 248) for i in range(823)}

    def test_all_partition_fold_and_count_invariants(self):
        development, test, folds = splitting.generate_assignments(self.labels, 42, 0.2, 4)
        self.assertEqual(len(development), 658)
        self.assertEqual(len(test), 165)
        self.assertFalse(set(development) & set(test))
        self.assertEqual(set(development) | set(test), set(self.labels))
        self.assertEqual(len(set(development + test)), 823)
        self.assertEqual(len(folds), 658)
        self.assertEqual({i for i, _ in folds}, set(development))
        self.assertEqual(len({i for i, _ in folds}), len(folds))
        self.assertFalse({i for i, _ in folds} & set(test))
        counts = splitting.validate_assignments(self.labels, development, test, folds, 4)
        self.assertEqual(counts["development"], {"total": 658, "positive": 198, "negative": 460})
        self.assertEqual(counts["test"], {"total": 165, "positive": 50, "negative": 115})
        for report in counts["folds"]:
            for role in ("training", "validation"):
                ids = [i for i, f in folds if (f == report["fold"]) == (role == "validation")]
                self.assertEqual(report[role], splitting.counts(ids, self.labels))

    def test_same_seed_and_different_row_order_reproduce_identical_assignments(self):
        first = splitting.generate_assignments(self.labels, 42, 0.2, 4)
        self.assertEqual(first, splitting.generate_assignments(self.labels, 42, 0.2, 4))
        reversed_labels = dict(reversed(list(self.labels.items())))
        self.assertEqual(first, splitting.generate_assignments(reversed_labels, 42, 0.2, 4))

    def test_each_identity_invariant_rejects_bad_assignments(self):
        dev, test, folds = splitting.generate_assignments(self.labels, 42, 0.2, 4)
        variants = [
            (dev + [dev[0]], test, folds),
            (dev, test + [test[0]], folds),
            (dev + [test[0]], test, folds),
            (dev[1:], test, folds),
            (dev, test, folds + [folds[0]]),
            (dev, test, folds[1:]),
            (dev, test, folds + [(test[0], 0)]),
            (dev, test, [(i, 9 if f == 0 else f) for i, f in folds]),
        ]
        for number, args in enumerate(variants):
            with self.subTest(invariant=number), self.assertRaises(ValueError):
                splitting.validate_assignments(self.labels, *args, 4)

    def test_processed_parser_preserves_identifiers_and_rejects_bad_labels(self):
        self.assertEqual(splitting.read_cohort(b"case_num,diagnosis_binary\n001,0\n002,1\n"), {"001": 0, "002": 1})
        for data in (b"case_num,diagnosis_binary\nx,0\nx,1\n", b"case_num,diagnosis_binary\n,0\nx,1\n",
                     b"case_num,diagnosis_binary\nx,yes\ny,1\n", b"case_num,diagnosis\nx,melanoma\n",
                     b"case_num,diagnosis_binary\nx,0,extra\ny,1\n"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                splitting.read_cohort(data)

    def fixture(self, root):
        (root / "configs").mkdir()
        cohort = root / "data/processed/stage2a/cohort.csv"
        summary = root / "artifacts/stage2a/summary.json"
        cohort.parent.mkdir(parents=True)
        summary.parent.mkdir(parents=True)
        data = splitting.csv_bytes(["case_num", "diagnosis_binary"], list(self.labels.items()))
        cohort.write_bytes(data)
        summary.write_text(json.dumps({"cohort_size": 823, "positive_cases": 248, "negative_cases": 575,
                                       "output_sha256": {"cohort.csv": splitting.sha256(data)}}))
        config = root / "configs/default.toml"
        config.write_text((ROOT / "configs/default.toml").read_text(encoding="utf-8"), encoding="utf-8")
        return config, cohort, summary, root / "data/splits"

    def test_freeze_reads_processed_only_and_rerun_never_regenerates_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, cohort, summary, output = self.fixture(Path(tmp))
            allowed = {config, cohort, summary, Path(splitting.__file__).resolve(),
                       *(output / name for name in splitting.FILES)}
            original = Path.open

            def guarded_open(path, *args, **kwargs):
                self.assertIn(path.resolve(), allowed, f"Unexpected read/write, including raw/index files: {path}")
                return original(path, *args, **kwargs)

            with patch.object(Path, "open", guarded_open), patch.object(
                    splitting, "generate_assignments", wraps=splitting.generate_assignments) as generate:
                metadata, status = splitting.freeze_splits(config)
                self.assertEqual(status, "created_and_frozen")
                before = {name: ((output / name).read_bytes(), (output / name).stat().st_mtime_ns)
                          for name in splitting.FILES}
                repeated, status = splitting.freeze_splits(config)
                self.assertEqual(status, "verified_existing_no_writes")
                self.assertEqual(metadata, repeated)
                self.assertEqual(generate.call_count, 1)
                after = {name: ((output / name).read_bytes(), (output / name).stat().st_mtime_ns)
                         for name in splitting.FILES}
                self.assertEqual(before, after)
            self.assertFalse(metadata["original_derm7pt_indexes_used"])
            self.assertFalse(metadata["raw_metadata_or_images_read"])
            self.assertIn("patient-level independence cannot be verified", metadata["limitation"])
            self.assertEqual(splitting.read_assignments(output), splitting.generate_assignments(self.labels, 42, 0.2, 4))

    def test_frozen_files_refuse_changed_seed_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _, output = self.fixture(Path(tmp))
            splitting.freeze_splits(config)
            original_config = config.read_text()
            original_files = {name: (output / name).read_bytes() for name in splitting.FILES}
            config.write_text(original_config.replace("seed = 42", "seed = 43"))
            with self.assertRaisesRegex(ValueError, "refusing to alter"):
                splitting.freeze_splits(config)
            self.assertEqual(original_files, {name: (output / name).read_bytes() for name in splitting.FILES})
            config.write_text(original_config)
            damaged = output / "test_ids.csv"
            damaged.write_bytes(original_files["test_ids.csv"] + b"unapproved\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                splitting.freeze_splits(config)
            self.assertTrue(damaged.read_bytes().endswith(b"unapproved\n"))

    def test_partial_freeze_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _, output = self.fixture(Path(tmp))
            output.mkdir()
            path = output / "development_ids.csv"
            path.write_bytes(b"existing partial file")
            with self.assertRaisesRegex(ValueError, "Partial frozen split"):
                splitting.freeze_splits(config)
            self.assertEqual(path.read_bytes(), b"existing partial file")
            self.assertEqual(len(list(output.iterdir())), 1)

    def test_changed_cohort_rejected_before_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, cohort, _, output = self.fixture(Path(tmp))
            cohort.write_bytes(cohort.read_bytes() + b"new,1\n")
            with self.assertRaisesRegex(ValueError, "Stage 2A hash"):
                splitting.freeze_splits(config)
            self.assertFalse(output.exists())

    def test_raw_input_path_rejected_before_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, _, _, output = self.fixture(Path(tmp))
            config.write_text(config.read_text().replace('cohort = "stage2a/cohort.csv"',
                              'cohort = "../raw/release_v0/meta/meta.csv"'))
            with self.assertRaisesRegex(ValueError, "Inputs must stay inside processed"):
                splitting.freeze_splits(config)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
