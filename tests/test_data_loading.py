"""Synthetic images/metadata only; these tests never access the real locked test."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import RandomSampler, SequentialSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from faithful_medical_cbm.data.dataset import DermoscopyDataset
from faithful_medical_cbm.data.loaders import LoaderFactory
from faithful_medical_cbm.reproducibility import seed_everything

CONCEPTS = (
    "atypical_pigment_network", "regression_structures_present", "irregular_pigmentation",
    "blue_whitish_veil_present", "atypical_vascular_structures", "irregular_dots_and_globules",
    "irregular_streaks",
)


def encoded_csv(columns, rows):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


class DataLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config_path = self.root / "configs/default.toml"
        self.cohort_path = self.root / "data/processed/stage2a/cohort.csv"
        self.image_root = self.root / "data/raw/release_v0/images"
        self.summary_path = self.root / "artifacts/stage2a/summary.json"
        self.split_root = self.root / "data/splits"
        for path in (self.config_path.parent, self.cohort_path.parent, self.image_root,
                     self.summary_path.parent, self.split_root):
            path.mkdir(parents=True, exist_ok=True)
        config = (ROOT / "configs/default.toml").read_text(encoding="utf-8")
        self.config_path.write_text(config.replace("image_size = 224", "image_size = 32")
                                   .replace("batch_size = 32", "batch_size = 5"), encoding="utf-8")
        self.ids = tuple(f"{i:04}" for i in range(1, 17))
        self.rows = []
        for i, identifier in enumerate(self.ids):
            pixels = np.zeros((36, 48, 3), dtype=np.uint8)
            pixels[:, :, 0] = np.arange(48, dtype=np.uint8) * 5
            pixels[:, :, 1] = i * 10
            pixels[:, :, 2] = np.arange(36, dtype=np.uint8)[:, None] * 7
            Image.fromarray(pixels).save(self.image_root / f"{identifier}.png")
            self.rows.append({"case_num": identifier, "diagnosis_binary": str(i % 2),
                              **{c: str(int((i + j) % 3 == 0)) for j, c in enumerate(CONCEPTS)},
                              "derm_path": f"{identifier}.png", "clinic": "DO_NOT_OPEN.jpg",
                              "diagnosis": "DO_NOT_MAP", "sex": "DO_NOT_USE"})
        self.refresh_metadata()

    def refresh_metadata(self):
        data = encoded_csv(list(self.rows[0]), self.rows)
        self.cohort_path.write_bytes(data)
        summary = (json.dumps({"concept_target_order": list(CONCEPTS),
                               "output_sha256": {"cohort.csv": sha(data)}}) + "\n").encode()
        self.summary_path.write_bytes(summary)
        payloads = {
            "development_ids.csv": encoded_csv(["case_num"], [{"case_num": i} for i in self.ids[:12]]),
            "test_ids.csv": encoded_csv(["case_num"], [{"case_num": i} for i in self.ids[12:]]),
            "development_folds.csv": encoded_csv(["case_num", "validation_fold"],
                [{"case_num": identifier, "validation_fold": i % 4} for i, identifier in enumerate(self.ids[:12])]),
        }
        for name, payload in payloads.items():
            (self.split_root / name).write_bytes(payload)
        meta = {"status": "frozen", "settings": {"num_folds": 4},
                "inputs": {"cohort_sha256": sha(data), "stage2a_summary_sha256": sha(summary)},
                "file_sha256": {name: sha(value) for name, value in payloads.items()},
                "counts": {"development": {"total": 12}, "test": {"total": 4}}}
        (self.split_root / "split_metadata.json").write_text(json.dumps(meta))

    def test_length_shape_labels_concept_order_and_case_number(self):
        dataset = LoaderFactory(self.config_path).development(training=False).dataset
        self.assertEqual(len(dataset), 12)
        self.assertEqual(dataset.case_nums, self.ids[:12])
        for i in (0, 1, 7):
            item = dataset[i]
            self.assertEqual(item["image"].shape, (3, 32, 32))
            self.assertEqual(item["image"].dtype, torch.float32)
            self.assertTrue(torch.isfinite(item["image"]).all())
            self.assertEqual(item["diagnosis"].shape, torch.Size([]))
            self.assertEqual(item["diagnosis"].item(), i % 2)
            self.assertEqual(item["concepts"].tolist(), [float((i + j) % 3 == 0) for j in range(7)])
            self.assertEqual(item["case_num"], self.ids[i])
            self.assertEqual(item["validation_fold"], i % 4)
            self.assertEqual(item["split"], "development")
            self.assertNotIn("clinic", item)
            self.assertNotIn("sex", item)

    def test_getitem_never_parses_metadata_and_only_opens_dermoscopy(self):
        dataset = LoaderFactory(self.config_path).development(training=False).dataset
        with patch("csv.DictReader", side_effect=AssertionError("No CSV parsing in getitem")), patch(
                "PIL.Image.open", wraps=Image.open) as opened:
            item = dataset[0]
        opened.assert_called_once_with(self.image_root / "0001.png")
        item["concepts"].zero_()
        self.assertEqual(dataset[0]["concepts"][0].item(), 1)

    def test_eval_is_deterministic_and_imagenet_normalized(self):
        dataset = LoaderFactory(self.config_path).development(training=False).dataset
        self.assertTrue(torch.equal(dataset[0]["image"], dataset[0]["image"]))
        black = dataset.transform(Image.new("RGB", (50, 30), (0, 0, 0)))
        expected = -torch.tensor([0.485, 0.456, 0.406]) / torch.tensor([0.229, 0.224, 0.225])
        self.assertTrue(torch.allclose(black[:, 0, 0], expected))

    def test_fold_membership_and_no_test_ids_in_development_batches(self):
        factory = LoaderFactory(self.config_path)
        held_out = []
        for fold in range(4):
            loaders = factory.fold(fold)
            training = set(loaders["train"].dataset.case_nums)
            validation = set(loaders["validation"].dataset.case_nums)
            self.assertEqual(len(training), 9)
            self.assertEqual(len(validation), 3)
            self.assertFalse(training & validation)
            self.assertEqual(training | validation, set(self.ids[:12]))
            self.assertFalse((training | validation) & set(factory.test_ids))
            self.assertTrue(all(factory.validation_folds[i] == fold for i in validation))
            held_out.extend(validation)
            self.assertIsInstance(loaders["train"].sampler, RandomSampler)
            self.assertIsInstance(loaders["validation"].sampler, SequentialSampler)
        self.assertEqual(len(held_out), len(set(held_out)))
        batches = list(factory.development(training=False))
        seen = [i for batch in batches for i in batch["case_num"]]
        self.assertEqual(seen, list(self.ids[:12]))
        self.assertEqual([b["image"].shape[0] for b in batches], [5, 5, 2])
        self.assertEqual(batches[0]["concepts"].shape, (5, 7))
        self.assertEqual(batches[0]["diagnosis"].shape, (5,))

    def test_development_does_not_parse_unselected_test_targets(self):
        for row in self.rows[12:]:
            row["diagnosis_binary"] = "not binary"
            row["derm_path"] = "../../forbidden.jpg"
        self.refresh_metadata()
        loader = LoaderFactory(self.config_path).development(training=False)
        self.assertEqual(next(iter(loader))["image"].shape, (5, 3, 32, 32))

    def test_locked_test_construction_is_lazy_and_access_guard_precedes_image_open(self):
        with patch("PIL.Image.open", side_effect=AssertionError("No image access")):
            loader = LoaderFactory(self.config_path).locked_test()
            self.assertEqual(len(loader.dataset), 4)
            self.assertIsInstance(loader.sampler, SequentialSampler)
            with self.assertRaisesRegex(RuntimeError, "Locked-test iteration is disabled"):
                loader.dataset[0]
        # Explicit opt-in exercised on SYNTHETIC test data only.
        allowed = LoaderFactory(self.config_path).locked_test(allow_locked_test_iteration=True)
        item = allowed.dataset[0]
        self.assertEqual(item["split"], "test")
        self.assertEqual(item["validation_fold"], -1)
        self.assertEqual(item["case_num"], self.ids[12])
        self.assertTrue(torch.equal(item["image"], allowed.dataset[0]["image"]))

    def test_training_reproducibility_with_seeded_main_process(self):
        seed_everything(42)
        first = next(iter(LoaderFactory(self.config_path).development()))
        seed_everything(42)
        second = next(iter(LoaderFactory(self.config_path).development()))
        self.assertEqual(first["case_num"], second["case_num"])
        self.assertTrue(torch.equal(first["image"], second["image"]))

    def test_spawn_worker_training_is_reproducible(self):
        self.config_path.write_text(self.config_path.read_text().replace("num_workers = 0", "num_workers = 1"))
        first = list(LoaderFactory(self.config_path).development())
        second = list(LoaderFactory(self.config_path).development())
        self.assertEqual([b["case_num"] for b in first], [b["case_num"] for b in second])
        self.assertTrue(all(torch.equal(a["image"], b["image"]) for a, b in zip(first, second)))

    def test_bad_targets_and_unsafe_paths_fail(self):
        for column, value in ((CONCEPTS[0], "2"), ("derm_path", "../outside.png")):
            original = self.rows[0][column]
            self.rows[0][column] = value
            self.refresh_metadata()
            with self.subTest(column=column), self.assertRaises(ValueError):
                LoaderFactory(self.config_path).development()
            self.rows[0][column] = original

    def test_tampered_split_hash_is_rejected_and_nothing_is_written(self):
        path = self.split_root / "development_ids.csv"
        path.write_bytes(path.read_bytes() + b"unexpected\n")
        before = {p: p.read_bytes() for p in self.split_root.iterdir()}
        with self.assertRaisesRegex(ValueError, "Frozen input hash mismatch"):
            LoaderFactory(self.config_path)
        self.assertEqual(before, {p: p.read_bytes() for p in self.split_root.iterdir()})

    def test_invalid_fold_and_workers_fail(self):
        with self.assertRaises(ValueError):
            LoaderFactory(self.config_path).fold(4)
        self.config_path.write_text(self.config_path.read_text().replace("num_workers = 0", "num_workers = -1"))
        with self.assertRaises(ValueError):
            LoaderFactory(self.config_path)


if __name__ == "__main__":
    unittest.main()
