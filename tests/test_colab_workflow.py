"""Offline Stage 4B structural checks: no training or dataset image access."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("colab_fold0", ROOT / "docs/colab_fold0.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class ColabWorkflowTests(unittest.TestCase):
    def test_notebook_compiles_and_has_one_fold_zero_command(self):
        notebook = json.loads((ROOT / "notebooks/colab_fold0.ipynb").read_text())
        code = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                compile(source, "colab_cell", "exec")
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                code.append(source)
        combined = "\n".join(code)
        self.assertEqual(combined.count('"faithful_medical_cbm.training.train_black_box"'), 1)
        self.assertIn('"--fold", "0", "--run-name", "baseline-v1"', combined)
        self.assertNotIn("locked_test(", combined)
        self.assertIn("preflight(REPO, OUTPUTS)\nrun(", combined)

    def test_manifest_matches_local_frozen_files(self):
        manifest = workflow.verify_manifest(ROOT)
        self.assertIn("configs/default.toml", manifest["files"])
        self.assertIn("data/splits/test_ids.csv", manifest["files"])

    def test_hash_tampering_rejected_and_lf_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            p = root / "config"
            p.write_bytes(b"a\nb\n")
            expected = workflow.digest(p, "lf")
            (root / "docs/colab_fold0_manifest.json").write_text(json.dumps(
                {"files": {"config": {"mode": "lf", "sha256": expected}}}))
            p.write_bytes(b"a\r\nb\r\n")
            workflow.verify_manifest(root)
            p.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                workflow.verify_manifest(root)

    def test_paths_missing_escape_and_test_root_rejected_without_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "development.jpg"
            image.touch()  # No decoding; existence only.
            diagnoses = SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 1))
            ds = SimpleNamespace(split="development", role="train", paths=[image],
                                 case_nums=["001"], diagnoses=diagnoses)
            loaders = {"train": SimpleNamespace(dataset=ds)}
            with patch("PIL.Image.open", side_effect=AssertionError("Must not open images")):
                self.assertEqual(workflow.verify_development_paths(loaders, root)["train"]["total"], 1)
                image.unlink()
                with self.assertRaises(FileNotFoundError):
                    workflow.verify_development_paths(loaders, root)
                ds.paths = [root.parent / "escaped.jpg"]
                with self.assertRaises(ValueError):
                    workflow.verify_development_paths(loaders, root)
                with self.assertRaisesRegex(ValueError, "Locked-test"):
                    workflow.verify_development_paths(loaders, root / "locked_test")
                ds.split = "test"
                with self.assertRaisesRegex(ValueError, "Only development"):
                    workflow.verify_development_paths(loaders, root)

    def test_existing_directory_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "link").mkdir()
            (root / "target").mkdir()
            (root / "link/data").write_text("preserve")
            with self.assertRaisesRegex(ValueError, "Refusing"):
                workflow.link_directory(root / "link", root / "target")
            self.assertEqual((root / "link/data").read_text(), "preserve")

    def test_post_run_summary_and_incomplete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                workflow.summarize(root)
            artifacts = root / "artifacts/baseline/baseline-v1/fold_0"
            checkpoints = root / "checkpoints/baseline/baseline-v1/fold_0"
            artifacts.mkdir(parents=True)
            checkpoints.mkdir(parents=True)
            (artifacts / "summary.json").write_text(json.dumps({"best_epoch": 2,
                "best_validation_auroc": .8, "early_stopped": True}))
            (artifacts / "history.csv").write_text("epoch,train_loss,validation_loss\n2,0.4,0.5\n")
            for name in ("best", "last"):
                (checkpoints / f"{name}.pt").touch()
            result = workflow.summarize(root)
            self.assertEqual(result["best_epoch"], 2)
            self.assertEqual(result["final_validation_loss"], .5)
            self.assertTrue(result["early_stopped"])


if __name__ == "__main__":
    unittest.main()
