"""Offline baseline checks: one B0 forward smoke and one tiny synthetic optimizer step."""
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from faithful_medical_cbm.config import load_config
from faithful_medical_cbm.models.black_box import BlackBoxEfficientNet
from faithful_medical_cbm.training import baseline
from faithful_medical_cbm.training.train_black_box import run_training
from faithful_medical_cbm.data.loaders import LoaderFactory
import test_data_loading as fixtures


class BatchOnlyLoader:
    def __init__(self, batch, *, role="train", split="development"):
        self.batch = batch
        self.dataset = SimpleNamespace(split=split, role=role, case_nums=tuple(batch["case_num"]))

    def __iter__(self):
        yield self.batch


class ImageDiagnosisOnly(dict):
    def __getitem__(self, key):
        if key in ("concepts", "clinic", "sex", "location"):
            raise AssertionError("Baseline may not consume concepts/metadata")
        return super().__getitem__(key)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/default.toml")
        self.settings = self.config["baseline"]
        self.addCleanup(patch.stopall)
        patch("torch.hub.download_url_to_file", side_effect=AssertionError("Tests must stay offline")).start()

    def test_b0_construction_forward_one_step_and_checkpoint_roundtrip(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(42)
        model = BlackBoxEfficientNet(pretrained=False)
        self.assertEqual(model.network.classifier[1].in_features, 1280)
        self.assertEqual(model.network.classifier[1].out_features, 1)
        model.train()
        self.assertTrue(all(not block.training for block in model.network.features))
        self.assertTrue(all(not p.requires_grad for p in model.network.features.parameters()))
        with torch.no_grad():
            output = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(output.shape, (2,))
        self.assertTrue(torch.isfinite(output).all())

        model.set_trainable_blocks(3)
        self.assertFalse(model.network.features[0].training)
        self.assertTrue(model.network.features[-1].training)
        before = model.network.classifier[1].weight.detach().clone()
        frozen_buffers = {name: value.clone() for name, value in model.network.features[0].state_dict().items()}
        optimizer = baseline.build_optimizer(model, self.settings, self.settings["finetune_learning_rate"])
        batch = ImageDiagnosisOnly(image=torch.randn(2, 3, 64, 64), diagnosis=torch.tensor([0., 1.]),
                                   case_num=["synthetic-a", "synthetic-b"], split=["development"] * 2)
        loss, predictions = baseline.run_epoch(model, BatchOnlyLoader(batch), torch.device("cpu"), optimizer)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(predictions, [])
        self.assertFalse(torch.equal(before, model.network.classifier[1].weight))
        self.assertTrue(any(p.grad is not None for p in model.network.features[-1].parameters()))
        self.assertTrue(all(torch.equal(value, model.network.features[0].state_dict()[name])
                            for name, value in frozen_buffers.items()))

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            stopper = baseline.EarlyStopping(7, 0.0)
            stopper.update(0.6)
            baseline.save_checkpoint(path, model, optimizer, epoch=1, config=self.config,
                                     provenance={"seed": 42, "validation_fold": 0},
                                     metrics={"train_loss": loss}, stopper=stopper)
            restored = BlackBoxEfficientNet(pretrained=False)
            restored.set_trainable_blocks(3)
            restored_optimizer = baseline.build_optimizer(restored, self.settings, 0.0001)
            payload = baseline.load_checkpoint(path, restored, restored_optimizer)
            self.assertEqual(payload["epoch"], 1)
            self.assertEqual(payload["provenance"]["validation_fold"], 0)
            self.assertEqual(restored.unfreeze_last_blocks, 3)
            self.assertTrue(all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items()))
            self.assertEqual(len(restored_optimizer.state), len(optimizer.state))

    def test_pretrained_default_selects_explicit_imagenet_enum_without_download(self):
        stub = nn.Module()
        stub.features = nn.Sequential(nn.Identity())
        stub.classifier = nn.Sequential(nn.Dropout(), nn.Linear(1280, 1000))
        with patch("faithful_medical_cbm.models.black_box.efficientnet_b0", return_value=stub) as constructor:
            BlackBoxEfficientNet()
        constructor.assert_called_once_with(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    def test_bce_and_validation_auroc(self):
        value = baseline.binary_loss(torch.zeros(2), torch.tensor([0., 1.]))
        self.assertAlmostEqual(value.item(), math.log(2), places=6)
        with self.assertRaises(ValueError):
            baseline.binary_loss(torch.zeros(2, 1), torch.zeros(2))
        rows = [{"diagnosis": y, "logit": x} for y, x in zip([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])]
        self.assertAlmostEqual(baseline.validation_auroc(rows), 0.75)
        with self.assertRaises(ValueError):
            baseline.validation_auroc(rows[:2])

    def test_validation_epoch_and_sample_weighted_loss(self):
        # Identity emits supplied synthetic logits, not a trained model.
        batches = [ImageDiagnosisOnly(image=torch.tensor([-2., 0.]), diagnosis=torch.tensor([0., 1.]),
                                     case_num=["a", "b"], split=["development"] * 2),
                   ImageDiagnosisOnly(image=torch.tensor([2.]), diagnosis=torch.tensor([1.]),
                                     case_num=["c"], split=["development"])]
        class Loader:
            dataset = SimpleNamespace(split="development", role="validation", case_nums=("a", "b", "c"))
            def __iter__(self):
                return iter(batches)
        loss, rows = baseline.run_epoch(nn.Identity(), Loader(), torch.device("cpu"))
        expected = baseline.binary_loss(torch.tensor([-2., 0., 2.]), torch.tensor([0., 1., 1.]))
        self.assertAlmostEqual(loss, expected.item(), places=6)
        self.assertEqual([r["case_num"] for r in rows], ["a", "b", "c"])

    def test_fold_isolation_without_any_image_or_locked_loader_access(self):
        fixture = fixtures.DataLoadingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with patch("PIL.Image.open", side_effect=AssertionError("No images required")), patch.object(
                LoaderFactory, "locked_test", side_effect=AssertionError("Never construct locked-test loader")):
            factory = LoaderFactory(fixture.config_path)
            for fold in range(4):
                loaders = baseline.development_fold_loaders(factory, fold)
                self.assertFalse(set(loaders["train"].dataset.case_nums) & set(loaders["validation"].dataset.case_nums))
                self.assertFalse(set(loaders["train"].dataset.case_nums) & set(factory.test_ids))
            bad = factory.fold(0)
            bad["train"].dataset.case_nums += (factory.test_ids[0],)
            with patch.object(factory, "fold", return_value=bad), self.assertRaises(ValueError):
                baseline.development_fold_loaders(factory, 0)

    def test_locked_loader_rejected_before_iteration(self):
        loader = Mock()
        loader.dataset.split = "test"
        with self.assertRaisesRegex(ValueError, "Refusing non-development"):
            baseline.run_epoch(nn.Identity(), loader, torch.device("cpu"))

    def test_cpu_run_stops_before_loader_or_model_construction(self):
        with patch("torch.cuda.is_available", return_value=False), patch(
                "faithful_medical_cbm.training.train_black_box.LoaderFactory") as factory, patch(
                "faithful_medical_cbm.training.train_black_box.BlackBoxEfficientNet") as model:
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                run_training(ROOT / "configs/default.toml", 0, "cpu-refused")
            factory.assert_not_called()
            model.assert_not_called()

    def test_gpu_entry_point_wiring_with_mocked_gpu_and_no_training(self):
        fixture = fixtures.DataLoadingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for i, row in enumerate(fixture.rows):
            row["diagnosis_binary"] = str((i // 4) % 2)
        fixture.refresh_metadata()
        with patch("torch.cuda.is_available", return_value=True), patch(
                "torch.cuda.get_device_name", return_value="mock GPU"), patch(
                "faithful_medical_cbm.training.train_black_box.seed_everything"), patch(
                "faithful_medical_cbm.training.train_black_box.BlackBoxEfficientNet") as model, patch(
                "faithful_medical_cbm.training.train_black_box.fit_fold", return_value={"mocked": True}) as fit, patch.object(
                LoaderFactory, "locked_test", side_effect=AssertionError("No locked loader")), patch(
                "PIL.Image.open", side_effect=AssertionError("No images for wiring test")):
            self.assertEqual(run_training(fixture.config_path, 0, "mock-run"), {"mocked": True})
            model.assert_called_once_with(pretrained=True)
            self.assertEqual(fit.call_args.args[2], torch.device("cuda"))
            artifact = fixture.root / "artifacts/baseline/mock-run/fold_0/run.json"
            record = json.loads(artifact.read_text())
            self.assertEqual(record["provenance"]["validation_fold"], 0)
            self.assertFalse(record["provenance"]["locked_test_images_accessed"])
            self.assertEqual(len(record["provenance"]["training_ids"]), 9)
            with self.assertRaises(FileExistsError):
                run_training(fixture.config_path, 0, "mock-run")
            self.assertEqual(fit.call_count, 1)

    def test_early_stopping_and_mocked_schedule_do_not_train(self):
        config = {**self.config, "baseline": {**self.settings, "epochs": 6, "head_epochs": 1, "patience": 2}}
        scores = iter([0.8, 0.7, 0.7])
        predictions = [{"case_num": "v", "diagnosis": 1, "logit": 0., "probability": 0.5}]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            checkpoint = Mock()
            with patch.object(baseline, "run_epoch", side_effect=lambda *a, **k: (0.5, predictions)), patch.object(
                    baseline, "build_optimizer", return_value=Mock()) as optimizers, patch.object(
                    baseline, "validation_auroc", side_effect=lambda rows: next(scores)), patch.object(
                    baseline, "save_checkpoint", checkpoint):
                result = baseline.fit_fold(Mock(), {"train": Mock(), "validation": Mock()}, torch.device("cpu"),
                                           config, directory, directory, {})
            self.assertEqual(result["epochs_completed"], 3)
            self.assertEqual(result["best_epoch"], 1)
            self.assertTrue(result["early_stopped"])
            self.assertEqual(optimizers.call_count, 2)
            names = [call.args[0].name for call in checkpoint.call_args_list]
            self.assertEqual(names.count("best.pt"), 1)
            self.assertEqual(names.count("last.pt"), 3)
            self.assertEqual(len(list(directory.glob("validation_epoch_*.csv"))), 3)


if __name__ == "__main__":
    unittest.main()
