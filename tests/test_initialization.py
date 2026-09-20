import importlib.util
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from faithful_medical_cbm.config import load_config
from faithful_medical_cbm.reproducibility import seed_everything


class ConfigurationTests(unittest.TestCase):
    def test_defaults_and_paths(self):
        config = load_config(ROOT / "configs/default.toml")
        self.assertEqual(config["experiment"]["num_folds"], 4)
        self.assertEqual(config["experiment"]["model_name"], "efficientnet_b0")
        self.assertEqual(len(config["concepts"]["names"]), 7)
        self.assertEqual(config["paths"]["raw"], ROOT / "data/raw")

    def test_invalid_batch_size_rejected(self):
        source = (ROOT / "configs/default.toml").read_text()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.toml"
            path.write_text(source.replace("batch_size = 32", "batch_size = 0"))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_invalid_seed_rejected_before_imports(self):
        with self.assertRaises(ValueError):
            seed_everything(-1)


@unittest.skipUnless(
    importlib.util.find_spec("numpy") and importlib.util.find_spec("torch"),
    "NumPy and PyTorch must be installed to test RNG reproducibility",
)
class ReproducibilityTests(unittest.TestCase):
    def test_reseeding_repeats_all_available_generators(self):
        import numpy as np
        import torch
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        old_cudnn = torch.backends.cudnn.deterministic
        old_benchmark = torch.backends.cudnn.benchmark
        def draw():
            values = [random.random(), np.random.rand(), torch.rand(3).tolist()]
            if torch.cuda.is_available():
                values.append(torch.rand(3, device="cuda").cpu().tolist())
            return values
        try:
            seed_everything(42)
            first = draw()
            seed_everything(42)
            self.assertEqual(first, draw())
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            seed_everything(43)
            self.assertNotEqual(first, draw())
        finally:
            torch.use_deterministic_algorithms(old_deterministic)
            torch.backends.cudnn.deterministic = old_cudnn
            torch.backends.cudnn.benchmark = old_benchmark


if __name__ == "__main__":
    unittest.main()
