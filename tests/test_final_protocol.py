"""Metadata-only Stage 14 checks. Never import a dataset, model or evaluator."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import freeze_development as freeze

ORDER = ["atypical_pigment_network", "regression_structures_present",
         "irregular_pigmentation", "blue_whitish_veil_present",
         "atypical_vascular_structures", "irregular_dots_and_globules", "irregular_streaks"]


class FinalProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = freeze.build_manifest()
        cls.protocol = cls.manifest["protocol"]

    def test_exact_order_epochs_and_thresholds(self):
        self.assertEqual(self.protocol["concept_order"], ORDER)
        self.assertEqual(self.protocol["selected_epochs"], {
            "black_box": [15, 9, 21, 10], "seven_concept": [19, 18, 16, 30],
            "joint_soft": [13, 15, 16, 14], "joint_hard_ste": [6, 18, 14, 16]})
        self.assertEqual(self.protocol["thresholds"], {"concept": .5, "diagnosis": .5, "operator": ">="})
        self.assertTrue(self.protocol["development_decisions_frozen"])
        for model, records in self.manifest["checkpoints"].items():
            self.assertEqual([r["epoch"] for r in records], self.protocol["selected_epochs"][model])
            self.assertEqual([r["fold"] for r in records], list(range(4)))
            for r in records:
                self.assertIsNone(r["binary_sha256"])
                self.assertFalse(r["binary_verified"])
                self.assertTrue(r["expected_path_relative_to_drive_outputs"].endswith(f"fold_{r['fold']}/best.pt"))

    def test_ensemble_rules(self):
        rules = self.protocol["ensemble_rules"]
        self.assertEqual(rules["black_box"], "arithmetic_mean_of_four_fold_diagnosis_probabilities")
        self.assertEqual(rules["sequential_concepts"], "arithmetic_mean_of_four_fold_soft_probabilities_per_concept")
        self.assertEqual(rules["joint_concepts"], rules["sequential_concepts"])
        self.assertEqual(rules["sequential_soft"], "full_development_soft_head(mean_concept_probabilities)")
        self.assertEqual(rules["sequential_hard"], "full_development_hard_head(mean_concept_probabilities >= 0.5)")
        self.assertEqual(rules["joint_soft"], "arithmetic_mean_of_four_end_to_end_fold_diagnosis_probabilities")
        self.assertEqual(rules["joint_hard_ste"], "arithmetic_mean_of_four_end_to_end_fold_diagnosis_probabilities; each_fold_uses_own_binary_concepts")
        self.assertEqual(rules["oracle"], "full_development_oracle_head(frozen_ground_truth_concepts); non_deployable")

    def test_full_development_heads_and_hashes(self):
        for model, head in self.manifest["full_development_heads"].items():
            self.assertEqual(head["path"], f"artifacts/cbm/{model}/full_development_model.json")
            self.assertTrue(head["label"].startswith("FULL-DEVELOPMENT"))
            self.assertEqual(head["training_cases"], 658)
            self.assertEqual(head["sha256"], freeze.sha(freeze.ROOT, head["path"]))
        self.assertEqual(self.manifest["split"]["counts"]["test"], {"total": 165, "positive": 50, "negative": 115})

    def test_policies_and_metrics(self):
        p = self.protocol["interventions"]
        self.assertEqual(p["models"], ["sequential_soft", "sequential_hard"])
        self.assertEqual(p["heads"], "saved_full_development_heads_only")
        self.assertEqual(p["budgets"], list(range(8)))
        self.assertEqual(p["q"], "original_four_model_mean_concept_probability")
        self.assertEqual(p["active"]["uncertainty"], "1 - 2 * abs(original q - 0.5)")
        self.assertEqual(p["active"]["impact"], "abs(p_force_1_current - p_force_0_current)")
        self.assertEqual(p["active"]["score"], "uncertainty * impact")
        self.assertEqual(p["active"]["truth_access"], "only selected concept after selection")
        self.assertEqual(p["tie_break"], "frozen concept order")
        random = p["random_error_oracle"]
        self.assertEqual((random["seed"], random["repetitions"]), (42, 100))
        self.assertEqual(random["rng"], "PCG64(SeedSequence([42, repetition]))")
        self.assertTrue(random["paired_orders_across_soft_hard"])
        for key in ("random_error_oracle", "confidently_wrong_oracle"):
            self.assertTrue(p[key]["non_deployable"])
            self.assertEqual(p[key]["exhaustion"], "stop and pad unchanged states")
        metrics = self.protocol["metrics"]
        self.assertEqual(metrics["diagnosis"], ["auroc", "macro_f1", "accuracy", "sensitivity", "specificity", "brier", "ece"])
        self.assertEqual(metrics["ece"]["bins"], 10)
        self.assertEqual(metrics["ece"]["formula"], "sum(n_bin/N * abs(mean_probability - fraction_positive))")

    def test_bootstrap(self):
        b = self.protocol["bootstrap"]
        self.assertTrue(b["enabled"])
        self.assertEqual((b["replicates"], b["seed"], self.protocol["seed"]), (2000, 42, 42))
        self.assertEqual((b["positive_count"], b["negative_count"]), (50, 115))
        self.assertEqual(b["rng"], "PCG64(42)")
        self.assertIn("same indices across all six models", b["sampling"])
        self.assertIn("all 15 unordered pairs", b["paired_differences"])
        self.assertEqual(b["metrics"], ["auroc", "macro_f1", "accuracy", "sensitivity", "specificity", "brier"])

    def test_read_allowlist_rejects_sensitive_paths_and_escape(self):
        for path in ["data/raw/release_v0/images/example.jpg", "data/processed/stage2a/cohort.csv",
                     "data/splits/test_ids.csv", "checkpoints/a.pt", "artifacts/final_test/results.json",
                     "../outside.json", "configs/../data/splits/test_ids.csv"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                freeze.read_bytes(freeze.ROOT, path)

    def test_builder_reads_metadata_only_and_has_no_loader_import(self):
        seen = []
        original = Path.read_bytes
        def guarded(path):
            relative = path.resolve().relative_to(freeze.ROOT).as_posix()
            self.assertFalse(relative.startswith(("data/raw/", "data/processed/", "checkpoints/")))
            self.assertNotEqual(relative, "data/splits/test_ids.csv")
            seen.append(relative)
            return original(path)
        with patch.object(Path, "read_bytes", guarded):
            result = freeze.build_manifest()
        self.assertTrue(seen)
        self.assertFalse(result["locked_test_accessed"])
        self.assertFalse(result["final_test_executed"])
        tree = ast.parse(Path(freeze.__file__).read_text(encoding="utf-8"))
        allowed = {"__future__", "csv", "hashlib", "json", "subprocess", "datetime", "pathlib"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue({a.name for a in node.names} <= allowed)
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)
        self.assertNotIn("DataLoader", ast.unparse(tree))

    def test_output_guards_and_no_final_results(self):
        freeze.ensure_no_final_outputs(freeze.ROOT)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / freeze.FINAL_OUTPUTS[0]).mkdir(parents=True)
            with self.assertRaises(ValueError):
                freeze.ensure_no_final_outputs(root)
            output = root / freeze.OUTPUT
            output.parent.mkdir(parents=True)
            output.write_text("sentinel", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                freeze.write_manifest(root)
            self.assertEqual(output.read_text(), "sentinel")

    def test_metadata_tampering_is_rejected(self):
        original = freeze.load
        for mode in ("epoch", "head"):
            def tampered(root, path):
                result = original(root, path)
                if mode == "epoch" and path.endswith("fold_0/summary.json"):
                    result["best_epoch"] = 999
                if mode == "head" and path.endswith("full_development_model.json"):
                    result["training_ids"] = []
                return result
            with self.subTest(mode=mode), patch.object(freeze, "load", side_effect=tampered), self.assertRaises(ValueError):
                freeze.build_manifest()

    def test_saved_manifest_matches_frozen_inputs(self):
        path = freeze.ROOT / freeze.OUTPUT
        if not path.exists():
            self.skipTest("Run again after one-time manifest creation")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["protocol"], self.protocol)
        self.assertEqual(manifest["checkpoints"], self.manifest["checkpoints"])
        self.assertEqual(manifest["full_development_heads"], self.manifest["full_development_heads"])
        self.assertEqual(manifest["split"], self.manifest["split"])
        for group in ("model_metadata_sha256", "source_config_docs_sha256"):
            self.assertEqual(manifest[group], self.manifest[group])
        self.assertFalse(manifest["locked_test_accessed"])
        self.assertFalse(manifest["final_test_executed"])


if __name__ == "__main__":
    unittest.main()
