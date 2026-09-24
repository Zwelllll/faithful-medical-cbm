"""Stage 8 tabular-only tests. No CNN construction, training or inference."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from faithful_medical_cbm.models import soft_cbm
from faithful_medical_cbm.evaluation.diagnosis_metrics import diagnosis_metrics, expected_calibration_error
from faithful_medical_cbm.evaluation.assemble_oof import read_csv, sha, OOF_COLUMNS
from faithful_medical_cbm.training import train_sequential_soft as trainer


class ProbabilityOnly(dict):
    def __getitem__(self, key):
        if key not in soft_cbm.FEATURE_COLUMNS:
            raise AssertionError(f"Forbidden feature: {key}")
        return super().__getitem__(key)


class SequentialSoftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = ROOT/"configs/sequential_soft.json"
        cls.data = trainer.load_inputs(cls.config_path)
        cls.raw = (ROOT/"artifacts/oof/seven_concept_oof.csv").read_bytes()

    def test_exact_feature_order_and_probability_only_allowlist(self):
        expected = (
            "atypical_pigment_network_probability", "regression_structures_present_probability",
            "irregular_pigmentation_probability", "blue_whitish_veil_present_probability",
            "atypical_vascular_structures_probability", "irregular_dots_and_globules_probability",
            "irregular_streaks_probability")
        self.assertEqual(soft_cbm.FEATURE_COLUMNS, expected)
        row = ProbabilityOnly({c: str(i/10) for i,c in enumerate(expected)})
        row.update(diagnosis_binary=1, image="forbidden", atypical_pigment_network_logit=100,
                   atypical_pigment_network_target=1, sex="forbidden")
        x = soft_cbm.extract_features([row])
        np.testing.assert_array_equal(x, [np.arange(7)/10])
        with self.assertRaises(ValueError): soft_cbm.extract_features([row], list(reversed(expected)))
        with self.assertRaises(ValueError): soft_cbm.extract_features([row], [c.replace("probability","logit") for c in expected])

    def test_frozen_coverage_and_malformed_rows(self):
        original = read_csv(self.raw, OOF_COLUMNS)
        cases = [
            lambda rows: rows.pop(),
            lambda rows: rows.append(rows[0].copy()),
            lambda rows: rows[0].update(case_num=next(iter(self.data["frozen"]["test"]))),
            lambda rows: rows[0].update(validation_fold=str((int(rows[0]["validation_fold"])+1)%4)),
            lambda rows: rows[0].update(diagnosis_binary="2"),
            lambda rows: rows[0].update(diagnosis_binary=str(1-int(rows[0]["diagnosis_binary"]))),
            lambda rows: rows[0].update(atypical_pigment_network_probability="nan"),
            lambda rows: rows[0].update(atypical_pigment_network_probability="inf"),
            lambda rows: rows[0].update(atypical_pigment_network_probability="1.01"),
            lambda rows: rows[0].update(atypical_pigment_network_probability="-.01"),
        ]
        for index, mutate in enumerate(cases):
            with self.subTest(index=index):
                rows = copy.deepcopy(original); mutate(rows)
                with self.assertRaises(ValueError): trainer.validate_rows(rows, self.data["frozen"])
        x, y, folds, ids = trainer.validate_rows(original[::-1], self.data["frozen"])
        self.assertEqual(ids, self.data["ids"])
        self.assertEqual(x.shape, (658,7)); self.assertEqual(y.sum(), 198)
        np.testing.assert_array_equal(x, self.data["features"])
        self.assertEqual(np.bincount(folds).tolist(), [165,165,164,164])

    def test_metrics_and_calibration_calculation(self):
        y = [0,0,1,1]; p = [.1,.6,.4,.8]
        result = diagnosis_metrics(y, np.log(np.array(p)/(1-np.array(p))), p)
        for key, expected in {"auroc":.75,"macro_f1":.5,"accuracy":.5,"sensitivity":.5,
                              "specificity":.5,"brier":.1925,"ece":.375}.items():
            self.assertAlmostEqual(result[key], expected)
        # Nonidentical sensitivity/specificity to detect a swapped denominator.
        result = diagnosis_metrics([0,0,0,1,1], [-2,-1,1,2,3], [.1,.2,.8,.9,.95])
        self.assertEqual(result["sensitivity"],1.)
        self.assertAlmostEqual(result["specificity"],2/3)
        boundary = expected_calibration_error([0,1,0,1,1], [0,.1,.3,.6,1])
        self.assertEqual([b["count"] for b in boundary["bins"]], [1,1,0,1,0,0,1,0,0,1])
        self.assertAlmostEqual(boundary["ece"], (.9+.3+.4)/5)
        self.assertEqual(expected_calibration_error([0,1],[0,1])["ece"],0.)
        self.assertAlmostEqual(expected_calibration_error([0,1],[.25,.75],bins=2)["ece"],.25)
        self.assertEqual(diagnosis_metrics([0,1],[-1,0],[.2,.5])["sensitivity"],1.)
        with self.assertRaises(ValueError): diagnosis_metrics([0,1],[0,np.nan],[.1,.9])
        with self.assertRaises(ValueError): expected_calibration_error([0,1],[.1,1.1])

    def test_fit_calls_exclude_validation_and_parameters_reproduce_scores(self):
        data = self.data
        real_fit = trainer.fit_head; calls = []
        def spy(x, y, settings):
            fold = len(calls); mask = data["folds"] != fold
            np.testing.assert_array_equal(x, data["features"][mask])
            np.testing.assert_array_equal(y, data["labels"][mask])
            calls.append(fold)
            return real_fit(x,y,settings)
        with patch.object(trainer,"fit_head",side_effect=spy):
            predictions, models = trainer.cross_fit(data,{"oof_sha256": data["oof_sha256"]})
        self.assertEqual(len(calls),4)
        self.assertEqual(len(predictions),658)
        before = data["features"].copy()
        for fold,record in enumerate(models):
            expected_val = {i for i,f in zip(data["ids"],data["folds"]) if f==fold}
            self.assertEqual(set(record["validation_ids"]),expected_val)
            self.assertEqual(set(record["training_ids"]),set(data["ids"])-expected_val)
            self.assertFalse(set(record["training_ids"]) & set(record["validation_ids"]))
            self.assertFalse(set(record["training_ids"]) & data["frozen"]["test"])
            self.assertEqual(record["feature_columns"],list(soft_cbm.FEATURE_COLUMNS))
            np.testing.assert_array_equal(list(record["coefficients_by_feature"].values()),record["coefficients"])
            for i,row in enumerate(predictions):
                if row["validation_fold"]==fold:
                    z = record["intercept"] + data["features"][i] @ np.array(record["coefficients"])
                    self.assertAlmostEqual(z,row["decision_score"],places=12)
                    self.assertAlmostEqual(float(expit(z)),row["melanoma_probability"],places=12)
        np.testing.assert_array_equal(data["features"],before)

    def test_serialization_full_head_never_predicts_and_no_overwrite(self):
        real_fit = trainer.fit_head; fits=[]; evaluation_calls=[]
        real_evaluate = trainer.evaluate_rows
        def evaluate(rows):
            evaluation_calls.append(len(rows))
            return real_evaluate(rows)
        def fit(x,y,settings):
            model=real_fit(x,y,settings); fits.append(len(x))
            if len(x)==658:
                self.assertEqual(evaluation_calls,[165,165,164,164,658])
                model.predict_proba=lambda *_: self.fail("Full head must not predict development")
                model.decision_function=lambda *_: self.fail("Full head must not score development")
            return model
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"soft"
            with patch.object(trainer,"fit_head",side_effect=fit), patch.object(trainer,"evaluate_rows",side_effect=evaluate):
                result=trainer.run(self.config_path,out)
            self.assertEqual(fits,[493,493,494,494,658])
            self.assertFalse(result["full_development_head_used"])
            records=read_csv((out/"cross_fitted_predictions.csv").read_bytes(),trainer.PREDICTION_COLUMNS)
            self.assertEqual(len(records),658)
            self.assertEqual([r["case_num"] for r in records],self.data["ids"])
            full=json.loads((out/"full_development_model.json").read_text())
            self.assertEqual(full["label"],"FULL-DEVELOPMENT SOFT CBM HEAD")
            self.assertFalse(full["used_for_development_metrics"])
            self.assertEqual(len(full["training_ids"]),658)
            integrity=json.loads((out/"integrity.json").read_text())
            self.assertTrue(all(sha((out/n).read_bytes())==h for n,h in integrity["output_sha256"].items()))
            with self.assertRaises(FileExistsError): trainer.run(self.config_path,out)

    def test_csv_preserves_ids_and_strict_header(self):
        row=dict(zip(trainer.PREDICTION_COLUMNS,["0001",0,1,.3,.574,1]))
        restored=read_csv(trainer.csv_bytes([row],trainer.PREDICTION_COLUMNS),trainer.PREDICTION_COLUMNS)
        self.assertEqual(restored[0]["case_num"],"0001")
        with self.assertRaises(ValueError): read_csv(self.raw, list(reversed(OOF_COLUMNS)))

    def test_oof_tampering_rejected_before_fit(self):
        original_read=Path.read_bytes
        def read(path):
            data=original_read(path)
            if path.name=="seven_concept_oof.csv":
                return data+b"\n"
            return data
        with patch.object(Path,"read_bytes",read), patch.object(trainer,"fit_head") as fit:
            with self.assertRaisesRegex(ValueError,"hash/schema"):
                trainer.run(self.config_path)
            fit.assert_not_called()


if __name__ == "__main__": unittest.main()
