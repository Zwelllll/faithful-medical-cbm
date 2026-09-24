"""Stage 9 tabular tests; no images, CNNs or inference."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from faithful_medical_cbm.models.binary_cbm import binary_features, feature_columns, HARD_COLUMNS, ORACLE_COLUMNS
from faithful_medical_cbm.models.soft_cbm import FEATURE_COLUMNS
from faithful_medical_cbm.training import train_binary_cbm as trainer
from faithful_medical_cbm.evaluation.assemble_oof import read_csv, sha


class Allowed(dict):
    def __init__(self,values,allowed):
        super().__init__(values); self.allowed = allowed
    def __getitem__(self,key):
        if key not in self.allowed:
            raise AssertionError(f"Forbidden feature {key}")
        return super().__getitem__(key)


class BinaryCBMTests(unittest.TestCase):
    def test_threshold_and_independent_allowlists(self):
        probs = [0,.499999,.5,.500001,1,.2,.9]
        row = dict(zip(FEATURE_COLUMNS,map(str,probs)))
        row.update(dict.fromkeys(ORACLE_COLUMNS,"0"))
        np.testing.assert_array_equal(binary_features([Allowed(row,FEATURE_COLUMNS)],"sequential_hard"),[[0,0,1,1,1,0,1]])
        np.testing.assert_array_equal(binary_features([Allowed(row,ORACLE_COLUMNS)],"oracle"),np.zeros((1,7)))
        self.assertEqual(tuple(c.replace("_probability","_hard") for c in FEATURE_COLUMNS),HARD_COLUMNS)
        self.assertEqual(tuple(c.replace("_probability","_target") for c in FEATURE_COLUMNS),ORACLE_COLUMNS)
        for mode,cols,bad in (("oracle",ORACLE_COLUMNS,"0.5"),("oracle",ORACLE_COLUMNS,"2"),
                              ("sequential_hard",FEATURE_COLUMNS,"nan"),("sequential_hard",FEATURE_COLUMNS,"1.1")):
            broken=row.copy(); broken[cols[0]]=bad
            with self.assertRaises(ValueError): binary_features([broken],mode)
        with self.assertRaises(ValueError): binary_features([row],"soft")
        with self.assertRaises(KeyError): binary_features([{}],"oracle")

    def test_end_to_end_isolation_coefficients_comparison_and_no_full_predictions(self):
        cfg=ROOT/"configs/binary_cbm.json"
        _,data,rows,soft_hashes=trainer.load_inputs(cfg)
        real_cv_fit=trainer.shared.fit_head; real_full_fit=trainer.fit_head
        calls=[]; full_calls=[]
        def cv_fit(x,y,settings):
            mode=("sequential_hard","oracle")[len(calls)//4]; fold=len(calls)%4
            expected=binary_features(rows,mode); mask=data["folds"]!=fold
            np.testing.assert_array_equal(x,expected[mask]); np.testing.assert_array_equal(y,data["labels"][mask])
            self.assertEqual(settings,data["config"]["logistic_regression"])
            calls.append((mode,fold)); return real_cv_fit(x,y,settings)
        def full_fit(x,y,settings):
            self.assertEqual(len(x),658); full_calls.append(len(x))
            model=real_full_fit(x,y,settings)
            model.predict_proba=lambda *_: self.fail("Full model predicted")
            model.decision_function=lambda *_: self.fail("Full model scored")
            return model
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with patch.object(trainer.shared,"fit_head",side_effect=cv_fit), patch.object(trainer,"fit_head",side_effect=full_fit):
                result=trainer.run(cfg,out)
            self.assertEqual(len(calls),8); self.assertEqual(full_calls,[658,658])
            for mode in ("sequential_hard","oracle"):
                dest=out/mode; columns=list(feature_columns(mode)); x=binary_features(rows,mode)
                predictions=read_csv((dest/"cross_fitted_predictions.csv").read_bytes(),trainer.shared.PREDICTION_COLUMNS[:3]+columns+trainer.shared.PREDICTION_COLUMNS[3:])
                self.assertEqual([r["case_num"] for r in predictions],data["ids"])
                self.assertEqual(len({r["case_num"] for r in predictions}),658)
                self.assertFalse({r["case_num"] for r in predictions}&data["frozen"]["test"])
                for fold in range(4):
                    record=json.loads((dest/f"fold_{fold}_model.json").read_bytes())
                    val={i for i,f in zip(data["ids"],data["folds"]) if f==fold}
                    self.assertEqual(set(record["validation_ids"]),val)
                    self.assertEqual(set(record["training_ids"]),set(data["ids"])-val)
                    self.assertEqual(record["feature_columns"],columns)
                    self.assertEqual(list(record["coefficients_by_feature"]),columns)
                    self.assertEqual(list(record["coefficients_by_feature"].values()),record["coefficients"])
                    for j,r in enumerate(predictions):
                        if int(r["validation_fold"])==fold:
                            np.testing.assert_array_equal([int(r[c]) for c in columns],x[j])
                            self.assertAlmostEqual(float(r["decision_score"]),record["intercept"]+x[j]@record["coefficients"],places=12)
                full=json.loads((dest/"full_development_model.json").read_bytes())
                self.assertFalse(full["used_for_development_metrics"])
                self.assertEqual(full["training_ids"],data["ids"])
                pooled=json.loads((dest/"pooled_metrics.json").read_bytes())
                evaluated=trainer.shared.evaluate_rows([{k:float(v) if k in ("diagnosis_binary","decision_score","melanoma_probability") else v for k,v in r.items()} for r in predictions])
                for metric in trainer.METRICS: self.assertEqual(pooled[metric],evaluated[metric])
                integrity=json.loads((dest/"integrity.json").read_bytes())
                self.assertTrue(all(sha((dest/n).read_bytes())==h for n,h in integrity["output_sha256"].items()))
            soft=json.loads((ROOT/"artifacts/cbm/sequential_soft/pooled_metrics.json").read_bytes())
            self.assertEqual(result["pooled_metrics"]["soft"],{m:soft[m] for m in trainer.METRICS})
            for a,b in (("soft","hard"),("oracle","soft"),("oracle","hard")):
                for metric in ("auroc","macro_f1"):
                    self.assertEqual(result["descriptive_gaps"][f"{a}_minus_{b}"][metric],result["pooled_metrics"][a][metric]-result["pooled_metrics"][b][metric])
            with self.assertRaises(FileExistsError): trainer.run(cfg,out)
        trainer.verify_hashes(soft_hashes)

    def test_soft_tampering_rejected_before_fit(self):
        original=Path.read_bytes
        def read(path):
            content=original(path)
            if path.name=="pooled_metrics.json" and path.parent.name=="sequential_soft": return content+b"\n"
            return content
        with patch.object(Path,"read_bytes",read),patch.object(trainer.shared,"cross_fit") as fit:
            with self.assertRaisesRegex(ValueError,"Frozen artifact changed"):
                trainer.load_inputs(ROOT/"configs/binary_cbm.json")
            fit.assert_not_called()


if __name__=="__main__": unittest.main()
