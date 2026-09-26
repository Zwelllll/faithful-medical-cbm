"""Stage 13 saved-prediction integrity tests; no models or checkpoints."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from faithful_medical_cbm.evaluation import joint_oof as j
from faithful_medical_cbm.evaluation.assemble_oof import frozen_inputs,read_csv,OOF_COLUMNS,sha


class JointOOFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=ROOT/'configs/joint_oof.json'
        cls.frozen=frozen_inputs(ROOT/'configs/joint_soft.toml')
        cls.truth={r['case_num']:r for r in read_csv((ROOT/'artifacts/oof/seven_concept_oof.csv').read_bytes(),OOF_COLUMNS)}
        cls.d=ROOT/'artifacts/joint_hard/joint-hard-v1/fold_0'
        cls.rows=read_csv((cls.d/'validation_epoch_006.csv').read_bytes(),j.columns('joint_hard_ste'))
        cls.run_record=json.loads((cls.d/'run.json').read_bytes()); cls.summary=json.loads((cls.d/'summary.json').read_bytes())

    def test_best_epoch_model_order_flags_and_fold_isolation(self):
        def validate(run,summary): j.validate_run(run,summary,'joint_hard_ste',0,6,self.frozen)
        validate(self.run_record,self.summary)
        for mutate in [lambda r:r['provenance']['training_ids'].append(r['provenance']['validation_ids'][0]),
                       lambda r:r['provenance'].update(validation_fold=1),
                       lambda r:r['provenance']['concept_order'].reverse(),
                       lambda r:r['provenance'].update(locked_test_used=True),
                       lambda r:r['provenance'].update(model_type='joint_soft')]:
            run=copy.deepcopy(self.run_record); mutate(run)
            with self.assertRaises(ValueError): validate(run,self.summary)
        for key,value in [('best_epoch',7),('locked_test_used',True),('model_type','joint_soft')]:
            summary=copy.deepcopy(self.summary); summary[key]=value
            with self.assertRaises(ValueError): validate(self.run_record,summary)

    def test_missing_duplicate_wrong_fold_test_and_invalid_fields(self):
        c=j.CONCEPT_ORDER[0]
        edits=[lambda r:r.pop(),lambda r:r.append(r[0].copy()),
               lambda r:r[0].update(case_num=next(iter(self.frozen['test']))),
               lambda r:r[0].update(case_num=next(i for i,f in self.frozen['folds'].items() if f==1)),
               lambda r:r[0].update(diagnosis_binary='2'),
               lambda r:r[0].update(diagnosis_binary=str(1-int(r[0]['diagnosis_binary']))),
               lambda r:r[0].update(diagnosis_probability='nan'),
               lambda r:r[0].update(diagnosis_probability='1.1'),
               lambda r:r[0].update(diagnosis_logit='inf'),
               lambda r:r[0].update({c+'_target':''}),
               lambda r:r[0].update({c+'_target':str(1-int(r[0][c+'_target']))}),
               lambda r:r[0].update({c+'_hard_input':'.5'}),
               lambda r:r[0].update({c+'_hard_input':str(1-int(r[0][c+'_hard_input']))})]
        for edit in edits:
            rows=copy.deepcopy(self.rows); edit(rows)
            with self.assertRaises(ValueError): j.validate_rows(rows,'joint_hard_ste',0,self.frozen,self.truth)
        with self.assertRaises(ValueError): read_csv((self.d/'validation_epoch_006.csv').read_bytes(),list(reversed(j.columns('joint_hard_ste'))))

    def test_metrics_calibration_and_probability_concept_auroc(self):
        rows=[]
        for i,(y,p) in enumerate([(0,.1),(0,.6),(1,.4),(1,.8)]):
            r={'case_num':str(i),'diagnosis_binary':y,'diagnosis_logit':float(np.log(p/(1-p))),'diagnosis_probability':p}
            for c in j.CONCEPT_ORDER: r.update({c+'_target':y,c+'_probability':p,c+'_logit':r['diagnosis_logit']})
            rows.append(r)
        diagnosis=j.evaluate(rows); concepts=j.concept_metrics(rows); diagnostics=j.diagnostics(rows)
        for key,value in {'auroc':.75,'macro_f1':.5,'accuracy':.5,'sensitivity':.5,'specificity':.5,'brier':.1925,'ece':.375}.items(): self.assertAlmostEqual(diagnosis[key],value)
        self.assertEqual(concepts['macro_auroc'],.75); self.assertEqual(concepts['macro_f1'],.5)
        self.assertAlmostEqual(diagnostics['mean_probability'],.475)
        self.assertAlmostEqual(diagnostics['mean_probability_melanoma'],.6)
        self.assertAlmostEqual(diagnostics['mean_probability_benign'],.35)
        self.assertEqual(diagnostics['fraction_positive'],.5)
        self.assertEqual(diagnostics['fraction_melanoma_positive'],.5)
        self.assertEqual(diagnostics['fraction_benign_positive'],.5)
        for r in rows:
            for c in j.CONCEPT_ORDER: r[c+'_logit']=-r[c+'_logit']
        self.assertEqual(j.concept_metrics(rows)['macro_auroc'],.75) # Uses probabilities as requested.

    def test_end_to_end_serialization_coverage_comparison_and_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'oof'; result=j.assemble(self.cfg,out)
            self.assertEqual(result['rows_per_fold'],[165,165,164,164])
            self.assertEqual([r['best_epoch'] for r in result['sources']],[13,15,16,14,6,18,14,16])
            for model,stem in [('joint_soft','joint_soft'),('joint_hard_ste','joint_hard')]:
                rows=read_csv((out/f'{stem}_oof.csv').read_bytes(),j.columns(model,assembled=True))
                self.assertEqual(len(rows),658); self.assertEqual([r['case_num'] for r in rows],self.frozen['development'])
                self.assertFalse({r['case_num'] for r in rows}&self.frozen['test'])
                self.assertEqual([sum(int(r['validation_fold'])==f for r in rows) for f in range(4)],[165,165,164,164])
                for r in rows:
                    self.assertEqual(int(r['validation_fold']),self.frozen['folds'][r['case_num']])
                    if model=='joint_hard_ste':
                        for c in j.CONCEPT_ORDER: self.assertEqual(int(r[c+'_hard_input']),int(float(r[c+'_probability'])>=.5))
                metrics=json.loads((out/f'{stem}_metrics.json').read_bytes())
                self.assertEqual(sum(b['count'] for b in metrics['calibration']['bins']),658)
                self.assertAlmostEqual(sum(b['weighted_absolute_gap'] for b in metrics['calibration']['bins']),metrics['diagnosis']['ece'])
            lookup={r['model']:r for r in result['diagnosis_comparison']}
            for m,folder in [('sequential_soft','sequential_soft'),('sequential_hard','sequential_hard'),('oracle','oracle')]:
                saved=json.loads((ROOT/f'artifacts/cbm/{folder}/pooled_metrics.json').read_bytes())
                for key in j.DIAGNOSIS_METRICS: self.assertEqual(lookup[m][key],saved[key])
            self.assertEqual(result['diagnosis_gaps']['joint_soft_minus_sequential_soft']['auroc'],lookup['joint_soft']['auroc']-lookup['sequential_soft']['auroc'])
            integrity=json.loads((out/'integrity.json').read_bytes())
            for name,h in integrity['input_sha256'].items(): self.assertEqual(sha(Path(name).read_bytes()),h)
            for name,h in integrity['output_sha256'].items(): self.assertEqual(sha((out/name).read_bytes()),h)
            with self.assertRaises(FileExistsError): j.assemble(self.cfg,out)

    def test_source_hash_tampering_fails_without_outputs(self):
        original=Path.read_bytes
        def read(path):
            raw=original(path)
            return raw+b'\n' if path.name=='validation_epoch_006.csv' else raw
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'oof'
            with patch.object(Path,'read_bytes',read):
                with self.assertRaisesRegex(ValueError,'Source hash mismatch'): j.assemble(self.cfg,out)
            self.assertFalse(out.exists())


if __name__=='__main__': unittest.main()
