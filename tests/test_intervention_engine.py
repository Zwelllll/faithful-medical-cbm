"""Stage 10 tests never fit a model or access images."""
import copy
from dataclasses import fields, FrozenInstanceError
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from faithful_medical_cbm.interventions.engine import InterventionEngine, CONCEPT_ORDER, validate_head
from faithful_medical_cbm.interventions import build_engine_artifacts as builder
from faithful_medical_cbm.evaluation.assemble_oof import read_csv, sha


class InterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=ROOT/'configs/intervention_engine.json'
        cls.engine=InterventionEngine(cls.config)

    def test_baseline_all_cases_both_models_and_fold_selection(self):
        e=self.engine
        self.assertEqual(len(e.ids),658)
        for mode,folder in [('soft','sequential_soft'),('hard','sequential_hard')]:
            rows=read_csv((ROOT/'artifacts/cbm'/folder/'cross_fitted_predictions.csv').read_bytes())
            for r in rows:
                a=e.intervene(r['case_num'],mode,[],[])
                self.assertLessEqual(abs(a['original_diagnosis_score']-float(r['decision_score'])),1e-12)
                self.assertLessEqual(abs(a['original_melanoma_probability']-float(r['melanoma_probability'])),1e-12)
                self.assertEqual(a['signed_probability_change'],0)
                self.assertEqual(a['validation_fold'],int(r['validation_fold']))
                head=json.loads((ROOT/'artifacts/cbm'/folder/f"fold_{r['validation_fold']}_model.json").read_bytes())
                self.assertNotIn(r['case_num'],head['training_ids'])
                self.assertIn(r['case_num'],head['validation_ids'])

    def test_replacements_untouched_values_forced_and_sequential(self):
        e=self.engine; case=e.ids[0]; names=[CONCEPT_ORDER[3],CONCEPT_ORDER[0],CONCEPT_ORDER[6]]
        for mode in ('soft','hard'):
            original=e.intervene(case,mode,[],[])
            a=e.correct(case,mode,names)
            for j,c in enumerate(CONCEPT_ORDER):
                expected=e._truth[case][c] if c in names else original['original_concept_inputs'][j]
                self.assertEqual(a['intervened_concept_inputs'][j],expected)
            trajectory=e.trajectory(case,mode,names)
            self.assertEqual([r['k'] for r in trajectory],[0,1,2,3])
            for k,r in enumerate(trajectory):
                self.assertEqual(r['intervened_melanoma_probability'],e.correct(case,mode,names[:k])['intervened_melanoma_probability'])
            self.assertEqual(trajectory[-1]['intervened_concept_inputs'],e.correct(case,mode,names[::-1])['intervened_concept_inputs'])
            intercept,weights=e._heads[mode,a['validation_fold']]
            for c in CONCEPT_ORDER:
                for value in (0,1):
                    forced=e.intervene(case,mode,[c],[value])
                    x=original['original_concept_inputs'].copy(); x[CONCEPT_ORDER.index(c)]=value
                    self.assertEqual(forced['intervened_concept_inputs'],x)
                    expected_score=intercept+np.array(x)@weights
                    self.assertAlmostEqual(forced['intervened_diagnosis_score'],expected_score,places=14)
                    self.assertAlmostEqual(forced['intervened_melanoma_probability'],float(expit(expected_score)),places=14)
                    self.assertEqual(forced['operation'],'forced_value')
            self.assertEqual(a['operation'],'ground_truth_correction')

    def test_policy_has_no_truth_and_executor_reads_only_after_selection(self):
        e=copy.copy(self.engine); case=e.ids[0]; chosen=CONCEPT_ORDER[2]; selected=False; reads=[]
        original=e._truth[case][chosen]
        class Targets(dict):
            def __getitem__(self,key):
                if not selected or key!=chosen: raise AssertionError('Unselected target access')
                reads.append(key); return original
        e._truth={case:Targets()}
        def selector(view):
            nonlocal selected
            self.assertEqual(tuple(view.concept_order),CONCEPT_ORDER)
            self.assertEqual({f.name for f in fields(view)}, {'case_num','model_type','concept_order','concept_probabilities',
                'original_inputs','original_melanoma_probability','forced_zero_probabilities','forced_one_probabilities'})
            self.assertFalse(hasattr(view,'__dict__'))
            with self.assertRaises(FrozenInstanceError): view.case_num='bad'
            self.assertEqual(reads,[]); selected=True
            return [chosen]
        e.selection_view(case,'soft')
        e.intervene(case,'hard',[chosen],[1])
        self.assertEqual(reads,[])
        result=e.select_then_correct(case,'soft',selector)
        self.assertEqual(reads,[chosen]); self.assertEqual(result['intervention_values'],[original])

    def test_malformed_and_test_ids_rejected(self):
        e=self.engine; case=e.ids[0]; c=CONCEPT_ORDER[0]
        for concepts,values in [(['bad'],[0]),([c,c],[0,1]),([c],[.2]),([c],[np.nan]),([c],[np.inf]),([c],['1']),([c],[]),(c,[1])]:
            with self.subTest(concepts=concepts,values=values):
                with self.assertRaises(ValueError): e.intervene(case,'soft',concepts,values)
        with self.assertRaises(ValueError): e.correct(case,'oracle',[c])
        with self.assertRaises(ValueError): e.trajectory(case,'soft',[c,c])
        with self.assertRaises(ValueError): e.correct(case,'soft',['unknown'])
        test_ids=read_csv((ROOT/'data/splits/test_ids.csv').read_bytes())
        for r in test_ids:
            with self.assertRaises(ValueError): e.selection_view(r['case_num'],'soft')

    def test_bad_heads_rejected(self):
        e=self.engine
        record=json.loads((ROOT/'artifacts/cbm/sequential_soft/fold_0_model.json').read_bytes())
        def check(r): validate_head(r,'soft',0,list(e.ids),e._folds,record['provenance']['oof_sha256'],record['logistic_regression'])
        check(record)
        mutations=[lambda r:r.update(validation_fold=1),lambda r:r.update(validation_fold=None),
            lambda r:r.update(label='FULL-DEVELOPMENT SOFT CBM HEAD'),lambda r:r['training_ids'].append(r['validation_ids'][0]),
            lambda r:r['validation_ids'].pop(),lambda r:r['feature_columns'].reverse(),
            lambda r:r['coefficients'].__setitem__(0,float('nan'))]
        for mutate in mutations:
            broken=copy.deepcopy(record); mutate(broken)
            with self.assertRaises(ValueError): check(broken)

    def test_reproduction_failure_is_fatal(self):
        original=InterventionEngine.intervene
        def altered(engine,*args,**kwargs):
            result=original(engine,*args,**kwargs); result['original_melanoma_probability']+=1e-6
            return result
        with patch.object(InterventionEngine,'intervene',altered):
            with self.assertRaisesRegex(ValueError,'reproduction failed'): InterventionEngine(self.config)

    def test_tables_summaries_determinism_no_fitting_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            first,second=Path(tmp)/'a',Path(tmp)/'b'
            with patch('sklearn.linear_model.LogisticRegression.fit',side_effect=AssertionError('No fitting')):
                one=builder.run(self.config,first); two=builder.run(self.config,second)
            self.assertEqual(one,two)
            for mode in ('soft','hard'):
                correction=read_csv((first/f'{mode}_single_concept_ground_truth_corrections.csv').read_bytes(),builder.CORRECTION_COLUMNS)
                forced=read_csv((first/f'{mode}_forced_value_counterfactuals.csv').read_bytes(),builder.FORCED_COLUMNS)
                self.assertEqual(len(correction),4606); self.assertEqual(len(forced),9212)
                self.assertEqual({r['case_num'] for r in correction},set(self.engine.ids))
                self.assertEqual([r['concept_name'] for r in correction[:7]],list(CONCEPT_ORDER))
                wrong=[r for r in correction if int(r['concept_was_wrong'])]
                self.assertEqual(one['models'][mode]['incorrect_concepts'],len(wrong))
                self.assertAlmostEqual(one['models'][mode]['absolute_effect_wrong_concepts']['mean'],np.mean([float(r['absolute_probability_change']) for r in wrong]))
                for r in correction:
                    if mode=='hard' and r['concept_was_wrong']=='0': self.assertEqual(float(r['absolute_probability_change']),0)
                for suffix in ('single_concept_ground_truth_corrections','forced_value_counterfactuals'):
                    name=f'{mode}_{suffix}.csv'; self.assertEqual((first/name).read_bytes(),(second/name).read_bytes())
            integrity=json.loads((first/'integrity.json').read_bytes())
            self.assertTrue(all(sha((first/n).read_bytes())==h for n,h in integrity['output_sha256'].items()))
            with self.assertRaises(FileExistsError): builder.run(self.config,first)
            with self.assertRaises(ValueError): builder.run(self.config,ROOT/'artifacts/cbm')


if __name__=='__main__': unittest.main()
