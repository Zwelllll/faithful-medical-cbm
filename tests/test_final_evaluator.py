"""Stage 15A synthetic/metadata checks. No image access or CNN inference."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.special import expit
from faithful_medical_cbm.evaluation import final_core as core
from faithful_medical_cbm.evaluation import final_verification as verify
from faithful_medical_cbm.evaluation.final_evaluator import run
from faithful_medical_cbm.evaluation.final_interventions import FullHeadEngine, run_interventions
from faithful_medical_cbm.interventions.policies import PolicySession, select_active, trajectory, repetition_rng

ROOT = Path(__file__).resolve().parents[1]


def fake_heads():
    return {name:{'coefficients':[.2,-.3,.4,-.5,.6,-.7,.8],'intercept':-.2}
            for name in ('sequential_soft','sequential_hard','oracle')}


def fake_folds(n):
    rng = np.random.default_rng(6)
    result = {}
    for family in ('black_box','seven_concept','joint_soft','joint_hard_ste'):
        values = {}
        if family != 'seven_concept':
            z = rng.normal(size=(n,4)); values.update(diagnosis_logit=z, diagnosis_probability=expit(z))
        if family != 'black_box':
            z = rng.normal(size=(n,4,7)); values.update(concept_logit=z, concept_probability=expit(z))
        result[family] = values
    return result


class FinalEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.state = verify.frozen_state(ROOT)
        cls.folds = verify.development_folds(ROOT,cls.manifest)
        cls.order = cls.manifest['protocol']['concept_order']

    def payload(self, family='black_box', fold=0):
        cfg = {'architecture':{'black_box':'efficientnet_b0','seven_concept':'efficientnet_b0_concept',
                               'joint_soft':'efficientnet_b0_joint_cbm','joint_hard_ste':'efficientnet_b0_joint_cbm'}[family],
               'pretrained':True,'unfreeze_last_blocks':3}
        if family == 'seven_concept': cfg['concept_names'] = self.order.copy()
        if family.startswith('joint_'): cfg.update(model_type=family,concept_order=self.order.copy(),diagnosis_inputs=7)
        pre = {'validation_fold':fold, 'seed':42, 'locked_test_images_accessed':False,
               'training_ids':[i for i,f in self.folds.items() if f!=fold], 'validation_ids':[i for i,f in self.folds.items() if f==fold],
               'cohort_sha256':self.manifest['split']['cohort_sha256'],
               'split_metadata_sha256':self.manifest['split']['metadata_sha256']}
        if family != 'black_box': pre['concept_order'] = self.order.copy()
        return {'epoch':self.manifest['protocol']['selected_epochs'][family][fold], 'model_config':cfg,'provenance':pre,
                'loss_weights':{'concept':1.,'diagnosis':1.},
                'config':{'experiment':{'image_size':224},'normalization':{'mean':[.485,.456,.406],'std':[.229,.224,.225]}}}

    def test_frozen_source_and_empty_results(self):
        self.assertTrue(self.state['frozen_hashes_match'])
        self.assertFalse((ROOT/'artifacts/final_test').exists())
        self.assertEqual(self.manifest['protocol']['thresholds'],{'concept':.5,'diagnosis':.5,'operator':'>='})

    def test_authorization_barrier_precedes_every_read_and_write(self):
        with patch.object(Path,'open',side_effect=AssertionError('File access before authorization')), patch('builtins.__import__',side_effect=AssertionError('Import before authorization')):
            with self.assertRaises(PermissionError):
                run(Path('missing'),Path('missing'),Path('missing'))

    def test_arithmetic_ensembles_and_routing(self):
        folds = fake_folds(3); truth = np.array([[0]*7,[1]*7,[0,1,0,1,0,1,0]])
        heads = fake_heads(); d,c = core.route_predictions(folds,heads,truth)
        q = folds['seven_concept']['concept_probability'].mean(axis=1)
        np.testing.assert_array_equal(c['sequential_concept_ensemble'],q)
        np.testing.assert_allclose(d['sequential_soft'],expit(q@np.array(heads['sequential_soft']['coefficients'])-.2))
        np.testing.assert_allclose(d['sequential_hard'],core.head_probability(heads['sequential_hard'],q>=.5))
        np.testing.assert_allclose(d['oracle'],core.head_probability(heads['oracle'],truth))
        for family in ('black_box','joint_soft','joint_hard_ste'):
            np.testing.assert_allclose(d[family],folds[family]['diagnosis_probability'].mean(axis=1))
        for family in ('joint_soft','joint_hard_ste'):
            np.testing.assert_allclose(c[family+'_concept_ensemble'],folds[family]['concept_probability'].mean(axis=1))
        np.testing.assert_array_equal(core.binary([.499999,.5,.500001]),[0,1,1])
        self.assertAlmostEqual(core.ensemble([[.1,.2,.6,.9]])[0],.45)
        with self.assertRaises(ValueError): core.ensemble(np.zeros((3,3)))

    def test_metrics_and_ece(self):
        m = core.metrics([0,0,1,1],[0,.25,.5,1])
        self.assertEqual(m['auroc'],1); self.assertEqual(m['macro_f1'],1)
        self.assertAlmostEqual(m['ece'],.1875)
        self.assertAlmostEqual(m['brier'],.078125)
        rows = core.concept_metrics(np.tile([[0],[1]],(1,7)),np.tile([[.1],[.9]],(1,7)),self.order)
        self.assertEqual(rows[-1]['auroc'],1); self.assertEqual(rows[-1]['macro_f1'],1)
        undefined = core.concept_metrics(np.zeros((2,7)),np.zeros((2,7)),self.order)
        self.assertIsNone(undefined[-1]['auroc']); self.assertTrue(undefined[-1]['reason'])

    def test_bootstrap_determinism_counts_and_pairing(self):
        ids = [str(i) for i in range(165)]; y = np.array([1]*50+[0]*115)
        indices = core.bootstrap_indices(ids,y)
        np.testing.assert_array_equal(indices,core.bootstrap_indices(ids,y))
        self.assertEqual(indices.shape,(2000,165))
        self.assertTrue((y[indices].sum(axis=1)==50).all())
        p = np.linspace(.01,.99,165)
        predictions = {name:p.copy() for name in core.MODEL_ORDER}
        a,b = core.bootstrap_statistics(ids,y,predictions,indices)
        self.assertEqual(len(a),36); self.assertEqual(len(b),90)
        self.assertTrue(all(r['lower_95']==r['upper_95']==0 for r in b))
        manual = [core.metrics(y[i],p[i])['auroc'] for i in indices[:4]]
        # Verify vectorized ranking via a deliberately selected small independent subset.
        from scipy.stats import rankdata
        actual = ((rankdata(p[indices[:4]],axis=1)*y[indices[:4]]).sum(axis=1)-1275)/5750
        np.testing.assert_allclose(actual,manual)
        with self.assertRaises(ValueError): core.bootstrap_indices(ids[:-1],y[:-1])

    def test_prediction_schemas_and_serialization(self):
        ids = ['1','2']; y = np.array([0,1]); truth = np.array([[0]*7,[1]*7])
        tables,_,_ = core.prediction_tables(ids,y,truth,fake_folds(2),fake_heads(),self.order)
        schema = core.schemas(self.order)
        self.assertEqual(len(tables),7)
        with tempfile.TemporaryDirectory() as tmp:
            for name,rows in tables.items():
                self.assertEqual(set(rows[0]),set(schema[name])); self.assertEqual([r['case_num'] for r in rows],ids)
                core.write_csv(Path(tmp)/name,rows,schema[name])
                self.assertEqual((Path(tmp)/name).read_text().splitlines()[0],','.join(schema[name]))
                with self.assertRaises(FileExistsError): core.write_csv(Path(tmp)/name,rows,schema[name])

    def test_checkpoint_metadata_every_family_and_fold(self):
        for family,epochs in self.manifest['protocol']['selected_epochs'].items():
            for fold,epoch in enumerate(epochs):
                result = verify.validate_payload(self.payload(family,fold),family,fold,epoch,self.manifest,self.folds)
                self.assertEqual(result['family'],family)

    def test_checkpoint_rejects_mismatches_and_missing_provenance(self):
        mutations = [lambda p:p.update(epoch=999), lambda p:p['provenance'].update(validation_fold=1),
                     lambda p:p['model_config'].update(architecture='other'), lambda p:p['model_config'].update(model_type='joint_hard_ste'),
                     lambda p:p['model_config'].update(concept_order=list(reversed(self.order))),
                     lambda p:p['provenance'].update(locked_test_used=True), lambda p:p['provenance'].update(training_ids=[]),
                     lambda p:p['provenance'].pop('validation_fold')]
        for mutate in mutations:
            p = self.payload('joint_soft'); mutate(p)
            with self.assertRaises((ValueError,KeyError)):
                verify.validate_payload(p,'joint_soft',0,13,self.manifest,self.folds)

    def test_missing_binaries_report_zero_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            report = verify.verify_checkpoints(ROOT,Path(temp))
        self.assertEqual(report['verified_count'],0); self.assertEqual(len(report['records']),16)
        self.assertEqual(report['status'],'blocked'); self.assertFalse(report['inference_performed'])
        with self.assertRaises(ValueError): verify.validate_attestation(report,ROOT,ROOT,self.manifest)

    def test_attestation_binding_to_all_selected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); records = []
            for family,rows in self.manifest['checkpoints'].items():
                for selected in rows:
                    path = folder/selected['expected_path_relative_to_drive_outputs']
                    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'synthetic attestation fixture')
                    records.append(dict(family=family,fold=selected['fold'],epoch=selected['epoch'],
                        relative_path=selected['expected_path_relative_to_drive_outputs'],status='verified',
                        strict_state_dict_compatible=True,sha256=verify.sha(path),size_bytes=path.stat().st_size))
            report = dict(status='passed',verified_count=16,expected_count=16,records=records,
                manifest_sha256=verify.sha(ROOT/verify.MANIFEST),evaluator_sha256=verify.evaluator_hashes(ROOT),
                locked_test_accessed=False,final_test_executed=False,inference_performed=False)
            verify.validate_attestation(report,ROOT,folder,self.manifest)
            invalid = copy.deepcopy(report); invalid['records'][-1] = invalid['records'][0]
            with self.assertRaises(ValueError): verify.validate_attestation(invalid,ROOT,folder,self.manifest)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError): verify.validate_attestation(report,ROOT,folder,self.manifest)

    def test_state_dict_compatibility_without_forward(self):
        import torch
        model = verify.construct_model('black_box')
        payload = self.payload(); payload['model_state'] = model.state_dict()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'best.pt'; torch.save(payload,path)
            with patch.object(type(model),'forward',side_effect=AssertionError('No inference')):
                loaded,evidence = verify.load_verified_model(path,'black_box',0,15,self.manifest,self.folds)
            self.assertTrue(evidence['strict_state_dict_compatible']); self.assertEqual(evidence['sha256'],verify.sha(path))
            with self.assertRaises(ValueError): verify.load_verified_model(path,'black_box',0,15,self.manifest,self.folds,'0'*64)
            payload['model_state'] = dict(payload['model_state']); payload['model_state'].pop('network.classifier.1.bias')
            torch.save(payload,path)
            with self.assertRaises(RuntimeError): verify.load_verified_model(path,'black_box',0,15,self.manifest,self.folds)
        del model,loaded

    def test_active_truth_separation_and_full_head_adapter(self):
        ids = ['1','2']; q = np.full((2,7),.5); truth = np.array([[0]*7,[1]*7])
        engine = FullHeadEngine(ids,q,truth,[0,1],fake_heads())
        session = PolicySession(engine,'1','soft')
        class NoTruth(dict):
            def __getitem__(self,key): raise AssertionError('Truth accessed before selection')
        original = engine._truth; engine._truth = NoTruth()
        view = session.view(); selected,details = select_active(view)
        self.assertEqual(details['uncertainty'],1.)
        j = self.order.index(selected)
        self.assertAlmostEqual(details['active_score'],abs(view.force_one[j]-view.force_zero[j]))
        self.assertFalse(hasattr(view,'truth')); engine._truth = original
        session.reveal_and_correct(selected)
        self.assertEqual(session.values,[0])
        for mode in ('soft','hard'):
            rows = trajectory(engine,'1',mode,'active')
            self.assertEqual(len(rows),8); self.assertEqual(rows[-1]['concepts_queried'],7)
            self.assertEqual(rows[-1]['validation_fold'],-1)
        soft = trajectory(engine,'1','soft','random_error_oracle',rng=repetition_rng(42,0))
        hard = trajectory(engine,'1','hard','random_error_oracle',rng=repetition_rng(42,0))
        self.assertEqual([r['selected_concept'] for r in soft],[r['selected_concept'] for r in hard])

    def test_synthetic_intervention_outputs_and_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)/'synthetic_interventions'
            run_interventions(destination,['a','b'],np.full((2,7),.5),np.array([[0]*7,[1]*7]),[0,1],fake_heads())
            self.assertEqual(len(list(destination.glob('*_trajectories.csv'))),6)
            area = json.loads((destination/'auc_summary.json').read_text())
            self.assertEqual(len(area['models']['soft']['random_error_oracle']['auroc']['per_repetition']),100)
            self.assertTrue((destination/'intervention_curves.csv').exists())


if __name__ == '__main__': unittest.main()
