"""Offline joint bottleneck tests; synthetic tensors, no downloads or dataset images."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from faithful_medical_cbm.config import load_config
from faithful_medical_cbm.models.joint_cbm import JointCBM, concept_representation, CONCEPT_ORDER
from faithful_medical_cbm.training import joint_cbm as t
from faithful_medical_cbm.training.train_joint import run_training
from faithful_medical_cbm.training.baseline import development_fold_loaders
from faithful_medical_cbm.data.loaders import LoaderFactory


class JointTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config(ROOT/'configs/joint_soft.toml')
        old=torch.get_num_threads(); torch.set_num_threads(2); self.addCleanup(torch.set_num_threads,old)
        self.addCleanup(patch.stopall)
        patch('torch.hub.download_url_to_file',side_effect=AssertionError('No downloads')).start()
        patch('PIL.Image.open',side_effect=AssertionError('No real images')).start()

    def test_shapes_strict_routes_and_soft_hard_inputs(self):
        for kind in ('joint_soft','joint_hard_ste'):
            model=JointCBM(kind,pretrained=False).eval()
            self.assertEqual(model.concept_names,CONCEPT_ORDER)
            self.assertEqual(set(dict(model.named_children())),{'concept_predictor','diagnosis_head'})
            self.assertIsInstance(model.diagnosis_head,nn.Linear)
            self.assertEqual((model.diagnosis_head.in_features,model.diagnosis_head.out_features),(7,1))
            captured=[]
            handle=model.diagnosis_head.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().clone()))
            with torch.no_grad(): out=model(torch.randn(1,3,224,224))
            handle.remove()
            self.assertEqual(out['concept_logits'].shape,(1,7)); self.assertEqual(out['diagnosis_logits'].shape,(1,))
            expected=out['concept_probabilities'] if kind=='joint_soft' else (out['concept_probabilities']>=.5).float()
            torch.testing.assert_close(captured[0],expected,rtol=0,atol=0)
            fixed=torch.arange(-3.,4.).reshape(1,7)
            with patch.object(model.concept_predictor,'forward',return_value=fixed):
                a=model(torch.zeros(1,3,32,32)); b=model(torch.ones(1,3,32,32))
            torch.testing.assert_close(a['diagnosis_logits'],b['diagnosis_logits'],rtol=0,atol=0)

    def test_ste_binary_forward_sigmoid_backward(self):
        logits=torch.tensor([[-3.,-1.,-.001,0.,.001,1.,3.]],requires_grad=True)
        representation=concept_representation(logits,'joint_hard_ste')
        self.assertEqual(representation.tolist(),[[0,0,0,1,1,1,1]])
        representation.sum().backward()
        q=torch.sigmoid(logits.detach())
        torch.testing.assert_close(logits.grad,q*(1-q))
        with self.assertRaises(ValueError): concept_representation(torch.zeros(1,6),'joint_soft')

    def test_loss_optimizer_transfer_and_checkpoint(self):
        for kind in ('joint_soft','joint_hard_ste'):
            model=JointCBM(kind,pretrained=False)
            self.assertTrue(all(not p.requires_grad for p in model.concept_predictor.network.features.parameters()))
            self.assertTrue(all(p.requires_grad for p in model.diagnosis_head.parameters()))
            self.assertTrue(all(p.requires_grad for p in model.concept_predictor.network.classifier.parameters()))
            model.set_trainable_blocks(3); model.train()
            blocks=model.concept_predictor.network.features
            for i,b in enumerate(blocks):
                self.assertEqual(b.training,i>=len(blocks)-3)
                self.assertTrue(all(p.requires_grad==(i>=len(blocks)-3) for p in b.parameters()))
            before=model.diagnosis_head.weight.detach().clone()
            frozen={k:v.clone() for k,v in blocks[0].state_dict().items()}
            outputs=model(torch.randn(2,3,64,64))
            concepts=torch.tensor([[0.]*7,[1.]*7]); diagnosis=torch.tensor([0.,1.]); weights=torch.arange(1.,8.)
            losses=t.joint_loss(outputs,concepts,diagnosis,weights)
            torch.testing.assert_close(losses['total_loss'],losses['concept_loss']+losses['diagnosis_loss'])
            torch.testing.assert_close(losses['diagnosis_loss'],nn.functional.binary_cross_entropy_with_logits(outputs['diagnosis_logits'],diagnosis))
            optimizer=t.build_optimizer(model,self.config['joint_training'],.0001)
            losses['total_loss'].backward(); optimizer.step()
            self.assertFalse(torch.equal(before,model.diagnosis_head.weight))
            self.assertTrue(any(p.grad is not None for p in blocks[-1].parameters()))
            self.assertTrue(all(torch.equal(v,blocks[0].state_dict()[k]) for k,v in frozen.items()))
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'best.pt'
                t.save_checkpoint(path,model,optimizer,epoch=4,config=self.config,provenance={'concept_order':list(CONCEPT_ORDER)},metrics={},stopper=t.EarlyStopping(7,0))
                restored=JointCBM(kind,pretrained=False); t.load_checkpoint(path,restored)
                self.assertTrue(all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items()))
                wrong=JointCBM('joint_soft' if kind=='joint_hard_ste' else 'joint_hard_ste',pretrained=False)
                with self.assertRaises(ValueError): t.load_checkpoint(path,wrong)
                payload=torch.load(path,weights_only=True); payload['model_config']['concept_order'].reverse(); torch.save(payload,path)
                with self.assertRaises(ValueError): t.load_checkpoint(path,restored)

    def test_frozen_configs_folds_training_weights_no_test_construction(self):
        with patch.object(LoaderFactory,'locked_test',side_effect=AssertionError('No test loader')):
            factory=LoaderFactory(ROOT/'configs/joint_soft.toml')
            for fold in range(4):
                loaders=development_fold_loaders(factory,fold)
                train,validation=loaders['train'].dataset,loaders['validation'].dataset
                self.assertFalse(set(train.case_nums)&set(validation.case_nums))
                self.assertFalse((set(train.case_nums)|set(validation.case_nums)) & set(factory.test_ids))
                weights=t.training_pos_weights(train)
                positives=train.concepts.double().sum(0)
                torch.testing.assert_close(weights,(len(train)-positives)/positives)
                with self.assertRaises(ValueError): t.training_pos_weights(validation)
                if fold==0: self.assertEqual(positives.tolist(),[129,133,155,107,33,229,134])
        for name in ('soft','hard'):
            cfg=load_config(ROOT/f'configs/joint_{name}.toml'); t.validate_config(cfg)
            for key,value in [('concept_loss_weight',2.),('selection_metric','validation_concept_macro_auroc'),('threshold',.4)]:
                changed=copy.deepcopy(cfg); changed['joint_training'][key]=value
                with self.assertRaises(ValueError): t.validate_config(changed)
            cfg['concepts']['target_order'].reverse()
            with self.assertRaises(ValueError): t.validate_config(cfg)

    def test_validation_metrics_and_raw_serialization(self):
        y=torch.tensor([0.,0.,1.,1.]); z=torch.logit(torch.tensor([.1,.6,.4,.8]))
        c=y[:,None].repeat(1,7); logits=z[:,None].repeat(1,7)
        for kind in ('joint_soft','joint_hard_ste'):
            out=dict(concept_logits=logits,concept_probabilities=logits.sigmoid(),diagnosis_logits=z,
                     diagnosis_inputs=concept_representation(logits,kind))
            rows=t.prediction_rows(['001','002','003','004'],c,y,out,kind)
            metrics=t.validation_metrics(rows)
            for key,value in {'auroc':.75,'macro_f1':.5,'accuracy':.5,'sensitivity':.5,'specificity':.5,'brier':.1925,'ece':.375}.items():
                self.assertAlmostEqual(metrics['validation_diagnosis_'+key],value,places=6)
            self.assertAlmostEqual(metrics['validation_concept_macro_auroc'],.75)
            self.assertAlmostEqual(metrics['validation_concept_macro_f1'],.5)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'pred.csv'; t.save_validation_predictions(path,rows,kind)
                with path.open() as stream:
                    reader=csv.DictReader(stream); restored=list(reader)
                    self.assertEqual(reader.fieldnames,t.prediction_columns(kind))
                    self.assertEqual(restored[0]['case_num'],'001')

    def test_cpu_guard_and_forbidden_loader_before_iteration(self):
        with patch('torch.cuda.is_available',return_value=False),patch('faithful_medical_cbm.training.train_joint.LoaderFactory') as factory:
            with self.assertRaisesRegex(RuntimeError,'requires CUDA'): run_training(ROOT/'configs/joint_soft.toml',0,'unit')
            factory.assert_not_called()
        class Forbidden:
            dataset=SimpleNamespace(split='test',role='validation')
            def __iter__(self): raise AssertionError('Test iteration')
        with self.assertRaisesRegex(ValueError,'non-development'): t.run_epoch(None,Forbidden(),'cpu',pos_weights=torch.ones(7))

    def test_synthetic_validation_epoch_and_bad_batch(self):
        model=JointCBM('joint_hard_ste',pretrained=False)
        batch=dict(image=torch.randn(2,3,64,64),concepts=torch.tensor([[0.]*7,[1.]*7]),
                   diagnosis=torch.tensor([0.,1.]),case_num=['a','b'],split=['development']*2)
        class Loader:
            dataset=SimpleNamespace(split='development',role='validation',case_nums=('a','b'),concept_columns=CONCEPT_ORDER)
            def __iter__(self): return iter([batch])
        losses,rows=t.run_epoch(model,Loader(),'cpu',pos_weights=torch.ones(7))
        self.assertEqual([r['case_num'] for r in rows],['a','b'])
        self.assertAlmostEqual(losses['total_loss'],losses['concept_loss']+losses['diagnosis_loss'],places=6)
        self.assertTrue(all(r[c+'_hard_input'] in (0,1) for r in rows for c in CONCEPT_ORDER))
        batch['case_num']=['a','a']
        with self.assertRaises(ValueError): t.run_epoch(model,Loader(),'cpu',pos_weights=torch.ones(7))

    def test_diagnosis_only_selection_and_raw_before_metrics(self):
        # Mock epochs, not a training experiment: concept metrics improve while diagnosis stagnates.
        cfg=self.config
        ds=lambda role,ids:SimpleNamespace(split='development',role=role,case_nums=ids,concept_columns=CONCEPT_ORDER,
                                          concepts=torch.tensor([[0.]*7,[1.]*7]))
        loaders={'train':SimpleNamespace(dataset=ds('train',['a','b'])),'validation':SimpleNamespace(dataset=ds('validation',['c','d']))}
        provenance={'training_ids':['a','b'],'validation_ids':['c','d'],'pos_weights':dict.fromkeys(CONCEPT_ORDER,1.)}
        model=JointCBM('joint_soft',pretrained=False)
        losses=dict(concept_loss=1.,diagnosis_loss=2.,total_loss=3.)
        saved=[]; checks=[]
        def metrics(rows):
            self.assertEqual(len(saved),len(checks)+1); checks.append(1)
            return {'validation_diagnosis_auroc':.7,'validation_concept_macro_auroc':len(checks)/10}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(t,'run_epoch',return_value=(losses,[])),patch.object(t,'save_validation_predictions',side_effect=lambda *args:saved.append(1)),patch.object(t,'validation_metrics',side_effect=metrics),patch.object(t,'save_checkpoint') as checkpoint:
                result=t.fit_fold(model,loaders,'cpu',cfg,Path(tmp),Path(tmp),provenance)
            self.assertEqual(result['best_epoch'],1); self.assertEqual(result['epochs_completed'],8)
            self.assertTrue(result['early_stopped']); self.assertEqual(checkpoint.call_count,9)


if __name__=='__main__': unittest.main()
