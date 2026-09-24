"""Offline Stage 6 checks; synthetic images/tensors and no downloads."""
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
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from faithful_medical_cbm.config import load_config
from faithful_medical_cbm.models.seven_concept import SevenConceptEfficientNet, CONCEPT_ORDER
from faithful_medical_cbm.models.concept import ConceptEfficientNet
from faithful_medical_cbm.training import seven_concept as training
from faithful_medical_cbm.training.train_seven_concept import run_training
from faithful_medical_cbm.training.baseline import development_fold_loaders
from faithful_medical_cbm.data.loaders import LoaderFactory
from test_concept import ConceptOnly, BatchLoader
import test_data_loading as fixtures


class SevenConceptTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config(ROOT/'configs/seven_concept.toml')
        self.addCleanup(patch.stopall)
        patch('torch.hub.download_url_to_file',side_effect=AssertionError('No downloads')).start()

    def test_model_forward_step_and_checkpoint(self):
        old=torch.get_num_threads(); torch.set_num_threads(2); self.addCleanup(torch.set_num_threads,old)
        model=SevenConceptEfficientNet(pretrained=False)
        self.assertEqual(model.concept_names,CONCEPT_ORDER)
        self.assertEqual(model.network.classifier[1].out_features,7)
        with torch.no_grad():
            self.assertEqual(model(torch.randn(1,3,224,224)).shape,(1,7))
        model.set_trainable_blocks(3)
        frozen={k:v.clone() for k,v in model.network.features[0].state_dict().items()}
        before=model.network.classifier[1].weight.detach().clone()
        batch=ConceptOnly(image=torch.randn(2,3,64,64),concepts=torch.tensor([[0.]*7,[1.]*7]),case_num=['a','b'],split=['development']*2)
        optimizer=training.build_optimizer(model,self.config['concept_training'],.0001)
        loss,rows=training.run_epoch(model,BatchLoader(batch),torch.device('cpu'),optimizer,pos_weights=torch.arange(1.,8.))
        self.assertTrue(math.isfinite(loss)); self.assertEqual(rows,[])
        self.assertFalse(torch.equal(before,model.network.classifier[1].weight))
        self.assertTrue(all(torch.equal(v,model.network.features[0].state_dict()[k]) for k,v in frozen.items()))
        self.assertTrue(any(p.grad is not None for p in model.network.features[-1].parameters()))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'best.pt'
            training.save_checkpoint(path,model,optimizer,epoch=1,config=self.config,
                provenance={'concept_order':list(CONCEPT_ORDER),'pos_weights':list(range(1,8))},metrics={'train_loss':loss},stopper=training.EarlyStopping(7,0))
            restored=SevenConceptEfficientNet(pretrained=False); restored.set_trainable_blocks(3)
            restored_optimizer=training.build_optimizer(restored,self.config['concept_training'],.0001)
            payload=training.load_checkpoint(path,restored,restored_optimizer)
            self.assertEqual(payload['model_config']['concept_names'],list(CONCEPT_ORDER))
            self.assertTrue(all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items()))
            self.assertEqual(len(optimizer.state),len(restored_optimizer.state))
            with self.assertRaises(ValueError): training.load_checkpoint(path,ConceptEfficientNet(pretrained=False))
            payload['model_config']['concept_names']=list(reversed(CONCEPT_ORDER)); torch.save(payload,path)
            with self.assertRaises(ValueError): training.load_checkpoint(path,restored)

    def test_order_and_each_target_and_unchanged_schedule(self):
        self.assertEqual(CONCEPT_ORDER,fixtures.CONCEPTS)
        training.check_order(self.config['concepts']['target_order'])
        targets=torch.eye(7)
        result=training.extract_targets(targets,CONCEPT_ORDER)
        for i in range(7): self.assertEqual(result[:,i].tolist(),targets[:,i].tolist())
        with self.assertRaises(ValueError): training.extract_targets(targets,tuple(reversed(CONCEPT_ORDER)))
        with self.assertRaises(ValueError): training.extract_targets(torch.zeros(2,6),CONCEPT_ORDER)
        with self.assertRaises(ValueError): training.extract_targets(torch.full((2,7),.5),CONCEPT_ORDER)
        previous=load_config(ROOT/'configs/concept_sanity.toml')['concept_training']
        for key in ('head_epochs','learning_rate','finetune_learning_rate','unfreeze_last_blocks','epochs','patience','optimizer','weight_decay'):
            self.assertEqual(previous[key],self.config['concept_training'][key])
        with self.assertRaises(ValueError): training.validate_settings({**self.config['concept_training'],'threshold':.4})

    def test_seven_training_weights_and_bce(self):
        values=torch.stack([(torch.arange(8)<i+1).float() for i in range(7)],dim=1)
        ds=SimpleNamespace(split='development',role='train',concepts=values,concept_columns=CONCEPT_ORDER)
        weights=training.training_pos_weights(ds)
        expected=torch.tensor([(7-i)/(i+1) for i in range(7)],dtype=torch.float64)
        self.assertTrue(torch.equal(weights,expected))
        loss=training.concept_loss(torch.zeros_like(values),values,weights)
        expected_loss=nn.functional.binary_cross_entropy_with_logits(torch.zeros_like(values),values,pos_weight=expected.float())
        self.assertAlmostEqual(loss.item(),expected_loss.item())
        ds.role='validation'
        with self.assertRaises(ValueError): training.training_pos_weights(ds)
        ds.role='train'; ds.split='test'
        with self.assertRaises(ValueError): training.training_pos_weights(ds)
        ds.split='development'; ds.concepts[:,0]=0
        with self.assertRaises(ValueError): training.training_pos_weights(ds)
        with self.assertRaises(ValueError): training.concept_loss(torch.zeros(2,7),torch.zeros(2,7),torch.ones(1))

    def test_metrics_serialization_and_validation_epoch(self):
        targets=torch.tensor([[0.]*7,[0.]*7,[1.]*7,[1.]*7])
        logits=torch.tensor([[-2.,-2.,-2.,-2.,-2.,-2.,-2.], [0.,-1.,-1.,-1.,-1.,-1.,-1.],[-1.,1.,1.,1.,1.,1.,1.],[2.,2.,2.,2.,2.,2.,2.]])
        batch=ConceptOnly(image=logits,concepts=targets,case_num=['001','002','003','004'],split=['development']*4)
        loss,rows=training.run_epoch(nn.Identity(),BatchLoader(batch,'validation'),torch.device('cpu'),pos_weights=torch.ones(7))
        self.assertAlmostEqual(loss,training.concept_loss(logits,targets,torch.ones(7)).item())
        scores=training.validation_metrics(rows)
        self.assertEqual(scores[CONCEPT_ORDER[0]+'_auroc'],.75)
        self.assertEqual(scores[CONCEPT_ORDER[0]+'_macro_f1'],.5)
        for name in CONCEPT_ORDER[1:]:
            self.assertEqual(scores[name+'_auroc'],1.)
            self.assertEqual(scores[name+'_macro_f1'],1.)
        self.assertAlmostEqual(scores['validation_macro_auroc'],6.75/7)
        self.assertAlmostEqual(scores['validation_macro_f1'],6.5/7)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'predictions.csv'; training.save_validation_predictions(p,rows)
            with p.open(newline='') as f:
                reader=csv.DictReader(f); saved=list(reader)
                self.assertEqual(reader.fieldnames,training.PREDICTION_COLUMNS)
                self.assertEqual(len(reader.fieldnames),22)
            self.assertEqual(saved[0]['case_num'],'001')
            for i,name in enumerate(CONCEPT_ORDER):
                self.assertEqual(float(saved[2][name+'_logit']),logits[2,i].item())
                self.assertEqual(int(saved[2][name+'_target']),1)
                self.assertAlmostEqual(float(saved[2][name+'_probability']),torch.sigmoid(logits)[2,i].item())
            with self.assertRaises(FileExistsError): training.save_validation_predictions(p,rows)
        with self.assertRaises(ValueError): training.validation_metrics(rows[:2])

    def test_fold_isolation_weights_and_gpu_wiring_all_four(self):
        fixture=fixtures.DataLoadingTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        fixture.config_path.write_text((ROOT/'configs/seven_concept.toml').read_text())
        with patch('PIL.Image.open',side_effect=AssertionError('No image access')),patch.object(LoaderFactory,'locked_test',side_effect=AssertionError('No test loader')):
            factory=LoaderFactory(fixture.config_path)
            for fold in range(4):
                loaders=development_fold_loaders(factory,fold)
                train=loaders['train'].dataset
                weight=training.training_pos_weights(train)
                loaders['validation'].dataset.concepts.fill_(1.)
                self.assertTrue(torch.equal(training.training_pos_weights(train),weight))
                self.assertFalse(set(train.case_nums)&set(factory.test_ids))
                with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.get_device_name',return_value='mock GPU'),patch('faithful_medical_cbm.training.train_seven_concept.seed_everything'),patch('faithful_medical_cbm.training.train_seven_concept.SevenConceptEfficientNet') as model,patch('faithful_medical_cbm.training.train_seven_concept.fit_fold',return_value={}) as fit:
                    run_training(fixture.config_path,fold,'mock-run')
                    model.assert_called_once_with(pretrained=True); self.assertEqual(fit.call_count,1)
                    record=json.loads((fixture.root/f'artifacts/seven_concept/mock-run/fold_{fold}/run.json').read_text())
                    self.assertEqual(record['provenance']['pos_weights'],dict(zip(CONCEPT_ORDER,weight.tolist())))
                    self.assertEqual(record['provenance']['concept_order'],list(CONCEPT_ORDER))
                    with self.assertRaises(FileExistsError): run_training(fixture.config_path,fold,'mock-run')
            bad=factory.fold(0); bad['train'].dataset.case_nums+=(factory.test_ids[0],)
            with patch.object(factory,'fold',return_value=bad),self.assertRaises(ValueError): development_fold_loaders(factory,0)

    def test_locked_test_cpu_and_invalid_fold_rejected(self):
        loader=Mock(); loader.dataset.split='test'
        with self.assertRaisesRegex(ValueError,'non-development'):
            training.run_epoch(nn.Identity(),loader,torch.device('cpu'),pos_weights=torch.ones(7))
        with patch('torch.cuda.is_available',return_value=False),patch('faithful_medical_cbm.training.train_seven_concept.LoaderFactory') as factory:
            with self.assertRaisesRegex(RuntimeError,'requires CUDA'): run_training(ROOT/'configs/seven_concept.toml',0,'cpu-refused')
            factory.assert_not_called()
            with self.assertRaises(ValueError): run_training(ROOT/'configs/seven_concept.toml',4,'invalid')

    def test_selection_uses_macro_auc_and_raw_outputs_precede_metrics(self):
        config={**self.config,'concept_training':{**self.config['concept_training'],'epochs':4,'head_epochs':1,'patience':1}}
        batch=ConceptOnly(image=torch.zeros(2,7),concepts=torch.tensor([[0.]*7,[1.]*7]),case_num=['a','b'],split=['development']*2)
        loaders={'train':BatchLoader(batch),'validation':BatchLoader(batch,'validation')}
        rows=training.prediction_rows(['a','b'],batch['concepts'],batch['image'],torch.full((2,7),.5))
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp); seen=[]
            def metrics(records):
                epoch=len(seen)+1
                self.assertTrue((directory/f'validation_epoch_{epoch:03}.csv').is_file())
                seen.append(epoch)
                return {'validation_macro_auroc':.8 if epoch==1 else .7,'validation_macro_f1':.2 if epoch==1 else .9,CONCEPT_ORDER[0]+'_auroc':.5 if epoch==1 else 1.}
            model=Mock(); model.concept_names=CONCEPT_ORDER
            with patch.object(training,'run_epoch',return_value=(.5,rows)),patch.object(training,'build_optimizer',return_value=Mock()) as optimizers,patch.object(training,'save_checkpoint') as checkpoints,patch.object(training,'validation_metrics',side_effect=metrics):
                result=training.fit_fold(model,loaders,torch.device('cpu'),config,directory,directory,{})
            self.assertEqual(result['best_epoch'],1); self.assertTrue(result['early_stopped'])
            self.assertEqual(optimizers.call_count,2)
            self.assertEqual([c.args[0].name for c in checkpoints.call_args_list],['best.pt','last.pt','last.pt'])
            self.assertEqual(len(list(directory.glob('validation_epoch_*.csv'))),2)

if __name__=='__main__': unittest.main()
