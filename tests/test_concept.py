"""Stage 5 offline/synthetic tests; no downloads or real image access."""
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from faithful_medical_cbm.config import load_config
from faithful_medical_cbm.models.concept import ConceptEfficientNet
from faithful_medical_cbm.training import concept
from faithful_medical_cbm.training.baseline import development_fold_loaders
from faithful_medical_cbm.training.train_concept import run_training
from faithful_medical_cbm.data.loaders import LoaderFactory
import test_data_loading as fixtures


class ConceptOnly(dict):
    def __getitem__(self, key):
        if key in ('diagnosis', 'clinic', 'sex', 'location'):
            raise AssertionError('No diagnosis or patient metadata')
        return super().__getitem__(key)


class BatchLoader:
    def __init__(self, batch, role='train'):
        self.batch = batch
        self.dataset = SimpleNamespace(split='development', role=role,
            case_nums=batch['case_num'], concepts=batch['concepts'], concept_columns=fixtures.CONCEPTS)
    def __iter__(self):
        yield self.batch


class ConceptTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / 'configs/concept_sanity.toml')
        self.addCleanup(patch.stopall)
        patch('torch.hub.download_url_to_file', side_effect=AssertionError('Offline tests')).start()

    def test_output_optimizer_checkpoint_and_frozen_blocks(self):
        old = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, old)
        model = ConceptEfficientNet(pretrained=False)
        self.assertEqual(model.concept_names, (concept.TARGET,))
        with torch.no_grad():
            self.assertEqual(model(torch.randn(2, 3, 224, 224)).shape, (2,))
        model.set_trainable_blocks(3)
        self.assertFalse(model.network.features[0].training)
        self.assertTrue(model.network.features[-1].training)
        batch = ConceptOnly(image=torch.randn(2, 3, 64, 64), concepts=torch.tensor([[0.]*7, [1.]*7]),
                            case_num=['a', 'b'], split=['development']*2)
        optimizer = concept.build_optimizer(model, self.config['concept_training'], .001)
        before = model.network.classifier[1].weight.detach().clone()
        loss, _ = concept.run_epoch(model, BatchLoader(batch), torch.device('cpu'), optimizer,
                                    target_index=0, pos_weight=2.)
        self.assertTrue(math.isfinite(loss))
        self.assertFalse(torch.equal(before, model.network.classifier[1].weight))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'concept.pt'
            concept.save_checkpoint(p, model, optimizer, epoch=1, config=self.config,
                provenance={'pos_weight': 2., 'validation_fold': 0}, metrics={'train_loss': loss},
                stopper=concept.EarlyStopping(7, 0.))
            restored = ConceptEfficientNet(pretrained=False)
            restored.set_trainable_blocks(3)
            restored_optimizer = concept.build_optimizer(restored, self.config['concept_training'], .001)
            result = concept.load_checkpoint(p, restored, restored_optimizer)
            self.assertEqual(result['model_config']['concept_names'], [concept.TARGET])
            self.assertTrue(all(torch.equal(v, restored.state_dict()[k]) for k,v in model.state_dict().items()))
            self.assertEqual(len(optimizer.state), len(restored_optimizer.state))
            payload = torch.load(p, weights_only=True)
            payload['model_config']['concept_names'] = ['wrong_concept']
            torch.save(payload, p)
            with self.assertRaises(ValueError):
                concept.load_checkpoint(p, restored)

    def test_target_order_selection_and_scope(self):
        index = concept.checked_target_index(fixtures.CONCEPTS, self.config)
        self.assertEqual(index, 0)
        values = torch.tensor([[1.,0.,0.,0.,0.,0.,0.], [0.,1.,1.,1.,1.,1.,1.]])
        self.assertEqual(concept.select_target(values, index).tolist(), [1.,0.])
        with self.assertRaises(ValueError):
            concept.checked_target_index(tuple(reversed(fixtures.CONCEPTS)), self.config)
        with self.assertRaises(ValueError):
            concept.select_target(torch.zeros(2,6), 0)
        for key,value in [('target','irregular_streaks'),('threshold',.4)]:
            with self.assertRaises(ValueError):
                concept.validate_settings({**self.config['concept_training'],key:value})

    def test_weight_uses_only_training_labels(self):
        ds = SimpleNamespace(split='development', role='train', concepts=torch.tensor([[1.]*7]+[[0.]*7]*3))
        self.assertEqual(concept.training_pos_weight(ds,0), 3.)
        ds.role='validation'
        with self.assertRaises(ValueError): concept.training_pos_weight(ds,0)
        ds.role='train'; ds.split='test'
        with self.assertRaises(ValueError): concept.training_pos_weight(ds,0)
        ds.split='development'; ds.concepts=torch.zeros(4,7)
        with self.assertRaises(ValueError): concept.training_pos_weight(ds,0)
        self.assertAlmostEqual(concept.concept_loss(torch.zeros(2),torch.tensor([0.,1.]),3.).item(),2*math.log(2),places=6)

    def test_metrics_fixed_threshold_and_concept_only_validation(self):
        logits = torch.tensor([-2.,0.,-1.,2.])
        targets = torch.tensor([0.,0.,1.,1.])
        batch=ConceptOnly(image=logits,concepts=targets[:,None].repeat(1,7),case_num=['a','b','c','d'],split=['development']*4)
        loss,rows=concept.run_epoch(nn.Identity(),BatchLoader(batch,'validation'),torch.device('cpu'),target_index=0,pos_weight=2.)
        self.assertAlmostEqual(loss,concept.concept_loss(logits,targets,2.).item())
        scores=concept.validation_metrics(rows)
        self.assertEqual(scores['validation_auroc'],.75)
        self.assertEqual(scores['validation_macro_f1'],.5) # p==.5 is positive
        with self.assertRaises(ValueError): concept.validation_metrics(rows[:2])

    def test_locked_loader_and_cpu_and_other_fold_refused(self):
        loader=Mock(); loader.dataset.split='test'
        with self.assertRaisesRegex(ValueError,'non-development'):
            concept.run_epoch(nn.Identity(),loader,torch.device('cpu'),target_index=0,pos_weight=1.)
        with patch('torch.cuda.is_available',return_value=False), patch('faithful_medical_cbm.training.train_concept.LoaderFactory') as factory:
            with self.assertRaisesRegex(RuntimeError,'requires CUDA'):
                run_training(ROOT/'configs/concept_sanity.toml',0,'cpu-refused')
            factory.assert_not_called()
            with self.assertRaisesRegex(ValueError,'Fold 0'):
                run_training(ROOT/'configs/concept_sanity.toml',1,'wrong-fold')

    def test_fold_isolation_weight_invariance_and_gpu_wiring(self):
        fixture=fixtures.DataLoadingTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        fixture.config_path.write_text((ROOT/'configs/concept_sanity.toml').read_text())
        with patch('PIL.Image.open',side_effect=AssertionError('No image access')), patch.object(LoaderFactory,'locked_test',side_effect=AssertionError('No test loader')):
            factory=LoaderFactory(fixture.config_path)
            loaders=development_fold_loaders(factory,0)
            train=loaders['train'].dataset
            weight=concept.training_pos_weight(train,0)
            loaders['validation'].dataset.concepts.fill_(1.)
            self.assertEqual(concept.training_pos_weight(train,0),weight)
            self.assertFalse(set(train.case_nums)&set(factory.test_ids))
            bad=factory.fold(0); bad['train'].dataset.case_nums+=(factory.test_ids[0],)
            with patch.object(factory,'fold',return_value=bad), self.assertRaises(ValueError):
                development_fold_loaders(factory,0)
            with patch('torch.cuda.is_available',return_value=True), patch('torch.cuda.get_device_name',return_value='mock GPU'), patch('faithful_medical_cbm.training.train_concept.seed_everything'), patch('faithful_medical_cbm.training.train_concept.ConceptEfficientNet') as model, patch('faithful_medical_cbm.training.train_concept.fit_fold',return_value={}) as fit:
                run_training(fixture.config_path,0,'mock-run')
                model.assert_called_once_with(pretrained=True)
                self.assertEqual(fit.call_count,1)
                record=json.loads((fixture.root/'artifacts/concept_sanity/mock-run/fold_0/run.json').read_text())
                self.assertEqual(record['provenance']['pos_weight'],weight)
                self.assertEqual(record['provenance']['concept_order'],list(fixtures.CONCEPTS))
                with self.assertRaises(FileExistsError): run_training(fixture.config_path,0,'mock-run')

    def test_mocked_schedule_predictions_saved_before_metrics(self):
        config={**self.config,'concept_training':{**self.config['concept_training'],'epochs':4,'head_epochs':1,'patience':1}}
        batch=ConceptOnly(image=torch.zeros(2),concepts=torch.tensor([[0.]*7,[1.]*7]),case_num=['a','b'],split=['development']*2)
        loaders={'train':BatchLoader(batch),'validation':BatchLoader(batch,'validation')}
        rows=[{'case_num':str(i),'concept':concept.TARGET,'target':i,'logit':float(i),'probability':.4+.2*i} for i in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp); scores=iter([.8,.7])
            def metrics(records):
                self.assertTrue(list(directory.glob('validation_epoch_*.csv')))
                return {'validation_auroc':next(scores),'validation_macro_f1':1.}
            with patch.object(concept,'run_epoch',return_value=(.5,rows)), patch.object(concept,'build_optimizer',return_value=Mock()) as optimizers, patch.object(concept,'save_checkpoint') as checkpoints, patch.object(concept,'validation_metrics',side_effect=metrics):
                result=concept.fit_fold(Mock(),loaders,torch.device('cpu'),config,directory,directory,{})
            self.assertTrue(result['early_stopped'])
            self.assertEqual(result['epochs_completed'],2)
            self.assertEqual(optimizers.call_count,2)
            self.assertEqual([c.args[0].name for c in checkpoints.call_args_list],['best.pt','last.pt','last.pt'])
            self.assertEqual(len(list(directory.glob('validation_epoch_*.csv'))),2)

if __name__=='__main__': unittest.main()
