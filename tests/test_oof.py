"""Synthetic prediction fixtures on frozen development IDs; no images or inference."""
import csv
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from faithful_medical_cbm.evaluation import assemble_oof as oof


def encode(rows,columns):
    s=io.StringIO(newline=''); writer=csv.DictWriter(s,fieldnames=columns,lineterminator='\n')
    writer.writeheader(); writer.writerows(rows); return s.getvalue().encode()


class OOFTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.run=self.root/'synthetic-run'; self.out=self.root/'oof'
        self.config=ROOT/'configs/seven_concept.toml'
        self.frozen=oof.frozen_inputs(self.config)
        self.original={}
        for fold in range(4):
            directory=self.run/f'fold_{fold}'; directory.mkdir(parents=True)
            ids=[i for i,f in self.frozen['folds'].items() if f==fold]
            rows=[]
            for identifier in ids:
                row={'case_num':identifier}
                for concept in oof.CONCEPT_ORDER:
                    y=self.frozen['truth'][identifier][concept]
                    logit=2. if y else -2.
                    row[concept+'_target']=str(y); row[concept+'_logit']=str(logit)
                    row[concept+'_probability']=str(1/(1+math.exp(-logit)))
                rows.append(row)
            scores=oof.metrics(rows)
            best={'epoch':2,'validation_macro_auroc':scores['macro_auroc'],'validation_macro_f1':scores['macro_f1']}
            best.update({c+'_auroc':m['auroc'] for c,m in scores['concepts'].items()})
            best.update({c+'_macro_f1':m['macro_f1'] for c,m in scores['concepts'].items()})
            summary={'best_epoch':2,'epochs_completed':3,'concept_order':list(oof.CONCEPT_ORDER),'locked_test_used':False,'threshold':.5,'best_epoch_metrics':best,'best_validation_macro_auroc':scores['macro_auroc']}
            run={'config':{'concepts':{'target_order':list(oof.CONCEPT_ORDER)},'concept_training':{'threshold':.5,'selection_metric':'validation_macro_auroc'}},'provenance':{'concept_order':list(oof.CONCEPT_ORDER),'locked_test_images_accessed':False,'validation_fold':fold,'validation_ids':ids,'training_ids':sorted(set(self.frozen['development'])-set(ids)),'cohort_sha256':self.frozen['cohort_sha256'],'split_metadata_sha256':self.frozen['split_metadata_sha256'],'git_commit':'synthetic'}}
            for name,data in [('summary.json',json.dumps(summary).encode()),('run.json',json.dumps(run).encode()),('validation_epoch_002.csv',encode(rows,oof.SOURCE_COLUMNS))]:
                path=directory/name; path.write_bytes(data); self.original[path]=data
        self.csv=self.run/'fold_0/validation_epoch_002.csv'

    def test_exact_coverage_diagnosis_join_schema_metrics_and_sources(self):
        result=oof.assemble(self.config,self.run,self.out)
        rows=oof.read_csv((self.out/'seven_concept_oof.csv').read_bytes(),oof.OOF_COLUMNS)
        self.assertEqual(len(rows),658)
        self.assertEqual([r['case_num'] for r in rows],self.frozen['development'])
        self.assertEqual(len({r['case_num'] for r in rows}),658)
        self.assertFalse({r['case_num'] for r in rows}&self.frozen['test'])
        self.assertEqual(result['rows_per_fold'],{'0':165,'1':165,'2':164,'3':164})
        self.assertEqual(result['diagnosis']['positive'],198)
        self.assertEqual(result['macro_auroc'],1.); self.assertEqual(result['macro_f1'],1.)
        for row in rows:
            self.assertEqual(int(row['diagnosis_binary']),self.frozen['truth'][row['case_num']]['diagnosis_binary'])
            self.assertEqual(int(row['validation_fold']),self.frozen['folds'][row['case_num']])
        integrity=json.loads((self.out/'seven_concept_oof_integrity.json').read_text())
        self.assertEqual(integrity['status'],'passed')
        self.assertFalse(integrity['inference_run'])
        self.assertEqual(result['sources'][0]['best_epoch'],2)
        self.assertEqual(result['sources'][0]['validation_sha256'],oof.sha(self.original[self.csv]))
        with self.assertRaises(FileExistsError): oof.assemble(self.config,self.run,self.out)
        self.assertTrue(all(p.read_bytes()==data for p,data in self.original.items()))

    def test_bad_rows_fail_without_outputs(self):
        mutations={
            'missing':lambda rows:rows.pop(),
            'duplicate':lambda rows:rows.append(rows[0].copy()),
            'test':lambda rows:rows[0].update(case_num=next(iter(self.frozen['test']))),
            'wrong_fold':lambda rows:rows[0].update(case_num=next(i for i,f in self.frozen['folds'].items() if f==1)),
            'bad_binary':lambda rows:rows[0].update(atypical_pigment_network_target='2'),
            'wrong_truth':lambda rows:rows[0].update(atypical_pigment_network_target=str(1-int(rows[0]['atypical_pigment_network_target']))),
            'nan_logit':lambda rows:rows[0].update(atypical_pigment_network_logit='nan'),
            'inf_logit':lambda rows:rows[0].update(atypical_pigment_network_logit='inf'),
            'nan_probability':lambda rows:rows[0].update(atypical_pigment_network_probability='nan'),
            'range_probability':lambda rows:rows[0].update(atypical_pigment_network_probability='1.2'),
            'sigmoid_mismatch':lambda rows:rows[0].update(atypical_pigment_network_probability='.25'),
        }
        for name,mutate in mutations.items():
            with self.subTest(name=name):
                rows=oof.read_csv(self.original[self.csv]); mutate(rows)
                self.csv.write_bytes(encode(rows,oof.SOURCE_COLUMNS))
                with self.assertRaises(ValueError): oof.assemble(self.config,self.run,self.out)
                self.assertFalse(self.out.exists())
        self.csv.write_bytes(self.original[self.csv])

    def test_wrong_order_missing_files_and_bad_provenance(self):
        rows=oof.read_csv(self.original[self.csv])
        self.csv.write_bytes(encode(rows,list(reversed(oof.SOURCE_COLUMNS))))
        with self.assertRaises(ValueError): oof.assemble(self.config,self.run,self.out)
        self.csv.unlink()
        with self.assertRaises(FileNotFoundError): oof.assemble(self.config,self.run,self.out)
        self.csv.write_bytes(self.original[self.csv])
        path=self.run/'fold_0/run.json'
        for key,value in [('validation_fold',1),('training_ids',self.frozen['development']),('concept_order',list(reversed(oof.CONCEPT_ORDER))),('cohort_sha256','bad'),('locked_test_images_accessed',True)]:
            with self.subTest(key=key):
                run=json.loads(self.original[path]); run['provenance'][key]=value; path.write_text(json.dumps(run))
                with self.assertRaises(ValueError): oof.assemble(self.config,self.run,self.out)
                self.assertFalse(self.out.exists())
        path.write_bytes(self.original[path])
        path=self.run/'fold_0/summary.json'
        data=json.loads(path.read_text()); data['best_validation_macro_auroc']=.5; path.write_text(json.dumps(data))
        with self.assertRaises(ValueError): oof.assemble(self.config,self.run,self.out)

    def test_pooled_metric_calculation_and_binary_validation(self):
        rows=[]
        for i,(y,logit) in enumerate(zip([0,0,1,1],[-2,0,-1,2])):
            row={'case_num':str(i)}
            for c in oof.CONCEPT_ORDER:
                row.update({c+'_target':str(y),c+'_logit':str(logit),c+'_probability':str(1/(1+math.exp(-logit)))})
            rows.append(row)
        scores=oof.metrics(rows)
        self.assertEqual(scores['macro_auroc'],.75)
        self.assertEqual(scores['macro_f1'],.5)
        for m in scores['concepts'].values(): self.assertEqual(m['prevalence'],.5)
        with self.assertRaises(ValueError): oof.binary('2')
        with self.assertRaises(ValueError): oof.metrics(rows[:2])

    def test_leading_zero_ids_and_ragged_csv(self):
        rows=oof.read_csv(b'case_num\n0001\n',['case_num'])
        self.assertEqual(rows[0]['case_num'],'0001')
        with self.assertRaises(ValueError): oof.read_csv(b'a,a\n1,2\n')
        with self.assertRaises(ValueError): oof.read_csv(b'a,b\n1\n')

    def test_output_cannot_replace_inputs(self):
        with self.assertRaises(ValueError): oof.assemble(self.config,self.run,self.run)

if __name__=='__main__': unittest.main()
