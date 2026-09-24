"""Stage 11 selection, execution and accounting tests. No fitting."""
import copy
import csv
from dataclasses import fields
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from faithful_medical_cbm.interventions.engine import InterventionEngine, CONCEPT_ORDER
from faithful_medical_cbm.interventions import policies as p
from faithful_medical_cbm.interventions.run_policies import budget_metrics,mean_sd,curve_auc,aggregate,METRICS


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.engine=InterventionEngine(ROOT/'configs/intervention_engine.json')

    def test_oracle_error_only_confidence_ties_and_seed(self):
        q=[.9,.1,.8,.6,.5,0,1]; y=[0,1,0,1,0,0,1]
        order=p.oracle_order(q,y,'confidently_wrong_oracle')
        self.assertEqual(order,[CONCEPT_ORDER[j] for j in [0,1,2,4]])
        for rep in range(100):
            a=p.oracle_order(q,y,'random_error_oracle',p.repetition_rng(42,rep))
            b=p.oracle_order(q,y,'random_error_oracle',p.repetition_rng(42,rep))
            self.assertEqual(a,b); self.assertEqual(set(a),set(order))
        self.assertGreater(len({tuple(p.oracle_order(q,y,'random_error_oracle',p.repetition_rng(42,r))) for r in range(20)}),1)
        with self.assertRaises(ValueError): p.oracle_order(q,y,'active')

    def test_active_formula_tie_break_and_exclusion(self):
        q=(0,.5,1,.25,.75,.1,.9)
        view=p.ActiveView(CONCEPT_ORDER,q,CONCEPT_ORDER,(.1,)*7,(.9,)*7)
        selected,d=p.select_active(view)
        self.assertEqual(selected,CONCEPT_ORDER[1]); self.assertEqual(d['uncertainty'],1)
        self.assertAlmostEqual(d['downstream_impact'],.8); self.assertAlmostEqual(d['active_score'],.8)
        tied=p.ActiveView(CONCEPT_ORDER,(.5,)*7,CONCEPT_ORDER[::-1],(.2,)*7,(.8,)*7)
        self.assertEqual(p.select_active(tied)[0],CONCEPT_ORDER[0])
        remaining=tuple(c for c in CONCEPT_ORDER if c!=CONCEPT_ORDER[1])
        selected,d=p.select_active(p.ActiveView(CONCEPT_ORDER,q,remaining,(.1,)*7,(.9,)*7))
        self.assertEqual(selected,CONCEPT_ORDER[3]); self.assertEqual(d['uncertainty'],.5)
        for j in [0,2]:
            _,d=p.select_active(p.ActiveView(CONCEPT_ORDER,q,(CONCEPT_ORDER[j],),(.1,)*7,(.9,)*7))
            self.assertEqual(d['active_score'],0)

    def test_active_truth_unavailable_until_selected(self):
        e=copy.copy(self.engine); case=e.ids[0]; truths=e._truth[case]; pending=[]; reads=[]
        class Guard(dict):
            def __getitem__(self,key):
                self_test.assertEqual(pending,[key]); pending.clear(); reads.append(key)
                return truths[key]
        self_test=self; e._truth={case:Guard()}
        real=p.select_active
        def select(view):
            self.assertEqual({f.name for f in fields(view)},{'concept_order','original_probabilities','remaining','force_zero','force_one'})
            self.assertFalse(hasattr(view,'__dict__'))
            selected,detail=real(view); pending.append(selected); return selected,detail
        with patch.object(p,'select_active',side_effect=select): rows=p.trajectory(e,case,'soft','active')
        self.assertEqual(reads,[r['selected_concept'] for r in rows[1:]])
        self.assertEqual(len(set(reads)),7)

    def test_current_state_forced_effects_and_distinct_execution(self):
        e=self.engine; case=e.ids[0]
        for mode in ('soft','hard'):
            session=p.PolicySession(e,case,mode)
            selected=CONCEPT_ORDER[2]; session.reveal_and_correct(selected)
            view=session.view()
            for j,c in enumerate(CONCEPT_ORDER):
                if c==selected: continue
                for v in (0,1):
                    expected=e.intervene(case,mode,session.selected+[c],session.values+[v])['intervened_melanoma_probability']
                    self.assertEqual((view.force_zero,view.force_one)[v][j],expected)
            with self.assertRaises(ValueError): session.reveal_and_correct(selected)
            active=p.trajectory(e,case,mode,'active')
            self.assertEqual(active[-1]['concepts_queried'],7)
            expected=e.correct(case,mode,CONCEPT_ORDER)
            self.assertEqual(active[-1]['melanoma_probability'],expected['intervened_melanoma_probability'])
        self.assertNotEqual(e.intervene(case,'soft',[],[])['original_concept_inputs'],e.intervene(case,'hard',[],[])['original_concept_inputs'])

    def test_oracle_padding_and_compact_replay(self):
        e=self.engine
        for mode in ('soft','hard'):
            for case in e.ids[:20]:
                rows=p.trajectory(e,case,mode,'random_error_oracle',3,p.repetition_rng(42,3))
                order=[r['selected_concept'] for r in rows if r['selected_concept']]
                self.assertTrue(all(r['selected_was_wrong']==1 for r in rows if r['selected_concept']))
                self.assertEqual(rows,p.trajectory(e,case,mode,'random_error_oracle',3,replay_order=order))
                self.assertEqual(len(set(order)),len(order))
                for r in rows[len(order)+1:]:
                    self.assertIsNone(r['selected_concept']); self.assertEqual(r['melanoma_probability'],rows[-1]['melanoma_probability'])

    def test_all_case_k0_metrics_match_saved(self):
        e=self.engine
        for mode,folder in [('soft','sequential_soft'),('hard','sequential_hard')]:
            rows=[p.trajectory(e,case,mode,'confidently_wrong_oracle')[0] for case in e.ids]
            actual=budget_metrics(rows)
            frozen=json.loads((ROOT/'artifacts/cbm'/folder/'pooled_metrics.json').read_bytes())
            for metric in METRICS[:7]: self.assertAlmostEqual(actual[metric],frozen[metric],places=12)
            self.assertIsNone(actual['query_precision']); self.assertEqual(actual['diagnosis_harm_rate'],0)

    def test_improvement_harm_query_denominators(self):
        # Two originally wrong diagnoses: one improved. Two originally right: one harmed.
        rows=[]
        for i,(y,prob,before,queries,errors) in enumerate([(0,.1,0,2,1),(1,.1,0,1,1),(0,.9,1,2,0),(1,.9,1,1,0)]):
            correct=int((prob>=.5)==y)
            rows.append(dict(case_num=str(i),query_budget=2,diagnosis_binary=y,diagnosis_score=np.log(prob/(1-prob)),
                melanoma_probability=prob,concepts_queried=queries,errors_corrected=errors,
                original_diagnosis_correct=before,diagnosis_correct=correct,diagnosis_changed=int(before!=correct)))
        result=budget_metrics(rows)
        self.assertEqual(result['diagnosis_improvement_rate'],.5); self.assertEqual(result['diagnosis_harm_rate'],.5)
        self.assertEqual(result['query_precision'],2/6); self.assertEqual(result['mean_errors_corrected'],.5)
        self.assertEqual(result['diagnosis_change_rate'],.5); self.assertEqual(result['accuracy'],.5)

    def test_aggregation_sample_sd_and_trapezoids(self):
        self.assertAlmostEqual(mean_sd([1,2,3])['sd'],1)
        self.assertEqual(mean_sd([None,None]),{'mean':None,'sd':None})
        self.assertEqual(curve_auc([dict(query_budget=k,auroc=.5) for k in range(8)],'auroc'),3.5)
        self.assertEqual(curve_auc([dict(query_budget=k,auroc=k/7) for k in range(8)],'auroc'),3.5)
        rows=[]
        for mode in ('soft','hard'):
            for policy in p.POLICIES:
                for rep in range(2 if policy=='random_error_oracle' else 1):
                    for k in range(8): rows.append(dict(model_type=mode,policy=policy,repetition=rep,query_budget=k,**{m:.5+rep*.1 for m in METRICS}))
        curves,auc=aggregate(rows)
        self.assertEqual(len(curves),48)
        random=auc['models']['soft']['random_error_oracle']['auroc']
        self.assertAlmostEqual(random['mean'],3.85); self.assertAlmostEqual(random['sd'],.7/np.sqrt(2))

    def test_saved_random_orders_coverage_errors_and_replay(self):
        folder=ROOT/'artifacts/interventions/policies'
        if not (folder/'integrity.json').exists(): self.skipTest('Run artifacts not generated yet')
        e=self.engine
        for mode in ('soft','hard'):
            with (folder/f'{mode}_random_error_orders.csv').open() as stream: rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),65800)
            seen=set()
            for r in rows:
                key=(r['repetition'],r['case_num']); self.assertNotIn(key,seen); seen.add(key)
                order=[CONCEPT_ORDER[j] for j in json.loads(r['concept_indices_in_query_order'])]
                case=r['case_num']; q=e._probabilities[e._case(case)]
                wrong={c for c,value in zip(CONCEPT_ORDER,q) if (value>=.5)!=e._truth[case][c]}
                self.assertEqual(set(order),wrong); self.assertEqual(len(order),len(set(order)))
                self.assertEqual(r['oracle_non_deployable'],'1')
            for rep in range(100):
                self.assertEqual({case for repetition,case in seen if repetition==str(rep)},set(e.ids))
            for rep in (0,99):
                rng=p.repetition_rng(42,rep)
                for r in [r for r in rows if r['repetition']==str(rep)]:
                    order=[CONCEPT_ORDER[j] for j in json.loads(r['concept_indices_in_query_order'])]
                    case=r['case_num']; q=e._probabilities[e._case(case)]
                    self.assertEqual(order,p.oracle_order(q,[e._truth[case][c] for c in CONCEPT_ORDER],'random_error_oracle',rng))
                    if case in e.ids[:3]:
                        replay=p.trajectory(e,case,mode,'random_error_oracle',rep,replay_order=order)
                        self.assertEqual(len(replay),8)

    def test_saved_active_trajectory_replay_and_curve_aggregation(self):
        folder=ROOT/'artifacts/interventions/policies'
        if not (folder/'integrity.json').exists(): self.skipTest('Run artifacts not generated yet')
        for mode in ('soft','hard'):
            with (folder/f'{mode}_active_trajectories.csv').open() as stream: rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),5264)
            for i,case in enumerate(self.engine.ids):
                expected=p.trajectory(self.engine,case,mode,'active')
                actual=rows[i*8:(i+1)*8]
                for a,b in zip(actual,expected):
                    for key in b:
                        if b[key] is None: self.assertEqual(a[key],'')
                        elif isinstance(b[key],(float,np.floating)): self.assertAlmostEqual(float(a[key]),b[key],places=14)
                        else: self.assertEqual(a[key],str(b[key]))
        with (folder/'budget_metrics.csv').open() as stream: raw=list(csv.DictReader(stream))
        metrics=[dict(model_type=r['model_type'],policy=r['policy'],repetition=int(r['repetition']),query_budget=int(r['query_budget']),
                      **{m:float(r[m]) if r[m] else None for m in METRICS}) for r in raw]
        self.assertEqual(len(metrics),1632)
        curves,auc=aggregate(metrics)
        saved=json.loads((folder/'auc_summary.json').read_bytes())
        self.assertEqual(auc,saved)
        with (folder/'intervention_curves.csv').open() as stream: stored=list(csv.DictReader(stream))
        for a,b in zip(stored,curves):
            for k,v in b.items():
                if isinstance(v,float): self.assertEqual(float(a[k]),v)
                else: self.assertEqual(a[k],'' if v is None else str(v))


if __name__=='__main__': unittest.main()
