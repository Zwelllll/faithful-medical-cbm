"""Array-backed full-development head adapter for unchanged Stage 11 policies."""
from __future__ import annotations
import csv
import numpy as np
from ..interventions.engine import InterventionEngine, CONCEPT_ORDER
from ..interventions.policies import POLICIES, trajectory, repetition_rng
from ..interventions.run_policies import budget_metrics, aggregate
from .final_core import probabilities, write_csv, metrics
from .final_verification import write_json


class FullHeadEngine(InterventionEngine):
    """Reuse forced-value/correction operations, without development-file loading."""
    def __init__(self, ids, q, truth, diagnosis, heads):
        if len(set(ids)) != len(ids) or list(ids) != sorted(ids):
            raise ValueError('Unique lexicographic case IDs required')
        self.ids = tuple(ids); self._index = {case:i for i,case in enumerate(ids)}
        self._probabilities = probabilities(q, (len(ids), 7)).copy()
        truth, diagnosis = np.asarray(truth), np.asarray(diagnosis)
        if truth.shape != (len(ids),7) or diagnosis.shape != (len(ids),) or not np.isin(truth,[0,1]).all() or not np.isin(diagnosis,[0,1]).all():
            raise ValueError('Invalid supplied binary targets')
        # -1 explicitly denotes full-development head, not a fabricated fold.
        self._folds = np.full(len(ids), -1)
        self._truth = {case:dict(zip(CONCEPT_ORDER, map(int, truth[i]))) for i,case in enumerate(ids)}
        self._diagnosis = dict(zip(ids,map(int,diagnosis)))
        self._heads = {}
        for mode in ('soft','hard'):
            head = heads['sequential_'+mode]
            self._heads[mode,-1] = (float(head['intercept']), np.asarray(head['coefficients'],dtype=float))


def run_interventions(destination, ids, q, truth, diagnosis, heads):
    """Caller has authorized final execution; supplied arrays also permit synthetic tests."""
    engine = FullHeadEngine(ids,q,truth,diagnosis,heads)
    destination.mkdir(exist_ok=False)
    metrics_rows = []
    for mode in ('soft','hard'):
        for policy in POLICIES:
            repetitions = 100 if policy == 'random_error_oracle' else 1
            path = destination / f'{mode}_{policy}_trajectories.csv'
            with path.open('x',newline='',encoding='utf-8') as stream:
                writer = None
                for rep in range(repetitions):
                    rng = repetition_rng(42,rep)
                    rows = [row for case in engine.ids for row in trajectory(engine,case,mode,policy,rep,rng)]
                    if writer is None:
                        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n'); writer.writeheader()
                    writer.writerows(rows); stream.flush()  # Persist raw steps before metrics.
                    for k in range(8):
                        selected = [r for r in rows if r['query_budget'] == k]
                        values = budget_metrics(selected)
                        # Stage 14 ranks final probabilities, not the LR scores.
                        values.update(metrics([r['diagnosis_binary'] for r in selected], [r['melanoma_probability'] for r in selected]))
                        metrics_rows.append(dict(model_type=mode,policy=policy,oracle_non_deployable=int(policy!='active'),
                                                 repetition=rep,query_budget=k,**values))
    write_csv(destination/'budget_metrics.csv',metrics_rows)
    curves,areas = aggregate(metrics_rows)
    write_csv(destination/'intervention_curves.csv',curves)
    write_json(destination/'auc_summary.json',areas)
    write_json(destination/'definitions.json',{'head':'full-development; validation_fold=-1',
        'query_precision':'errors corrected / actual cumulative queries',
        'improvement':'initially incorrect now correct / initially incorrect',
        'harm':'initially correct now incorrect / initially correct',
        'undefined':'null when denominator zero', 'random_sd':'sample SD ddof=1, 100 repetitions',
        'active_truth_access':'selection view has no truth; reveal selected concept afterward'})
