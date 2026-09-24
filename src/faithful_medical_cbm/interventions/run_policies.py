"""Stage 11 policy experiments on frozen development heads; no model fitting."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import numpy as np
from .engine import InterventionEngine, CONCEPT_ORDER
from .policies import POLICIES, trajectory, repetition_rng
from ..evaluation.assemble_oof import sha, unique_ids
from ..evaluation.diagnosis_metrics import diagnosis_metrics
from ..training.train_binary_cbm import verify_hashes, write_json
from ..training.train_sequential_soft import csv_bytes

METRICS = ('auroc','macro_f1','accuracy','sensitivity','specificity','brier','ece',
           'query_precision','mean_errors_corrected','mean_concepts_queried','diagnosis_change_rate',
           'diagnosis_improvement_rate','diagnosis_harm_rate')


def budget_metrics(rows: list[dict]) -> dict:
    unique_ids([r['case_num'] for r in rows],'budget cases')
    if len({r['query_budget'] for r in rows})!=1: raise ValueError('Mixed budgets')
    result=diagnosis_metrics([r['diagnosis_binary'] for r in rows],[r['diagnosis_score'] for r in rows],
                             [r['melanoma_probability'] for r in rows])
    queries=sum(r['concepts_queried'] for r in rows)
    errors=sum(r['errors_corrected'] for r in rows)
    originally_wrong=[r for r in rows if not r['original_diagnosis_correct']]
    originally_right=[r for r in rows if r['original_diagnosis_correct']]
    improved=sum(r['diagnosis_correct'] for r in originally_wrong)
    harmed=sum(not r['diagnosis_correct'] for r in originally_right)
    result.update(query_precision=errors/queries if queries else None,
                  mean_errors_corrected=errors/len(rows),mean_concepts_queried=queries/len(rows),
                  diagnosis_change_rate=sum(r['diagnosis_changed'] for r in rows)/len(rows),
                  diagnosis_improvement_rate=improved/len(originally_wrong) if originally_wrong else None,
                  diagnosis_harm_rate=harmed/len(originally_right) if originally_right else None,
                  queries=queries,errors_corrected=errors,originally_wrong=len(originally_wrong),
                  originally_right=len(originally_right),diagnoses_improved=improved,diagnoses_harmed=harmed)
    return result


def mean_sd(values: list) -> dict:
    if all(v is None for v in values): return {'mean':None,'sd':None}
    if any(v is None for v in values): raise ValueError('Partly undefined metric')
    return {'mean':float(np.mean(values)),'sd':float(np.std(values,ddof=1)) if len(values)>1 else 0.0}


def curve_auc(rows: list[dict], metric: str) -> float:
    rows=sorted(rows,key=lambda r:r['query_budget'])
    if [r['query_budget'] for r in rows]!=list(range(8)):
        raise ValueError('Exactly budgets 0..7 required')
    y=np.array([r[metric] for r in rows],dtype=float)
    return float(np.sum((y[:-1]+y[1:])/2))  # Unit-width trapezoids, unnormalized range 0..7.


def aggregate(metrics: list[dict]) -> tuple[list[dict],dict]:
    curves=[]; auc={}
    for mode in ('soft','hard'):
        auc[mode]={}
        for policy in POLICIES:
            selected=[r for r in metrics if r['model_type']==mode and r['policy']==policy]
            repetitions=sorted({r['repetition'] for r in selected})
            for k in range(8):
                subset=[r for r in selected if r['query_budget']==k]
                record=dict(model_type=mode,policy=policy,oracle_non_deployable=int(policy!='active'),query_budget=k,repetitions=len(repetitions))
                for m in METRICS:
                    record.update({f'{m}_{stat}':v for stat,v in mean_sd([r[m] for r in subset]).items()})
                curves.append(record)
            auc[mode][policy]={}
            for m in ('auroc','macro_f1','accuracy'):
                per_rep=[curve_auc([r for r in selected if r['repetition']==rep],m) for rep in repetitions]
                auc[mode][policy][m]={**mean_sd(per_rep),'per_repetition':per_rep}
    return curves,{'definition':'Unnormalized trapezoidal area over budgets 0..7; descriptive only, not causal effects or deployment utility',
                  'models':auc}


def run(config_path: Path) -> dict:
    config_path=config_path.resolve(); root=config_path.parent.parent
    cfg=json.loads(config_path.read_bytes())
    expected={'stage':11,'seed':42,'random_repetitions':100,'budgets':list(range(8)),
              'uncertainty':'1 - 2 * abs(q - 0.5)','impact':'abs(p_force_1_current - p_force_0_current)',
              'tie_break':'frozen concept order','sd_ddof':1}
    if any(cfg[k]!=v for k,v in expected.items()): raise ValueError('Frozen Stage 11 protocol differs')
    engine=InterventionEngine(root/cfg['engine_config'])
    source=root/cfg['engine_artifacts']
    previous=json.loads((source/'integrity.json').read_bytes())
    if previous['status']!='passed': raise ValueError('Stage 10 integrity failed')
    hashes={**engine.hashes,str(config_path):sha(config_path.read_bytes()),
            str(source/'integrity.json'):sha((source/'integrity.json').read_bytes()),
            **{str(source/n):h for n,h in previous['output_sha256'].items()}}
    verify_hashes(hashes)
    destination=(root/cfg['output_dir']).resolve()
    for p in [*engine.protected_paths,source]:
        p=p.resolve()
        if destination==p or p in destination.parents or destination in p.parents: raise ValueError('Protected output path')
    if destination.exists() and any(p.name!='.gitattributes' for p in destination.iterdir()):
        raise FileExistsError('Policy outputs exist; refusing overwrite')
    destination.mkdir(parents=True,exist_ok=True)
    metrics=[]; counts={}; baseline_errors={}
    for mode in ('soft','hard'):
        counts[mode]={}
        saved_folder='sequential_soft' if mode=='soft' else 'sequential_hard'
        baseline=json.loads((root/'artifacts/cbm'/saved_folder/'pooled_metrics.json').read_bytes())
        for policy in POLICIES:
            is_random=policy=='random_error_oracle'
            reps=cfg['random_repetitions'] if is_random else 1
            name=f'{mode}_random_error_orders.csv' if is_random else f'{mode}_{"active" if policy=="active" else "confidently_wrong"}_trajectories.csv'
            with (destination/name).open('x',newline='',encoding='utf-8') as stream:
                writer=None; written=0
                for rep in range(reps):
                    rng=repetition_rng(cfg['seed'],rep)
                    rows=[]
                    for case in engine.ids:
                        records=trajectory(engine,case,mode,policy,rep,rng)
                        rows.extend(records)
                        if is_random:
                            order=[CONCEPT_ORDER.index(r['selected_concept']) for r in records if r['selected_concept'] is not None]
                            output=[dict(case_num=case,model_type=mode,policy=policy,oracle_non_deployable=1,repetition=rep,
                                         concept_indices_in_query_order=json.dumps(order,separators=(',',':')))]
                        else: output=records
                        if writer is None:
                            writer=csv.DictWriter(stream,fieldnames=list(output[0]),lineterminator='\n'); writer.writeheader()
                        writer.writerows(output); written+=len(output)
                    stream.flush()  # Raw trajectories/orders precede their aggregate metrics.
                    for k in range(8):
                        subset=[r for r in rows if r['query_budget']==k]
                        if len(subset)!=658 or {r['case_num'] for r in subset}!=set(engine.ids): raise ValueError('Development coverage differs')
                        values=budget_metrics(subset)
                        if k==0:
                            error=max(abs(values[m]-baseline[m]) for m in METRICS[:7])
                            if error>1e-12: raise ValueError('k=0 metrics differ from frozen baseline')
                            baseline_errors[f'{mode}/{policy}']=error
                        metrics.append(dict(model_type=mode,policy=policy,oracle_non_deployable=int(policy!='active'),
                                            repetition=rep,query_budget=k,**values))
                    if is_random and (rep+1)%25==0: print(f'{mode}: {rep+1}/100 random repetitions complete',flush=True)
                counts[mode][policy]={'saved_rows':written,'represented_trajectory_rows':658*8*reps,'repetitions':reps}
    with (destination/'budget_metrics.csv').open('xb') as stream: stream.write(csv_bytes(metrics,list(metrics[0])))
    curves,auc=aggregate(metrics)
    with (destination/'intervention_curves.csv').open('xb') as stream: stream.write(csv_bytes(curves,list(curves[0])))
    write_json(destination/'auc_summary.json',auc)
    summary={'stage':11,'config':cfg,'development_cases':658,'concept_order':list(CONCEPT_ORDER),'counts':counts,
             'random_storage':'One exact selection order per case/repetition; replay with trajectory(..., replay_order=...). Frozen inputs reconstruct every step without RNG.',
             'random_streams':'PCG64 SeedSequence([42,repetition]); independent repetition streams, cases in frozen ID order. Same stream per model pairs random query orders across soft/hard.',
             'oracle_semantics':'Ground-truth-aware/non-deployable error-selection references; not guaranteed optimal bounds on diagnosis performance.',
             'query_precision_definition':'Cumulative true errors corrected / actual queries; undefined at zero queries',
             'improvement_definition':'Originally wrong diagnoses now correct / originally wrong diagnoses',
             'harm_definition':'Originally correct diagnoses now wrong / originally correct diagnoses',
             'sd_definition':'Sample SD across repetitions (ddof=1), not sampling uncertainty across patients; deterministic policy SD=0',
             'k0_metric_max_errors':baseline_errors,'baseline_reproduction':engine.reproduction,
             'endpoints':[r for r in curves if r['query_budget'] in (0,7)],
             'limitations':'Development diagnostics inherit non-nested CNN/LR evaluation, unverified patient independence and simulated perfect human corrections. Oracle policies stop after errors; active spends all queries and soft also snaps correct concepts to 0/1. No winner/ranking or causal claims.'}
    write_json(destination/'summary.json',summary)
    verify_hashes(hashes)
    package=Path(__file__).resolve().parents[1]
    source_files=[Path(__file__).resolve(),Path(__file__).with_name('policies.py'),Path(__file__).with_name('engine.py'),package/'evaluation/diagnosis_metrics.py']
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=False)
    write_json(destination/'integrity.json',{'status':'passed','input_sha256':hashes,
        'output_sha256':{p.name:sha(p.read_bytes()) for p in destination.iterdir() if p.suffix in ('.csv','.json')},
        'source_sha256':{str(p.relative_to(root)):sha(p.read_bytes()) for p in source_files},
        'git_commit':git.stdout.strip(),'python':platform.python_version(),'numpy':np.__version__,
        'development_cases':658,'locked_test_overlap':0,'fold_specific_heads_only':True,'full_development_heads_used':False,
        'baseline_reproduction':engine.reproduction,'k0_metric_max_errors':baseline_errors,
        'active_selector_receives_ground_truth':False,'oracle_policies_explicitly_labelled':True,
        'no_repeated_queries':True,'tie_break':'frozen concept order','models_retrained':False,'images_accessed':False,
        'thresholds_optimized':False,'source_artifacts_unchanged':True,'seed':42,'random_repetitions':100})
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/intervention_policies.json'))
    args=parser.parse_args(); result=run(args.config)
    print(json.dumps(result['counts'],indent=2))


if __name__=='__main__': main()
