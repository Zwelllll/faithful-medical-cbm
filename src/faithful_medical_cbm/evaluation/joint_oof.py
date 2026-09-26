"""Stage 13 saved joint OOF assembly. No torch, images, checkpoints or fitting."""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
import platform
import subprocess
import importlib.metadata
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score, f1_score
from .assemble_oof import CONCEPT_ORDER, OOF_COLUMNS, binary, frozen_inputs, read_csv, sha, unique_ids
from .diagnosis_metrics import diagnosis_metrics, expected_calibration_error

DIAGNOSIS_METRICS=('auroc','macro_f1','accuracy','sensitivity','specificity','brier','ece')
BEST_EPOCHS={'joint_soft':[13,15,16,14],'joint_hard_ste':[6,18,14,16]}


def columns(model: str, *, assembled: bool=False) -> list[str]:
    if model not in BEST_EPOCHS: raise ValueError('Unknown joint model')
    names=['case_num']+(['validation_fold'] if assembled else [])+['diagnosis_binary','diagnosis_logit','diagnosis_probability']
    names += [f'{c}_{kind}' for kind in ('target','logit','probability') for c in CONCEPT_ORDER]
    if model=='joint_hard_ste': names += [f'{c}_hard_input' for c in CONCEPT_ORDER]
    return names


def validate_rows(rows: list[dict], model: str, fold: int, frozen: dict, stage7_truth: dict) -> list[dict]:
    ids=unique_ids([r['case_num'] for r in rows],'joint validation')
    expected={i for i,f in frozen['folds'].items() if f==fold}
    if ids & frozen['test']: raise ValueError('Locked-test overlap')
    if ids!=expected: raise ValueError('Missing or wrong-fold validation cases')
    result=[]
    for raw in rows:
        case=raw['case_num']; row=dict(raw,validation_fold=fold)
        row['diagnosis_binary']=binary(raw['diagnosis_binary'])
        if row['diagnosis_binary']!=frozen['truth'][case]['diagnosis_binary']: raise ValueError('Diagnosis truth mismatch')
        for prefix in ('diagnosis',*CONCEPT_ORDER):
            z,p=float(raw[prefix+'_logit']),float(raw[prefix+'_probability'])
            if not np.isfinite([z,p]).all() or not 0<=p<=1: raise ValueError('Invalid logit/probability')
            if not np.isclose(p,expit(z),atol=1e-6,rtol=1e-6): raise ValueError('Sigmoid inconsistency')
            row[prefix+'_logit'],row[prefix+'_probability']=z,p
            if prefix!='diagnosis':
                target=binary(raw[prefix+'_target']); row[prefix+'_target']=target
                if target!=frozen['truth'][case][prefix] or target!=binary(stage7_truth[case][prefix+'_target']):
                    raise ValueError('Concept truth mismatch')
                if model=='joint_hard_ste':
                    hard=binary(raw[prefix+'_hard_input']); row[prefix+'_hard_input']=hard
                    if hard!=int(p>=.5): raise ValueError('Hard input threshold mismatch')
        result.append(row)
    return result


def validate_run(run: dict, summary: dict, model: str, fold: int, expected_epoch: int, frozen: dict) -> None:
    prov=run['provenance']; settings=run['config']['joint_training']
    if summary['best_epoch']!=expected_epoch or summary['best_epoch_metrics']['epoch']!=expected_epoch:
        raise ValueError('Mismatched best epoch')
    if any(x!=model for x in (prov['model_type'],settings['model_type'],summary['model_type'])):
        raise ValueError('Model type mismatch')
    if prov['validation_fold']!=fold: raise ValueError('Run fold mismatch')
    for order in (prov['concept_order'],summary['concept_order'],run['config']['concepts']['target_order']):
        if tuple(order)!=CONCEPT_ORDER: raise ValueError('Frozen concept order mismatch')
    if prov['locked_test_used'] is not False or summary['locked_test_used'] is not False or prov['locked_test_images_accessed'] is not False:
        raise ValueError('Locked-test flag is not false')
    val=unique_ids(prov['validation_ids'],'run validation'); train=unique_ids(prov['training_ids'],'run training')
    expected={i for i,f in frozen['folds'].items() if f==fold}
    if val!=expected or train!=set(frozen['development'])-expected or train & val:
        raise ValueError('Training/validation isolation mismatch')
    if prov['cohort_sha256']!=frozen['cohort_sha256'] or prov['split_metadata_sha256']!=frozen['split_metadata_sha256']:
        raise ValueError('Frozen input provenance mismatch')
    if settings['threshold']!=.5 or settings['selection_metric']!='validation_diagnosis_auroc':
        raise ValueError('Selection/threshold mismatch')


def concept_metrics(rows: list[dict], *, use_logits: bool=False) -> dict:
    results={}
    for c in CONCEPT_ORDER:
        y=[r[c+'_target'] for r in rows]; p=[r[c+'_probability'] for r in rows]
        if set(y)!={0,1}: raise ValueError('Both concept classes required')
        results[c]={'auroc':float(roc_auc_score(y,[r[c+'_logit'] for r in rows] if use_logits else p)),
                    'macro_f1':float(f1_score(y,[int(v>=.5) for v in p],average='macro',labels=[0,1],zero_division=0))}
    return {'concepts':results,'macro_auroc':float(np.mean([r['auroc'] for r in results.values()])),
            'macro_f1':float(np.mean([r['macro_f1'] for r in results.values()])),
            'auroc_input':'logits' if use_logits else 'saved soft probabilities'}


def evaluate(rows: list[dict]) -> dict:
    return diagnosis_metrics([r['diagnosis_binary'] for r in rows],[r['diagnosis_logit'] for r in rows],
                             [r['diagnosis_probability'] for r in rows],bins=10)


def diagnostics(rows: list[dict]) -> dict:
    y=np.array([r['diagnosis_binary'] for r in rows]); p=np.array([r['diagnosis_probability'] for r in rows])
    return {'mean_probability':float(p.mean()),'mean_probability_melanoma':float(p[y==1].mean()),
            'mean_probability_benign':float(p[y==0].mean()),'fraction_positive':float((p>=.5).mean()),
            'fraction_melanoma_positive':float((p[y==1]>=.5).mean()),'fraction_benign_positive':float((p[y==0]>=.5).mean()),
            'minimum_probability':float(p.min()),'maximum_probability':float(p.max())}


def write_csv(path: Path, rows: list[dict], names=None) -> None:
    with path.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=names or list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    with path.open('xb') as stream: stream.write((json.dumps(value,indent=2,allow_nan=False)+'\n').encode())


def assemble(config_path: Path, output_dir: Path | None=None) -> dict:
    config_path=config_path.resolve(); root=config_path.parent.parent; hashes={}
    def read(path, expected=None):
        path=path.resolve(); content=path.read_bytes(); actual=sha(content)
        if expected is not None and expected!=actual: raise ValueError(f'Source hash mismatch: {path}')
        hashes[str(path)]=actual; return content
    cfg=json.loads(read(config_path))
    if cfg['stage']!=13 or set(cfg['models'])!=set(BEST_EPOCHS): raise ValueError('Stage 13 models required')
    if any(cfg['models'][m]['best_epochs']!=e for m,e in BEST_EPOCHS.items()): raise ValueError('Frozen best epochs differ')
    frozen=frozen_inputs(root/cfg['frozen_config']); hashes.update(frozen['hashes'])
    read(root/cfg['frozen_config'])
    stage7=root/cfg['stage7_dir']; integrity7=json.loads(read(stage7/'seven_concept_oof_integrity.json'))
    if integrity7['status']!='passed' or integrity7['schema']!=OOF_COLUMNS: raise ValueError('Stage 7 integrity/schema failed')
    truth_rows=read_csv(read(stage7/'seven_concept_oof.csv',integrity7['oof_sha256']),OOF_COLUMNS)
    if unique_ids([r['case_num'] for r in truth_rows],'Stage 7')!=set(frozen['development']): raise ValueError('Stage 7 coverage differs')
    truth={r['case_num']:r for r in truth_rows}
    concepts7=json.loads(read(stage7/'seven_concept_oof_summary.json'))
    if tuple(concepts7['concept_order'])!=CONCEPT_ORDER: raise ValueError('Stage 7 order mismatch')
    inventory=json.loads(read(root/cfg['organization_inventory']))
    destination=(output_dir or root/cfg['output_dir']).resolve()
    protected=[root/'artifacts/oof',root/'artifacts/cbm',root/'artifacts/interventions',
               *(root/v['run_dir'] for v in cfg['models'].values()),
               *(frozen['config']['paths'][k] for k in ('raw','processed','splits'))]
    for p in protected:
        p=p.resolve()
        if destination==p or p in destination.parents or destination in p.parents: raise ValueError('Output overlaps protected inputs')
    if destination.exists() and any(p.name!='.gitattributes' for p in destination.iterdir()): raise FileExistsError('Refusing OOF overwrite')
    pooled_rows={}; fold_results={}; sources=[]
    for model,settings in cfg['models'].items():
        run_dir=root/settings['run_dir']; kind='soft' if model=='joint_soft' else 'hard'
        reference=inventory['runs'][kind]
        if (root/reference['path']).resolve()!=run_dir.resolve(): raise ValueError('Run path differs from verified inventory')
        all_rows=[]; fold_results[model]=[]
        for fold,epoch in enumerate(settings['best_epochs']):
            d=run_dir/f'fold_{fold}'
            def read_source(name): return read(d/name,reference['files'][f'fold_{fold}/{name}']['sha256'])
            run=json.loads(read_source('run.json')); summary=json.loads(read_source('summary.json'))
            validate_run(run,summary,model,fold,epoch,frozen)
            history=read_csv(read_source('history.csv'))
            selected=[r for r in history if r['epoch']==str(epoch)]
            if len(selected)!=1: raise ValueError('Best epoch missing/duplicate in history')
            name=f'validation_epoch_{epoch:03}.csv'
            rows=validate_rows(read_csv(read_source(name),columns(model)),model,fold,frozen,truth)
            diagnosis=evaluate(rows); concepts=concept_metrics(rows); saved_concepts=concept_metrics(rows,use_logits=True)
            for metric in DIAGNOSIS_METRICS:
                expected=summary['best_epoch_metrics']['validation_diagnosis_'+metric]
                if not np.isclose(diagnosis[metric],expected,atol=1e-12,rtol=0) or not np.isclose(float(selected[0]['validation_diagnosis_'+metric]),expected,atol=1e-12,rtol=0):
                    raise ValueError('Best-epoch diagnosis metrics mismatch')
            for c in CONCEPT_ORDER:
                for metric in ('auroc','macro_f1'):
                    key='validation_concept_'+c+'_'+metric
                    if not np.isclose(saved_concepts['concepts'][c][metric],summary['best_epoch_metrics'][key],atol=1e-12,rtol=0):
                        raise ValueError('Best-epoch concept metrics mismatch')
            fold_results[model].append({'fold':fold,'best_epoch':epoch,'diagnosis':diagnosis,'concepts':concepts,
                'saved_validation_concept_macro_auroc':summary['best_epoch_metrics']['validation_concept_macro_auroc'],
                'saved_validation_concept_macro_f1':summary['best_epoch_metrics']['validation_concept_macro_f1']})
            sources.append({'model':model,'fold':fold,'best_epoch':epoch,'run_name':run_dir.name,
                'prediction_file':str((d/name).relative_to(root)),'prediction_sha256':hashes[str((d/name).resolve())],
                'run_sha256':hashes[str((d/'run.json').resolve())],'summary_sha256':hashes[str((d/'summary.json').resolve())],
                'source_git_commit':run['provenance'].get('git_commit')})
            all_rows.extend(rows)
        if len(all_rows)!=658 or unique_ids([r['case_num'] for r in all_rows],'assembled OOF')!=set(frozen['development']): raise ValueError('Joint OOF coverage differs')
        lookup={r['case_num']:r for r in all_rows}; pooled_rows[model]=[lookup[i] for i in frozen['development']]
    sequential={}
    for model,folder in cfg['sequential_dirs'].items():
        d=root/folder; source=json.loads(read(d/'integrity.json'))
        if source['status']!='passed': raise ValueError('Sequential integrity failed')
        predictions=read_csv(read(d/'cross_fitted_predictions.csv',source['output_sha256']['cross_fitted_predictions.csv']))
        if len(predictions)!=658 or unique_ids([r['case_num'] for r in predictions],'sequential predictions')!=set(frozen['development']): raise ValueError('Sequential coverage differs')
        for r in predictions:
            if int(r['validation_fold'])!=frozen['folds'][r['case_num']] or binary(r['diagnosis_binary'])!=frozen['truth'][r['case_num']]['diagnosis_binary']: raise ValueError('Sequential fold/label mismatch')
        sequential[model]=json.loads(read(d/'pooled_metrics.json',source['output_sha256']['pooled_metrics.json']))
    destination.mkdir(parents=True,exist_ok=True)
    metrics={}; calibration=[]; comparison=[]; concept_table=[]
    for model,rows in pooled_rows.items():
        stem='joint_soft' if model=='joint_soft' else 'joint_hard'
        write_csv(destination/f'{stem}_oof.csv',rows,columns(model,assembled=True))  # Raw OOF precedes pooled metrics.
        metrics[model]={'diagnosis':evaluate(rows),'concepts':concept_metrics(rows),'folds':fold_results[model],
                       'fold_means':{'diagnosis':{m:float(np.mean([r['diagnosis'][m] for r in fold_results[model]])) for m in DIAGNOSIS_METRICS},
                            'concept_macro_auroc':float(np.mean([r['saved_validation_concept_macro_auroc'] for r in fold_results[model]])),
                            'concept_macro_f1':float(np.mean([r['saved_validation_concept_macro_f1'] for r in fold_results[model]]))},
                       'probability_diagnostics':diagnostics(rows)}
        bins=expected_calibration_error([r['diagnosis_binary'] for r in rows],[r['diagnosis_probability'] for r in rows])
        metrics[model]['calibration']=bins
        write_json(destination/f'{stem}_metrics.json',metrics[model])
        calibration.append({'model':model,**metrics[model]['probability_diagnostics']})
    for model in ('sequential_soft','sequential_hard','joint_soft','joint_hard_ste','oracle'):
        values=metrics[model]['diagnosis'] if model in metrics else sequential[model]
        comparison.append({'model':model,**{m:values[m] for m in DIAGNOSIS_METRICS}})
    for model,values in [('sequential_concepts',concepts7),*((m,r['concepts']) for m,r in metrics.items())]:
        for c in CONCEPT_ORDER: concept_table.append({'model':model,'concept':c,**{k:values['concepts'][c][k] for k in ('auroc','macro_f1')}})
        concept_table.append({'model':model,'concept':'macro_mean','auroc':values['macro_auroc'],'macro_f1':values['macro_f1']})
    write_csv(destination/'development_model_comparison.csv',comparison)
    write_csv(destination/'concept_model_comparison.csv',concept_table)
    write_csv(destination/'joint_concept_metrics.csv',[r for r in concept_table if r['model']!='sequential_concepts'])
    write_csv(destination/'calibration_diagnostics.csv',calibration)
    bins_rows=[{'model':m,'bin':j,**r} for m,v in metrics.items() for j,r in enumerate(v['calibration']['bins'])]
    write_csv(destination/'calibration_bins.csv',bins_rows)
    lookup={r['model']:r for r in comparison}
    gaps={f'{a}_minus_{b}':{m:lookup[a][m]-lookup[b][m] for m in ('auroc','macro_f1')} for a,b in (
        ('joint_soft','sequential_soft'),('joint_hard_ste','sequential_hard'),('oracle','joint_soft'),('oracle','joint_hard_ste'))}
    summary={'stage':13,'total_rows_per_model':658,'rows_per_fold':[165,165,164,164],'concept_order':list(CONCEPT_ORDER),
        'sources':sources,'diagnosis_comparison':comparison,'diagnosis_gaps':gaps,'concept_comparison':concept_table,
        'fold_means':{m:v['fold_means'] for m,v in metrics.items()},'calibration_diagnostics':calibration,
        'sequential_metrics_recomputed':False,'threshold':.5,'ece_bins':10,
        'interpretation':'Pooled development OOF, not means of folds or locked-test results. AUROC is threshold-independent; F1/accuracy/sensitivity/specificity use fixed 0.5. No tuning or recalibration.',
        'limitations':'Best epoch used held-out validation diagnosis AUROC; these are selected development results, not unbiased test estimates. Sequential LR evaluation is not fully nested CNN/LR CV. Patient independence is unverified. Fold-specific logit scales can change pooled ranking relative to fold means.'}
    write_json(destination/'summary.json',summary)
    for name,h in hashes.items():
        if sha(Path(name).read_bytes())!=h: raise ValueError('Source changed during assembly')
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=False)
    write_json(destination/'integrity.json',{'status':'passed','input_sha256':hashes,
        'output_sha256':{p.name:sha(p.read_bytes()) for p in destination.iterdir() if p.suffix in ('.csv','.json')},
        'source_sha256':{str(p.relative_to(root)):sha(p.read_bytes()) for p in (Path(__file__).resolve(),Path(__file__).with_name('assemble_oof.py').resolve(),Path(__file__).with_name('diagnosis_metrics.py').resolve())},
        'git_commit':git.stdout.strip(),'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn')},
        'models':{m:{'rows':658,'unique_case_num':658,'rows_per_fold':[165,165,164,164],'locked_test_overlap':0,
                    'exact_development_coverage':True,'training_ids_exclude_validation':True,'truth_matches_stage7_and_processed':True,
                    'hard_binary_and_threshold_verified':m=='joint_hard_ste','schema':columns(m,assembled=True)} for m in metrics},
        'checkpoints_opened':False,'images_opened':False,'inference_run':False,'models_trained':False,
        'threshold_optimized':False,'recalibrated':False,'sequential_values_unchanged':True})
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',type=Path,default=Path('configs/joint_oof.json'))
    args=parser.parse_args(); result=assemble(args.config)
    print(json.dumps(result['diagnosis_comparison'],indent=2))


if __name__=='__main__': main()
