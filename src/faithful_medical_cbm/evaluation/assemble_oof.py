"""Assemble saved best-epoch development predictions; no image/model dependencies."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import importlib.metadata
from sklearn.metrics import roc_auc_score, f1_score
from ..config import load_config

CONCEPT_ORDER = (
    'atypical_pigment_network', 'regression_structures_present', 'irregular_pigmentation',
    'blue_whitish_veil_present', 'atypical_vascular_structures',
    'irregular_dots_and_globules', 'irregular_streaks',
)
SOURCE_COLUMNS = ['case_num'] + [f'{c}_{kind}' for kind in ('target','logit','probability') for c in CONCEPT_ORDER]
OOF_COLUMNS = ['case_num','validation_fold','diagnosis_binary'] + SOURCE_COLUMNS[1:]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(data: bytes, columns=None) -> list[dict]:
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')), strict=True)
    header = reader.fieldnames or []
    if not header or len(header) != len(set(header)) or (columns is not None and header != columns):
        raise ValueError('Missing, duplicate or incorrectly ordered CSV columns')
    rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError('Malformed CSV row')
    return rows


def unique_ids(ids, label: str) -> set[str]:
    if any(not isinstance(i,str) or not i or i != i.strip() for i in ids) or len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate or invalid case_num in {label}')
    return set(ids)


def binary(value: str) -> int:
    if value not in ('0','1'):
        raise ValueError('Invalid binary label')
    return int(value)


def frozen_inputs(config_path: Path) -> dict:
    config = load_config(config_path)
    if tuple(config['concepts']['target_order']) != CONCEPT_ORDER:
        raise ValueError('Configured concept order differs')
    split = config['paths']['splits']
    hashes = {}
    def read(path, expected=None):
        data = path.read_bytes()
        hashes[str(path)] = sha(data)
        if expected is not None and sha(data) != expected:
            raise ValueError(f'Frozen hash mismatch: {path}')
        return data
    metadata_path = split/'split_metadata.json'
    metadata_bytes = read(metadata_path)
    meta = json.loads(metadata_bytes)
    if meta['status'] != 'frozen' or meta['settings']['num_folds'] != 4 or meta['counts']['development']['total'] != 658:
        raise ValueError('Expected frozen four-fold 658-case development pool')
    stage2a = json.loads(read(config['paths']['artifacts']/config['splitting']['cohort_summary'],meta['inputs']['stage2a_summary_sha256']))
    if tuple(stage2a['concept_target_order']) != CONCEPT_ORDER:
        raise ValueError('Frozen concept order differs')
    development = [r['case_num'] for r in read_csv(read(split/'development_ids.csv',meta['file_sha256']['development_ids.csv']),['case_num'])]
    test = [r['case_num'] for r in read_csv(read(split/'test_ids.csv',meta['file_sha256']['test_ids.csv']),['case_num'])]
    dev_set, test_set = unique_ids(development,'development'), unique_ids(test,'test exclusion manifest')
    if len(dev_set)!=658 or len(test_set)!=165 or dev_set & test_set:
        raise ValueError('Development/test split coverage or overlap mismatch')
    fold_rows = read_csv(read(split/'development_folds.csv',meta['file_sha256']['development_folds.csv']),['case_num','validation_fold'])
    if unique_ids([r['case_num'] for r in fold_rows],'folds') != dev_set or any(r['validation_fold'] not in ('0','1','2','3') for r in fold_rows):
        raise ValueError('Invalid frozen folds')
    folds = {r['case_num']:int(r['validation_fold']) for r in fold_rows}
    if [sum(f==i for f in folds.values()) for i in range(4)] != [165,165,164,164]:
        raise ValueError('Frozen fold counts differ')
    cohort_path = (config['paths']['processed']/config['splitting']['cohort']).resolve()
    if config['paths']['processed'] not in cohort_path.parents:
        raise ValueError('Cohort must be processed metadata')
    cohort_bytes=read(cohort_path,meta['inputs']['cohort_sha256'])
    if sha(cohort_bytes)!=stage2a['output_sha256']['cohort.csv']:
        raise ValueError('Stage 2A cohort mismatch')
    cohort_rows=read_csv(cohort_bytes)
    if unique_ids([r['case_num'] for r in cohort_rows],'cohort') != dev_set|test_set:
        raise ValueError('Cohort IDs differ from frozen split')
    # Never inspect test labels, paths, or raw categorical annotations.
    truth={r['case_num']:{c:binary(r[c]) for c in ('diagnosis_binary',*CONCEPT_ORDER)}
           for r in cohort_rows if r['case_num'] in dev_set}
    if sum(r['diagnosis_binary'] for r in truth.values()) != 198:
        raise ValueError('Development diagnosis counts differ')
    return dict(config=config,development=development,test=test_set,folds=folds,truth=truth,
                hashes=hashes,cohort_sha256=sha(cohort_bytes),split_metadata_sha256=sha(metadata_bytes))


def metrics(rows: list[dict]) -> dict:
    concepts={}
    for c in CONCEPT_ORDER:
        labels=[int(r[c+'_target']) for r in rows]
        if set(labels)!={0,1}:
            raise ValueError(f'Both concept classes required: {c}')
        concepts[c]={'positive':sum(labels),'negative':len(labels)-sum(labels),
                     'prevalence':sum(labels)/len(labels),
                     'auroc':float(roc_auc_score(labels,[float(r[c+'_logit']) for r in rows])),
                     'macro_f1':float(f1_score(labels,[int(float(r[c+'_probability'])>=.5) for r in rows],
                                              labels=[0,1],average='macro',zero_division=0))}
    return {'concepts':concepts,'macro_auroc':sum(c['auroc'] for c in concepts.values())/7,
            'macro_f1':sum(c['macro_f1'] for c in concepts.values())/7}


def validate_rows(rows: list[dict], fold: int, frozen: dict) -> list[dict]:
    expected={i for i,f in frozen['folds'].items() if f==fold}
    ids=unique_ids([r['case_num'] for r in rows],f'fold {fold} predictions')
    if ids & frozen['test']:
        raise ValueError('Locked-test ID in predictions')
    if ids!=expected:
        raise ValueError(f'Fold {fold} predictions do not exactly match held-out IDs')
    joined=[]
    for row in rows:
        identifier=row['case_num']; truth=frozen['truth'][identifier]
        for c in CONCEPT_ORDER:
            if binary(row[c+'_target'])!=truth[c]:
                raise ValueError(f'Concept ground truth mismatch: {identifier}/{c}')
            logit=float(row[c+'_logit']); probability=float(row[c+'_probability'])
            if not math.isfinite(logit) or not math.isfinite(probability) or not 0<=probability<=1:
                raise ValueError('Nonfinite logit or invalid probability')
            expected_p=1/(1+math.exp(-logit)) if logit>=0 else math.exp(logit)/(1+math.exp(logit))
            if not math.isclose(probability,expected_p,abs_tol=1e-6,rel_tol=1e-6):
                raise ValueError('Probability does not match saved logit')
        joined.append({'case_num':identifier,'validation_fold':fold,
                       'diagnosis_binary':truth['diagnosis_binary'],**{c:row[c] for c in SOURCE_COLUMNS[1:]}})
    return joined


def assemble(config_path: Path, run_dir: Path, output_dir: Path | None = None) -> dict:
    frozen=frozen_inputs(config_path)
    run_dir=run_dir.resolve()
    output_dir=(output_dir or frozen['config']['paths']['artifacts']/'oof').resolve()
    for protected in [run_dir,*(frozen['config']['paths'][k] for k in ('raw','processed','splits'))]:
        if output_dir==protected or protected in output_dir.parents or output_dir in protected.parents:
            raise ValueError('OOF outputs overlap protected inputs')
    names=['seven_concept_oof.csv','seven_concept_oof_summary.json','seven_concept_oof_integrity.json']
    if any((output_dir/n).exists() for n in names):
        raise FileExistsError('OOF outputs already exist; refusing overwrite')
    all_rows=[]; sources=[]; source_hashes={}
    for fold in range(4):
        directory=run_dir/f'fold_{fold}'
        def read_source(name):
            path=directory/name; data=path.read_bytes(); source_hashes[str(path)]=sha(data)
            return data
        summary=json.loads(read_source('summary.json'))
        run=json.loads(read_source('run.json')); provenance=run['provenance']
        epoch=summary['best_epoch']
        if type(epoch) is not int or epoch<1 or epoch>summary['epochs_completed']:
            raise ValueError('Invalid best epoch')
        if tuple(summary['concept_order'])!=CONCEPT_ORDER or tuple(provenance['concept_order'])!=CONCEPT_ORDER or tuple(run['config']['concepts']['target_order'])!=CONCEPT_ORDER:
            raise ValueError('Source concept order mismatch')
        if summary['locked_test_used'] is not False or provenance['locked_test_images_accessed'] is not False:
            raise ValueError('Source does not attest locked-test exclusion')
        if summary['threshold']!=.5 or run['config']['concept_training']['threshold']!=.5 or run['config']['concept_training']['selection_metric']!='validation_macro_auroc':
            raise ValueError('Source threshold/selection protocol differs')
        if type(provenance['validation_fold']) is not int or provenance['validation_fold']!=fold:
            raise ValueError('Source fold provenance mismatch')
        expected={i for i,f in frozen['folds'].items() if f==fold}
        if unique_ids(provenance['validation_ids'],'run validation IDs')!=expected or unique_ids(provenance['training_ids'],'run training IDs')!=set(frozen['development'])-expected:
            raise ValueError('Source training/validation provenance differs from frozen fold')
        for key in ('cohort_sha256','split_metadata_sha256'):
            if provenance[key]!=frozen[key]: raise ValueError(f'Source frozen input mismatch: {key}')
        filename=f'validation_epoch_{epoch:03}.csv'
        data=read_source(filename)
        joined=validate_rows(read_csv(data,SOURCE_COLUMNS),fold,frozen)
        scores=metrics(joined)
        best=summary['best_epoch_metrics']
        if best['epoch']!=epoch:
            raise ValueError('Best epoch metadata mismatch')
        comparisons={'validation_macro_auroc':scores['macro_auroc'],'validation_macro_f1':scores['macro_f1']}
        comparisons.update({f'{c}_auroc':s['auroc'] for c,s in scores['concepts'].items()})
        comparisons.update({f'{c}_macro_f1':s['macro_f1'] for c,s in scores['concepts'].items()})
        if not math.isclose(summary['best_validation_macro_auroc'],scores['macro_auroc'],abs_tol=1e-8):
            raise ValueError('Best summary AUROC differs from saved predictions')
        if any(not math.isclose(float(best[k]),v,rel_tol=1e-8,abs_tol=1e-8) for k,v in comparisons.items()):
            raise ValueError('Best-epoch metrics differ from saved predictions')
        sources.append({'fold':fold,'best_epoch':epoch,'validation_file':str(directory/filename),
                        'validation_sha256':sha(data),'summary_sha256':source_hashes[str(directory/'summary.json')],
                        'run_sha256':source_hashes[str(directory/'run.json')],'run_name':provenance.get('run_name',run_dir.name),
                        'git_commit':provenance.get('git_commit'),'checkpoint_reference':f'fold_{fold}/best.pt',
                        'checkpoint_loaded':False})
        all_rows.extend(joined)
    if len(all_rows)!=658 or unique_ids([r['case_num'] for r in all_rows],'OOF')!=set(frozen['development']):
        raise ValueError('OOF must cover all 658 development cases exactly once')
    by_id={r['case_num']:r for r in all_rows}
    all_rows=[by_id[i] for i in frozen['development']]
    scores=metrics(all_rows)
    positive=sum(r['diagnosis_binary'] for r in all_rows)
    summary={'stage':7,'total_rows':len(all_rows),'rows_per_fold':{str(i):sum(r['validation_fold']==i for r in all_rows) for i in range(4)},
             'diagnosis':{'positive':positive,'negative':len(all_rows)-positive,'prevalence':positive/len(all_rows)},
             'concept_order':list(CONCEPT_ORDER),'threshold':.5,**scores,'sources':sources,
             'metric_note':'Pooled OOF metrics, not the arithmetic average of fold metrics; AUROC uses logits.',
             'python':platform.python_version(),'scikit_learn':importlib.metadata.version('scikit-learn')}
    text=io.StringIO(newline=''); writer=csv.DictWriter(text,fieldnames=OOF_COLUMNS,lineterminator='\n')
    writer.writeheader(); writer.writerows(all_rows); csv_bytes=text.getvalue().encode('utf-8')
    integrity={'status':'passed','total_rows':658,'unique_case_num':658,'duplicates':0,'missing_development_ids':0,
               'locked_test_overlap':0,'exact_fold_membership':True,'training_ids_excluded_per_fold':True,
               'concept_order_preserved':True,'concept_truth_matches_frozen_cohort':True,'diagnosis_join_verified':True,
               'finite_logits':True,'valid_probabilities':True,'sigmoid_consistency':True,'best_epoch_metrics_verified':True,
               'images_opened':False,'inference_run':False,'threshold_optimized':False,
               'checkpoint_provenance_limit':'Uses saved run IDs and summary best epoch; checkpoint weights not loaded.',
               'schema':OOF_COLUMNS,'input_sha256':{**frozen['hashes'],**source_hashes},'oof_sha256':sha(csv_bytes)}
    # Check source stability before committing any outputs.
    for name,expected in integrity['input_sha256'].items():
        if sha(Path(name).read_bytes())!=expected: raise ValueError('Input changed during assembly')
    payloads=[csv_bytes,(json.dumps(summary,indent=2,allow_nan=False)+'\n').encode(),(json.dumps(integrity,indent=2,allow_nan=False)+'\n').encode()]
    output_dir.mkdir(parents=True,exist_ok=True)
    for name,data in zip(names,payloads):
        with (output_dir/name).open('xb') as stream: stream.write(data)
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/seven_concept.toml'))
    parser.add_argument('--run-dir',type=Path,required=True,help='Parent containing fold_0 through fold_3')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    result=assemble(args.config,args.run_dir,args.output_dir)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
