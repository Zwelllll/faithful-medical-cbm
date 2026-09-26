"""Explicitly gated frozen final evaluation. Default invocation cannot access test data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def run(root: Path, drive_outputs: Path, attestation_path: Path, *, authorize_locked_test=False, device='cuda'):
    # This must precede imports, config reads, checkpoint reads, loader creation and writes.
    if authorize_locked_test is not True:
        raise PermissionError('Final execution refused: explicit --authorize-locked-test required')
    from .final_verification import (frozen_state, development_folds, validate_attestation,
                                     load_verified_model, sha, write_json)
    from .final_core import prediction_tables, schemas, metrics, concept_metrics, bootstrap_indices, bootstrap_statistics, write_csv, MODEL_ORDER
    from .final_interventions import run_interventions
    import numpy as np
    import torch
    import importlib.metadata
    from ..reproducibility import seed_everything

    root = root.resolve(); drive_outputs = drive_outputs.resolve()
    destination = drive_outputs/'artifacts/final_test'
    if destination.exists():
        raise FileExistsError('Final-test namespace already exists; refusing rerun/overwrite')
    if destination == root or root in destination.parents:
        # Local outputs are supported only via the canonical project artifact path.
        if destination != root/'artifacts/final_test':
            raise ValueError('Invalid nested repository output path')
    for protected in (root/'data', root/'checkpoints', root/'src', root/'configs'):
        if destination == protected or protected in destination.parents:
            raise ValueError('Output overlaps protected repository inputs')
    manifest, source_state = frozen_state(root)
    report = json.loads(attestation_path.read_bytes())
    validate_attestation(report,root,drive_outputs,manifest)
    folds = development_folds(root,manifest)
    seed_everything(seed=42, deterministic=True)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not begin final test')
    heads = {}
    for name, record in manifest['full_development_heads'].items():
        if sha(root/record['path']) != record['sha256']:
            raise ValueError('Frozen LR head changed')
        heads[name] = json.loads((root/record['path']).read_bytes())
    models = {}
    for record in report['records']:
        family,fold = record['family'],record['fold']
        model,_ = load_verified_model(drive_outputs/record['relative_path'],family,fold,record['epoch'],manifest,folds,record['sha256'])
        models[family,fold] = model.to(device).eval()

    # Reserve the run before possible exposure. Failed/partial runs cannot silently rerun.
    destination.mkdir(parents=True,exist_ok=False)
    write_json(destination/'execution_started.json',{'status':'started_not_yet_complete',
        'authorization_explicit':True,'manifest_sha256':source_state['manifest_sha256'],
        'test_access_may_follow':True,'completion_requires':'integrity.json status=passed'})
    # FIRST possible access to test IDs/cohort/targets/images is below this boundary.
    from ..data.loaders import LoaderFactory
    factory = LoaderFactory(root/'configs/default.toml')
    order = manifest['protocol']['concept_order']
    if list(factory.concept_columns) != order:
        raise ValueError('Loader concept order mismatch')
    loader = factory.locked_test(allow_locked_test_iteration=True)
    if len(loader.dataset) != 165 or len(set(factory.test_ids)) != 165 or set(factory.test_ids) & set(factory.development_ids):
        raise ValueError('Frozen test membership mismatch')
    ids, ys, truths = [],[],[]
    collected = {family:{} for family in manifest['checkpoints']}
    with torch.inference_mode():
        for batch in loader:
            ids.extend(batch['case_num']); ys.append(batch['diagnosis'].numpy()); truths.append(batch['concepts'].numpy())
            image = batch['image'].to(device)
            for family in collected:
                per_fold = []
                for fold in range(4):
                    output = models[family,fold](image)
                    if family == 'black_box':
                        values = {'diagnosis_logit':output, 'diagnosis_probability':torch.sigmoid(output)}
                    elif family == 'seven_concept':
                        values = {'concept_logit':output, 'concept_probability':torch.sigmoid(output)}
                    else:
                        values = {'diagnosis_logit':output['diagnosis_logits'], 'diagnosis_probability':torch.sigmoid(output['diagnosis_logits']),
                                  'concept_logit':output['concept_logits'], 'concept_probability':output['concept_probabilities']}
                    per_fold.append({k:v.cpu().numpy() for k,v in values.items()})
                for key in per_fold[0]:
                    collected[family].setdefault(key,[]).append(np.stack([r[key] for r in per_fold],axis=1))
    if len(ids) != 165 or len(set(ids)) != 165 or set(ids) != set(factory.test_ids):
        raise ValueError('Inference coverage mismatch')
    permutation = np.array(sorted(range(165),key=lambda i:ids[i]))
    ids = [ids[i] for i in permutation]
    y,truth = np.concatenate(ys)[permutation],np.concatenate(truths)[permutation]
    if int(y.sum()) != 50 or not np.isin(y,[0,1]).all() or not np.isin(truth,[0,1]).all():
        raise ValueError('Frozen test label/count mismatch')
    collected = {f:{k:np.concatenate(v)[permutation] for k,v in values.items()} for f,values in collected.items()}
    for values in collected.values():
        for key,value in values.items():
            if not np.isfinite(value).all(): raise ValueError('Nonfinite model outputs')
    tables,diagnoses,concepts = prediction_tables(ids,y,truth,collected,heads,order)
    schema = schemas(order)
    for name,rows in tables.items():
        write_csv(destination/name,rows,schema[name])
    # Raw predictions are now durable before any aggregate calculation.
    diagnosis_rows = [{'model':m,**{k:v for k,v in metrics(y,diagnoses[m]).items() if k in schema['diagnosis_metrics.csv']}} for m in MODEL_ORDER]
    write_csv(destination/'diagnosis_metrics.csv',diagnosis_rows,schema['diagnosis_metrics.csv'])
    concept_rows = [{'model':m,**r} for m,p in concepts.items() for r in concept_metrics(truth,p,order)]
    write_csv(destination/'concept_metrics.csv',concept_rows,schema['concept_metrics.csv'])
    indices = bootstrap_indices(ids,y)
    np.save(destination/'bootstrap_indices.npy',indices,allow_pickle=False)
    write_json(destination/'bootstrap_case_order.json',{'case_num':ids})
    intervals,paired = bootstrap_statistics(ids,y,diagnoses,indices)
    write_csv(destination/'bootstrap_intervals.csv',intervals,schema['bootstrap_intervals.csv'])
    write_csv(destination/'paired_bootstrap_differences.csv',paired,schema['paired_bootstrap_differences.csv'])
    run_interventions(destination/'interventions',ids,concepts['sequential_concept_ensemble'],truth,y,heads)
    summary = {'status':'completed','cases':165,'positive':50,'negative':115,'concept_order':order,
               'models':list(MODEL_ORDER),'threshold':.5,'locked_test_accessed':True,'final_test_executed':True,
               'oracle_non_deployable':True,'limitations':manifest['protocol']['limitations']}
    write_json(destination/'summary.json',summary)
    frozen_state(root)  # Frozen source/head metadata must still match after execution.
    validate_attestation(report,root,drive_outputs,manifest)
    write_json(destination/'integrity.json',{'status':'passed',**source_state,'checkpoint_attestation':report,
        'attestation_sha256':sha(attestation_path),'head_records':manifest['full_development_heads'],
        'case_num':ids,'exact_test_coverage':True,'development_overlap':0,'concept_order':order,
        'packages':{p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','torch','torchvision','Pillow')},
        'output_sha256':{str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()},
        'locked_test_accessed':True,'final_test_executed':True})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--drive-outputs',type=Path,required=True)
    parser.add_argument('--attestation',type=Path,required=True)
    parser.add_argument('--authorize-locked-test',action='store_true')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args = parser.parse_args()
    run(args.root,args.drive_outputs,args.attestation,authorize_locked_test=args.authorize_locked_test,device=args.device)


if __name__ == '__main__':
    main()
