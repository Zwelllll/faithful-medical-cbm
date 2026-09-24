"""Joint CBM losses, validation, checkpoints and development-only training."""
from __future__ import annotations
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import torch
from .baseline import EarlyStopping, build_optimizer, binary_loss, validate_settings as validate_schedule
from .seven_concept import (check_order, extract_targets, training_pos_weights, prevalence, concept_loss,
                            prediction_rows as concept_rows, validation_metrics as concept_metrics)
from ..models.joint_cbm import MODEL_TYPES, CONCEPT_ORDER
from ..evaluation.diagnosis_metrics import diagnosis_metrics


def validate_config(config: dict) -> None:
    settings=config['joint_training']; validate_schedule(settings)
    fixed={'concept_loss_weight':1.,'diagnosis_loss_weight':1.,'threshold':.5,'selection_metric':'validation_diagnosis_auroc',
           'optimizer':'adamw','learning_rate':.001,'finetune_learning_rate':.0001,'weight_decay':.0001,
           'epochs':30,'head_epochs':3,'unfreeze_last_blocks':3,'patience':7,'min_delta':0.}
    if settings['model_type'] not in MODEL_TYPES or any(settings[k]!=v for k,v in fixed.items()):
        raise ValueError('Frozen joint objective/schedule/selection differs')
    if config['reproducibility']['seed']!=42 or config['experiment']['image_size']!=224 or config['experiment']['batch_size']!=32 or config['experiment']['num_folds']!=4 or config['experiment']['model_name']!='efficientnet_b0':
        raise ValueError('Frozen seed, size, batch, folds or backbone differs')
    check_order(config['concepts']['target_order'])


def joint_loss(outputs: dict, concepts: torch.Tensor, diagnosis: torch.Tensor, weights: torch.Tensor) -> dict:
    c=concept_loss(outputs['concept_logits'],concepts,weights)
    d=binary_loss(outputs['diagnosis_logits'],diagnosis)
    return {'concept_loss':c,'diagnosis_loss':d,'total_loss':c+d}


def prediction_columns(model_type: str) -> list[str]:
    if model_type not in MODEL_TYPES: raise ValueError('Unknown joint model')
    return ['case_num','diagnosis_binary','diagnosis_logit','diagnosis_probability']+[
        f'{c}_{kind}' for kind in ('target','logit','probability') for c in CONCEPT_ORDER]+(
        [f'{c}_hard_input' for c in CONCEPT_ORDER] if model_type=='joint_hard_ste' else [])


def prediction_rows(ids, targets, diagnosis, outputs, model_type: str) -> list[dict]:
    rows=concept_rows(ids,targets,outputs['concept_logits'],outputs['concept_probabilities'])
    for i,row in enumerate(rows):
        row.update(diagnosis_binary=int(diagnosis[i].item()),diagnosis_logit=float(outputs['diagnosis_logits'][i].item()),
                   diagnosis_probability=float(torch.sigmoid(outputs['diagnosis_logits'][i]).item()))
        if model_type=='joint_hard_ste':
            values=outputs['diagnosis_inputs'][i]
            if not torch.all((values==0)|(values==1)): raise ValueError('Hard forward inputs must be binary')
            row.update({f'{c}_hard_input':int(values[j].item()) for j,c in enumerate(CONCEPT_ORDER)})
    return rows


def save_validation_predictions(path: Path, rows: list[dict], model_type: str) -> None:
    with path.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=prediction_columns(model_type),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def validation_metrics(rows: list[dict]) -> dict:
    diagnosis=diagnosis_metrics([r['diagnosis_binary'] for r in rows],[r['diagnosis_logit'] for r in rows],
                                [r['diagnosis_probability'] for r in rows],bins=10)
    concepts=concept_metrics(rows)
    return {**{f'validation_diagnosis_{k}':v for k,v in diagnosis.items()},
            **{f'validation_concept_{k.removeprefix("validation_")}':v for k,v in concepts.items()}}


def run_epoch(model, loader, device, optimizer=None, *, pos_weights) -> tuple[dict,list[dict]]:
    training=optimizer is not None
    if loader.dataset.split!='development' or loader.dataset.role!=('train' if training else 'validation'):
        raise ValueError('Refusing non-development or wrong-role loader before iteration')
    check_order(loader.dataset.concept_columns)
    expected=set(loader.dataset.case_nums)
    if len(expected)!=len(loader.dataset.case_nums): raise ValueError('Duplicate dataset IDs')
    model.train(training); seen=set(); rows=[]
    sums={k:0. for k in ('concept_loss','diagnosis_loss','total_loss')}
    with torch.set_grad_enabled(training):
        for batch in loader:
            ids=list(batch['case_num'])
            if set(ids)-expected or seen&set(ids) or len(ids)!=len(set(ids)) or any(s!='development' for s in batch['split']):
                raise ValueError('Unexpected development batch IDs')
            seen.update(ids)
            targets=extract_targets(batch['concepts'],loader.dataset.concept_columns).to(device,dtype=torch.float32)
            diagnosis=batch['diagnosis'].to(device,dtype=torch.float32)
            if training: optimizer.zero_grad(set_to_none=True)
            outputs=model(batch['image'].to(device))
            losses=joint_loss(outputs,targets,diagnosis,pos_weights)
            if training:
                losses['total_loss'].backward(); optimizer.step()
            for k,v in losses.items(): sums[k]+=v.item()*len(ids)
            if not training: rows.extend(prediction_rows(ids,targets,diagnosis,outputs,model.model_type))
    if not seen or seen!=expected: raise ValueError('Epoch must cover every selected development case once')
    return {k:v/len(seen) for k,v in sums.items()},rows


def save_checkpoint(path, model, optimizer, *, epoch, config, provenance, metrics, stopper):
    payload={'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'epoch':epoch,
             'config':json.loads(json.dumps(config,default=str)),'provenance':provenance,'metrics':metrics,
             'early_stopping':asdict(stopper),'model_config':{'architecture':'efficientnet_b0_joint_cbm',
                 'model_type':model.model_type,'concept_order':list(CONCEPT_ORDER),'diagnosis_inputs':7,
                 'pretrained':model.pretrained,'unfreeze_last_blocks':model.unfreeze_last_blocks,
                 'phase':'head_only' if model.unfreeze_last_blocks==0 else 'finetune'},
             'loss_weights':{'concept':1.,'diagnosis':1.},'locked_test_used':False}
    temporary=path.with_suffix('.pt.tmp'); torch.save(payload,temporary); os.replace(temporary,path)


def load_checkpoint(path, model, optimizer=None, *, map_location='cpu'):
    payload=torch.load(path,map_location=map_location,weights_only=True)
    cfg=payload['model_config']
    if cfg['architecture']!='efficientnet_b0_joint_cbm' or cfg['model_type']!=model.model_type or cfg['concept_order']!=list(CONCEPT_ORDER) or cfg['diagnosis_inputs']!=7:
        raise ValueError('Incompatible joint checkpoint type/order/architecture')
    if payload['loss_weights']!={'concept':1.,'diagnosis':1.} or payload['locked_test_used'] is not False:
        raise ValueError('Incompatible joint checkpoint protocol')
    model.set_trainable_blocks(cfg['unfreeze_last_blocks'])
    model.load_state_dict(payload['model_state'],strict=True)
    model.concept_predictor.pretrained=cfg['pretrained']
    if optimizer is not None: optimizer.load_state_dict(payload['optimizer_state'])
    return payload


def fit_fold(model, loaders, device, config, artifact_dir, checkpoint_dir, provenance):
    validate_config(config); s=config['joint_training']
    if model.model_type!=s['model_type']: raise ValueError('Model/config type differs')
    for role in ('train','validation'):
        ds=loaders[role].dataset
        if ds.split!='development' or ds.role!=role: raise ValueError('Development-only fold required')
        check_order(ds.concept_columns)
        if list(ds.case_nums)!=provenance[f'{"training" if role=="train" else "validation"}_ids']:
            raise ValueError('Fold IDs differ from verified provenance')
    if set(provenance['training_ids'])&set(provenance['validation_ids']): raise ValueError('Fold overlap')
    weights=training_pos_weights(loaders['train'].dataset)
    if weights.tolist()!=[provenance['pos_weights'][c] for c in CONCEPT_ORDER]: raise ValueError('Training weights differ')
    stopper=EarlyStopping(s['patience'],s['min_delta']); optimizer=None; blocks_previous=None; history=[]
    for epoch in range(1,s['epochs']+1):
        blocks=0 if epoch<=s['head_epochs'] else s['unfreeze_last_blocks']
        lr=s['learning_rate'] if blocks==0 else s['finetune_learning_rate']
        if blocks!=blocks_previous:
            model.set_trainable_blocks(blocks); optimizer=build_optimizer(model,s,lr); blocks_previous=blocks
        train,_=run_epoch(model,loaders['train'],device,optimizer,pos_weights=weights)
        validation,rows=run_epoch(model,loaders['validation'],device,pos_weights=weights)
        save_validation_predictions(artifact_dir/f'validation_epoch_{epoch:03}.csv',rows,model.model_type)
        scores=validation_metrics(rows)  # Raw predictions are persisted FIRST.
        improved,stop=stopper.update(scores['validation_diagnosis_auroc'])
        metrics={'epoch':epoch,**{f'train_{k}':v for k,v in train.items()},**{f'validation_{k}':v for k,v in validation.items()},
                 **scores,'learning_rate':lr,'trainable_feature_blocks':blocks}
        history.append(metrics)
        with (artifact_dir/'history.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(metrics)); writer.writeheader(); writer.writerows(history)
        if improved:
            best_epoch=epoch; best_metrics=metrics.copy()
            save_checkpoint(checkpoint_dir/'best.pt',model,optimizer,epoch=epoch,config=config,provenance=provenance,metrics=metrics,stopper=stopper)
        save_checkpoint(checkpoint_dir/'last.pt',model,optimizer,epoch=epoch,config=config,provenance=provenance,metrics=metrics,stopper=stopper)
        print(json.dumps(metrics),flush=True)
        if stop: break
    result={'model_type':model.model_type,'best_epoch':best_epoch,'best_validation_diagnosis_auroc':stopper.best,
            'best_epoch_metrics':best_metrics,'final_epoch_metrics':history[-1],'epochs_completed':len(history),
            'early_stopped':stop,'locked_test_used':False,'concept_order':list(CONCEPT_ORDER),'pos_weights':provenance['pos_weights'],
            'loss_weights':{'concept':1.,'diagnosis':1.},'best_checkpoint':str(checkpoint_dir/'best.pt'),'last_checkpoint':str(checkpoint_dir/'last.pt')}
    (artifact_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result
