"""Seven-concept development training; fixed ordering, no diagnosis/test inference."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, f1_score
from ..models.seven_concept import CONCEPT_ORDER
from .baseline import EarlyStopping, build_optimizer, validate_settings as validate_schedule
from .concept import save_checkpoint, load_checkpoint


def check_order(columns) -> None:
    if tuple(columns) != CONCEPT_ORDER:
        raise ValueError("Concept order must exactly match the frozen seven concepts")


def validate_settings(settings: dict) -> None:
    validate_schedule(settings)
    if settings['threshold'] != 0.5 or settings['selection_metric'] != 'validation_macro_auroc':
        raise ValueError('Use threshold 0.5 and seven-concept macro AUROC selection')
    if 'target' in settings:
        raise ValueError('A single target is not allowed in seven-concept training')


def extract_targets(concepts: torch.Tensor, columns) -> torch.Tensor:
    check_order(columns)
    if concepts.ndim != 2 or concepts.shape[1] != 7 or concepts.shape[0] == 0:
        raise ValueError('Expected nonempty [B,7] concept targets')
    if not torch.all((concepts == 0) | (concepts == 1)):
        raise ValueError('Concept targets must be binary')
    return concepts


def prevalence(dataset) -> dict:
    if dataset.split != 'development':
        raise ValueError('Only development labels are permitted')
    targets = extract_targets(dataset.concepts, dataset.concept_columns)
    total = len(targets)
    return {name: {'total': total, 'positive': int(targets[:,i].sum().item()),
                   'negative': total-int(targets[:,i].sum().item()),
                   'prevalence': float(targets[:,i].double().mean().item())}
            for i,name in enumerate(CONCEPT_ORDER)}


def training_pos_weights(dataset) -> torch.Tensor:
    if dataset.split != 'development' or dataset.role != 'train':
        raise ValueError('Weights require development training labels only')
    counts = prevalence(dataset)
    if any(not c['positive'] or not c['negative'] for c in counts.values()):
        raise ValueError('Each training concept requires both classes')
    return torch.tensor([c['negative']/c['positive'] for c in counts.values()], dtype=torch.float64)


def concept_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weights: torch.Tensor) -> torch.Tensor:
    extract_targets(targets, CONCEPT_ORDER)
    if logits.shape != targets.shape or not torch.isfinite(logits).all():
        raise ValueError('Expected matching finite [B,7] logits')
    weights = torch.as_tensor(pos_weights, device=logits.device, dtype=logits.dtype)
    if weights.shape != (7,) or not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError('Expected seven finite positive training weights')
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=weights)


PREDICTION_COLUMNS = ['case_num'] + [f'{name}_{kind}' for kind in ('target','logit','probability') for name in CONCEPT_ORDER]


def prediction_rows(ids, targets, logits, probabilities) -> list[dict]:
    arrays = {kind: values.detach().cpu().tolist() for kind,values in
              [('target',targets),('logit',logits),('probability',probabilities)]}
    rows = []
    for row_index, identifier in enumerate(ids):
        row = {'case_num': identifier}
        for kind,values in arrays.items():
            row.update({f'{name}_{kind}': int(values[row_index][i]) if kind=='target' else float(values[row_index][i])
                        for i,name in enumerate(CONCEPT_ORDER)})
        rows.append(row)
    return rows


def save_validation_predictions(path: Path, rows: list[dict]) -> None:
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def validation_metrics(rows: list[dict]) -> dict:
    scores, aucs, f1s = {}, [], []
    for name in CONCEPT_ORDER:
        targets = [r[f'{name}_target'] for r in rows]
        logits = [r[f'{name}_logit'] for r in rows]
        probabilities = [r[f'{name}_probability'] for r in rows]
        if set(targets) != {0,1}:
            raise ValueError(f'Both validation classes required for {name}; do not omit from macro AUROC')
        if any(not math.isfinite(x) for x in logits) or any(not math.isfinite(p) or not 0<=p<=1 for p in probabilities):
            raise ValueError('Invalid validation predictions')
        auc = float(roc_auc_score(targets, logits))
        f1 = float(f1_score(targets, [int(p>=.5) for p in probabilities], labels=[0,1], average='macro', zero_division=0))
        scores[f'{name}_auroc'], scores[f'{name}_macro_f1'] = auc, f1
        aucs.append(auc); f1s.append(f1)
    scores['validation_macro_auroc'] = sum(aucs)/7
    scores['validation_macro_f1'] = sum(f1s)/7
    return scores


def run_epoch(model: nn.Module, loader, device: torch.device,
              optimizer: torch.optim.Optimizer | None = None, *, pos_weights: torch.Tensor) -> tuple[float, list[dict]]:
    """Sample-weighted concept BCE; diagnosis and metadata are never consumed."""
    training = optimizer is not None
    expected_role = "train" if training else "validation"
    if loader.dataset.split != "development" or loader.dataset.role != expected_role:
        raise ValueError("Refusing non-development or wrong-role loader before iteration")
    expected = set(loader.dataset.case_nums)
    model.train(training)
    seen, predictions = set(), []
    loss_sum = 0.0
    with torch.set_grad_enabled(training):
        for batch in loader:
            ids = list(batch["case_num"])
            if (set(ids) - expected or seen & set(ids) or len(ids) != len(set(ids))
                    or any(s != "development" for s in batch["split"])):
                raise ValueError("Unexpected or repeated case IDs in development batch")
            seen.update(ids)
            image = batch["image"].to(device)
            targets = extract_targets(batch["concepts"], loader.dataset.concept_columns).to(device, dtype=torch.float32)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = concept_loss(logits, targets, pos_weights)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(ids)
            if not training:
                probabilities = torch.sigmoid(logits)
                predictions.extend(prediction_rows(ids, targets, logits, probabilities))
    if not seen or seen != expected:
        raise ValueError("Epoch must cover every selected development case exactly once")
    return loss_sum / len(seen), predictions



def fit_fold(model, loaders: dict, device: torch.device, config: dict,
             artifact_dir: Path, checkpoint_dir: Path, provenance: dict) -> dict:
    """Run the configured schedule; callers supply verified development-fold loaders."""
    settings = config["concept_training"]
    validate_settings(settings)
    for role in ("train", "validation"):
        if loaders[role].dataset.split != "development" or loaders[role].dataset.role != role:
            raise ValueError("Only development training/validation loaders are permitted")
    check_order(config["concepts"]["target_order"])
    check_order(model.concept_names)
    for loader in loaders.values():
        check_order(loader.dataset.concept_columns)
    pos_weights = training_pos_weights(loaders["train"].dataset)
    for counts in prevalence(loaders["validation"].dataset).values():
        if not counts["positive"] or not counts["negative"]:
            raise ValueError("All seven validation concepts require both classes")
    stopper = EarlyStopping(settings["patience"], settings["min_delta"])
    optimizer, current_blocks, best_epoch = None, None, 0
    history = []
    for epoch in range(1, settings["epochs"] + 1):
        blocks = 0 if epoch <= settings["head_epochs"] else settings["unfreeze_last_blocks"]
        learning_rate = settings["learning_rate"] if blocks == 0 else settings["finetune_learning_rate"]
        if blocks != current_blocks:
            model.set_trainable_blocks(blocks)
            # Explicit phase transition: rebuild optimizer, resetting its momentum/state.
            optimizer = build_optimizer(model, settings, learning_rate)
            current_blocks = blocks
        train_loss, _ = run_epoch(model, loaders["train"], device, optimizer, pos_weights=pos_weights)
        validation_loss, rows = run_epoch(model, loaders["validation"], device, pos_weights=pos_weights)
        save_validation_predictions(artifact_dir / f"validation_epoch_{epoch:03}.csv", rows)
        scores = validation_metrics(rows)
        auroc = scores["validation_macro_auroc"]  # Raw predictions are saved before aggregate metrics.
        improved, stop = stopper.update(auroc)
        metrics = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
                   **scores, "learning_rate": learning_rate,
                   "trainable_feature_blocks": blocks}
        history.append(metrics)
        with (artifact_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics))
            writer.writeheader()
            writer.writerows(history)
        if improved:
            best_epoch = epoch
            best_metrics = metrics.copy()
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch=epoch, config=config,
                            provenance=provenance, metrics=metrics, stopper=stopper)
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch=epoch, config=config,
                        provenance=provenance, metrics=metrics, stopper=stopper)
        print(json.dumps(metrics), flush=True)
        if stop:
            break
    result = {"best_epoch": best_epoch, "best_validation_macro_auroc": stopper.best, "best_epoch_metrics": best_metrics, "final_epoch_metrics": history[-1],
              "epochs_completed": len(history), "early_stopped": stop,
              "locked_test_used": False, "concept_order": list(CONCEPT_ORDER), "pos_weights": dict(zip(CONCEPT_ORDER, pos_weights.tolist())), "threshold": 0.5}
    (artifact_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
