"""Single-concept development sanity training; no diagnosis prediction or test path."""
from __future__ import annotations
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, f1_score
from .baseline import EarlyStopping, build_optimizer, validate_settings as validate_schedule

TARGET = "atypical_pigment_network"


def validate_settings(settings: dict) -> None:
    validate_schedule(settings)
    if settings["target"] != TARGET or settings["threshold"] != 0.5 or settings["selection_metric"] != "validation_auroc":
        raise ValueError("Stage 5 permits only atypical_pigment_network at threshold 0.5 and AUROC selection")


def checked_target_index(columns, config: dict) -> int:
    expected = tuple(config["concepts"]["target_order"])
    if len(expected) != 7 or len(set(expected)) != 7 or tuple(columns) != expected:
        raise ValueError("Concept order differs from frozen Stage 2A order")
    if config["concept_training"]["target"] != TARGET:
        raise ValueError("Only the Stage 5 target is allowed")
    return expected.index(TARGET)


def select_target(concepts: torch.Tensor, index: int) -> torch.Tensor:
    if concepts.ndim != 2 or concepts.shape[1] != 7 or not 0 <= index < 7:
        raise ValueError("Expected a batch of seven ordered processed targets")
    target = concepts[:, index]
    if not torch.all((target == 0) | (target == 1)):
        raise ValueError("Nonbinary concept target")
    return target


def prevalence(dataset, index: int) -> dict:
    if dataset.split != "development":
        raise ValueError("Only development labels are allowed")
    target = select_target(dataset.concepts, index)
    positive = int(target.sum().item())
    total = len(target)
    if total == 0:
        raise ValueError("Empty subset")
    return {"total": total, "positive": positive, "negative": total-positive, "prevalence": positive/total}


def training_pos_weight(dataset, index: int) -> float:
    if dataset.split != "development" or dataset.role != "train":
        raise ValueError("pos_weight requires the development training subset only")
    counts = prevalence(dataset, index)
    if not counts["positive"] or not counts["negative"]:
        raise ValueError("Training concept requires both classes")
    return counts["negative"] / counts["positive"]


def concept_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: float) -> torch.Tensor:
    if logits.ndim != 1 or logits.shape != targets.shape or not torch.isfinite(logits).all():
        raise ValueError("Expected finite matching [B] logits and targets")
    if not torch.all((targets == 0) | (targets == 1)) or not math.isfinite(pos_weight) or pos_weight <= 0:
        raise ValueError("Invalid binary targets or training weight")
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=logits.new_tensor(pos_weight))


def save_validation_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_num", "concept", "target", "logit", "probability"])
        writer.writeheader()
        writer.writerows(rows)


def validation_metrics(rows: list[dict]) -> dict:
    if {r["target"] for r in rows} != {0, 1}:
        raise ValueError("Validation AUROC requires both concept classes")
    targets = [r["target"] for r in rows]
    return {"validation_auroc": float(roc_auc_score(targets, [r["logit"] for r in rows])),
            "validation_macro_f1": float(f1_score(targets, [int(r["probability"] >= 0.5) for r in rows],
                                                  labels=[0, 1], average="macro", zero_division=0))}


def run_epoch(model: nn.Module, loader, device: torch.device,
              optimizer: torch.optim.Optimizer | None = None, *, target_index: int, pos_weight: float) -> tuple[float, list[dict]]:
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
            targets = select_target(batch["concepts"], target_index).to(device, dtype=torch.float32)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = concept_loss(logits, targets, pos_weight)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(ids)
            if not training:
                probabilities = torch.sigmoid(logits)
                predictions.extend({"case_num": identifier, "concept": TARGET, "target": int(y), "logit": float(logit),
                                    "probability": float(probability)}
                                   for identifier, y, logit, probability in zip(
                                       ids, targets.cpu().tolist(), logits.cpu().tolist(), probabilities.cpu().tolist()))
    if not seen or seen != expected:
        raise ValueError("Epoch must cover every selected development case exactly once")
    return loss_sum / len(seen), predictions



def save_checkpoint(path: Path, model, optimizer, *, epoch: int, config: dict,
                    provenance: dict, metrics: dict, stopper: EarlyStopping) -> None:
    payload = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
               "epoch": epoch, "config": json.loads(json.dumps(config, default=str)),
               "provenance": provenance, "metrics": metrics, "early_stopping": asdict(stopper),
               "model_config": {"architecture": "efficientnet_b0_concept", "concept_names": list(model.concept_names), "pretrained": model.pretrained,
                                "unfreeze_last_blocks": model.unfreeze_last_blocks}}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, model, optimizer=None, *, map_location="cpu") -> dict:
    """Construct model with pretrained=False first: restore never needs a download.

    Restores model/optional optimizer state, not an exact interrupted RNG/loader resume.
    Configure an optional optimizer for the saved trainable blocks before calling.
    """
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload["model_config"]["architecture"] != "efficientnet_b0_concept" or tuple(payload["model_config"].get("concept_names", ())) != model.concept_names:
        raise ValueError("Checkpoint architecture mismatch")
    model.set_trainable_blocks(payload["model_config"]["unfreeze_last_blocks"])
    model.load_state_dict(payload["model_state"])
    model.pretrained = payload["model_config"]["pretrained"]
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def fit_fold(model, loaders: dict, device: torch.device, config: dict,
             artifact_dir: Path, checkpoint_dir: Path, provenance: dict) -> dict:
    """Run the configured schedule; callers supply verified development-fold loaders."""
    settings = config["concept_training"]
    validate_settings(settings)
    for role in ("train", "validation"):
        if loaders[role].dataset.split != "development" or loaders[role].dataset.role != role:
            raise ValueError("Only development training/validation loaders are permitted")
    target_index = checked_target_index(loaders["train"].dataset.concept_columns, config)
    if tuple(loaders["validation"].dataset.concept_columns) != tuple(loaders["train"].dataset.concept_columns):
        raise ValueError("Validation concept order differs")
    pos_weight = training_pos_weight(loaders["train"].dataset, target_index)
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
        train_loss, _ = run_epoch(model, loaders["train"], device, optimizer, target_index=target_index, pos_weight=pos_weight)
        validation_loss, rows = run_epoch(model, loaders["validation"], device, target_index=target_index, pos_weight=pos_weight)
        save_validation_predictions(artifact_dir / f"validation_epoch_{epoch:03}.csv", rows)
        scores = validation_metrics(rows)
        auroc = scores["validation_auroc"]  # Raw predictions are saved before aggregate metrics.
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
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch=epoch, config=config,
                            provenance=provenance, metrics=metrics, stopper=stopper)
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch=epoch, config=config,
                        provenance=provenance, metrics=metrics, stopper=stopper)
        print(json.dumps(metrics), flush=True)
        if stop:
            break
    result = {"best_epoch": best_epoch, "best_validation_auroc": stopper.best,
              "epochs_completed": len(history), "early_stopped": stop,
              "locked_test_used": False, "concept": TARGET, "pos_weight": pos_weight, "threshold": 0.5}
    (artifact_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
