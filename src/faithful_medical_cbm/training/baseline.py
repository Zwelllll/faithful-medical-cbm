"""Development-fold baseline training primitives. No locked-test execution path."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path

from sklearn.metrics import roc_auc_score
import torch
from torch import nn


def validate_settings(settings: dict) -> None:
    for key in ("epochs", "patience"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(settings["head_epochs"]) is not int or not 0 <= settings["head_epochs"] <= settings["epochs"]:
        raise ValueError("head_epochs must be between zero and epochs")
    if type(settings["unfreeze_last_blocks"]) is not int or not 1 <= settings["unfreeze_last_blocks"] <= 9:
        raise ValueError("unfreeze_last_blocks must be between 1 and 9")
    for key in ("learning_rate", "finetune_learning_rate", "weight_decay", "min_delta", "momentum"):
        value = settings[key]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key}")
    if settings["learning_rate"] == 0 or settings["finetune_learning_rate"] == 0:
        raise ValueError("Learning rates must be positive")
    if settings["optimizer"] not in ("adamw", "adam", "sgd"):
        raise ValueError("Unsupported optimizer")
    if settings["pretrained"] is not True:
        raise ValueError("Real baseline runs require ImageNet pretraining; tests construct weights=None directly")


def build_optimizer(model: nn.Module, settings: dict, learning_rate: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    options = {"lr": learning_rate, "weight_decay": settings["weight_decay"]}
    if settings["optimizer"] == "adamw":
        return torch.optim.AdamW(params, **options)
    if settings["optimizer"] == "adam":
        return torch.optim.Adam(params, **options)
    if settings["optimizer"] == "sgd":
        return torch.optim.SGD(params, momentum=settings["momentum"], **options)
    raise ValueError("Unsupported optimizer")


def development_fold_loaders(factory, fold: int) -> dict:
    """Check exact fold isolation before any loader is iterated."""
    loaders = factory.fold(fold)
    train_ids = set(loaders["train"].dataset.case_nums)
    val_ids = set(loaders["validation"].dataset.case_nums)
    for role, expected_ids in (("train", train_ids), ("validation", val_ids)):
        dataset = loaders[role].dataset
        if dataset.split != "development" or dataset.role != role:
            raise ValueError("Only development training/validation loaders are permitted")
        if len(dataset.case_nums) != len(expected_ids) or expected_ids & set(factory.test_ids):
            raise ValueError("Duplicate IDs or locked-test contamination")
    if train_ids & val_ids or train_ids | val_ids != set(factory.development_ids):
        raise ValueError("Training and validation must partition development")
    expected_val = {i for i, f in factory.validation_folds.items() if f == fold}
    if val_ids != expected_val:
        raise ValueError("Validation IDs differ from the frozen selected fold")
    return loaders


def binary_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape or logits.ndim != 1:
        raise ValueError("Binary logits and targets must both have shape [B]")
    if not torch.isfinite(logits).all() or not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("Nonfinite logits or invalid binary targets")
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def run_epoch(model: nn.Module, loader, device: torch.device,
              optimizer: torch.optim.Optimizer | None = None) -> tuple[float, list[dict]]:
    """Sample-weighted BCE and raw validation predictions; concepts are never consumed."""
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
            targets = batch["diagnosis"].to(device, dtype=torch.float32)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = binary_loss(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(ids)
            if not training:
                probabilities = torch.sigmoid(logits)
                predictions.extend({"case_num": identifier, "diagnosis": int(y), "logit": float(logit),
                                    "probability": float(probability)}
                                   for identifier, y, logit, probability in zip(
                                       ids, targets.cpu().tolist(), logits.cpu().tolist(), probabilities.cpu().tolist()))
    if not seen or seen != expected:
        raise ValueError("Epoch must cover every selected development case exactly once")
    return loss_sum / len(seen), predictions


def save_validation_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_num", "diagnosis", "logit", "probability"])
        writer.writeheader()
        writer.writerows(rows)


def validation_auroc(rows: list[dict]) -> float:
    if {row["diagnosis"] for row in rows} != {0, 1}:
        raise ValueError("Validation AUROC requires both diagnosis classes")
    # Logits preserve ranking even when float32 sigmoid saturates; no threshold tuning.
    return float(roc_auc_score([r["diagnosis"] for r in rows], [r["logit"] for r in rows]))


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float
    best: float = -math.inf
    bad_epochs: int = 0

    def update(self, auroc: float) -> tuple[bool, bool]:
        if not math.isfinite(auroc):
            raise ValueError("Validation AUROC must be finite")
        improved = auroc > self.best + self.min_delta
        if improved:
            self.best, self.bad_epochs = auroc, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def save_checkpoint(path: Path, model, optimizer, *, epoch: int, config: dict,
                    provenance: dict, metrics: dict, stopper: EarlyStopping) -> None:
    payload = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
               "epoch": epoch, "config": json.loads(json.dumps(config, default=str)),
               "provenance": provenance, "metrics": metrics, "early_stopping": asdict(stopper),
               "model_config": {"architecture": "efficientnet_b0", "pretrained": model.pretrained,
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
    if payload["model_config"]["architecture"] != "efficientnet_b0":
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
    settings = config["baseline"]
    validate_settings(settings)
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
        train_loss, _ = run_epoch(model, loaders["train"], device, optimizer)
        validation_loss, rows = run_epoch(model, loaders["validation"], device)
        save_validation_predictions(artifact_dir / f"validation_epoch_{epoch:03}.csv", rows)
        auroc = validation_auroc(rows)  # Raw predictions are saved before aggregate metrics.
        improved, stop = stopper.update(auroc)
        metrics = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
                   "validation_auroc": auroc, "learning_rate": learning_rate,
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
              "locked_test_used": False}
    (artifact_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
