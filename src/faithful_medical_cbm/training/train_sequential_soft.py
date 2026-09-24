"""Stage 8: fixed logistic regression cross-fitted on frozen OOF probabilities."""
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
from ..models.soft_cbm import FEATURE_COLUMNS, extract_features, fit_head, parameters, constructor_kwargs
from ..evaluation.assemble_oof import OOF_COLUMNS, binary, frozen_inputs, read_csv, sha, unique_ids
from ..evaluation.diagnosis_metrics import diagnosis_metrics, expected_calibration_error

PREDICTION_COLUMNS = ["case_num", "validation_fold", "diagnosis_binary", "decision_score",
                      "melanoma_probability", "predicted_diagnosis"]
MODEL_LIMITATION = (
    "LR-head cross-fitting on existing OOF features, not fully nested CNN-plus-LR cross-validation. "
    "Concept-model early stopping used its held-out concept labels, and concept models producing "
    "LR-training features may have trained on cases in the LR-validation fold. "
    "These are development diagnostics, not a final independent test estimate. "
    "Patient-level independence is unverifiable; OOF-to-ensemble feature shift remains accepted."
)


def load_inputs(config_path: Path) -> dict:
    config_path = config_path.resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config["stage"] != 8 or tuple(config["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError("Stage 8 requires exactly seven ordered probability features")
    if config["target"] != "diagnosis_binary" or config["threshold"] != .5 or config["ece_bins"] != 10 or config["preprocessing"] != "none":
        raise ValueError("Fixed diagnosis/threshold/binning/preprocessing protocol required")
    settings = config["logistic_regression"]
    if settings["class_weight"] is not None or settings["fit_intercept"] is not True:
        raise ValueError("Use an intercept and no class weighting")
    constructor_kwargs(settings)
    root = config_path.parent.parent
    frozen_config = (root / config["frozen_data_config"]).resolve()
    frozen = frozen_inputs(frozen_config)
    oof_path, integrity_path = [(root/config[k]).resolve() for k in ("input_oof", "input_integrity")]
    oof_root = (frozen["config"]["paths"]["artifacts"] / "oof").resolve()
    if oof_root not in oof_path.parents or oof_root not in integrity_path.parents:
        raise ValueError("Inputs must be the Stage 7 OOF artifacts")
    data, integrity_bytes = oof_path.read_bytes(), integrity_path.read_bytes()
    integrity = json.loads(integrity_bytes)
    if integrity["status"] != "passed" or integrity["oof_sha256"] != sha(data) or integrity["schema"] != OOF_COLUMNS:
        raise ValueError("Stage 7 OOF hash/schema/integrity mismatch")
    for key in ("exact_fold_membership", "training_ids_excluded_per_fold", "concept_order_preserved", "diagnosis_join_verified"):
        if integrity[key] is not True:
            raise ValueError(f"Missing Stage 7 integrity assertion: {key}")
    for key in ("duplicates", "missing_development_ids", "locked_test_overlap"):
        if integrity[key] != 0:
            raise ValueError(f"Stage 7 integrity failure: {key}")
    rows = read_csv(data, OOF_COLUMNS)
    features, labels, folds, ids = validate_rows(rows, frozen)
    features.setflags(write=False)
    hashes = {**frozen["hashes"], str(config_path): sha(config_bytes),
              str(frozen_config): sha(frozen_config.read_bytes()),
              str(oof_path): sha(data), str(integrity_path): sha(integrity_bytes)}
    return dict(config=config, root=root, frozen=frozen, features=features, labels=labels,
                folds=folds, ids=ids, hashes=hashes, oof_sha256=sha(data))


def validate_rows(rows: list[dict], frozen: dict) -> tuple:
    ids = [r["case_num"] for r in rows]
    selected = unique_ids(ids, "soft OOF input")
    if selected & frozen["test"]:
        raise ValueError("Locked-test ID in soft OOF input")
    if len(rows) != 658 or selected != set(frozen["development"]):
        raise ValueError("Soft OOF must cover exactly 658 development cases")
    by_id = {r["case_num"]: r for r in rows}
    # Canonical frozen order; never modify individual feature values.
    rows = [by_id[i] for i in frozen["development"]]
    ids = [r["case_num"] for r in rows]
    folds, labels = [], []
    for row in rows:
        i = row["case_num"]
        if row["validation_fold"] not in ("0", "1", "2", "3") or int(row["validation_fold"]) != frozen["folds"][i]:
            raise ValueError("OOF fold differs from frozen fold")
        y = binary(row["diagnosis_binary"])
        if y != frozen["truth"][i]["diagnosis_binary"]:
            raise ValueError("OOF diagnosis differs from frozen processed target")
        folds.append(int(row["validation_fold"]))
        labels.append(y)
    return extract_features(rows), np.array(labels, dtype=np.int64), np.array(folds, dtype=np.int64), ids


def model_record(model, *, role: str, fold, train_ids: list[str], validation_ids: list[str],
                 config: dict, provenance: dict) -> dict:
    if set(train_ids) & set(validation_ids):
        raise ValueError("LR training/validation overlap")
    return {"label": role, "validation_fold": fold, "used_for_development_metrics": fold is not None,
            **parameters(model), "training_ids": train_ids, "validation_ids": validation_ids,
            "logistic_regression": config["logistic_regression"],
            "constructor_kwargs": constructor_kwargs(config["logistic_regression"]),
            "preprocessing": "none", "threshold": .5, "provenance": provenance}


def cross_fit(data: dict, provenance: dict) -> tuple[list[dict], list[dict]]:
    features, y, folds, ids = (data[k] for k in ("features", "labels", "folds", "ids"))
    before = features.copy()
    predictions, models = [], []
    for fold in range(4):
        train = np.flatnonzero(folds != fold)
        validation = np.flatnonzero(folds == fold)
        if not len(train) or not len(validation) or set(train) & set(validation):
            raise ValueError("Invalid LR fold partition")
        model = fit_head(features[train], y[train], data["config"]["logistic_regression"])
        scores = model.decision_function(features[validation])
        probabilities = model.predict_proba(features[validation])[:, 1]
        for j, score, probability in zip(validation, scores, probabilities):
            if not np.isfinite(score) or not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Invalid diagnosis prediction")
            predictions.append({"case_num": ids[j], "validation_fold": fold, "diagnosis_binary": int(y[j]),
                                "decision_score": float(score), "melanoma_probability": float(probability),
                                "predicted_diagnosis": int(probability >= .5)})
        models.append(model_record(model, role="CROSS-FITTED DEVELOPMENT SOFT CBM HEAD", fold=fold,
                                  train_ids=[ids[j] for j in train], validation_ids=[ids[j] for j in validation],
                                  config=data["config"], provenance=provenance))
    if not np.array_equal(features, before):
        raise ValueError("Concept probabilities were modified")
    if unique_ids([r["case_num"] for r in predictions], "cross-fitted predictions") != set(ids):
        raise ValueError("Cross-fitted predictions must cover every development case once")
    lookup = {r["case_num"]: r for r in predictions}
    return [lookup[i] for i in ids], models


def csv_bytes(rows: list[dict], columns: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def collect_provenance(data: dict, config_path: Path) -> dict:
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=data["root"], capture_output=True, text=True, check=False)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=data["root"], capture_output=True, text=True, check=False)
        commit = git.stdout.strip() if git.returncode == 0 else None
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    except OSError:
        commit, dirty = None, None
    package = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), package/"models/soft_cbm.py", package/"evaluation/diagnosis_metrics.py",
               package/"evaluation/assemble_oof.py", package/"config.py"]
    return {"oof_sha256": data["oof_sha256"], "stage8_config_sha256": sha(config_path.read_bytes()),
            "git_commit": commit, "git_working_tree_dirty": dirty,
            "source_sha256": {str(p.relative_to(package)): sha(p.read_bytes()) for p in sources},
            "python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn")},
            "feature_columns": list(FEATURE_COLUMNS), "limitation": MODEL_LIMITATION}


def evaluate_rows(rows: list[dict]) -> dict:
    return diagnosis_metrics([r["diagnosis_binary"] for r in rows], [r["decision_score"] for r in rows],
                             [r["melanoma_probability"] for r in rows], bins=10)


def run(config_path: Path, output_dir: Path | None = None) -> dict:
    data = load_inputs(config_path)
    destination = (output_dir or data["root"]/data["config"]["output_dir"]).resolve()
    for protected in [data["root"]/"artifacts/oof", *(data["frozen"]["config"]["paths"][k] for k in ("raw", "processed", "splits"))]:
        protected = protected.resolve()
        if destination == protected or protected in destination.parents or destination in protected.parents:
            raise ValueError("Output path overlaps protected inputs")
    if destination.exists() and any(p.name != ".gitattributes" for p in destination.iterdir()):
        raise FileExistsError("Soft CBM outputs already exist; refusing overwrite")
    provenance = collect_provenance(data, config_path)
    predictions, models = cross_fit(data, provenance)
    for name, expected in data["hashes"].items():
        if sha(Path(name).read_bytes()) != expected:
            raise ValueError("Input changed during cross-fitting")
    destination.mkdir(parents=True, exist_ok=True)
    raw = csv_bytes(predictions, PREDICTION_COLUMNS)
    with (destination/"cross_fitted_predictions.csv").open("xb") as stream:
        stream.write(raw)  # Raw held-out predictions precede aggregate metrics.
    fold_metrics = [{"validation_fold": f, **evaluate_rows([r for r in predictions if r["validation_fold"] == f])}
                    for f in range(4)]
    pooled = {"evaluation": "Cross-fitted LR heads on existing development OOF probabilities",
              **evaluate_rows(predictions), "full_development_head_used": False,
              "ece_definition": expected_calibration_error([r["diagnosis_binary"] for r in predictions],
                  [r["melanoma_probability"] for r in predictions])["rule"], "limitation": MODEL_LIMITATION}
    calibration = {"pooled": expected_calibration_error([r["diagnosis_binary"] for r in predictions],
                                                       [r["melanoma_probability"] for r in predictions]),
                   "folds": {str(f): expected_calibration_error(
                       [r["diagnosis_binary"] for r in predictions if r["validation_fold"] == f],
                       [r["melanoma_probability"] for r in predictions if r["validation_fold"] == f]) for f in range(4)}}
    # Fit only AFTER held-out evaluation is complete. Never call predict on this head here.
    full = fit_head(data["features"], data["labels"], data["config"]["logistic_regression"])
    full_record = model_record(full, role="FULL-DEVELOPMENT SOFT CBM HEAD", fold=None,
                              train_ids=data["ids"], validation_ids=[], config=data["config"], provenance=provenance)
    outputs = {"pooled_metrics.json": pooled, "full_development_model.json": full_record,
               "calibration_bins.json": calibration, "run_config.json": data["config"]}
    outputs.update({f"fold_{i}_model.json": m for i, m in enumerate(models)})
    payloads = {name: (json.dumps(value, indent=2, allow_nan=False)+"\n").encode() for name, value in outputs.items()}
    payloads["fold_metrics.csv"] = csv_bytes(fold_metrics, list(fold_metrics[0]))
    for name, expected in data["hashes"].items():
        if sha(Path(name).read_bytes()) != expected:
            raise ValueError("Input changed during full-development fit")
    integrity = {"status": "passed", "input_rows": 658, "prediction_rows": len(predictions),
                 "unique_case_num": len({r["case_num"] for r in predictions}), "duplicates": 0,
                 "missing_development_cases": 0, "locked_test_overlap": 0,
                 "feature_columns": list(FEATURE_COLUMNS), "features_used": 7,
                 "probabilities_only": True, "probabilities_modified": False,
                 "diagnosis_matches_frozen_cohort": True, "folds_match_frozen_split": True,
                 "all_lr_validation_ids_excluded_from_training": True,
                 "full_development_head_used_for_metrics": False,
                 "cnn_training_or_inference": False, "images_accessed": False,
                 "threshold_optimized": False, "ece_bins": 10, "input_sha256": data["hashes"],
                 "output_sha256": {"cross_fitted_predictions.csv": sha(raw), **{k: sha(v) for k, v in payloads.items()}},
                 "provenance": provenance}
    for name, payload in payloads.items():
        with (destination/name).open("xb") as stream:
            stream.write(payload)
    with (destination/"integrity.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(integrity, indent=2, allow_nan=False)+"\n")
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sequential_soft.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
