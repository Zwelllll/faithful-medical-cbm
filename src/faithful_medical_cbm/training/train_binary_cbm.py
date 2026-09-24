"""Stage 9: fixed hard/oracle LR heads and a read-only Stage 8 comparison."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from . import train_sequential_soft as shared
from ..models.soft_cbm import fit_head
from ..models.binary_cbm import binary_features, feature_columns
from ..evaluation.assemble_oof import CONCEPT_ORDER, OOF_COLUMNS, read_csv, sha
from ..evaluation.diagnosis_metrics import expected_calibration_error

METRICS = ("auroc", "macro_f1", "accuracy", "sensitivity", "specificity", "brier", "ece")


def verify_hashes(hashes: dict) -> None:
    for name, expected in hashes.items():
        if sha(Path(name).read_bytes()) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")


def load_inputs(config_path: Path) -> tuple[dict, dict, list[dict], dict]:
    config_path = config_path.resolve()
    cfg = json.loads(config_path.read_bytes())
    if cfg["stage"] != 9 or cfg["concept_threshold"] != .5:
        raise ValueError("Frozen Stage 9 threshold required")
    root = config_path.parent.parent
    stage8 = root/cfg["stage8_config"]
    data = shared.load_inputs(stage8)
    soft = root/cfg["soft_outputs"]
    integrity = json.loads((soft/"integrity.json").read_bytes())
    if integrity["status"] != "passed":
        raise ValueError("Stage 8 integrity did not pass")
    hashes = {str(soft/name): h for name,h in integrity["output_sha256"].items()}
    hashes[str(soft/"integrity.json")] = sha((soft/"integrity.json").read_bytes())
    verify_hashes(hashes)
    if json.loads((soft/"run_config.json").read_bytes()) != data["config"]:
        raise ValueError("Stage 8 configuration changed")
    if integrity["provenance"]["oof_sha256"] != data["oof_sha256"]:
        raise ValueError("Stage 8 OOF source changed")
    data["hashes"].update(hashes)
    data["hashes"][str(config_path)] = sha(config_path.read_bytes())
    rows = read_csv((root/data["config"]["input_oof"]).read_bytes(), OOF_COLUMNS)
    lookup = {r["case_num"]: r for r in rows}
    rows = [lookup[i] for i in data["ids"]]
    targets = binary_features(rows, "oracle")
    expected = [[data["frozen"]["truth"][i][c] for c in CONCEPT_ORDER] for i in data["ids"]]
    if not np.array_equal(targets, expected):
        raise ValueError("Oracle labels differ from frozen processed concepts")
    return cfg, data, rows, hashes


def rename_record(record: dict, mode: str) -> dict:
    columns = feature_columns(mode)
    role = "HARD CBM HEAD" if mode == "sequential_hard" else "ORACLE CONCEPT HEAD"
    prefix = "FULL-DEVELOPMENT" if record["validation_fold"] is None else "CROSS-FITTED DEVELOPMENT"
    record.update(label=f"{prefix} {role}", feature_columns=list(columns),
                  coefficients_by_feature=dict(zip(columns, record["coefficients"])))
    return record


def comparison(soft: dict, hard: dict, oracle: dict) -> dict:
    values = {name: {m: result[m] for m in METRICS}
              for name,result in (("soft",soft),("hard",hard),("oracle",oracle))}
    gaps = {f"{a}_minus_{b}": {m: values[a][m]-values[b][m] for m in ("auroc","macro_f1")}
            for a,b in (("soft","hard"),("oracle","soft"),("oracle","hard"))}
    return {"pooled_metrics": values, "descriptive_gaps": gaps,
            "soft_results_recomputed": False, "winner_label": None,
            "interpretation": "Development diagnostics; descriptive differences, not causal effects or independent test estimates."}


def write_json(path: Path, value: dict) -> None:
    with path.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, allow_nan=False)+"\n").encode())


def run(config_path: Path, output_root: Path | None = None) -> dict:
    cfg, data, rows, soft_hashes = load_inputs(config_path)
    root = (output_root or data["root"]/cfg["output_root"]).resolve()
    destinations = [root/n for n in ("sequential_hard","oracle","comparison")]
    protected = [data["root"]/"artifacts/oof", data["root"]/cfg["soft_outputs"],
                 *(data["frozen"]["config"]["paths"][k] for k in ("raw","processed","splits"))]
    for dest in destinations:
        for path in protected:
            path = path.resolve()
            if dest == path or path in dest.parents or dest in path.parents:
                raise ValueError("Output overlaps protected inputs")
        if dest.exists() and any(p.name != ".gitattributes" for p in dest.iterdir()):
            raise FileExistsError(f"Refusing to overwrite {dest}")
    base_provenance = shared.collect_provenance(data, data["root"]/cfg["stage8_config"])
    package = Path(__file__).resolve().parents[1]
    for path in (Path(__file__), package/"models/binary_cbm.py"):
        base_provenance["source_sha256"][str(path.relative_to(package))] = sha(path.read_bytes())
    base_provenance.update(stage9_config=cfg, stage9_config_sha256=sha(config_path.read_bytes()),
                           concept_order=list(CONCEPT_ORDER), stage8_outputs_sha256=soft_hashes)
    results = {}
    for mode,dest in zip(("sequential_hard","oracle"),destinations):
        columns = list(feature_columns(mode))
        features = binary_features(rows, mode)
        features.setflags(write=False)
        current = {**data, "features": features}
        limitation = shared.MODEL_LIMITATION if mode == "sequential_hard" else (
            "Oracle uses true concepts, not deployable image predictions. This fixed linear model is an analytical "
            "reference, not a guaranteed empirical upper bound. Patient-level independence is unverifiable.")
        provenance = {**base_provenance, "feature_columns": columns, "model": mode, "limitation": limitation,
                      "feature_rule": "OOF probability >= 0.5" if mode == "sequential_hard" else "frozen target columns only"}
        # Shared cross-fitting is feature-agnostic; pass ONLY the selected seven binary inputs.
        predictions, models = shared.cross_fit(current, provenance)
        models = [rename_record(m,mode) for m in models]
        for i,row in enumerate(predictions):
            row.update({c: int(v) for c,v in zip(columns,features[i])})
        prediction_columns = shared.PREDICTION_COLUMNS[:3]+columns+shared.PREDICTION_COLUMNS[3:]
        verify_hashes(data["hashes"])
        dest.mkdir(parents=True, exist_ok=True)
        with (dest/"cross_fitted_predictions.csv").open("xb") as stream:
            stream.write(shared.csv_bytes(predictions,prediction_columns))
        # Raw predictions are persisted before any aggregate calculation.
        fold_metrics = [{"validation_fold": f, **shared.evaluate_rows([r for r in predictions if r["validation_fold"]==f])}
                        for f in range(4)]
        pooled = {**shared.evaluate_rows(predictions), "full_development_head_used": False, "limitation": limitation}
        calibration = {"pooled": expected_calibration_error(data["labels"], [r["melanoma_probability"] for r in predictions]),
                       "folds": {str(f): expected_calibration_error(
                           [r["diagnosis_binary"] for r in predictions if r["validation_fold"]==f],
                           [r["melanoma_probability"] for r in predictions if r["validation_fold"]==f]) for f in range(4)}}
        full = fit_head(features,data["labels"],data["config"]["logistic_regression"])
        full_record = rename_record(shared.model_record(full,role="",fold=None,train_ids=data["ids"],
            validation_ids=[],config=data["config"],provenance=provenance),mode)
        outputs = {"pooled_metrics.json": pooled, "calibration_bins.json": calibration,
                   "full_development_model.json": full_record,
                   "run_config.json": {"stage":9,"model":mode,"stage8_config":data["config"],
                                       "feature_columns":columns,"concept_threshold":.5}}
        outputs.update({f"fold_{f}_model.json": m for f,m in enumerate(models)})
        for name,value in outputs.items():
            write_json(dest/name,value)
        with (dest/"fold_metrics.csv").open("xb") as stream:
            stream.write(shared.csv_bytes(fold_metrics,list(fold_metrics[0])))
        verify_hashes(data["hashes"])
        write_json(dest/"integrity.json", {"status":"passed","prediction_rows":len(predictions),
            "unique_case_num":len({r["case_num"] for r in predictions}),"exact_development_coverage":True,
            "locked_test_overlap":0,"features_used":7,"feature_columns":columns,
            "feature_source_verified":True,"diagnosis_matches_frozen_cohort":True,"fold_isolation":True,
            "full_development_head_used_for_metrics":False,"images_accessed":False,
            "cnn_training_or_inference":False,"threshold_optimized":False,"soft_outputs_unchanged":True,
            "input_sha256":data["hashes"],"provenance":provenance,
            "output_sha256":{p.name:sha(p.read_bytes()) for p in dest.iterdir() if p.suffix in (".csv",".json")}})
        results[mode] = pooled
    soft = json.loads((data["root"]/cfg["soft_outputs"]/"pooled_metrics.json").read_bytes())
    report = comparison(soft,results["sequential_hard"],results["oracle"])
    report.update(stage8_outputs_sha256=soft_hashes, input_oof_sha256=data["oof_sha256"])
    destinations[2].mkdir(parents=True, exist_ok=True)
    write_json(destinations[2]/"comparison.json",report)
    verify_hashes(data["hashes"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/binary_cbm.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.config),indent=2))


if __name__ == "__main__":
    main()
