"""Enumerate single-concept effects, not intervention-policy experiments."""
from __future__ import annotations
import argparse
from pathlib import Path
import platform
import importlib.metadata
import subprocess
import numpy as np
from .engine import InterventionEngine, CONCEPT_ORDER
from ..training.train_sequential_soft import csv_bytes
from ..training.train_binary_cbm import verify_hashes, write_json
from ..evaluation.assemble_oof import sha

EFFECT_FIELDS = ["original_diagnosis_score","intervened_diagnosis_score","original_melanoma_probability",
                 "intervened_melanoma_probability","signed_probability_change","absolute_probability_change",
                 "original_predicted_diagnosis","intervened_predicted_diagnosis","diagnosis_prediction_changed"]
BASE_COLUMNS = ["case_num","validation_fold","diagnosis_binary","concept_name","original_concept_value"]
CORRECTION_COLUMNS = BASE_COLUMNS+["ground_truth_concept_value","concept_was_wrong"]+EFFECT_FIELDS
FORCED_COLUMNS = BASE_COLUMNS+["forced_value"]+EFFECT_FIELDS


def tables(engine: InterventionEngine, mode: str) -> tuple[list[dict], list[dict]]:
    corrections, forced = [],[]
    for case in engine.ids:
        for j,concept in enumerate(CONCEPT_ORDER):
            result = engine.correct(case,mode,[concept])
            truth = result["intervention_values"][0]
            base = {"case_num":case,"validation_fold":result["validation_fold"],
                    "diagnosis_binary":engine._diagnosis[case],"concept_name":concept,
                    "original_concept_value":result["original_concept_inputs"][j]}
            corrections.append({**base,"ground_truth_concept_value":truth,
                "concept_was_wrong":int((base["original_concept_value"]>=.5)!=truth),
                **{k:result[k] for k in EFFECT_FIELDS}})
            for value in (0,1):
                effect = engine.intervene(case,mode,[concept],[value])
                forced.append({**base,"forced_value":value,**{k:effect[k] for k in EFFECT_FIELDS}})
    if len(corrections)!=658*7 or len(forced)!=658*7*2:
        raise ValueError("Unexpected case-concept coverage")
    if len({(r['case_num'],r['concept_name']) for r in corrections})!=len(corrections):
        raise ValueError("Duplicate correction pair")
    if len({(r['case_num'],r['concept_name'],r['forced_value']) for r in forced})!=len(forced):
        raise ValueError("Duplicate forced-value pair")
    return corrections,forced


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count":0,"mean":None,"min":None,"q25":None,"median":None,"q75":None,"q95":None,"max":None}
    return {"count":len(values),"mean":float(np.mean(values)),
            **dict(zip(("min","q25","median","q75","q95","max"),map(float,np.quantile(values,[0,.25,.5,.75,.95,1]))))}


def summarize(rows: list[dict]) -> dict:
    wrong = [r for r in rows if r["concept_was_wrong"]]
    flips = sum(r["diagnosis_prediction_changed"] for r in wrong)
    return {"case_concept_pairs":len(rows),"incorrect_concepts":len(wrong),
            "incorrect_fraction":len(wrong)/len(rows),
            "absolute_effect_all_corrections":distribution([r["absolute_probability_change"] for r in rows]),
            "absolute_effect_wrong_concepts":distribution([r["absolute_probability_change"] for r in wrong]),
            "wrong_corrections_flipping_diagnosis":flips,
            "wrong_correction_flip_fraction":flips/len(wrong) if wrong else None}


def run(config_path: Path, output_dir: Path | None = None) -> dict:
    engine = InterventionEngine(config_path)
    destination = (output_dir or engine.root/engine.config["output_dir"]).resolve()
    for path in engine.protected_paths:
        path = path.resolve()
        if destination==path or path in destination.parents or destination in path.parents:
            raise ValueError("Output overlaps protected inputs")
    if destination.exists() and any(p.name!=".gitattributes" for p in destination.iterdir()):
        raise FileExistsError("Intervention outputs exist; refusing overwrite")
    destination.mkdir(parents=True,exist_ok=True)
    results = {}
    for mode in ("soft","hard"):
        corrections,forced = tables(engine,mode)
        for name,rows,columns in ((f"{mode}_single_concept_ground_truth_corrections.csv",corrections,CORRECTION_COLUMNS),
                                  (f"{mode}_forced_value_counterfactuals.csv",forced,FORCED_COLUMNS)):
            with (destination/name).open("xb") as stream:
                stream.write(csv_bytes(rows,columns))
        # Persist raw outputs before computing descriptive summaries.
        results[mode] = {**summarize(corrections),"forced_value_rows":len(forced),
                         "per_concept":{c:summarize([r for r in corrections if r["concept_name"]==c]) for c in CONCEPT_ORDER}}
    summary = {"stage":10,"development_cases":658,"concept_order":list(CONCEPT_ORDER),"models":results,
               "baseline_reproduction":engine.reproduction,"reproduction_atol":1e-12,"reproduction_rtol":0,
               "concept_error_definition":"Original concept probability >=0.5 disagrees with frozen binary target",
               "interpretation":"Descriptive model-input effects, not causal clinical effects or policy comparisons. "
                   "Soft corrections also snap already-correct thresholded concepts to 0/1. A diagnosis flip is not necessarily an improvement. "
                   "Existing non-nested development evaluation and unverified patient independence limitations remain."}
    write_json(destination/"summary.json",summary)
    verify_hashes(engine.hashes)
    source_files = [Path(__file__).resolve(),Path(__file__).with_name("engine.py").resolve()]
    source_files += [engine.root/p for p in ("src/faithful_medical_cbm/training/train_binary_cbm.py",
        "src/faithful_medical_cbm/training/train_sequential_soft.py","src/faithful_medical_cbm/evaluation/assemble_oof.py",
        "src/faithful_medical_cbm/models/binary_cbm.py","src/faithful_medical_cbm/models/soft_cbm.py")]
    git = subprocess.run(["git","rev-parse","HEAD"],cwd=engine.root,capture_output=True,text=True,check=False)
    status = subprocess.run(["git","status","--porcelain"],cwd=engine.root,capture_output=True,text=True,check=False)
    integrity = {"status":"passed","development_cases":658,"locked_test_overlap":0,
                 "baseline_reproduction":engine.reproduction,"fold_specific_heads_only":True,
                 "validation_ids_excluded_from_training":True,"full_development_heads_used":False,
                 "concept_order":list(CONCEPT_ORDER),"nonselected_inputs_unchanged":True,
                 "selection_view_contains_ground_truth":False,"policy_experiments_run":False,
                 "models_retrained":False,"images_accessed":False,"thresholds_optimized":False,
                 "source_artifacts_unchanged":True,"input_sha256":engine.hashes,
                 "output_sha256":{p.name:sha(p.read_bytes()) for p in destination.iterdir() if p.suffix in (".csv",".json")},
                 "source_sha256":{str(p.relative_to(engine.root)):sha(p.read_bytes()) for p in source_files},
                 "git_commit":git.stdout.strip() if git.returncode==0 else None,
                 "git_working_tree_dirty":bool(status.stdout.strip()),"python":platform.python_version(),
                 "packages":{p:importlib.metadata.version(p) for p in ("numpy","scipy","scikit-learn")},
                 "schemas":{"correction":CORRECTION_COLUMNS,"forced_value":FORCED_COLUMNS}}
    write_json(destination/"integrity.json",integrity)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/intervention_engine.json"))
    args = parser.parse_args()
    result = run(args.config)
    print({m:{k:v for k,v in r.items() if k!='per_concept'} for m,r in result['models'].items()})


if __name__=="__main__":
    main()
