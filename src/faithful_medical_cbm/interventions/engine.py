"""Read-only cross-fitted concept intervention executor. No fitting or policies."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
import json
import numbers
import numpy as np
from scipy.special import expit
from ..evaluation.assemble_oof import CONCEPT_ORDER, read_csv, sha, unique_ids, binary
from ..models.soft_cbm import FEATURE_COLUMNS
from ..models.binary_cbm import HARD_COLUMNS
from ..training.train_binary_cbm import load_inputs, verify_hashes


@dataclass(frozen=True, slots=True)
class SelectionView:
    """Detached policy input: no truths, diagnosis targets, rows or executor reference."""
    case_num: str
    model_type: str
    concept_order: tuple[str, ...]
    concept_probabilities: tuple[float, ...]
    original_inputs: tuple[float, ...]
    original_melanoma_probability: float
    forced_zero_probabilities: tuple[float, ...]
    forced_one_probabilities: tuple[float, ...]


def validate_head(record: dict, mode: str, fold: int, ids: list[str], folds: np.ndarray,
                  oof_hash: str, settings: dict) -> None:
    expected = list(FEATURE_COLUMNS if mode == "soft" else HARD_COLUMNS)
    label = "CROSS-FITTED DEVELOPMENT SOFT CBM HEAD" if mode == "soft" else "CROSS-FITTED DEVELOPMENT HARD CBM HEAD"
    if record["label"] != label or record["validation_fold"] != fold or record["used_for_development_metrics"] is not True:
        raise ValueError("Only the matching cross-fitted head is allowed; full-development heads rejected")
    if record["feature_columns"] != expected or record["classes"] != [0,1]:
        raise ValueError("Head feature order/classes differ")
    if record["threshold"] != .5 or record["preprocessing"] != "none" or record["logistic_regression"] != settings:
        raise ValueError("Head configuration differs from frozen protocol")
    if record["provenance"]["oof_sha256"] != oof_hash:
        raise ValueError("Head OOF provenance differs")
    train = unique_ids(record["training_ids"], "head training IDs")
    val = unique_ids(record["validation_ids"], "head validation IDs")
    expected_val = {i for i,f in zip(ids,folds) if f==fold}
    if val != expected_val or train != set(ids)-expected_val or train & val:
        raise ValueError("Head training/validation fold isolation failure")
    weights = np.asarray(record["coefficients"],dtype=float)
    if weights.shape != (7,) or not np.isfinite(weights).all() or not np.isfinite(record["intercept"]):
        raise ValueError("Invalid head parameters")
    if list(record["coefficients_by_feature"]) != expected or list(record["coefficients_by_feature"].values()) != record["coefficients"]:
        raise ValueError("Coefficient order mismatch")


class InterventionEngine:
    """Uses only frozen development records and eight saved cross-fitted LR heads."""

    def __init__(self, config_path: Path):
        config_path = config_path.resolve()
        self.config = json.loads(config_path.read_bytes())
        if self.config["stage"] != 10 or self.config["reproduction_atol"] != 1e-12 or self.config["reproduction_rtol"] != 0:
            raise ValueError("Stage 10 requires absolute tolerance 1e-12 and zero relative tolerance")
        self.root = config_path.parent.parent
        cfg9,data,rows,_ = load_inputs(self.root/self.config["stage9_config"])
        self.ids = tuple(data["ids"])
        self._index = {identifier:i for i,identifier in enumerate(self.ids)}
        self._folds = data["folds"].copy()
        self._probabilities = data["features"].copy()
        self._truth = {r["case_num"]: {c:binary(r[c+"_target"]) for c in CONCEPT_ORDER} for r in rows}
        self._diagnosis = dict(zip(self.ids,map(int,data["labels"])))
        self.hashes = dict(data["hashes"])
        self.hashes[str(config_path)] = sha(config_path.read_bytes())
        self._heads = {}
        self.reproduction = {}
        self.protected_paths = [self.root/"artifacts/cbm",self.root/"artifacts/oof",
                                *(data["frozen"]["config"]["paths"][k] for k in ("raw","processed","splits"))]
        for mode,folder in (("soft",cfg9["soft_outputs"]),("hard",self.config["hard_outputs"])):
            path = self.root/folder
            integrity_bytes = (path/"integrity.json").read_bytes()
            integrity = json.loads(integrity_bytes)
            if integrity["status"] != "passed":
                raise ValueError("Source integrity did not pass")
            source_hashes = {str(path/name): h for name,h in integrity["output_sha256"].items()}
            verify_hashes(source_hashes)
            self.hashes.update(source_hashes)
            self.hashes[str(path/"integrity.json")] = sha(integrity_bytes)
            if integrity["provenance"]["oof_sha256"] != data["oof_sha256"]:
                raise ValueError("Source OOF differs")
            for fold in range(4):
                record = json.loads((path/f"fold_{fold}_model.json").read_bytes())
                validate_head(record,mode,fold,list(self.ids),self._folds,data["oof_sha256"],data["config"]["logistic_regression"])
                self._heads[mode,fold] = (float(record["intercept"]),np.array(record["coefficients"],dtype=float))
            saved = read_csv((path/"cross_fitted_predictions.csv").read_bytes())
            if unique_ids([r["case_num"] for r in saved],"saved baseline") != set(self.ids):
                raise ValueError("Saved baseline coverage differs")
            errors = []
            for row in saved:
                i = self._case(row["case_num"])
                baseline = self.intervene(row["case_num"],mode,[],[])
                if int(row["validation_fold"]) != self._folds[i] or binary(row["diagnosis_binary"]) != self._diagnosis[row["case_num"]]:
                    raise ValueError("Saved baseline fold/label differs")
                score_error = abs(baseline["original_diagnosis_score"]-float(row["decision_score"]))
                prob_error = abs(baseline["original_melanoma_probability"]-float(row["melanoma_probability"]))
                if not np.isfinite([score_error,prob_error]).all() or max(score_error,prob_error)>1e-12 or baseline["original_predicted_diagnosis"] != binary(row["predicted_diagnosis"]):
                    raise ValueError("Baseline prediction reproduction failed")
                errors.append((score_error,prob_error))
            self.reproduction[mode] = {"cases":len(saved),"max_score_absolute_error":max(e[0] for e in errors),
                                       "max_probability_absolute_error":max(e[1] for e in errors)}
        verify_hashes(self.hashes)

    def _case(self, case_num: str) -> int:
        if not isinstance(case_num,str) or case_num not in self._index:
            raise ValueError("Only frozen development case identifiers are allowed")
        return self._index[case_num]

    def _inputs(self, case_num: str, model_type: str) -> tuple[np.ndarray, tuple]:
        i = self._case(case_num)
        if model_type not in ("soft","hard"):
            raise ValueError("Model type must be soft or hard")
        x = self._probabilities[i].copy()
        if model_type == "hard":
            x = (x>=.5).astype(float)
        return x,self._heads[model_type,int(self._folds[i])]

    @staticmethod
    def _selected(concepts: Sequence[str]) -> tuple[str, ...]:
        if isinstance(concepts,str):
            raise ValueError("Supply an ordered sequence of distinct concept names")
        names = tuple(concepts)
        if any(not isinstance(c,str) or c not in CONCEPT_ORDER for c in names) or len(names)!=len(set(names)):
            raise ValueError("Unknown or duplicate concept")
        return names

    def intervene(self, case_num: str, model_type: str, concepts: Sequence[str], values: Sequence[int]) -> dict:
        """Forced-value operation: supplied binary values, no ground-truth access."""
        names = self._selected(concepts)
        values = tuple(values)
        if len(names)!=len(values) or any(not isinstance(v,numbers.Real) or not np.isfinite(v) or v not in (0,1) for v in values):
            raise ValueError("Exactly one finite binary numeric value per selected concept required")
        x,(intercept,weights) = self._inputs(case_num,model_type)
        changed = x.copy()
        for name,value in zip(names,values):
            changed[CONCEPT_ORDER.index(name)] = value
        original_score = float(intercept + x @ weights)
        changed_score = float(intercept + changed @ weights)
        original_p,changed_p = float(expit(original_score)),float(expit(changed_score))
        return {"case_num":case_num,"validation_fold":int(self._folds[self._case(case_num)]),"model_type":model_type,
                "operation":"forced_value","selected_concepts":list(names),"intervention_values":list(map(int,values)),
                "original_concept_inputs":x.tolist(),"intervened_concept_inputs":changed.tolist(),
                "original_diagnosis_score":original_score,"intervened_diagnosis_score":changed_score,
                "original_melanoma_probability":original_p,"intervened_melanoma_probability":changed_p,
                "signed_probability_change":changed_p-original_p,"absolute_probability_change":abs(changed_p-original_p),
                "original_predicted_diagnosis":int(original_p>=.5),"intervened_predicted_diagnosis":int(changed_p>=.5),
                "diagnosis_prediction_changed":int((original_p>=.5)!=(changed_p>=.5))}

    def correct(self, case_num: str, model_type: str, concepts: Sequence[str]) -> dict:
        """Read ONLY already-selected targets, then execute their replacement."""
        names = self._selected(concepts)
        self._inputs(case_num,model_type)  # Validate case/model before reading any target.
        result = self.intervene(case_num,model_type,names,[self._truth[case_num][c] for c in names])
        result["operation"] = "ground_truth_correction"
        return result

    def trajectory(self, case_num: str, model_type: str, concept_order: Sequence[str]) -> list[dict]:
        names = self._selected(concept_order)
        return [{"k":k,**self.correct(case_num,model_type,names[:k])} for k in range(len(names)+1)]

    def selection_view(self, case_num: str, model_type: str) -> SelectionView:
        original = self.intervene(case_num,model_type,[],[])
        forced = [[self.intervene(case_num,model_type,[c],[v])["intervened_melanoma_probability"] for c in CONCEPT_ORDER] for v in (0,1)]
        return SelectionView(case_num,model_type,CONCEPT_ORDER,tuple(self._probabilities[self._case(case_num)]),
                             tuple(original["original_concept_inputs"]),original["original_melanoma_probability"],
                             tuple(forced[0]),tuple(forced[1]))

    def select_then_correct(self, case_num: str, model_type: str,
                            selector: Callable[[SelectionView], Sequence[str]]) -> dict:
        """No built-in policy: callback returns selections before target lookup."""
        selected = self._selected(selector(self.selection_view(case_num,model_type)))
        return self.correct(case_num,model_type,selected)
