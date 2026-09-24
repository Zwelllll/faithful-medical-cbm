"""Frozen Stage 11 selectors and execution; oracle APIs are explicitly separate."""
from dataclasses import dataclass
import numpy as np
from .engine import InterventionEngine, CONCEPT_ORDER

POLICIES = ("random_error_oracle", "confidently_wrong_oracle", "active")


@dataclass(frozen=True, slots=True)
class ActiveView:
    """Detached current-state counterfactuals; no targets or executor reference."""
    concept_order: tuple[str, ...]
    original_probabilities: tuple[float, ...]
    remaining: tuple[str, ...]
    force_zero: tuple[float, ...]
    force_one: tuple[float, ...]


def select_active(view: ActiveView) -> tuple[str, dict]:
    if view.concept_order != CONCEPT_ORDER or not view.remaining:
        raise ValueError("Frozen order and remaining concepts required")
    if len(set(view.remaining)) != len(view.remaining) or any(c not in CONCEPT_ORDER for c in view.remaining):
        raise ValueError("Invalid remaining concepts")
    q,p0,p1 = [np.asarray(v,dtype=float) for v in (view.original_probabilities,view.force_zero,view.force_one)]
    if any(a.shape!=(7,) or not np.isfinite(a).all() or ((a<0)|(a>1)).any() for a in (q,p0,p1)):
        raise ValueError("Expected seven finite probabilities per vector")
    uncertainty = 1-2*np.abs(q-.5)
    impact = np.abs(p1-p0)
    scores = uncertainty*impact
    candidates = [j for j,c in enumerate(CONCEPT_ORDER) if c in view.remaining]
    j = max(candidates,key=lambda i:scores[i])  # First maximum in frozen order.
    return CONCEPT_ORDER[j],dict(uncertainty=float(uncertainty[j]),downstream_impact=float(impact[j]),active_score=float(scores[j]))


def oracle_order(probabilities, targets, policy: str, rng=None) -> list[str]:
    """Ground-truth-aware, NON-DEPLOYABLE. Only original threshold errors qualify."""
    q,y = np.asarray(probabilities,dtype=float),np.asarray(targets)
    if q.shape!=(7,) or y.shape!=(7,) or not np.isfinite(q).all() or ((q<0)|(q>1)).any() or not np.isin(y,[0,1]).all():
        raise ValueError("Invalid oracle inputs")
    wrong = np.flatnonzero((q>=.5)!=y).tolist()
    if policy == "random_error_oracle":
        if rng is None: raise ValueError("Explicit seeded generator required")
        wrong = rng.permutation(wrong).tolist()
    elif policy == "confidently_wrong_oracle":
        wrong.sort(key=lambda j:(-abs(q[j]-.5),j))
    else:
        raise ValueError("Only explicit oracle policies may receive targets")
    return [CONCEPT_ORDER[j] for j in wrong]


def repetition_rng(seed: int, repetition: int):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,repetition])))


class PolicySession:
    """Execution state; selectors receive only detached ActiveView values."""
    def __init__(self, engine: InterventionEngine, case: str, mode: str):
        self.engine,self.case,self.mode = engine,case,mode
        self.initial = engine.intervene(case,mode,[],[])
        self.q = tuple(engine._probabilities[engine._case(case)])
        self.selected,self.values = [],[]
        self.errors_corrected = 0
        self.current = self.initial

    def view(self) -> ActiveView:
        # Recompute both forced values against the CURRENT cumulative intervention state.
        remaining = tuple(c for c in CONCEPT_ORDER if c not in self.selected)
        forced = [[],[]]
        for c in CONCEPT_ORDER:
            for v in (0,1):
                if c in self.selected:
                    forced[v].append(self.current['intervened_melanoma_probability'])
                else:
                    result = self.engine.intervene(self.case,self.mode,self.selected+[c],self.values+[v])
                    forced[v].append(result['intervened_melanoma_probability'])
        return ActiveView(CONCEPT_ORDER,self.q,remaining,tuple(forced[0]),tuple(forced[1]))

    def reveal_and_correct(self, concept: str) -> tuple[int,int]:
        if concept not in CONCEPT_ORDER or concept in self.selected:
            raise ValueError("Unknown or repeated query")
        # Selection is complete before this is called. Only this chosen target is read.
        revealed = self.engine.correct(self.case,self.mode,[concept])['intervention_values'][0]
        wrong = int((self.q[CONCEPT_ORDER.index(concept)]>=.5)!=revealed)
        self.selected.append(concept); self.values.append(revealed)
        self.errors_corrected += wrong
        self.current = self.engine.intervene(self.case,self.mode,self.selected,self.values)
        return revealed,wrong


def trajectory(engine: InterventionEngine, case: str, mode: str, policy: str,
               repetition: int = 0, rng=None, replay_order: list[str] | None = None) -> list[dict]:
    if policy not in POLICIES: raise ValueError("Unknown policy")
    session = PolicySession(engine,case,mode)
    order = None
    if policy != 'active':
        targets = [engine._truth[case][c] for c in CONCEPT_ORDER]  # Explicit oracle-only boundary.
        wrong = {c for c,q,y in zip(CONCEPT_ORDER,session.q,targets) if (q>=.5)!=y}
        order = replay_order if replay_order is not None else oracle_order(session.q,targets,policy,rng)
        if len(order)!=len(set(order)) or set(order)!=wrong:
            raise ValueError("Oracle order must contain every error exactly once")
    elif replay_order is not None:
        raise ValueError("Active selection must be recomputed on current state")
    rows=[]
    diagnosis = engine._diagnosis[case]
    for k in range(8):
        selected = None; revealed = None; wrong = None
        details = dict(uncertainty=None,downstream_impact=None,active_score=None)
        if k:
            if policy=='active':
                selected,details = select_active(session.view())
            elif k<=len(order):
                selected=order[k-1]
            if selected is not None:
                revealed,wrong=session.reveal_and_correct(selected)
        result = session.current
        rows.append(dict(case_num=case,validation_fold=result['validation_fold'],diagnosis_binary=diagnosis,
            model_type=mode,policy=policy,oracle_non_deployable=int(policy!='active'),repetition=repetition,query_budget=k,
            selected_concept=selected,selected_probability=session.q[CONCEPT_ORDER.index(selected)] if selected else None,
            revealed_ground_truth=revealed,selected_was_wrong=wrong,concepts_queried=len(session.selected),
            errors_corrected=session.errors_corrected,diagnosis_score=result['intervened_diagnosis_score'],
            melanoma_probability=result['intervened_melanoma_probability'],predicted_diagnosis=result['intervened_predicted_diagnosis'],
            diagnosis_changed=int(result['intervened_predicted_diagnosis']!=session.initial['original_predicted_diagnosis']),
            diagnosis_correct=int(result['intervened_predicted_diagnosis']==diagnosis),
            original_diagnosis_correct=int(session.initial['original_predicted_diagnosis']==diagnosis),**details))
    return rows
