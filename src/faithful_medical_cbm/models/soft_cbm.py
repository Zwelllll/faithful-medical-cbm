"""Probability-only linear diagnosis head; no torch, images or CNN features."""
from __future__ import annotations
import inspect
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from ..evaluation.assemble_oof import CONCEPT_ORDER

FEATURE_COLUMNS = tuple(f"{name}_probability" for name in CONCEPT_ORDER)


def extract_features(rows: list[dict], columns=FEATURE_COLUMNS) -> np.ndarray:
    if tuple(columns) != FEATURE_COLUMNS:
        raise ValueError("Soft features must be the seven frozen probability columns in order")
    # Explicit allowlist: never inspect logits, concept truths, diagnosis or metadata.
    features = np.array([[float(row[name]) for name in FEATURE_COLUMNS] for row in rows], dtype=np.float64)
    if features.shape != (len(rows), 7) or not len(rows):
        raise ValueError("Expected a nonempty N-by-7 probability matrix")
    if not np.isfinite(features).all() or np.any((features < 0) | (features > 1)):
        raise ValueError("Concept probabilities must be finite and in [0,1]")
    return features


def constructor_kwargs(settings: dict) -> dict:
    """Same L2 objective on supported old/new sklearn APIs; record actual kwargs."""
    if settings.get("penalty") != "l2" or settings.get("solver") != "lbfgs":
        raise ValueError("The soft head uses L2 logistic regression with lbfgs")
    kwargs = dict(settings)
    penalty_parameter = inspect.signature(LogisticRegression).parameters.get("penalty")
    if penalty_parameter is None or penalty_parameter.default == "deprecated":
        kwargs.pop("penalty")
        kwargs["l1_ratio"] = 0.0
    return kwargs


def fit_head(features: np.ndarray, targets: np.ndarray, settings: dict) -> LogisticRegression:
    if features.ndim != 2 or features.shape[1] != 7 or len(features) != len(targets):
        raise ValueError("Expected seven probability features and matching labels")
    if not np.isfinite(features).all() or np.any((features < 0) | (features > 1)):
        raise ValueError("Invalid soft features")
    if targets.ndim != 1 or set(targets.tolist()) != {0, 1}:
        raise ValueError("Binary diagnosis training requires both classes")
    model = LogisticRegression(**constructor_kwargs(settings))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(features, targets)
    if model.classes_.tolist() != [0, 1] or not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise ValueError("Invalid fitted diagnosis parameters")
    return model


def parameters(model: LogisticRegression) -> dict:
    return {"feature_columns": list(FEATURE_COLUMNS), "classes": model.classes_.tolist(),
            "intercept": float(model.intercept_[0]), "coefficients": model.coef_[0].tolist(),
            "coefficients_by_feature": dict(zip(FEATURE_COLUMNS, model.coef_[0].tolist())),
            "iterations": model.n_iter_.tolist(),
            "coefficient_interpretation": "Descriptive conditional model parameters, not causal concept importance"}
