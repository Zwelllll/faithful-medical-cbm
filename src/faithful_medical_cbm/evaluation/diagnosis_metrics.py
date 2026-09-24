"""Fixed-threshold binary diagnosis metrics and probability calibration."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def probability_arrays(targets, probabilities) -> tuple[np.ndarray, np.ndarray]:
    y, p = np.asarray(targets), np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or not len(y) or p.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected matching nonempty binary labels and probabilities")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and in [0,1]")
    return y, p


def expected_calibration_error(targets, probabilities, bins: int = 10) -> dict:
    """Positive-class ECE: sum n_bin/N * |mean(p)-mean(y)|.

    Equal-width bins [j/K,(j+1)/K), except the last includes 1. Empty bins
    contribute zero. Boundaries go in the bin to their right (p=1 in last).
    """
    y, p = probability_arrays(targets, probabilities)
    if type(bins) is not int or bins < 1:
        raise ValueError("ECE bins must be a positive integer")
    edges = np.arange(bins + 1, dtype=np.float64) / bins
    index = np.searchsorted(edges[1:-1], p, side="right")
    details, ece = [], 0.0
    for j in range(bins):
        selected = index == j
        count = int(selected.sum())
        mean_p = float(p[selected].mean()) if count else None
        fraction_positive = float(y[selected].mean()) if count else None
        contribution = count / len(y) * abs(mean_p - fraction_positive) if count else 0.0
        ece += contribution
        details.append({"lower": float(edges[j]), "upper": float(edges[j+1]), "count": count,
                        "mean_probability": mean_p, "fraction_positive": fraction_positive,
                        "weighted_absolute_gap": contribution})
    return {"ece": ece, "bins": details,
            "rule": "10 equal-width positive-class probability bins; left-inclusive, right-exclusive except p=1 in last" if bins == 10 else
                    f"{bins} equal-width positive-class probability bins; left-inclusive, right-exclusive except p=1 in last"}


def diagnosis_metrics(targets, scores, probabilities, *, bins: int = 10) -> dict:
    y, p = probability_arrays(targets, probabilities)
    z = np.asarray(scores, dtype=np.float64)
    if z.shape != y.shape or not np.isfinite(z).all() or set(y.tolist()) != {0, 1}:
        raise ValueError("Finite scores and both diagnosis classes required")
    predicted = p >= 0.5
    tp = int(((y == 1) & predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    calibration = expected_calibration_error(y, p, bins)
    return {"total": len(y), "positive": int(y.sum()), "negative": int(len(y)-y.sum()),
            "auroc": float(roc_auc_score(y, z)),
            "macro_f1": float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0)),
            "accuracy": float((predicted == y).mean()), "sensitivity": tp/(tp+fn),
            "specificity": tn/(tn+fp), "brier": float(np.mean((p-y)**2)), "ece": calibration["ece"],
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": 0.5}
