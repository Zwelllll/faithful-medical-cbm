"""Strict, separate feature allowlists for hard and oracle concept heads."""
import numpy as np
from ..evaluation.assemble_oof import CONCEPT_ORDER, binary
from .soft_cbm import extract_features

HARD_COLUMNS = tuple(f"{c}_hard" for c in CONCEPT_ORDER)
ORACLE_COLUMNS = tuple(f"{c}_target" for c in CONCEPT_ORDER)


def binary_features(rows: list[dict], mode: str) -> np.ndarray:
    if mode == "sequential_hard":
        return (extract_features(rows) >= .5).astype(np.float64)
    if mode == "oracle":
        return np.array([[binary(row[c]) for c in ORACLE_COLUMNS] for row in rows], dtype=np.float64)
    raise ValueError("Expected sequential_hard or oracle")


def feature_columns(mode: str) -> tuple[str, ...]:
    if mode == "sequential_hard":
        return HARD_COLUMNS
    if mode == "oracle":
        return ORACLE_COLUMNS
    raise ValueError("Unknown binary concept model")
