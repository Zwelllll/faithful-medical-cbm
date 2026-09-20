"""Explicit random-state initialization; no import-time side effects."""
import os
import random


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch and available CUDA generators.

    Call before creating models or initializing CUDA. Deterministic operations
    are enforced when requested; unsupported operations raise an error.
    Exact results are not guaranteed across hardware or library versions.
    PYTHONHASHSEED must be set before launching Python if hash order matters.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
