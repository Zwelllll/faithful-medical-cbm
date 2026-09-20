"""Load central configuration without accessing datasets."""
from pathlib import Path
import tomllib
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Read TOML; resolve relative data paths from the config's project root.

    Configuration files live directly inside the project's configs directory.
    This function reads configuration only and never creates data directories.
    """
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    for key in ("image_size", "batch_size", "num_folds"):
        value = config["experiment"][key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    seed = config["reproducibility"]["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(config["reproducibility"]["deterministic"]) is not bool:
        raise ValueError("deterministic must be a boolean")
    names = config["concepts"]["names"]
    if (len(names) != 7 or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != 7):
        raise ValueError("Exactly seven distinct concept names are required")
    config["paths"] = {
        name: (config_path.parent.parent / value).resolve()
        for name, value in config["paths"].items()
    }
    return config
