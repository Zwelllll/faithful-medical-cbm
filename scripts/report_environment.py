"""Print runtime provenance as JSON without touching research data."""
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


def collect_environment() -> dict:
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
        "cuda": None,
        "cuda_available": False,
        "cudnn": None,
        "gpus": [],
    }
    for package in ("numpy", "torch", "torchvision", "timm", "scikit-learn", "pandas", "Pillow"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    try:
        import torch
        report["pytorch"] = torch.__version__
        report["cuda"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        report["cudnn"] = torch.backends.cudnn.version()
        report["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:
        report["torch_runtime_error"] = f"{type(exc).__name__}: {exc}"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=False,
        )
        report["git_commit"] = result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        report["git_commit"] = None
    return report


if __name__ == "__main__":
    print(json.dumps(collect_environment(), indent=2))
