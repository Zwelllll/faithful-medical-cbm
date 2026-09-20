"""Image-only preprocessing; no model/weights are instantiated or downloaded."""
from typing import Any

from torchvision import transforms as T


def build_transform(config: dict[str, Any], *, training: bool) -> T.Compose:
    """Train: mild stochastic augmentation. Eval: deterministic full-frame resize."""
    size = config["experiment"]["image_size"]
    if type(size) is not int or size <= 0:
        raise ValueError("image_size must be a positive integer")
    normalization = config["normalization"]
    if len(normalization["mean"]) != 3 or len(normalization["std"]) != 3 or any(
            value <= 0 for value in normalization["std"]):
        raise ValueError("RGB normalization requires three means and positive standard deviations")
    steps = []
    if training:
        aug = config["augmentation"]
        for name in ("horizontal_flip_probability", "vertical_flip_probability"):
            if not 0 <= aug[name] <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("crop_scale", "crop_ratio"):
            bounds = aug[name]
            if len(bounds) != 2 or not 0 < bounds[0] <= bounds[1]:
                raise ValueError(f"Invalid {name}")
        if aug["crop_scale"][1] > 1 or aug["rotation_degrees"] < 0:
            raise ValueError("Crop scale must be <= 1 and rotation nonnegative")
        steps.extend([
            T.RandomResizedCrop(size, scale=tuple(aug["crop_scale"]), ratio=tuple(aug["crop_ratio"]),
                                interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.RandomHorizontalFlip(aug["horizontal_flip_probability"]),
            T.RandomVerticalFlip(aug["vertical_flip_probability"]),
            T.RandomRotation(aug["rotation_degrees"], interpolation=T.InterpolationMode.BILINEAR),
            T.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"], saturation=aug["saturation"]),
        ])
    else:
        steps.append(T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC, antialias=True))
    steps.extend([T.ToTensor(), T.Normalize(normalization["mean"], normalization["std"])])
    return T.Compose(steps)
