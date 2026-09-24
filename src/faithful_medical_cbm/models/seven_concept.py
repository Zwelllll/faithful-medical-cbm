"""Seven ordered concept logits sharing one EfficientNet-B0 backbone."""
import torch
from torch import nn
from .concept import ConceptEfficientNet

CONCEPT_ORDER = (
    "atypical_pigment_network", "regression_structures_present",
    "irregular_pigmentation", "blue_whitish_veil_present",
    "atypical_vascular_structures", "irregular_dots_and_globules", "irregular_streaks",
)


class SevenConceptEfficientNet(ConceptEfficientNet):
    """A linear 1280-to-7 layer is seven independent binary heads, not a softmax."""
    concept_names = CONCEPT_ORDER

    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__(pretrained=pretrained)
        self.network.classifier[1] = nn.Linear(self.network.classifier[1].in_features, 7)
        self.set_trainable_blocks(0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)
