"""Direct dermoscopic image -> one melanoma logit; no concept pathway."""
import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class BlackBoxEfficientNet(nn.Module):
    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.network = efficientnet_b0(weights=weights)
        self.network.classifier[1] = nn.Linear(self.network.classifier[1].in_features, 1)
        self.pretrained = pretrained
        self.unfreeze_last_blocks = 0
        self.set_trainable_blocks(0)

    def set_trainable_blocks(self, count: int) -> None:
        """Head always trains; count refers to torchvision feature children (including final conv)."""
        if type(count) is not int or not 0 <= count <= len(self.network.features):
            raise ValueError("Invalid number of trainable feature blocks")
        self.unfreeze_last_blocks = count
        boundary = len(self.network.features) - count
        for index, block in enumerate(self.network.features):
            block.requires_grad_(index >= boundary)
        self.network.classifier.requires_grad_(True)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen feature blocks must not update BatchNorm buffers or stochastic depth.
        if mode:
            for block in self.network.features[:len(self.network.features) - self.unfreeze_last_blocks]:
                block.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image).squeeze(-1)
