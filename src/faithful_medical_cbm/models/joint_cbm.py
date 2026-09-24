"""Strict seven-concept joint bottlenecks; no image-feature diagnosis bypass."""
import torch
from torch import nn
from .seven_concept import SevenConceptEfficientNet, CONCEPT_ORDER

MODEL_TYPES = ('joint_soft','joint_hard_ste')


def concept_representation(logits: torch.Tensor, model_type: str) -> torch.Tensor:
    if logits.ndim!=2 or logits.shape[1]!=7 or model_type not in MODEL_TYPES:
        raise ValueError('Expected [B,7] logits and a supported joint type')
    soft=torch.sigmoid(logits)
    if model_type=='joint_soft': return soft
    hard=(soft>=.5).to(soft.dtype)
    # Exactly binary forward; derivative is sigmoid's derivative, NOT threshold's.
    return hard + (soft-soft.detach())


class JointCBM(nn.Module):
    concept_names=CONCEPT_ORDER

    def __init__(self, model_type: str, *, pretrained: bool=True):
        super().__init__()
        if model_type not in MODEL_TYPES: raise ValueError('Unknown joint model type')
        self.model_type=model_type
        self.concept_predictor=SevenConceptEfficientNet(pretrained=pretrained)
        self.diagnosis_head=nn.Linear(7,1,bias=True)
        self.set_trainable_blocks(0)

    @property
    def pretrained(self): return self.concept_predictor.pretrained

    @property
    def unfreeze_last_blocks(self): return self.concept_predictor.unfreeze_last_blocks

    def set_trainable_blocks(self, count: int) -> None:
        self.concept_predictor.set_trainable_blocks(count)
        self.diagnosis_head.requires_grad_(True)

    def forward(self, image: torch.Tensor) -> dict:
        logits=self.concept_predictor(image)
        representation=concept_representation(logits,self.model_type)
        # This is the ONLY diagnosis route: seven concepts and an intercept.
        diagnosis=self.diagnosis_head(representation).squeeze(-1)
        return {'concept_logits':logits,'concept_probabilities':torch.sigmoid(logits),
                'diagnosis_inputs':representation,'diagnosis_logits':diagnosis}
