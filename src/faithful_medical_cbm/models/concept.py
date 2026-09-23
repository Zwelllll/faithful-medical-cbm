"""One named concept head; reuse the baseline's EfficientNet transfer mechanics."""
from .black_box import BlackBoxEfficientNet


class ConceptEfficientNet(BlackBoxEfficientNet):
    """Image -> [B] atypical-pigment-network logits, with no diagnosis head.

    The parent supplies only the B0 backbone, single linear output and freezing
    mechanics; task targets/losses live separately. Ordered concept_names are saved
    in checkpoints so future multi-head work can extend the explicit output schema.
    Stage 5 deliberately exposes no multi-head constructor option.
    """
    concept_names = ("atypical_pigment_network",)
