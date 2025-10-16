from collections.abc import Mapping, Sequence
from typing import TypeAlias, TypedDict

import torch

EmbedderName: TypeAlias = str
EmbedderParams: TypeAlias = Mapping[str, torch.Tensor]


class SynthesisRepr(TypedDict):
    token_types: torch.Tensor
    bb_indices: torch.Tensor
    rxn_indices: torch.Tensor


PropertyRepr: TypeAlias = Sequence[Mapping[EmbedderName, EmbedderParams]]


class SynthesisTrainingBatch(TypedDict):
    synthesis_repr: SynthesisRepr
    property_repr: PropertyRepr
