from collections.abc import Mapping, Sequence
from typing import TypeAlias, TypedDict

import torch

EmbedderName: TypeAlias = str
EmbedderParams: TypeAlias = Mapping[str, torch.Tensor]


class SynthesisTrainingBatch(TypedDict):
    token_types: torch.Tensor
    bb_indices: torch.Tensor
    rxn_indices: torch.Tensor
    property_repr: Sequence[Mapping[EmbedderName, EmbedderParams]]
