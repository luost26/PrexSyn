import abc
from typing import TYPE_CHECKING

import torch
from torch import nn

from .base import Embedding


class BasePropertyEmbedder(nn.Module, abc.ABC):
    if TYPE_CHECKING:

        def __call__(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> Embedding: ...
