import abc
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from torch import nn

from prexsyn.data.struct import PropertyRepr
from prexsyn.models.embeddings import BasePropertyEmbedder, Embedding
from prexsyn.queries import Condition
from prexsyn_engine.featurizer.fingerprint import FingerprintFeaturizer
from prexsyn_engine.fingerprints import get_fingerprints
from prexsyn_engine.synthesis import Synthesis

from .base import BasePropertyDef


class FingerprintEmbedder(BasePropertyEmbedder, nn.Module):
    def __init__(self, fingerprint_dim: int, embedding_dim: int, num_tokens: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.mlp = nn.Sequential(
            nn.Linear(fingerprint_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim * num_tokens),
        )

    def forward(self, fingerprint: torch.Tensor) -> Embedding:
        h = self.mlp(fingerprint)
        h = h.reshape(*h.shape[:-1], self.num_tokens, -1)
        return Embedding(
            embedding=h,
            padding_mask=Embedding.create_full_padding_mask(h),
        )


class FingerprintCondition(Condition):
    def __init__(
        self,
        fingerprint: torch.Tensor,
        property_def: "StandardFingerprintProperty",
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight=weight)
        if not (fingerprint.ndim == 2 and fingerprint.shape[0] == 1):
            raise ValueError("FingerprintCondition expects a fingerprint tensor of shape (1, fp_dim).")
        self.fingerprint = fingerprint
        self._property_def = property_def

    def get_property_repr(self) -> PropertyRepr:
        return {self._property_def.name: {"fingerprint": self.fingerprint}}

    def score(self, synthesis: Synthesis, product: Chem.Mol) -> float:
        fp_product = self._property_def.evaluate_mol(product)["fingerprint"].squeeze(0).cpu().numpy()
        fp_query = self.fingerprint.squeeze(0).cpu().numpy()

        intersection = np.minimum(fp_product, fp_query).sum()
        union = np.maximum(fp_product, fp_query).sum()
        if union == 0:
            return 1.0
        return float(intersection / union)

    def __repr__(self) -> str:
        return f"{self._property_def.name}={self.fingerprint[0].nonzero(as_tuple=True)[0]}"


class StandardFingerprintProperty(BasePropertyDef, abc.ABC):
    def __init__(self, name: str, num_embedding_tokens: int = 4) -> None:
        super().__init__()
        self._name = name
        self._num_embedding_tokens = num_embedding_tokens

    @property
    @abc.abstractmethod
    def fp_type(self) -> str: ...

    @property
    @abc.abstractmethod
    def fp_dim(self) -> int: ...

    @property
    def name(self) -> str:
        return self._name

    def get_featurizer(self) -> FingerprintFeaturizer:
        return FingerprintFeaturizer(name=self._name, fp_type=self.fp_type)

    def get_embedder(self, model_dim: int) -> FingerprintEmbedder:
        return FingerprintEmbedder(
            fingerprint_dim=self.fp_dim,
            embedding_dim=model_dim,
            num_tokens=self._num_embedding_tokens,
        )

    def evaluate_mol(self, mol: Chem.Mol | list[Chem.Mol]) -> Mapping[str, torch.Tensor]:
        mol = [mol] if isinstance(mol, Chem.Mol) else mol
        return {"fingerprint": torch.from_numpy(get_fingerprints(mol, fp_type=self.fp_type))}

    def eq(
        self,
        mol: Chem.Mol | np.ndarray[tuple[int], np.dtype[Any]] | torch.Tensor,
        weight: float = 1.0,
    ) -> FingerprintCondition:
        if isinstance(mol, Chem.Mol):
            fp_tensor = self.evaluate_mol(mol)["fingerprint"]
        elif isinstance(mol, np.ndarray):
            fp_tensor = torch.from_numpy(mol)
        elif isinstance(mol, torch.Tensor):
            fp_tensor = mol
        fp_tensor = fp_tensor.view(1, self.fp_dim)
        return FingerprintCondition(fingerprint=fp_tensor, property_def=self, weight=weight)


class ECFP4(StandardFingerprintProperty):
    def __init__(self, name: str = "ecfp4", num_embedding_tokens: int = 4) -> None:
        super().__init__(name, num_embedding_tokens)

    @property
    def fp_type(self) -> str:
        return "ecfp4"

    @property
    def fp_dim(self) -> int:
        return 2048


class FCFP4(StandardFingerprintProperty):
    def __init__(self, name: str = "fcfp4", num_embedding_tokens: int = 4) -> None:
        super().__init__(name, num_embedding_tokens)

    @property
    def fp_type(self) -> str:
        return "fcfp4"

    @property
    def fp_dim(self) -> int:
        return 2048
