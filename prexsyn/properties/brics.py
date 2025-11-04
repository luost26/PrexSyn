import torch
from rdkit import Chem
from torch import nn

from prexsyn.data.struct import PropertyRepr
from prexsyn.models.embeddings import BasePropertyEmbedder, Embedding
from prexsyn.queries import Condition
from prexsyn_engine.featurizer.substructures import BRICSFragmentsFeaturizer
from prexsyn_engine.fingerprints import get_fingerprints, get_fp_dim
from prexsyn_engine.synthesis import Synthesis

from .base import BasePropertyDef


class MultiFingerprintEmbedder(BasePropertyEmbedder):
    def __init__(self, fingerprint_dim: int, embedding_dim: int, fp_dropout: float = 0.5) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(fingerprint_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.fp_dropout = fp_dropout

    def forward(self, fingerprints: torch.Tensor, fingerprint_exists: torch.Tensor) -> Embedding:
        h = self.mlp(fingerprints)

        m = torch.full(fingerprint_exists.shape, fill_value=float("-inf"), dtype=h.dtype, device=h.device)
        m.masked_fill_(fingerprint_exists, 0.0)

        if self.training and self.fp_dropout > 0.0:
            p = torch.rand_like(m)
            m.masked_fill_(p < self.fp_dropout, float("-inf"))

        return Embedding(h, m)


class BRICSCondition(Condition):
    def __init__(self, fragments: tuple[Chem.Mol, ...], property_def: "BRICSFragments", weight: float = 1.0) -> None:
        super().__init__(weight)
        self.property_def = property_def

        self._fingerprints = (
            torch.from_numpy(get_fingerprints(list(fragments), fp_type=self.property_def.fp_type)).float().unsqueeze(0)
        )  # (1, num_fragments, fp_dim)
        self._exists = torch.ones((1, len(fragments)), dtype=torch.bool)

    def get_property_repr(self) -> PropertyRepr:
        return {self.property_def.name: {"fingerprints": self._fingerprints, "fingerprint_exists": self._exists}}

    def score(self, synthesis: Synthesis, product: Chem.Mol) -> float:
        # TODO: implement scoring
        raise NotImplementedError("Scoring for BRICSCondition is not implemented.")


class BRICSFragments(BasePropertyDef):
    def __init__(self, name: str = "brics", max_num_fragments: int = 4, fp_type: str = "ecfp4") -> None:
        super().__init__()
        self._name = name
        self._max_num_fragments = max_num_fragments
        self._fp_type = fp_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def fp_type(self) -> str:
        return self._fp_type

    def get_featurizer(self) -> BRICSFragmentsFeaturizer:
        return BRICSFragmentsFeaturizer(
            name=self._name,
            fp_type=self._fp_type,
            max_num_fragments=self._max_num_fragments,
        )

    def get_embedder(self, model_dim: int) -> MultiFingerprintEmbedder:
        return MultiFingerprintEmbedder(
            fingerprint_dim=get_fp_dim(self._fp_type),
            embedding_dim=model_dim,
            fp_dropout=0.5,
        )

    def has(self, *fragments: Chem.Mol, weight: float = 1.0) -> BRICSCondition:
        return BRICSCondition(fragments, property_def=self, weight=weight)
