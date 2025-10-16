from collections.abc import Mapping

import torch
from rdkit import Chem
from torch import nn

from prexsyn.models.embeddings import BasePropertyEmbedder, Embedding
from prexsyn_engine.featurizer.rdkit_descriptors import RDKitDescriptorsFeaturizer

from .base import BasePropertyDef


class ScalarPropertySetEmbedder(BasePropertyEmbedder):
    def __init__(self, max_property_types: int, embedding_dim: int, prop_dropout: float = 0.5) -> None:
        super().__init__()
        self.prop_dropout = prop_dropout
        self.value_mlp = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.type_embedding = nn.Embedding(max_property_types + 1, embedding_dim)
        self.mix_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, values: torch.Tensor, types: torch.Tensor) -> Embedding:
        value_embeddings = self.value_mlp(values[..., None])
        type_embeddings = self.type_embedding(types)
        combined = value_embeddings * type_embeddings
        h = self.mix_mlp(combined)
        m = torch.full(types.shape, fill_value=float("-inf"), dtype=h.dtype, device=h.device)
        m.masked_fill_(types != 0, 0.0)

        if self.training and self.prop_dropout > 0.0:
            p = torch.rand_like(m)
            m.masked_fill_(p < self.prop_dropout, float("-inf"))

        return Embedding(h, m)


class ScalarPropertyUpperBoundEmbedder(BasePropertyEmbedder):
    def __init__(self, max_property_types: int, embedding_dim: int, num_tokens: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.value_mlp = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.type_embedding = nn.Embedding(max_property_types + 1, embedding_dim)
        self.mixer = nn.Linear(embedding_dim, num_tokens * embedding_dim)

        self.observed_max_values = nn.Parameter(torch.zeros(max_property_types + 1), requires_grad=False)

    def get_upper_bound_for_training(self, values: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            types_flat = types.flatten()
            self.observed_max_values[types_flat] = torch.max(self.observed_max_values[types_flat], values.flatten())

        r = torch.rand_like(values)
        upper_bound = values + r * (self.observed_max_values[types] - values).clamp_min(0)
        return upper_bound

    def forward(self, values: torch.Tensor, types: torch.Tensor) -> Embedding:
        if values.size(-1) != 1 or types.size(-1) != 1:
            raise ValueError("values and types must have shape (..., 1), only single property supported.")

        if self.training:
            values = self.get_upper_bound_for_training(values, types)

        value_embeddings = self.value_mlp(values[..., None])
        type_embeddings = self.type_embedding(types)
        combined = value_embeddings * type_embeddings
        h: torch.Tensor = self.mixer(combined)
        h = h.reshape(*values.shape[:-1], self.num_tokens, self.embedding_dim)

        m = torch.full(h.shape[:-1], fill_value=float("-inf"), dtype=h.dtype, device=h.device)
        m.masked_fill_(types.expand(h.shape[:-1]) != 0, 0.0)

        return Embedding(h, m)


class RDKitDescriptors(BasePropertyDef):
    def __init__(self, name: str = "rdkit_descriptors", num_evaluated_descriptors: int = 4) -> None:
        super().__init__()
        self._name = name
        self._num_evaluated_descriptors = num_evaluated_descriptors

    @property
    def name(self) -> str:
        return self._name

    def get_featurizer(self) -> RDKitDescriptorsFeaturizer:
        return RDKitDescriptorsFeaturizer(
            name=self._name,
            num_evaluated_descriptors=self._num_evaluated_descriptors,
        )

    def get_embedder(self, model_dim: int) -> ScalarPropertySetEmbedder:
        return ScalarPropertySetEmbedder(
            max_property_types=100,
            embedding_dim=model_dim,
            prop_dropout=0.5,
        )

    @property
    def available_descriptors(self) -> list[str]:
        return list(self.get_featurizer().descriptor_names)

    def evaluate_single_mol(self, mol: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor]:
        from rdkit.Chem.rdMolDescriptors import Properties

        types: list[int] = []
        values: list[float] = []

        for i, desc_name in enumerate(self.available_descriptors, start=1):
            types.append(i)
            values.append(Properties.GetProperty(desc_name)(mol))

        return torch.tensor(values, dtype=torch.float32), torch.tensor(types, dtype=torch.long)

    def evaluate_mol(self, mol: Chem.Mol | list[Chem.Mol]) -> Mapping[str, torch.Tensor]:
        all_types: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []

        if isinstance(mol, Chem.Mol):
            mol = [mol]

        for m in mol:
            values, types = self.evaluate_single_mol(m)
            all_types.append(types)
            all_values.append(values)

        return {f"{self.name}.values": torch.stack(all_values), f"{self.name}.types": torch.stack(all_types)}


class RDKitDescriptorUpperBounds(BasePropertyDef):
    def __init__(self, name: str = "rdkit_descriptor_upper_bounds") -> None:
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_featurizer(self) -> RDKitDescriptorsFeaturizer:
        return RDKitDescriptorsFeaturizer(name=self._name, num_evaluated_descriptors=1)

    def get_embedder(self, model_dim: int) -> ScalarPropertyUpperBoundEmbedder:
        return ScalarPropertyUpperBoundEmbedder(
            max_property_types=100,
            embedding_dim=model_dim,
            num_tokens=4,
        )
