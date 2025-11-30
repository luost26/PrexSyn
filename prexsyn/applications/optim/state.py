import dataclasses
from collections.abc import Sequence

import torch
from rdkit import Chem

from prexsyn_engine.synthesis import Synthesis


@dataclasses.dataclass(frozen=True)
class State:
    coords: torch.Tensor
    scores: torch.Tensor
    constraint_scores: torch.Tensor
    ages: torch.Tensor
    syntheses: Sequence[Synthesis | None]
    products: Sequence[Chem.Mol | None]

    @property
    def total_scores(self) -> torch.Tensor:
        return self.scores + self.constraint_scores

    def concat(self, other: "State") -> "State":
        return State(
            coords=torch.cat([self.coords, other.coords], dim=0),
            scores=torch.cat([self.scores, other.scores], dim=0),
            constraint_scores=torch.cat([self.constraint_scores, other.constraint_scores], dim=0),
            ages=torch.cat([self.ages, other.ages], dim=0),
            syntheses=list(self.syntheses) + list(other.syntheses),
            products=list(self.products) + list(other.products),
        )

    def deduplicate(self) -> "State":
        seen: set[str] = set()
        unique_indices: list[int] = []
        for i, prod in enumerate(self.products):
            if prod is None:
                continue
            smi = Chem.MolToSmiles(prod, canonical=True)
            if smi in seen:
                continue
            seen.add(smi)
            unique_indices.append(i)
        if len(unique_indices) == len(self.scores):
            return self
        indices = torch.tensor(unique_indices, dtype=torch.long, device=self.scores.device)
        return self.index_select(indices)

    def index_select(self, indices: torch.Tensor) -> "State":
        indices_list: list[int] = indices.cpu().tolist()
        return State(
            coords=self.coords[indices],
            scores=self.scores[indices],
            constraint_scores=self.constraint_scores[indices],
            ages=self.ages[indices],
            syntheses=[self.syntheses[i] for i in indices_list],
            products=[self.products[i] for i in indices_list],
        )

    def topk(self, k: int) -> "State":
        indices = torch.topk(self.total_scores, k=min(k, len(self.scores)), largest=True).indices
        return self.index_select(indices)

    def samplek(self, k: int, temperature: float) -> "State":
        prob = torch.softmax(self.total_scores / temperature, dim=0)
        indices = torch.multinomial(prob, num_samples=min(k, len(self.total_scores)), replacement=False)
        return self.index_select(indices)

    def update_if_better(self, other: "State") -> "State":
        accept = other.total_scores > self.total_scores  # scores might be negative so donot do ratio test
        accept_list = accept.cpu().tolist()
        state_next = State(
            coords=torch.where(accept[:, None].expand_as(self.coords), other.coords, self.coords),
            scores=torch.where(accept, other.scores, self.scores),
            constraint_scores=torch.where(accept, other.constraint_scores, self.constraint_scores),
            ages=torch.where(accept, torch.zeros_like(self.ages), self.ages + 1),
            syntheses=[other.syntheses[i] if a else self.syntheses[i] for i, a in enumerate(accept_list)],
            products=[other.products[i] if a else self.products[i] for i, a in enumerate(accept_list)],
        )
        return state_next

    def update_if_mh_accept(self, other: "State", temperature: float) -> "State":
        prob = torch.exp((other.total_scores - self.total_scores) / temperature)
        accept = torch.rand_like(prob) < prob
        accept_list = accept.cpu().tolist()
        state_next = State(
            coords=torch.where(accept[:, None].expand_as(self.coords), other.coords, self.coords),
            scores=torch.where(accept, other.scores, self.scores),
            constraint_scores=torch.where(accept, other.constraint_scores, self.constraint_scores),
            ages=torch.where(accept, torch.zeros_like(self.ages), self.ages + 1),
            syntheses=[other.syntheses[i] if a else self.syntheses[i] for i, a in enumerate(accept_list)],
            products=[other.products[i] if a else self.products[i] for i, a in enumerate(accept_list)],
        )
        return state_next


@dataclasses.dataclass(frozen=True)
class DeltaState(State):
    pass
