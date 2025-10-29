import heapq
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdDepictor import Compute2DCoords

from .state import State


class OptimTracker:
    def __init__(self) -> None:
        super().__init__()
        self._data: dict[str, list[Any]] = {
            "step": [],
            "inbatch_index": [],
            "synthesis": [],
            "product": [],
            "smiles": [],
            "score": [],
            "constraint": [],
            "prev_age": [],
            "timestamp": [],
        }
        self._visited: set[str] = set()
        self._top10: list[float] = []
        self._moving_top10_avg: list[float] = []

    def __len__(self) -> int:
        return len(self._data["step"])

    def add(self, step: int, state: State) -> None:
        for i, (score, constraint, syn, prod, age) in enumerate(
            zip(
                state.scores.cpu().tolist(),
                state.constraint_scores.cpu().tolist(),
                state.syntheses,
                state.products,
                state.ages.cpu().tolist(),
            )
        ):
            if syn is None or prod is None:
                continue
            smi = Chem.MolToSmiles(prod, canonical=True)
            if smi in self._visited:
                continue
            self._visited.add(smi)
            Compute2DCoords(prod)

            self._data["step"].append(step)
            self._data["inbatch_index"].append(i)
            self._data["synthesis"].append(syn)
            self._data["product"].append(prod)
            self._data["smiles"].append(smi)
            self._data["score"].append(score)
            self._data["constraint"].append(constraint)
            self._data["prev_age"].append(age)
            self._data["timestamp"].append(pd.Timestamp.now())

            heapq.heappush(self._top10, score)
            if len(self._top10) > 10:
                heapq.heappop(self._top10)
            self._moving_top10_avg.append(sum(self._top10) / len(self._top10) if self._top10 else 0.0)

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._data)

    def moving_top10_avg(self) -> float:
        if self._moving_top10_avg:
            return self._moving_top10_avg[-1]
        return 0.0

    def auc_top10(self, max_steps: int | None = None) -> float:
        if self._moving_top10_avg:
            if max_steps is None:
                max_steps = len(self._moving_top10_avg)
            pad_length = max(0, max_steps - len(self._moving_top10_avg))
            return sum(self._moving_top10_avg + [self._moving_top10_avg[-1]] * pad_length) / max_steps
        return 0.0
