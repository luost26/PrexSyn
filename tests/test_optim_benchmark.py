import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np


def load_optim_module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "benchmarks" / "optim.py"
    spec = importlib.util.spec_from_file_location("optim_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubObjective:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def score_list(self, smiles: list[str]) -> list[float]:
        self.calls.append(smiles)
        return [0.1 * (index + 1) for index in range(len(smiles))]


class StubMolecule:
    def __init__(self, smiles: str) -> None:
        self._smiles = smiles

    def smiles(self) -> str:
        return self._smiles


def test_budgeted_scoring_counts_unique_molecules_and_honors_budget() -> None:
    optim = load_optim_module()
    objective = StubObjective()
    scorer = optim.BudgetedScoringFunction(objective, max_evals=2)
    phenotypes = [(None, StubMolecule(smiles)) for smiles in ("CCO", "CCN", "CCC", "CCO")]

    scores = scorer(cast(Any, phenotypes))

    assert objective.calls == [["CCO", "CCN"]]
    assert scorer.num_evaluations == 2
    assert np.array_equal(scores, np.array([0.1, 0.2, 0.0, 0.1]))
    assert scorer.dataframe()["smiles"].tolist() == ["CCO", "CCN"]


def test_auc_top10_pads_early_termination_with_last_value() -> None:
    optim = load_optim_module()

    assert optim.auc_top10([], 10) == 0.0
    assert optim.auc_top10([0.1, 0.5, 0.2], 5) == 0.24


def test_paper_comparison_covers_every_standard_task() -> None:
    optim = load_optim_module()

    assert tuple(optim.PAPER_AUC_TOP10) == optim.STANDARD_TASKS
    assert tuple(optim.PAPER_AUC_TOP10_STD) == optim.STANDARD_TASKS


def test_population_dedup_uses_smiles_values() -> None:
    optim = load_optim_module()
    phenotypes = [(None, StubMolecule(smiles)) for smiles in ("CCO", "CCN", "CCO")]
    population = optim.Population(
        genotypes=np.zeros((3, 2), dtype=bool),
        phenotypes=cast(Any, phenotypes),
        fitnesses=np.array([0.1, 0.2, 0.3]),
        unique_identifiers=np.arange(3),
        parents=np.full((3, 2), -1),
    )

    deduplicated = population.dedup()

    assert deduplicated.size() == 2
    assert deduplicated.unique_identifiers.tolist() == [0, 1]
