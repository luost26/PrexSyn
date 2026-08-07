import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest


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


def test_budgeted_scoring_detects_a_mathematically_saturated_top10() -> None:
    optim = load_optim_module()
    objective = StubObjective()
    scorer = optim.BudgetedScoringFunction(objective, max_evals=100)
    scorer.evaluations = [
        optim.Evaluation(evaluation=index + 1, generation=0, smiles=f"C{index}", score=1.0) for index in range(10)
    ]

    assert scorer.saturated

    scorer.evaluations[-1] = optim.Evaluation(evaluation=10, generation=0, smiles="C9", score=0.9)
    assert not scorer.saturated


def test_saturated_partial_run_is_complete(tmp_path: Path) -> None:
    optim = load_optim_module()
    path = tmp_path / "run.csv"
    pd.DataFrame({"score": [1.0] * 10}).to_csv(path, index=False)

    assert optim.run_is_complete(path, max_evals=10_000)

    pd.DataFrame({"score": [1.0] * 9 + [0.9]}).to_csv(path, index=False)
    assert not optim.run_is_complete(path, max_evals=10_000)


def test_auc_top10_pads_early_termination_with_last_value() -> None:
    optim = load_optim_module()

    assert optim.auc_top10([], 10) == 0.0
    assert optim.auc_top10([0.1, 0.5, 0.2], 5) == 0.24


def test_run_selection_preserves_default_seed_indices() -> None:
    optim = load_optim_module()

    assert optim.resolve_run_indices((), 5) == (1, 2, 3, 4, 5)
    assert optim.resolve_run_indices((3, 5), 5) == (3, 5)
    with pytest.raises(optim.click.ClickException, match="at most once"):
        optim.resolve_run_indices((3, 3), 5)
    with pytest.raises(optim.click.ClickException, match="less than or equal"):
        optim.resolve_run_indices((6,), 5)


def test_collect_run_summaries_requires_complete_runs(tmp_path: Path) -> None:
    optim = load_optim_module()
    task_dir = tmp_path / "amlodipine"
    task_dir.mkdir()
    pd.DataFrame(
        {
            "evaluation": [1, 2],
            "generation": [0, 0],
            "smiles": ["CCO", "CCN"],
            "score": [0.2, 0.8],
        }
    ).to_csv(task_dir / "run_01.csv", index=False)

    summaries = optim.collect_run_summaries(tmp_path, ("amlodipine",), (1,), num_runs=1, max_evals=2, seed=17)

    assert summaries == [
        {
            "task": "amlodipine",
            "run": 1,
            "seed": 17,
            "evaluations": 2,
            "auc_top10": 0.35,
            "top10": 0.5,
            "best": 0.8,
        }
    ]
    assert (task_dir / "summary.csv").exists()

    with pytest.raises(optim.click.ClickException, match="missing or incomplete"):
        optim.collect_run_summaries(tmp_path, ("amlodipine",), (2,), num_runs=2, max_evals=2, seed=17)


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


def test_lightweight_history_tracks_ids_without_retaining_individuals() -> None:
    optim = load_optim_module()
    phenotypes = [(None, StubMolecule("CCO")), (None, StubMolecule("CCN"))]
    population = optim.Population(
        genotypes=np.zeros((2, 2), dtype=bool),
        phenotypes=cast(Any, phenotypes),
        fitnesses=np.array([0.1, 0.2]),
        unique_identifiers=np.array([4, 9]),
        parents=np.full((2, 2), -1),
    )
    history = optim.History(record_individuals=False)

    history.add_population(population)

    assert history.individuals == {}
    assert history.next_unique_id() == 10
