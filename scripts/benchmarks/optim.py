import heapq
import importlib
import json
import logging
import pathlib
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import click
import numpy as np
import pandas as pd
import torch

from prexsyn.shortcuts import AllInOneLoader, MoleculeProjector
from prexsyn.shortcuts.genetic import History, Population, evolve, initialize
from prexsyn_engine.chemistry import Molecule
from prexsyn_engine.chemspace import Synthesis

STANDARD_TASKS = (
    "amlodipine",
    "fexofenadine",
    "osimertinib",
    "perindopril",
    "ranolazine",
    "sitagliptin",
    "zaleplon",
    "celecoxib_rediscovery",
)

# PrexSyn values from tables/optim_guacamol.tex in arXiv:2512.00384.
PAPER_AUC_TOP10 = {
    "amlodipine": 0.781,
    "fexofenadine": 0.837,
    "osimertinib": 0.855,
    "perindopril": 0.714,
    "ranolazine": 0.807,
    "sitagliptin": 0.471,
    "zaleplon": 0.504,
    "celecoxib_rediscovery": 0.801,
}
PAPER_AUC_TOP10_STD = {
    "amlodipine": 0.023,
    "fexofenadine": 0.013,
    "osimertinib": 0.007,
    "perindopril": 0.010,
    "ranolazine": 0.009,
    "sitagliptin": 0.030,
    "zaleplon": 0.018,
    "celecoxib_rediscovery": 0.005,
}


class GuacaMolScoringFunction(Protocol):
    def score_list(self, smiles_list: list[str]) -> list[float]: ...


def make_scoring_function(task_name: str) -> GuacaMolScoringFunction:
    """Build one of the GuacaMol objectives bundled by prexsyn-third-party."""
    try:
        standard_benchmarks = importlib.import_module("guacamol.standard_benchmarks")
    except ModuleNotFoundError as exc:
        raise click.ClickException(
            "The optimization benchmark requires prexsyn-third-party. "
            "Install the evaluation dependencies with `uv sync --extra eval`."
        ) from exc

    factories: dict[str, Callable[[], Any]] = {
        "amlodipine": standard_benchmarks.amlodipine_rings,
        "fexofenadine": standard_benchmarks.hard_fexofenadine,
        "osimertinib": standard_benchmarks.hard_osimertinib,
        "perindopril": standard_benchmarks.perindopril_rings,
        "ranolazine": standard_benchmarks.ranolazine_mpo,
        "sitagliptin": standard_benchmarks.sitagliptin_replacement,
        "zaleplon": standard_benchmarks.zaleplon_with_other_formula,
        "celecoxib_rediscovery": lambda: standard_benchmarks.similarity(
            smiles="CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F",
            name="Celecoxib",
            fp_type="ECFP4",
            threshold=1.0,
            rediscovery=True,
        ),
    }
    if task_name not in factories:
        raise ValueError(f"Unknown task {task_name!r}; expected one of {STANDARD_TASKS}.")
    return factories[task_name]().objective


@dataclass(frozen=True)
class Evaluation:
    evaluation: int
    generation: int
    smiles: str
    score: float


class BudgetedScoringFunction:
    """Adapt a GuacaMol objective to the GA while counting unique oracle calls."""

    def __init__(self, scoring_function: GuacaMolScoringFunction, max_evals: int):
        if max_evals <= 0:
            raise ValueError("max_evals must be positive.")
        self.scoring_function = scoring_function
        self.max_evals = max_evals
        self.generation = 0
        self.cache: dict[str, float] = {}
        self.evaluations: list[Evaluation] = []

    @property
    def num_evaluations(self) -> int:
        return len(self.evaluations)

    @property
    def exhausted(self) -> bool:
        return self.num_evaluations >= self.max_evals

    @property
    def saturated(self) -> bool:
        if self.num_evaluations < 10:
            return False
        top10 = heapq.nlargest(10, (item.score for item in self.evaluations))
        return all(np.isclose(score, 1.0, rtol=0.0, atol=1e-12) for score in top10)

    def __call__(self, phenotypes: Sequence[tuple[Synthesis, Molecule]]) -> np.ndarray:
        smiles = [molecule.smiles() for _, molecule in phenotypes]
        unseen: list[str] = []
        unseen_set: set[str] = set()
        remaining = self.max_evals - self.num_evaluations
        for smi in smiles:
            if smi not in self.cache and smi not in unseen_set and len(unseen) < remaining:
                unseen.append(smi)
                unseen_set.add(smi)

        if unseen:
            scores = self.scoring_function.score_list(unseen)
            if len(scores) != len(unseen):
                raise RuntimeError(f"Scoring function returned {len(scores)} scores for {len(unseen)} molecules.")
            for smi, score_raw in zip(unseen, scores, strict=True):
                score = float(score_raw)
                self.cache[smi] = score
                self.evaluations.append(
                    Evaluation(
                        evaluation=self.num_evaluations + 1,
                        generation=self.generation,
                        smiles=smi,
                        score=score,
                    )
                )

        # A batch can cross the budget. Molecules beyond it are deliberately not
        # sent to the oracle and receive the minimum GuacaMol score.
        return np.asarray([self.cache.get(smi, 0.0) for smi in smiles], dtype=float)

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(item) for item in self.evaluations)


def auc_top10(scores: Sequence[float], max_evals: int) -> float:
    """Compute the oracle-call AUC of the running mean of the best ten scores."""
    if max_evals <= 0:
        raise ValueError("max_evals must be positive.")
    if not scores:
        return 0.0

    top10: list[float] = []
    moving_average: list[float] = []
    for score in scores[:max_evals]:
        heapq.heappush(top10, float(score))
        if len(top10) > 10:
            heapq.heappop(top10)
        moving_average.append(float(np.mean(top10)))

    if len(moving_average) < max_evals:
        moving_average.extend([moving_average[-1]] * (max_evals - len(moving_average)))
    return float(np.mean(moving_average))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_run_indices(selected_runs: Sequence[int], num_runs: int) -> tuple[int, ...]:
    run_indices = tuple(selected_runs) or tuple(range(1, num_runs + 1))
    if len(set(run_indices)) != len(run_indices):
        raise click.ClickException("Each --run index must be specified at most once.")
    if any(run_index > num_runs for run_index in run_indices):
        raise click.ClickException("Every --run index must be less than or equal to --num-runs.")
    return run_indices


def run_optimization(
    projector: MoleculeProjector,
    scoring_function: BudgetedScoringFunction,
    *,
    population_size: int,
    offspring_size: int,
    temperature: float,
    time_limit: float | None,
    logger: logging.Logger,
) -> tuple[Population, History]:
    start = time.monotonic()
    population, history = initialize(
        size=population_size,
        projector=projector,
        fn=scoring_function,
        oversample_factor=1,
        record_history=False,
    )
    logger.info(
        "generation=0 evals=%d/%d population=%d best=%.4f auc_top10=%.4f",
        scoring_function.num_evaluations,
        scoring_function.max_evals,
        population.size(),
        float(population.fitnesses.max()) if population.size() else 0.0,
        auc_top10([item.score for item in scoring_function.evaluations], scoring_function.max_evals),
    )

    generation = 1
    stagnant_generations = 0
    while not scoring_function.exhausted and not scoring_function.saturated:
        if time_limit is not None and time.monotonic() - start >= time_limit:
            logger.info("Time limit reached after %.1f seconds.", time.monotonic() - start)
            break
        if population.size() < 2:
            logger.warning("Stopping because fewer than two candidates hatched.")
            break

        scoring_function.generation = generation
        previous_evaluations = scoring_function.num_evaluations
        evolve(
            population,
            history,
            projector,
            scoring_function,
            k=min(offspring_size, population.size()),
            t=temperature,
        )
        if scoring_function.num_evaluations == previous_evaluations:
            stagnant_generations += 1
        else:
            stagnant_generations = 0
        scores = [item.score for item in scoring_function.evaluations]
        logger.info(
            "generation=%d evals=%d/%d population=%d best=%.4f auc_top10=%.4f",
            generation,
            scoring_function.num_evaluations,
            scoring_function.max_evals,
            population.size(),
            float(population.fitnesses.max()) if population.size() else 0.0,
            auc_top10(scores, scoring_function.max_evals),
        )
        if stagnant_generations >= 20:
            logger.warning("Stopping after 20 generations without a new unique molecule.")
            break
        generation += 1

    if scoring_function.saturated and not scoring_function.exhausted:
        logger.info(
            "Stopping at %d evaluations because the Top-10 is saturated at the maximum score.",
            scoring_function.num_evaluations,
        )

    return population, history


def read_completed_run(path: pathlib.Path, max_evals: int) -> dict[str, float | int]:
    frame = pd.read_csv(path)
    scores = frame["score"].astype(float).tolist()[:max_evals]
    return {
        "evaluations": len(scores),
        "auc_top10": auc_top10(scores, max_evals),
        "top10": float(np.mean(sorted(scores, reverse=True)[:10])) if scores else 0.0,
        "best": max(scores, default=0.0),
    }


def run_is_complete(path: pathlib.Path, max_evals: int) -> bool:
    frame = pd.read_csv(path)
    if len(frame) >= max_evals:
        return True
    scores = frame["score"].astype(float).tolist()
    return len(scores) >= 10 and all(
        np.isclose(score, 1.0, rtol=0.0, atol=1e-12) for score in heapq.nlargest(10, scores)
    )


def configure_logger(task_dir: pathlib.Path) -> logging.Logger:
    logger = logging.getLogger(f"prexsyn.optim.{task_dir.name}")
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(task_dir / "benchmark.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def collect_run_summaries(
    output_dir: pathlib.Path,
    tasks: Sequence[str],
    run_indices: Sequence[int],
    *,
    num_runs: int,
    max_evals: int,
    seed: int,
) -> list[dict[str, float | int | str]]:
    summaries: list[dict[str, float | int | str]] = []
    for task_name in tasks:
        task_dir = output_dir / task_name
        for run_index in run_indices:
            run_path = task_dir / f"run_{run_index:02d}.csv"
            if not run_path.exists() or not run_is_complete(run_path, max_evals):
                raise click.ClickException(f"Cannot summarize missing or incomplete run: {run_path}")
            summaries.append(
                {
                    "task": task_name,
                    "run": run_index,
                    "seed": seed + STANDARD_TASKS.index(task_name) * num_runs + run_index - 1,
                    **read_completed_run(run_path, max_evals),
                }
            )
        pd.DataFrame(item for item in summaries if item["task"] == task_name).to_csv(
            task_dir / "summary.csv", index=False
        )
    return summaries


def write_comparison(output_dir: pathlib.Path, run_summaries: Sequence[dict[str, float | int | str]]) -> pd.DataFrame:
    run_frame = pd.DataFrame(run_summaries)
    run_frame.to_csv(output_dir / "runs.csv", index=False)
    comparison = (
        run_frame.groupby("task", sort=False)
        .agg(
            runs=("run", "count"),
            evaluations_mean=("evaluations", "mean"),
            auc_top10_mean=("auc_top10", "mean"),
            auc_top10_std=("auc_top10", lambda values: float(np.std(values, ddof=0))),
            top10_mean=("top10", "mean"),
            best_mean=("best", "mean"),
        )
        .reset_index()
    )
    comparison["paper_auc_top10"] = comparison["task"].map(lambda task: PAPER_AUC_TOP10[str(task)])
    comparison["paper_auc_top10_std"] = comparison["task"].map(lambda task: PAPER_AUC_TOP10_STD[str(task)])
    comparison["delta_from_paper"] = comparison["auc_top10_mean"] - comparison["paper_auc_top10"]
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    return comparison


@click.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    default=pathlib.Path("./data/trained_models/enamine2310_rxn115_202511.yml"),
    show_default=True,
)
@click.option(
    "--out",
    "output_dir",
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    default=pathlib.Path("./outputs/benchmarks/optim"),
    show_default=True,
)
@click.option("--device", default="cuda", show_default=True)
@click.option("--task", "selected_tasks", multiple=True, type=click.Choice(STANDARD_TASKS))
@click.option("--num-runs", type=click.IntRange(min=1), default=5, show_default=True)
@click.option(
    "--run", "selected_runs", multiple=True, type=click.IntRange(min=1), help="Run only these 1-based seeds."
)
@click.option("--max-evals", type=click.IntRange(min=1), default=10_000, show_default=True)
@click.option("--population-size", type=click.IntRange(min=2), default=500, show_default=True)
@click.option("--offspring-size", type=click.IntRange(min=2), default=50, show_default=True)
@click.option("--temperature", type=click.FloatRange(min=0.0, min_open=True), default=0.5, show_default=True)
@click.option("--num-samples", type=click.IntRange(min=1), default=8, show_default=True)
@click.option("--batch-size-limit", type=click.IntRange(min=1), default=64, show_default=True)
@click.option("--seed", type=int, default=2026, show_default=True)
@click.option("--time-limit", type=click.FloatRange(min=0.0, min_open=True), default=None)
@click.option("--overwrite", is_flag=True, help="Rerun and replace existing per-run CSV files.")
@click.option(
    "--summarize-only", is_flag=True, help="Consolidate existing complete run CSVs without loading the model."
)
def main(
    config_path: pathlib.Path,
    output_dir: pathlib.Path,
    device: str,
    selected_tasks: tuple[str, ...],
    num_runs: int,
    selected_runs: tuple[int, ...],
    max_evals: int,
    population_size: int,
    offspring_size: int,
    temperature: float,
    num_samples: int,
    batch_size_limit: int,
    seed: int,
    time_limit: float | None,
    overwrite: bool,
    summarize_only: bool,
) -> None:
    """Run the standard GuacaMol optimization tasks with the fingerprint GA."""
    torch.set_grad_enabled(False)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = selected_tasks or STANDARD_TASKS
    run_indices = resolve_run_indices(selected_runs, num_runs)

    if summarize_only:
        run_summaries = collect_run_summaries(
            output_dir,
            tasks,
            run_indices,
            num_runs=num_runs,
            max_evals=max_evals,
            seed=seed,
        )
        comparison = write_comparison(output_dir, run_summaries)
        click.echo(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        return

    loader = AllInOneLoader(config_path)
    model = loader.model().to(device).eval()
    projector = MoleculeProjector(
        model=model,
        detokenizer=loader.detokenizer(),
        descriptor="ecfp4",
        num_samples=num_samples,
        batch_size_limit=batch_size_limit,
    )

    config = {
        "config": str(config_path),
        "device": device,
        "tasks": list(tasks),
        "num_runs": num_runs,
        "runs": list(run_indices),
        "max_evals": max_evals,
        "population_size": population_size,
        "offspring_size": offspring_size,
        "temperature": temperature,
        "num_samples": num_samples,
        "batch_size_limit": batch_size_limit,
        "seed": seed,
        "time_limit": time_limit,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    run_summaries: list[dict[str, float | int | str]] = []
    for task_name in tasks:
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        logger = configure_logger(task_dir)

        for run_index in run_indices:
            run_path = task_dir / f"run_{run_index:02d}.csv"
            run_seed = seed + STANDARD_TASKS.index(task_name) * num_runs + run_index - 1
            existing_evaluations = len(pd.read_csv(run_path)) if run_path.exists() else 0
            if run_path.exists() and not overwrite and run_is_complete(run_path, max_evals):
                logger.info("Reusing %s", run_path)
                metrics = read_completed_run(run_path, max_evals)
            else:
                if run_path.exists() and not overwrite:
                    logger.info(
                        "Existing run is incomplete (%d/%d evaluations); rerunning it from its seed.",
                        existing_evaluations,
                        max_evals,
                    )
                logger.info("Starting task=%s run=%d/%d seed=%d", task_name, run_index, num_runs, run_seed)
                seed_everything(run_seed)
                scoring_function = BudgetedScoringFunction(make_scoring_function(task_name), max_evals)
                run_optimization(
                    projector,
                    scoring_function,
                    population_size=population_size,
                    offspring_size=offspring_size,
                    temperature=temperature,
                    time_limit=time_limit,
                    logger=logger,
                )
                scoring_function.dataframe().to_csv(run_path, index=False)
                metrics = read_completed_run(run_path, max_evals)

            summary = {
                "task": task_name,
                "run": run_index,
                "seed": run_seed,
                **metrics,
            }
            run_summaries.append(summary)
            logger.info(
                "Finished task=%s run=%d evals=%d auc_top10=%.4f top10=%.4f best=%.4f",
                task_name,
                run_index,
                metrics["evaluations"],
                metrics["auc_top10"],
                metrics["top10"],
                metrics["best"],
            )

        pd.DataFrame(item for item in run_summaries if item["task"] == task_name).to_csv(
            task_dir / "summary.csv", index=False
        )

    comparison = write_comparison(output_dir, run_summaries)
    click.echo(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
