import random
import time

import torch
from rdkit import Chem

from prexsyn.data.struct import move_to_device
from prexsyn.factories.facade import Facade
from prexsyn.models.prexsyn import PrexSyn
from prexsyn.queries import Query
from prexsyn.samplers.query import QuerySampler
from prexsyn.utils.oracles import OracleProtocol, get_oracle
from prexsyn_engine.synthesis import Synthesis

from .state import State
from .step import StepStrategy
from .tracker import OptimTracker


class Optimizer:
    def __init__(
        self,
        facade: Facade,
        model: PrexSyn,
        init_query: Query,
        num_init_samples: int,
        max_evals: int,
        step_strategy: StepStrategy,
        oracle_fn: OracleProtocol,
        constraint_fn: OracleProtocol | None = None,
        cond_query: Query | None = None,
        time_limit: int | None = None,
        handle_interrupt: bool = False,
    ) -> None:
        super().__init__()
        self.facade = facade
        self.model = model
        self.init_query = init_query
        self.num_init_samples = num_init_samples
        self.max_evals = max_evals
        self.step_strategy = step_strategy
        self.oracle_fn = oracle_fn
        self.constraint_fn = constraint_fn
        self.cond_query = cond_query
        self.time_limit = time_limit
        self.handle_interrupt = handle_interrupt

    def _get_init_state(self) -> State:
        sampler = QuerySampler(
            model=self.model,
            token_def=self.facade.get_token_def(),
            num_samples=self.num_init_samples,
        )
        syns = self.facade.get_detokenizer()(**sampler.sample(self.init_query))

        syn_prod_pairs: list[tuple[Synthesis, Chem.Mol]] = []
        for syn in syns:
            if syn.stack_size() != 1:
                continue
            for product in syn.top().to_list():
                syn_prod_pairs.append((syn, product))
        random.shuffle(syn_prod_pairs)
        syn_prod_pairs = syn_prod_pairs[: self.num_init_samples]

        selected_syntheses = [item[0] for item in syn_prod_pairs]
        selected_products = [item[1] for item in syn_prod_pairs]

        property_repr = move_to_device(
            {"ecfp4": self.facade.property_set["ecfp4"].evaluate_mol(selected_products)},
            device=self.model.device,
        )

        constraint_fn = self.constraint_fn or get_oracle("null")
        return State(
            coords=property_repr["ecfp4"]["fingerprint"],
            scores=torch.tensor(self.oracle_fn(selected_products), device=self.model.device),
            constraint_scores=torch.tensor(constraint_fn(selected_products), device=self.model.device),
            ages=torch.zeros(len(selected_products), device=self.model.device, dtype=torch.long),
            syntheses=selected_syntheses,
            products=selected_products,
        )

    def run(self) -> OptimTracker:
        tracker = OptimTracker()
        state = self._get_init_state()
        tracker.add(0, state)

        i = 1
        t_start = time.time()
        try:
            while len(tracker) < self.max_evals:
                state, explored = self.step_strategy(
                    state=state,
                    facade=self.facade,
                    model=self.model,
                    oracle_fn=self.oracle_fn,
                    constraint_fn=self.constraint_fn or get_oracle("null"),
                    cond_query=self.cond_query,
                )
                tracker.add(i, explored)

                print(
                    f"Step {i}: "
                    f"time = {time.time() - t_start:.1f}s, "
                    f"evals = {len(tracker)} max = {state.scores.max().item():.4f}, "
                    f"mean = {state.scores.mean().item():.4f}, "
                    f"p90 = {state.scores.quantile(0.9).item():.4f}, "
                    + (
                        f"max total = {state.total_scores.max().item():.4f}, "
                        if self.constraint_fn is not None
                        else ""
                    )
                    + f"moving_avg_top10 = {tracker.moving_top10_avg():.4f}, "
                    f"auc_top10({self.max_evals / 1000}k) = {tracker.auc_top10(self.max_evals):.4f}, "
                )

                if self.time_limit is not None and time.time() - t_start > self.time_limit:
                    print("Time limit exceeded. Stopping optimization.")
                    break
                i += 1
        except KeyboardInterrupt as e:
            if self.handle_interrupt:
                print("Interrupted by user... Saving results.")
            else:
                raise e

        return tracker
