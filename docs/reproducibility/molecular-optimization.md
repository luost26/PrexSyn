# Molecular optimization

The v1 repository includes the eight standard optimization tasks reported in Table 2 of the [PrexSyn paper](https://arxiv.org/abs/2512.00384): Amlodipine, Fexofenadine, Osimertinib, Perindopril, Ranolazine, Sitagliptin, Zaleplon, and Celecoxib rediscovery.

## Run the benchmark

Install the evaluation dependencies, then run the benchmark from the repository root:

```bash
uv sync --extra eval
uv run python scripts/benchmarks/optim.py --device cuda
```

The paper settings are the defaults: five runs per task, a budget of 10,000 unique oracle calls per run, an initial population of 500, 50 offspring per generation, and eight sampled pathways per fingerprint. Use one or more `--task` options for a subset:

```bash
uv run python scripts/benchmarks/optim.py \
    --task amlodipine \
    --task celecoxib_rediscovery
```

Independent seeds can be scheduled separately with `--run`; for example, `--num-runs 5 --run 3` writes `run_03.csv` using the same seed as run 3 of the default invocation.
After all scheduled runs finish, pass `--summarize-only` with the same task and run selection to consolidate their CSVs without loading the model.

The scoring functions come from the `prexsyn-third-party` PyPI package, which bundles GuacaMol. Generated products are cached by canonical SMILES, so repeated molecules do not consume the oracle budget.

## Outputs and paper comparison

Each run is written to `<out>/<task>/run_NN.csv` in oracle-call order. Existing complete runs are reused unless `--overwrite` is passed. The output root also contains:

- `runs.csv`: per-run AUC-Top-10, final Top-10, and best score;
- `comparison.csv`: per-task mean and population standard deviation, the paper value, and the difference from the paper;
- `config.json`: the settings used for the invocation.

The paper reports these PrexSyn AUC-Top-10 means (standard deviations in parentheses):

| Task | Paper AUC-Top-10 |
| --- | ---: |
| Amlodipine | 0.781 (0.023) |
| Fexofenadine | 0.837 (0.013) |
| Osimertinib | 0.855 (0.007) |
| Perindopril | 0.714 (0.010) |
| Ranolazine | 0.807 (0.009) |
| Sitagliptin | 0.471 (0.030) |
| Zaleplon | 0.504 (0.018) |
| Celecoxib rediscovery | 0.801 (0.005) |

The v1 implementation searches ECFP4 fingerprint space using the released paper checkpoint and chemical space. It does not restore the deprecated property-query or composite-query APIs. Sampling is stochastic, so compare aggregate statistics rather than expecting identical individual runs.
