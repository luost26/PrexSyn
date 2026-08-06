# Chemical space projection

The v1 repository includes the projection benchmark used for Table 1 of the [PrexSyn paper](https://arxiv.org/abs/2512.00384).

## Run the full benchmark

```bash
uv run python scripts/benchmarks/projection.py
```

The script evaluates the included 1,000-molecule Enamine REAL and ChEMBL subsets with 64, 128, and 256 samples per target. It performs five runs for every dataset/sample-count pair: 30 runs and 30,000 projections in total.

The default device is CUDA. Useful options are:

```bash
uv run python scripts/benchmarks/projection.py \
    --device cuda \
    --num-runs 5 \
    --out ./outputs/benchmarks/analog
```

Each target result is cached in an LMDB file under the output directory. Rerunning the same settings resumes from cached entries. Delete or choose another output directory when you need an independent rerun.

After all runs finish, the script writes `summary.csv` and prints grouped similarity, reconstruction-rate, and timing statistics.

## Paper settings

The paper reports results from one NVIDIA RTX 4090. Sampling is stochastic, and timing depends on the GPU, CPU, storage, and system load. Compare aggregate means and standard deviations rather than expecting identical rows.

The released configuration downloads the paper checkpoint and its matching Enamine/Rxn115 chemical-space cache automatically.
