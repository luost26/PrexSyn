# Quick examples

Run these commands from the repository root.

## Project a molecule

```bash
uv run python scripts/examples/projection.py --smiles "COc1ccc(-c2ccnc(Nc3ccccc3)n2)cc1"
```

The script samples 64 pathways, ranks their products by ECFP4 Tanimoto similarity, and prints the top 10 as YAML. The first run downloads the default model and chemical space.

Useful options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--num-samples` | `64` | Number of pathways sampled per target |
| `--top` | `10` | Maximum number of products printed |
| `--device` | `cuda` | PyTorch device, such as `cuda` or `cpu` |
| `--config` | Default released model YAML | Model and chemical-space configuration |

To render pathways, install Graphviz and add an output directory:

```bash
uv run python scripts/examples/projection.py \
    --smiles "COc1ccc(-c2ccnc(Nc3ccccc3)n2)cc1" \
    --draw-output-dir ./draw
```

Images are written as `synthesis_<rank>_sim<similarity>.png`.

![Projection example](./imgs/projection-example.png)

## Run the molecular sampler

The current sampler uses a genetic algorithm over ECFP4 fingerprints. The shipped example maximizes RDKit QED:

```bash
uv run python scripts/examples/sampling.py --out-fig ./sampling.png
```

It initializes a population, runs 20 generations, and prints the best and mean fitness at each step. `--out-fig` is optional and requires Graphviz. Use `--device cpu` only when CUDA is unavailable.

To optimize another objective, implement a callable that accepts a sequence of `(Synthesis, Molecule)` pairs and returns a one-dimensional NumPy array of scores. See [`scripts/examples/sampling.py`](https://github.com/luost26/prexsyn/blob/main/scripts/examples/sampling.py) for the complete interface.
