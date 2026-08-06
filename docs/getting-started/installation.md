# Installation

PrexSyn requires Python 3.11 or newer. The published PrexSyn Engine wheels currently target Linux x86-64; CUDA is recommended for model inference and required by the training script.

## Install for the examples

```bash
git clone https://github.com/luost26/prexsyn.git
cd prexsyn
uv sync
```

[uv](https://docs.astral.sh/uv/) creates the environment from `pyproject.toml` and `uv.lock`. Verify both packages:

```bash
uv run python -c "import prexsyn, prexsyn_engine; print(prexsyn.__version__)"
```

Run commands from the repository root. Shipped configuration files use paths relative to that directory.

## Install in another project

```bash
python -m pip install "prexsyn @ git+https://github.com/luost26/prexsyn.git"
```

PrexSyn Engine is installed as a dependency. A PrexSyn release on PyPI is not yet available.

## Choose a device

Example scripts default to `--device cuda`. Use `--device cpu` when CUDA is unavailable, but expect substantially slower inference. To select a different PyTorch/CUDA build, follow the [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/#using-a-pytorch-index) before syncing the environment.

## Optional pathway images

Install the [Graphviz](https://graphviz.org/download/) system package to use `--draw-output-dir` or `item.get_image()`. Python dependencies alone are not enough because `pydot` calls the Graphviz executable.
