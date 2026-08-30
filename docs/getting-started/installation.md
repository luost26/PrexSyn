# Installation

PrexSyn requires Python 3.11 or newer. The published PrexSyn Engine wheels currently target Linux x86-64; CUDA is recommended for model inference and required by the training script.

## Install from PyPI

For use in your own project, install the latest release from PyPI:

```bash
python -m pip install prexsyn
```

PrexSyn Engine and the other required dependencies are installed automatically. Verify the installation:

```bash
python -c "import prexsyn, prexsyn_engine; print(prexsyn.__version__)"
```

## Install from source

Clone the repository and use [uv](https://docs.astral.sh/uv/) to create an environment from `pyproject.toml` and `uv.lock`:

```bash
git clone https://github.com/luost26/prexsyn.git
cd prexsyn
uv sync
```

Verify the source installation:

```bash
uv run python -c "import prexsyn, prexsyn_engine; print(prexsyn.__version__)"
```

Run commands from the repository root. Shipped configuration files use paths relative to that directory.

## Choose a device

Example scripts default to `--device cuda`. Use `--device cpu` when CUDA is unavailable, but expect substantially slower inference. To select a different PyTorch/CUDA build, follow the [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/#using-a-pytorch-index) before syncing the environment.

## Optional pathway images

Install the [Graphviz](https://graphviz.org/download/) system package to use `--draw-output-dir` or `item.get_image()`. Python dependencies alone are not enough because `pydot` calls the Graphviz executable.
