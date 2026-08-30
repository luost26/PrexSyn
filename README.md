# PrexSyn

[![arXiv](https://img.shields.io/badge/arXiv-2512.00384-b31b1b.svg)](https://arxiv.org/abs/2512.00384)
[![readthedocs](https://app.readthedocs.org/projects/prexsyn/badge/?version=v0.1.0)](https://prexsyn.readthedocs.io)
[![data](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-blue)](https://huggingface.co/datasets/luost26/prexsyn-data/tree/main)

PrexSyn is an efficient, accurate, and programmable framework for exploring synthesizable chemical space.

PrexSyn is based on a decoder-only transformer architecture that autoregressively generates [*postfix notations of
synthesis*](https://arxiv.org/abs/2406.04628) (a molecular representation based on chemical reactions and purchasable building blocks) conditioned on molecular descriptors.


PrexSyn is trained on a billion-scale datastream of postfix notations paired with molecular descriptors using only two GPUs and 32 CPU cores in two days. This is made possible by [PrexSyn Engine](https://github.com/luost26/prexsyn-engine), a real-time, high-throughput C++-based data generation pipeline.

**Note:** features described in the original paper (e.g. property-based queries) are not supported in the current version (v1) of PrexSyn for now. If you need these features, please switch to the [v0 branch](https://github.com/luost26/prexsyn/tree/dev-v0).

[[Documentation]](https://prexsyn.readthedocs.io)
[[Paper]](https://arxiv.org/abs/2512.00384)
[[PrexSyn Engine]](https://github.com/luost26/prexsyn-engine)
[[Data and Model Weights]](https://huggingface.co/datasets/luost26/prexsyn-data/tree/main)

## Capabilities

| Capability | Input | Output | Performance |
| :---: | :---: | :---: | :---: |
| **Chemical space projection** | ![](https://raw.githubusercontent.com/luost26/prexsyn/refs/heads/main/docs/imgs/proj-in.png) | ![](https://raw.githubusercontent.com/luost26/prexsyn/refs/heads/main/docs/imgs/proj-out.png) | ![](https://raw.githubusercontent.com/luost26/prexsyn/refs/heads/main/docs/imgs/projection-compare.png) |
| **Fingerprint/descriptor based generation** | ![](https://raw.githubusercontent.com/luost26/prexsyn/refs/heads/main/docs/imgs/fp-in.png) | ![](https://raw.githubusercontent.com/luost26/prexsyn/refs/heads/main/docs/imgs/proj-out.png) | ![](https://raw.githubusercontent.com/luost26/prexsyn/refs/heads/main/docs/imgs/projection-compare.png) |
| **Molecular sampling** | ![](https://raw.githubusercontent.com/luost26/prexsyn/refs/heads/main/docs/imgs/sample-in.png) | ![](https://raw.githubusercontent.com/luost26/prexsyn/refs/heads/main/docs/imgs/sample-out.png) | ![](https://raw.githubusercontent.com/luost26/prexsyn/refs/heads/main/docs/imgs/sampling-compare-1.png) |


## Usage

### Documentation

Please refer to the [documentation](https://prexsyn.readthedocs.io) for detailed usage instructions on installation, data setup, reproducibility, and customization.

### Quick example

To run a quick example, make sure [uv](https://docs.astral.sh/uv/) is installed, then clone this repository. The command below is **all you need** to get started. No need to manually configure or download anything! On the first run, the preprocessed chemical space and model checkpoints will be downloaded automatically.

```
uv run python scripts/examples/projection.py \
    --smiles "COc1ccc(-c2ccnc(Nc3ccccc3)n2)cc1" \
    --draw-output-dir ./draw
```

The diagrams of the synthesis pathways will be saved in the `./draw` directory.

![examples](https://raw.githubusercontent.com/luost26/prexsyn/refs/heads/main/docs/getting-started/imgs/projection-example.png)

If you need to customize the environment (e.g., specific PyTorch/CUDA versions), please refer to the [installation instructions](https://prexsyn.readthedocs.io/en/latest/getting-started/installation/) for guidance.

### Use PrexSyn in your own project

PrexSyn is designed to be modular and easy to integrate into your own projects.

#### From PyPI

Install the latest release from PyPI:

```bash
python -m pip install prexsyn
```

#### From source

Install the latest source directly from GitHub:

```bash
python -m pip install "prexsyn @ git+https://github.com/luost26/prexsyn.git"
```

Both options install PrexSyn Engine and the other required dependencies automatically. See the [installation instructions](https://prexsyn.readthedocs.io/en/latest/getting-started/installation/) for more details.

The example below demonstrates how to use PrexSyn to generate synthesis pathways for a target SMILES string:

```python
from prexsyn.shortcuts import AllInOneLoader, MoleculeProjector

config_path = "./data/trained_models/enamine2310_rxn115_202511.yml"

loader = AllInOneLoader(config_path)
projector = MoleculeProjector(
    model=loader.model().to("cuda").eval(),
    detokenizer=loader.detokenizer(),
    descriptor="ecfp4",
    num_samples=16,
)

result = projector.one("COc1ccc(-c2ccnc(Nc3ccccc3)n2)cc1")
for i, item in enumerate(result.items):
    print(item.get_tree())  # print the synthesis tree in python dict format

    img = item.get_image()
    img.save(f"output_{i}.png")  # save the synthesis tree diagram as a PNG image
    img.close()
```

More examples can be found in the [`scripts/examples`](./scripts/examples) directory.

## Upgrade to PrexSyn v1

We have substantially refactored both the PrexSyn codebase and the [PrexSyn Engine](https://github.com/luost26/prexsyn-engine) to improve usability, performance, and extensibility. Key updates include:

- ✅ **Improved usability:** PrexSyn Engine is now available via PyPI.
- ✅ **Higher performance and stability:** Data generation is now approximately 2× faster than reported in the paper, with improved robustness thanks to a more reliable compilation pipeline.
- ✅ **Greater flexibility:** Chemical space definitions and training workflows are now easier to customize for new use cases.
- ✅ **Cleaner interfaces:** Simplified and more consistent APIs for projection, fingerprint/descriptor-based generation, and sampling.
- ⚠️ Some features described in the original paper (mostly property-based queries) are no longer supported in the current version of PrexSyn. If you need these features, please use the [v0 branch](https://github.com/luost26/prexsyn/tree/dev-v0).


## Citation

```bibtex
@article{luo2025prexsyn,
  title   = {Efficient and Programmable Exploration of Synthesizable Chemical Space},
  author  = {Shitong Luo and Connor W. Coley},
  year    = {2025},
  journal = {arXiv preprint arXiv: 2512.00384}
}

@inproceedings{luo2024chemprojector,
  title={Projecting Molecules into Synthesizable Chemical Spaces},
  author={Shitong Luo and Wenhao Gao and Zuofan Wu and Jian Peng and Connor W. Coley and Jianzhu Ma},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}
```
