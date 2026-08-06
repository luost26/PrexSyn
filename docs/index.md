# PrexSyn

## Introduction

PrexSyn is an efficient, accurate, and programmable framework for synthesizable molecular design.

It uses a decoder-only transformer to autoregressively generate *postfix notations of synthesis*[^chemprojector]: a molecular representation based on chemical reactions and purchasable building blocks. Generation is conditioned on molecular descriptors.

PrexSyn was trained on a billion-scale datastream of postfix notations paired with molecular descriptors using two GPUs and 32 CPU cores in two days. This scale is enabled by [PrexSyn Engine](prexsyn-engine/index.md), a real-time, high-throughput C++ data generation pipeline.

[^chemprojector]: *Projecting Molecules into Synthesizable Chemical Spaces*. [https://arxiv.org/abs/2406.04628](https://arxiv.org/abs/2406.04628)

!!! important "Need the exact paper features?"
    Use the [`dev-v0` branch](https://github.com/luost26/prexsyn/tree/dev-v0). General physicochemical-property conditioning and composite logical queries from the original paper are deprecated in v1. See [Paper and v1 differences](reproducibility/version-differences.md).

“Synthesizable” means constructible under the configured reaction templates and building blocks. It is not a guarantee of experimental success.

## Capabilities

| Capability | Input | Output |
| :---: | :---: | :---: |
| **Chemical-space projection** | ![Molecule used as projection input](imgs/proj-in.png)<br>Graph or SMILES | ![Synthesizable analog and pathway](imgs/proj-out.png)<br>Ranked analogs and pathways |
| **Fingerprint/descriptor-based generation** | ![Fingerprint used as generation input](imgs/fp-in.png)<br>ECFP4 or FCFP4 | ![Generated molecule and pathway](imgs/proj-out.png)<br>Molecules and pathways |
| **Molecular sampling** | ![Scoring function used for sampling](imgs/sample-in.png)<br>Scoring function | ![Optimized synthesizable molecules](imgs/sample-out.png)<br>Optimized candidates |

## Performance

The following figures show the results reported in the PrexSyn paper. The projection benchmark is maintained in v1. Migration of the optimization benchmark to v1 is work in progress.

| Capability | Result |
| --- | :---: |
| Record-high accuracy and speed in chemical-space projection and fingerprint/descriptor-based generation | ![Projection performance comparison](imgs/projection-compare.png) |
| Record-high sample efficiency in molecular sampling against scoring functions | ![Molecular sampling performance comparison](imgs/sampling-compare-1.png) |

## Start here

1. [Install PrexSyn](getting-started/installation.md).
2. Run the [projection example](getting-started/examples.md#project-a-molecule).
3. Use the [Python API](getting-started/import.md) or [define a chemical space](customization/chemical-space.md).

## Resources

### Repositories

- **PrexSyn**: [https://github.com/luost26/prexsyn](https://github.com/luost26/prexsyn)
- **PrexSyn Engine**: C++ backend for high-throughput training data generation and synthesis detokenization. [https://github.com/luost26/prexsyn-engine](https://github.com/luost26/prexsyn-engine)
- **Data and model weights**: Preprocessed chemical spaces and trained model weights. [https://huggingface.co/datasets/luost26/prexsyn-data/tree/main](https://huggingface.co/datasets/luost26/prexsyn-data/tree/main)

### Papers and documentation

- **PrexSyn paper**: *Efficient and Programmable Exploration of Synthesizable Chemical Space*. [https://arxiv.org/abs/2512.00384](https://arxiv.org/abs/2512.00384)
- **PrexSyn documentation**: [https://prexsyn.readthedocs.io](https://prexsyn.readthedocs.io)

### Community

- **MIT Coley Research Group**: [https://coley.mit.edu/](https://coley.mit.edu/)

## Citation

```bibtex
@article{luo2025prexsyn,
  title   = {Efficient and Programmable Exploration of Synthesizable Chemical Space},
  author  = {Shitong Luo and Connor W. Coley},
  year    = {2025},
  journal = {arXiv preprint arXiv: 2512.00384}
}
```
