# PrexSyn



[[Paper]](https://arxiv.org/abs/2512.00384)
[[Documentation]](https://prexsyn.readthedocs.io)
[[Data]](https://huggingface.co/datasets/luost26/prexsyn-data/tree/main)

Work in progress...

## Installation

We highly recommend using pixi to setup the environment, which is way easier and faster.
If you are interested, please check out the [pixi-based installation guide](https://prexsyn.readthedocs.io/en/latest/getting-started/installation/). 
Alternatively, you can follow the steps below to setup the environment manually.

Create and activate conda (mamba) environment:

```bash
conda create -n prexsyn
conda activate prexsyn
```

Install [PrexSyn Engine](https://github.com/luost26/prexsyn-engine). This package is only available via conda for now. RDKit will be installed as a dependency in this step.

```bash
conda install luost26::prexsyn-engine
```

Setup PrexSyn package. PyTorch and other dependencies will be installed in this step.

```bash
pip install -e .
```
