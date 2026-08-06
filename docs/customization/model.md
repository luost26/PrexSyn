# Training your own model

The training script uses PyTorch Lightning and the PrexSyn Engine datastream. It requires one or more CUDA GPUs.

## Configure training

Start with [`configs/enamine-test-small_rxn115.yml`](https://github.com/luost26/prexsyn/blob/main/configs/enamine-test-small_rxn115.yml). It is a complete, runnable example that uses the bundled small building-block set and `rxn115` reactions:

```bash
cp configs/enamine-test-small_rxn115.yml configs/my_space.yml
```

Edit the copy for your chemical space and training resources. For the full released architecture, refer to [`configs/enamine2310_rxn115_202511.yml`](https://github.com/luost26/prexsyn/blob/main/configs/enamine2310_rxn115_202511.yml); [`configs/enamine2310_rxn115_bsz512.yml`](https://github.com/luost26/prexsyn/blob/main/configs/enamine2310_rxn115_bsz512.yml) is the lower-batch-size variant.

Each configuration contains these required sections:

| Section | Controls |
| --- | --- |
| `note` | Configuration version, display name, and description |
| `chemical_space` | Cache and source-library paths |
| `descriptors` | Descriptor names and prompt-token counts |
| `featurizer` | Maximum postfix length and optional token IDs |
| `model` | Transformer and building-block embedding dimensions |
| `training` | Batch size, validation interval, workers, losses, optimizer, and scheduler |

The current descriptor registry accepts `ecfp4` and `fcfp4`. The high-level inference sampler uses a maximum postfix length of 16, so keep `featurizer.max_length: 16` when training for `MoleculeProjector`.

`training.batch_size` is the batch requested by each training process. Reduce it with the model dimensions if GPU memory is limited. `training.data_pipeline_num_cpus` controls physical CPU cores; the code expands this to the corresponding number of logical threads.

## Start a run

First [build the configured chemical space](chemical-space.md), then run:

```bash
uv run python scripts/train.py configs/my_space.yml --devices 1 --max-epochs 1000
```

The trainer always uses the GPU accelerator. It logs to `logs/<config-name>-<timestamp>/` and uses Weights & Biases. Run `wandb login` for online logging, or set `WANDB_MODE=offline`:

```bash
WANDB_MODE=offline uv run python scripts/train.py configs/my_space.yml --devices 1
```

Checkpoints are saved every 25 epochs and on exceptions. Resume a Lightning run with:

```bash
uv run python scripts/train.py configs/my_space.yml \
    --devices 1 \
    --ckpt-path logs/<run>/last.ckpt
```

## Export for `AllInOneLoader`

Lightning checkpoints include trainer state and prefix model parameters with `model.`. Export the raw model state expected by `AllInOneLoader`:

```python
from pathlib import Path

import torch

lightning_path = Path("logs/<run>/last.ckpt")
output_path = Path("data/trained_models/my_space.ckpt")

checkpoint = torch.load(lightning_path, map_location="cpu", weights_only=False)
model_state = {
    key.removeprefix("model."): value
    for key, value in checkpoint["state_dict"].items()
    if key.startswith("model.")
}
torch.save(model_state, output_path)
```

Copy the training YAML to `data/trained_models/my_space.yml` and keep its `chemical_space.cache_path` pointed at the training cache. The YAML and `.ckpt` must share a filename stem.

The `--ckpt-path` option resumes Lightning training; it does not accept the raw checkpoint used by `AllInOneLoader`.
