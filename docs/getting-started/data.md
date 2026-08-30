# Data and model weights

## Automatic download

The default loader downloads two files on first use:

- `data/chemical_spaces/enamine2310_rxn115.chemspace`
- `data/trained_models/enamine2310_rxn115_202511.ckpt`

The URLs and local paths come from `data/trained_models/enamine2310_rxn115_202511.yml`. Trigger both downloads with:

```bash
uv run python scripts/examples/projection.py --smiles "COc1ccc(-c2ccnc(Nc3ccccc3)n2)cc1"
```

The loader downloads a missing file only when its corresponding `remote` URL is set. Existing files are reused.

## Use another model configuration

Pass a local path or an HTTP(S) URL to `--config`:

```bash
uv run python scripts/examples/projection.py \
    --config ./data/trained_models/enamine2310_rxn115_202511.yml \
    --smiles "COc1ccc(-c2ccnc(Nc3ccccc3)n2)cc1"
```

For a remote configuration, `AllInOneLoader` first stores the YAML under `data/trained_models/remote/`. It then resolves the checkpoint beside that downloaded YAML and uses the chemical-space path written in the configuration.

## Manual download

Files are also available in the [PrexSyn data repository](https://huggingface.co/datasets/luost26/prexsyn-data/tree/main). Preserve the paths and filenames used by the YAML, or update `checkpoint_url`, `chemical_space_url`, and `chemical_space.cache_path` together.

Do not mix a checkpoint with another chemical-space cache. Building blocks and reactions are represented by their indices, so even reordered libraries are incompatible.
