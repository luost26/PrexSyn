# Troubleshooting

## PrexSyn Engine does not install

Published engine wheels target Linux x86-64. On another platform, use a compatible Linux environment or build the engine from source. The source build requires the toolchain listed in the [engine guide](../prexsyn-engine/index.md#installation).

## CUDA is unavailable

Pass `--device cpu` to an example script, or move the model to `cpu` in Python. CPU inference works but is much slower. The training script is GPU-only.

## A model or chemical-space file is missing

`AllInOneLoader` downloads missing assets only when the YAML contains the corresponding `remote.checkpoint_url` or `remote.chemical_space_url`. Check the configured local path and URL. Run from the repository root when using a shipped configuration.

## A checkpoint has size-mismatch errors

Use the exact chemical-space cache for which the checkpoint was trained. The checkpoint dimensions and token indices depend on the building-block and reaction libraries.

## Pathway image rendering fails

Install the Graphviz system package and confirm its `dot` executable is on `PATH`. Installing `pydot` alone is insufficient.
