# How PrexSyn works

## Chemical space

A PrexSyn chemical space contains:

- purchasable or otherwise available building blocks;
- reaction templates that define allowed transformations;
- generated intermediates and lookup tables that match molecules to reaction inputs.

Changing the building-block or reaction order changes token indices. A checkpoint therefore belongs to one exact chemical-space cache.

## Postfix notation of synthesis

PrexSyn represents a pathway as a stack-based sequence. A building-block token pushes a molecule onto the stack. A reaction token consumes the required reactants and pushes its products. The representation records both the final molecule and the route used to construct it.

## Projection flow

For a target molecule, PrexSyn:

1. computes an ECFP4 or FCFP4 fingerprint;
2. samples postfix token sequences from the transformer;
3. uses PrexSyn Engine to reconstruct each pathway and its products;
4. ranks products by Tanimoto similarity to the target fingerprint.

One sampled pathway can produce more than one product, and invalid or incomplete samples may produce none. The number of returned products can therefore differ from `num_samples`.

## Training flow

PrexSyn Engine continuously enumerates random pathways, calculates fingerprints, and encodes pathways as tensors in C++ worker threads. The Python training loop consumes these batches directly; it does not require a pre-enumerated molecule dataset.

The current registry contains `ecfp4` and `fcfp4`. The model is trained with one descriptor type per sample and can later generate pathways from either descriptor.
