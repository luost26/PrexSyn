# PrexSyn

PrexSyn generates molecules together with synthesis pathways built from a defined building-block library and reaction set. A decoder-only transformer generates a postfix notation of synthesis conditioned on a molecular fingerprint. [PrexSyn Engine](prexsyn-engine/index.md) converts the notation into products and pathways.

## What you can do

| Task | Input | Result |
| --- | --- | --- |
| Chemical-space projection | A molecule as SMILES, an RDKit molecule, or a PrexSyn Engine molecule | Synthesizable analogs ranked by fingerprint similarity |
| Descriptor-conditioned generation | An ECFP4 or FCFP4 fingerprint | Molecules and synthesis pathways |
| Molecular sampling | A Python scoring function | Synthesizable candidates optimized by the current genetic sampler |
| Model training | Building blocks and reaction SMARTS | A model trained from an on-the-fly C++ datastream |

“Synthesizable” here means constructible under the configured reaction templates and building blocks. It is not a guarantee of experimental success.

!!! note "Current v1 scope"
    General physicochemical-property conditioning and composite logical queries from the original paper are not available in v1. See [Paper and v1 differences](reproducibility/version-differences.md).

## Start here

1. [Install PrexSyn](getting-started/installation.md).
2. Run the [projection example](getting-started/examples.md#project-a-molecule).
3. Use the [Python API](getting-started/import.md) or [define a chemical space](customization/chemical-space.md).

## Resources

- [PrexSyn source](https://github.com/luost26/prexsyn)
- [PrexSyn Engine source](https://github.com/luost26/prexsyn-engine)
- [Data and model weights](https://huggingface.co/datasets/luost26/prexsyn-data/tree/main)
- [PrexSyn paper](https://arxiv.org/abs/2512.00384)

## Citation

```bibtex
@article{luo2025prexsyn,
  title   = {Efficient and Programmable Exploration of Synthesizable Chemical Space},
  author  = {Shitong Luo and Connor W. Coley},
  year    = {2025},
  journal = {arXiv preprint arXiv: 2512.00384}
}
```
