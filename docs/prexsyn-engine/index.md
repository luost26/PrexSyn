# PrexSyn Engine

PrexSyn Engine is the C++ backend used by PrexSyn. Version 1.1 provides Python bindings and statically links its C++ chemistry dependencies in published wheels.

It provides:

- molecule, reaction, and synthesis primitives;
- building-block, reaction, intermediate, and chemical-space libraries;
- random synthesis enumeration;
- ECFP4 and FCFP4 calculation;
- postfix-notation encoding and multithreaded detokenization;
- a multithreaded producer-consumer training data pipeline.

## Installation

PrexSyn installs a compatible engine automatically. To use the engine alone:

```bash
python -m pip install "prexsyn-engine>=1.1,<1.2"
```

Published wheels currently target CPython 3.11–3.14 on Linux x86-64. Building from source requires a C++20 toolchain, CMake 3.28 or newer, OpenMP, and the dependencies configured by the engine's CMake project.

## Molecules and fingerprints

```python
from prexsyn_engine import chemistry, descriptor

molecules = [
    chemistry.Molecule.from_smiles("CCO"),
    chemistry.Molecule.from_smiles("c1ccccc1"),
]

ecfp4 = descriptor.MorganFingerprint.ecfp4()
fingerprints = ecfp4(molecules)

print(molecules[0].smiles())
print(fingerprints.shape, fingerprints.dtype)
```

`Molecule.from_smiles()` canonicalizes valid input and raises `MoleculeError` for invalid SMILES. `Molecule.from_rdkit_mol()` and `to_rdkit_mol()` convert between engine and RDKit molecule objects.

## Chemical-space pipeline

A complete data-producing space is built in this order:

1. load building blocks and reactions;
2. construct `ChemicalSpace` with an empty `IntermediateLibrary`;
3. build building-block reactant lists;
4. generate intermediates;
5. build intermediate reactant lists;
6. serialize the result.

The main repository implements this sequence in `scripts/create_chemspace.py`. See [Defining chemical space](../customization/chemical-space.md) for the supported input formats and command.

During training, `DataPipeline.start_workers(seeds)` starts one producer per seed. `get(batch_size)` returns NumPy arrays keyed by descriptor name. Always call `stop_workers()` when managing a pipeline directly.

## Detokenization

`MultiThreadedDetokenizer` accepts an integer array shaped `(batch, sequence_length, 3)`. The last dimension stores token type, building-block index, and reaction index. It returns one `chemspace.Synthesis` per sequence; `products()` returns the final stack products.

See the [engine repository](https://github.com/luost26/prexsyn-engine) for C++ sources, type stubs, and executable API tests.
