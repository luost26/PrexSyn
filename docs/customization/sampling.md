# Sampling molecules using your scoring function

The genetic sampler searches ECFP4 fingerprint space and uses PrexSyn to map each fingerprint to a molecule and synthesis pathway. Candidate selection maximizes a Python scoring function.

Start from the runnable [`scripts/examples/sampling.py`](https://github.com/luost26/prexsyn/blob/main/scripts/examples/sampling.py) example:

```bash
cp scripts/examples/sampling.py scripts/examples/my_sampling.py
```

## Define the scoring function

The scoring function receives valid `(Synthesis, Molecule)` pairs and returns one score per pair. Higher scores are better. Preserve the input order and return a one-dimensional NumPy array.

The shipped example maximizes RDKit QED:

```python
from collections.abc import Sequence

import numpy as np
from rdkit.Chem import QED

from prexsyn_engine.chemistry import Molecule
from prexsyn_engine.chemspace import Synthesis


class ExampleScoringFunction:
    def __call__(self, phenotypes: Sequence[tuple[Synthesis, Molecule]]) -> np.ndarray:
        scores = [QED.qed(molecule.to_rdkit_mol()) for _, molecule in phenotypes]
        return np.asarray(scores, dtype=float)
```

Replace the calculation inside `__call__` with your scoring function. It may use the product molecule, its synthesis pathway, or both. Batch expensive evaluations where possible because the function is called once for every newly generated population.

## Run sampling

The copied script loads the released model, initializes a population, and runs 20 generations:

```bash
uv run python scripts/examples/my_sampling.py --device cuda
```

The script prints the best and mean fitness after every generation. Use `--config` to select another compatible model and chemical space.

Add this after the optimization loop to print the best retained molecule:

```python
best_index = int(np.argmax(ppl.fitnesses))
_, best_molecule = ppl.phenotypes[best_index]

print(best_molecule.smiles())
print(float(ppl.fitnesses[best_index]))
```

## Tune the search

The main controls in the example are:

| Setting | Example value | Effect |
| --- | ---: | --- |
| `MoleculeProjector.num_samples` | `8` | Pathways sampled for each fingerprint |
| `initialize(size=...)` | `100` | Initial population size; by default the implementation attempts `2 × size` embryo fingerprints and retains the best unique candidates |
| Number of `evolve` calls | `20` | Generations |
| `evolve(k=...)` | `50` | Children generated and candidates retained after each generation |
| `evolve(t=...)` | `0.5` | Selection temperature; lower positive values favor high-scoring parents more strongly |

Larger populations and more generations require more scoring-function evaluations. The process is stochastic unless you seed NumPy and PyTorch.

!!! note "Current limitation"
    The genetic sampler currently supports boolean descriptors only. The shipped example uses ECFP4.
