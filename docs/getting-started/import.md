# Use PrexSyn in your own project

PrexSyn is designed to be modular and easy to integrate into your own projects. To get started, install PrexSyn directly from this repository (a PyPI release is planned), which will automatically install all required dependencies:

```bash
pip install git+https://github.com/luost26/prexsyn.git
```

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

More examples can be found in the `scripts/examples` directory.
