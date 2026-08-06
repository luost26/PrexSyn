# Defining chemical space

A chemical space is a serialized set of building blocks, reactions, generated intermediates, and reactant-match lookup tables. Build one before training on a new catalog or reaction set.

## 1. Prepare building blocks

`scripts/create_chemspace.py` reads an SDF file. Each record may contain an `id` property; otherwise its canonical SMILES is used as the identifier. Duplicate identifiers are suffixed automatically.

Invalid molecules are skipped with a warning. Review the final loaded count before training.

## 2. Prepare reactions

Use a UTF-8 text file with one reaction SMARTS per line. Blank lines and lines beginning with `#` are ignored. Fields are tab-separated:

```text
<reaction SMARTS>\t<reaction name>\t<reactant 1 name>\t<reactant 2 name>
```

Only the SMARTS field is required. Missing reaction names become `RXN_<line number>`, and missing reactant names become `R0`, `R1`, and so on.

The repository's `data/reactions/rxn115.txt` is a working example.

## 3. Create a configuration

Copy the small configuration and change the `chemical_space` section:

```bash
cp configs/enamine-test-small_rxn115.yml configs/my_space.yml
```

```yaml
chemical_space:
  cache_path: data/chemical_spaces/my_space.chemspace
  bb_path: data/building_blocks/my_building_blocks.sdf
  rxn_path: data/reactions/my_reactions.txt
  building_block_selectivity_cutoff: 2
```

`building_block_selectivity_cutoff` rejects a building block when a reaction template matches it in more than the given number of ways. Omit it to use the engine default.

## 4. Build and inspect the cache

```bash
uv run python scripts/create_chemspace.py configs/my_space.yml
```

The script caches parsed input libraries as `.cache` files beside the SDF and reaction file. If an input file changes, remove its corresponding `.cache` before rebuilding so the new input is parsed.

Inspect the serialized result:

```python
from prexsyn_engine.chemspace import ChemicalSpace

stats = ChemicalSpace.peek("data/chemical_spaces/my_space.chemspace")
print(stats.num_building_blocks)
print(stats.num_reactions)
print(stats.num_intermediates)
```

## 5. Train a matching model

A released checkpoint cannot be used with a custom cache. The model's embedding and output layers depend on the number and order of building blocks and reactions. Continue with [Training your own model](model.md).
