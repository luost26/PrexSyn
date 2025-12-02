import pathlib

import click
import torch
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from prexsyn.applications.analog import generate_analogs
from prexsyn.factories import load_model
from prexsyn.utils.draw import draw_synthesis
from prexsyn.samplers.basic import BasicSampler
from prexsyn_engine.synthesis import Synthesis
from prexsyn_engine.fingerprints import tanimoto_similarity


def indent_lines(text: str, level: int, indent: str = "  ") -> str:
    return "\n".join(indent * level + line for line in text.splitlines())


def synthesis_to_string(synthesis: Synthesis) -> str:
    replay = Synthesis()
    pfn_list = synthesis.get_postfix_notation().to_list()
    text_stack: list[str] = []
    for i, item in enumerate(pfn_list):
        if isinstance(item, Chem.Mol):
            replay.push_mol(item)
            smi = Chem.MolToSmiles(item, canonical=True)
            idx = item.GetProp("building_block_index")
            text = f"- SMILES: {smi}\n"
            text += f"  Building Block Index: {idx}\n"
            if item.HasProp("id"):
                name = item.GetProp("id")
                text += f"  ID: {name}\n"
            text_stack.append(text.strip())

        elif isinstance(item, rdChemReactions.ChemicalReaction):
            replay.push_reaction(item)
            prod_list = replay.top().to_list()
            prod_smi_set = set(Chem.MolToSmiles(mol, canonical=True) for mol in prod_list)
            num_reactants = item.GetNumReactantTemplates()

            idx = item.GetProp("reaction_index")
            text = f"- Reaction Index: {idx}\n"
            text += "  Possible Products:\n"
            product_text = "\n".join(f"- {smi}" for smi in prod_smi_set)
            text += indent_lines(product_text, 1) + "\n"
            text += "  Reactants:\n"
            reactant_text = "\n".join(text_stack[-num_reactants:])
            text += indent_lines(reactant_text, 1) + "\n"

            text_stack = text_stack[:-num_reactants]
            text_stack.append(text.strip())

    synthesis_text = "\n".join(text_stack)
    return synthesis_text


@click.command()
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="./data/trained_models/v1_converted.yaml",
)
@click.option("--smiles", type=str, required=True)
@click.option("--draw-output", type=click.Path(path_type=pathlib.Path), default=None)
@click.option("--top", type=int, default=10)
@click.option("--num-samples", type=int, default=64)
def main(
    model_path: pathlib.Path,
    smiles: str,
    draw_output: pathlib.Path | None,
    top: int,
    num_samples: int,
) -> None:
    torch.set_grad_enabled(False)
    facade, model = load_model(model_path, train=False)
    model = model.to("cuda")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    canonical_smi = Chem.MolToSmiles(mol, canonical=True)

    sampler = BasicSampler(
        model,
        token_def=facade.tokenization.token_def,
        num_samples=num_samples,
        max_length=16,
    )

    result = generate_analogs(
        facade=facade,
        model=model,
        sampler=sampler,
        fp_property=facade.property_set["ecfp4"],
        mol=mol,
    )

    visited: set[str] = set()
    result_list: list[tuple[Chem.Mol, Synthesis, float]] = []
    for synthesis in result["synthesis"]:
        if synthesis.stack_size() != 1:
            continue
        for prod in synthesis.top().to_list():
            prod_smi = Chem.MolToSmiles(prod, canonical=True)
            if prod_smi in visited:
                continue
            visited.add(prod_smi)
            sim = tanimoto_similarity(prod, mol, fp_type="ecfp4")
            result_list.append((prod, synthesis, sim))

    if draw_output is not None:
        draw_output.mkdir(parents=True, exist_ok=True)

    result_list.sort(key=lambda x: x[2], reverse=True)
    print(f"Input: {smiles}")
    print(f"Target (Canonical SMILES): {canonical_smi}")
    print("Results:")
    for i, (prod, synthesis, sim) in enumerate(result_list[:top]):
        smi = Chem.MolToSmiles(prod, canonical=True)
        print(f"- SMILES: {smi}")
        print(f"  Similarity: {sim:.4f}")
        print("  Synthesis:")
        print(indent_lines(synthesis_to_string(synthesis), 1) + "\n")

        if draw_output is not None:
            im = draw_synthesis(synthesis, show_intermediate=True, show_num_cases=True)
            im.save(draw_output / f"{i}.png")


if __name__ == "__main__":
    main()
