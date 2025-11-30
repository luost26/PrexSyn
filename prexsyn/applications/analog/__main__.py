import pathlib
from collections.abc import Sequence

import click
import pandas as pd
import torch
from rdkit import Chem
from tqdm.auto import tqdm

from prexsyn.factories import load_model
from prexsyn.samplers.basic import BasicSampler

from .analog import generate_analogs
from .data import AnalogGenerationDatabase


@click.command()
@click.option("--model", "model_path", type=click.Path(exists=True, path_type=pathlib.Path), required=True)
@click.option("-i", "csv_path", type=click.Path(exists=True, path_type=pathlib.Path), required=True)
@click.option("-o", "output_path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--max-length", default=16)
@click.option("--num-samples", default=128)
@click.option("--device", default="cuda")
def analog_generation_cli(
    model_path: pathlib.Path,
    csv_path: pathlib.Path,
    output_path: pathlib.Path,
    max_length: int,
    num_samples: int,
    device: torch.device | str,
) -> None:
    torch.set_grad_enabled(False)
    df = pd.read_csv(csv_path)
    smi_list: Sequence[str] = df["SMILES"].tolist()

    with AnalogGenerationDatabase(output_path) as db:
        existing = set(db.keys())
        smi_list = [smi for smi in smi_list if smi not in existing]

        try:
            if len(smi_list) == 0:
                print("All SMILES already processed.")
                raise KeyboardInterrupt  # this is just a hack to jump to the final print

            facade, model = load_model(model_path)
            model = model.eval().to(device)
            sampler = BasicSampler(
                model,
                token_def=facade.tokenization.token_def,
                num_samples=num_samples,
                max_length=max_length,
            )
            with tqdm(total=len(smi_list)) as pbar:
                for smi in smi_list:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        print(f"Invalid SMILES: {smi}")
                        continue
                    entry = generate_analogs(
                        facade=facade,
                        model=model,
                        sampler=sampler,
                        fp_property=facade.property_set["ecfp4"],
                        mol=mol,
                    )
                    db[smi] = entry
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "count": len(db),
                            "avg_sim": f"{db.get_average_similarity():.4f}",
                            "recons": f"{db.get_reconstruction_rate():.4f}",
                        }
                    )
        except KeyboardInterrupt:
            print("Interrupted. Saving progress...")

        print("[Results]")
        print(f"Model path: {model_path}")
        print(f"CSV path: {csv_path}")
        print(f"Output path: {output_path}")
        print(f"Total entries: {len(db)}")
        print(f"Average similarity:  {db.get_average_similarity():.4f}")
        print(f"Reconstruction rate: {db.get_reconstruction_rate():.4f}")
        time_mean, time_std = db.get_time_statistics()
        print(f"Time per target: {time_mean:.4f} +/- {time_std:.4f} sec")
        print(
            "[NOTE] The reported time only includes the model inference time (`generate_analogs` call), "
            "which is shorter than the actual time taken. "
            "Extra time is mainly taken for loading the model and saving outputs to the database."
        )


if __name__ == "__main__":
    analog_generation_cli()
