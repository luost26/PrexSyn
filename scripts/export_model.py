from collections.abc import Mapping
from pathlib import Path

import click
import torch


MODEL_PREFIX = "model."


def extract_model_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Lightning checkpoint must be a mapping.")

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Lightning checkpoint does not contain a 'state_dict' mapping.")

    model_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if isinstance(key, str) and key.startswith(MODEL_PREFIX):
            model_state[key.removeprefix(MODEL_PREFIX)] = value

    if not model_state:
        raise ValueError(f"No parameters with the '{MODEL_PREFIX}' prefix were found.")

    return model_state


def convert_checkpoint(input_path: Path, output_path: Path) -> int:
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    model_state = extract_model_state_dict(checkpoint)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, output_path)
    return len(model_state)


@click.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_path", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--force", is_flag=True, help="Overwrite the output file without confirmation.")
def main(input_path: Path, output_path: Path, force: bool):
    """Export a Lightning checkpoint for use with AllInOneLoader."""
    if output_path.exists() and not force:
        click.confirm(f"Output file {output_path} already exists. Overwrite it?", abort=True)

    try:
        parameter_count = convert_checkpoint(input_path, output_path)
    except (TypeError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(f"Saved {parameter_count} model tensors to {output_path}")


if __name__ == "__main__":
    main()

