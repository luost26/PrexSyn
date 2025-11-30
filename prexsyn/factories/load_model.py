import pathlib
from omegaconf import OmegaConf, DictConfig
from .facade import Facade
from prexsyn.models.prexsyn import PrexSyn
from prexsyn.utils.download import download
from .chemical_space import check_chemical_space_data_dir, chemical_space_data_files


_remote_model_url = "https://huggingface.co/datasets/luost26/prexsyn-data/resolve/main/trained_models"
_remote_chemical_space_url = "https://huggingface.co/datasets/luost26/prexsyn-data/resolve/main/chemical_spaces"


def load_model(path: pathlib.Path | str, train: bool = False) -> tuple[Facade, PrexSyn]:
    path = pathlib.Path(path)
    c_path = path.with_suffix(".yaml")
    if not c_path.exists():
        raise FileNotFoundError(f"Config file not found: {c_path}")
    config = OmegaConf.load(c_path)
    if not isinstance(config, DictConfig):
        raise ValueError("Config file does not contain a valid DictConfig.")

    m_path = path.with_suffix(".ckpt")
    if not m_path.exists():
        url = f"{_remote_model_url}/{m_path.name}"
        print(f"Model checkpoint not found locally at {m_path}, trying to download from {url}...")
        download(url, m_path)

    cs_dir = pathlib.Path(config.chemical_space.cache_dir)
    cs_dir.mkdir(parents=True, exist_ok=True)
    cs_name = cs_dir.name
    if not check_chemical_space_data_dir(cs_dir):
        print(
            f"Chemical space data not found locally at {cs_dir}, "
            f"trying to download from {_remote_chemical_space_url}..."
        )
        for file_name in chemical_space_data_files:
            url = f"{_remote_chemical_space_url}/{cs_name}/{file_name}"
            download(url, cs_dir / file_name, desc=f"Downloading {file_name}")

    facade = Facade.from_file(c_path)
    model = facade.load_model_from_checkpoint(m_path)
    if train:
        model.train()
    else:
        model.eval()
    return facade, model
