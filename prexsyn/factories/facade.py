from omegaconf import DictConfig

from prexsyn_engine.detokenizer import Detokenizer

from .chemical_space import ChemicalSpace
from .property import PropertySet
from .tokenization import Tokenization


class Facade:
    def __init__(self, cfg: DictConfig):
        self._cfg = cfg
        self._chemical_space = ChemicalSpace.from_config(cfg.chemical_space)
        self._property_set = PropertySet.from_config(cfg.property)
        self._tokenization = Tokenization.from_config(cfg.tokenization)

    @property
    def chemical_space(self) -> ChemicalSpace:
        return self._chemical_space

    @property
    def property_set(self) -> PropertySet:
        return self._property_set

    @property
    def tokenization(self) -> Tokenization:
        return self._tokenization

    def get_detokenizer(self) -> Detokenizer:
        csd = self.chemical_space.get_csd()
        return Detokenizer(
            building_blocks=csd.get_primary_building_blocks(),
            reactions=csd.get_reactions(),
            token_def=self.tokenization.token_def,
        )
