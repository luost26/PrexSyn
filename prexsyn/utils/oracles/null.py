from typing import overload

from rdkit import Chem

from ._registry import register


@register
class null:
    @overload
    def __call__(self, mol: Chem.Mol) -> float: ...
    @overload
    def __call__(self, mol: list[Chem.Mol]) -> list[float]: ...

    def __call__(self, mol: list[Chem.Mol] | Chem.Mol) -> list[float] | float:
        if isinstance(mol, list):
            return [0.0] * len(mol)
        else:
            return 0.0
