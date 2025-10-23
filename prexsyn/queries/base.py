import abc

import sympy as sp
from rdkit import Chem

from prexsyn.data.struct import PropertyRepr
from prexsyn_engine.synthesis import Synthesis


class Node(abc.ABC):
    def __invert__(self) -> "Not":
        return Not(self)

    def __and__(self, other: "Node") -> "And":
        return And(self, other)

    def __or__(self, other: "Node") -> "Or":
        return Or(self, other)


class And(Node):
    def __init__(self, *children: "Node") -> None:
        self.children = children

    def __repr__(self) -> str:
        return "(" + " & ".join(repr(child) for child in self.children) + ")"


class Or(Node):
    def __init__(self, *children: "Node") -> None:
        self.children = children

    def __repr__(self) -> str:
        return "(" + " | ".join(repr(child) for child in self.children) + ")"


class Not(Node):
    def __init__(self, child: "Node") -> None:
        self.child = child

    def __repr__(self) -> str:
        return "~" + repr(self.child)


class Condition(Node, abc.ABC):
    @property
    def weight(self) -> float:
        return 1.0

    @abc.abstractmethod
    def get_property_repr(self) -> PropertyRepr: ...

    @abc.abstractmethod
    def score(self, synthesis: Synthesis, product: Chem.Mol) -> float: ...


def _condition_to_symbol(condition: Condition, mapping: dict[Condition, sp.Symbol]) -> sp.Symbol:
    if condition in mapping:
        return mapping[condition]
    name = f"c{len(mapping)}"
    sym = sp.Symbol(name)
    mapping[condition] = sym
    return sym


def _node_to_sympy_expr(node: Node, mapping: dict[Condition, sp.Symbol]) -> sp.Expr:
    if isinstance(node, Condition):
        return _condition_to_symbol(node, mapping)
    elif isinstance(node, And):
        return sp.And(*[_node_to_sympy_expr(child, mapping) for child in node.children])
    elif isinstance(node, Or):
        return sp.Or(*[_node_to_sympy_expr(child, mapping) for child in node.children])
    elif isinstance(node, Not):
        return sp.Not(_node_to_sympy_expr(node.child, mapping))
    else:
        raise ValueError(f"Unknown node type: {type(node)}")


def _sympy_expr_to_node(expr: sp.Expr, reverse_mapping: dict[sp.Symbol, Condition]) -> Node:
    if isinstance(expr, sp.Symbol):
        return reverse_mapping[expr]
    elif isinstance(expr, sp.And):
        return And(*[_sympy_expr_to_node(arg, reverse_mapping) for arg in expr.args])
    elif isinstance(expr, sp.Or):
        return Or(*[_sympy_expr_to_node(arg, reverse_mapping) for arg in expr.args])
    elif isinstance(expr, sp.Not):
        return Not(_sympy_expr_to_node(expr.args[0], reverse_mapping))
    else:
        raise ValueError(f"Unknown sympy expression type: {type(expr)}")


def to_dnf(node: Node) -> list[list[tuple[Condition, bool]]]:
    mapping: dict[Condition, sp.Symbol] = {}
    expr = _node_to_sympy_expr(node, mapping)
    reverse_mapping: dict[sp.Symbol, Condition] = {v: k for k, v in mapping.items()}

    dnf_expr = _sympy_expr_to_node(sp.to_dnf(expr, simplify=True), reverse_mapping)
    if not isinstance(dnf_expr, Or):
        dnf_expr = Or(dnf_expr)

    dnf: list[list[tuple[Condition, bool]]] = []
    for disjunct in dnf_expr.children:
        if isinstance(disjunct, And):
            conjuncts = list(disjunct.children)
        else:
            conjuncts = [disjunct]

        clause: list[tuple[Condition, bool]] = []
        for literal in conjuncts:
            if isinstance(literal, Not):
                condition = literal.child
                is_positive = False
            else:
                condition = literal
                is_positive = True
            if not isinstance(condition, Condition):
                raise ValueError("Expected Condition node")
            clause.append((condition, is_positive))
        dnf.append(clause)

    return dnf


if __name__ == "__main__":

    class TestCondition(Condition):
        def __init__(self, name: str) -> None:
            self.name = name

        def get_property_repr(self) -> PropertyRepr:
            return {}

        def score(self, synthesis: Synthesis, product: Chem.Mol) -> float:
            return 1.0

        def __repr__(self) -> str:
            return self.name

    expr = (
        TestCondition("A1")
        & TestCondition("A2")
        & (TestCondition("B") | ~TestCondition("C"))
        & ~(TestCondition("D") | TestCondition("E"))
        & ~(TestCondition("F") & ~TestCondition("G"))
    )
    print(expr)
    print(_node_to_sympy_expr(expr, {}))

    print(to_dnf(expr))
