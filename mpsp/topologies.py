"""Topologias T1..T5: só mudam o agrupamento dos mesmos fragmentos de prompt.

O primeiro nó sempre emite `resumo_processo`; os demais recebem dele a
classificação e os nomes por papel (contexto). O custo de uma topologia é a soma dos custos dos seus nós.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .schemas import ROLES


@dataclass(frozen=True)
class Node:
    name: str
    emits: Tuple[str, ...]


_ALL = ("resumo_processo", "inquerito") + ROLES

TOPOLOGIES: Dict[str, List[Node]] = {
    "T1": [Node("tudo", _ALL)],
    "T2": [Node("crime_inquerito", ("resumo_processo", "inquerito")), Node("pessoas", ROLES)],
    "T3": [Node("crime", ("resumo_processo",)), Node("inquerito", ("inquerito",)), Node("pessoas", ROLES)],
    "T4": [Node("crime_inquerito", ("resumo_processo", "inquerito"))]
          + [Node(r, (r,)) for r in ROLES],
    "T5": [Node("crime", ("resumo_processo",)), Node("inquerito", ("inquerito",))]
          + [Node(r, (r,)) for r in ROLES],
}


def is_natural_death(ctx: dict) -> bool:
    return bool(ctx.get("classificacao")) and "naturais" in str(ctx["classificacao"]).lower()


def should_run(node: Node, ctx: dict, early_exit: bool = True) -> bool:
    """Nós de pessoas são pulados em morte natural (early_exit) ou sem nomes."""
    roles = [r for r in node.emits if r in ROLES]
    if not roles:
        return True
    if early_exit and is_natural_death(ctx):
        return False
    return any(ctx.get(r) for r in roles)
