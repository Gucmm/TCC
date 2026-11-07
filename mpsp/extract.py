"""Executa uma topologia sobre documentos e devolve a predição canônica.

`extract_batch` roda a topologia nó a nó para VÁRIOS documentos de uma vez (cada nó vira um lote
para o backend); `extract_document` é o caso de um documento, pelo mesmo código.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from . import canonical
from .backends.base import Backend
from .prompts import build_prompt
from .schemas import InqueritoTotal, make_output_model
from .topologies import TOPOLOGIES, Node, should_run

Shot = Dict[str, Any]   # {"doc_id", "text", "canon"}


@dataclass
class ExtractionResult:
    doc_id: str
    prediction: Dict[str, Any]
    ok: bool                         # saída final válida no schema canônico
    nodes: List[dict] = field(default_factory=list)   # nome, emits, tokens, segundos, erro
    context: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def _node_shots(node: Node, shots: Sequence[Shot], early_exit: bool) -> List[dict]:
    out = []
    for s in shots:
        canon = s["canon"]
        ctx = canonical.context_from(canon)
        if node.emits[0] != "resumo_processo" and not should_run(node, ctx, early_exit):
            continue      # exemplo sem nada a mostrar neste nó
        out.append({"text": s["text"],
                    "ctx": None if "resumo_processo" in node.emits else ctx,
                    "output": canonical.project(canon, node.emits)})
    return out


def extract_batch(items: Sequence[dict], backend: Backend, topology: str = "T1", *,
                  early_exit: bool = True) -> List[ExtractionResult]:
    """`items`: [{"doc_id", "text", "shots"?, "oracle_context"?}].

    `oracle_context` (E2): usa classificação/nomes do GT nos nós seguintes, isolando a propagação
    de erro do primeiro nó."""
    S = [{"state": {}, "meta": [], "ctx": {}, "alive": True} for _ in items]
    for i, node in enumerate(TOPOLOGIES[topology]):
        run = []
        for k, s in enumerate(S):
            if not s["alive"]:
                continue
            if i > 0 and not should_run(node, s["ctx"], early_exit):
                s["meta"].append({"name": node.name, "emits": list(node.emits), "skipped": True})
                continue
            run.append(k)
        if not run:
            continue
        prompts = [build_prompt(node.emits, items[k]["text"], None if i == 0 else S[k]["ctx"],
                                _node_shots(node, items[k].get("shots", ()), early_exit),
                                with_schema=backend.needs_schema_text) for k in run]
        gens = backend.generate_batch(prompts, make_output_model(node.emits))
        for k, gen in zip(run, gens):
            s = S[k]
            s["meta"].append({"name": node.name, "emits": list(node.emits), "skipped": False, **gen.meta()})
            if gen.parsed is None:
                if i == 0:
                    s["alive"] = False          # sem contexto não há como continuar
                continue
            s["state"].update(gen.parsed.model_dump(mode="json"))
            if i == 0:
                s["ctx"] = items[k].get("oracle_context") or canonical.context_from(
                    canonical.canonicalize(s["state"]))
    results = []
    for it, s in zip(items, S):
        canon = canonical.canonicalize(s["state"])
        try:
            ok = InqueritoTotal.model_validate(canon) is not None
        except Exception:  # noqa: BLE001
            ok = False
        results.append(ExtractionResult(it["doc_id"], canon, ok, s["meta"], s["ctx"]))
    return results


def extract_document(doc_id: str, text: str, backend: Backend, topology: str = "T1", *,
                     shots: Sequence[Shot] = (), early_exit: bool = True,
                     oracle_context: dict = None) -> ExtractionResult:
    return extract_batch([{"doc_id": doc_id, "text": text, "shots": shots, "oracle_context": oracle_context}],
                         backend, topology, early_exit=early_exit)[0]
