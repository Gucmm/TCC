"""Tarefa NER por LLM: prompt, schema de saída, alinhamento em spans e métricas."""
from collections import Counter
from typing import Dict, List, Literal, Sequence, Tuple

from pydantic import Field, create_model

from ..prompts import describe_schema, dumps
from .conll import Unit

BASE = """Você é um especialista em reconhecimento de entidades nomeadas em textos jurídicos brasileiros.
Extraia do texto SOMENTE as entidades dos tipos abaixo:
{tipos}

Regras:
- Copie cada entidade exatamente como aparece no texto (mesma grafia, sem acrescentar ou remover palavras).
- Liste cada ocorrência, na ordem em que aparece. Não invente entidades.
- Se não houver entidades dos tipos pedidos, devolva a lista vazia.
- Responda somente com o JSON do schema."""


def ner_model(types: Sequence[str]):
    Tipo = Literal[tuple(types)]
    Ent = create_model("Entidade", texto=(str, ...), tipo=(Tipo, ...))
    return create_model("NERSaida", entidades=(List[Ent], Field(default_factory=list)))


def build_prompt(types: Dict[str, str], unit: Unit, shots: Sequence[Unit], with_schema: bool) -> str:
    blocks = [BASE.format(tipos="\n".join(f"- {t}: {d}" for t, d in types.items()))]
    if with_schema:
        blocks.append(describe_schema(ner_model(list(types))))

    def out(u: Unit):
        return {"entidades": [{"texto": " ".join(u.tokens[s:e]), "tipo": t} for s, e, t in u.spans if t in types]}

    if shots:
        blocks.append("Exemplos resolvidos:\n\n" + "\n\n".join(
            f"### Exemplo {i}\nTexto:\n{u.text}\nResposta (JSON):\n{dumps(out(u))}" for i, u in enumerate(shots, 1)))
        blocks.append("### Texto a processar")
    blocks.append(f"Texto:\n{unit.text}\n\nResposta (JSON):")
    return "\n\n".join(blocks)


def align(tokens: List[str], entities: List[Tuple[str, str]]) -> List[Tuple[int, int, str]]:
    """Cada entidade prevista é localizada na primeira ocorrência livre (cópia literal).
    Entidade não localizável vira span inexistente (conta como falso positivo)."""
    taken, out = [False] * len(tokens), []
    for k, (texto, tipo) in enumerate(entities):
        tt = texto.split()
        found = None
        for s in range(len(tokens) - len(tt) + 1):
            if tt and tokens[s:s + len(tt)] == tt and not any(taken[s:s + len(tt)]):
                found = s
                break
        if found is None:
            out.append((-1 - k, -1 - k, tipo))
        else:
            taken[found:found + len(tt)] = [True] * len(tt)
            out.append((found, found + len(tt), tipo))
    return out


def _prf(tp: int, n_pred: int, n_gold: int) -> Dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "support": n_gold}


def unit_counts(unit: Unit, entities: List[Tuple[str, str]]) -> Dict[str, Counter]:
    """Contagens por tipo: span exato (strict) e (tipo, texto) como multiconjunto (relaxed)."""
    gold = set(unit.spans)
    pred = align(unit.tokens, entities)
    strict = Counter(); relaxed = Counter()
    for t in {t for _, _, t in gold} | {t for _, _, t in pred}:
        g = {x for x in gold if x[2] == t}; p = [x for x in pred if x[2] == t]
        strict[(t, "tp")] = len(g & set(p)); strict[(t, "pred")] = len(p); strict[(t, "gold")] = len(g)
        gt_txt = Counter(" ".join(unit.tokens[s:e]) for s, e, _ in g)
        pr_txt = Counter(txt for txt, tp_ in entities if tp_ == t)
        relaxed[(t, "tp")] = sum((gt_txt & pr_txt).values())
        relaxed[(t, "pred")] = sum(pr_txt.values()); relaxed[(t, "gold")] = sum(gt_txt.values())
    return {"strict": strict, "relaxed": relaxed}


def score(units: Sequence[Unit], preds: Dict[str, List[Tuple[str, str]]]) -> dict:
    """Micro P/R/F1 (strict e relaxed), por tipo, e F1 por unidade (vazio×vazio = 1)."""
    out, per_unit = {}, {"strict": {}, "relaxed": {}}
    totals = {m: Counter() for m in ("strict", "relaxed")}
    for u in units:
        c = unit_counts(u, preds.get(u.id, []))
        for m in totals:
            totals[m].update(c[m])
            tp = sum(v for (t, k), v in c[m].items() if k == "tp")
            npred = sum(v for (t, k), v in c[m].items() if k == "pred")
            ngold = sum(v for (t, k), v in c[m].items() if k == "gold")
            per_unit[m][u.id] = 1.0 if npred == 0 and ngold == 0 else _prf(tp, npred, ngold)["f1"]
    for m, tot in totals.items():
        types = sorted({t for t, _ in tot})
        agg = lambda ts: _prf(sum(tot[(t, "tp")] for t in ts), sum(tot[(t, "pred")] for t in ts),  # noqa: E731
                              sum(tot[(t, "gold")] for t in ts))
        out[m] = {"micro": agg(types), "by_type": {t: agg([t]) for t in types}}
        out[m]["macro_f1"] = sum(v["f1"] for v in out[m]["by_type"].values()) / max(len(types), 1)
    return {**out, "per_unit": per_unit}


def training_records(units: Sequence[Unit], types: Dict[str, str]) -> List[dict]:
    """Exemplos (prompt, completion) para ajustar o modelo em NER, com o MESMO prompt do benchmark (padrão `all`)."""
    recs = []
    for u in units:
        out = {"entidades": [{"texto": " ".join(u.tokens[s:e]), "tipo": t} for s, e, t in u.spans if t in types]}
        recs.append({"doc_id": u.id, "node": "ner", "prompt": build_prompt(types, u, (), True), "completion": dumps(out)})
    return recs
