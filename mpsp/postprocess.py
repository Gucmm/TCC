"""Diagnósticos sobre a predição: regras de negócio e ancoragem no texto.

Não corrige a saída (para não esconder erro do modelo): apenas mede.
"""
from typing import Dict, List

from .schemas import ROLES, ClassificacaoCrime
from .textnorm import norm, tokens

HOMICIDIO = ClassificacaoCrime.HOMICIDIO.value
LATROCINIO = ClassificacaoCrime.LATROCINIO.value
NATURAL = ClassificacaoCrime.MORTE_CAUSAS_NATURAIS.value


def rule_violations(pred: dict) -> List[str]:
    resumo = pred.get("resumo_processo") or {}
    classe, inq = resumo.get("classificacao_crime"), pred.get("inquerito") or {}
    v = []
    if inq.get("é_feminicidio") and classe != HOMICIDIO:
        v.append("feminicidio_fora_de_homicidio")
    if inq.get("bem_roubado") and classe != LATROCINIO:
        v.append("bem_roubado_fora_de_latrocinio")
    if classe == LATROCINIO and inq.get("resultado") not in (None, "Consumado"):
        v.append("latrocinio_nao_consumado")
    if classe == NATURAL and inq.get("resultado"):
        v.append("resultado_em_morte_natural")
    for role in ROLES:
        for p in pred.get(role) or []:
            if p.get("é_policial") and not p.get("corporacao_policial"):
                v.append("policial_sem_corporacao")
    return v


def _coverage(value: str, doc_tokens: set) -> float:
    t = tokens(value)
    return sum(x in doc_tokens for x in t) / len(t) if t else 1.0


def grounding(pred: dict, text: str) -> Dict[str, int]:
    """Valores literais da predição que de fato aparecem no documento."""
    doc_norm, doc_tokens = norm(text), set(tokens(text))
    checked = grounded = 0

    def check(value, min_cov, substring=False):
        nonlocal checked, grounded
        if value is None or not str(value).strip():
            return
        checked += 1
        ok = _coverage(str(value), doc_tokens) >= min_cov
        if substring:
            ok = ok or norm(value) in doc_norm
        grounded += int(ok)

    inq = pred.get("inquerito") or {}
    check(inq.get("razao_arquivamento"), 0.95, substring=True)
    check(inq.get("delegacia_registro"), 0.8)
    check(inq.get("municipio"), 1.0)
    for role in ROLES:
        for p in pred.get(role) or []:
            nome = [t for t in tokens(p.get("nome") or "") if len(t) > 1]
            if nome:
                check(" ".join(nome), 1.0)
    return {"checked": checked, "grounded": grounded}
