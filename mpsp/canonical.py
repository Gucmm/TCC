"""Normalização de qualquer formato legado de saída/GT para o formato canônico.

Formatos aceitos para as listas de pessoas:
  - lista direta                     {"vitimas": [...]}
  - wrapper com o mesmo nome         {"vítimas": {"vitimas": [...]}}
  - wrapper com capitalização antiga {"suspeitos": {"Suspeitos": [...]}}
  - envelope de execução             {"status": "success", "result": {...}}
"""
from typing import Any, Dict, Iterable, List

from .schemas import InqueritoTotal

_ALIASES = {
    "vitimas": ("vitimas", "vítimas", "Vitimas", "Vítimas"),
    "suspeitos": ("suspeitos", "Suspeitos"),
    "testemunhas": ("testemunhas", "Testemunhas"),
}


def _as_list(value: Any, names: Iterable[str]) -> List[dict]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for k in names:
            if isinstance(value.get(k), list):
                return value[k]
        lists = [v for v in value.values() if isinstance(v, list)]
        if len(lists) == 1:
            return lists[0]
    return []


def canonicalize(raw: Any) -> Dict[str, Any]:
    """Devolve um dict com as 5 seções, listas de pessoas sempre como `list`."""
    if not isinstance(raw, dict):
        raw = {}
    if isinstance(raw.get("result"), dict):
        raw = raw["result"]
    out: Dict[str, Any] = {
        "resumo_processo": raw.get("resumo_processo"),
        "inquerito": raw.get("inquerito"),
    }
    for role, names in _ALIASES.items():
        value = next((raw[k] for k in names if k in raw), None)
        out[role] = _as_list(value, names)
    return out


def validate(raw: Any) -> InqueritoTotal:
    return InqueritoTotal.model_validate(canonicalize(raw))


def dump(model: InqueritoTotal) -> Dict[str, Any]:
    return model.model_dump(mode="json")


def project(canon: Dict[str, Any], emits: Iterable[str]) -> Dict[str, Any]:
    return {k: canon.get(k) for k in emits}


def context_from(canon: Dict[str, Any]) -> Dict[str, Any]:
    """Contexto repassado aos nós seguintes: classificação e nomes por papel."""
    resumo = canon.get("resumo_processo") or {}
    pessoas = resumo.get("pessoas_envolvidas") or {}
    return {
        "classificacao": resumo.get("classificacao_crime"),
        "vitimas": list(pessoas.get("vitimas") or []),
        "suspeitos": list(pessoas.get("suspeitos_investigados") or []),
        "testemunhas": list(pessoas.get("testemunhas") or []),
    }
