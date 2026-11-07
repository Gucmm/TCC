"""Dados sigilosos: nada de API externa e rede só onde declarado."""
import os
import re
from pathlib import Path

import pytest

from mpsp import cli
from mpsp.backends import get_backend, load_registry

PKG = Path(__file__).resolve().parents[1] / "mpsp"
PROIBIDOS = re.compile(r"^\s*(?:import|from)\s+(requests|httpx|aiohttp|openai|anthropic|google|urllib3)\b", re.M)


def test_nenhum_cliente_de_api_ou_http_no_pacote():
    achados = [str(p.relative_to(PKG)) for p in PKG.rglob("*.py") if PROIBIDOS.search(p.read_text(encoding="utf-8"))]
    assert achados == []


def test_rede_so_no_download_de_pesos_e_do_corpus_ner_publico():
    uso = {"urllib.request": set(), "snapshot_download": set()}
    for p in PKG.rglob("*.py"):
        txt = p.read_text(encoding="utf-8")
        for k in uso:
            if k in txt:
                uso[k].add(p.name)
    assert uso["urllib.request"] == {"conll.py"}      # baixa o LeNER-Br (dado público)
    assert uso["snapshot_download"] == {"cli.py"}     # `mpsp download` (arquivos públicos de modelo)


def test_registro_so_tem_modelos_locais():
    assert set(load_registry()) == {"slm"}
    with pytest.raises(KeyError, match="não está em configs"):
        get_backend("gemini-2.5-pro")


def test_comandos_comuns_bloqueiam_o_hugging_face(monkeypatch):
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.delenv(k, raising=False)
    cli.main(["models"])
    assert all(os.environ[k] == "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"))
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
