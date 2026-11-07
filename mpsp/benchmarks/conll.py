"""Leitura de corpora NER em CoNLL/BIO e montagem de amostras (janelas de sentenças).

Para adicionar outro corpus em português jurídico/legislativo basta registrar uma entrada
em DATASETS (URLs dos arquivos CoNLL + descrição dos tipos). Verificado: LeNER-Br.
"""
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_LENER = "https://raw.githubusercontent.com/peluz/lener-br/master/leNER-Br/{s}/{s}.conll"

DATASETS: Dict[str, dict] = {
    "lener_br": {
        "urls": {s: _LENER.format(s=s) for s in ("train", "dev", "test")},
        "types": {
            "PESSOA": "nomes de pessoas físicas (partes, ministros, advogados, servidores)",
            "ORGANIZACAO": "órgãos públicos, tribunais, empresas e demais instituições",
            "LOCAL": "locais geográficos (cidades, estados, países)",
            "TEMPO": "expressões de data e de tempo",
            "LEGISLACAO": "leis, decretos, artigos e demais normas citadas",
            "JURISPRUDENCIA": "acórdãos, processos, súmulas e decisões judiciais citadas",
        },
    },
}


@dataclass
class Unit:
    id: str
    tokens: List[str]
    text: str
    spans: List[Tuple[int, int, str]]     # (início, fim exclusivo, tipo) em índices de token


def fetch(name: str, split: str, cache_dir: str) -> Path:
    path = Path(cache_dir) / name / f"{split}.conll"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATASETS[name]["urls"][split], path)
    return path


def read_conll(path: Path) -> List[List[Tuple[str, str]]]:
    sents, cur = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if cur:
                sents.append(cur)
                cur = []
            continue
        parts = line.split()
        cur.append((" ".join(parts[:-1]), parts[-1]))
    if cur:
        sents.append(cur)
    return sents


def bio_to_spans(tags: List[str]) -> List[Tuple[int, int, str]]:
    spans, start, typ = [], None, None
    for i, t in enumerate(tags + ["O"]):
        inside = t.startswith("I-") and typ == t[2:]
        if start is not None and not inside:
            spans.append((start, i, typ))
            start, typ = None, None
        if t.startswith("B-") or (t.startswith("I-") and start is None):
            start, typ = i, t[2:]
    return spans


def make_units(sents: List[List[Tuple[str, str]]], window: int, prefix: str) -> List[Unit]:
    units = []
    for g in range(0, len(sents), window):
        group = sents[g:g + window]
        tokens = [w for s in group for w, _ in s]
        tags = [t for s in group for _, t in s]
        text = "\n".join(" ".join(w for w, _ in s) for s in group)
        units.append(Unit(f"{prefix}{g // window:05d}", tokens, text, bio_to_spans(tags)))
    return units


def sample_units(units: List[Unit], n: int, seed: int) -> List[Unit]:
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(units)), min(n, len(units))))
    return [units[i] for i in idx]


def load(name: str, split: str, window: int, cache_dir: str, n: Optional[int] = None, seed: int = 42) -> List[Unit]:
    units = make_units(read_conll(fetch(name, split, cache_dir)), window, f"{name}-{split}-")
    return sample_units(units, n, seed) if n else units
