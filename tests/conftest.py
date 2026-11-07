import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NOMES = ["Maria da Silva", "João Carlos Pereira", "Ana Paula Souza", "Pedro Henrique Lima", "Carla Mendes",
         "Rafael Alves", "Juliana Costa", "Marcos Oliveira", "Beatriz Rocha", "Lucas Martins"]


def make_doc(i: int, classe: str):
    rng = random.Random(i)
    v, s, t = rng.sample(NOMES, 3)
    natural = "naturais" in classe
    gt = {
        "resumo_processo": {"pessoas_envolvidas": {
            "vitimas": [v], "suspeitos_investigados": [] if natural else [s], "testemunhas": [t]},
            "classificacao_crime": classe},
        "inquerito": {"resultado": None if natural else "Consumado", "data_ocorrencia": f"{i % 27 + 1:02d}/03/2024",
                      "municipio": "São Paulo", "natureza_da_autoria": "Conhecida",
                      "razao_arquivamento": f"ausência de provas de autoria no caso {i}"},
        "vitimas": [{"nome": v, "sexo": "feminino", "faleceu": True, "causa_juridica_da_morte": "homicídio"}],
        "suspeitos": [] if natural else [{"nome": s, "sexo": "masculino"}],
        "testemunhas": [{"nome": t, "profissao": "vizinha"}],
    }
    text = (f"Promoção de arquivamento {i}. A vítima {v} faleceu em São Paulo. "
            + ("" if natural else f"O investigado {s} foi ouvido. ") + f"A testemunha {t} confirmou. "
            f"Arquiva-se por ausência de provas de autoria no caso {i}. " + "texto " * rng.randint(5, 60))
    return gt, text


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("mpsp")
    classes = ["Homicídio"] * 6 + ["Latrocínio (roubo seguido de morte)"] * 4 + ["Morte por causas naturais"] * 2
    docs = d / "docs"
    docs.mkdir()
    gt = {}
    for i, c in enumerate(classes):
        g, text = make_doc(i, c)
        gt[f"{1000 + i}"] = g
        (docs / f"{1000 + i}.txt").write_text(text, encoding="utf-8")
    (d / "gt.json").write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
    return {"dir": d, "docs": str(docs), "gt_path": str(d / "gt.json"), "gt_raw": gt}
