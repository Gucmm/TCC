import json

import pytest

from mpsp import canonical, data, evaluate as ev, postprocess, prompts
from mpsp.backends.base import MockBackend
from mpsp.extract import extract_document
from mpsp.fewshot import TfidfRetriever, select_shots
from mpsp.schemas import SECTIONS, make_output_model
from mpsp.topologies import TOPOLOGIES


def oracle(gt_by_text):
    def responder(prompt, schema):
        for text, canon in gt_by_text.items():
            if prompt.rstrip().endswith(text + "\n\nResposta (JSON):"):
                return json.loads(prompts.dumps({k: canon[k] for k in schema.model_fields}))
        raise AssertionError("documento não encontrado no prompt")
    return responder


@pytest.fixture()
def corpus(dataset):
    gt, errors = data.load_gt(dataset["gt_path"])
    texts = data.load_texts(gt, dataset["docs"])
    assert not errors and len(texts) == len(gt) == 12
    return gt, texts


def test_t1_com_backend_oraculo_reproduz_o_gt(corpus):
    gt, texts = corpus
    doc = "1000"      # homicídio com as 3 categorias de pessoas
    be = MockBackend(oracle({texts[doc]: gt[doc]}))
    r = extract_document(doc, texts[doc], be, "T1")
    assert r.ok and len(be.calls) == 1
    rep = ev.evaluate({doc: gt[doc]}, {doc: r.prediction})
    assert rep.overall("strict")["f1"] == 1.0
    assert all(len(r.prediction[k]) == len(gt[doc][k]) for k in ("vitimas", "suspeitos", "testemunhas"))


def test_falha_do_primeiro_no_aborta_e_conta_como_invalido(corpus):
    gt, texts = corpus
    r = extract_document("1000", texts["1000"], MockBackend(lambda p, s: "isto não é json"), "T1")
    assert not r.ok and r.nodes[0]["error"].startswith("json_invalido") and r.nodes[0]["retries"] == 1


def test_retentativa_recupera_saida_invalida(corpus):
    gt, texts = corpus
    state = {"n": 0}

    def flaky(prompt, schema):
        state["n"] += 1
        return "```json\n{quebrado" if state["n"] == 1 else {"resumo_processo": {"pessoas_envolvidas": {}, "classificacao_crime": "Homicídio"}}
    r = extract_document("1000", texts["1000"], MockBackend(flaky), "T1")
    assert r.ok and r.nodes[0]["retries"] == 1


def test_estouro_de_contexto_nao_reenvia(corpus):
    from mpsp.backends.base import Backend, ContextOverflow, RawCompletion

    class Curto(Backend):
        name, calls = "curto", 0
        def _complete(self, prompt, schema):
            Curto.calls += 1
            raise ContextOverflow("x")
    _, texts = corpus
    r = extract_document("1000", texts["1000"], Curto(), "T1")
    assert not r.ok and r.nodes[0]["error"] == "context_overflow" and Curto.calls == 1


def test_prompts_compartilham_as_mesmas_regras_em_todas_as_topologias():
    regra = "é_policial só pode ser true se o texto afirmar"
    for topo, nodes in TOPOLOGIES.items():
        com_regra = [n.name for n in nodes if regra in prompts.build_prompt(n.emits, "doc")]
        com_pessoas = [n.name for n in nodes if any(r in n.emits for r in ("vitimas", "suspeitos", "testemunhas"))]
        assert com_regra == com_pessoas, topo


def test_schema_em_texto_e_few_shots_no_prompt():
    shots = [{"text": "DOC-EXEMPLO", "ctx": None, "output": {"vitimas": [{"nome": "X", "idade": None}]}}]
    p = prompts.build_prompt(("vitimas",), "DOC-ALVO", {"classificacao": "Homicídio", "vitimas": ["X"]}, shots, with_schema=True)
    assert "- nome (texto)" in p and "causa_juridica_da_morte" in p
    assert p.index("DOC-EXEMPLO") < p.index("### Documento a processar") < p.index("DOC-ALVO")
    assert '"idade"' not in p       # nulos são omitidos no exemplo


def test_fewshot_recupera_mais_parecido_e_exclui_o_proprio(corpus):
    gt, texts = corpus
    retr = TfidfRetriever(texts)
    shots = select_shots(retr, texts, gt, "1000", texts["1000"], k=3)
    assert len(shots) == 3 and all(s["doc_id"] != "1000" for s in shots)
    assert retr.query(texts["1003"], 1)[0] == "1003"
    assert select_shots(retr, texts, gt, "1000", texts["1000"], k=0) == []
    assert len(select_shots(retr, texts, gt, "1000", texts["1000"], k=3, max_chars=10)) == 0


def test_splits_cobrem_todos_uma_vez_e_sao_disjuntos(corpus):
    gt, texts = corpus
    s = data.make_splits(gt, texts, k=3, seed=1, val_frac=0.2)
    tests = [i for f in s["folds"] for i in f["test"]]
    assert sorted(tests) == sorted(gt) and len(set(tests)) == len(tests)
    for f in s["folds"]:
        assert not (set(f["train"]) & set(f["val"])) and not (set(f["train"] + f["val"]) & set(f["test"]))
        assert len(f["train"]) + len(f["val"]) + len(f["test"]) == len(gt)
    assert s == data.make_splits(gt, texts, k=3, seed=1, val_frac=0.2)     # determinístico


def test_regras_de_negocio_e_ancoragem():
    pred = {"resumo_processo": {"classificacao_crime": "Latrocínio (roubo seguido de morte)"},
            "inquerito": {"é_feminicidio": True, "resultado": "Tentado", "razao_arquivamento": "ausência de provas",
                          "municipio": "Campinas"},
            "vitimas": [{"nome": "Maria Silva", "é_policial": True}], "suspeitos": [], "testemunhas": []}
    v = postprocess.rule_violations(pred)
    assert set(v) == {"feminicidio_fora_de_homicidio", "latrocinio_nao_consumado", "policial_sem_corporacao"}
    g = postprocess.grounding(pred, "Arquiva-se por ausência de provas. Maria Silva foi ouvida em Santos.")
    assert g == {"checked": 3, "grounded": 2}      # Campinas não está no texto
