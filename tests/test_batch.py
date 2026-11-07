import json
import subprocess
import sys

import pytest

from mpsp import data, prompts
from mpsp.backends.base import MockBackend
from mpsp.extract import extract_batch, extract_document


def oracle(gt_by_text):
    def responder(prompt, schema):
        for text, canon in gt_by_text.items():
            if prompt.rstrip().endswith(text + "\n\nResposta (JSON):"):
                return json.loads(prompts.dumps({k: canon[k] for k in schema.model_fields}))
        raise AssertionError("documento não encontrado no prompt")
    return responder


class SpyBackend(MockBackend):
    """Registra o tamanho de cada lote enviado ao engine."""
    def __init__(self, responder):
        super().__init__(responder)
        self.batches = []

    def _complete_batch(self, prompts_, schema):
        self.batches.append(len(prompts_))
        return super()._complete_batch(prompts_, schema)


@pytest.fixture()
def corpus(dataset):
    gt, _ = data.load_gt(dataset["gt_path"])
    return gt, data.load_texts(gt, dataset["docs"])


def strip_time(nodes):
    return [{k: v for k, v in n.items() if k != "seconds"} for n in nodes]


def test_lote_e_sequencial_dao_o_mesmo_resultado(corpus):
    topo = "T1"
    gt, texts = corpus
    resp = oracle({texts[d]: gt[d] for d in gt})
    items = [{"doc_id": d, "text": texts[d]} for d in sorted(gt)]
    lote = extract_batch(items, MockBackend(resp), topo)
    seq = [extract_document(d, texts[d], MockBackend(resp), topo) for d in sorted(gt)]
    for a, b in zip(lote, seq):
        assert (a.doc_id, a.prediction, a.ok, a.context) == (b.doc_id, b.prediction, b.ok, b.context)
        assert strip_time(a.nodes) == strip_time(b.nodes)


def test_t1_vira_um_unico_lote_com_todos_os_documentos(corpus):
    gt, texts = corpus
    be = SpyBackend(oracle({texts[d]: gt[d] for d in gt}))
    extract_batch([{"doc_id": d, "text": texts[d]} for d in sorted(gt)], be, "T1")
    assert be.batches == [12]


def test_retentativa_reenvia_so_os_invalidos_do_lote():
    from mpsp.schemas import make_output_model
    schema = make_output_model(("resumo_processo",))
    ok = {"resumo_processo": {"pessoas_envolvidas": {}, "classificacao_crime": "Homicídio"}}
    visto = {}

    def responder(prompt, sch):
        base = prompt.split("\n\nA resposta anterior")[0]
        visto[base] = visto.get(base, 0) + 1
        return "lixo" if base == "B" and visto[base] == 1 else ok

    be = SpyBackend(responder)
    gens = be.generate_batch(["A", "B", "C"], schema)
    assert be.batches == [3, 1]                         # 2ª rodada só com o prompt inválido
    assert [g.retries for g in gens] == [0, 1, 0] and all(g.parsed is not None for g in gens)
    assert all(g.error is None for g in gens)


def test_falha_de_um_nao_derruba_o_lote(corpus):
    gt, texts = corpus
    resp = oracle({texts[d]: gt[d] for d in gt})
    ruim = sorted(gt)[1]

    def responder(prompt, schema):
        return "não é json" if prompt.rstrip().endswith(texts[ruim] + "\n\nResposta (JSON):") else resp(prompt, schema)
    r = {x.doc_id: x for x in extract_batch([{"doc_id": d, "text": texts[d]} for d in sorted(gt)[:3]],
                                            MockBackend(responder), "T1")}
    assert not r[ruim].ok and len(r[ruim].nodes) == 1       # abortou só esse documento
    assert all(r[d].ok for d in sorted(gt)[:3] if d != ruim)


def test_cli_shards_e_lotes_reproduzem_a_execucao_unica(dataset, tmp_path):
    py = [sys.executable, "-m", "mpsp"]
    run = lambda *a: subprocess.run(py + list(a), capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout  # noqa: E731
    base = ["--gt", dataset["gt_path"], "--docs", dataset["docs"]]
    run("split", *base, "--k", "3", "--out", str(tmp_path / "s.json"))
    comuns = [*base, "--splits", str(tmp_path / "s.json"), "--model", "mock", "--topology", "T1"]
    run("run", *comuns, "--out", str(tmp_path / "unico.jsonl"))
    for i in (0, 1):
        run("run", *comuns, "--shard", f"{i}/2", "--batch-size", "3", "--out", str(tmp_path / "shard.jsonl"))
    shards = sorted(tmp_path.glob("shard.shard*of2.jsonl"))
    assert [p.name for p in shards] == ["shard.shard0of2.jsonl", "shard.shard1of2.jsonl"]
    ids = [set(json.loads(l)["doc_id"] for l in p.read_text().splitlines()) for p in shards]
    assert not (ids[0] & ids[1]) and len(ids[0] | ids[1]) == 12          # partição sem sobreposição
    # retomada de um shard não refaz nada
    run("run", *comuns, "--shard", "0/2", "--out", str(tmp_path / "shard.jsonl"))
    assert len(shards[0].read_text().splitlines()) == len(ids[0])
    # eval junta os shards automaticamente e bate com a execução única
    o1 = run("eval", *base[:2], "--pred", str(tmp_path / "unico.jsonl"), "--out", str(tmp_path / "r1"))
    o2 = run("eval", *base[:2], "--pred", str(tmp_path / "shard.jsonl"), "--out", str(tmp_path / "r2"))
    s1, s2 = (json.loads((tmp_path / r / "summary.json").read_text()) for r in ("r1", "r2"))
    assert s1["n_docs"] == s2["n_docs"] == 12 and s1["strict"] == s2["strict"] and s1["soft"] == s2["soft"]


def test_shard_invalido_e_engine_vllm_sem_dependencia(dataset, tmp_path):
    r = subprocess.run([sys.executable, "-m", "mpsp", "run", "--gt", dataset["gt_path"], "--docs", dataset["docs"],
                        "--splits", str(tmp_path / "x.json"), "--model", "mock", "--shard", "3/2",
                        "--out", str(tmp_path / "o.jsonl")], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode != 0 and "--shard deve ser i/n" in r.stderr
