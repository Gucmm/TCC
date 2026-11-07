import json
import subprocess
import sys

import numpy as np
import pytest

from mpsp import chat, stats
from mpsp.benchmarks import conll, ner

def test_bootstrap_ic_contem_a_media():
    x = np.random.default_rng(0).normal(0.7, 0.1, 100)
    m, lo, hi = stats.bootstrap_ci(x)
    assert lo < m < hi and hi - lo < 0.06


def test_bootstrap_pareado_detecta_diferenca_e_nao_inventa():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.4, 0.9, 100)
    melhor = stats.paired_bootstrap(base + 0.05 + rng.normal(0, 0.02, 100), base)
    igual = stats.paired_bootstrap(base + rng.normal(0, 0.02, 100), base)
    assert melhor["lo"] > 0 and melhor["p"] < 0.01 and igual["lo"] < 0 < igual["hi"]
    assert stats.wilcoxon_p(base + 0.05, base) < 0.001


def test_holm():
    h = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h["a"] == pytest.approx(0.03) and h["c"] == pytest.approx(0.06) and h["b"] == pytest.approx(0.06)


class StubTok:
    """Tokenizador de brinquedo: 1 token por caractere; template estilo ChatML."""
    eos_token, pad_token_id = "<e>", 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        s = "".join(f"<{m['role']}>{m['content']}<e>" for m in msgs)
        return s + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_mascara_so_na_resposta_e_nao_trunca():
    tok = StubTok()
    ex = chat.build_example(tok, "PROMPT", "RESP", max_len=200)
    p = len(chat.prompt_text(tok, "PROMPT"))
    assert ex["labels"][:p] == [-100] * p and ex["labels"][p:] == ex["input_ids"][p:]
    assert bytes(ex["input_ids"][p:]).decode() == "RESP<e>"     # inclui o fim de turno do template
    assert chat.build_example(tok, "PROMPT", "RESP", max_len=10) is None


CONLL = """Acórdão O
1160/2016 O
PLENÁRIO B-ORGANIZACAO
Relator O
AUGUSTO B-PESSOA
NARDES I-PESSOA
.  O

Ministro O
Ana B-PESSOA
Arraes I-PESSOA
no O
TCU B-ORGANIZACAO
em O
11/05/2016 B-TEMPO
.  O
"""


@pytest.fixture()
def units(tmp_path):
    f = tmp_path / "x.conll"
    f.write_text(CONLL, encoding="utf-8")
    return conll.make_units(conll.read_conll(f), window=2, prefix="u")


def test_bio_e_janelas(units):
    assert len(units) == 1
    u = units[0]
    assert [(" ".join(u.tokens[s:e]), t) for s, e, t in u.spans] == [
        ("PLENÁRIO", "ORGANIZACAO"), ("AUGUSTO NARDES", "PESSOA"), ("Ana Arraes", "PESSOA"),
        ("TCU", "ORGANIZACAO"), ("11/05/2016", "TEMPO")]
    assert conll.bio_to_spans(["I-PESSOA", "I-PESSOA", "O"]) == [(0, 2, "PESSOA")]     # I- sem B- inicia entidade


def test_cli_run_eval_compare(dataset, tmp_path):
    py = [sys.executable, "-m", "mpsp"]
    run = lambda *a: subprocess.run(py + list(a), capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout  # noqa: E731
    run("split", "--gt", dataset["gt_path"], "--docs", dataset["docs"], "--k", "3", "--out", str(tmp_path / "s.json"))
    for name, shots in (("t1", "0"), ("t1k2", "2")):
        run("run", "--gt", dataset["gt_path"], "--docs", dataset["docs"], "--splits", str(tmp_path / "s.json"),
            "--model", "mock", "--topology", "T1", "--shots", shots,
            "--out", str(tmp_path / f"{name}.jsonl"))
    out = run("eval", "--gt", dataset["gt_path"], "--pred", str(tmp_path / "t1.jsonl"), "--docs", dataset["docs"],
              "--out", str(tmp_path / "rep"))
    summ = json.loads((tmp_path / "rep" / "summary.json").read_text())
    assert summ["n_docs"] == 12 and summ["run"]["ok_rate"] == 1.0 and "Por seção" in out
    run("run", "--gt", dataset["gt_path"], "--docs", dataset["docs"], "--splits", str(tmp_path / "s.json"),
        "--model", "mock", "--out", str(tmp_path / "t1.jsonl"))      # retomada: nada a refazer
    assert len((tmp_path / "t1.jsonl").read_text().splitlines()) == 12
    cmp = run("compare", "--gt", dataset["gt_path"], "--pred", str(tmp_path / "t1.jsonl"), str(tmp_path / "t1k2.jsonl"))
    assert "12 documentos em comum" in cmp
