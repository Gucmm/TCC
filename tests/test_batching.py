"""Geração em lote do backend HF: lógica pura (sem torch) e, quando há torch, equivalência com a geração de um prompt por vez."""
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mpsp.backends.base import ContextOverflow
from mpsp.backends.batching import generate_all, plan_batches
from mpsp.backends.local import LocalBackend


def test_lotes_agrupam_por_tamanho_e_respeitam_os_limites():
    lengths = [100, 5000, 120, 110, 4900, 130]
    batches = plan_batches(lengths, max_seqs=3, max_tokens=6000)
    assert sorted(i for b in batches for i in b) == list(range(6))                  # todos os prompts, uma vez
    assert all(len(b) <= 3 for b in batches)
    assert all(len(b) * max(lengths[i] for i in b) <= 6000 or len(b) == 1 for b in batches)
    longos = [b for b in batches if 1 in b or 4 in b]
    assert len(longos) == 2 and all(len(b) == 1 for b in longos)                    # 2 × 5000 > 6000: cada um sozinho
    assert max(lengths[i] for i in batches[0]) <= min(lengths[i] for i in batches[-1])   # ordem crescente: pouco padding


def test_prompt_acima_do_orcamento_vai_sozinho_e_lista_vazia_nao_gera_lote():
    assert plan_batches([9000], 4, 1000) == [[0]]
    assert plan_batches([], 4, 1000) == []


def fake_gen(batch):                                  # "gera" o dobro do primeiro token, em um passo
    return [[p[0] * 2] for p in batch]


def kw(**o):
    return dict(ctx_tokens=100, max_new_tokens=10, max_seqs=4, max_tokens=1000, **o)


def test_resultados_ficam_alinhados_com_os_prompts():
    prompts = [[7] * 30, [1] * 5, [4] * 20, [9] * 10]
    assert generate_all(prompts, fake_gen, **kw()) == [[14], [2], [8], [18]]


def test_estouro_de_contexto_e_por_item_e_nao_chega_ao_gerador():
    chamados = []

    def gen(batch):
        chamados.extend(len(p) for p in batch)
        return fake_gen(batch)
    out = generate_all([[1] * 95, [2] * 5], gen, **kw())
    assert isinstance(out[0], ContextOverflow) and out[1] == [4]
    assert chamados == [5]


def test_falta_de_memoria_divide_o_lote_ate_caber():
    limpezas, tamanhos = [], []

    def gen(batch):
        tamanhos.append(len(batch))
        if len(batch) > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return fake_gen(batch)
    prompts = [[i + 1] * 10 for i in range(4)]
    out = generate_all(prompts, gen, **kw(on_oom=lambda: limpezas.append(1)))
    assert out == [[2], [4], [6], [8]] and tamanhos == [4, 2, 2] and len(limpezas) == 1


def test_oom_em_prompt_isolado_e_erro_so_daquele_item():
    def gen(batch):
        if any(p[0] == 3 for p in batch):
            raise RuntimeError("out of memory")
        return fake_gen(batch)
    out = generate_all([[1] * 10, [3] * 10, [5] * 10], gen, **kw())
    assert out[0] == [2] and out[2] == [10] and isinstance(out[1], RuntimeError)


def test_erro_que_nao_e_de_memoria_nao_divide_e_marca_o_lote():
    n = []

    def gen(batch):
        n.append(len(batch))
        raise ValueError("falha qualquer")
    out = generate_all([[1] * 5, [2] * 5], gen, **kw())
    assert n == [2] and all(isinstance(o, ValueError) for o in out)


class Tok:
    eos_token, eos_token_id, pad_token_id = "<e>", 1, 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        return msgs[0]["content"] + ("<a>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 90 + 2 for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + i % 26) for i in ids)


def backend(micro_batch=4, constrained=False):
    b = object.__new__(LocalBackend)
    b.name, b.ctx_tokens, b.max_new_tokens = "fake", 200, 10
    b.constrained, b.micro_batch, b.max_batch_tokens, b.tok = constrained, micro_batch, 1000, Tok()
    b.model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[1, 7]))
    b.batches = []

    def gen(batch_ids, **_):
        b.batches.append(len(batch_ids))
        return [[len(p), 5] for p in batch_ids]
    b._generate_ids = gen
    return b


def test_backend_em_lote_gera_uma_chamada_por_micro_lote_e_mede_tokens():
    b = backend(micro_batch=2)
    out = b._complete_batch(["a" * 10, "b" * 20, "c" * 30], schema=None)
    assert b.batches == [2, 1]
    assert [o.prompt_tokens for o in out] == [13, 23, 33] and all(o.completion_tokens == 2 for o in out)   # +3 do "<a>"
    assert out[0].text == Tok().decode([13, 5])


def test_backend_micro_lote_1_e_restricao_por_schema_voltam_ao_um_por_vez():
    for b in (backend(micro_batch=1), backend(constrained=True)):
        b._constrained_kw = lambda schema: {}
        out = b._complete_batch(["a" * 10, "b" * 20], schema=None)
        assert b.batches == [1, 1] and len(out) == 2


def test_backend_estouro_de_contexto_vira_excecao_do_item():
    b = backend()
    out = b._complete_batch(["a" * 10, "b" * 300], schema=None)
    assert isinstance(out[1], ContextOverflow) and out[0].prompt_tokens == 13


def test_lote_gera_o_mesmo_que_um_por_vez_com_padding_a_esquerda():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=256, tie_word_embeddings=False)).eval()
    b = object.__new__(LocalBackend)
    b.tok, b.model, b.max_new_tokens = Tok(), model, 8
    b.tok.pad_token_id = 0
    prompts = [[5, 9, 12, 33, 8, 40], [7, 2, 51], [60, 61, 62, 63, 64, 65, 66, 67, 68, 69]]
    juntos = b._generate_ids(prompts)
    sozinhos = [b._generate_ids([p])[0] for p in prompts]
    assert juntos == sozinhos


def test_validate_gt_reconcilia_documentos(dataset, tmp_path):
    raw = dict(dataset["gt_raw"])
    raw["9999"] = {"lixo": 1}                                        # GT com erro de schema
    raw["1011"] = raw["1011"]                                        # existe; vamos apagar o texto dele
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    for p in __import__("pathlib").Path(dataset["docs"]).glob("*.txt"):
        if p.stem != "1011":
            (docs / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    sp = tmp_path / "splits.json"
    subprocess.run([sys.executable, "-m", "mpsp", "split", "--gt", str(gt), "--docs", str(docs), "--k", "2",
                    "--out", str(sp)], check=True, capture_output=True)
    out = subprocess.run([sys.executable, "-m", "mpsp", "validate-gt", "--gt", str(gt), "--docs", str(docs),
                          "--splits", str(sp)], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout
    assert "13 entradas" in out and "válidas no schema: 12" in out and "com erro de schema: 1" in out
    assert "sem texto: 1" in out and "1011" in out
    assert "Documentos utilizáveis (schema válido e texto): 11" in out
    assert "11 documentos de teste em 2 folds (0 repetidos)" in out
