"""Lógica de treino que não precisa de torch: dados, fases, o que é treinável (com módulos de mentira), parada, estimativas."""
import argparse
import json
from types import SimpleNamespace

import pytest

from mpsp import data
from mpsp.backends import resolve_adapter
from mpsp.slm import train
from mpsp.slm.earlystop import EarlyStop
from mpsp.slm.methods import METHODS, estimate_infer_gb, estimate_train_gb


class StubTok:
    eos_token, pad_token_id = "<e>", 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        s = "".join(f"<{m['role']}>{m['content']}<e>" for m in msgs)
        return s + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 500 for c in text]}


@pytest.fixture()
def corpus(dataset):
    gt, _ = data.load_gt(dataset["gt_path"])
    return gt, data.load_texts(gt, dataset["docs"])


def args(**kw):
    base = dict(lr=None, epochs=None, patience=None, phase1_lr=None, phase1_epochs=None, topology="T1", seed=42,
                ner_window=10, ner_n=4, ner_val=2, ner_cache="x")
    base.update(kw)
    return SimpleNamespace(**base)


def test_registro_de_metodos():
    assert set(METHODS) == {"qlora", "lora", "full", "frozen", "two_phase", "two_phase_ner"}
    assert METHODS["qlora"].load_4bit and not METHODS["lora"].load_4bit
    assert [p.name for p in METHODS["two_phase"].phases] == ["head", "lora"]
    assert [p.data for p in METHODS["two_phase_ner"].phases] == ["ner", "mpsp"]
    assert METHODS["full"].full_save and METHODS["frozen"].full_save and METHODS["two_phase"].hf_only


def test_sobrescritas_valem_para_a_ultima_fase_e_phase1_para_a_primeira():
    ph = train.plan_phases(METHODS["two_phase"], args(lr=5e-5, epochs=7, phase1_lr=1e-3, phase1_epochs=1))
    assert (ph[0].lr, ph[0].epochs) == (1e-3, 1) and (ph[1].lr, ph[1].epochs) == (5e-5, 7)
    solo = train.plan_phases(METHODS["lora"], args(epochs=3))
    assert len(solo) == 1 and solo[0].epochs == 3 and solo[0].lr == METHODS["lora"].phases[0].lr


def test_estimativas_de_memoria_seguem_a_ordem_esperada():
    e = {n: estimate_train_gb(8.0, m, 4096) for n, m in METHODS.items()}
    assert e["qlora"] < e["frozen"] < e["lora"] < e["full"] < 48
    assert estimate_train_gb(8.0, METHODS["lora"], 8192) > e["lora"]          # sequência maior, mais ativações
    assert estimate_infer_gb(8.0, True, "hf") < estimate_infer_gb(8.0, False, "hf") < estimate_infer_gb(8.0, True, "vllm")


def test_build_records_t1(corpus):
    gt, texts = corpus
    recs = train.build_records(gt, texts, sorted(gt), "T1")
    assert len(recs) == 12                                   # um exemplo (nó "tudo") por documento
    r = next(x for x in recs if json.loads(x["completion"]).get("vitimas"))
    assert "Schema da resposta" in r["prompt"]


def test_subsample_e_deterministico():
    ids = [str(i) for i in range(40)]
    a = train.subsample(ids, 0.25, 7)
    assert a == train.subsample(ids, 0.25, 7) and len(a) == 10 and set(a) <= set(ids)
    assert train.subsample(ids, 1.0, 7) == sorted(ids) and len(train.subsample(ids, 0.001, 7)) == 1


def test_encode_descarta_o_que_nao_cabe_sem_truncar():
    recs = [{"prompt": "p" * 5, "completion": "c" * 5}, {"prompt": "p" * 500, "completion": "c" * 5}]
    kept, dropped = train.encode_records(StubTok(), recs, max_len=200)
    assert len(kept) == 1 and dropped == 1
    assert kept[0]["labels"].count(-100) == len("<user>ppppp<e><assistant>")


def test_registros_de_ner_usam_o_prompt_do_benchmark(tmp_path):
    from mpsp.benchmarks import conll, ner
    f = tmp_path / "x.conll"
    f.write_text("Ana B-PESSOA\nArraes I-PESSOA\nno O\nTCU B-ORGANIZACAO\n.  O\n", encoding="utf-8")
    units = conll.make_units(conll.read_conll(f), 10, "u")
    recs = ner.training_records(units, conll.DATASETS["lener_br"]["types"])
    assert len(recs) == 1 and "Ana Arraes no TCU" in recs[0]["prompt"] and "Schema da resposta" in recs[0]["prompt"]
    assert json.loads(recs[0]["completion"]) == {"entidades": [{"texto": "Ana Arraes", "tipo": "PESSOA"},
                                                              {"texto": "TCU", "tipo": "ORGANIZACAO"}]}


class P:
    def __init__(self, n):
        self.n, self.requires_grad = n, True

    def numel(self):
        return self.n


class Mod:
    def __init__(self, *ps):
        self.ps = list(ps)

    def parameters(self):
        return iter(self.ps)


class FakeLM(Mod):
    def __init__(self, tied=False, n_layers=4):
        self.embed = Mod(P(100))
        self.head = self.embed if tied else Mod(P(100))
        self.norm = Mod(P(4))
        self.layers = [Mod(P(10), P(10)) for _ in range(n_layers)]
        self.model = SimpleNamespace(layers=self.layers, norm=self.norm)
        self.config = SimpleNamespace(tie_word_embeddings=tied)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head

    def parameters(self):
        seen, out = set(), []
        for m in [self.embed, self.head, self.norm, *self.layers]:
            for p in m.ps:
                if id(p) not in seen:
                    seen.add(id(p)); out.append(p)
        return iter(out)

    def named_modules(self):
        return []


def trainable_ids(m):
    return {id(p) for p in m.parameters() if p.requires_grad}


def test_all_congela_as_embeddings():
    m = FakeLM()
    info = train.set_trainable(m, "all")
    assert info["trainable"] == info["total"] - 100 and not m.embed.ps[0].requires_grad and m.head.ps[0].requires_grad


def test_head_treina_so_norma_e_lm_head():
    m = FakeLM()
    info = train.set_trainable(m, "head")
    assert info["trainable"] == 104 and not info["tied_head"]
    assert all(not p.requires_grad for b in m.layers for p in b.ps) and not m.embed.ps[0].requires_grad


def test_head_com_ultimos_k_blocos():
    m = FakeLM()
    info = train.set_trainable(m, "head", last_k=2)
    assert info["trainable"] == 104 + 2 * 20
    assert [all(p.requires_grad for p in b.ps) for b in m.layers] == [False, False, True, True]


def test_lm_head_amarrada_e_sinalizada():
    m = FakeLM(tied=True)
    info = train.set_trainable(m, "head")
    assert info["tied_head"] and m.embed.ps[0].requires_grad       # treinar o head treina as embeddings junto
    with pytest.raises(ValueError):
        train.set_trainable(m, "xyz")


def test_find_blocks():
    assert len(train.find_blocks(FakeLM(n_layers=6))) == 6


def test_early_stop():
    es = EarlyStop(patience=2)
    seq = [(1, 3.0), (2, 2.0), (3, 2.5), (4, 2.6)]
    out = [es.update(l, e) for e, l in seq]
    assert out == [(True, False), (True, False), (False, False), (False, True)]
    assert es.best == 2.0 and es.best_epoch == 2


def test_resolve_adapter(tmp_path):
    lora = tmp_path / "lora"; lora.mkdir()
    (lora / "adapter_config.json").write_text("{}"); (lora / "train_info.json").write_text(json.dumps({"load_4bit": False}))
    cfg = {"hf_id": "org/base", "load_4bit": True}
    assert resolve_adapter(str(lora), "hf", cfg, {}) == str(lora) and cfg["load_4bit"] is False   # segue o treino
    cfg2 = {"hf_id": "org/base", "load_4bit": True}
    resolve_adapter(str(lora), "hf", cfg2, {"load_4bit": True})                                    # chamador manda
    assert cfg2["load_4bit"] is True

    full = tmp_path / "full"; full.mkdir()
    (full / "config.json").write_text("{}"); (full / "train_info.json").write_text(json.dumps({"load_4bit": False}))
    cfg3 = {"hf_id": "org/base"}
    assert resolve_adapter(str(full), "hf", cfg3, {}) is None and cfg3["hf_id"] == str(full)

    two = tmp_path / "two"; two.mkdir()
    (two / "adapter_config.json").write_text("{}"); (two / "train_info.json").write_text(json.dumps({"hf_only": True}))
    with pytest.raises(ValueError, match="vLLM"):
        resolve_adapter(str(two), "vllm", {"hf_id": "x"}, {})
    with pytest.raises(FileNotFoundError):
        resolve_adapter(str(tmp_path / "nao_existe"), "hf", {}, {})
    vazio = tmp_path / "vazio"; vazio.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_adapter(str(vazio), "hf", {}, {})


def test_cleanup_so_apaga_checkpoint_e_guarda_o_info(tmp_path):
    import subprocess, sys
    ck = tmp_path / "ck"; ck.mkdir()
    (ck / "train_info.json").write_text('{"peak_gb": 12}'); (ck / "model.safetensors").write_text("x")
    keep = tmp_path / "rep" / "info.json"
    subprocess.run([sys.executable, "-m", "mpsp", "cleanup", "--dir", str(ck), "--keep-info", str(keep)], check=True, capture_output=True)
    assert not ck.exists() and json.loads(keep.read_text()) == {"peak_gb": 12}
    outro = tmp_path / "docs"; outro.mkdir(); (outro / "a.txt").write_text("importante")
    r = subprocess.run([sys.executable, "-m", "mpsp", "cleanup", "--dir", str(outro)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode != 0 and (outro / "a.txt").exists()          # não apaga o que não é checkpoint
