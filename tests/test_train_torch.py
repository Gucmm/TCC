"""Treino de verdade, em CPU, com um Llama minúsculo de pesos aleatórios (sem baixar nada).

Precisa de torch, transformers e peft; sem eles os testes são ignorados. Rode no servidor ANTES dos jobs longos:
    pytest tests/test_train_torch.py -q        # ~1–3 min em CPU
Não cobre o que exige GPU: quantização 4 bits (qlora), AdamW de 8 bits (full) e bf16 — esses o preflight só confere
com `--dry-run` e o smoke test.
"""
import dataclasses
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from mpsp.slm import train  # noqa: E402
from mpsp.slm.earlystop import EarlyStop  # noqa: E402
from mpsp.slm.methods import METHODS  # noqa: E402


class Tok:
    eos_token, pad_token_id = "<e>", 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        s = "".join(f"<{m['role']}>{m['content']}<e>" for m in msgs)
        return s + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 500 + 1 for c in text]}

    def save_pretrained(self, path):
        (path / "tok.txt").write_text("stub")


def tiny(tie=False):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(vocab_size=520, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
                                        tie_word_embeddings=tie))


def recs(n, tag):
    return [{"prompt": f"{tag} documento {i} com algum texto", "completion": f'{{"x":{i}}}'} for i in range(n)]


def args(**kw):
    base = dict(max_seq_len=300, batch=2, grad_accum=1, lora_r=4, seed=0, unfreeze_last_k=0)
    base.update(kw)
    return SimpleNamespace(**base)


def names(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


def test_set_trainable_no_llama_real():
    m = tiny()
    info = train.set_trainable(m, "head")
    assert names(m) == {"lm_head.weight", "model.norm.weight"} and not info["tied_head"]
    train.set_trainable(m, "head", last_k=1)
    assert any(n.startswith("model.layers.2.") for n in names(m)) and not any(n.startswith("model.layers.1.") for n in names(m))
    train.set_trainable(m, "all")
    assert "model.embed_tokens.weight" not in names(m) and "model.layers.0.self_attn.q_proj.weight" in names(m)
    t = tiny(tie=True)
    assert train.set_trainable(t, "head")["tied_head"] and t.lm_head.weight is t.model.embed_tokens.weight


def test_restauracao_dos_melhores_pesos():
    model = torch.nn.Linear(3, 3)
    model.config = SimpleNamespace()
    stopper = EarlyStop(patience=1)
    cb = train._make_callback(model, stopper)
    ctl = SimpleNamespace(should_training_stop=False)
    cb.on_evaluate(None, SimpleNamespace(epoch=1), ctl, metrics={"eval_loss": 2.0})
    best = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(5.0)
    cb.on_evaluate(None, SimpleNamespace(epoch=2), ctl, metrics={"eval_loss": 3.0})        # piorou: para
    assert ctl.should_training_stop
    cb.restore()
    assert torch.allclose(model.weight, best)


@pytest.mark.parametrize("name", ["lora", "full", "frozen", "two_phase", "two_phase_ner"])
def test_metodo_treina_e_salva(name, tmp_path):
    m = METHODS[name]
    m = dataclasses.replace(m, optim="adamw_torch",
                            phases=tuple(dataclasses.replace(p, epochs=2, patience=1, lr=1e-3) for p in m.phases))
    datasets = {"mpsp": (recs(6, "mpsp"), recs(2, "mpsp")), "ner": (recs(6, "ner"), recs(2, "ner"))}
    model = tiny()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    res = train.train_model(model, Tok(), m, m.phases, datasets, args(), tmp_path)
    assert len(res["phases"]) == len(m.phases) and all(p["epochs_run"] >= 1 for p in res["phases"])
    assert all(p["best_eval_loss"] is not None for p in res["phases"])
    if name in ("full", "frozen"):
        assert (tmp_path / "config.json").exists()
        from transformers import AutoModelForCausalLM
        loaded = AutoModelForCausalLM.from_pretrained(tmp_path)
        changed = {n for n, p in loaded.named_parameters() if not torch.equal(p, before[n])}
        assert changed                                                       # algo foi treinado
        if name == "frozen":
            assert changed <= {"lm_head.weight", "model.norm.weight"}        # backbone intacto
        else:
            assert "model.embed_tokens.weight" not in changed                # embeddings congeladas
    else:
        assert (tmp_path / "adapter_config.json").exists()
        from peft import PeftModel
        PeftModel.from_pretrained(tiny(), tmp_path)                          # o adapter recarrega
        if name == "two_phase":
            import json
            assert "lm_head" in json.loads((tmp_path / "adapter_config.json").read_text())["modules_to_save"]


def test_two_phase_recusa_lm_head_amarrada(tmp_path):
    m = METHODS["two_phase"]
    with pytest.raises(SystemExit, match="amarradas"):
        train.train_model(tiny(tie=True), Tok(), m, m.phases, {"mpsp": (recs(4, "a"), recs(2, "a"))}, args(), tmp_path)


def test_sem_val_roda_as_epocas_fixas(tmp_path):
    m = dataclasses.replace(METHODS["lora"], phases=(dataclasses.replace(METHODS["lora"].phases[0], epochs=1, lr=1e-3),))
    res = train.train_model(tiny(), Tok(), m, m.phases, {"mpsp": (recs(4, "a"), [])}, args(), tmp_path)
    assert res["phases"][0]["best_eval_loss"] is None and (tmp_path / "adapter_config.json").exists()
