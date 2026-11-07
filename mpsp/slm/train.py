"""Treino unificado dos SLMs: um único caminho de dados/avaliação para todos os métodos (ver methods.py).

  python -m mpsp.slm.train --method lora --model falcon3-7b --gt data/labels/ground_truth.json \
      --docs "data/scrappings/treino+teste" --splits splits.json --fold 0 --topology T1 \
      --out ckpts/falcon3-7b/lora/T1/fold0

Cada nó da topologia é um exemplo (prompt do nó → JSON do nó); prompt e renderização de chat são os da inferência.
Exemplos que excedem --max-seq-len são descartados, nunca truncados. A parada antecipada guarda os melhores pesos
treináveis na RAM (sem checkpoints em disco) e os restaura ao final. Saída: adapter LoRA (métodos LoRA) ou modelo
completo (full, frozen), mais train_info.json (contagem de parâmetros, pico de VRAM, tempo, curvas).
"""
import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .. import canonical, data
from ..backends import model_info
from ..prompts import build_prompt, dumps
from ..topologies import TOPOLOGIES, should_run
from .earlystop import EarlyStop
from .methods import METHODS, Method, Phase, estimate_train_gb


def build_records(gt: Dict[str, dict], texts: Dict[str, str], ids: Sequence[str], topology: str,
                  early_exit: bool = True) -> List[dict]:
    recs = []
    for doc in ids:
        canon = gt[doc]
        ctx = canonical.context_from(canon)
        for i, node in enumerate(TOPOLOGIES[topology]):
            if i > 0 and not should_run(node, ctx, early_exit):
                continue
            recs.append({"doc_id": doc, "node": node.name,
                         "prompt": build_prompt(node.emits, texts[doc], None if i == 0 else ctx, (), with_schema=True),
                         "completion": dumps(canonical.project(canon, node.emits))})
    return recs


def subsample(ids: Sequence[str], frac: float, seed: int) -> List[str]:
    """Fração determinística dos documentos de treino (curva de aprendizado)."""
    ids = sorted(ids)
    if frac >= 1.0:
        return ids
    return sorted(random.Random(seed).sample(ids, max(1, round(len(ids) * frac))))


def encode_records(tok, recs: Sequence[dict], max_len: int) -> Tuple[List[dict], int]:
    from .. import chat
    kept, dropped = [], 0
    for r in recs:
        ex = chat.build_example(tok, r["prompt"], r["completion"], max_len)
        if ex is None:
            dropped += 1
        else:
            kept.append(ex)
    return kept, dropped


def plan_phases(method: Method, args) -> List[Phase]:
    """Fases do método com as sobrescritas da linha de comando (--lr/--epochs/--patience valem para a última fase)."""
    phases = list(method.phases)
    last = phases[-1]
    phases[-1] = replace(last, lr=args.lr or last.lr, epochs=args.epochs or last.epochs,
                         patience=args.patience or last.patience)
    if len(phases) > 1:
        first = phases[0]
        phases[0] = replace(first, lr=args.phase1_lr or first.lr, epochs=args.phase1_epochs or first.epochs)
    return phases


def phase_records(phases: Sequence[Phase], gt, texts, train_ids, val_ids, args) -> Dict[str, Tuple[list, list]]:
    out: Dict[str, Tuple[list, list]] = {}
    if any(p.data == "mpsp" for p in phases):
        out["mpsp"] = (build_records(gt, texts, train_ids, args.topology), build_records(gt, texts, val_ids, args.topology))
    if any(p.data == "ner" for p in phases):
        from ..benchmarks import conll, ner
        types = conll.DATASETS["lener_br"]["types"]
        tr = conll.load("lener_br", "train", args.ner_window, args.ner_cache, args.ner_n, args.seed)
        va = conll.load("lener_br", "dev", args.ner_window, args.ner_cache, args.ner_val, args.seed)
        out["ner"] = (ner.training_records(tr, types), ner.training_records(va, types))
    return out


def find_blocks(model) -> list:
    """Blocos do decoder (model.model.layers nas arquiteturas Llama-like)."""
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    best = None
    for _, m in model.named_modules():
        if m.__class__.__name__ == "ModuleList" and (best is None or len(m) > len(best)):
            best = m
    if best is None:
        raise ValueError("não encontrei os blocos do decoder")
    return list(best)


def count_params(model) -> Dict[str, Any]:
    total = trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        trainable += n if p.requires_grad else 0
    return {"total": total, "trainable": trainable, "pct": round(100 * trainable / max(total, 1), 4)}


def set_trainable(model, mode: str, last_k: int = 0, freeze_embeddings: bool = True) -> Dict[str, Any]:
    """`all`: tudo (embeddings congeladas); `head`: norma final + lm_head (+ últimos `last_k` blocos)."""
    for p in model.parameters():
        p.requires_grad = False
    tied = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))
    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
        if freeze_embeddings:
            for p in model.get_input_embeddings().parameters():
                p.requires_grad = False
    elif mode == "head":
        for p in model.get_output_embeddings().parameters():
            p.requires_grad = True
        norm = getattr(getattr(model, "model", None), "norm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad = True
        if last_k:
            for b in find_blocks(model)[-last_k:]:
                for p in b.parameters():
                    p.requires_grad = True
    else:
        raise ValueError(f"modo inválido: {mode}")
    info = count_params(model)
    info.update(mode=mode, last_k=last_k, tied_head=tied)
    return info


def load_base(info: dict, load_4bit: bool, attn: str = None):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from ..compat import dtype_kw
    kw = dict(trust_remote_code=info.get("trust_remote_code", False),
              attn_implementation=attn or info.get("attn_implementation"), **dtype_kw(torch.bfloat16))
    if torch.cuda.is_available():
        kw["device_map"] = {"": torch.cuda.current_device()}      # sem offload silencioso para a CPU
    if load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    return AutoModelForCausalLM.from_pretrained(info["hf_id"], **kw)


def wrap_lora(model, r: int, modules_to_save=None):
    from peft import LoraConfig, get_peft_model
    return get_peft_model(model, LoraConfig(
        r=r, lora_alpha=2 * r, lora_dropout=0.1, target_modules="all-linear", task_type="CAUSAL_LM",
        modules_to_save=modules_to_save))


def _make_callback(model, stopper: EarlyStop):
    from transformers import TrainerCallback

    class BestState(TrainerCallback):
        def __init__(self):
            self.snap = None

        def on_evaluate(self, args, state, control, metrics=None, **kw):
            loss = (metrics or {}).get("eval_loss")
            if loss is None:
                return
            improved, stop = stopper.update(float(loss), float(state.epoch or 0))
            if improved:     # melhores pesos treináveis ficam na RAM
                self.snap = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad}
            if stop:
                control.should_training_stop = True

        def restore(self):
            if self.snap is None:
                return
            import torch
            params = dict(model.named_parameters())
            with torch.no_grad():
                for n, v in self.snap.items():
                    params[n].copy_(v.to(params[n].device))

    return BestState()


def fit(model, train_ds, val_ds, collate, *, lr, epochs, patience, batch, grad_accum, optim, bf16, out_dir, seed,
        grad_ckpt=True) -> Dict[str, Any]:
    """Uma fase de treino com parada antecipada e restauração dos melhores pesos."""
    import torch
    from transformers import Trainer, TrainingArguments
    has_val = len(val_ds) > 0
    kw = dict(output_dir=str(out_dir), num_train_epochs=epochs, learning_rate=lr, lr_scheduler_type="cosine",
              warmup_ratio=0.05, per_device_train_batch_size=batch, per_device_eval_batch_size=batch,
              gradient_accumulation_steps=grad_accum, bf16=bf16, gradient_checkpointing=grad_ckpt,
              gradient_checkpointing_kwargs={"use_reentrant": False},
              eval_strategy="epoch" if has_val else "no", save_strategy="no", logging_steps=5, report_to="none",
              remove_unused_columns=False, seed=seed, optim=optim, max_grad_norm=1.0)
    if not torch.cuda.is_available():
        kw["use_cpu"] = True
    model.config.use_cache = False
    stopper = EarlyStop(patience)
    cb = _make_callback(model, stopper) if has_val else None
    trainer = Trainer(model=model, args=TrainingArguments(**kw), train_dataset=train_ds,
                      eval_dataset=val_ds if has_val else None, data_collator=collate,
                      callbacks=[cb] if cb else [])
    trainer.train()
    if cb:
        cb.restore()
    return {"epochs_run": trainer.state.epoch, "best_eval_loss": None if not has_val else stopper.best,
            "best_epoch": stopper.best_epoch, "eval_history": stopper.history,
            "log_history": trainer.state.log_history}


def train_model(model, tok, method: Method, phases: Sequence[Phase], datasets, args, out: Path) -> Dict[str, Any]:
    """Executa as fases do método sobre `model` e salva o resultado em `out`."""
    import torch
    from .. import chat
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        col = lambda k, v: torch.tensor([b[k] + [v] * (n - len(b[k])) for b in batch])  # noqa: E731
        return {"input_ids": col("input_ids", pad), "attention_mask": col("attention_mask", 0), "labels": col("labels", -100)}

    if method.load_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    if method.name == "two_phase" and getattr(model.config, "tie_word_embeddings", False):
        raise SystemExit("two_phase (lm_head) não suporta embeddings amarradas a lm_head neste modelo; use two_phase_ner.")

    results, peft_done = [], False
    for ph in phases:
        tr_recs, va_recs = datasets[ph.data]
        tr, d_tr = encode_records(tok, tr_recs, args.max_seq_len)
        va, d_va = encode_records(tok, va_recs, args.max_seq_len)
        if not tr:
            raise SystemExit(f"fase {ph.name}: nenhum exemplo coube em --max-seq-len {args.max_seq_len}")
        if ph.trainable == "lora":
            if not peft_done:
                model = wrap_lora(model, args.lora_r,
                                  modules_to_save=["lm_head"] if method.name == "two_phase" else None)
                peft_done = True
            counts = count_params(model)
        else:
            counts = set_trainable(model, ph.trainable, last_k=args.unfreeze_last_k if ph.trainable == "head" else 0)
            if counts["tied_head"]:
                print("AVISO: lm_head amarrada às embeddings; treinar o 'head' altera também as embeddings de entrada.")
        print(f"[{ph.name}] treino={len(tr)}/{len(tr_recs)} (descartados: {d_tr}) val={len(va)}/{len(va_recs)} | "
              f"treináveis={counts['trainable']:,} ({counts['pct']}%) | lr={ph.lr} épocas<={ph.epochs}", flush=True)
        t0 = time.time()
        res = fit(model, tr, va, collate, lr=ph.lr, epochs=ph.epochs, patience=ph.patience, batch=args.batch,
                  grad_accum=args.grad_accum, optim=method.optim, bf16=torch.cuda.is_available(), out_dir=out / f"_{ph.name}",
                  seed=args.seed, grad_ckpt=ph.trainable != "head")
        results.append({"phase": ph.name, "data": ph.data, "lr": ph.lr, "train_examples": len(tr), "train_dropped": d_tr,
                        "val_examples": len(va), "val_dropped": d_va, "params": counts, "seconds": round(time.time() - t0, 1), **res})
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return {"phases": results}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="qlora", choices=list(METHODS))
    ap.add_argument("--model", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--docs", required=True); ap.add_argument("--splits", required=True)
    ap.add_argument("--fold", type=int, required=True, help="-1 = treina com todo o GT (modelo final)")
    ap.add_argument("--topology", default="T1"); ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--epochs", type=int); ap.add_argument("--lr", type=float); ap.add_argument("--patience", type=int)
    ap.add_argument("--batch", type=int, default=1); ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=1.0, help="fração dos documentos de treino (curva de aprendizado)")
    ap.add_argument("--unfreeze-last-k", type=int, default=0, help="frozen/two_phase: libera também os últimos K blocos")
    ap.add_argument("--phase1-epochs", type=int); ap.add_argument("--phase1-lr", type=float)
    ap.add_argument("--ner-n", type=int, default=200, help="two_phase_ner: janelas de treino do LeNER-Br")
    ap.add_argument("--ner-val", type=int, default=30); ap.add_argument("--ner-window", type=int, default=10)
    ap.add_argument("--ner-cache", default="data/benchmarks")
    ap.add_argument("--attn", help="sobrescreve attn_implementation do registro")
    ap.add_argument("--dry-run", action="store_true", help="só monta os dados e estima a memória (sem carregar o modelo)")
    a = ap.parse_args(argv)

    from ..offline import enforce_offline
    enforce_offline()      # dados sigilosos: nada sai da máquina
    method, info = METHODS[a.method], model_info(a.model)
    gt, _ = data.load_gt(a.gt)
    texts = data.load_texts(gt, a.docs)
    splits = json.loads(Path(a.splits).read_text(encoding="utf-8"))
    if a.fold >= 0:
        train_ids, val_ids = splits["folds"][a.fold]["train"], splits["folds"][a.fold]["val"]
    else:   # modelo final: todo o GT; o val do fold 0 serve só de parada antecipada
        all_ids = sorted({i for fo in splits["folds"] for i in fo["test"]})
        val_ids = splits["folds"][0]["val"]
        train_ids = [i for i in all_ids if i not in set(val_ids)]
    train_ids = subsample([i for i in train_ids if i in texts], a.train_frac, a.seed)
    val_ids = [i for i in val_ids if i in texts]
    phases = plan_phases(method, a)
    datasets = phase_records(phases, gt, texts, train_ids, val_ids, a)
    est = estimate_train_gb(info["params_b"], method, a.max_seq_len, a.batch)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(info["hf_id"], trust_remote_code=info.get("trust_remote_code", False))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if a.dry_run:
        for ph in phases:
            tr, d1 = encode_records(tok, datasets[ph.data][0], a.max_seq_len)
            va, d2 = encode_records(tok, datasets[ph.data][1], a.max_seq_len)
            print(f"[{ph.name}] dados={ph.data} treino={len(tr)} (descartados {d1}) val={len(va)} (descartados {d2}) "
                  f"lr={ph.lr} épocas<={ph.epochs}")
        print(f"memória estimada ≈ {est:.1f} GB ({method.desc})")
        return

    import torch
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        if est > 0.95 * total:
            print(f"AVISO: estimativa {est:.0f} GB perto/acima da VRAM ({total:.0f} GB); pode dar OOM. "
                  "Reduza --max-seq-len ou use um método mais leve.", flush=True)
    t0 = time.time()
    model = load_base(info, method.load_4bit, a.attn)
    res = train_model(model, tok, method, phases, datasets, a, out)
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    (out / "train_info.json").write_text(json.dumps({
        "method": a.method, "model": a.model, "topology": a.topology, "fold": a.fold, "train_frac": a.train_frac,
        "max_seq_len": a.max_seq_len, "load_4bit": method.load_4bit, "full_model": method.full_save,
        "hf_only": method.hf_only, "estimate_gb": round(est, 1), "peak_gb": peak,
        "seconds": round(time.time() - t0, 1), **res}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"pronto: {out} | pico de VRAM: {peak and round(peak, 1)} GB")


if __name__ == "__main__":
    main()
