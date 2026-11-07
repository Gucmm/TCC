"""
Consolida os logs de avaliação do mpsp-extraction em UM único CSV/JSON limpo.

- Lê os eval_*.log dos 5 diretórios de estado (lab, lab_granite_tp, lab_lora,
  lab_outros, lab_restante).
- Cada log pode ter MAIS de uma execução do eval anexada (retentativas); usa SEMPRE
  a ÚLTIMA (que é o resultado final, já com os folds completos).
- Normaliza o nome em (modelo, condição), classifica a "família" do método e
  calcula a taxa de saídas inválidas (1 - ok_rate).

Uso:
    python consolidate.py --logs C:/caminho/logs --out results_unified.csv
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

MODELS = ("qwen2.5-7b", "llama3.1-8b", "falcon3-7b", "granite3.3-8b")

def family(cond: str) -> str:
    return {
        "0shot": "Zero-shot",
        "1shot": "Few-shot",
        "3shot": "Few-shot",
        "qlora": "QLoRA",
        "qlora_lr1e-4": "QLoRA (lr 1e-4)",
        "lora": "LoRA",
        "full": "Full",
        "frozen": "Frozen",
        "two_phase": "Two-phase",
        "two_phase_ner": "Two-phase NER",
        "two_phase_ner_lr2e-4": "Two-phase NER (lr 2e-4)",
    }.get(cond, cond)

def kind(cond: str) -> str:
    return "baseline" if cond in ("0shot", "1shot", "3shot") else "finetuned"

RE_HEADER = re.compile(r"==\s*runs/([A-Za-z0-9._-]+)\.jsonl")
RE_STRICT = re.compile(
    r"\[strict\]\s+P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+)\s+\(n=\d+\)\s*"
    r"\|\s*macro-F1 campos=([\d.]+)\s*\|\s*F1/doc=([\d.]+)\s*IC95%\[([\d.]+),\s*([\d.]+)\]"
)
RE_SOFT = re.compile(
    r"\[soft\s*\]\s+P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+)\s+\(n=\d+\)\s*"
    r"\|\s*macro-F1 campos=([\d.]+)\s*\|\s*F1/doc=([\d.]+)\s*IC95%\[([\d.]+),\s*([\d.]+)\]"
)

def parse_eval_log(path: str):
    txt = open(path, encoding="utf-8").read()
    # divide por blocos de eval (cada um começa com "== runs/...")
    parts = RE_HEADER.split(txt)  # [antes, nome1, corpo1, nome2, corpo2, ...]
    # reconstrói blocos (nome, corpo) e pega o último
    blocks = []
    for i in range(1, len(parts) - 1, 2):
        blocks.append((parts[i], parts[i + 1]))
    if not blocks:
        return None
    name, body = blocks[-1]  # ÚLTIMO bloco = resultado final

    m = re.match(r"(" + "|".join(MODELS) + r")_T1_(.+)", name)
    if not m:
        return None
    model, cond = m.group(1), m.group(2)

    strict = RE_STRICT.search(body)
    soft = RE_SOFT.search(body)
    exec_json = re.search(r"Execução:\s*(\{.*\})", body, flags=re.S)
    run = json.loads(exec_json.group(1)) if exec_json else {}

    f = lambda s: float(s)
    return {
        "model": model,
        "condition": cond,
        "family": family(cond),
        "kind": kind(cond),
        "strict_f1": f(strict.group(3)) if strict else None,
        "soft_f1": f(soft.group(3)) if soft else None,
        "macro_f1_fields_soft": f(soft.group(4)) if soft else None,
        "doc_f1_soft": f(soft.group(5)) if soft else None,
        "ci_lo": f(soft.group(6)) if soft else None,
        "ci_hi": f(soft.group(7)) if soft else None,
        "ok_rate": run.get("ok_rate"),
        "invalid_rate": (1 - run["ok_rate"]) if "ok_rate" in run else None,
        "context_overflow_rate": run.get("context_overflow_rate"),
        "node_failure_rate": run.get("node_failure_rate"),
        "calls_per_doc": run.get("calls_per_doc"),
        "prompt_tokens": run.get("prompt_tokens"),
        "completion_tokens": run.get("completion_tokens"),
        "seconds_per_doc": run.get("seconds"),
        "grounding_rate": run.get("grounding_rate"),
        "rule_violation_rate": run.get("rule_violation_rate"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=r"C:\Users\gucmm\Downloads\logs")
    ap.add_argument("--out", default="results_unified.csv")
    args = ap.parse_args()

    rows, seen = [], {}
    for f in sorted(glob.glob(os.path.join(args.logs, "*", "eval_*.log"))):
        r = parse_eval_log(f)
        if not r:
            print("ignorado (sem bloco válido):", f)
            continue
        key = (r["model"], r["condition"])
        if key in seen:
            # já visto: mantém o que veio antes (mesma condição em dirs distintos não deve ocorrer)
            print("duplicado:", key, "<-", os.path.basename(f))
            continue
        seen[key] = r
        rows.append(r)

    rows.sort(key=lambda r: (-(r["soft_f1"] or 0)))

    cols = ["model", "condition", "family", "kind", "strict_f1", "soft_f1",
            "macro_f1_fields_soft", "doc_f1_soft", "ci_lo", "ci_hi", "ok_rate",
            "invalid_rate", "context_overflow_rate", "node_failure_rate",
            "calls_per_doc", "prompt_tokens", "completion_tokens",
            "seconds_per_doc", "grounding_rate", "rule_violation_rate"]

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

    # também grava JSON (para quem prefere)
    jout = os.path.splitext(args.out)[0] + ".json"
    with open(jout, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)

    print(f"total de execuções: {len(rows)}")
    print("famílias:", {f: sum(1 for r in rows if r['family'] == f) for f in sorted({r['family'] for r in rows})})
    print("→", args.out, "e", jout)


if __name__ == "__main__":
    main()
