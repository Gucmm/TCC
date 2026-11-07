"""Métricas de execução (tokens, latência, validade, ancoragem) e resumo com IC."""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import canonical, evaluate as ev, postprocess, stats


def preds_exist(path: str) -> bool:
    p = Path(path)
    return p.exists() or bool(list(p.parent.glob(f"{p.stem}.shard*of*.jsonl")) or list(p.parent.glob(f"{p.stem}.fold*.jsonl")))


def load_preds(path: str) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """JSONL de execução (ou JSON legado {id: saída}). → (predições canônicas, registros)."""
    p = Path(path)
    parts = lambda: sorted(set(p.parent.glob(f"{p.stem}.shard*of*.jsonl")) | set(p.parent.glob(f"{p.stem}.fold*.jsonl")))  # noqa: E731
    if p.suffix == ".jsonl" or parts():
        # junta o arquivo e suas partes (`run --shard i/n` → .shard<i>of<n>; `run --fold f` → .fold<f>)
        files = ([p] if p.exists() else []) + parts()
        recs = {}
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    recs[r["doc_id"]] = r
        return {i: (r["prediction"] if r.get("ok") or r.get("prediction") else None) for i, r in recs.items()}, recs
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {i: canonical.canonicalize(v) for i, v in raw.items()}, {}


def run_metrics(recs: Dict[str, dict], texts: Dict[str, str]) -> Dict[str, float]:
    if not recs:
        return {}
    R = list(recs.values())
    nodes = [[n for n in r.get("nodes", []) if not n.get("skipped")] for r in R]
    errs = [[n.get("error") for n in ns] for ns in nodes]
    checked = grounded = viol_docs = viol_total = 0
    for r in R:
        v = postprocess.rule_violations(r["prediction"])
        viol_docs += bool(v)
        viol_total += len(v)
        g = postprocess.grounding(r["prediction"], texts.get(r["doc_id"], ""))
        checked += g["checked"]
        grounded += g["grounded"]
    s = lambda key: float(np.mean([sum(n.get(key, 0) for n in ns) for ns in nodes]))  # noqa: E731
    return {
        "docs": len(R), "ok_rate": float(np.mean([bool(r["ok"]) for r in R])),
        "context_overflow_rate": float(np.mean([any(e == "context_overflow" for e in es) for es in errs])),
        "node_failure_rate": float(np.mean([any(e for e in es) for es in errs])),
        "calls_per_doc": float(np.mean([len(ns) for ns in nodes])),
        "prompt_tokens": s("prompt_tokens"), "completion_tokens": s("completion_tokens"), "seconds": s("seconds"),
        "rule_violation_rate": viol_docs / len(R), "rule_violations_per_doc": viol_total / len(R),
        "grounding_rate": grounded / checked if checked else float("nan"),
    }


def summarize(gt: Dict[str, dict], preds: Dict[str, dict], recs: Dict[str, dict],
              texts: Dict[str, str], ids: List[str] = None, n_boot: int = 2000) -> dict:
    ids = ids or sorted(gt)
    rep = ev.evaluate(gt, preds, ids)
    out = {"n_docs": len(ids), "failed_docs": len(rep.failed_docs)}
    for mode in ("strict", "soft"):
        mean, lo, hi = stats.bootstrap_ci(rep.doc_scores(mode).reindex(ids).fillna(0.0).values, n_boot)
        out[mode] = {**rep.overall(mode), "macro_f1_fields": rep.macro_f1(mode),
                     "doc_f1_mean": mean, "doc_f1_ci95": [lo, hi]}
    out["run"] = run_metrics({i: r for i, r in recs.items() if i in set(ids)}, texts)
    return out, rep
