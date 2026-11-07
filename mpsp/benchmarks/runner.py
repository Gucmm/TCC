"""Executa um benchmark NER sob as mesmas condições da extração do MPSP."""
import json
from pathlib import Path

from .. import stats
from ..backends import get_backend
from ..fewshot import TfidfRetriever
from . import conll, ner


def run_pattern(backend, types, units, train_by_id, retr, pattern, shots, out_path):
    """`all`: 1 chamada por amostra; `per_type`: 1 por tipo. Cada chamada é enviada ao backend em lote."""
    groups = [dict(types)] if pattern == "all" else [{t: d} for t, d in types.items()]
    ex = {u.id: ([train_by_id[i] for i in retr.query(u.text, shots)][::-1] if retr else []) for u in units}
    recs = {u.id: {"unit_id": u.id, "pattern": pattern, "ok": True, "entities": [], "calls": []} for u in units}
    for g in groups:
        gens = backend.generate_batch([ner.build_prompt(g, u, ex[u.id], backend.needs_schema_text) for u in units],
                                      ner.ner_model(list(g)))
        for u, gen in zip(units, gens):
            r = recs[u.id]
            r["calls"].append(gen.meta())
            if gen.parsed is None:
                r["ok"] = False
            else:
                r["entities"] += [(e.texto, e.tipo) for e in gen.parsed.entidades]
    out = [recs[u.id] for u in units]
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    return out


def run_cli(a):
    cfg = conll.DATASETS[a.name]
    types = cfg["types"]
    units = conll.load(a.name, a.split, a.window, a.cache_dir, a.n, a.seed)
    train = conll.load(a.name, "train", a.window, a.cache_dir)
    train_by_id = {u.id: u for u in train}
    retr = TfidfRetriever({u.id: u.text for u in train}) if a.shots else None
    backend = get_backend(a.model, adapter=a.adapter)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    patterns = ["all", "per_type"] if a.pattern == "both" else [a.pattern]
    tag = f"{a.name}_{a.model}_{a.shots}shot_n{len(units)}_w{a.window}_s{a.seed}"
    print(f"{a.name}: {len(units)} amostras ({a.window} sentenças cada) | modelo={a.model} | {a.shots}-shot")

    scored = {}
    for pat in patterns:
        path = out_dir / f"{tag}_{pat}.jsonl"
        path.unlink(missing_ok=True)
        recs = run_pattern(backend, types, units, train_by_id, retr, pat, a.shots, path)
        s = ner.score(units, {r["unit_id"]: [tuple(e) for e in r["entities"]] for r in recs})
        s["run"] = {"ok_rate": sum(r["ok"] for r in recs) / len(recs),
                    "seconds_per_unit": sum(c["seconds"] for r in recs for c in r["calls"]) / len(recs),
                    "calls_per_unit": sum(len(r["calls"]) for r in recs) / len(recs)}
        scored[pat] = s
        (out_dir / f"{tag}_{pat}.summary.json").write_text(json.dumps(
            {k: v for k, v in s.items() if k != "per_unit"}, ensure_ascii=False, indent=1), encoding="utf-8")
        for m in ("strict", "relaxed"):
            x = s[m]["micro"]
            print(f"[{pat:8}] {m:7} P={x['precision']:.3f} R={x['recall']:.3f} F1={x['f1']:.3f} "
                  f"macro-F1={s[m]['macro_f1']:.3f}")
        print(f"[{pat:8}] ok={s['run']['ok_rate']:.2f} chamadas/amostra={s['run']['calls_per_unit']:.1f} "
              f"s/amostra={s['run']['seconds_per_unit']:.1f}")
    if len(patterns) == 2:
        ids = [u.id for u in units]
        d = stats.paired_bootstrap([scored["per_type"]["per_unit"]["strict"][i] for i in ids],
                                   [scored["all"]["per_unit"]["strict"][i] for i in ids])
        print(f"per_type − all (F1 strict/amostra): Δ={d['diff']:+.3f} IC95%[{d['lo']:+.3f}, {d['hi']:+.3f}] p={d['p']:.3f}")
