"""CLI: python -m mpsp <comando>

  models        lista o registro de SLMs locais
  download      baixa os pesos de um modelo (só arquivos públicos; não envia dados do projeto)
  validate-gt   valida o GT contra o schema canônico
  split         k-fold estratificado do GT
  run           extrai (GT em k-fold, ou corpus não rotulado)
  eval          métricas + IC + custo + validade
  compare       teste pareado entre execuções
  bench         benchmarks externos nas mesmas condições
  plan          lista os jobs do plano de experimentos (configs/experiments.yaml)
  schedule      executa o plano na GPU, vários jobs em paralelo respeitando a VRAM (--budget-hours para prazo)
  status        progresso do plano, duração média por tipo de job e ETA grosseira
  leaderboard   tabela com os resultados de todas as execuções avaliadas (reports/*/summary.json)
  cleanup       apaga um checkpoint pesado depois de avaliado (guarda o train_info.json)
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import canonical, data, offline, report, stats
from .backends import get_backend, load_registry, model_info
from .extract import extract_batch
from .fewshot import TfidfRetriever, select_shots

DEFAULT_CONFIG = os.environ.get("MPSP_EXPERIMENTS", "configs/experiments.yaml")


def _configure_stdio() -> None:
    """Garante UTF-8 no console (evita UnicodeEncodeError em cp932/cp1252 no Windows)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def _resolve_config(path: str) -> str:
    """Aceita caminho relativo ao cwd, à raiz do repositório ou ao prefixo de instalação."""
    p = Path(path)
    if p.exists():
        return str(p)
    for base in (Path(__file__).resolve().parents[1], Path(sys.prefix)):
        candidate = base / path
        if candidate.exists():
            return str(candidate)
    return str(p)


def _split_ids(splits: dict, fold: int):
    f = splits["folds"][fold]
    return f["test"], f["train"] + f["val"]


def cmd_models(a):
    print(f"{'modelo':16} {'parâmetros':>10} {'ctx':>7} {'gated':>5}  licença")
    for n, i in load_registry()["slm"].items():
        print(f"{n:16} {i['params_b']:>10} {i['ctx_tokens']:>7} {str(i['gated']):>5}  {i['license']}")


def cmd_download(a):
    """Baixa arquivos públicos do modelo (nenhum dado do projeto é enviado)."""
    from huggingface_hub import snapshot_download
    info = model_info(a.model)
    path = snapshot_download(repo_id=info["hf_id"], token=os.environ.get("HF_TOKEN") or None)
    print(f"{info['hf_id']} → {path}\nA partir daqui, use `run`/treino: rodam com a rede do Hugging Face bloqueada.")


def cmd_validate_gt(a):
    """Confere o GT e explica de onde vem cada documento (ou a falta dele): schema, texto em disco, partições, predições."""
    raw = json.loads(Path(a.gt).read_text(encoding="utf-8"))
    gt, errors = data.load_gt(a.gt)
    print(f"GT: {len(raw)} entradas | válidas no schema: {len(gt)} | com erro de schema: {len(errors)}")
    for e in errors:
        print("  ✗", e)
    texts = data.load_texts(gt, a.docs) if a.docs else {}
    if a.docs:
        miss = [i for i in gt if i not in texts]
        print(f"Com texto não vazio em {a.docs}: {len(texts)} | sem texto: {len(miss)} {miss[:10]}")
    usable = set(gt) & set(texts) if a.docs else set(gt)
    print(f"Documentos utilizáveis (schema válido e texto): {len(usable)}")
    if a.splits:
        sp = json.loads(Path(a.splits).read_text(encoding="utf-8"))
        tests = [i for f in sp["folds"] for i in f["test"]]
        dup = len(tests) - len(set(tests))
        print(f"Partições: {len(set(tests))} documentos de teste em {sp['k']} folds ({dup} repetidos) | "
              f"utilizáveis fora das partições: {len(usable - set(tests))} {sorted(usable - set(tests))[:10]}")
        if a.pred:
            _, recs = report.load_preds(a.pred)
            ok = {i for i, r in recs.items() if r.get("ok")}
            print(f"Predições em {a.pred}: {len(recs)} documentos | saída válida: {len(ok)} | "
                  f"testes sem predição: {len(set(tests) - set(recs))} {sorted(set(tests) - set(recs))[:10]}")
    cls = pd.Series([g["resumo_processo"]["classificacao_crime"] for g in gt.values()]).value_counts()
    print(cls.to_string())


def cmd_split(a):
    gt, _ = data.load_gt(a.gt)
    texts = data.load_texts(gt, a.docs)
    s = data.make_splits(gt, texts, a.k, a.seed, a.val_frac)
    Path(a.out).write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    for i, f in enumerate(s["folds"]):
        print(f"fold {i}: treino={len(f['train'])} val={len(f['val'])} teste={len(f['test'])}")
    print("→", a.out)


def _parse_shard(spec):
    if not spec:
        return 0, 1
    i, n = (int(x) for x in spec.split("/"))
    if not 0 <= i < n:
        raise SystemExit("--shard deve ser i/n com 0 <= i < n (ex.: 0/4)")
    return i, n


def cmd_run(a):
    gt, _ = data.load_gt(a.gt)
    shard_i, shard_n = _parse_shard(a.shard)
    out = Path(a.out)
    if a.fold is not None:      # um fold por processo/job: arquivos separados, juntados pelo eval
        out = out.with_name(f"{out.stem}.fold{a.fold}{out.suffix}")
    if shard_n > 1:      # cada processo grava no próprio arquivo (sem escrita concorrente)
        out = out.with_name(f"{out.stem}.shard{shard_i}of{shard_n}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["doc_id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    backends = {}

    if a.source == "gt":
        splits = json.loads(Path(a.splits).read_text(encoding="utf-8"))
        jobs = [(f, *_split_ids(splits, f)) for f in range(splits["k"]) if a.fold in (None, f)]
        texts = data.load_texts(gt, a.docs)
    else:
        ids = [i for i in data.list_corpus(a.docs, a.include or [], a.exclude or []) if i not in gt]
        texts = data.load_texts(ids + list(gt), a.docs)
        jobs = [(-1, ids, [i for i in gt if i in texts])]

    def backend_for(fold):
        key = fold if a.adapter else 0
        if key not in backends:
            ad = a.adapter.format(fold=fold) if a.adapter else None
            extra = {"gpu_memory_utilization": a.gpu_util} if a.gpu_util else {}
            if a.gen_batch:
                extra["micro_batch"] = a.gen_batch
            backends[key] = get_backend(a.model, adapter=ad, constrained=a.constrained, engine=a.engine, **extra)
        return backends[key]

    n_done = 0
    for fold, test_ids, pool_ids in jobs:
        # o shard é definido sobre a lista completa (estável entre retomadas), depois filtra o que já foi feito
        mine = [d for k, d in enumerate(sorted(test_ids)) if k % shard_n == shard_i]
        todo = [i for i in mine if i in texts and i not in done][: a.limit or None]
        if not todo:
            continue
        backend = backend_for(fold)
        pool = {i: texts[i] for i in pool_ids if i in texts}
        retr = TfidfRetriever(pool) if a.shots and pool else None
        for s in range(0, len(todo), a.batch_size):
            chunk = todo[s:s + a.batch_size]
            items = [{"doc_id": d, "text": texts[d],
                      "shots": select_shots(retr, texts, gt, d, texts[d], a.shots, a.max_shot_chars) if retr else [],
                      "oracle_context": canonical.context_from(gt[d]) if a.oracle and d in gt else None}
                     for d in chunk]
            t0 = time.perf_counter()
            results = extract_batch(items, backend, a.topology, early_exit=not a.no_early_exit)
            with open(out, "a", encoding="utf-8") as f:
                f.write("".join(r.to_json() + "\n" for r in results))
            dt = time.perf_counter() - t0
            n_done += len(results)
            print(f"[fold {fold}] +{len(results)} docs em {dt:.0f}s ({dt / len(results):.1f}s/doc) | "
                  f"ok={sum(r.ok for r in results)}/{len(results)} | total {n_done}", flush=True)
    print(f"{n_done} documentos processados → {out}")


def _fmt(d):
    return f"P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} (n={d['support']})"


def cmd_eval(a):
    if not report.preds_exist(a.pred):
        print(f"sem predições em {a.pred} (nada para avaliar ainda)")
        return
    gt, _ = data.load_gt(a.gt)
    preds, recs = report.load_preds(a.pred)
    ids = sorted(gt) if a.require_all or not recs else sorted(set(recs) & set(gt))
    texts = data.load_texts(ids, a.docs) if a.docs else {}
    summary, rep = report.summarize(gt, preds, recs, texts, ids)
    print(f"== {a.pred} | {summary['n_docs']} docs | sem predição: {summary['failed_docs']}")
    for m in ("strict", "soft"):
        s = summary[m]
        print(f"[{m:6}] {_fmt(s)} | macro-F1 campos={s['macro_f1_fields']:.3f} | "
              f"F1/doc={s['doc_f1_mean']:.3f} IC95%[{s['doc_f1_ci95'][0]:.3f}, {s['doc_f1_ci95'][1]:.3f}]")
    print("\nPor seção (soft):\n" + rep.by_section("soft").round(3).to_string())
    print("\n5 piores campos (soft, com suporte ≥ 5):")
    bf = rep.by_field("soft")
    print(bf[bf.support >= 5].head(5).round(3).to_string())
    if summary["run"]:
        print("\nExecução:", json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                       for k, v in summary["run"].items()}, ensure_ascii=False))
    if not rep.role_confusion().empty:
        print("\nConfusão de papéis (linhas=GT, colunas=predição):\n" + rep.role_confusion().to_string())
    if a.out:
        o = Path(a.out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        rep.rows.to_csv(o / "rows.csv", index=False)
        for m in ("strict", "soft"):
            rep.by_field(m).to_csv(o / f"by_field_{m}.csv")
            rep.doc_scores(m).to_csv(o / f"doc_scores_{m}.csv")
        rep.role_confusion().to_csv(o / "role_confusion.csv")


def cmd_compare(a):
    gt, _ = data.load_gt(a.gt)
    scores, names = [], []
    existing = [p for p in a.pred if report.preds_exist(p)]
    for p in set(a.pred) - set(existing):
        print(f"(ignorado: sem predições em {p})")
    if len(existing) < 2:
        print("menos de 2 execuções com predições; nada a comparar")
        return
    a.pred = existing
    for p in a.pred:
        preds, recs = report.load_preds(p)
        ids = sorted(set(recs) & set(gt)) if recs else sorted(gt)
        scores.append(report.ev.evaluate(gt, preds, ids).doc_scores(a.mode))
        names.append(Path(p).stem)
    common = sorted(set.intersection(*[set(s.index) for s in scores]))
    base = scores[0].loc[common].values
    rows, pv = [], {}
    for n, s in zip(names[1:], scores[1:]):
        r = stats.paired_bootstrap(s.loc[common].values, base)
        r["wilcoxon_p"] = stats.wilcoxon_p(s.loc[common].values, base)
        rows.append({"comparação": f"{n} − {names[0]}", **r})
        pv[f"{n} − {names[0]}"] = r["p"]
    h = stats.holm(pv)
    df = pd.DataFrame(rows)
    df["p_holm"] = df["comparação"].map(h)
    text = f"{len(common)} documentos em comum | métrica: F1/doc ({a.mode})\n" + df.round(4).to_string(index=False)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n", encoding="utf-8")


def cmd_bench(a):
    from .benchmarks.runner import run_cli
    run_cli(a)


def cmd_cleanup(a):
    d = Path(a.dir)
    if not any((d / f).exists() for f in ("train_info.json", "config.json", "adapter_config.json")):
        raise SystemExit(f"{d} não parece um checkpoint (sem train_info/config); nada apagado")
    if a.keep_info and (d / "train_info.json").exists():
        Path(a.keep_info).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(d / "train_info.json", a.keep_info)
    shutil.rmtree(d)
    print(f"apagado: {d}")


def _jobs_from(a):
    from .scheduler import build_jobs, load_config, restrict
    cfg = load_config(_resolve_config(a.config))
    for key in ("max_train", "max_parallel"):          # sobrescritas da linha de comando
        if getattr(a, key, None):
            cfg.setdefault("gpu", {})[key] = getattr(a, key)
    return cfg, restrict(build_jobs(cfg), a.only.split(",") if a.only else None)


def cmd_plan(a):
    cfg, jobs = _jobs_from(a)
    kinds = pd.Series([j.kind for j in jobs]).value_counts()
    print(f"{len(jobs)} jobs\n" + kinds.to_string())
    print(f"exclusivos (rodam sozinhos): {sum(j.exclusive for j in jobs)} | maior estimativa de VRAM: "
          f"{max((j.est_gb for j in jobs), default=0):.0f} GB (GPU utilizável: {cfg.get('gpu', {}).get('total_gb', 46)} GB)")
    if a.verbose:
        for j in sorted(jobs, key=lambda j: (j.prio, j.id)):
            print(f"{j.prio:3d} {j.kind:8} {j.est_gb:5.1f} GB {'EXCL ' if j.exclusive else '     '}{j.id}")
    ckpt_gb = sum(float(j.meta.get("gb_ckpt", 0.0)) for j in jobs if j.kind == "train")
    if ckpt_gb:
        cleaned = any(j.kind == "cleanup" for j in jobs)
        print(f"checkpoints completos (full/frozen): ≈ {ckpt_gb:.0f} GB em disco"
              + (" (apagados pelo cleanup depois da inferência)" if cleaned else " — ficam no disco (este plano não faz inferência/cleanup); confira `df -h /store`"))
    if getattr(a, "estimate", False):
        from .scheduler import _tm, estimate_hours
        e = estimate_hours(jobs, cfg)
        rows = [{"tipo": k, "jobs": e["counts"][k], "horas (otimista)": round(v[0], 1), "horas (pessimista)": round(v[1], 1)}
                for k, v in sorted(e["by_kind"].items())]
        print("\nESTIMATIVA DE TEMPO (premissas, não medições):\n" + pd.DataFrame(rows).to_string(index=False))
        lo, hi = e["makespan"]
        print(f"\nem série: {e['serial'][0]:.0f}–{e['serial'][1]:.0f} h ({e['serial'][0] / 24:.1f}–{e['serial'][1] / 24:.1f} dias)")
        print(f"com sobreposição (1 treino + inferências juntas; exclusivos sozinhos): {lo:.0f}–{hi:.0f} h "
              f"({lo / 24:.1f}–{hi / 24:.1f} dias)")
        tm = _tm(cfg)
        print(f"premissas: {tm['infer_s_per_doc'][0]:.0f}–{tm['infer_s_per_doc'][1]:.0f} s/doc (T1 zero-shot) | treino/fold (min): "
              + ", ".join(f"{k} {v[0]}–{v[1]}" for k, v in tm["train_min"].items())
              + f" | inferências ganham {tm['overlap_speedup'][1]}–{tm['overlap_speedup'][0]}x em paralelo")
        print("Troque por valores medidos com `time_model:` no YAML (ou use `status` depois da 1ª hora de execução).")


def cmd_schedule(a):
    from .scheduler import Scheduler
    cfg, jobs = _jobs_from(a)
    if a.dry_run:
        return cmd_plan(argparse.Namespace(config=a.config, only=a.only, verbose=True, estimate=True,
                                           max_train=a.max_train, max_parallel=a.max_parallel))
    g = cfg.get("gpu", {})
    res = Scheduler(jobs, state_dir=a.state, total_gb=float(g.get("total_gb", 46)), max_parallel=int(g.get("max_parallel", 4)),
                    max_train=int(g.get("max_train", 1)), settle=float(g.get("settle_s", 45)),
                    budget_h=a.budget_hours, tail_h=a.tail_hours).run()
    print(f"concluídos={len(res['done'])} falhas={res['failed']} bloqueados={len(res['blocked'])} "
          f"adiados={len(res['postponed'])}")
    raise SystemExit(1 if res["failed"] else 0)


def cmd_status(a):
    from .scheduler import done_ids, eta, marker_name
    cfg, jobs = _jobs_from(a)
    state = Path(a.state)
    done = done_ids(state)
    done |= {marker_name(j.id) for j in jobs if j.id in done}
    tfile = state / "times.jsonl"
    times = [json.loads(l) for l in tfile.read_text(encoding="utf-8").splitlines() if l.strip()] if tfile.exists() else []
    e = eta(jobs, done, times)
    rows = []
    for kind in sorted({j.kind for j in jobs}):
        ks = [j for j in jobs if j.kind == kind]
        rows.append({"tipo": kind, "feitos": f"{sum(j.id in done or marker_name(j.id) in done for j in ks)}/{len(ks)}",
                     "duração média (min)": round(e["mean_s"].get(kind, float("nan")) / 60, 1)})
    print(pd.DataFrame(rows).to_string(index=False))
    recent_stems = [p.stem for p in state.glob("*.log") if time.time() - p.stat().st_mtime < 180]
    by_marker = {marker_name(j.id): j.id for j in jobs}
    recent = [by_marker.get(s, s) for s in recent_stems]
    print(f"\nem execução (log recente): {sorted(recent) or '—'}")
    print(f"trabalho restante ≈ {e['serial_h']:.1f} h em série | concorrência observada ≈ {e['concurrency']:.1f}x | "
          f"ETA grosseira ≈ {e['eta_h']:.1f} h")
    if e["unknown_kinds"]:
        print(f"(sem duração observada ainda para: {', '.join(e['unknown_kinds'])}; a ETA subestima)")


_TRAINING_TYPE_ORDER = [
    "Zero-shot", "Few-shot", "QLoRA", "LoRA", "Full fine-tuning",
    "Frozen (head only)", "Two-phase (LP-FT)", "Two-phase NER",
    "QLoRA (lr alternativo)", "Two-phase NER (lr alternativo)", "Outros",
]


def _run_parts(name: str):
    """`modelo_T1_tag` -> (modelo, tag). Tolera nomes fora do padrão."""
    if "_T1_" in name:
        model, tag = name.split("_T1_", 1)
        return model, tag
    return name, name


def _training_type(tag: str) -> str:
    """Agrupa a condição de execução em um tipo de treino legível."""
    t = tag.lower()
    if t.startswith("qlora_lr"):
        return "QLoRA (lr alternativo)"
    if t.startswith("two_phase_ner_lr"):
        return "Two-phase NER (lr alternativo)"
    if t.startswith("qlora"):
        return "QLoRA"
    if t.startswith("two_phase_ner"):
        return "Two-phase NER"
    if t.startswith("two_phase"):
        return "Two-phase (LP-FT)"
    if t.startswith("full"):
        return "Full fine-tuning"
    if t.startswith("frozen"):
        return "Frozen (head only)"
    if t.startswith("lora"):
        return "LoRA"
    if t.endswith("shot"):
        k = t.removesuffix("shot")
        return "Zero-shot" if k == "0" else "Few-shot"
    return "Outros"


def cmd_leaderboard(a):
    rows = []
    for f in sorted(Path(a.reports).glob("*/summary.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        run = s.get("run", {})
        soft = s.get("soft", {})
        ci = soft.get("doc_f1_ci95") or [float("nan"), float("nan")]
        ok = run.get("ok_rate", float("nan"))
        model, tag = _run_parts(f.parent.name)
        rows.append({
            "tipo": _training_type(tag),
            "execução": f.parent.name,
            "modelo": model,
            "condição": tag,
            "docs": s.get("n_docs", 0),
            "F1 strict": round(s.get("strict", {}).get("f1", float("nan")), 3),
            "F1 soft": round(soft.get("f1", float("nan")), 3),
            "F1/doc soft IC95%": f"[{ci[0]:.3f}, {ci[1]:.3f}]",
            "válidas": round(ok, 2),
            "inválidas": round(1 - ok, 2),
            "⚠": "⚠" if (ok == ok and ok < 0.8) else "",
            "estouro ctx": round(run.get("context_overflow_rate", float("nan")), 2),
            "s/doc": round(run.get("seconds", float("nan")), 1),
        })
    if not rows:
        print(f"nenhum summary.json em {a.reports}/*/")
        return

    df = pd.DataFrame(rows)
    if getattr(a, "max_invalid", None) is not None:
        df = df[df["inválidas"] <= a.max_invalid]
        if df.empty:
            print(f"nenhuma execução com taxa de inválidas <= {a.max_invalid:.0%}")
            return

    by = getattr(a, "by", "method")
    if by == "method":
        present = [t for t in _TRAINING_TYPE_ORDER if t in set(df["tipo"])]
        groups = [(t, df[df["tipo"] == t]) for t in present]
    elif by == "model":
        groups = [(m, df[df["modelo"] == m]) for m in sorted(df["modelo"].unique())]
    else:
        groups = [("todas as execuções", df)]

    cols = ["execução", "modelo", "condição", "docs", "F1 strict", "F1 soft",
            "F1/doc soft IC95%", "válidas", "inválidas", "⚠", "estouro ctx", "s/doc"]
    for label, sub in groups:
        sub = sub.sort_values("F1 soft", ascending=False)
        print(f"\n=== {label} ({len(sub)} execuções) ===")
        print(sub[cols].to_string(index=False))

    print("\n⚠ = menos de 80% das saídas válidas no schema (taxa alta de inválidas).")
    print("Atenção: execuções com nº de docs diferente (folds incompletos) não são comparáveis; use `compare`.")
    if a.out:
        df.sort_values(["tipo", "F1 soft"], ascending=[True, False]).to_csv(a.out, index=False)


def main(argv=None):
    _configure_stdio()
    p = argparse.ArgumentParser(prog="mpsp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models").set_defaults(fn=cmd_models)

    s = sub.add_parser("download")
    s.add_argument("--model", required=True)
    s.set_defaults(fn=cmd_download)

    s = sub.add_parser("validate-gt", help="valida o GT e reconcilia GT × textos × partições × predições (use --splits/--pred)")
    s.add_argument("--gt", required=True); s.add_argument("--docs")
    s.add_argument("--splits"); s.add_argument("--pred", help="JSONL de uma execução (junta .fold*/.shard*); exige --splits")
    s.set_defaults(fn=cmd_validate_gt)

    s = sub.add_parser("split")
    s.add_argument("--gt", required=True); s.add_argument("--docs", required=True)
    s.add_argument("--k", type=int, default=5); s.add_argument("--seed", type=int, default=42)
    s.add_argument("--val-frac", type=float, default=0.15); s.add_argument("--out", default="splits.json")
    s.set_defaults(fn=cmd_split)

    s = sub.add_parser("run")
    s.add_argument("--gt", required=True); s.add_argument("--docs", required=True)
    s.add_argument("--model", required=True); s.add_argument("--topology", default="T1", choices=["T1", "T2", "T3", "T4", "T5"])
    s.add_argument("--source", choices=["gt", "corpus"], default="gt")
    s.add_argument("--splits", help="obrigatório com --source gt")
    s.add_argument("--shots", type=int, default=0, help="0 = zero-shot; k>0 = few-shot recuperado por TF-IDF")
    s.add_argument("--max-shot-chars", type=int, default=60000)
    s.add_argument("--adapter", help="caminho do adapter LoRA; aceita {fold}, ex.: ckpts/qwen/T1/fold{fold}")
    s.add_argument("--oracle", action="store_true", help="contexto do GT nos nós seguintes (propagação de erro)")
    s.add_argument("--no-early-exit", action="store_true"); s.add_argument("--constrained", action="store_true")
    s.add_argument("--include", nargs="*"); s.add_argument("--exclude", nargs="*")
    s.add_argument("--limit", type=int)
    s.add_argument("--engine", choices=["hf", "vllm"], default="hf", help="vllm = lote contínuo (muito mais rápido)")
    s.add_argument("--batch-size", type=int, default=64, help="documentos por lote; resultados são gravados a cada lote")
    s.add_argument("--fold", type=int, help="processa só este fold (arquivo <nome>.fold<f>.jsonl)")
    s.add_argument("--gen-batch", type=int, help="hf: prompts por geração simultânea (padrão 4; 1 = um por vez). "
                                                 "Se faltar memória, o lote é dividido sozinho")
    s.add_argument("--gpu-util", type=float, help="vLLM: fração da GPU a reservar (ex.: 0.45 para dividir a GPU)")
    s.add_argument("--shard", help="i/n: processa só a fatia i de n (rode n processos em paralelo; ex.: 0/4)")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("eval")
    s.add_argument("--gt", required=True); s.add_argument("--pred", required=True)
    s.add_argument("--docs", help="para ancoragem no texto"); s.add_argument("--out")
    s.add_argument("--require-all", action="store_true", help="documentos sem predição contam como falha")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("compare")
    s.add_argument("--gt", required=True); s.add_argument("--pred", nargs="+", required=True)
    s.add_argument("--mode", choices=["strict", "soft"], default="soft"); s.add_argument("--out")
    s.set_defaults(fn=cmd_compare)

    for name, fn in (("plan", cmd_plan), ("schedule", cmd_schedule)):
        s = sub.add_parser(name)
        s.add_argument("--config", default=DEFAULT_CONFIG)
        s.add_argument("--only", help="tipos de job, separados por vírgula (smoke,infer,bench,train,ft,eval,cleanup,report)")
        s.add_argument("--max-train", type=int, help="treinos simultâneos (2 = sobreposição, se a VRAM permitir)")
        s.add_argument("--max-parallel", type=int, help="processos simultâneos")
        if name == "plan":
            s.add_argument("--verbose", action="store_true"); s.add_argument("--estimate", action="store_true")
        else:
            s.add_argument("--dry-run", action="store_true"); s.add_argument("--state", default="logs/lab")
            s.add_argument("--budget-hours", type=float, default=0.0,
                           help="prazo total (h): para de lançar jobs em (prazo − tail) e encerra no prazo")
            s.add_argument("--tail-hours", type=float, default=1.5, help="folga para os jobs em andamento terminarem")
        s.set_defaults(fn=fn)

    s = sub.add_parser("status")
    s.add_argument("--config", default=DEFAULT_CONFIG); s.add_argument("--only")
    s.add_argument("--state", default="logs/lab"); s.set_defaults(fn=cmd_status)

    s = sub.add_parser("leaderboard")
    s.add_argument("--reports", default="reports")
    s.add_argument("--by", choices=["method", "model", "none"], default="method",
                   help="agrupa por tipo de treino (method), modelo (model) ou tabela única (none)")
    s.add_argument("--max-invalid", type=float,
                   help="oculta execuções com taxa de saída inválida acima deste valor (0-1)")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_leaderboard)

    s = sub.add_parser("cleanup")
    s.add_argument("--dir", required=True); s.add_argument("--keep-info")
    s.set_defaults(fn=cmd_cleanup)

    s = sub.add_parser("bench")
    s.add_argument("--name", default="lener_br"); s.add_argument("--model", required=True)
    s.add_argument("--pattern", choices=["all", "per_type", "both"], default="both")
    s.add_argument("--shots", type=int, default=0)
    s.add_argument("--n", type=int, default=100, help="nº de amostras (use o mesmo nº de documentos do GT)")
    s.add_argument("--window", type=int, default=10, help="sentenças por amostra")
    s.add_argument("--seed", type=int, default=42); s.add_argument("--split", default="test")
    s.add_argument("--cache-dir", default="data/benchmarks"); s.add_argument("--adapter")
    s.add_argument("--out-dir", default="runs/bench")
    s.set_defaults(fn=cmd_bench)

    args = p.parse_args(argv)
    offline.disable_telemetry() if args.cmd == "download" else offline.enforce_offline()
    args.fn(args)


if __name__ == "__main__":
    main()
