"""Agendador de jobs em uma GPU: roda vários processos ao mesmo tempo, respeitando VRAM e dependências.

Processos pequenos podem compartilhar a GPU quando há memória livre; treinos grandes rodam sozinhos. O ganho de
desempenho dessa concorrência não foi medido. A estimativa de VRAM é heurística e é conferida contra a memória livre
real (nvidia-smi).

Retomável: cada job concluído deixa um marcador em <state>/done/<id> (nome portável, conteúdo = id); reexecutar pula o que já terminou.
"""
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

from .backends import load_registry
from .slm.methods import METHODS, estimate_infer_gb, estimate_train_gb


@dataclass
class Job:
    id: str
    kind: str                       # smoke | infer | bench | train | ft | eval | cleanup | report | corpus
    cmd: List[str]
    est_gb: float = 0.0
    deps: List[str] = field(default_factory=list)
    prio: int = 50                  # menor roda primeiro
    meta: Dict[str, object] = field(default_factory=dict)   # dados para a estimativa de tempo
    exclusive: bool = False         # roda sozinho na GPU
    env: Dict[str, str] = field(default_factory=dict)
    retried: bool = False


def nvidia_free_gb() -> Optional[float]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0]) / 1024
    except Exception:  # noqa: BLE001
        return None


OOM = re.compile(r"out of memory|OutOfMemoryError", re.I)
CHEAP = {"eval", "report", "cleanup"}        # só CPU/disco: rodam mesmo depois do prazo de lançamento


def marker_name(jid: str) -> str:
    """Nome de arquivo portável para o id do job (`:` e separadores não valem no Windows)."""
    return jid.replace("/", "_").replace("\\", "_").replace(":", "_")


def _terminate(proc) -> None:
    """Encerra o processo em Unix (grupo) e Windows (sem `killpg`)."""
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except (OSError, AttributeError):
        try:
            proc.terminate()
        except OSError:
            pass


def done_ids(state_dir) -> set:
    """Ids concluídos: lê o conteúdo do marcador (ou o nome, para marcadores antigos)."""
    d = Path(state_dir) / "done"
    if not d.exists():
        return set()
    out = set()
    for p in d.glob("*"):
        try:
            txt = p.read_text(encoding="utf-8").strip()
        except OSError:
            txt = ""
        out.add(txt or p.name)
    return out


class Scheduler:
    def __init__(self, jobs: List[Job], state_dir: str = "logs/lab", total_gb: float = 46.0, max_parallel: int = 4,
                 max_train: int = 1, settle: float = 45.0, poll: float = 5.0, margin_gb: float = 2.0,
                 stall_s: float = 1800.0, free_fn: Callable[[], Optional[float]] = nvidia_free_gb, log=print,
                 budget_h: float = 0.0, tail_h: float = 1.5):
        self.jobs = {j.id: j for j in jobs}
        for j in jobs:
            for d in j.deps:
                if d not in self.jobs:
                    raise ValueError(f"job {j.id} depende de {d}, que não existe")
        self.state = Path(state_dir)
        self.done_dir = self.state / "done"
        self.done_dir.mkdir(parents=True, exist_ok=True)
        self.total, self.max_parallel, self.max_train = total_gb, max_parallel, max_train
        self.settle, self.poll, self.margin, self.stall = settle, poll, margin_gb, stall_s
        self.free_fn, self.log = free_fn, log
        self.budget_h, self.tail_h = budget_h, tail_h      # budget_h=0: sem prazo

    def _is_done(self, jid: str) -> bool:
        return (self.done_dir / marker_name(jid)).exists() or (self.done_dir / jid).exists()

    def _oom(self, jid: str) -> bool:
        p = self.state / f"{marker_name(jid)}.log"
        try:
            return bool(OOM.search(p.read_text(errors="ignore")[-20000:]))
        except OSError:
            return False

    def _start(self, job: Job):
        logf = open(self.state / f"{marker_name(job.id)}.log", "ab")
        proc = subprocess.Popen(job.cmd, stdout=logf, stderr=subprocess.STDOUT, env={**os.environ, **job.env},
                                start_new_session=(os.name != "nt"))
        return proc, logf, time.time()

    def _record(self, jid: str, rc: int, t0: float):
        with open(self.state / "times.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": jid, "kind": self.jobs[jid].kind, "rc": rc, "seconds": round(time.time() - t0, 1),
                                "ts": time.time()}) + "\n")

    def run(self) -> Dict[str, List[str]]:
        done = {j for j in self.jobs if self._is_done(j)}
        failed, blocked, postponed = set(), set(), set()
        pending = [j for j in self.jobs.values() if j.id not in done]
        running: Dict[str, tuple] = {}
        t_start = time.time()
        launch_until = t_start + (self.budget_h - self.tail_h) * 3600 if self.budget_h else None
        hard_until = t_start + self.budget_h * 3600 if self.budget_h else None
        announced = False
        last_launch, last_progress, last_status = -1e9, time.time(), 0.0
        try:
            while pending or running:
                now = time.time()
                past = launch_until is not None and now > launch_until
                # 0) prazo: depois do prazo de lançamento só rodam avaliações/relatórios; no prazo duro, encerra o que sobrou
                if past and not announced:
                    announced = True
                    for j in [j for j in pending if j.kind not in CHEAP]:
                        pending.remove(j)
                        postponed.add(j.id)
                    self.log(f"⏱ prazo de lançamento atingido: {len(postponed)} jobs adiados (rode de novo para continuar); "
                             "esperando os em andamento e avaliando o que existe")
                if hard_until is not None and now > hard_until and running:
                    self.log("⏱ prazo duro atingido: encerrando os jobs em andamento")
                    for jid, (proc, logf, _) in running.items():
                        _terminate(proc)
                    time.sleep(2)
                # 1) colher términos
                for jid, (proc, logf, t0) in list(running.items()):
                    rc = proc.poll()
                    if rc is None:
                        continue
                    logf.close()
                    del running[jid]
                    job = self.jobs[jid]
                    last_progress = time.time()
                    self._record(jid, rc, t0)
                    if rc == 0:
                        (self.done_dir / marker_name(jid)).write_text(jid, encoding="utf-8")
                        done.add(jid)
                        self.log(f"✓ {jid}")
                    elif hard_until is not None and time.time() > hard_until:
                        postponed.add(jid)
                    elif not job.retried and self._oom(jid):
                        job.retried, job.exclusive = True, True
                        pending.append(job)
                        self.log(f"✗ {jid}: sem memória; reenfileirado para rodar sozinho")
                    else:
                        failed.add(jid)
                        self.log(f"✗ {jid} (código {rc}); veja {self.state}/{marker_name(jid)}.log")
                # 2) dependentes de jobs que falharam ficam bloqueados
                changed = True
                while changed:
                    changed = False
                    for j in list(pending):
                        if any(d in failed or d in blocked for d in j.deps):
                            pending.remove(j)
                            blocked.add(j.id)
                            changed = True
                            self.log(f"⊘ {j.id}: bloqueado por dependência que falhou")
                # 3) lançar (no máximo um por vez, para a memória do anterior aparecer no nvidia-smi)
                unfinished = set(running) | {j.id for j in pending}
                if past:     # depois do prazo, avaliações rodam com o que existir (dependências adiadas não esperam)
                    ready_ids = {j.id for j in pending if not any(d in unfinished for d in j.deps)}
                else:
                    ready_ids = {j.id for j in pending if all(d in done for d in j.deps)}
                if hard_until is not None and now > hard_until:
                    ready_ids = set()
                ready = sorted((j for j in pending if j.id in ready_ids), key=lambda j: (j.prio, -j.est_gb))
                heavy_ok = not running or now - last_launch >= self.settle      # espera a memória do anterior aparecer
                if ready:
                    reserved = sum(self.jobs[i].est_gb for i in running)
                    n_train = sum(self.jobs[i].kind == "train" for i in running)
                    any_excl = any(self.jobs[i].exclusive for i in running)
                    for j in ready:
                        if len(running) >= self.max_parallel or any_excl:
                            break
                        if j.est_gb > 0 and j.kind not in CHEAP and not heavy_ok:
                            continue                        # sem GPU (dryrun/avaliação/limpeza): não espera o intervalo
                        if j.kind == "train" and n_train >= self.max_train:
                            continue
                        if j.exclusive:
                            if running:
                                break                      # drena: nenhum job menor fura a fila de um exclusivo
                            fits = True
                        else:
                            free = self.free_fn()
                            fits = reserved + j.est_gb <= self.total and (free is None or free >= j.est_gb + self.margin)
                        if not fits:
                            if j.est_gb >= 0.5 * self.total:
                                break                      # job grande: não deixa os pequenos o adiarem para sempre
                            continue
                        pending.remove(j)
                        running[j.id] = self._start(j)
                        last_progress = time.time()
                        if j.est_gb > 0 and j.kind not in CHEAP:
                            last_launch = last_progress
                        self.log(f"▶ {j.id} (≈{j.est_gb:.0f} GB{', exclusivo' if j.exclusive else ''}) | "
                                 f"rodando={len(running)}")
                        break
                if not running and pending and time.time() - last_progress > self.stall:
                    raise RuntimeError("nenhum job coube na GPU por muito tempo (outro processo ocupando a VRAM?): "
                                       + ", ".join(j.id for j in pending[:5]))
                if hard_until is not None and now > hard_until and not running:
                    postponed.update(j.id for j in pending)
                    break
                if time.time() - last_status > 60:
                    last_status = time.time()
                    left = (hard_until - time.time()) / 3600 if hard_until else float("nan")
                    self.log(f"… rodando={sorted(running)} | pendentes={len(pending)} | concluídos={len(done)} | "
                             f"falhas={len(failed)} | VRAM livre={self.free_fn()} | prazo restante={left:.1f} h")
                time.sleep(self.poll)
        except KeyboardInterrupt:
            for proc, _, _ in running.values():
                _terminate(proc)
            self.log("interrompido; rode de novo para continuar do que falta")
        return {"done": sorted(done), "failed": sorted(failed), "blocked": sorted(blocked), "postponed": sorted(postponed)}


def eta(jobs: List[Job], done_ids, times: List[dict]) -> dict:
    """ETA grosseira: trabalho restante (duração média observada por tipo) ÷ concorrência observada."""
    done_norm = {str(d).replace("/", "_").replace("\\", "_").replace(":", "_") for d in (done_ids or ())}
    done_norm |= {str(d) for d in (done_ids or ())}
    mean: Dict[str, float] = {}
    for kind in {t["kind"] for t in times}:
        d = [t["seconds"] for t in times if t["kind"] == kind and t["rc"] == 0]
        if d:
            mean[kind] = sum(d) / len(d)
    pending = [j for j in jobs if j.id not in done_norm and marker_name(j.id) not in done_norm]
    unknown = sorted({j.kind for j in pending if j.kind not in mean and j.kind not in CHEAP})
    serial = sum(mean.get(j.kind, 0.0) for j in pending)
    conc = 1.0
    if times:
        span = max(t["ts"] for t in times) - min(t["ts"] - t["seconds"] for t in times)
        if span > 0:
            conc = max(1.0, sum(t["seconds"] for t in times) / span)
    return {"mean_s": mean, "pending": len(pending), "serial_h": serial / 3600, "concurrency": conc,
            "eta_h": serial / conc / 3600, "unknown_kinds": unknown}


def _cmd(py: str, *args) -> List[str]:
    return [py, "-m", *[str(a) for a in args]]


def _lr_tag(lr: float) -> str:
    """1e-4 → "1e-4", 1.5e-4 → "1.5e-4" (curto, sem espaços nem barras: vira parte de caminhos e ids de job)."""
    mant, exp = f"{lr:.1e}".split("e")
    return f"{mant.removesuffix('.0')}e{int(exp)}"


def build_jobs(cfg: dict) -> List[Job]:
    """Transforma o YAML de experimentos em jobs"""
    P, g = cfg["paths"], cfg.get("gpu", {})
    py, py_vllm = cfg.get("python", sys.executable), cfg.get("python_vllm", cfg.get("python", sys.executable))
    reg = load_registry()["slm"]
    env = {"HF_HUB_OFFLINE": "1", **{k: str(v) for k, v in cfg.get("env", {}).items()}}
    if "device" in g:
        env["CUDA_VISIBLE_DEVICES"] = str(g["device"])
    total = float(g.get("total_gb", 46))
    n_docs = int(cfg.get("n_docs", 100))
    sch = cfg.get("scheduling", {})
    tbase = 5 if sch.get("train_first", False) else 30       # treino é o gargalo: começa cedo, inferência ocupa o resto
    heavy = 40 if sch.get("heavy_last", False) else 0          # exclusivos (full) por último: rodam sozinhos
    common = ["--gt", P["gt"], "--docs", P["docs"], "--splits", P["splits"]]
    jobs: List[Job] = []

    def add(**kw):
        jobs.append(Job(env=dict(env), **kw))
        return kw["id"]

    inf = cfg.get("inference", {})
    engine, shards, util = inf.get("engine", "hf"), int(inf.get("shards", 1)), float(inf.get("vllm_gpu_util", 0.85))
    topos = inf.get("topologies", ["T1"])

    def infer_gb(model, eng):
        return estimate_infer_gb(reg[model]["params_b"], reg[model].get("load_4bit", True), eng, util, 48.0)

    def run_args(model, topo, shots, out, extra=()):
        return ["mpsp", "run", *common, "--model", model, "--topology", topo, "--shots", shots, "--out", out, *extra]

    # --- E0 smoke
    smoke = {}
    if inf.get("smoke", True):
        for m in inf.get("models", []):
            smoke[m] = add(id=f"smoke:{m}", kind="smoke", prio=0, est_gb=infer_gb(m, "hf"), meta={"docs": 3},
                           cmd=_cmd(py, *run_args(m, topos[0], 0, f"{P['runs']}/smoke/{m}.jsonl",
                                                  ["--fold", 0, "--limit", 3, "--batch-size", 3])))
    # --- E1/E2/E3: topologias, oracle e few-shot
    mode = inf.get("oracle", False)
    variants = [False] if not mode else ([True] if mode is True else [False, True])
    pred_files = []
    for m in inf.get("models", []):
        for t in topos:
            for sh in inf.get("shots", [0]):
                for orc in variants:
                    if orc and (t == "T1" or int(sh) != 0):
                        continue                          # oracle só faz sentido com mais de um nó e zero-shot
                    tag, sfx = f"{t}:{sh}shot" + (":oracle" if orc else ""), "_oracle" if orc else ""
                    out = f"{P['runs']}/{m}_{t}_{sh}shot{sfx}.jsonl"
                    pred_files.append(out)
                    deps = [smoke[m]] if m in smoke else []
                    extra_o = ["--oracle"] if orc else []
                    prio = 10 + int(sh) + (6 if orc else 0)
                    if engine == "vllm":
                        n = 1
                        ids = [add(id=f"infer:{m}:{tag}", kind="infer", prio=prio, deps=deps, est_gb=infer_gb(m, "vllm"),
                                   meta={"docs": n_docs, "topo": t, "shots": int(sh)},
                                   cmd=_cmd(py_vllm, *run_args(m, t, sh, out, ["--engine", "vllm", "--gpu-util", util,
                                                                              "--batch-size", 128, *extra_o])))]
                    else:
                        n = max(shards, 1)
                        ids = [add(id=f"infer:{m}:{tag}:s{i}", kind="infer", prio=prio, deps=deps, est_gb=infer_gb(m, "hf"),
                                   meta={"docs": n_docs / n, "topo": t, "shots": int(sh)},
                                   cmd=_cmd(py, *run_args(m, t, sh, out, ["--shard", f"{i}/{n}", "--batch-size", 16, *extra_o])))
                               for i in range(n)]
                    add(id=f"eval:{m}:{tag}", kind="eval", prio=90, deps=ids, est_gb=0,
                        cmd=_cmd(py, "mpsp", "eval", "--gt", P["gt"], "--docs", P["docs"], "--pred", out,
                                 "--out", f"{P['reports']}/{m}_{t}_{sh}shot{sfx}"))
    # --- E5 benchmark externo
    b = cfg.get("benchmark", {})
    if b.get("enabled"):
        for m in b.get("models", []):
            for sh in b.get("shots", [0]):
                calls = int(b.get("n", 100)) * (7 if b.get("pattern", "both") == "both" else 1)
                add(id=f"bench:{m}:{sh}shot", kind="bench", prio=15, est_gb=infer_gb(m, "hf"), meta={"calls": calls, "shots": int(sh)},
                    deps=[smoke[m]] if m in smoke else [],
                    cmd=_cmd(py, "mpsp", "bench", "--model", m, "--shots", sh, "--n", b.get("n", 100),
                             "--pattern", b.get("pattern", "both"), "--out-dir", f"{P['runs']}/bench"))
    # --- E4: ajuste fino (modelo × método × topologia × fração × fold) + inferência no fold + avaliação
    tr = cfg.get("training", {})
    L = int(tr.get("max_seq_len", 4096))
    folds = tr.get("folds", [0, 1, 2, 3, 4])
    groups = tr.get("groups") or [{"models": tr.get("models", []), "methods": tr.get("methods", [])}]
    combos = []                                                # (modelo, método, fração, lr ou None)
    for grp in groups:
        lr = grp.get("lr")                                     # controle de lr: sobrescreve o lr da última fase
        for m in grp.get("models", []):
            for method in grp.get("methods", []):
                for frac in tr.get("train_fracs", [1.0]):
                    if (m, method, frac, lr) not in combos:
                        combos.append((m, method, frac, lr))
    cv = tr.get("curve", {})                                   # E4i: curva de aprendizado
    for m in cv.get("models", []):
        for method in cv.get("methods", []):
            for frac in cv.get("fracs", []):
                if (m, method, frac, None) not in combos:
                    combos.append((m, method, frac, None))
    infer_after = tr.get("infer", True)                          # False: só treino (sem inferência/avaliação por fold)
    batch_kw = [x for k, flag in (("batch", "--batch"), ("grad_accum", "--grad-accum")) if tr.get(k) for x in (flag, tr[k])]
    dry = {}
    for (m, method, frac, lr) in combos:
        meth = METHODS[method]
        if tr.get("dry_run") and (m, method) not in dry:          # confere dados/tokenizador/memória estimada antes das horas de treino
            dry[(m, method)] = add(id=f"dryrun:{m}:{method}", kind="dryrun", prio=0, est_gb=0, meta={},
                                   cmd=_cmd(py, "mpsp.slm.train", "--method", method, "--model", m, *common, "--fold", 0,
                                            "--topology", tr.get("topologies", ["T1"])[0], "--max-seq-len", L,
                                            "--out", f"/tmp/mpsp_dry_{m}_{method}", "--dry-run"))
        for t in tr.get("topologies", ["T1"]):
            tag = method + ("" if frac >= 1 else f"_f{frac}") + (f"_lr{_lr_tag(lr)}" if lr else "")
            ckpt = f"{P['ckpts']}/{m}/{tag}/{t}"
            est = estimate_train_gb(reg[m]["params_b"], meth, L, int(tr.get("batch", 1)))
            excl = est > 0.75 * total
            off = (20 if frac < 1 else 0) + (heavy if excl else 0)
            out = f"{P['runs']}/{m}_{t}_{tag}.jsonl"
            fts = []
            for k in folds:
                tid = add(id=f"train:{m}:{tag}:{t}:fold{k}", kind="train", prio=tbase + 2 * k + off, est_gb=est, exclusive=excl,
                          deps=[dry[(m, method)]] if (m, method) in dry else [],
                          meta={"method": method, "frac": frac, "topo": t,
                                "gb_ckpt": 2.0 * reg[m]["params_b"] if meth.full_save else 0.0},
                          cmd=_cmd(py, "mpsp.slm.train", "--method", method, "--model", m, *common, "--fold", k,
                                   "--topology", t, "--train-frac", frac, "--max-seq-len", L, *batch_kw,
                                   *(["--lr", lr] if lr else []), "--out", f"{ckpt}/fold{k}"))
                if not infer_after:
                    continue
                eng = tr.get("engine", "hf") if not meth.hf_only else "hf"
                extra = ["--fold", k, "--adapter", f"{ckpt}/fold{k}", "--batch-size", 16 if eng == "hf" else 128]
                if eng == "vllm":
                    extra += ["--engine", "vllm", "--gpu-util", util]
                ft = add(id=f"ft:{m}:{tag}:{t}:fold{k}", kind="ft", prio=tbase + 1 + 2 * k + off, deps=[tid], est_gb=infer_gb(m, eng),
                         meta={"docs": n_docs / len(folds), "topo": t, "shots": 0},
                         cmd=_cmd(py_vllm if eng == "vllm" else py, *run_args(m, t, 0, out, extra)))
                fts.append(ft)
                if meth.full_save and tr.get("cleanup_full_models", True):
                    add(id=f"cleanup:{m}:{tag}:{t}:fold{k}", kind="cleanup", prio=tbase + 1 + 2 * k + off, deps=[ft], est_gb=0,
                        cmd=_cmd(py, "mpsp", "cleanup", "--dir", f"{ckpt}/fold{k}",
                                 "--keep-info", f"{P['reports']}/train_info/{m}_{tag}_{t}_fold{k}.json"))
            if not infer_after:
                continue
            pred_files.append(out)
            add(id=f"eval:{m}:{tag}:{t}", kind="eval", prio=90, deps=fts, est_gb=0,
                cmd=_cmd(py, "mpsp", "eval", "--gt", P["gt"], "--docs", P["docs"], "--pred", out,
                         "--out", f"{P['reports']}/{m}_{t}_{tag}"))
    # --- E7: corpus completo com um modelo final treinado em todo o GT (precisa do caminho do corpus)
    cp = cfg.get("corpus", {})
    if cp.get("enabled"):
        m, method, t = cp["model"], cp.get("method", "qlora"), tr.get("topologies", ["T1"])[0]
        ck = f"{P['ckpts']}/{m}/{method}/{t}/final"
        final = add(id=f"train:{m}:{method}:{t}:final", kind="train", prio=150, est_gb=estimate_train_gb(reg[m]["params_b"], METHODS[method], L),
                    meta={"method": method, "frac": 1.25, "topo": t},
                    cmd=_cmd(py, "mpsp.slm.train", "--method", method, "--model", m, *common, "--fold", -1, "--topology", t,
                             "--max-seq-len", L, "--out", ck))
        n = int(cp.get("shards", 3))
        for i in range(n):
            add(id=f"corpus:{m}:{method}:{t}:s{i}", kind="corpus", prio=160, deps=[final], est_gb=infer_gb(m, "hf"),
                meta={"docs": cp.get("n_docs", 40000) / n, "topo": t, "shots": 0},
                cmd=_cmd(py, "mpsp", "run", "--source", "corpus", "--gt", P["gt"], "--docs", cp["docs"], "--model", m,
                         "--topology", t, "--adapter", ck, "--shard", f"{i}/{n}", "--batch-size", 16,
                         "--out", f"{P['runs']}/corpus_{m}_{method}_{t}.jsonl"))
    # --- comparação final (pareada) de tudo contra a primeira execução
    if len(pred_files) > 1:
        ev = [j.id for j in jobs if j.kind == "eval"]
        add(id="report:compare", kind="report", prio=99, deps=ev, est_gb=0,
            cmd=_cmd(py, "mpsp", "compare", "--gt", P["gt"], "--pred", *pred_files, "--out", f"{P['reports']}/compare_all.txt"))
    return jobs


# [otimista, pessimista]: premissas, não medidas na A40. Substitua pelos valores de `status` em `time_model:` no YAML.
TIME_MODEL = {
    "infer_s_per_doc": [6.0, 12.0],              # T1 zero-shot, HF 4 bits, um processo (ref.: ~11 s/doc na máquina anterior)
    "topo_mult": {"T1": 1.0, "T2": 1.3, "T3": 1.6, "T4": 1.9, "T5": 2.2},
    "shot_mult": {"0": 1.0, "1": 1.3, "3": 1.8, "5": 2.2},
    "bench_s_per_call": [2.0, 5.0],
    "train_min": {                                # minutos por fold (100% dos documentos de treino)
        "qlora": [20, 35], "lora": [18, 32], "frozen": [8, 15], "two_phase": [15, 30],
        "two_phase_ner": [25, 45], "full": [60, 150]},
    "overlap_speedup": [2.0, 1.3],                # ganho das inferências em paralelo na mesma GPU
    "train_overlap_speedup": [1.25, 1.05],        # ganho de rodar 2 treinos juntos (gpu.max_train >= 2); treino é limitado por cálculo
    "train_slowdown": [1.2, 1.5],                 # lentidão do treino ao dividir a GPU
    "load_s": 120, "cheap_min": 2,
}


def _tm(cfg: dict) -> dict:
    tm = {k: (dict(v) if isinstance(v, dict) else v) for k, v in TIME_MODEL.items()}
    for k, v in (cfg.get("time_model") or {}).items():
        if isinstance(v, dict) and isinstance(tm.get(k), dict):
            tm[k].update(v)
        else:
            tm[k] = v
    return tm


def job_hours(j: Job, tm: dict, w: int) -> float:
    """Horas de um job (w=0 otimista, 1 pessimista)."""
    md = j.meta
    if j.kind in CHEAP:
        return tm["cheap_min"] / 60
    s_doc = tm["infer_s_per_doc"][w] * tm["topo_mult"].get(md.get("topo", "T1"), 1.0) * tm["shot_mult"].get(str(md.get("shots", 0)), 1.0)
    if j.kind == "dryrun":
        return tm["load_s"] / 3600
    if j.kind == "smoke":
        return (tm["load_s"] + md.get("docs", 3) * s_doc) / 3600
    if j.kind in ("infer", "ft", "corpus"):
        return (tm["load_s"] + md.get("docs", 100) * s_doc) / 3600
    if j.kind == "bench":
        return (tm["load_s"] + md.get("calls", 700) * tm["bench_s_per_call"][w] * (1.5 if md.get("shots") else 1.0)) / 3600
    if j.kind == "train":
        base = tm["train_min"][md.get("method", "qlora")][w] / 60
        return base * max(float(md.get("frac", 1.0)), 0.15)
    return 0.0


def estimate_hours(jobs: List[Job], cfg: dict) -> dict:
    """Serial e com sobreposição (treino 1 por vez + inferências juntas; exclusivos rodam sozinhos). Heurística."""
    tm = _tm(cfg)
    out = {"by_kind": {}, "serial": [0.0, 0.0], "makespan": [0.0, 0.0]}
    for w in (0, 1):
        by = {}
        for j in jobs:
            by[j.kind] = by.get(j.kind, 0.0) + job_hours(j, tm, w)
        excl = sum(job_hours(j, tm, w) for j in jobs if j.kind == "train" and j.exclusive)
        train = by.get("train", 0.0)
        infer = sum(v for k, v in by.items() if k not in CHEAP and k != "train")
        cheap = sum(v for k, v in by.items() if k in CHEAP)
        out["serial"][w] = sum(by.values())
        mt = int(cfg.get("gpu", {}).get("max_train", 1))
        slow = tm["train_slowdown"][w] if infer > 0 else 1.0                 # sem inferência dividindo a GPU, não há lentidão
        tr_speed = tm["train_overlap_speedup"][w] if mt >= 2 else 1.0
        overlapped = max((train - excl) * slow / tr_speed, infer / tm["overlap_speedup"][w] if infer else 0.0) + excl
        out["makespan"][w] = overlapped + cheap / 4
        for k, v in by.items():
            out["by_kind"].setdefault(k, [0.0, 0.0])[w] = v
    out["counts"] = {k: sum(j.kind == k for j in jobs) for k in out["by_kind"]}
    return out


def load_config(path: str) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: configuração vazia ou depreciada; use configs/experiments.yaml")
    return data


def restrict(jobs: List[Job], only: Optional[List[str]]) -> List[Job]:
    """Mantém só os tipos pedidos; dependências removidas assumem-se já cumpridas (execuções anteriores)."""
    if not only:
        return jobs
    keep = {j.id for j in jobs if j.kind in only}
    return [Job(**{**j.__dict__, "deps": [d for d in j.deps if d in keep]}) for j in jobs if j.id in keep]
