"""Plano único `configs/experiments.yaml`: escopo efetivamente executado (ver reports/).

Cobre apenas o que foi rodado:
  - topologia T1
  - zero-shot (k=0) e few-shot (k=1,3)
  - métodos qlora, lora, full, frozen, two_phase, two_phase_ner
  - controles de lr: qlora lr=1e-4 e two_phase_ner lr=2e-4

Não cobre T2-T5, oracle (E2), benchmark LeNER-Br (E5), corpus completo (E7),
curva de aprendizado (E4i) nem múltiplas sementes, pois esses experimentos não foram executados.
"""
import os
import subprocess
import sys
from pathlib import Path

from mpsp.scheduler import Scheduler, build_jobs, estimate_hours, load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "experiments.yaml"
ALL = load_config(str(CFG))

MODELS = ["qwen2.5-7b", "llama3.1-8b", "falcon3-7b", "granite3.3-8b"]
METHODS = {"qlora", "lora", "full", "frozen", "two_phase", "two_phase_ner"}


def by_kind(jobs):
    out = {}
    for j in jobs:
        out.setdefault(j.kind, []).append(j)
    return out


def test_config_unica_e_escopo_executado():
    assert CFG.exists(), "configs/experiments.yaml deve ser a fonte única"
    assert ALL["n_docs"] == 95
    jobs = build_jobs(ALL)
    k = by_kind(jobs)

    # 4 modelos × 3 valores de k-shot × 2 shards
    assert len(k["infer"]) == len(MODELS) * 3 * 2
    assert {j.meta["shots"] for j in k["infer"]} == {0, 1, 3}
    assert {j.meta["topo"] for j in jobs if j.kind in ("infer", "ft")} == {"T1"}
    assert len(k.get("smoke", [])) == 0          # smoke desligado no plano executado
    assert not any(j.kind == "bench" for j in jobs)   # E5 não executado
    assert not any(j.kind == "corpus" for j in jobs)  # E7 não executado


def test_metodos_e_controles_de_lr_do_escopo():
    jobs = build_jobs(ALL)
    train = [j for j in jobs if j.kind == "train"]
    assert {j.meta["method"] for j in train} == METHODS

    ids = {j.id for j in train}
    for m in MODELS:
        for fold in range(5):
            assert f"train:{m}:qlora_lr1e-4:T1:fold{fold}" in ids
            assert f"train:{m}:two_phase_ner_lr2e-4:T1:fold{fold}" in ids


def test_lr_so_na_ultima_fase_e_sem_colisao_de_id():
    jobs = {j.id: j for j in build_jobs(ALL)}
    q = jobs["train:qwen2.5-7b:qlora_lr1e-4:T1:fold0"]
    n = jobs["train:qwen2.5-7b:two_phase_ner_lr2e-4:T1:fold3"]
    assert q.cmd[q.cmd.index("--lr") + 1] == "0.0001"
    assert n.cmd[n.cmd.index("--lr") + 1] == "0.0002"
    assert "ckpts/qwen2.5-7b/qlora_lr1e-4/T1/fold0" in q.cmd
    assert not any("--lr" in j.cmd for j in jobs.values() if j.kind == "train" and "_lr" not in j.id)


def test_grafo_sem_ciclos_e_dependencias_existem():
    jobs = {j.id: j for j in build_jobs(ALL)}
    assert len(jobs) == len(build_jobs(ALL))          # ids únicos
    for j in jobs.values():
        assert all(d in jobs for d in j.deps)

    seen, stack = set(), set()

    def dfs(i):
        assert i not in stack, f"ciclo em {i}"
        if i in seen:
            return
        stack.add(i)
        for d in jobs[i].deps:
            dfs(d)
        stack.discard(i)
        seen.add(i)

    for i in jobs:
        dfs(i)


def test_full_roda_sozinho():
    heavy = [j for j in build_jobs(ALL) if j.exclusive]
    assert heavy and {j.meta["method"] for j in heavy} == {"full"}


def test_estimativa_de_tempo():
    jobs = build_jobs(ALL)
    e = estimate_hours(jobs, ALL)
    assert e["serial"][0] < e["serial"][1]
    assert e["makespan"][0] <= e["serial"][0]
    assert e["counts"]["train"] == len([j for j in jobs if j.kind == "train"])


def test_cli_plan_estimate():
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    out = subprocess.run([sys.executable, "-m", "mpsp", "plan", "--config", str(CFG), "--estimate"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace",
                         check=True, cwd=ROOT, env=env).stdout
    assert "ESTIMATIVA DE TEMPO" in out and "premissas" in out


def test_config_padrao_da_cli_existe():
    from mpsp.cli import DEFAULT_CONFIG
    assert Path(DEFAULT_CONFIG).name == "experiments.yaml"
    assert (ROOT / "configs" / "experiments.yaml").exists()
