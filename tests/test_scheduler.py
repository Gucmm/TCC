import subprocess
import sys
import time

import pytest

from mpsp.scheduler import Job, Scheduler, build_jobs, restrict

PY = sys.executable
SLEEPER = "import sys,time; open(sys.argv[1],'w').write(str(time.time())); time.sleep(float(sys.argv[2])); open(sys.argv[1]+'.end','w').write(str(time.time()))"


def sleeper(tmp, name, secs=0.4, **kw):
    return Job(id=name, kind=kw.pop("kind", "infer"), cmd=[PY, "-c", SLEEPER, str(tmp / name), str(secs)], **kw)


def span(tmp, name):
    return float((tmp / name).read_text()), float((tmp / f"{name}.end").read_text())


def overlap(tmp, a, b):
    (s1, e1), (s2, e2) = span(tmp, a), span(tmp, b)
    return min(e1, e2) > max(s1, s2)


def sched(jobs, tmp, **kw):
    log = []
    s = Scheduler(jobs, state_dir=str(tmp / "state"), settle=0, poll=0.03, free_fn=lambda: None, log=log.append, **kw)
    return s, log


def test_respeita_a_vram_e_roda_em_paralelo_o_que_cabe(tmp_path):
    jobs = [sleeper(tmp_path, n, est_gb=20, prio=i) for i, n in enumerate("abc")]
    s, _ = sched(jobs, tmp_path, total_gb=45)
    res = s.run()
    assert res["done"] == ["a", "b", "c"] and not res["failed"]
    assert overlap(tmp_path, "a", "b")                      # 20+20 ≤ 45: juntos
    assert not (overlap(tmp_path, "a", "c") and overlap(tmp_path, "b", "c"))     # 3×20 > 45: nunca os três


def test_dependencias_e_prioridade(tmp_path):
    jobs = [sleeper(tmp_path, "d", est_gb=1, prio=1, deps=["a"]), sleeper(tmp_path, "a", est_gb=1, prio=5)]
    s, _ = sched(jobs, tmp_path, total_gb=45)
    s.run()
    assert span(tmp_path, "d")[0] >= span(tmp_path, "a")[1]


def test_falha_bloqueia_dependentes_e_nao_derruba_os_outros(tmp_path):
    bad = Job(id="f", kind="train", cmd=[PY, "-c", "import sys; sys.exit(3)"], est_gb=1)
    jobs = [bad, sleeper(tmp_path, "g", est_gb=1, deps=["f"]), sleeper(tmp_path, "ok", est_gb=1)]
    s, log = sched(jobs, tmp_path, total_gb=45)
    res = s.run()
    assert res["failed"] == ["f"] and res["blocked"] == ["g"] and "ok" in res["done"] and not (tmp_path / "g").exists()


def test_retomada_pula_o_que_ja_terminou(tmp_path):
    jobs = lambda: [sleeper(tmp_path, "a", est_gb=1), sleeper(tmp_path, "b", est_gb=1, deps=["a"])]  # noqa: E731
    sched(jobs(), tmp_path)[0].run()
    t = span(tmp_path, "a")
    res = sched(jobs(), tmp_path)[0].run()
    assert res["done"] == ["a", "b"] and span(tmp_path, "a") == t           # nada foi reexecutado


def test_exclusivo_roda_sozinho_e_nao_e_furado_por_menores(tmp_path):
    jobs = [sleeper(tmp_path, "pequeno1", est_gb=2, prio=1, secs=0.6),
            sleeper(tmp_path, "grande", est_gb=40, prio=2, exclusive=True),
            sleeper(tmp_path, "pequeno2", est_gb=2, prio=3)]
    s, _ = sched(jobs, tmp_path, total_gb=46)
    s.run()
    assert not overlap(tmp_path, "grande", "pequeno1") and not overlap(tmp_path, "grande", "pequeno2")
    assert span(tmp_path, "pequeno2")[0] >= span(tmp_path, "grande")[1]      # pequeno2 esperou a fila do exclusivo


def test_no_maximo_um_treino_por_vez(tmp_path):
    jobs = [sleeper(tmp_path, f"t{i}", kind="train", est_gb=5, prio=i) for i in range(3)] + [sleeper(tmp_path, "i", est_gb=5, prio=9)]
    s, _ = sched(jobs, tmp_path, total_gb=45, max_train=1)
    s.run()
    assert not any(overlap(tmp_path, a, b) for a, b in (("t0", "t1"), ("t0", "t2"), ("t1", "t2")))
    assert overlap(tmp_path, "t0", "i") or overlap(tmp_path, "t1", "i") or overlap(tmp_path, "t2", "i")   # inferência divide a GPU


def test_sem_memoria_reenfileira_sozinho_uma_vez(tmp_path):
    flag = tmp_path / "flag"
    code = f"import os,sys; p={str(flag)!r}\nif not os.path.exists(p):\n open(p,'w'); print('CUDA out of memory'); sys.exit(1)"
    jobs = [Job(id="oom", kind="train", cmd=[PY, "-c", code], est_gb=5), sleeper(tmp_path, "x", est_gb=5)]
    s, log = sched(jobs, tmp_path, total_gb=45)
    res = s.run()
    assert "oom" in res["done"] and not res["failed"] and any("sem memória" in l for l in log)
    assert not overlap(tmp_path, "oom", "x") if (tmp_path / "oom").exists() else True


def test_respeita_a_memoria_livre_real(tmp_path):
    jobs = [sleeper(tmp_path, "a", est_gb=20)]
    s = Scheduler(jobs, state_dir=str(tmp_path / "state"), settle=0, poll=0.02, stall_s=0.4, free_fn=lambda: 5.0, log=lambda *_: None)
    with pytest.raises(RuntimeError, match="nenhum job coube"):
        s.run()                                                          # outro processo ocupa a GPU: não dispara às cegas
    assert not (tmp_path / "a").exists()


def test_dependencia_inexistente_e_erro():
    with pytest.raises(ValueError, match="não existe"):
        Scheduler([Job(id="a", kind="x", cmd=["true"], deps=["fantasma"])], state_dir="/tmp/_s")


CFG = {
    "paths": {"gt": "gt.json", "docs": "docs", "splits": "s.json", "runs": "runs", "ckpts": "ck", "reports": "rep"},
    "gpu": {"total_gb": 46, "device": 0},
    "inference": {"models": ["qwen2.5-7b", "falcon3-7b"], "topologies": ["T1"], "shots": [0, 3], "shards": 2},
    "training": {"models": ["qwen2.5-7b"], "topologies": ["T1"], "methods": ["qlora", "full", "two_phase"],
                 "folds": [0, 1]},
}


def by_kind(jobs):
    out = {}
    for j in jobs:
        out.setdefault(j.kind, []).append(j)
    return out


def test_plano_gera_os_jobs_esperados():
    jobs = build_jobs(CFG)
    k = by_kind(jobs)
    assert len(k["smoke"]) == 2
    assert len(k["infer"]) == 2 * 2 * 2                      # 2 modelos × 2 shots × 2 shards
    assert len(k["train"]) == 3 * 2                          # 3 métodos × 2 folds
    assert len(k["ft"]) == len(k["train"])
    assert len(k["cleanup"]) == 2                            # só o `full` salva modelo completo
    assert len(k["report"]) == 1
    ids = {j.id for j in jobs}
    assert all(d in ids for j in jobs for d in j.deps) and len(ids) == len(jobs)


def test_plano_deps_exclusividade_e_comandos():
    jobs = {j.id: j for j in build_jobs(CFG)}
    assert jobs["train:qwen2.5-7b:full:T1:fold0"].exclusive and not jobs["train:qwen2.5-7b:qlora:T1:fold0"].exclusive
    ft = jobs["ft:qwen2.5-7b:two_phase:T1:fold0"]
    assert ft.deps == ["train:qwen2.5-7b:two_phase:T1:fold0"] and "--engine" not in ft.cmd        # hf_only: sem vLLM
    assert jobs["eval:qwen2.5-7b:T1:0shot"].deps == ["infer:qwen2.5-7b:T1:0shot:s0", "infer:qwen2.5-7b:T1:0shot:s1"]
    assert jobs["train:qwen2.5-7b:qlora:T1:fold0"].prio < jobs["train:qwen2.5-7b:qlora:T1:fold1"].prio   # fold 0 primeiro
    assert jobs["smoke:qwen2.5-7b"].prio == 0 and jobs["infer:qwen2.5-7b:T1:0shot:s0"].deps == ["smoke:qwen2.5-7b"]
    assert jobs["train:qwen2.5-7b:qlora:T1:fold0"].env["CUDA_VISIBLE_DEVICES"] == "0"
    assert jobs["train:qwen2.5-7b:qlora:T1:fold0"].env["HF_HUB_OFFLINE"] == "1"


def test_plano_vllm_usa_um_job_e_restrict_remove_deps():
    cfg = {**CFG, "inference": {**CFG["inference"], "engine": "vllm", "vllm_gpu_util": 0.8}}
    k = by_kind(build_jobs(cfg))
    assert len(k["infer"]) == 2 * 2 and all("--engine" in j.cmd and abs(j.est_gb - 0.8 * 48) < 1e-6 for j in k["infer"])
    jobs = restrict(build_jobs(CFG), ["train"])
    assert {j.kind for j in jobs} == {"train"} and all(d.startswith("train:") for j in jobs for d in j.deps)


def test_cli_plan_e_dry_run(tmp_path):
    import yaml
    (tmp_path / "e.yaml").write_text(yaml.safe_dump(CFG))
    out = subprocess.run([PY, "-m", "mpsp", "plan", "--config", str(tmp_path / "e.yaml")], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout
    assert "jobs" in out and "exclusivos" in out
    out2 = subprocess.run([PY, "-m", "mpsp", "schedule", "--dry-run", "--only", "train", "--config", str(tmp_path / "e.yaml")],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout
    assert "train:qwen2.5-7b:full:T1:fold0" in out2 and "infer:" not in out2


def test_dois_treinos_sobrepostos_quando_max_train_2_e_cabem(tmp_path):
    jobs = [sleeper(tmp_path, f"t{i}", secs=0.6, kind="train", est_gb=15, prio=i) for i in range(3)]
    s, _ = sched(jobs, tmp_path, total_gb=40, max_train=2)
    s.run()
    assert overlap(tmp_path, "t0", "t1") and not overlap(tmp_path, "t0", "t2") or overlap(tmp_path, "t1", "t2")
    assert not (overlap(tmp_path, "t0", "t1") and overlap(tmp_path, "t0", "t2") and overlap(tmp_path, "t1", "t2"))   # nunca os 3


def test_dois_treinos_nao_se_sobrepoem_se_a_vram_nao_cabe(tmp_path):
    jobs = [sleeper(tmp_path, f"t{i}", secs=0.4, kind="train", est_gb=27, prio=i) for i in range(2)]
    s, _ = sched(jobs, tmp_path, total_gb=46, max_train=2)
    s.run()
    assert not overlap(tmp_path, "t0", "t1")                        # 27 + 27 > 46: sequenciais mesmo com max_train=2
