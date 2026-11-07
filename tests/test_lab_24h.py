import json
import subprocess
import sys
import time

import yaml

from mpsp.scheduler import Job, Scheduler, build_jobs, eta, marker_name

PY = sys.executable
SLEEPER = "import sys,time; open(sys.argv[1],'w').write(str(time.time())); time.sleep(float(sys.argv[2]))"


def job(tmp, name, secs=0.7, **kw):
    return Job(id=name, kind=kw.pop("kind", "infer"), cmd=[PY, "-c", SLEEPER, str(tmp / name), str(secs)], **kw)


def sched(jobs, tmp, **kw):
    log = []
    kw = {"settle": 0, "poll": 0.03, **kw}
    return Scheduler(jobs, state_dir=str(tmp / "state"), free_fn=lambda: None, log=log.append, **kw), log


def test_prazo_para_de_lancar_adia_o_resto_e_ainda_avalia(tmp_path):
    jobs = [job(tmp_path, f"j{i}", prio=i, est_gb=1) for i in range(6)]
    jobs.append(Job(id="ev", kind="eval", cmd=[PY, "-c", f"open({str(tmp_path / 'ev')!r}, 'w').write('ok')"],
                    deps=[j.id for j in jobs], prio=99))
    s, log = sched(jobs, tmp_path, max_parallel=1, total_gb=45, budget_h=0.0006, tail_h=0.0003)   # lança até ~1,1 s
    res = s.run()
    assert 1 <= len(res["done"]) - 1 < 6 and res["postponed"]               # alguns rodaram, o resto foi adiado
    assert "ev" in res["done"] and (tmp_path / "ev").exists()                # a avaliação roda com o que existe
    assert any("prazo de lançamento" in l for l in log)
    assert set(res["postponed"]).isdisjoint(res["failed"])


def test_prazo_duro_encerra_job_longo_sem_marcar_falha(tmp_path):
    jobs = [job(tmp_path, "longo", secs=60, est_gb=1)]
    s, _ = sched(jobs, tmp_path, budget_h=0.001, tail_h=0.0)                 # 3,6 s
    t0 = time.time()
    res = s.run()
    assert time.time() - t0 < 20 and res["postponed"] == ["longo"] and not res["failed"] and not res["done"]


def test_grava_duracoes_por_job(tmp_path):
    jobs = [job(tmp_path, "a", secs=0.2, est_gb=1, kind="train"), job(tmp_path, "b", secs=0.2, est_gb=1)]
    sched(jobs, tmp_path, total_gb=45)[0].run()
    rows = [json.loads(l) for l in (tmp_path / "state" / "times.jsonl").read_text().splitlines()]
    assert {r["id"] for r in rows} == {"a", "b"} and all(r["rc"] == 0 and r["seconds"] >= 0.2 for r in rows)
    assert {r["kind"] for r in rows} == {"train", "infer"}


def test_avaliacao_nao_espera_o_intervalo_entre_lancamentos(tmp_path):
    jobs = [job(tmp_path, "pesado", secs=1.0, est_gb=5, prio=1),
            Job(id="ev", kind="eval", est_gb=0, prio=2, cmd=[PY, "-c", f"open({str(tmp_path / 'ev')!r}, 'w').write(str(__import__('time').time()))"])]
    sched(jobs, tmp_path, total_gb=45, settle=30)[0].run()
    assert float((tmp_path / "ev").read_text()) - float((tmp_path / "pesado").read_text()) < 3      # settle=30 s não valeu


def test_eta_grosseira():
    jobs = [Job(id=f"t{i}", kind="train", cmd=[]) for i in range(6)] + [Job(id="i0", kind="infer", cmd=[])]
    times = [{"id": "t0", "kind": "train", "rc": 0, "seconds": 600, "ts": 1000},
             {"id": "t1", "kind": "train", "rc": 0, "seconds": 600, "ts": 1000}]          # duas rodaram juntas
    e = eta(jobs, {"t0", "t1"}, times)
    assert e["mean_s"] == {"train": 600} and e["pending"] == 5 and e["unknown_kinds"] == ["infer"]
    assert abs(e["serial_h"] - 4 * 600 / 3600) < 1e-9 and abs(e["concurrency"] - 2.0) < 1e-9
    assert abs(e["eta_h"] - e["serial_h"] / 2) < 1e-9


CFG = {"paths": {"gt": "g.json", "docs": "d", "splits": "s.json", "runs": "r", "ckpts": "c", "reports": "rep"},
       "scheduling": {"train_first": True},
       "inference": {"models": ["qwen2.5-7b"], "shots": [0]},
       "training": {"models": ["qwen2.5-7b"], "methods": ["qlora", "frozen"], "folds": [0, 1]}}


def test_treino_primeiro_e_limpeza_logo_depois_da_inferencia_do_fold():
    jobs = {j.id: j for j in build_jobs(CFG)}
    tr0, ft0 = jobs["train:qwen2.5-7b:qlora:T1:fold0"], jobs["ft:qwen2.5-7b:qlora:T1:fold0"]
    assert jobs["smoke:qwen2.5-7b"].prio < tr0.prio < ft0.prio < jobs["infer:qwen2.5-7b:T1:0shot:s0"].prio
    assert tr0.prio < jobs["train:qwen2.5-7b:qlora:T1:fold1"].prio                  # largura primeiro: fold 0 de todos antes do 1
    cl = jobs["cleanup:qwen2.5-7b:frozen:T1:fold0"]
    assert cl.prio == jobs["ft:qwen2.5-7b:frozen:T1:fold0"].prio                    # libera disco logo, sem esperar o fim
    cfg = yaml.safe_load(open("configs/experiments.yaml"))
    plan = build_jobs(cfg)
    assert {j.meta["method"] for j in plan if j.exclusive} == {"full"}              # só o `full` roda sozinho na GPU
    assert max(j.est_gb for j in plan if not j.exclusive) < 46                        # o resto coexiste com a inferência


def test_cli_leaderboard_status_e_eval_tolerante(dataset, tmp_path):
    rep = tmp_path / "reports"
    for name, f1 in (("b_run", 0.5), ("a_run", 0.7)):
        (rep / name).mkdir(parents=True)
        (rep / name / "summary.json").write_text(json.dumps({
            "n_docs": 12, "strict": {"f1": f1 - 0.1}, "soft": {"f1": f1, "doc_f1_ci95": [f1 - 0.05, f1 + 0.05]},
            "run": {"ok_rate": 1.0, "context_overflow_rate": 0.0, "seconds": 3.2}}))
    out = subprocess.run([PY, "-m", "mpsp", "leaderboard", "--reports", str(rep)], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout
    assert out.index("a_run") < out.index("b_run") and "0.7" in out                    # ordenado por F1 soft

    state = tmp_path / "state"; (state / "done").mkdir(parents=True)
    (state / "done" / marker_name("smoke:qwen2.5-7b")).write_text("smoke:qwen2.5-7b", encoding="utf-8")
    (state / "times.jsonl").write_text(json.dumps({"id": "smoke:qwen2.5-7b", "kind": "smoke", "rc": 0, "seconds": 90, "ts": 2000}) + "\n")
    cfg_st = tmp_path / "cfg.yaml"
    cfg_st.write_text(yaml.safe_dump({**CFG, "inference": {**CFG["inference"], "models": ["qwen2.5-7b", "falcon3-7b"]}}))
    st = subprocess.run([PY, "-m", "mpsp", "status", "--config", str(cfg_st), "--state", str(state)],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout
    assert "smoke" in st and "1/2" in st and "ETA grosseira" in st

    ev = subprocess.run([PY, "-m", "mpsp", "eval", "--gt", dataset["gt_path"], "--pred", str(tmp_path / "nada.jsonl")],
                        capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert ev.returncode == 0 and "sem predições" in ev.stdout                          # job de avaliação não quebra o plano
    cp = subprocess.run([PY, "-m", "mpsp", "compare", "--gt", dataset["gt_path"], "--pred", str(tmp_path / "x.jsonl"), str(tmp_path / "y.jsonl")],
                        capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert cp.returncode == 0 and "menos de 2" in cp.stdout
