"""
Seção 3.4 da monografia: recalcula métricas das 44 execuções a partir dos summary.json,
gera tabelas (CSV) e figuras da seção.

Entradas (pasta de relatórios com um diretório por execução):
  reports/<modelo>_T1_<condição>/summary.json   métricas por execução
  reports/significancia/pares.csv               testes pareados (Wilcoxon + Holm) das 946 comparações
  per_field_top5.csv                            F1 soft por campo (5 melhores execuções)

Uso:
  python secao_3_4.py --reports <pasta reports> --fields per_field_top5.csv --out figs/secao_3_4
"""

import argparse
import itertools
import json
import os
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["qwen2.5-7b", "llama3.1-8b", "falcon3-7b", "granite3.3-8b"]
MODEL_LABEL = {"qwen2.5-7b": "Qwen2.5-7B", "llama3.1-8b": "Llama-3.1-8B",
               "falcon3-7b": "Falcon3-7B", "granite3.3-8b": "Granite-3.3-8B"}
COND_LABEL = {"0shot": "zero-shot", "1shot": "1-shot", "3shot": "3-shot", "qlora": "QLoRA",
              "qlora_lr1e-4": "QLoRA lr1e-4", "lora": "LoRA", "full": "full", "frozen": "frozen",
              "two_phase": "two-phase", "two_phase_ner": "two-phase NER",
              "two_phase_ner_lr2e-4": "two-phase NER lr2e-4"}
FAMILY = {"0shot": "Zero-shot", "1shot": "Few-shot", "3shot": "Few-shot", "qlora": "QLoRA",
          "qlora_lr1e-4": "QLoRA lr1e-4", "lora": "LoRA", "full": "Full", "frozen": "Frozen",
          "two_phase": "Two-phase", "two_phase_ner": "Two-phase NER",
          "two_phase_ner_lr2e-4": "Two-phase NER lr2e-4"}
FAMILY_ORDER = ["Zero-shot", "Few-shot", "Frozen", "Full", "QLoRA", "QLoRA lr1e-4", "LoRA",
                "Two-phase", "Two-phase NER", "Two-phase NER lr2e-4"]
BAR_COLOR = "#2a78d6"  # cor única para as barras dos dez melhores/piores
COLORS = {"Zero-shot": "#d62728", "Few-shot": "#ff7f0e", "Frozen": "#9467bd", "Full": "#8c564b",
          "QLoRA": "#1f77b4", "QLoRA lr1e-4": "#17becf", "LoRA": "#2ca02c", "Two-phase": "#7f7f7f",
          "Two-phase NER": "#bcbd22", "Two-phase NER lr2e-4": "#e377c2"}
MODEL_COLOR = {"qwen2.5-7b": "#1f77b4", "llama3.1-8b": "#ff7f0e", "falcon3-7b": "#2ca02c",
               "granite3.3-8b": "#d62728"}
MODEL_MARKER = {"qwen2.5-7b": "o", "llama3.1-8b": "s", "falcon3-7b": "^", "granite3.3-8b": "D"}
SHOTS = {"0shot": 0, "1shot": 1, "3shot": 3}


def num(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x):
    return f"{x * 100:.1f}".replace(".", ",") + "%"


def load_runs(reports):
    rows = []
    for d in sorted(os.listdir(reports)):
        f = os.path.join(reports, d, "summary.json")
        m = re.match(r"(" + "|".join(MODELS) + r")_T1_(.+)$", d)
        if not os.path.isfile(f) or not m:
            continue
        s = json.load(open(f, encoding="utf-8"))
        model, cond = m.group(1), m.group(2)
        rows.append({
            "model": model, "condition": cond, "family": FAMILY[cond],
            "strict_f1": s["strict"]["f1"], "soft_f1": s["soft"]["f1"],
            "doc_f1_soft": s["soft"]["doc_f1_mean"],
            "ci_lo": s["soft"]["doc_f1_ci95"][0], "ci_hi": s["soft"]["doc_f1_ci95"][1],
            "ok_rate": s["run"]["ok_rate"], "seconds": s["run"]["seconds"],
            "run": f"{model}_T1_{cond}",
        })
    df = pd.DataFrame(rows)
    assert len(df) == 44, f"esperava 44 execuções, achei {len(df)}"
    return df


def letters(order, sig):
    """Compact letter display (Piepho): quem divide uma letra não difere significativamente."""
    groups = [set(order)]
    for a, b in itertools.combinations(order, 2):
        if not sig.get(frozenset((a, b)), False):
            continue
        new = []
        for g in groups:
            if a in g and b in g:
                new += [g - {b}, g - {a}]
            else:
                new.append(g)
        groups = [g for g in new if g and not any(g < h for h in new)]
        groups = [g for i, g in enumerate(groups) if g not in groups[:i]]
    pos = {n: i for i, n in enumerate(order)}
    groups.sort(key=lambda g: min(pos[x] for x in g))
    alpha = "abcdefghijklmnopqrstuvwxyz"
    return {n: "".join(alpha[i] for i, g in enumerate(groups) if n in g) for n in order}


def holm(pvals):
    """Holm–Bonferroni (mesma regra de mpsp/stats.holm)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def per_model_letters(runs, pares):
    """Letras por modelo: Wilcoxon pareado (p bruto de pares.csv) corrigido por Holm
    dentro de cada modelo (55 comparações). Ordem = F1/doc médio."""
    out = {}
    for m in MODELS:
        d = runs[runs.model == m].sort_values("doc_f1_soft", ascending=False)
        order = list(d.run)
        sub = pares[pares.a.isin(order) & pares.b.isin(order)]
        ph = holm({f"{r.a}|||{r.b}": r.p_wilcoxon for r in sub.itertuples()})
        sig = {frozenset((r.a, r.b)): ph[f"{r.a}|||{r.b}"] < 0.05 for r in sub.itertuples()}
        out[m] = (order, letters(order, sig), ph)
    return out


def global_letters(runs, pares):
    """Letras do ranking global (44 execuções): Holm sobre as 946 comparações (p_holm de pares.csv)."""
    order = list(runs.sort_values("doc_f1_soft", ascending=False).run)
    sig = {frozenset((r.a, r.b)): bool(r.significativo) for r in pares.itertuples()}
    return order, letters(order, sig)


def fig_ranking_global(runs, order, let, path):
    d = runs.set_index("run").loc[order]
    n = len(d)
    fig, ax = plt.subplots(figsize=(9.5, 0.27 * n + 1.8))
    y = np.arange(n)[::-1]
    for yi, (name, r) in zip(y, d.iterrows()):
        c = COLORS[r.family]
        ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax.plot(r.doc_f1_soft, yi, "o", ms=5.5, color=c, mec="white", mew=0.8)
        ax.text(1.0, yi, "  " + let[name], transform=ax.get_yaxis_transform(), va="center",
                fontsize=8.5, fontweight="bold", color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i + 1:>2}. {MODEL_LABEL[r.model]} · {COND_LABEL[r.condition]}  "
                        f"(F1/doc {num(r.doc_f1_soft)})" for i, (_, r) in enumerate(d.iterrows())], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1/doc soft (média e IC95% por bootstrap)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle("Ranking global das 44 execuções (ordem = F1/doc médio)\n"
                 "Letras iguais = sem diferença significativa (Wilcoxon pareado + Holm sobre 946 pares, p<0,05)",
                 x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def table_leaderboard(runs, worst):
    d = runs.sort_values("soft_f1", ascending=worst).head(10)
    rows = [["Execução", "F1 strict", "F1 soft", "F1/doc (IC95%)", "Saídas válidas", "Tempo/doc (s)"]]
    for _, r in d.iterrows():
        rows.append([f"{MODEL_LABEL[r.model]} · {COND_LABEL[r.condition]}", num(r.strict_f1),
                     num(r.soft_f1), f"{num(r.doc_f1_soft)} [{num(r.ci_lo)}; {num(r.ci_hi)}]",
                     pct(r.ok_rate), num(r.seconds, 1)])
    return rows


def table_fewshot(runs):
    rows = [["Modelo", "0 exemplos", "1 exemplo", "3 exemplos"]]
    for m in MODELS:
        row = [MODEL_LABEL[m]]
        for c in SHOTS:
            r = runs[(runs.model == m) & (runs.condition == c)].iloc[0]
            row.append(f"{num(r.soft_f1)} / {pct(r.ok_rate)} / {num(r.seconds, 0)} s")
        rows.append(row)
    return rows


def best_run(runs, m):
    """Melhor condição do modelo por F1 soft (mesma métrica das tabelas de melhores execuções)."""
    return runs[runs.model == m].sort_values("soft_f1", ascending=False).iloc[0]


def table_best(runs, lets):
    rows = [["Modelo", "Melhor condição", "F1 soft", "Condições com a mesma letra (sem diferença significativa)"]]
    for m in MODELS:
        _, letter, _ = lets[m]
        b = best_run(runs, m)
        bl = letter[b.run]
        same = [COND_LABEL[runs[runs.run == r].condition.iloc[0]] for r in letter
                if r != b.run and set(bl) & set(letter[r])]
        rows.append([MODEL_LABEL[m], COND_LABEL[b.condition], num(b.soft_f1), ", ".join(same)])
    return rows


def table_method(runs):
    rows = [["Método", "F1 médio", "Mediana", "Desvio", "Saídas válidas (média)"]]
    for c in ["two_phase_ner", "qlora", "two_phase", "qlora_lr1e-4", "two_phase_ner_lr2e-4", "lora",
              "full", "frozen", "1shot", "3shot", "0shot"]:
        d = runs[runs.condition == c]
        name = COND_LABEL[c] if c not in ("1shot", "3shot", "0shot") else (
            "zero-shot" if c == "0shot" else "few-shot " + COND_LABEL[c].replace("-shot", "") + " exemplo(s)")
        if c in ("1shot", "3shot"):
            name = "few-shot, " + ("1 exemplo" if c == "1shot" else "3 exemplos")
        rows.append([name, num(d.soft_f1.mean()), num(d.soft_f1.median()),
                     num(d.soft_f1.std()), pct(d.ok_rate.mean())])
    return rows


def table_fields(fields):
    fd = fields.sort_values("mean_f1", ascending=False)
    rows = [["Dez melhores campos", "F1", "Dez piores campos", "F1"]]
    for (_, b), (_, w) in zip(fd.head(10).iterrows(), fd.tail(10).iterrows()):
        rows.append([b.field, num(b.mean_f1), w.field, num(w.mean_f1)])
    return rows


def table_lr(runs):
    rows = [["Modelo", "QLoRA → QLoRA lr1e-4", "two-phase NER → lr2e-4"]]
    for m in MODELS:
        g = lambda c: runs[(runs.model == m) & (runs.condition == c)].soft_f1.iloc[0]
        rows.append([MODEL_LABEL[m], f"{num(g('qlora'))} → {num(g('qlora_lr1e-4'))}",
                     f"{num(g('two_phase_ner'))} → {num(g('two_phase_ner_lr2e-4'))}"])
    return rows


def save_csv(rows, path):
    pd.DataFrame(rows[1:], columns=rows[0]).to_csv(path, index=False, encoding="utf-8-sig")


def style(ax):
    ax.grid(axis="x", alpha=0.3, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_leaderboard(runs, worst, path, title):
    d = runs.sort_values("soft_f1", ascending=worst).head(10).sort_values("soft_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    y = np.arange(len(d))
    ax.barh(y, d.soft_f1, color=BAR_COLOR, edgecolor="black", linewidth=0.5)
    for yi, v in zip(y, d.soft_f1):
        ax.text(v + 0.005, yi, num(v), va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{MODEL_LABEL[r.model]} · {COND_LABEL[r.condition]}" for _, r in d.iterrows()],
                       fontsize=8.5)
    ax.set_xlabel("F1 soft")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_method(runs, path):
    g = runs.groupby("family")["soft_f1"].agg(["mean", "median"]).reindex(FAMILY_ORDER).dropna()
    x = np.arange(len(g))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, g["mean"], w, label="Média", color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, g["median"], w, label="Mediana", color="#ff7f0e", edgecolor="black", linewidth=0.5)
    for xi, f in enumerate(g.index):
        vals = runs[runs.family == f].soft_f1
        ax.scatter([xi] * len(vals), vals, color="black", s=14, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, rotation=30, ha="right")
    ax.set_ylabel("F1 soft")
    ax.set_ylim(0, 1)
    ax.set_title("F1 soft por método: média e mediana dos quatro modelos (pontos = execuções)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_heatmap(runs, path):
    pv = runs.pivot_table(index="model", columns="condition", values="soft_f1").reindex(MODELS)
    order = ["two_phase_ner", "qlora", "two_phase", "qlora_lr1e-4", "two_phase_ner_lr2e-4", "lora",
             "full", "frozen", "0shot", "1shot", "3shot"]
    pv = pv[order]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    im = ax.imshow(pv.values, cmap="viridis", aspect="auto", vmin=0, vmax=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([COND_LABEL[c] for c in order], rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_LABEL[m] for m in MODELS], fontsize=9)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            ax.text(j, i, num(v, 2), ha="center", va="center", fontsize=8.5,
                    color="white" if v < 0.55 else "black")
    fig.colorbar(im, ax=ax, label="F1 soft")
    ax.set_title("F1 soft por modelo × condição")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_fewshot(runs, path):
    sub = runs[runs.condition.isin(SHOTS)].copy()
    sub["n"] = sub.condition.map(SHOTS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, col, lab in [(axes[0], "soft_f1", "F1 soft"), (axes[1], "ok_rate", "Saídas válidas")]:
        for m in MODELS:
            d = sub[sub.model == m].sort_values("n")
            ax.plot(d.n, d[col], marker=MODEL_MARKER[m], color=MODEL_COLOR[m], lw=2, ms=7,
                    label=MODEL_LABEL[m])
        ax.set_xticks([0, 1, 3])
        ax.set_xticklabels(["0 exemplos", "1 exemplo", "3 exemplos"])
        ax.set_ylabel(lab)
        ax.set_title(f"{lab} × número de exemplos")
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_best(runs, lets, path):
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    models = MODELS[::-1]
    vals, labels = [], []
    for m in models:
        b = best_run(runs, m)
        vals.append(b.soft_f1)
        labels.append(f"{MODEL_LABEL[m]} — {COND_LABEL[b.condition]}")
    y = np.arange(len(models))
    ax.barh(y, vals, color=[MODEL_COLOR[m] for m in models], edgecolor="black", linewidth=0.5)
    for yi, v in zip(y, vals):
        ax.text(v + 0.005, yi, num(v), va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("F1 soft da melhor condição")
    ax.set_title("Melhor condição por modelo")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_regua(runs, lets, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.6))
    for ax, m in zip(axes.ravel(), MODELS):
        order, letter, _ = lets[m]
        d = runs.set_index("run").loc[order]
        y = np.arange(len(d))[::-1]
        for yi, (name, r) in zip(y, d.iterrows()):
            c = COLORS[r.family]
            ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=c, lw=2.2, solid_capstyle="round")
            ax.plot(r.doc_f1_soft, yi, "o", ms=5.5, color=c, mec="white", mew=0.8)
            ax.text(1.0, yi, letter[name], transform=ax.get_yaxis_transform(), va="center",
                    fontsize=9, fontweight="bold", color="#333")
        ax.set_yticks(y)
        ax.set_yticklabels([COND_LABEL[c] for c in d.condition], fontsize=8.5)
        ax.set_title(MODEL_LABEL[m], fontsize=10.5)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.supxlabel("F1/doc soft (média e IC95% por bootstrap); letras iguais = sem diferença significativa "
                  "(Wilcoxon pareado + Holm dentro de cada modelo, p<0,05)", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_method_validity(runs, path):
    g = runs.groupby("family").agg(f1=("soft_f1", "mean"), ok=("ok_rate", "mean")).reindex(FAMILY_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for f in g.index:
        ax.scatter(g.loc[f, "ok"], g.loc[f, "f1"], color=COLORS[f], s=120, edgecolor="black", lw=0.5, zorder=3)
        ax.annotate(f, (g.loc[f, "ok"], g.loc[f, "f1"]), fontsize=8, xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("Saídas válidas (média dos quatro modelos)")
    ax.set_ylabel("F1 soft médio")
    ax.set_xlim(0.45, 1.0)
    ax.set_title("F1 médio × saídas válidas por método")
    style(ax)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_validity_scatter(runs, path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for f in FAMILY_ORDER:
        d = runs[runs.family == f]
        if len(d):
            ax.scatter(d.ok_rate, d.soft_f1, color=COLORS[f], label=f, s=55, edgecolor="black", lw=0.4)
    for _, r in runs[runs.ok_rate < 0.7].iterrows():
        ax.annotate(f"{MODEL_LABEL[r.model]}\n{COND_LABEL[r.condition]}", (r.ok_rate, r.soft_f1),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Saídas válidas (fração no esquema)")
    ax.set_ylabel("F1 soft")
    ax.set_title("F1 soft × saídas válidas por execução")
    ax.legend(fontsize=7.5, ncol=2, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_fields(fields, path):
    fd = fields.sort_values("mean_f1", ascending=False)
    best = fd.head(10).sort_values("mean_f1")
    worst = fd.tail(10).sort_values("mean_f1")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, sub, title, color in [(axes[0], best, "Dez melhores campos", "#2ca02c"),
                                  (axes[1], worst, "Dez piores campos", "#d62728")]:
        ax.barh(sub.field, sub.mean_f1, color=color, alpha=0.85, edgecolor="black", lw=0.5)
        for yi, v in enumerate(sub.mean_f1):
            ax.text(v + 0.01, yi, num(v), va="center", fontsize=8)
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("F1 soft médio (5 melhores execuções)")
        ax.set_title(title)
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_matriz(runs, order, pares, path):
    """Matriz pareada (Holm global): azul = linha melhor, laranja = pior, cinza = sem diferença."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    n = len(order)
    pos = {k: i for i, k in enumerate(order)}
    idx = runs.set_index("run")
    M = np.zeros((n, n))
    for r in pares.itertuples():
        if not r.significativo:
            continue
        i, j = pos[r.a], pos[r.b]
        M[i, j], M[j, i] = np.sign(r.diff_media), -np.sign(r.diff_media)
    fig, ax = plt.subplots(figsize=(8.6, 8.6))
    ax.imshow(M, cmap=ListedColormap(["#eb6834", "#d9d8d3", "#2a78d6"]), vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(i + 1) for i in range(n)], fontsize=6.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{i + 1}. {MODEL_LABEL[idx.loc[k, 'model']]} · {COND_LABEL[idx.loc[k, 'condition']]}"
                        for i, k in enumerate(order)], fontsize=6.5)
    ax.set_xticks(np.arange(-.5, n), minor=True)
    ax.set_yticks(np.arange(-.5, n), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="both", length=0)
    ax.xaxis.tick_top()
    fig.legend(handles=[Patch(color="#2a78d6", label="linha melhor que coluna"),
                        Patch(color="#eb6834", label="linha pior que coluna"),
                        Patch(color="#d9d8d3", label="sem diferença significativa (Holm, p<0,05)")],
               loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True)
    ap.add_argument("--fields", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    figs = os.path.join(a.out, "figuras")
    tabs = os.path.join(a.out, "tabelas")
    os.makedirs(figs, exist_ok=True)
    os.makedirs(tabs, exist_ok=True)

    runs = load_runs(a.reports)
    pares = pd.read_csv(os.path.join(a.reports, "significancia", "pares.csv"))
    fields = pd.read_csv(a.fields)
    lets = per_model_letters(runs, pares)
    g_order, g_let = global_letters(runs, pares)
    print("execuções:", len(runs), "| pares:", len(pares), "| significativos:", int(pares.significativo.sum()))

    tables = {
        "tab03_dez_melhores": table_leaderboard(runs, worst=False),
        "tab04_dez_piores": table_leaderboard(runs, worst=True),
        "tab05_zero_few_shot": table_fewshot(runs),
        "tab06_melhor_condicao": table_best(runs, lets),
        "tab07_metodos": table_method(runs),
        "tab08_campos": table_fields(fields),
        "tab09_taxa_aprendizado": table_lr(runs),
    }
    for name, rows in tables.items():
        save_csv(rows, os.path.join(tabs, name + ".csv"))

    fig_leaderboard(runs, False, os.path.join(figs, "fig01_dez_melhores.png"),
                    "Dez melhores execuções por F1 soft")
    fig_leaderboard(runs, True, os.path.join(figs, "fig02_dez_piores.png"),
                    "Dez piores execuções por F1 soft")
    fig_method(runs, os.path.join(figs, "fig03_metodo_media_mediana.png"))
    fig_heatmap(runs, os.path.join(figs, "fig04_heatmap.png"))
    fig_fewshot(runs, os.path.join(figs, "fig05_fewshot.png"))
    fig_best(runs, lets, os.path.join(figs, "fig06_melhor_condicao.png"))
    fig_regua(runs, lets, os.path.join(figs, "fig07_regua_modelo.png"))
    fig_method_validity(runs, os.path.join(figs, "fig08_metodo_validade.png"))
    fig_validity_scatter(runs, os.path.join(figs, "fig09_f1_validade.png"))
    fig_fields(fields, os.path.join(figs, "fig10_campos.png"))
    fig_ranking_global(runs, g_order, g_let, os.path.join(figs, "fig11_ranking_global.png"))
    fig_matriz(runs, g_order, pares, os.path.join(figs, "fig12_matriz_pareada.png"))

    runs.to_csv(os.path.join(a.out, "execucoes_44.csv"), index=False, encoding="utf-8-sig")
    print("figuras em", figs)
    print("tabelas em", tabs)


if __name__ == "__main__":
    main()
