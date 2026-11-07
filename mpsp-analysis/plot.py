"""
Gera os gráficos a partir de results_unified.csv.

Uso:
    python plot.py --csv results_unified.csv --outdir .
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # sem janela; salva PNG direto
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILY_ORDER = [
    "Zero-shot", "Few-shot", "Frozen", "Full",
    "QLoRA", "QLoRA (lr 1e-4)", "LoRA",
    "Two-phase", "Two-phase NER", "Two-phase NER (lr 2e-4)",
]
MODEL_ORDER = ["qwen2.5-7b", "llama3.1-8b", "falcon3-7b", "granite3.3-8b"]
COLORS = {
    "Zero-shot": "#d62728",
    "Few-shot": "#ff7f0e",
    "Frozen": "#9467bd",
    "Full": "#8c564b",
    "QLoRA": "#1f77b4",
    "QLoRA (lr 1e-4)": "#17becf",
    "LoRA": "#2ca02c",
    "Two-phase": "#7f7f7f",
    "Two-phase NER": "#bcbd22",
    "Two-phase NER (lr 2e-4)": "#e377c2",
}
MARKERS = {"qwen2.5-7b": "o", "llama3.1-8b": "s", "falcon3-7b": "^", "granite3.3-8b": "D"}


def load(path):
    df = pd.read_csv(path)
    df["family"] = pd.Categorical(df["family"], FAMILY_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    return df


def fig1_family(df, outdir):
    """F1 soft médio por família (barra = média dos 4 modelos, pontos = modelos)."""
    g = df.groupby("family", observed=True)["soft_f1"].agg(["mean", "std"])
    g = g.reindex(FAMILY_ORDER).dropna(subset=["mean"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(g))
    ax.bar(x, g["mean"], yerr=g["std"], capsize=4, alpha=0.85,
           color=[COLORS[f] for f in g.index], edgecolor="black", linewidth=0.6)
    for f in g.index:
        pts = df[df["family"] == f]
        j = list(g.index).index(f)
        ax.scatter([j] * len(pts), pts["soft_f1"], color="black", s=22, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, rotation=30, ha="right")
    ax.set_ylabel("F1 soft (micro)")
    ax.set_title("F1 soft por método (média dos 4 modelos, ±1 desvio; pontos = modelos)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_f1_por_familia.png"), dpi=160)
    plt.close(fig)


def fig2_top_ci(df, outdir, top=15):
    """Top execuções por F1/doc com IC95% (mostra o empate estatístico)."""
    d = df.sort_values("doc_f1_soft", ascending=False).head(top)
    err = np.array([(d["doc_f1_soft"] - d["ci_lo"]).values, (d["ci_hi"] - d["doc_f1_soft"]).values])
    fig, ax = plt.subplots(figsize=(9, top * 0.42))
    y = np.arange(len(d))[::-1]
    ax.errorbar(d["doc_f1_soft"], y, xerr=err, fmt="none", ecolor="gray", capsize=3, zorder=1)
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.scatter(r["doc_f1_soft"], yi, color=COLORS.get(r["family"], "gray"), s=55, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['model']} · {r['condition']}" for _, r in d.iterrows()], fontsize=8)
    ax.set_xlabel("F1/doc (soft) com IC95%")
    ax.set_title(f"Top {top} execuções — CIs sobrepostos = empate estatístico")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_top_ci.png"), dpi=160)
    plt.close(fig)


def fig3_validity(df, outdir):
    """F1 soft × taxa de saídas válidas (ok_rate): mostra o confundidor de formato."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for fam in FAMILY_ORDER:
        d = df[df["family"] == fam]
        if d.empty:
            continue
        ax.scatter(d["ok_rate"], d["soft_f1"], color=COLORS[fam], label=fam, s=70,
                   edgecolor="black", linewidth=0.4)
    for _, r in df[df["ok_rate"] < 0.7].iterrows():
        ax.annotate(f"{r['model']}\n{r['condition']}", (r["ok_rate"], r["soft_f1"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("ok_rate (fração de saídas válidas no schema)")
    ax.set_ylabel("F1 soft")
    ax.set_title("F1 × validade de formato — pontos à esquerda têm F1 baixo por falha de JSON/schema")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_f1_vs_okrate.png"), dpi=160)
    plt.close(fig)


def fig4_heatmap(df, outdir):
    """Matriz modelo × condição (heatmap de F1 soft)."""
    pivot = df.pivot_table(index="model", columns="condition", values="soft_f1", observed=True)
    pivot = pivot.reindex(index=MODEL_ORDER)
    # ordem das colunas por família
    fam_of = {c: df[df["condition"] == c]["family"].iloc[0] for c in df["condition"].unique()}
    cols = sorted(pivot.columns, key=lambda c: FAMILY_ORDER.index(fam_of[c]))
    pivot = pivot.reindex(columns=cols)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=0.0, vmax=0.8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v < 0.55 else "black")
    fig.colorbar(im, ax=ax, label="F1 soft")
    ax.set_title("F1 soft por modelo × condição (heatmap)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig4_heatmap.png"), dpi=160)
    plt.close(fig)


def fig5_efficiency(df, outdir):
    """F1 soft × tempo de inferência (s/doc)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in MODEL_ORDER:
        d = df[df["model"] == m]
        ax.scatter(d["seconds_per_doc"], d["soft_f1"], marker=MARKERS[m], s=90,
                   label=m, edgecolor="black", linewidth=0.4)
    for _, r in df[df["soft_f1"] > 0.76].iterrows():
        ax.annotate(r["condition"], (r["seconds_per_doc"], r["soft_f1"]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Tempo de inferência (s/doc)")
    ax.set_ylabel("F1 soft")
    ax.set_title("F1 × eficiência — alto e à esquerda é o melhor custo/benefício")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig5_f1_vs_tempo.png"), dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results_unified.csv")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    df = load(args.csv)
    fig1_family(df, args.outdir)
    fig2_top_ci(df, args.outdir)
    fig3_validity(df, args.outdir)
    fig4_heatmap(df, args.outdir)
    fig5_efficiency(df, args.outdir)
    print("gerados:", ", ".join(sorted(os.listdir(args.outdir))))


if __name__ == "__main__":
    main()
