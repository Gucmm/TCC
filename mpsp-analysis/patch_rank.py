import io
p = "secao_3_4.py"
s = io.open(p, encoding="utf-8").read()
reps = [
 ("        _, letter = lets[m]\n        b = best_run(runs, m)", "        _, letter, _ = lets[m]\n        b = best_run(runs, m)"),
 ("        order, letter = lets[m]\n        d = runs.set_index", "        order, letter, _ = lets[m]\n        d = runs.set_index"),
]
for a, b in reps:
    assert s.count(a) == 1, a[:40]
    s = s.replace(a, b)
old_main = """    lets = per_model_letters(runs, pares)"""
new_main = """    lets = per_model_letters(runs, pares)
    g_order, g_let = global_letters(runs, pares)"""
assert s.count(old_main) == 1
s = s.replace(old_main, new_main)
old_copy = """    if a.ranking_global:
        shutil.copy(a.ranking_global, os.path.join(figs, "fig11_ranking_global.png"))
    if a.matriz:
        shutil.copy(a.matriz, os.path.join(figs, "fig12_matriz_pareada.png"))"""
new_copy = """    fig_ranking_global(runs, g_order, g_let, os.path.join(figs, "fig11_ranking_global.png"))
    fig_matriz(runs, g_order, pares, os.path.join(figs, "fig12_matriz_pareada.png"))"""
assert s.count(old_copy) == 1
s = s.replace(old_copy, new_copy)
s = s.replace('''    ap.add_argument("--ranking-global", help="figura global ja existente (ranking com letras)")
    ap.add_argument("--matriz", help="figura existente da matriz pareada")
''', "")
matriz = '''

def fig_matriz(runs, order, pares, path):
    """Matriz pareada (Holm global): azul = linha melhor, laranja = pior, cinza = sem diferenca."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    n = len(order)
    pos = {k: i for i, k in enumerate(order)}
    M = np.zeros((n, n))
    dm = {}
    for r in pares.itertuples():
        dm[(r.a, r.b)] = r.diff_media
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
    ax.set_yticklabels([f"{i + 1}. {MODEL_LABEL[runs.set_index('run').loc[k, 'model']]} ÅE "
                        f"{COND_LABEL[runs.set_index('run').loc[k, 'condition']]}" for i, k in enumerate(order)],
                       fontsize=6.5)
    ax.set_xticks(np.arange(-.5, n), minor=True)
    ax.set_yticks(np.arange(-.5, n), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="both", length=0)
    ax.xaxis.tick_top()
    fig.legend(handles=[Patch(color="#2a78d6", label="linha melhor que coluna"),
                        Patch(color="#eb6834", label="linha pior que coluna"),
                        Patch(color="#d9d8d3", label="sem diferenca significativa (Holm, p<0,05)")],
               loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
'''
s = s.replace("\n\ndef main():", matriz + "\n\ndef main():")
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
