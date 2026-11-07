"""Intervalos de confiança e testes pareados sobre F1 por documento."""
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import stats


def bootstrap_ci(x: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 5000, seed: int = 0) -> Dict[str, float]:
    """Diferença média a−b com IC95% e p bilateral (mesmos documentos nos dois vetores)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"diff": float(d.mean()), "lo": float(np.quantile(boot, 0.025)),
            "hi": float(np.quantile(boot, 0.975)), "p": float(min(p, 1.0))}


def wilcoxon_p(a: Sequence[float], b: Sequence[float]) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(d).pvalue)


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Correção de Holm–Bonferroni para múltiplas comparações."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
