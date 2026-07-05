"""Statistics utilities for rigorous reporting (RESEARCH_PLAN §6).

All estimates come with uncertainty: bootstrap CIs for means, Wilcoxon signed-rank for
paired seed comparisons, Holm-Bonferroni for multiplicity, Cliff's delta for effect size,
and Sarle's bimodality coefficient (the 0%-100% patching results in the legacy work were
bimodal and must not be averaged away).

Only numpy is required; scipy is used for Wilcoxon if present, else a normal approximation.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def bootstrap_ci(values: Sequence[float], n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap CI for the mean."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boot = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(x.size)}


def wilcoxon_signed_rank(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test (two-sided). x, y aligned by seed."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    assert a.shape == b.shape, "paired samples must align"
    try:
        from scipy.stats import wilcoxon
        d = a - b
        if np.allclose(d, 0):
            return {"stat": 0.0, "p": 1.0, "n": int(a.size)}
        stat, p = wilcoxon(a, b)
        return {"stat": float(stat), "p": float(p), "n": int(a.size)}
    except Exception:
        # normal approximation fallback
        d = a - b
        d = d[d != 0]
        n = d.size
        if n == 0:
            return {"stat": 0.0, "p": 1.0, "n": 0}
        ranks = np.argsort(np.argsort(np.abs(d))) + 1
        w = float(np.sum(ranks[d > 0]))
        mean_w = n * (n + 1) / 4.0
        se_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        z = (w - mean_w) / se_w if se_w > 0 else 0.0
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
        return {"stat": w, "p": float(min(1.0, p)), "n": int(n)}


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> List[Dict]:
    """Holm-Bonferroni step-down. Returns per-hypothesis {p, p_adj, reject} in input order."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    p_adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        p_adj[idx] = min(1.0, running)
    return [{"p": float(p[i]), "p_adj": float(p_adj[i]), "reject": bool(p_adj[i] < alpha)}
            for i in range(m)]


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta effect size in [-1, 1]; >0 means x tends to exceed y."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = np.sum(a[:, None] > b[None, :])
    lt = np.sum(a[:, None] < b[None, :])
    return float((gt - lt) / (a.size * b.size))


def bimodality_coefficient(values: Sequence[float]) -> float:
    """Sarle's bimodality coefficient. BC > ~0.555 suggests bimodality/non-unimodality."""
    x = np.asarray(values, dtype=float)
    n = x.size
    if n < 4:
        return float("nan")
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0:
        return float("nan")
    z = (x - m) / s
    skew = np.mean(z ** 3)
    kurt = np.mean(z ** 4) - 3.0  # excess kurtosis
    return float((skew ** 2 + 1) / (kurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def paired_report(a: Sequence[float], b: Sequence[float], seed: int = 0) -> Dict:
    """Convenience: CI of each, CI of the difference, Wilcoxon p, Cliff's delta."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return {
        "a": bootstrap_ci(a, seed=seed),
        "b": bootstrap_ci(b, seed=seed),
        "diff": bootstrap_ci(diff, seed=seed),
        "wilcoxon": wilcoxon_signed_rank(a, b),
        "cliffs_delta": cliffs_delta(a, b),
    }
