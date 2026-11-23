"""Concordance decomposition of pooled AUROC and downstream metrics.

Conventions: y = 1 means the answer is INCORRECT (positive class); a detector score s is a
risk score (higher = more likely incorrect). Ties count 1/2.

For items grouped by question q, every (incorrect i, correct j) pair is either a
within-question pair (q_i = q_j) or a between-question pair. With U the number of concordant
pairs (ties 1/2) and N the number of pairs,

    AUROC_pooled = pi_W * AUROC_W + (1 - pi_W) * AUROC_B,   pi_W = N_W / N.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _u_stat(s: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Mann-Whitney U (ties 1/2) and number of pairs for positives y==1 vs negatives y==0."""
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return 0.0, 0.0
    r = rankdata(s)
    u = r[y == 1].sum() - npos * (npos + 1) / 2.0
    return float(u), float(npos * nneg)


def _group_stats(s: np.ndarray, y: np.ndarray, g: np.ndarray):
    """Vectorised per-question Mann-Whitney U and pair counts.

    Returns (u_q, n_q, inv) over the sorted unique question ids; n_q = 0 for unmixed questions."""
    _, inv = np.unique(g, return_inverse=True)
    G = inv.max() + 1
    r = pd.Series(s).groupby(inv).rank(method="average").to_numpy()
    yf = y.astype(float)
    npos = np.bincount(inv, weights=yf, minlength=G)
    cnt = np.bincount(inv, minlength=G).astype(float)
    u = np.bincount(inv, weights=r * yf, minlength=G) - npos * (npos + 1) / 2.0
    n = npos * (cnt - npos)
    return u, n, inv


def _summarise(u, n, uw_q, nw_q) -> dict:
    mixed = nw_q > 0
    uw, nw = uw_q[mixed].sum(), nw_q[mixed].sum()
    return {
        "auroc": u / n if n else np.nan,
        "auroc_w": uw / nw if nw else np.nan,
        "auroc_w_macro": float(np.mean(uw_q[mixed] / nw_q[mixed])) if mixed.any() else np.nan,
        "auroc_b": (u - uw) / (n - nw) if n - nw > 0 else np.nan,
        "pi_w": nw / n if n else np.nan,
        "n_mixed": int(mixed.sum()),
    }


def decompose(s: np.ndarray, y: np.ndarray, g: np.ndarray) -> dict:
    """Pooled, within-question and between-question AUROC.

    s, y, g: 1-D arrays (score, label, question id). Returns dict with
    auroc, auroc_w (pair-weighted), auroc_w_macro (mean over mixed questions), auroc_b,
    pi_w, n_mixed.
    """
    s = np.asarray(s, float)
    y = np.asarray(y, int)
    g = np.asarray(g)
    u, n = _u_stat(s, y)
    uq, nq, _ = _group_stats(s, y, g)
    return _summarise(u, n, uq, nq)


def bootstrap_decompose(s, y, g, n_boot: int = 300, seed: int = 0) -> list[dict]:
    """Cluster bootstrap of decompose(), equal to bootstrap(decompose, ...) with the same seed.

    A resampled copy of a question keeps its own within-question U and pair count, so these are
    computed once; only the pooled U is recomputed per resample."""
    s = np.asarray(s, float)
    y = np.asarray(y, int)
    uq, nq, inv = _group_stats(s, y, np.asarray(g))
    G = len(uq)
    order = np.argsort(inv, kind="stable")
    cnt = np.bincount(inv, minlength=G)
    start = np.r_[0, np.cumsum(cnt)[:-1]]
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(n_boot):
        pick = rng.integers(0, G, G)
        lens = cnt[pick]
        offs = np.repeat(start[pick] - np.cumsum(lens) + lens, lens) + np.arange(lens.sum())
        idx = order[offs]
        u, n = _u_stat(s[idx], y[idx])
        res.append(_summarise(u, n, uq[pick], nq[pick]))
    return res


def best_of_n(s: np.ndarray, correct: np.ndarray, g: np.ndarray) -> dict:
    """Pick the minimum-risk answer per question; ties broken uniformly at random (expected).

    Returns selection accuracy, random-pick accuracy, oracle accuracy, computed over all
    questions and over mixed questions.
    """
    s = np.asarray(s, float)
    c = np.asarray(correct, float)
    g = np.asarray(g)
    order = np.argsort(g, kind="stable")
    gs, ss, cs = g[order], s[order], c[order]
    bounds = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    sel, rnd, orc, mixed = [], [], [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        sq, cq = ss[a:b], cs[a:b]
        m = sq <= sq.min() + 1e-12
        sel.append(cq[m].mean())
        rnd.append(cq.mean())
        orc.append(cq.max())
        mixed.append(0 < cq.sum() < len(cq))
    sel, rnd, orc, mixed = map(np.array, (sel, rnd, orc, mixed))
    return {
        "sel_acc": sel.mean(), "rand_acc": rnd.mean(), "oracle_acc": orc.mean(),
        "gain": (sel - rnd).mean(),
        "sel_acc_mixed": sel[mixed].mean() if mixed.any() else np.nan,
        "rand_acc_mixed": rnd[mixed].mean() if mixed.any() else np.nan,
        "gain_mixed": (sel - rnd)[mixed].mean() if mixed.any() else np.nan,
    }


def aurc(s: np.ndarray, y: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better); ties broken by averaging."""
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    n = len(s)
    # sort by increasing risk; accept lowest-risk first
    r = rankdata(s, method="average")
    order = np.argsort(r, kind="stable")
    ys = y[order]
    # average y within tied blocks so the result does not depend on tie order
    rs = r[order]
    blocks = np.flatnonzero(np.r_[True, rs[1:] != rs[:-1], True])
    for a, b in zip(blocks[:-1], blocks[1:]):
        ys[a:b] = ys[a:b].mean()
    risks = np.cumsum(ys) / np.arange(1, n + 1)
    return float(risks.mean())


def bootstrap(fn, groups: np.ndarray, n_boot: int = 500, seed: int = 0, **arrays):
    """Cluster bootstrap over questions. fn(**arrays_resampled, g=new_group_ids) -> dict."""
    rng = np.random.default_rng(seed)
    uq, inv = np.unique(groups, return_inverse=True)
    idx_by_q = [np.flatnonzero(inv == k) for k in range(len(uq))]
    res = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uq), len(uq))
        idx = np.concatenate([idx_by_q[k] for k in pick])
        newg = np.concatenate([np.full(len(idx_by_q[k]), j) for j, k in enumerate(pick)])
        res.append(fn(**{k: v[idx] for k, v in arrays.items()}, g=newg))
    return res
