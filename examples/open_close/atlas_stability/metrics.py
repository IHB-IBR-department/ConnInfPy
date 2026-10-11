"""EC−EO edge effects, common-unit summaries and participant bootstrap metrics."""
import hashlib
import numpy as np
from scipy import stats

SEED, MIN_SUPPORT = 42, 1.0

def fisher_z(ts: np.ndarray) -> np.ndarray:
    out = np.empty((ts.shape[0], ts.shape[2], ts.shape[2]))
    for s, x in enumerate(ts):
        c = np.clip(np.corrcoef(x.T), -0.9999, 0.9999)
        np.fill_diagonal(c, 0.0)
        out[s] = np.arctanh(c)
    return out

def _digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        if a is not None:
            h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()[:16]

def unit_pairs(units):
    U = len(units)
    a, b = np.triu_indices(U)
    return a, b, [f"{units[i]} | {units[j]}" for i, j in zip(a, b)]

def unit_summaries(D: np.ndarray, M: np.ndarray, base: np.ndarray):
    """Participant-level weighted mean delta-z per unit pair, plus baseline split and heterogeneity.

    D: (S, N, N) EC−EO delta-z (zero diagonal); M: (N, U) memberships; base:
    (N, N) full-sample EO mean z. An edge (i, j) has weight m_i[a] m_j[b] + m_i[b] m_j[a]
    for a != b and m_i[a] m_j[a] for a = b.
    """
    S, N, _ = D.shape
    U = M.shape[1]
    off = 1.0 - np.eye(N)
    W = M.T @ off @ M                                       # summed weights (both orientations)
    num = np.matmul(M.T[None], np.matmul(D, M))              # (S, U, U)
    a, b, _ = unit_pairs([str(k) for k in range(U)])
    support = W[a, b] / np.where(a == b, 2.0, 1.0)          # edge-equivalents
    with np.errstate(invalid="ignore", divide="ignore"):
        E = num[:, a, b] / W[a, b]
    iu = np.triu_indices(N, 1)
    De = D[:, iu[0], iu[1]]                                 # (S, E)
    be = base[iu]
    dbar, se2 = De.mean(0), De.var(0, ddof=1) / S
    strong, weak, tau = np.full((S, len(a)), np.nan), np.full((S, len(a)), np.nan), np.full(len(a), np.nan)
    for k, (x, y) in enumerate(zip(a, b)):
        w = M[iu[0], x] * M[iu[1], y] + (M[iu[0], y] * M[iu[1], x] if x != y else 0.0)
        on = w > 0
        if w[on].sum() < MIN_SUPPORT:
            continue
        wk, bk = w[on], be[on]
        order = np.argsort(bk)
        cw = np.cumsum(wk[order]) / wk.sum()
        med = bk[order][np.searchsorted(cw, 0.5)]
        hi = bk >= med
        for out, sel in ((strong, hi), (weak, ~hi)):
            if wk[sel].sum() > 0:
                out[:, k] = De[:, on][:, sel] @ wk[sel] / wk[sel].sum()
        m = np.average(dbar[on], weights=wk)
        tau[k] = np.sqrt(max(0.0, np.average((dbar[on] - m) ** 2, weights=wk) - np.average(se2[on], weights=wk)))
    return {"E": E, "support": support, "strong": strong, "weak": weak, "tau": tau}

def draws(n_subjects: int, n_boot: int) -> np.ndarray:
    """(B, S) bootstrap count matrix (seed 42); identical for every branch of a cohort."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_subjects, size=(n_boot, n_subjects))
    C = np.zeros((n_boot, n_subjects))
    np.add.at(C, (np.repeat(np.arange(n_boot), n_subjects), idx.ravel()), 1.0)
    return C / n_subjects

def agreement(x: np.ndarray, y: np.ndarray) -> dict:
    """Agreement of two unit-effect vectors (rows = draws allowed)."""
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    mx, my = x.mean(1, keepdims=True), y.mean(1, keepdims=True)
    vx, vy = x.var(1), y.var(1)
    cov = ((x - mx) * (y - my)).mean(1)
    rx, ry = stats.rankdata(x, axis=1), stats.rankdata(y, axis=1)
    return {"ccc": 2 * cov / (vx + vy + (mx - my)[:, 0] ** 2), "pearson": cov / np.sqrt(vx * vy),
            "spearman": np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(rx, ry)]),
            "sign_agreement": (np.sign(x) == np.sign(y)).mean(1)}

def pct(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v.size else (np.nan, np.nan)
