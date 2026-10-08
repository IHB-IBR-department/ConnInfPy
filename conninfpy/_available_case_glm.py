"""Available-case GLM inference (Freedman-Lane) for connectivity with missing values.

Each edge is fitted with the subjects who have it. Edges are grouped by their
missingness pattern (the set of subjects present); every pattern gets its own
design rows, full- and reduced-model fits and residual degrees of freedom.

Permutations stay coherent across patterns: each permutation draws one
global (within-site, when strata are given) permutation of the subjects, the
same draw as the complete-data path, and every pattern uses its restriction
to the subjects it has — within each block, the global order with the absent
subjects removed. Removing fixed subjects from a uniformly random permutation
leaves a uniformly random permutation of the rest, so each pattern sees a
valid Freedman-Lane shuffle, and on complete data this is exactly the
standard permutation. Freedman-Lane is approximate (exchangeable errors,
estimated nuisance), with or without missing values.

An edge is tested when at least ``min_coverage`` of the subjects have it, its
pattern keeps at least 2 residual degrees of freedom, and the contrast is
estimable on the pattern's design rows (e.g. a site whose subjects all lack
the edge drops out of the site dummies without affecting the interest
contrast). Other edges get p = NaN and leave the family.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import numpy.typing as npt
from scipy import stats

from . import _degenerate
from ._engine import (
    _collect_results_to_arrays,
    _compute_p_values_from_null,
    _encode_strata,
    _extract_max_stats,
    _is_worker_process,
    _stratified_perm,
    get_available_cores,
)
from ._prepare import nonfinite_report
from .glm_stats import contrast_is_estimable
from ._progress import run_permutations

DEFAULT_MIN_COVERAGE = 0.8
MIN_DF = 2


@dataclass
class _Pattern:
    """Edges sharing one set of present subjects, with their fitted design."""

    edges: npt.NDArray[np.intp]            # indices into the upper-triangle edge list
    rows: npt.NDArray[np.intp]             # present subjects (sorted)
    X: npt.NDArray[np.float64]
    pinv: npt.NDArray[np.float64]
    df: int
    cXc: Optional[float] = None            # t: c (X'X)^+ c'
    M_inv: Optional[npt.NDArray[np.float64]] = None   # F: [C (X'X)^+ C']^-1
    Y_hat0: Optional[npt.NDArray[np.float64]] = None  # reduced-model fit (n_p, E_p)
    R0: Optional[npt.NDArray[np.float64]] = None      # reduced-model residuals
    blocks: List[tuple] = field(default_factory=list)  # (block positions, block positions present, local index of present)


def _stats(Yp: npt.NDArray[np.float64], pat: _Pattern, contrast, stat_type: str):
    """Signed t / beta, or F, for the edges of one pattern."""
    beta = pat.pinv @ Yp
    if stat_type == "beta":
        return contrast @ beta
    resid = Yp - pat.X @ beta
    sigma2 = np.sum(resid ** 2, axis=0) / pat.df
    floor = _degenerate.RTOL * np.mean(Yp ** 2, axis=0)   # zero variance -> 0 (see _degenerate)
    if stat_type == "fstat":
        C = contrast if contrast.ndim == 2 else contrast[None, :]
        Cb = C @ beta
        quad = np.einsum("ie,ij,je->e", Cb, pat.M_inv, Cb)
        with np.errstate(divide="ignore", invalid="ignore"):
            F = quad / (C.shape[0] * sigma2)
        return np.where((sigma2 > floor) & np.isfinite(F), np.maximum(F, 0.0), 0.0)
    t = _degenerate.t_ratio(contrast @ beta, pat.cXc * sigma2, pat.cXc * floor)
    return np.where(np.isfinite(t), t, 0.0)


def _to_stat_dict(vec: npt.NDArray[np.float64], N: int, triu, stat_type: str):
    mat = np.zeros((N, N))
    mat[triu] = vec
    mat = mat + mat.T
    if stat_type == "fstat":
        return {"omnibus": mat}
    return {"positive": np.where(mat > 0, mat, 0.0), "negative": np.where(mat < 0, -mat, 0.0)}


def _restricted(perm_idx: npt.NDArray[np.intp], pat: _Pattern) -> npt.NDArray[np.intp]:
    """Local permutation of the pattern's subjects induced by a global one.

    ``perm_idx[k]`` is the subject whose residual lands in row ``k``. Within
    each block, the global sequence with the absent subjects removed is
    assigned, in order, to the present rows of that block.
    """
    local = np.empty(pat.rows.size, dtype=np.intp)
    for positions, present_positions, loc in pat.blocks:
        sources = perm_idx[positions]
        keep = sources[np.isin(sources, pat.rows)]
        local[loc[present_positions]] = loc[keep]
    return local


def _vector_stats(Y_by_pattern, patterns, contrast, stat_type, n_edges):
    out = np.zeros(n_edges)
    for Yp, pat in zip(Y_by_pattern, patterns):
        out[pat.edges] = _stats(Yp, pat, contrast, stat_type)
    return out


def _task(patterns, contrast, stat_type, N, triu, n_edges, strata_codes, n_subjects,
          scheme, enhance_func, enhance_kwargs, seed):
    rng = np.random.RandomState(seed)
    if scheme == "sign-flip":
        signs = rng.choice([1.0, -1.0], n_subjects)
        Y_perm = [pat.Y_hat0 + pat.R0 * signs[pat.rows][:, None] for pat in patterns]
    else:
        perm_idx = rng.permutation(n_subjects) if strata_codes is None else _stratified_perm(strata_codes, rng)
        Y_perm = [pat.Y_hat0 + pat.R0[_restricted(perm_idx, pat)] for pat in patterns]
    stat = _to_stat_dict(_vector_stats(Y_perm, patterns, contrast, stat_type, n_edges), N, triu, stat_type)
    if enhance_func is not None:
        stat = enhance_func(stat, **enhance_kwargs)
    return _extract_max_stats(stat, (N, N))


def available_case_glm(
    Y: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    contrast: npt.NDArray[np.float64],
    X_reduced: npt.NDArray[np.float64],
    stat_type: str,
    enhance_func: Optional[Callable],
    enhance_kwargs: Dict[str, Any],
    *,
    n_permutations: int,
    random_state: Optional[int],
    use_mp: bool,
    n_processes: Optional[int],
    verbose: bool,
    strata: Optional[npt.NDArray[Any]],
    acceleration: Optional[str],
    two_tailed: bool,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    scheme: str = "permutation",
    block_inference: Optional[Dict[str, Any]] = None,
):
    """p-values, raw statistic maps and the coverage report (GLM, missing values).

    ``scheme`` is the resolved resampling scheme ('permutation' or
    'sign-flip'); sign flips are drawn per subject and restricted to each
    pattern's subjects. ``block_inference`` (``net_labels``, ``correction``)
    switches to cNBS block inference over the tested edges; the block table
    is returned in the report under ``'block_report'``.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError(f"min_coverage must be in (0, 1]; got {min_coverage}.")
    n, N, _ = Y.shape
    if not np.allclose(Y, Y.transpose(0, 2, 1), equal_nan=True):
        raise NotImplementedError(
            "available-case GLM supports symmetric (undirected) connectivity only."
        )
    triu = np.triu_indices(N, k=1)
    E = Y[:, triu[0], triu[1]]
    present = np.isfinite(E)
    n_edges = E.shape[1]
    contrast = np.asarray(contrast, dtype=np.float64)
    C = contrast if contrast.ndim == 2 else contrast[None, :]

    strata_codes = None
    if strata is not None:
        strata_codes = _encode_strata(strata)
        if strata_codes.shape[0] != n:
            raise ValueError(f"strata length {strata_codes.shape[0]} does not match n_subjects={n}.")
    block_ids = strata_codes if strata_codes is not None else np.zeros(n, dtype=np.int_)
    block_positions = [np.flatnonzero(block_ids == b) for b in np.unique(block_ids)]

    need = max(int(np.ceil(min_coverage * n)), MIN_DF + 1)
    covered = present.sum(0) >= need
    reasons = {"coverage": int((~covered).sum()), "degrees_of_freedom": 0, "not_estimable": 0}

    # Group covered edges by missingness pattern
    packed = np.packbits(present, axis=0)
    keys = [packed[:, j].tobytes() for j in range(n_edges)]
    groups: Dict[bytes, List[int]] = {}
    for j in np.flatnonzero(covered):
        groups.setdefault(keys[j], []).append(j)

    patterns: List[_Pattern] = []
    tested = np.zeros(n_edges, dtype=bool)
    for edges in groups.values():
        edges = np.asarray(edges, dtype=np.intp)
        rows = np.flatnonzero(present[:, edges[0]])
        Xp = X[rows]
        rank = int(np.linalg.matrix_rank(Xp))
        df = rows.size - rank
        if df < MIN_DF:
            reasons["degrees_of_freedom"] += edges.size
            continue
        pinv = np.linalg.pinv(Xp)
        if not contrast_is_estimable(C, Xp, pinv):
            reasons["not_estimable"] += edges.size
            continue
        XtX = pinv @ pinv.T
        pat = _Pattern(edges=edges, rows=rows, X=Xp, pinv=pinv, df=df)
        if stat_type == "fstat":
            pat.M_inv = np.linalg.inv(C @ XtX @ C.T)
        else:
            pat.cXc = float(contrast @ XtX @ contrast)
        Yp = E[np.ix_(rows, edges)]
        Xr = X_reduced[rows]
        pat.Y_hat0 = (np.zeros_like(Yp) if Xr.shape[1] == 0 or not np.any(Xr)
                      else Xr @ (np.linalg.pinv(Xr) @ Yp))
        pat.R0 = Yp - pat.Y_hat0
        loc = np.full(n, -1, dtype=np.intp)
        loc[rows] = np.arange(rows.size)
        in_rows = np.zeros(n, dtype=bool)
        in_rows[rows] = True
        pat.blocks = [(pos, pos[in_rows[pos]], loc) for pos in block_positions]
        patterns.append(pat)
        tested[edges] = True
    if not tested.any():
        raise ValueError(
            "no edge can be tested: too few subjects per edge "
            f"(min_coverage={min_coverage}), too few residual degrees of freedom, "
            "or a contrast that is not estimable on the available subjects."
        )

    # An exact fit with a nonzero contrast has an infinite t (or F)
    infinite = np.zeros((N, N), dtype=bool)
    for pat in patterns:
        hit = _degenerate.glm_infinite(E[np.ix_(pat.rows, pat.edges)], pat.X, contrast, stat_type)
        infinite[triu[0][pat.edges[hit]], triu[1][pat.edges[hit]]] = True
    _degenerate.raise_if_infinite(infinite, "F" if stat_type == "fstat" else "t")

    observed_vec = _vector_stats([E[np.ix_(p.rows, p.edges)] for p in patterns], patterns,
                                 contrast, stat_type, n_edges)
    raw = _to_stat_dict(observed_vec, N, triu, stat_type)
    tested_mat = np.zeros((N, N), dtype=bool)
    tested_mat[triu] = tested
    tested_mat = tested_mat | tested_mat.T
    if block_inference is not None:
        enhance_kwargs = {**enhance_kwargs, "edge_mask": tested_mat}
    emp = raw if enhance_func is None else enhance_func(raw, **enhance_kwargs)

    seeds = np.random.RandomState(random_state).randint(0, 2**32 - 1, size=n_permutations, dtype=np.int64)
    _use_mp = use_mp and not _is_worker_process()
    if _use_mp and n_processes is None:
        n_processes = get_available_cores()
    if _use_mp and n_processes is not None:
        n_processes = min(n_processes, n_permutations)
    task = partial(_task, patterns, contrast, stat_type, N, triu, n_edges, strata_codes, n,
                   scheme, enhance_func, enhance_kwargs)
    results = run_permutations(task, list(seeds), use_mp=_use_mp, n_processes=n_processes,
                               verbose=verbose, desc="available_case_glm")
    null = _collect_results_to_arrays(results, n_permutations)
    if two_tailed and stat_type != "fstat":
        joint = np.maximum(null["positive"], null["negative"])
        null = {"positive": joint, "negative": joint}
    block_report = None
    if block_inference is not None:
        from ._cnbs import block_p_values
        p_values, block_report = block_p_values(
            emp, null, block_inference["net_labels"], N,
            correction=block_inference["correction"], edge_mask=tested_mat,
            acceleration=acceleration,
        )
    elif acceleration is not None:
        from .acceleration import compute_p_values_accelerated
        p_values = compute_p_values_accelerated(emp, null, method=acceleration)
    else:
        p_values = _compute_p_values_from_null(emp, null)
    out_i, out_j = triu[0][~tested], triu[1][~tested]
    for key in p_values:
        p = np.array(p_values[key], dtype=np.float64)
        p[out_i, out_j, ...] = np.nan
        p[out_j, out_i, ...] = np.nan
        p_values[key] = p

    report = _report(Y, X, C, present, tested, triu, N, min_coverage, reasons)
    if block_report is not None:
        report["block_report"] = block_report
    return p_values, raw, report


def _report(Y, X, C, present, tested, triu, N, min_coverage, reasons) -> Dict[str, Any]:
    edge_n = present.sum(0)
    n_mat = np.zeros((N, N))
    n_mat[triu] = edge_n
    n_mat = n_mat + n_mat.T
    r = nonfinite_report(Y)
    missing = np.zeros(Y.shape[0], dtype=bool)
    missing[r["subjects"]] = True
    rep: Dict[str, Any] = {
        "policy": "available_case",
        "min_coverage": min_coverage,
        "n_edges_tested": int(tested.sum()),
        "n_edges_untested": int((~tested).sum()),
        "untested_edges": [(int(i), int(j)) for i, j in zip(triu[0][~tested], triu[1][~tested])],
        "untested_reasons": reasons,
        "edge_n": n_mat,
        "edge_n_min": int(edge_n[tested].min()),
        "edge_n_median": float(np.median(edge_n[tested])),
        "subjects_with_missing": {"all": r["subjects"]},
        "regions_without_signal": dict(sorted(r["rois"].items())),
        "missing_by_group_p": None,
        "missing_by_interest_p": None,
    }
    # Is missingness related to the tested variable? (single-column contrasts)
    nz = np.flatnonzero(np.any(C != 0, axis=0))
    if nz.size == 1 and missing.any() and (~missing).any():
        x = X[:, nz[0]]
        values = np.unique(x)
        if values.size == 2:
            a = [[int(np.sum(missing & (x == v))), int(np.sum(~missing & (x == v)))] for v in values]
            rep["missing_by_interest_p"] = float(stats.fisher_exact(a)[1])
        else:
            rep["missing_by_interest_p"] = float(stats.mannwhitneyu(x[missing], x[~missing]).pvalue)
    return rep
