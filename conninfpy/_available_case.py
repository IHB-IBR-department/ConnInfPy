"""Available-case inference for pairwise designs (two-sample, paired, one-sample).

Missing connectivity values (NaN) are common in small fMRI cohorts — e.g. a
region outside the field of view in one subject. Dropping that subject loses
a scarce observation; filling in zeros invents connectivity. Here each
edge's t-statistic uses the subjects who have that edge:

- two-sample (Welch): per-edge group sizes ``n1(e)``, ``n2(e)``;
- paired: the pairs complete for that edge; one-sample: the subjects present.

Permutations shuffle (or sign-flip) whole subjects, so a subject's missing
pattern travels with them and every edge sees the same permutation. Observed
and permuted maps are computed by the same function with the same masks, so
the max-statistic test stays exact under exchangeability given the
missingness (which must not depend on the connectivity values themselves).

Edges kept by fewer than ``min_coverage`` of the subjects of a group (or of
the pairs), or by fewer than three, are not tested: their p-value is NaN and
they are listed in the report. They enter neither the max-statistic family
nor the parametric multiplicity count.

The permutation draws are the same as in the complete-data fast path, so on
complete data this path reproduces its p-values.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import stats

from . import _degenerate
from ._engine import (
    _bh_fdr_correction,
    _collect_results_to_arrays,
    _compute_p_values_from_null,
    _encode_strata,
    _extract_max_stats,
    _is_worker_process,
    _stratified_choice_n1,
    get_available_cores,
)
from ._prepare import nonfinite_report
from ._progress import run_permutations

DEFAULT_MIN_COVERAGE = 0.8
MIN_PER_GROUP = 3


@dataclass
class _Design:
    """Masked edge data for one analysis (edges = upper triangle)."""

    kind: str                          # 'two-sample' or 'one-sample' (paired = one-sample on diffs)
    X0: npt.NDArray[np.float64]        # (n_subjects, n_edges), missing -> 0
    X02: npt.NDArray[np.float64]       # X0 ** 2
    M: npt.NDArray[np.float64]         # (n_subjects, n_edges), 1 = present
    n1: int                            # subjects in group 1 (two-sample)
    tested: npt.NDArray[np.bool_]      # (n_edges,)
    triu: Tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]
    N: int
    strata: Optional[npt.NDArray[np.int_]] = None
    n1_per_stratum: Optional[npt.NDArray[np.int_]] = None


# ----------------------------------------------------------------------
# t-statistics from masked sums (per-edge n)
# ----------------------------------------------------------------------
def _t_one(sum_, sumsq, n):
    """One-sample t with per-edge n; undefined edges (n < 2, zero variance) -> 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sum_ / n
        var = np.maximum((sumsq - sum_ ** 2 / n) / (n - 1), 0.0)
        t = _degenerate.t_ratio(mean, var / n, _degenerate.RTOL * sumsq / n ** 2)
    t = np.where((n >= 2) & np.isfinite(t), t, 0.0)
    return np.maximum(t, 0.0), np.maximum(-t, 0.0)


def _group_moments(sum_, sumsq, n):
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sum_ / n
        var = np.maximum((sumsq - sum_ ** 2 / n) / (n - 1), 0.0)
    return mean, var


def _t_two(sum1, sumsq1, n1, sum2, sumsq2, n2):
    """Welch t (group 2 - group 1) with per-edge group sizes; undefined -> 0."""
    m1, v1 = _group_moments(sum1, sumsq1, n1)
    m2, v2 = _group_moments(sum2, sumsq2, n2)
    with np.errstate(divide="ignore", invalid="ignore"):
        floor = _degenerate.RTOL * (sumsq1 + sumsq2) / (n1 + n2) * (1 / n1 + 1 / n2)
        t = _degenerate.t_ratio(m2 - m1, v1 / n1 + v2 / n2, floor)
    t = np.where((n1 >= 2) & (n2 >= 2) & np.isfinite(t), t, 0.0)
    return np.maximum(t, 0.0), np.maximum(-t, 0.0)


def _welch_df(sum1, sumsq1, n1, sum2, sumsq2, n2):
    """Edge-wise Welch-Satterthwaite df with per-edge group sizes."""
    _, v1 = _group_moments(sum1, sumsq1, n1)
    _, v2 = _group_moments(sum2, sumsq2, n2)
    with np.errstate(divide="ignore", invalid="ignore"):
        se1, se2 = v1 / n1, v2 / n2
        denom = se1 ** 2 / (n1 - 1) + se2 ** 2 / (n2 - 1)
        df = (se1 + se2) ** 2 / denom
    return np.where(denom > 0, df, n1 + n2 - 2.0)


# ----------------------------------------------------------------------
# Observed and permuted statistics
# ----------------------------------------------------------------------
def _observed(d: _Design):
    if d.kind == "two-sample":
        s1, q1, k1 = d.X0[:d.n1].sum(0), d.X02[:d.n1].sum(0), d.M[:d.n1].sum(0)
        s, q, k = d.X0.sum(0), d.X02.sum(0), d.M.sum(0)
        t_pos, t_neg = _t_two(s1, q1, k1, s - s1, q - q1, k - k1)
    else:
        t_pos, t_neg = _t_one(d.X0.sum(0), d.X02.sum(0), d.M.sum(0))
    return np.where(d.tested, t_pos, 0.0), np.where(d.tested, t_neg, 0.0)


def _permuted(d: _Design, seed: int):
    """Same random draws as the complete-data fast path."""
    rng = np.random.RandomState(seed)
    if d.kind == "two-sample":
        n_all = d.X0.shape[0]
        if d.strata is None:
            g = np.zeros(n_all, dtype=np.float64)
            g[rng.choice(n_all, d.n1, replace=False)] = 1.0
        else:
            g = _stratified_choice_n1(d.strata, d.n1_per_stratum, rng)
        s1, q1, k1 = g @ d.X0, g @ d.X02, g @ d.M
        s, q, k = d.X0.sum(0), d.X02.sum(0), d.M.sum(0)
        t_pos, t_neg = _t_two(s1, q1, k1, s - s1, q - q1, k - k1)
    else:
        signs = rng.choice([1, -1], d.X0.shape[0]).astype(np.float64)
        t_pos, t_neg = _t_one(signs @ d.X0, d.X02.sum(0), d.M.sum(0))
    return np.where(d.tested, t_pos, 0.0), np.where(d.tested, t_neg, 0.0)


def _to_matrices(t_pos, t_neg, d: _Design):
    out = {}
    for key, vec in (("positive", t_pos), ("negative", t_neg)):
        mat = np.zeros((d.N, d.N))
        mat[d.triu] = vec
        out[key] = mat + mat.T
    return out


def _task_max(d: _Design, enhance_func, enhance_kwargs, seed):
    stat = _to_matrices(*_permuted(d, seed), d)
    if enhance_func is not None:
        stat = enhance_func(stat, **enhance_kwargs)
    return _extract_max_stats(stat, (d.N, d.N))


def _task_edges(d: _Design, seed):
    t_pos, t_neg = _permuted(d, seed)
    return {"positive": t_pos, "negative": t_neg}


# ----------------------------------------------------------------------
# Design and report
# ----------------------------------------------------------------------
def _build_design(group1, group2, test_type, strata, min_coverage) -> Tuple[_Design, Dict[str, Any]]:
    N = group1.shape[1]
    triu = np.triu_indices(N, k=1)
    if test_type == "two-sample":
        A = np.concatenate((group1, group2), axis=0)[:, triu[0], triu[1]]
        n1 = group1.shape[0]
    else:
        A = (group2 - group1 if test_type == "paired" else group1)[:, triu[0], triu[1]]
        n1 = A.shape[0]
    present = np.isfinite(A)
    X0 = np.where(present, A, 0.0)
    M = present.astype(np.float64)

    if test_type == "two-sample":
        k1, k2 = M[:n1].sum(0), M[n1:].sum(0)
        need1 = max(int(np.ceil(min_coverage * n1)), MIN_PER_GROUP)
        need2 = max(int(np.ceil(min_coverage * (A.shape[0] - n1))), MIN_PER_GROUP)
        tested = (k1 >= need1) & (k2 >= need2)
        edge_n = k1 + k2
    else:
        k = M.sum(0)
        tested = k >= max(int(np.ceil(min_coverage * A.shape[0])), MIN_PER_GROUP)
        edge_n = k

    strata_codes = n1_per_stratum = None
    if strata is not None and test_type == "two-sample":
        strata_codes = _encode_strata(strata)
        if strata_codes.shape[0] != A.shape[0]:
            raise ValueError(
                f"strata length {strata_codes.shape[0]} does not match "
                f"n1+n2={A.shape[0]} for the two-sample test."
            )
        n1_per_stratum = np.bincount(strata_codes[:n1], minlength=int(strata_codes.max()) + 1)

    design = _Design(
        kind="two-sample" if test_type == "two-sample" else "one-sample",
        X0=np.ascontiguousarray(X0), X02=np.ascontiguousarray(X0 ** 2),
        M=np.ascontiguousarray(M), n1=n1, tested=tested, triu=triu, N=N,
        strata=strata_codes, n1_per_stratum=n1_per_stratum,
    )
    return design, _report(group1, group2, test_type, design, edge_n, min_coverage)


def _report(group1, group2, test_type, d: _Design, edge_n, min_coverage) -> Dict[str, Any]:
    """Missingness and coverage summary attached to the result."""
    n_mat = np.zeros((d.N, d.N))
    n_mat[d.triu] = edge_n
    n_mat = n_mat + n_mat.T
    untested = [(int(i), int(j)) for i, j, ok in zip(d.triu[0], d.triu[1], d.tested) if not ok]
    rep: Dict[str, Any] = {
        "policy": "available_case",
        "min_coverage": min_coverage,
        "n_edges_tested": int(d.tested.sum()),
        "n_edges_untested": len(untested),
        "untested_edges": untested,
        "edge_n": n_mat,
        "edge_n_min": int(edge_n[d.tested].min()) if d.tested.any() else 0,
        "edge_n_median": float(np.median(edge_n[d.tested])) if d.tested.any() else 0.0,
    }
    groups = {"group1": group1} if test_type == "one-sample" else {"group1": group1, "group2": group2}
    regions: Dict[int, int] = {}
    missing_subjects = {}
    for name, g in groups.items():
        r = nonfinite_report(g)
        missing_subjects[name] = r["subjects"]
        for roi, c in r["rois"].items():
            regions[roi] = regions.get(roi, 0) + c
    rep["subjects_with_missing"] = missing_subjects
    rep["regions_without_signal"] = dict(sorted(regions.items()))
    rep["missing_by_group_p"] = None
    if test_type == "two-sample":
        a, n_a = len(missing_subjects["group1"]), group1.shape[0]
        b, n_b = len(missing_subjects["group2"]), group2.shape[0]
        rep["missing_by_group_p"] = float(stats.fisher_exact([[a, n_a - a], [b, n_b - b]])[1])
    return rep


def describe(report: Dict[str, Any]) -> str:
    """One-line summary of an available-case report (for flags / printing)."""
    subjects = sum(len(v) for v in report["subjects_with_missing"].values())
    regions = ", ".join(f"ROI {r} in {c}" for r, c in report["regions_without_signal"].items())
    text = (
        f"available-case analysis: {subjects} subject(s) with missing connectivity"
        f"{f' ({regions})' if regions else ''}; {report['n_edges_tested']} edges tested, "
        f"{report['n_edges_untested']} below {report['min_coverage']:.0%} coverage not tested; "
        f"per-edge n {report['edge_n_min']}–{int(report['edge_n'].max())}"
    )
    if report.get("missing_by_group_p") is not None:
        text += f"; missingness by group, Fisher p = {report['missing_by_group_p']:.3g}"
    if report.get("missing_by_interest_p") is not None:
        text += f"; missingness vs tested variable, p = {report['missing_by_interest_p']:.3g}"
    return text + "."


# ----------------------------------------------------------------------
# Entry point (called by compute_p_val)
# ----------------------------------------------------------------------
def available_case_p_val(
    group1: npt.NDArray[np.float64],
    group2: Optional[npt.NDArray[np.float64]],
    test_type: str,
    method: str,
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
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    block_inference: Optional[Dict[str, Any]] = None,
    two_tailed: bool = False,
):
    """p-values, raw t maps and the coverage report for one available-case run.

    ``method`` is ``'parametric_bonferroni'``, ``'parametric_bh_fdr'``,
    ``'bh_fdr_perm'`` or ``'max'`` (max-statistic, with ``enhance_func``
    applied when given). ``block_inference`` (``net_labels``, ``correction``)
    switches the permutation path to cNBS block inference: block means over
    the tested edges, one null per block; the block table is returned in the
    report under ``'block_report'``.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError(f"min_coverage must be in (0, 1]; got {min_coverage}.")
    d, report = _build_design(group1, group2, test_type, strata, min_coverage)
    if not d.tested.any():
        raise ValueError(
            "no edge has enough subjects for available-case inference "
            f"(min_coverage={min_coverage}, at least {MIN_PER_GROUP} per group)."
        )
    t_pos, t_neg = _observed(d)
    raw_t = _to_matrices(t_pos, t_neg, d)
    untested = ~d.tested

    def _matrix(vec):
        mat = np.full((d.N, d.N), np.nan)
        mat[d.triu] = vec
        mat[(d.triu[1], d.triu[0])] = vec
        np.fill_diagonal(mat, 1.0)
        return mat

    if method.startswith("parametric_"):
        if d.kind == "two-sample":
            s1, q1, k1 = d.X0[:d.n1].sum(0), d.X02[:d.n1].sum(0), d.M[:d.n1].sum(0)
            s, q, k = d.X0.sum(0), d.X02.sum(0), d.M.sum(0)
            df = _welch_df(s1, q1, k1, s - s1, q - q1, k - k1)
        else:
            df = d.M.sum(0) - 1.0
        p_values = {}
        for key, t in (("positive", t_pos), ("negative", t_neg)):
            p = np.full(t.shape, np.nan)
            raw = stats.t.sf((t_pos + t_neg if two_tailed else t)[d.tested], df[d.tested])
            if two_tailed:
                raw *= 2
            if method == "parametric_bonferroni":
                p[d.tested] = np.minimum(raw * d.tested.sum(), 1.0)
            else:
                p[d.tested] = _bh_fdr_correction(raw)
            if two_tailed:
                p[d.tested & (t == 0)] = 1.0
            p_values[key] = _matrix(p)
        return p_values, raw_t, report

    seeds = np.random.RandomState(random_state).randint(0, 2**32 - 1, size=n_permutations, dtype=np.int64)
    _use_mp = use_mp and not _is_worker_process()
    if _use_mp and n_processes is None:
        n_processes = get_available_cores()
    if _use_mp and n_processes is not None:
        n_processes = min(n_processes, n_permutations)

    if method == "bh_fdr_perm":
        perm = run_permutations(partial(_task_edges, d), list(seeds), use_mp=_use_mp,
                                n_processes=n_processes, verbose=verbose, desc="available_case")
        p_values = {}
        for key, t in (("positive", t_pos), ("negative", t_neg)):
            count = np.zeros(t.shape)
            for r in perm:
                count += ((r["positive"] + r["negative"] >= t_pos + t_neg)
                          if two_tailed else (r[key] >= t))
            per_edge = (count + 1.0) / (n_permutations + 1.0)
            p = np.full(t.shape, np.nan)
            p[d.tested] = _bh_fdr_correction(per_edge[d.tested])
            if two_tailed:
                p[d.tested & (t == 0)] = 1.0
            p_values[key] = _matrix(p)
        return p_values, raw_t, report

    tested_mat = np.zeros((d.N, d.N), dtype=bool)
    tested_mat[d.triu] = d.tested
    tested_mat = tested_mat | tested_mat.T
    from .enhance import apply_ni_tfnbs
    if block_inference is not None or enhance_func is apply_ni_tfnbs:
        enhance_kwargs = {**enhance_kwargs, "edge_mask": tested_mat}
    emp = raw_t if enhance_func is None else enhance_func(raw_t, **enhance_kwargs)
    results = run_permutations(
        partial(_task_max, d, enhance_func, enhance_kwargs), list(seeds),
        use_mp=_use_mp, n_processes=n_processes, verbose=verbose, desc="available_case",
    )
    null = _collect_results_to_arrays(results, n_permutations)
    if two_tailed and block_inference is None:
        joint = np.maximum(null["positive"], null["negative"])
        null = {"positive": joint, "negative": joint}
    if block_inference is not None:
        from ._cnbs import block_p_values
        p_values, report["block_report"] = block_p_values(
            emp, null, block_inference["net_labels"], d.N,
            correction=block_inference["correction"], edge_mask=tested_mat,
            acceleration=acceleration, two_tailed=two_tailed,
        )
    elif acceleration is not None:
        from .acceleration import compute_p_values_accelerated
        p_values = compute_p_values_accelerated(emp, null, method=acceleration)
    else:
        p_values = _compute_p_values_from_null(emp, null)
    # Untested edges are outside the family: NaN, never "not significant" by accident
    for key in p_values:
        p = np.array(p_values[key], dtype=np.float64)
        if p.ndim == 3:
            p[d.triu[0][untested], d.triu[1][untested], :] = np.nan
            p[d.triu[1][untested], d.triu[0][untested], :] = np.nan
        else:
            p[d.triu[0][untested], d.triu[1][untested]] = np.nan
            p[d.triu[1][untested], d.triu[0][untested]] = np.nan
        p_values[key] = p
    return p_values, raw_t, report
