"""Edges whose residual variance is zero.

A t (or F) statistic divides an effect by its standard error. When an edge's
residual variance is zero (constant values, an exact fit) the ratio is 0/0 if
the effect is zero as well, and infinite otherwise. The two cases differ:

* zero effect, zero variance (e.g. an edge that is 0 in every subject): no
  evidence of an effect; the statistic is 0;
* nonzero effect, zero variance (e.g. an edge that is 0.3 in every subject):
  the statistic is infinite. It cannot be enhanced (TFNBS heights) or ranked
  in a permutation null, and it usually signals duplicated subjects or a value
  filled in during preprocessing. The inference entry points reject it.

"Zero" is relative to the edge's mean square (``RTOL``): floating-point sums
leave ~1e-16 noise where the exact variance is zero, and dividing a nonzero
effect by that noise gave t ~ 1e16. A standard deviation below 1e-5 of the
root mean square is never real connectivity data.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import numpy.typing as npt

RTOL = 1e-10


def t_ratio(effect, se2, floor):
    """``effect / sqrt(se2)``, and 0 where ``se2`` is at or below ``floor``.

    ``floor`` is ``RTOL`` times the mean square, on the scale of ``se2``.
    NaN passes through.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        t = effect / np.sqrt(se2)
    return np.where(se2 <= floor, 0.0, t)


def infinite(effect2, se2, floor):
    """True where the variance is zero (at or below ``floor``) but the effect is not."""
    return (se2 <= floor) & (effect2 > floor)


def pairwise_infinite(
    group1: npt.NDArray[np.float64],
    group2: Optional[npt.NDArray[np.float64]],
    test_type: str,
) -> npt.NDArray[np.bool_]:
    """(N, N) mask of upper-triangle edges whose observed t is infinite.

    NaN-aware: each edge uses the subjects who have it (complete pairs for
    'paired'); edges with fewer than 2 values per group are skipped.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN edges
        return _pairwise_infinite(group1, group2, test_type)


def _pairwise_infinite(group1, group2, test_type):
    N = group1.shape[1]
    iu = np.triu_indices(N, k=1)
    out = np.zeros((N, N), dtype=bool)
    if test_type == "two-sample":
        a, b = group1[:, iu[0], iu[1]], group2[:, iu[0], iu[1]]
        n1, n2 = np.isfinite(a).sum(0), np.isfinite(b).sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            m1, m2 = np.nanmean(a, 0), np.nanmean(b, 0)
            h = 1.0 / n1 + 1.0 / n2
            se2 = np.nanvar(a, 0, ddof=1) / n1 + np.nanvar(b, 0, ddof=1) / n2
            meansq = (np.nansum(a ** 2, 0) + np.nansum(b ** 2, 0)) / (n1 + n2)
        ok = (n1 >= 2) & (n2 >= 2)
        mask = ok & infinite((m2 - m1) ** 2, se2, RTOL * meansq * h)
    else:
        d = group1 if test_type == "one-sample" else group2 - group1
        d = d[:, iu[0], iu[1]]
        k = np.isfinite(d).sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.nanmean(d, 0)
            se2 = np.nanvar(d, 0, ddof=1) / k
            meansq = np.nanmean(d ** 2, 0)
        mask = (k >= 2) & infinite(m ** 2, se2, RTOL * meansq / k)
    out[iu] = mask
    return out


def glm_infinite(
    Y: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    contrast: npt.NDArray[np.float64],
    stat_type: str,
) -> npt.NDArray[np.bool_]:
    """Columns of ``Y`` (n_subjects, n_edges) whose observed t or F is infinite.

    Raw beta has no variance in its denominator and is never infinite.
    """
    if stat_type == "beta" or Y.shape[1] == 0:
        return np.zeros(Y.shape[1], dtype=bool)
    pinv = np.linalg.pinv(X)
    beta = pinv @ Y
    df = X.shape[0] - int(np.linalg.matrix_rank(X))
    sigma2 = np.sum((Y - X @ beta) ** 2, axis=0) / max(df, 1)
    floor = RTOL * np.mean(Y ** 2, axis=0)
    C = np.atleast_2d(np.asarray(contrast, dtype=np.float64))
    Cb = C @ beta
    XtX = pinv @ pinv.T
    if stat_type == "fstat":
        # F = (quad / r) / sigma2: the effect is quad / r, on the scale of sigma2
        effect2 = np.einsum("ie,ij,je->e", Cb, np.linalg.pinv(C @ XtX @ C.T), Cb) / C.shape[0]
    else:
        # t^2 = (c'b)^2 / (c'(X'X)^+c sigma2)
        effect2 = Cb[0] ** 2 / float(C[0] @ XtX @ C[0])
    return infinite(effect2, sigma2, floor)


def raise_if_infinite(mask: npt.NDArray[np.bool_], statistic: str = "t") -> None:
    """Reject edges whose observed statistic is infinite (see the module docstring).

    ``mask`` is (N, N); only its upper triangle is read.
    """
    i, j = np.nonzero(np.triu(mask, k=1))
    if i.size == 0:
        return
    shown = ", ".join(f"({a}, {b})" for a, b in zip(i[:5], j[:5]))
    more = f" and {i.size - 5} more" if i.size > 5 else ""
    raise ValueError(
        f"{i.size} edge(s) have zero residual variance but a nonzero effect, so "
        f"their {statistic} statistic is infinite: {shown}{more}. Constant nonzero "
        "values usually mean duplicated subjects, a constant ROI signal, or values "
        "filled in during preprocessing. Fix the input, or set these edges to NaN "
        "and pass missing='available_case' to leave them untested."
    )
