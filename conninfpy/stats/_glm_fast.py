"""Edge-vector Freedman-Lane kernel for the single-contrast GLM permuter.

Internal module (NETSTAT-34). The permutation null of
:func:`conninfpy.stats.glm.compute_p_val_glm` only ever needs the ``E = N(N-1)/2``
upper-triangle edges: the stack is exactly symmetric with a zero diagonal
(enforced by ``check_connectivity``) and every statistic, floor and
enhancement is per-edge. Rebuilding ``(N, N)`` matrices per permutation
allocated two or three ``n x N**2`` arrays per draw, which dominated the
memory of the pipeline at realistic ``N``.

These helpers work on persistent ``(n_subjects, E)`` edge payloads:

* :func:`reduced_payload` — reduced-model fitted values and residuals in edge
  form, verified bit-for-bit against the full-matrix computation once per run
  (with that computation as the fallback).
* :func:`build_plan` — constant quantities (design, pseudoinverse, ``df``,
  contrast scale, ``M_inv`` for the F statistic, enhancement route) collected
  once, plus the ``_EdgeMaxSpec`` used to score each tail without a matrix.
* :func:`permutation_task` — one Freedman-Lane draw: reconstruct
  ``Y* = yhat + shuffled residuals`` as one ``(n, E)`` array, refit, compute
  the statistic explicitly from the residual, and score the tail.

Per-draw allocation is ``Y*`` plus the full-model fit and residual, i.e. a
small constant number of ``n x E`` arrays — never ``n x N**2``.

Everything here reproduces ``compute_glm_stat`` and the matrix enhancement
wrappers bit-for-bit: same formulas, same operation order, same RNG draws.
Nothing is relaxed.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import numpy as np
import numpy.typing as npt

from conninfpy._compat import make_tail_result
from conninfpy.stats import _degenerate as _degenerate
from conninfpy.stats._design import _residual_degrees_of_freedom
from conninfpy.stats._engine import _extract_max_stats, _stratified_perm
from conninfpy.stats.pairwise import (
    _EdgeMaxSpec,
    _build_edge_max_spec,
    _eval_edge_max_tail,
)

_FSTAT = "fstat"
_TSTAT = "tstat"
_BETA = "beta"


class EdgeStats(NamedTuple):
    """Per-edge statistic and, where the statistic divides by it, its scale."""

    stat: npt.NDArray[np.float64]
    sigma2: Optional[npt.NDArray[np.float64]] = None
    floor: Optional[npt.NDArray[np.float64]] = None


class EdgeGlmPlan(NamedTuple):
    """Constants of one Freedman-Lane run, shared by every permutation."""

    X: npt.NDArray[np.float64]
    X_pinv: npt.NDArray[np.float64]
    contrast: npt.NDArray[np.float64]
    stat_type: str
    df: int
    cXc: float
    M_inv: Optional[npt.NDArray[np.float64]]
    n_roi: int
    rows: npt.NDArray[np.intp]
    cols: npt.NDArray[np.intp]
    spec: Optional[_EdgeMaxSpec]
    enhance_func: Optional[Callable]
    enhance_kwargs: Dict[str, Any]
    scheme: str
    strata_codes: Optional[npt.NDArray[np.int_]]


def upper_triangle(N: int) -> Tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Row and column indices of the strict upper triangle of an (N, N) map."""
    return np.triu_indices(N, k=1)


def edge_payload_usable(Y: npt.NDArray[np.float64]) -> bool:
    """True when the upper triangle carries the whole statistic.

    The permutation path is a per-edge computation; it equals the full-matrix
    computation only when every edge has one value and the diagonal is zero
    (both guaranteed by ``check_connectivity`` at the entry points). Checked
    once per run, not per permutation.
    """
    if Y.shape[-1] < 2:
        return False
    if not np.array_equal(Y, Y.transpose(0, 2, 1)):
        return False
    idx = np.arange(Y.shape[-1])
    return not np.any(Y[:, idx, idx] != 0)


def reduced_payload(
    Y: npt.NDArray[np.float64],
    X_reduced: npt.NDArray[np.float64],
    rows: npt.NDArray[np.intp],
    cols: npt.NDArray[np.intp],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Reduced-model fitted values and residuals for the upper-triangle edges.

    Primary path: the reduced fit is evaluated on the ``(n, E)`` edge matrix.
    The full-matrix computation used by the matrix path is evaluated once as
    well and compared bit-for-bit; on any mismatch (BLAS blocking may depend
    on the number of columns) its upper-triangle slice is returned instead,
    so the values fed to the permutation kernel are exactly those the matrix
    path would use.

    Returns
    -------
    yhat_e, resid_e : ndarray of shape (n_subjects, E)
    """
    from conninfpy.stats.glm import _reduced_fit

    Y_e = np.ascontiguousarray(Y[:, rows, cols])
    yhat_e = _reduced_fit(X_reduced, Y_e)

    yhat_full = _reduced_fit(X_reduced, Y.reshape(Y.shape[0], -1)).reshape(Y.shape)
    yhat_full_e = yhat_full[:, rows, cols]
    if not np.array_equal(yhat_full_e, yhat_e):
        yhat_e = np.ascontiguousarray(yhat_full_e)

    return yhat_e, Y_e - yhat_e


def build_plan(
    Y: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    contrast: npt.NDArray[np.float64],
    stat_type: str,
    X_pinv: npt.NDArray[np.float64],
    XtX_inv: npt.NDArray[np.float64],
    enhance_func: Optional[Callable],
    enhance_kwargs: Dict[str, Any],
    scheme: str,
    strata_codes: Optional[npt.NDArray[np.int_]],
) -> Optional[EdgeGlmPlan]:
    """Collect the constants of one permutation run, or None if unused.

    ``None`` (the caller keeps the matrix path) when the stack is not exactly
    symmetric with a zero diagonal, or when ``N < 2`` leaves no edges.
    """
    if stat_type not in (_TSTAT, _BETA, _FSTAT):
        return None
    if not edge_payload_usable(Y):
        return None

    N = Y.shape[1]
    rows, cols = upper_triangle(N)
    cXc = 0.0
    M_inv: Optional[npt.NDArray[np.float64]] = None
    if stat_type == _FSTAT:
        C = contrast if contrast.ndim == 2 else contrast[None, :]
        # The observed call of compute_glm_stat has already rejected a
        # singular C(X'X)^+ C'; the same inverse is reused here.
        M_inv = np.linalg.inv(C @ XtX_inv @ C.T)
        contrast = C
    else:
        cXc = float(contrast @ XtX_inv @ contrast)

    spec = _build_edge_max_spec(enhance_func, enhance_kwargs, (rows, cols), N)

    return EdgeGlmPlan(
        X=X,
        X_pinv=X_pinv,
        contrast=contrast,
        stat_type=stat_type,
        df=_residual_degrees_of_freedom(X),
        cXc=cXc,
        M_inv=M_inv,
        n_roi=N,
        rows=rows,
        cols=cols,
        spec=spec,
        enhance_func=enhance_func,
        enhance_kwargs=enhance_kwargs,
        scheme=scheme,
        strata_codes=strata_codes,
    )


def _draw_y_edge(
    yhat_e: npt.NDArray[np.float64],
    r_e: npt.NDArray[np.float64],
    rng: np.random.RandomState,
    scheme: str,
    strata_codes: Optional[npt.NDArray[np.int_]],
) -> npt.NDArray[np.float64]:
    """``Y* = yhat + shuffled residuals`` as one (n, E) array.

    The RNG calls are exactly those of ``glm._shuffle_residuals``: one
    ``choice`` for sign flipping, one ``permutation`` (per stratum when
    blocked) otherwise, in the same order.
    """
    n = r_e.shape[0]
    if scheme == "sign-flip":
        signs = rng.choice([1.0, -1.0], n)
        Y_e = np.multiply(r_e, signs[:, None])
    else:
        perm = (
            rng.permutation(n)
            if strata_codes is None
            else _stratified_perm(strata_codes, rng)
        )
        Y_e = r_e[perm]
    Y_e += yhat_e
    return Y_e


def _edge_sigma2(
    Y_e: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    df: int,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Residual variance and mean square per edge, explicit residual.

    ``sigma2 = sum(resid**2) / df`` with ``resid = Y* - X beta`` — the same
    reductions (``np.sum(..., axis=0)`` for the residual, ``einsum`` for the
    mean square) as ``compute_glm_stat``, so the values are bit-identical
    rather than merely close. Never ``y'y - ...``: at the floor that
    difference is numerically meaningless.
    """
    resid = Y_e - X @ beta
    sigma2 = np.sum(resid * resid, axis=0) / df
    meansq = np.einsum("ij,ij->j", Y_e, Y_e) / Y_e.shape[0]
    return sigma2, meansq


def _tails_from_edge_stats(
    stat: npt.NDArray[np.float64], stat_type: str
) -> Dict[str, npt.NDArray[np.float64]]:
    """Split a signed edge statistic into the non-negative tail vectors."""
    if stat_type == _FSTAT:
        return {"omnibus": stat}
    return make_tail_result(
        np.where(stat > 0, stat, 0.0),
        np.where(stat < 0, -stat, 0.0),
    )


def edge_statistics(
    plan: EdgeGlmPlan, Y_e: npt.NDArray[np.float64]
) -> EdgeStats:
    """t/beta/F per edge, with exactly the formulas and operation order of
    :func:`conninfpy.stats.glm.compute_glm_stat` (the oracle).

    ``Y_e`` is ``(n_subjects, E)``; nothing here allocates ``n x N**2``.
    """
    beta = plan.X_pinv @ Y_e

    if plan.stat_type == _FSTAT:
        Cbeta = plan.contrast @ beta
        sigma2, meansq = _edge_sigma2(Y_e, beta, plan.X, plan.df)
        floor = _degenerate.RTOL * meansq
        quad = np.einsum("ie,ij,je->e", Cbeta, plan.M_inv, Cbeta)
        with np.errstate(divide="ignore", invalid="ignore"):
            F = quad / (plan.contrast.shape[0] * sigma2)
        F = np.where(sigma2 <= floor, 0.0, F)
        F = np.where(F < 0, 0.0, F)  # guard numerical noise near zero
        return EdgeStats(stat=F, sigma2=sigma2, floor=floor)

    c_beta = plan.contrast @ beta
    if plan.stat_type == _BETA:
        return EdgeStats(stat=c_beta)

    sigma2, meansq = _edge_sigma2(Y_e, beta, plan.X, plan.df)
    floor = _degenerate.RTOL * meansq
    stat = _degenerate.t_ratio(c_beta, plan.cXc * sigma2, plan.cXc * floor)
    return EdgeStats(stat=stat, sigma2=sigma2, floor=floor)


def _edge_stat_matrices(
    stat_dict: Dict[str, npt.NDArray[np.float64]],
    rows: npt.NDArray[np.intp],
    cols: npt.NDArray[np.intp],
    N: int,
) -> Dict[str, npt.NDArray[np.float64]]:
    """Assemble (N, N) maps from edge vectors for the legacy matrix enhancers."""
    out: Dict[str, npt.NDArray[np.float64]] = {}
    for key, vec in stat_dict.items():
        mat = np.zeros((N, N), dtype=np.float64)
        mat[rows, cols] = vec
        out[key] = mat + mat.T
    return out


def _max_tails(
    stat_dict: Dict[str, npt.NDArray[np.float64]],
) -> Dict[str, np.float64]:
    """Max of each non-negative tail vector.

    The rest of the (N, N) map is zero (structural zeros and the mirrored
    edges hold the same values), so this equals the matrix maximum.
    """
    out: Dict[str, np.float64] = {}
    for key, vec in stat_dict.items():
        out[key] = (
            np.float64(max(0.0, float(np.max(vec))))
            if vec.size
            else np.float64(0.0)
        )
    return out


def permutation_task(
    plan: EdgeGlmPlan,
    yhat_e: npt.NDArray[np.float64],
    r_e: npt.NDArray[np.float64],
    seed: int,
) -> Dict[str, Any]:
    """One Freedman-Lane permutation, scored on the edge vector.

    Reconstruct ``Y*`` (one ``n x E`` array), refit the full model, compute
    ``t``/``beta``/``F`` from the explicit residual, then either score each
    tail directly from the edge vector (``_EdgeMaxSpec``) or — for the legacy
    explicit-n and cNBS routes, which need a matrix — assemble the (N, N) map
    and call the same enhancer the matrix path uses.

    Returns the same per-permutation contract as
    ``glm._freedman_lane_permutation_task``: a tail dict of ``np.float64`` or
    of ``(K,)`` arrays for a parameter grid.
    """
    rng = np.random.RandomState(seed)
    Y_e = _draw_y_edge(yhat_e, r_e, rng, plan.scheme, plan.strata_codes)
    stat_dict = _tails_from_edge_stats(edge_statistics(plan, Y_e).stat, plan.stat_type)

    if plan.enhance_func is None:
        return _max_tails(stat_dict)

    if plan.spec is not None:
        if plan.stat_type == _FSTAT:
            return {"omnibus": _eval_edge_max_tail(stat_dict["omnibus"], plan.spec)}
        return make_tail_result(
            _eval_edge_max_tail(stat_dict["positive"], plan.spec),
            _eval_edge_max_tail(stat_dict["negative"], plan.spec),
        )

    # Legacy matrix route (explicit threshold grid, cNBS): same enhancer,
    # same input maps as the matrix path, only assembled from the edge vector.
    mats = _edge_stat_matrices(stat_dict, plan.rows, plan.cols, plan.n_roi)
    enhanced = plan.enhance_func(mats, **plan.enhance_kwargs)
    return _extract_max_stats(enhanced, (plan.n_roi, plan.n_roi))


__all__ = [
    "EdgeGlmPlan",
    "EdgeStats",
    "build_plan",
    "edge_payload_usable",
    "edge_statistics",
    "permutation_task",
    "reduced_payload",
    "upper_triangle",
]
