"""Descriptive per-subject leave-one-out influence on explicitly chosen edges.

This module answers one question about a *finished* inference result: which
subjects drive the native contrast effect ``c'β`` and the native statistic on
a small, caller-chosen set of edges? Nothing is tested again. There is no
permutation run, no random number generator, no p-value, no selected-edge
search and no threshold; the source result is read only.

Contract
--------
* **Explicit edges only.** ``edges`` is a finite list of unique, off-diagonal
  ROI index pairs, canonicalised to ``i < j``, capped at :data:`MAX_EDGES`
  (64). Duplicates, diagonal entries, out-of-range indices, more than 64 pairs
  and edges outside the producer's ``estimable``/``tested`` provenance masks
  raise :class:`ValueError` *before* any computation. There is no
  ``select='significant'`` option and no component/block aggregation. The
  report is also capped at :data:`MAX_RECORDS` (4096) subject × edge records:
  the refit grows quadratically with the number of subjects, so a larger
  request is refused as ``unavailable`` before any refit, never run quietly.
* **Leave-one-out, not resampling.** For every subject and every chosen edge
  the native model is refitted without that subject's row: ``Δ_s = c'β_full −
  c'β_(−s)`` on the analysis scale, the statistic and its df recomputed with
  the *same* formula the source test used, and the leverage of *that edge's*
  design recorded. Leverage is per edge, on the rows the edge actually uses:
  ``1/n_edge`` of the relevant group for a pairwise equivalent design, and
  the hat-matrix diagonal of the edge's coverage-row design for a GLM
  (``X_pinv[:, s] · X[s, :]``, never by materialising an ``n × n`` hat
  matrix). A row outside an edge's coverage has no leverage on that edge and
  gets ``None`` with the ``not_in_coverage`` status; no full-design leverage
  is reported, so two designs never share one column. For a paired design the
  deletion unit is the whole subject pair (the row leaves **both**
  conditions).
* **Native units.** Effects and ``Δ`` are in the response units of the
  source model (e.g. Fisher z, or the Fisher z difference ``B − A`` for a
  paired result), never standardised and never rescaled.
* **Inputs are not reconstructed.** A result carries coefficient maps and a
  design description, not the data that produced it. The caller supplies the
  tested inputs (the :func:`conninfpy.analyze` arguments): ``group1``,
  ``group2``, or ``Y`` plus ``design_matrix``. A compatibility check refits
  the full model on those inputs and requires it to reproduce ``result.beta``
  at :data:`CHECK_TOLERANCE` (off-diagonal cells; the diagonal is
  structurally zero in a connectivity result, and available-case GLM edges
  are checked on their own coverage rows); a mismatch is reported as
  ``unavailable`` with the reason "inputs do not reproduce the tested model".
  A positive ``Δ`` means the full-sample effect is larger than without that
  subject, i.e. the subject pulled the effect down. The check does not
  prove that every row is correctly aligned — row order, subject IDs and the
  analysis scale must be supplied correctly by the caller.
* **Paired alignment.** When the source result recorded a non-identity
  ``design['paired_alignment']``, either name the subjects with
  ``ids_group1``/``ids_group2`` (the subject-ID permutation is verified
  against the record and applied exactly once) or declare the inputs already
  aligned with ``paired_inputs_aligned=True`` (nothing is reordered, and the
  β reproduction check catches a wrong order). For a paired design ``c'β`` is
  the mean ``B − A`` and is therefore invariant to the row pairing: there the
  beta check cannot detect a wrong order, and the observed-statistic check is
  the guard. IDs without a recorded alignment, or a non-identity alignment
  without IDs or that flag, are reported as ``unavailable`` with the reason.
* **Harmonization is not refit.** ComBat and other sample-dependent
  transforms are fixed: the caller passes the already processed inputs, and
  the report notes that LOO is conditional on the fitted harmonization.
* **Unavailable, never invented.** Missing input data, legacy results without
  coefficient maps, omnibus F results, multi-row contrasts, undeclared
  ``var_equal``, statistic mismatches and missing provenance masks are
  reported as ``unavailable`` with a reason. A subject × edge entry whose
  refit is undefined (constant values, rank loss, no residual df,
  non-estimable contrast, no coverage) gets ``delta``/``t`` set to ``None``
  with a status, never a made-up number.

Statuses (per subject × edge, field ``status``)
-----------------------------------------------
``ok``, ``zero_mad`` (the edge's MAD is 0 without its values being identical,
i.e. majority ties plus at least one outlying value — the LOO ``Δ`` shows
which), ``constant`` (all values identical, variance 0, t undefined),
``rank_loss`` (the design lost rank after the deletion), ``not_estimable``
(the contrast is not a combination of the rows of the reduced design),
``no_residual_df``, ``not_in_coverage`` (the subject has no value on that edge;
``Δ`` is factually 0 and the statistic does not change) and ``nan_edge``.
``zero_mad`` is deliberately distinct from ``constant``: a zero MAD alone is
not a constant edge. Field ``loo_fit_status`` keeps the raw status of the
reduced fit, so a ``zero_mad`` row whose reduced data happen to be tied also
says why its ``t`` is ``None``.

This is a descriptive diagnostic. It says nothing about significance, about
the correctness of the original inference, or about which subjects to exclude.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import numpy.typing as npt

from conninfpy._shared.alignment import IdAlignmentError, paired_id_permutation
from conninfpy.stats._design import contrast_is_estimable
from conninfpy.stats._engine import _twosample_se2
from conninfpy.stats._estimates import pairwise_estimates
from conninfpy.stats.glm import compute_glm_stat
from conninfpy.stats.pairwise import (
    _compute_welch_degrees_of_freedom,
    _onesample_tstat_from_sums,
    _twosample_tstat_from_sums,
)

__all__ = [
    "CHECK_TOLERANCE",
    "MAX_EDGES",
    "MAX_RECORDS",
    "SCHEMA_VERSION",
    "subject_influence",
]

SCHEMA_VERSION = 1
MAX_EDGES = 64
MAX_RECORDS = 4096
CHECK_TOLERANCE = {"rtol": 1e-5, "atol": 1e-8}

_SUPPORTED_KINDS = ("one-sample", "two-sample", "paired", "glm")
_STATUSES = (
    "ok",
    "zero_mad",
    "constant",
    "rank_loss",
    "not_estimable",
    "no_residual_df",
    "not_in_coverage",
    "nan_edge",
)
_MAD_DEFINITION = (
    "unscaled median absolute deviation: median(|x - median(x)|), scale factor 1"
)


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _num(value: Any) -> Optional[float]:
    """JSON-safe float, or None for a missing or non-finite value."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mad(values: npt.NDArray[np.float64]) -> Optional[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    centre = float(np.median(finite))
    return float(np.median(np.abs(finite - centre)))


def _var_from_sums(values: npt.NDArray[np.float64]) -> float:
    """Sample variance (ddof=1) via the sums identity used by the pairwise tests."""
    array = np.asarray(values, dtype=np.float64)
    n = array.size
    total = float(array.sum())
    sumsq = float((array**2).sum())
    return max((sumsq - total * total / n) / (n - 1), 0.0)


def _nan_allclose(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    tolerance: Dict[str, float],
) -> Tuple[bool, Optional[float]]:
    """NaN-aware ``allclose``: same finite pattern first, then the values."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False, None
    finite_a, finite_b = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(finite_a, finite_b):
        return False, None
    if not finite_a.any():
        return True, 0.0
    diff = float(np.max(np.abs(a[finite_a] - b[finite_b])))
    return bool(np.allclose(a[finite_a], b[finite_b], **cast(Dict[str, Any], tolerance))), diff


def _empty_fit(
    status: str,
    n: int,
    *,
    df: Optional[float] = None,
    rank: Optional[int] = None,
    n_group1: Optional[int] = None,
    n_group2: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "effect": None,
        "stat": None,
        "df": df,
        "n": int(n),
        "n_group1": n_group1,
        "n_group2": n_group2,
        "variance": None,
        "se2": None,
        "variance_group1": None,
        "variance_group2": None,
        "rank": rank,
        "status": status,
    }


# ----------------------------------------------------------------------
# Native model fits (full model and each leave-one-out refit)
# ----------------------------------------------------------------------
def _onesample_fit(values: npt.NDArray[np.float64]) -> Dict[str, Any]:
    """One-sample mean and ``mean / (sd/sqrt(n))`` on ``values`` (paired: differences)."""
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    if n == 0:
        return _empty_fit("nan_edge", 0)
    mean = float(values.mean())
    if n < 2:
        fit = _empty_fit("no_residual_df", n, df=n - 1)
        fit["effect"] = mean
        return fit
    if bool(np.ptp(values) == 0.0):
        fit = _empty_fit("constant", n, df=n - 1)
        fit["effect"] = mean
        fit["variance"] = 0.0
        return fit
    variance = _var_from_sums(values)
    positive, negative = _onesample_tstat_from_sums(
        np.float64(values.sum()), np.float64((values**2).sum()), n
    )
    fit = _empty_fit("ok", n, df=n - 1)
    fit.update(
        {
            "effect": mean,
            "stat": float(positive - negative),
            "variance": variance,
            "se2": variance / n,
        }
    )
    return fit


def _twosample_fit(
    y1: npt.NDArray[np.float64],
    y2: npt.NDArray[np.float64],
    n1: int,
    keep: npt.NDArray[np.bool_],
    var_equal: bool,
) -> Dict[str, Any]:
    """Welch (``var_equal=False``) or Student two-sample t on the rows in ``keep``."""
    a = y1[keep[:n1]]
    b = y2[keep[n1:]]
    na, nb = int(a.size), int(b.size)
    n = na + nb
    if na == 0 or nb == 0:
        return _empty_fit("nan_edge", n, df=n - 2, n_group1=na, n_group2=nb)
    effect = float(b.mean() - a.mean())
    if na < 2 or nb < 2:
        fit = _empty_fit("no_residual_df", n, df=n - 2, n_group1=na, n_group2=nb)
        fit["effect"] = effect
        return fit
    var1, var2 = _var_from_sums(a), _var_from_sums(b)
    if var1 == 0.0 and var2 == 0.0:
        fit = _empty_fit("constant", n, df=n - 2, n_group1=na, n_group2=nb)
        fit["effect"] = effect
        fit["variance_group1"] = 0.0
        fit["variance_group2"] = 0.0
        return fit
    positive, negative = _twosample_tstat_from_sums(
        np.float64(a.sum()),
        np.float64((a**2).sum()),
        na,
        np.float64(b.sum()),
        np.float64((b**2).sum()),
        nb,
        var_equal,
    )
    if var_equal:
        df: Union[int, float] = n - 2
        variance: Optional[float] = float(
            ((na - 1) * var1 + (nb - 1) * var2) / (na + nb - 2)
        )
    else:
        df = float(_compute_welch_degrees_of_freedom(a, b))
        variance = None
    fit = _empty_fit("ok", n, df=df, n_group1=na, n_group2=nb)
    fit.update(
        {
            "effect": effect,
            "stat": float(positive - negative),
            "variance": variance,
            "se2": float(_twosample_se2(var1, na, var2, nb, var_equal)),
            "variance_group1": var1,
            "variance_group2": var2,
        }
    )
    return fit


def _glm_fit(
    y: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    contrast: npt.NDArray[np.float64],
    keep: npt.NDArray[np.bool_],
    full_rank: Optional[int] = None,
) -> Dict[str, Any]:
    """OLS fit on the rows in ``keep``: ``c'β``, residual df and the native t.

    ``full_rank`` is the rank of the *same edge's* full-coverage design: a
    reduced fit with a lower rank has lost a design direction (``rank_loss``).
    The absolute ``rank < p`` comparison is deliberately not used, because a
    design can be rank deficient before any subject is removed.
    """
    yk = y[keep]
    Xk = np.asarray(X[keep], dtype=np.float64)
    n = int(yk.size)
    if n == 0:
        return _empty_fit("nan_edge", 0)
    rank = int(np.linalg.matrix_rank(Xk))
    df = n - rank
    X_pinv = np.linalg.pinv(Xk)
    if not contrast_is_estimable(contrast, Xk, X_pinv):
        return _empty_fit("not_estimable", n, df=df, rank=rank)
    beta = X_pinv @ yk
    effect = float(contrast @ beta)
    if df <= 0:
        fit = _empty_fit("no_residual_df", n, df=df, rank=rank)
        fit["effect"] = effect
        return fit
    if bool(np.ptp(yk) == 0.0):
        fit = _empty_fit("constant", n, df=df, rank=rank)
        fit["effect"] = effect
        fit["variance"] = 0.0
        return fit
    XtX_inv = X_pinv @ X_pinv.T
    stat = compute_glm_stat(
        yk.reshape(-1, 1, 1),
        Xk,
        contrast,
        stat_type="tstat",
        X_pinv=X_pinv,
        XtX_inv=XtX_inv,
    )
    t = float(stat["positive"][0, 0] - stat["negative"][0, 0])
    residuals = yk - Xk @ beta
    sigma2 = float(np.sum(residuals**2) / df)
    lost_rank = full_rank is not None and rank < int(full_rank)
    fit = _empty_fit("rank_loss" if lost_rank else "ok", n, df=df, rank=rank)
    fit.update(
        {
            "effect": effect,
            "stat": t,
            "variance": sigma2,
            "se2": float(contrast @ XtX_inv @ contrast) * sigma2,
        }
    )
    return fit


# ----------------------------------------------------------------------
# Report scaffolding
# ----------------------------------------------------------------------
def _skeleton(result: Any) -> Dict[str, Any]:
    design = getattr(result, "design", None)
    design = design if isinstance(design, dict) else {}
    provenance = getattr(result, "analysis_provenance", None)
    provenance = provenance if isinstance(provenance, dict) else {}
    contrast = design.get("contrast")
    unit = getattr(result, "inference_unit", None)
    unit_context: Dict[str, Any] = {
        "inference_unit": unit,
        "method": getattr(result, "method", None),
        "correction": getattr(result, "correction", None),
        "note": (
            "The source result's inference unit is the edge; this diagnostic is "
            "descriptive and adds no multiplicity statement."
            if unit in (None, "edge")
            else (
                f"The source result's inference unit is the {unit}. This "
                "diagnostic describes the native effect at explicitly selected "
                "edges only: those are NOT independent edge discoveries, and the "
                "source p-value map still belongs to its "
                f"{unit}-level (corrected) inference."
            )
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "descriptive_only": True,
        "test_type": design.get("kind"),
        "kind": design.get("kind"),
        "stat_type": getattr(result, "stat_type", None),
        "units": provenance.get("units") or design.get("scale"),
        "scale": provenance.get("scale") or design.get("scale"),
        "harmonized": bool(getattr(result, "harmonized", False)),
        "contrast": (
            [float(value) for value in np.ravel(contrast)]
            if contrast is not None
            else None
        ),
        "tolerance": dict(CHECK_TOLERANCE),
        "inference_unit": unit_context,
        "n_subjects": None,
        "n_edges": 0,
        "checks": [],
        "edges": [],
        "per_subject_edge": [],
        "subject_summary": [],
        "ranking": [],
        "summary": {},
        "paired_alignment": None,
        "mad": {"definition": _MAD_DEFINITION, "per": "edge"},
        "delta": {
            "definition": "c'beta_full - c'beta_(-s) in the native units of the "
            "tested model; positive means the subject pulled the effect down",
            "removal_unit": (
                "whole subject pair (both conditions) for a paired design, "
                "otherwise one observation"
            ),
        },
        "leverage": {
            "definition": "leverage of this edge's design on the rows the edge "
            "actually uses: the hat-matrix diagonal for a GLM, 1/n of the "
            "row's group (1/n_edge for a paired/one-sample edge) for a "
            "pairwise equivalent design; None when the row is outside the "
            "edge's coverage",
            "scope": "per edge; no full-design leverage is reported",
        },
        "statuses": list(_STATUSES),
        "unavailable": [],
        "limitations": [
            "Descriptive only: no p-values, no permutation, no random numbers, "
            "no selected-edge search and no threshold are involved; the source "
            "result is not modified.",
            "Leave-one-out is conditional on the supplied inputs: row order, "
            "subject IDs and the analysis scale must be the ones that produced "
            "the result. A mismatch is reported as unavailable, never repaired.",
            "Each deletion refits the native model on one fewer observation; "
            "the resulting t values are not hypotheses and are not corrected "
            "for multiplicity.",
            "MAD is the unscaled median absolute deviation. MAD = 0 alone is "
            "not a constant edge (majority ties plus an outlying value); only "
            "an edge whose values are all identical is reported as constant.",
            "Leverage is per edge and covers only the rows that edge uses "
            "(hat-matrix diagonal for a GLM, 1/n_edge of the group for a "
            "pairwise equivalent design); a row outside an edge's coverage has "
            "no leverage there and reports None.",
            "A reduced fit whose design rank falls below the full fit's rank on "
            "the same edge is flagged rank_loss; its delta/statistic are "
            "reported but excluded from the per-subject ranking and from the "
            "ok counts. A contrast that stops being estimable is "
            "not_estimable with None values.",
            "The diagnostic cannot tell whether the original inference, its "
            "enhancement or its thresholding was appropriate.",
        ],
    }


def _stop(report: Dict[str, Any], code: str, reason: str) -> Dict[str, Any]:
    report["unavailable"].append({"code": code, "reason": reason})
    return report


def _check(
    report: Dict[str, Any],
    name: str,
    status: str,
    detail: str,
    *,
    max_abs_diff: Optional[float] = None,
) -> None:
    entry: Dict[str, Any] = {"check": name, "status": status, "detail": detail}
    if max_abs_diff is not None:
        entry["max_abs_diff"] = _num(max_abs_diff)
        entry["tolerance"] = dict(CHECK_TOLERANCE)
    report["checks"].append(entry)


# ----------------------------------------------------------------------
# Preconditions and edge selection
# ----------------------------------------------------------------------
def _structure(
    result: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[str, str]]]:
    """Validate the parts of the result the diagnostic needs."""
    if type(result).__name__ == "OmnibusInferenceResult":
        return None, (
            "omnibus_not_supported",
            "an F-stat omnibus result has no signed contrast effect c'β; there "
            "is no native single-contrast effect to leave a subject out of. Use "
            "a per-contrast (tstat/beta) result.",
        )
    design = getattr(result, "design", None)
    if not isinstance(design, dict):
        return None, (
            "no_design",
            "the result carries no design description (columns, contrast, "
            "n_subjects), so there is no tested model to reproduce.",
        )
    contrast = design.get("contrast")
    if contrast is None:
        return None, ("no_contrast", "design['contrast'] is missing; there is no c'β.")
    c = np.asarray(contrast, dtype=np.float64)
    if c.ndim != 1 or c.size == 0:
        return None, (
            "multi_row_contrast",
            "a multi-row (omnibus F) contrast has no single native effect c'β; "
            "the diagnostic is defined for a one-row contrast only.",
        )
    beta = getattr(result, "beta", None)
    if beta is None:
        return None, (
            "no_beta",
            "the result carries no coefficient maps (beta): a legacy result "
            "cannot be refitted, so no subject can be left out.",
        )
    beta_array = np.asarray(beta, dtype=np.float64)
    if beta_array.ndim != 3:
        return None, ("bad_beta", f"beta must have shape (p, N, N); got {beta_array.shape}.")
    if beta_array.shape[0] != c.size:
        return None, (
            "contrast_beta_mismatch",
            f"the contrast has {c.size} entries but beta has "
            f"{beta_array.shape[0]} coefficient maps.",
        )
    provenance = getattr(result, "analysis_provenance", None)
    if not isinstance(provenance, dict):
        return None, (
            "no_provenance",
            "the result has no analysis provenance: the producer's estimable/"
            "tested edge masks are unknown, so an explicitly selected edge "
            "cannot be checked against the tested family.",
        )
    kind = design.get("kind")
    if kind not in _SUPPORTED_KINDS:
        return None, (
            "unsupported_design",
            f"unsupported design kind {kind!r}; expected one of "
            f"{list(_SUPPORTED_KINDS)}.",
        )
    columns = design.get("columns") or []
    if len(columns) != c.size:
        return None, (
            "columns_contrast_mismatch",
            f"design['columns'] names {len(columns)} columns but the contrast "
            f"has {c.size} entries.",
        )
    n_subjects = design.get("n_subjects")
    if not isinstance(n_subjects, (int, np.integer)) or int(n_subjects) <= 0:
        return None, (
            "no_n_subjects",
            "design['n_subjects'] is missing or not a positive integer.",
        )
    return (
        {
            "design": design,
            "provenance": provenance,
            "beta": beta_array,
            "contrast": c,
            "N": int(beta_array.shape[1]),
            "p": int(c.size),
            "kind": str(kind),
            "n_subjects": int(n_subjects),
            "stat_type": getattr(result, "stat_type", None),
        },
        None,
    )


def _mask_arrays(
    provenance: Dict[str, Any], N: int
) -> Tuple[Optional[npt.NDArray[np.bool_]], Optional[Dict[str, Any]]]:
    estimable = provenance.get("estimable")
    tested = provenance.get("tested")
    if estimable is not None:
        estimable = np.asarray(estimable, dtype=bool)
        if estimable.shape != (N, N):
            raise ValueError(
                f"provenance 'estimable' mask has shape {estimable.shape}, "
                f"expected ({N}, {N})."
            )
    if tested is not None:
        if isinstance(tested, dict):
            masks = {}
            for tail, mask in tested.items():
                array = np.asarray(mask, dtype=bool)
                if array.shape != (N, N):
                    raise ValueError(
                        f"provenance 'tested'[{tail!r}] mask has shape "
                        f"{array.shape}, expected ({N}, {N})."
                    )
                masks[tail] = array
            tested = masks
        else:
            array = np.asarray(tested, dtype=bool)
            if array.shape != (N, N):
                raise ValueError(
                    f"provenance 'tested' mask has shape {array.shape}, "
                    f"expected ({N}, {N})."
                )
            tested = {"map": array}
    return estimable, tested


def _validate_edges(
    edges: Any, N: int, provenance: Dict[str, Any]
) -> List[Tuple[int, int]]:
    """Canonicalise, bound and mask-check the explicit edge list (ValueError)."""
    if edges is None:
        raise ValueError(
            "edges must be an explicit finite list of (i, j) ROI index pairs; "
            "this diagnostic never searches for edges."
        )
    try:
        requested = list(edges)
    except TypeError as exc:
        raise ValueError(
            "edges must be an iterable of (i, j) ROI index pairs."
        ) from exc
    if not requested:
        raise ValueError("edges must not be empty.")
    if len(requested) > MAX_EDGES:
        raise ValueError(
            f"at most {MAX_EDGES} edges can be diagnosed; got {len(requested)}. "
            "Select the edges explicitly instead of passing a whole family."
        )
    pairs: List[Tuple[int, int]] = []
    seen = set()
    for position, entry in enumerate(requested):
        try:
            pair = tuple(entry)
        except TypeError as exc:
            raise ValueError(
                f"edge #{position} ({entry!r}) is not an (i, j) pair."
            ) from exc
        if len(pair) != 2:
            raise ValueError(
                f"edge #{position} ({entry!r}) does not have exactly two entries."
            )
        indices = []
        for value in pair:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(
                    f"edge #{position} ({entry!r}) has a non-integer ROI index."
                )
            indices.append(int(value))
        i, j = indices
        if not (0 <= i < N and 0 <= j < N):
            raise ValueError(
                f"edge #{position} ({i}, {j}) is outside the ROI range 0..{N - 1}."
            )
        if i == j:
            raise ValueError(
                f"edge #{position} ({i}, {j}) is on the diagonal; only "
                "off-diagonal edges carry a connectivity effect."
            )
        canonical = (i, j) if i < j else (j, i)
        if canonical in seen:
            raise ValueError(
                f"edge #{position} ({i}, {j}) duplicates {canonical}; each edge "
                "may be requested once."
            )
        seen.add(canonical)
        pairs.append(canonical)
    estimable, tested = _mask_arrays(provenance, N)
    for i, j in pairs:
        if estimable is not None and not bool(estimable[i, j]):
            raise ValueError(
                f"edge ({i}, {j}) is not in the producer's estimable mask; its "
                "contrast effect is undefined for the tested model."
            )
        if tested is not None and not any(
            bool(mask[i, j]) for mask in tested.values()
        ):
            raise ValueError(
                f"edge ({i}, {j}) is not in the producer's tested mask; the "
                "source result did not test it, so it cannot be diagnosed."
            )
    return sorted(pairs)


def _resolve_inputs(
    kind: str, N: int, group1: Any, group2: Any, Y: Any, design_matrix: Any
) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[str, str]]]:
    if kind == "glm":
        if group1 is not None or group2 is not None:
            raise ValueError(
                "group1/group2 describe a pairwise design; a glm result takes "
                "Y= (the tested response) and design_matrix= (the tested X)."
            )
        if Y is None or design_matrix is None:
            return None, (
                "no_input_data",
                "no input data: a glm result needs both the tested response Y= "
                "and the tested design_matrix= (including intercept, confounds "
                "and any site dummies). Nothing is reconstructed from the "
                "result.",
            )
        response = np.asarray(Y, dtype=np.float64)
        matrix = np.asarray(design_matrix, dtype=np.float64)
        if response.ndim != 3 or response.shape[1:] != (N, N):
            raise ValueError(
                "Y must have shape (n_subjects, N, N) matching the result; got "
                f"{response.shape} for N={N}."
            )
        if matrix.ndim != 2 or matrix.shape[0] != response.shape[0]:
            raise ValueError(
                "design_matrix must be 2D with one row per subject; got shape "
                f"{matrix.shape} for {response.shape[0]} subjects."
            )
        if not np.isfinite(matrix).all():
            raise ValueError("design_matrix must be finite.")
        return (
            {
                "Y": response,
                "X": matrix,
                "group1": None,
                "group2": None,
                "n_rows": int(response.shape[0]),
                "n1": None,
            },
            None,
        )

    if Y is not None or design_matrix is not None:
        raise ValueError(
            "Y/design_matrix describe a glm design; a pairwise result takes "
            "group1= (and group2= for two-sample/paired)."
        )
    if group1 is None:
        return None, (
            "no_input_data",
            "no input data: this result was produced from group1"
            + (" and group2" if kind in ("two-sample", "paired") else "")
            + "; pass the same arrays on the analysis scale. Nothing is "
            "reconstructed from the result.",
        )
    first = np.asarray(group1, dtype=np.float64)
    if first.ndim != 3 or first.shape[1:] != (N, N):
        raise ValueError(
            "group1 must have shape (n_subjects, N, N) matching the result; got "
            f"{first.shape} for N={N}."
        )
    if kind == "one-sample":
        return (
            {
                "group1": first,
                "group2": None,
                "Y": None,
                "X": None,
                "n_rows": int(first.shape[0]),
                "n1": None,
            },
            None,
        )
    if group2 is None:
        return None, (
            "no_input_data",
            f"no input data: the {kind} design needs group2= as well. Nothing "
            "is reconstructed from the result.",
        )
    second = np.asarray(group2, dtype=np.float64)
    if second.ndim != 3 or second.shape[1:] != (N, N):
        raise ValueError(
            "group2 must have shape (n_subjects, N, N) matching the result; got "
            f"{second.shape} for N={N}."
        )
    if kind == "paired" and second.shape[0] != first.shape[0]:
        raise ValueError(
            "a paired design needs the same number of subject rows in both "
            f"conditions; got {first.shape[0]} and {second.shape[0]}."
        )
    # A paired deletion unit is the whole subject pair, so the row axis of the
    # diagnostic is the pair axis (design['n_subjects'] counts pairs).
    n_rows = (
        int(first.shape[0])
        if kind == "paired"
        else int(first.shape[0] + second.shape[0])
    )
    return (
        {
            "group1": first,
            "group2": second,
            "Y": None,
            "X": None,
            "n_rows": n_rows,
            "n1": int(first.shape[0]),
        },
        None,
    )


def _align_paired(
    result: Any,
    group2: npt.NDArray[np.float64],
    ids_group1: Optional[Sequence[Any]],
    ids_group2: Optional[Sequence[Any]],
    paired_inputs_aligned: bool,
) -> Tuple[
    Optional[npt.NDArray[np.float64]], Optional[Tuple[str, str]], Dict[str, Any]
]:
    """Apply the recorded subject-ID alignment exactly once (or none at all)."""
    design = getattr(result, "design", None) or {}
    record = design.get("paired_alignment")
    named = ids_group1 is not None or ids_group2 is not None
    note: Dict[str, Any] = {
        "recorded": record is not None,
        "identity_permutation": None,
        "permutation_applied": False,
        "paired_inputs_aligned": bool(paired_inputs_aligned),
        "note": "",
    }
    if paired_inputs_aligned:
        note["note"] = (
            "paired_inputs_aligned=True: the caller declares group2 already in "
            "group1's row order. No permutation is applied (a second reorder is "
            "forbidden); the beta reproduction check is the guard against a "
            "wrong order."
        )
        if named:
            note["note"] += " The supplied subject IDs were not used for reordering."
        return group2, None, note
    if record is None:
        if named:
            return None, (
                "paired_alignment_unrecorded",
                "subject IDs were supplied but the result recorded no "
                "paired_alignment; the permutation the producer applied cannot "
                "be verified, and applying the IDs would be a second reorder. "
                "Pass the aligned input with paired_inputs_aligned=True instead.",
            ), note
        note["note"] = (
            "The result recorded no paired_alignment and no IDs were supplied: "
            "group2 is assumed to follow group1's row order, which the beta "
            "reproduction check verifies."
        )
        return group2, None, note
    permutation = np.asarray(record.get("permutation_b_to_a"), dtype=np.intp)
    identity = bool(record.get("identity_permutation", False))
    note["identity_permutation"] = identity
    n_pairs = int(group2.shape[0])
    if permutation.shape != (n_pairs,):
        return None, (
            "paired_alignment_mismatch",
            f"the recorded alignment has {permutation.size} entries but group2 "
            f"has {n_pairs} subject rows.",
        ), note
    if identity:
        if named:
            try:
                # One-sided IDs are rejected inside ``paired_id_permutation``.
                check = paired_id_permutation(
                    cast(Sequence[Any], ids_group1), cast(Sequence[Any], ids_group2),
                    n_group1=n_pairs, n_group2=n_pairs,
                )
            except IdAlignmentError as exc:
                raise ValueError(str(exc)) from exc
            if not np.array_equal(check, np.arange(n_pairs, dtype=np.intp)):
                return None, (
                    "paired_alignment_mismatch",
                    "the result recorded an identity alignment but the supplied "
                    "subject IDs imply a non-identity permutation; the inputs "
                    "are not the ones the result was computed from.",
                ), note
        note["note"] = (
            "The result recorded an identity alignment (condition B already "
            "followed condition A's order); nothing was reordered."
        )
        return group2, None, note
    if not named:
        return None, (
            "paired_alignment_ids_required",
            "the result recorded a non-identity subject-ID alignment, so the "
            "original condition-B order cannot be guessed: pass ids_group1/"
            "ids_group2 (verified against the record) or the already aligned "
            "input with paired_inputs_aligned=True.",
        ), note
    try:
        check = paired_id_permutation(
            cast(Sequence[Any], ids_group1), cast(Sequence[Any], ids_group2), n_group1=n_pairs, n_group2=n_pairs
        )
    except IdAlignmentError as exc:
        raise ValueError(str(exc)) from exc
    if not np.array_equal(check, permutation):
        return None, (
            "paired_alignment_mismatch",
            "the supplied subject IDs imply a different permutation than the "
            "one recorded in design['paired_alignment']; refusing to apply a "
            "second, different reorder.",
        ), note
    note["permutation_applied"] = True
    note["note"] = (
        "The recorded subject-ID permutation was verified against ids_group1/ "
        "ids_group2 and applied to group2 exactly once."
    )
    return group2[permutation], None, note


# ----------------------------------------------------------------------
# Per-edge values and fits
# ----------------------------------------------------------------------
def _edge_arrays(
    kind: str, inputs: Dict[str, Any], i: int, j: int
) -> Tuple[Dict[str, Any], npt.NDArray[np.bool_]]:
    if kind == "glm":
        y = inputs["Y"][:, i, j]
        matrix = inputs["X"]
        mask = np.isfinite(y) & np.isfinite(matrix).all(axis=1)
        return {"y": y}, mask
    first = inputs["group1"][:, i, j]
    if kind == "one-sample":
        return {"y1": first}, np.isfinite(first)
    second = inputs["group2"][:, i, j]
    if kind == "paired":
        difference = second - first
        return (
            {"y1": first, "y2": second, "difference": difference},
            np.isfinite(difference),
        )
    return (
        {"y1": first, "y2": second, "n1": int(inputs["n1"])},
        np.concatenate([np.isfinite(first), np.isfinite(second)]),
    )


def _edge_values(
    kind: str, arrays: Dict[str, Any], mask: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """The response values actually used by the test on this edge."""
    if kind == "glm":
        return arrays["y"][mask]
    if kind == "two-sample":
        n1 = int(arrays["n1"])
        return np.concatenate([arrays["y1"][mask[:n1]], arrays["y2"][mask[n1:]]])
    values = arrays["y1"] if kind == "one-sample" else arrays["difference"]
    return values[mask]


def _fit_edge(
    kind: str,
    arrays: Dict[str, Any],
    mask: npt.NDArray[np.bool_],
    X: Optional[npt.NDArray[np.float64]],
    contrast: npt.NDArray[np.float64],
    var_equal: Optional[bool],
    full_rank: Optional[int] = None,
) -> Dict[str, Any]:
    if kind == "glm":
        return _glm_fit(arrays["y"], cast(npt.NDArray[np.float64], X), contrast, mask, full_rank)
    if kind == "two-sample":
        return _twosample_fit(
            arrays["y1"], arrays["y2"], int(arrays["n1"]), mask, bool(var_equal)
        )
    values = arrays["y1"] if kind == "one-sample" else arrays["difference"]
    return _onesample_fit(values[mask])


def _edge_leverage(
    kind: str,
    arrays: Dict[str, Any],
    mask: npt.NDArray[np.bool_],
    X: Optional[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Leverage on this edge's rows only (NaN outside the edge's coverage).

    GLM: diagonal of the edge's design hat matrix, one entry at a time as
    ``X_pinv[:, s] · X[s, :]`` so no ``n × n`` matrix is ever formed.
    Pairwise: ``1/n`` of the row's group on this edge (the equivalent design
    has one mean parameter per group).
    """
    out = np.full(int(mask.size), np.nan)
    used = np.flatnonzero(mask)
    if used.size == 0:
        return out
    if kind == "glm":
        design = np.asarray(X, dtype=np.float64)[used]
        pinv = np.linalg.pinv(design)
        # H_ss = X[s, :] @ pseudoinverse @ X[s, :]^T, one row at a time
        out[used] = np.einsum("sj,sj->s", pinv.T, design)
        return out
    if kind == "two-sample":
        n1 = int(arrays["n1"])
        n_group1 = float(mask[:n1].sum())
        n_group2 = float(mask[n1:].sum())
        group1_rows = used[used < n1]
        group2_rows = used[used >= n1]
        if n_group1 > 0:
            out[group1_rows] = 1.0 / n_group1
        if n_group2 > 0:
            out[group2_rows] = 1.0 / n_group2
        return out
    out[used] = 1.0 / float(used.size)
    return out


def _row_coverage_counts(
    kind: str, arrays: Dict[str, Any], mask: npt.NDArray[np.bool_]
) -> Dict[str, int]:
    if kind == "two-sample":
        n1 = int(arrays["n1"])
        return {
            "n_group1": int(mask[:n1].sum()),
            "n_group2": int(mask[n1:].sum()),
        }
    return {}


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def subject_influence(
    result: Any,
    *,
    edges: Sequence[Tuple[int, int]],
    group1: Optional[npt.NDArray[np.float64]] = None,
    group2: Optional[npt.NDArray[np.float64]] = None,
    Y: Optional[npt.NDArray[np.float64]] = None,
    design_matrix: Optional[npt.NDArray[np.float64]] = None,
    var_equal: Optional[bool] = None,
    ids_group1: Optional[Sequence[Any]] = None,
    ids_group2: Optional[Sequence[Any]] = None,
    paired_inputs_aligned: bool = False,
) -> Dict[str, Any]:
    """Descriptive per-subject leave-one-out influence on explicitly chosen edges.

    Parameters
    ----------
    result : InferenceResult
        A finished result with ``beta``, ``design`` and
        ``analysis_provenance``. It is only read: no p-value, mask, sample,
        enhancement setting or preprocessing is touched.
    edges : sequence of (i, j)
        Explicit, unique, off-diagonal ROI index pairs (canonicalised to
        ``i < j``), at most :data:`MAX_EDGES`. Each must lie inside the
        producer's ``estimable`` and ``tested`` provenance masks, otherwise
        :class:`ValueError` is raised before any computation.
    group1, group2 : ndarray of shape (n_subjects, N, N), optional
        The tested pairwise inputs on the analysis scale (Fisher z already
        applied; for a paired design the two condition matrices). ``group1``
        for one-sample; ``group1`` and ``group2`` for two-sample and paired.
    Y : ndarray of shape (n_subjects, N, N), optional
        The tested GLM response (for a paired GLM the ``Δz`` difference).
    design_matrix : ndarray of shape (n_subjects, p), optional
        The tested GLM design matrix, including intercept, confounds and any
        site dummies.
    var_equal : bool, optional
        The two-sample variance assumption **as used by the source test**. It
        is required explicitly for a two-sample result; ``None`` makes the
        diagnostic unavailable instead of guessing Welch.
    ids_group1, ids_group2 : sequence, optional
        Subject IDs of the paired design. With a recorded non-identity
        alignment they are verified against the record and the recorded
        permutation is applied exactly once.
    paired_inputs_aligned : bool, default False
        Declare ``group2`` already in ``group1``'s row order. Nothing is
        reordered; a wrong order is caught by the β reproduction check.

    Returns
    -------
    dict
        JSON-serialisable descriptive report: schema version, test type,
        units and scale, contrast, inference-unit context, the executed
        compatibility checks with their tolerance, per-edge full estimates,
        per-subject × edge ``Δ``/t/df/n/leverage/MAD/status entries, a
        per-subject summary with a ranking, ``unavailable`` reasons and
        ``limitations``. ``descriptive_only`` is always ``True``.

    Notes
    -----
    Only the checks listed under ``checks`` are possible from a result plus
    the supplied inputs: design shape, full-model β reproduction, the native
    statistic on the selected edges and (with available-case output) per-edge
    coverage. Row order, subject IDs and the analysis scale are the caller's
    responsibility; the reproduction check only catches a mismatch, it does
    not prove row-by-row identity. Multiplying the two-sample Welch degrees of
    freedom by a leave-one-out variance ratio is *not* attempted: the df are
    recomputed from the reduced sample with the same Welch–Satterthwaite
    formula.
    """
    inference = getattr(result, "inference", None)
    if not isinstance(getattr(result, "design", None), dict) and inference is not None:
        result = inference  # accept the AnalyzeResult from analyze() as well
    report = _skeleton(result)
    info, unavailable = _structure(result)
    if unavailable is not None:
        return _stop(report, *unavailable)

    assert info is not None  # _structure returns (info, None) or (None, unavailable)
    N = int(info["N"])
    pairs = _validate_edges(edges, N, info["provenance"])
    kind = info["kind"]
    contrast = info["contrast"]
    design = info["design"]
    provenance = info["provenance"]
    stat_type = info["stat_type"]

    if kind == "two-sample":
        if var_equal is None:
            return _stop(
                report,
                "var_equal_not_declared",
                "a two-sample result needs var_equal= declared as an explicit "
                "bool (the source test's assumption). Welch is never inferred "
                "from a default.",
            )
        if not isinstance(var_equal, (bool, np.bool_)):
            raise ValueError(f"var_equal must be an explicit bool; got {var_equal!r}.")
        var_equal = bool(var_equal)
    elif var_equal is not None:
        raise ValueError(
            "var_equal is only meaningful for a two-sample result; this result "
            f"is a {kind} design."
        )
    if paired_inputs_aligned and kind != "paired":
        raise ValueError(
            "paired_inputs_aligned is only meaningful for the paired design; "
            f"this result is a {kind} design."
        )

    inputs, unavailable = _resolve_inputs(kind, N, group1, group2, Y, design_matrix)
    if unavailable is not None:
        return _stop(report, *unavailable)
    assert inputs is not None  # _resolve_inputs returns (inputs, None) or (None, unavailable)
    report["n_subjects"] = int(inputs["n_rows"])

    # ---- Work/output cap before any refit (the refit is quadratic in n) ----
    n_records = int(inputs["n_rows"]) * len(pairs)
    if n_records > MAX_RECORDS:
        return _stop(
            report,
            "too_many_records",
            f"the request covers {inputs['n_rows']} subjects x {len(pairs)} "
            f"edges = {n_records} subject x edge records, above MAX_RECORDS="
            f"{MAX_RECORDS}. Narrow the edge list or the subject set: the "
            "leave-one-out refit grows with both, so nothing was refitted.",
        )

    invalidate: Optional[Tuple[str, str]] = None
    if kind == "glm":
        if inputs["X"].shape[1] != contrast.size:
            raise ValueError(
                f"design_matrix has {inputs['X'].shape[1]} columns but the "
                f"tested contrast has {contrast.size} entries."
            )
        shape_note = f"glm: {inputs['n_rows']} observations"
    elif kind == "two-sample":
        shape_note = (
            f"two-sample: {inputs['n1']} + {inputs['n_rows'] - inputs['n1']} rows"
        )
    else:
        shape_note = f"{kind}: {inputs['n_rows']} rows"
    if int(design["n_subjects"]) != int(inputs["n_rows"]):
        _check(
            report,
            "n_subjects",
            "failed",
            f"the tested model has n_subjects={design['n_subjects']} but the "
            f"supplied inputs carry {inputs['n_rows']} rows.",
        )
        invalidate = (
            "input_rows_mismatch",
            "the supplied inputs do not have the same number of rows as the "
            "tested model; refusing to delete rows from a different sample.",
        )
    else:
        _check(report, "n_subjects", "passed", shape_note)

    coverage = provenance.get("coverage")
    if isinstance(coverage, dict) and invalidate is None:
        recorded_n = None
        supplied_n = None
        if "group1" in coverage and kind != "glm":
            recorded_n = int(coverage["group1"])
            supplied_n = inputs["n1"] if kind == "two-sample" else inputs["n_rows"]
        elif "observations" in coverage and kind == "glm":
            recorded_n = int(coverage["observations"])
            supplied_n = inputs["n_rows"]
        if recorded_n is not None:
            if recorded_n != supplied_n:
                _check(
                    report,
                    "provenance_coverage",
                    "failed",
                    f"provenance records {recorded_n} rows but the inputs carry "
                    f"{supplied_n}.",
                )
                invalidate = (
                    "coverage_mismatch",
                    "the supplied rows disagree with the recorded coverage of "
                    "the tested model.",
                )
            else:
                _check(report, "provenance_coverage", "passed", shape_note)

    if kind == "paired":
        aligned, unavailable, note = _align_paired(
            result, inputs["group2"], ids_group1, ids_group2, paired_inputs_aligned
        )
        report["paired_alignment"] = note
        if unavailable is not None:
            return _stop(report, *unavailable)
        inputs["group2"] = aligned
        _check(report, "paired_alignment", "passed", note["note"])
    if invalidate is not None:
        return _stop(report, *invalidate)

    # ---- The full model on the supplied inputs must reproduce result.beta ----
    edge_cells = ~np.eye(N, dtype=bool)
    matrix = inputs["X"]
    response = inputs["Y"]
    if kind == "glm":
        dense = bool(np.isfinite(matrix).all() and np.isfinite(response).all())
        pairwise_reference = None
    else:
        dense = True
        pairwise_reference = pairwise_estimates(
            inputs["group1"],
            inputs["group2"] if kind != "one-sample" else None,
            kind,
        )[0]
    if kind == "glm" and not dense:
        # Available case: every edge was fitted on its own coverage rows, so a
        # dense pinv would drag one NaN row into every cell. Reproduce beta per
        # edge on exactly those rows instead.
        reproduced = True
        max_diff: Optional[float] = 0.0
        mismatched: List[Tuple[int, int]] = []
        for i, j in pairs:
            y = response[:, i, j]
            mask = np.isfinite(y) & np.isfinite(matrix).all(axis=1)
            if not mask.any():
                continue
            reference = np.linalg.pinv(matrix[mask]) @ y[mask]
            same, difference = _nan_allclose(
                reference, info["beta"][:, i, j], CHECK_TOLERANCE
            )
            if not same:
                reproduced = False
                mismatched.append((int(i), int(j)))
            if difference is not None:
                max_diff = difference if max_diff is None else max(max_diff, difference)
        scope = (
            "on the selected edges, each on its own coverage rows "
            "(available-case inputs contain NaN, so no dense map exists)"
            if reproduced
            else f"on {mismatched}"
        )
    else:
        if kind == "glm":
            reference_map = np.linalg.pinv(matrix) @ response.reshape(
                matrix.shape[0], -1
            )
            reference_map = reference_map.reshape(contrast.size, N, N)
        else:
            reference_map = pairwise_reference
        reproduced, max_diff = _nan_allclose(
            reference_map[:, edge_cells],
            info["beta"][:, edge_cells],
            CHECK_TOLERANCE,
        )
        scope = "on every off-diagonal cell (the diagonal is structurally zero)"
    _check(
        report,
        "beta_reproduced",
        "passed" if reproduced else "failed",
        "the model refitted on the supplied inputs "
        + (
            f"reproduces result.beta {scope}."
            if reproduced
            else f"does not reproduce result.beta {scope}."
        ),
        max_abs_diff=max_diff,
    )
    if not reproduced:
        return _stop(
            report,
            "model_mismatch",
            "inputs do not reproduce the tested model: check the row order, the "
            "subject IDs, the analysis scale (Fisher z already applied) and the "
            "design matrix. Nothing was left out.",
        )

    # ---- Observed statistic: available for verification? ----
    stat_maps = getattr(result, "stat_signed", None)
    observed: Optional[npt.NDArray[np.float64]] = None
    if stat_maps is None:
        _check(
            report,
            "stat_maps",
            "unavailable",
            "the result carries no observed statistic maps (stat_positive/"
            "stat_negative are None); only the beta part is checked.",
        )
    else:
        candidate = np.asarray(stat_maps, dtype=np.float64)
        if candidate.ndim != 2:
            _check(
                report,
                "stat_maps",
                "unavailable",
                "the observed statistic is a parameter grid, not a single map; "
                "select one cell before diagnosing influence. Only the beta "
                "part is checked.",
            )
        elif candidate.shape != (N, N):
            _check(
                report,
                "stat_maps",
                "unavailable",
                f"the observed statistic map has shape {candidate.shape}, not "
                f"({N}, {N}); only the beta part is checked.",
            )
        elif stat_type not in ("tstat", "beta"):
            _check(
                report,
                "stat_maps",
                "unavailable",
                f"stat_type={stat_type!r} is not a signed native statistic; "
                "only the beta part is checked.",
            )
        else:
            observed = candidate

    # ---- Full per-edge fits on exactly the selected edges ----
    marginals: List[Dict[str, Any]] = []
    stat_differences: List[float] = []
    stat_mismatches: List[Tuple[int, int]] = []
    for i, j in pairs:
        arrays, mask = _edge_arrays(kind, inputs, i, j)
        leverage = _edge_leverage(kind, arrays, mask, inputs["X"])
        full = _fit_edge(kind, arrays, mask, inputs["X"], contrast, var_equal)
        mad = _mad(_edge_values(kind, arrays, mask))
        entry: Dict[str, Any] = {
            "edge": [int(i), int(j)],
            "effect": _num(full["effect"]),
            "stat": _num(full["stat"]),
            "n": int(full["n"]),
            "variance": _num(full["variance"]),
            "se2": _num(full["se2"]),
            "variance_group1": _num(full["variance_group1"]),
            "variance_group2": _num(full["variance_group2"]),
            "df": _num(full["df"]),
            "rank": full["rank"],
            "status": full["status"],
            "mad": _num(mad),
            "stat_matches_result": None,
            "max_abs_delta": None,
            "max_abs_delta_subject": None,
        }
        entry.update(_row_coverage_counts(kind, arrays, mask))
        if observed is not None:
            target = entry["stat"] if stat_type == "tstat" else entry["effect"]
            recorded = observed[i, j]
            if target is not None and np.isfinite(recorded):
                difference = abs(float(target) - float(recorded))
                matches = bool(
                    np.isclose(float(target), float(recorded), **cast(Dict[str, Any], CHECK_TOLERANCE))
                )
                entry["stat_matches_result"] = matches
                stat_differences.append(difference)
                if not matches:
                    stat_mismatches.append((int(i), int(j)))
        marginals.append(
            {"edge": (int(i), int(j)), "arrays": arrays, "mask": mask,
             "full": full, "mad": mad, "leverage": leverage, "entry": entry}
        )

    if observed is not None:
        if stat_mismatches:
            _check(
                report,
                "stat_reproduced",
                "failed",
                "the native statistic recomputed on the supplied inputs does "
                f"not match the result's stat map on {stat_mismatches}.",
                max_abs_diff=max(stat_differences) if stat_differences else None,
            )
            return _stop(
                report,
                "statistic_mismatch",
                "the recomputed native statistic does not match the result's "
                "observed statistic at the declared tolerance; the input scale "
                "or the test formula (Welch/Student/df) does not match the "
                "source analysis.",
            )
        if stat_differences:
            _check(
                report,
                "stat_reproduced",
                "passed",
                "the native statistic recomputed on the supplied inputs matches "
                "the result's observed statistic on every selected edge.",
                max_abs_diff=max(stat_differences),
            )
        else:
            _check(
                report,
                "stat_reproduced",
                "unavailable",
                "no finite observed statistic on the selected edges.",
            )

    # ---- Available-case coverage against the recorded NaN pattern ----
    missing_report = getattr(result, "missing_report", None)
    if isinstance(missing_report, dict) and "edge_n" in missing_report:
        edge_n = np.asarray(missing_report["edge_n"], dtype=np.float64)
        recorded_group1 = missing_report.get("edge_n_group1")
        recorded_group2 = missing_report.get("edge_n_group2")
        if recorded_group1 is not None:
            recorded_group1 = np.asarray(recorded_group1, dtype=np.float64)
        if recorded_group2 is not None:
            recorded_group2 = np.asarray(recorded_group2, dtype=np.float64)
        problems: List[str] = []
        for row in marginals:
            i, j = row["edge"]
            supplied = int(row["mask"].sum())
            recorded = edge_n[i, j]
            if not np.isfinite(recorded) or int(recorded) != supplied:
                problems.append(
                    f"({i}, {j}): recorded edge_n={_num(recorded)}, supplied "
                    f"{supplied}"
                )
                continue
            counts = _row_coverage_counts(kind, row["arrays"], row["mask"])
            if (
                recorded_group1 is not None
                and "n_group1" in counts
                and int(recorded_group1[i, j]) != counts["n_group1"]
            ):
                problems.append(
                    f"({i}, {j}): recorded edge_n_group1="
                    f"{int(recorded_group1[i, j])}, supplied "
                    f"{counts['n_group1']}"
                )
            if (
                recorded_group2 is not None
                and "n_group2" in counts
                and int(recorded_group2[i, j]) != counts["n_group2"]
            ):
                problems.append(
                    f"({i}, {j}): recorded edge_n_group2="
                    f"{int(recorded_group2[i, j])}, supplied "
                    f"{counts['n_group2']}"
                )
        if problems:
            _check(
                report,
                "coverage_matches_missing_report",
                "failed",
                "per-edge coverage of the supplied inputs disagrees with the "
                f"recorded available-case coverage: {problems}.",
            )
            return _stop(
                report,
                "coverage_mismatch",
                "the NaN pattern of the supplied inputs does not reproduce the "
                "per-edge coverage recorded for the tested model.",
            )
        _check(
            report,
            "coverage_matches_missing_report",
            "passed",
            "per-edge coverage (edge_n and per-group counts where recorded) "
            "matches the NaN pattern of the supplied inputs.",
        )

    # ---- Leave-one-out over subjects ----
    n_rows = int(inputs["n_rows"])
    per_subject_edge: List[Dict[str, Any]] = []
    subject_rows: List[Dict[str, Any]] = []
    for subject in range(n_rows):
        best_magnitude: Optional[float] = None
        best_edge: Optional[List[int]] = None
        magnitudes: List[float] = []
        counts = {"ok": 0, "unavailable": 0, "not_in_coverage": 0,
                  "rank_loss": 0}
        for row in marginals:
            i, j = row["edge"]
            mask = row["mask"]
            full = row["full"]
            if full["status"] == "nan_edge":
                status = "nan_edge"
                loo_fit_status = "nan_edge"
                loo: Optional[Dict[str, Any]] = None
                delta: Optional[float] = None
            elif not bool(mask[subject]):
                status = "not_in_coverage"
                loo_fit_status = "not_in_coverage"
                loo = None
                delta = 0.0
            else:
                keep = mask.copy()
                keep[subject] = False
                loo = _fit_edge(
                    kind,
                    row["arrays"],
                    keep,
                    inputs["X"],
                    contrast,
                    var_equal,
                    row["full"]["rank"] if kind == "glm" else None,
                )
                loo_fit_status = loo["status"]
                if full["effect"] is not None and loo["effect"] is not None:
                    delta = float(full["effect"]) - float(loo["effect"])
                else:
                    delta = None
                status = loo["status"]
                if (
                    row["mad"] == 0.0
                    and full["status"] != "constant"
                    and status in ("ok", "constant")
                ):
                    status = "zero_mad"
            n_loo = int(full["n"]) if loo is None else int(loo["n"])
            df_loo = full["df"] if loo is None else loo["df"]
            stat_loo = full["stat"] if loo is None else loo["stat"]
            record: Dict[str, Any] = {
                "subject": int(subject),
                "edge": [int(i), int(j)],
                "delta": _num(delta),
                "effect_full": _num(full["effect"]),
                "effect_loo": None if loo is None else _num(loo["effect"]),
                "stat_loo": _num(stat_loo),
                "df_loo": _num(df_loo),
                "n_loo": n_loo,
                "leverage": (
                    _num(row["leverage"][subject]) if bool(mask[subject]) else None
                ),
                "mad": _num(row["mad"]),
                "status": status,
                "loo_fit_status": loo_fit_status,
            }
            if kind == "two-sample":
                source = full if loo is None else loo
                record["n_loo_group1"] = int(source["n_group1"])
                record["n_loo_group2"] = int(source["n_group2"])
            per_subject_edge.append(record)
            if status in ("ok", "zero_mad"):
                counts["ok"] += 1
            elif status == "rank_loss":
                counts["rank_loss"] += 1
            elif status == "not_in_coverage":
                counts["not_in_coverage"] += 1
            else:
                counts["unavailable"] += 1
            # rank_loss rows keep their delta on the record but never rank
            # as "influential": the reduced design is not the tested one.
            if record["delta"] is not None and status != "rank_loss":
                magnitude = abs(record["delta"])
                magnitudes.append(magnitude)
                if best_magnitude is None or magnitude > best_magnitude:
                    best_magnitude = magnitude
                    best_edge = [int(i), int(j)]
                current = row["entry"]["max_abs_delta"]
                if current is None or magnitude > current:
                    row["entry"]["max_abs_delta"] = _num(magnitude)
                    row["entry"]["max_abs_delta_subject"] = int(subject)
        subject_rows.append(
            {
                "subject": int(subject),
                "max_abs_delta": _num(best_magnitude),
                "max_abs_delta_edge": best_edge,
                "mean_abs_delta": (
                    _num(float(np.mean(magnitudes))) if magnitudes else None
                ),
                "n_edges": len(marginals),
                "n_ok": counts["ok"],
                "n_unavailable": counts["unavailable"],
                "n_not_in_coverage": counts["not_in_coverage"],
                "n_rank_loss": counts["rank_loss"],
            }
        )

    report["n_edges"] = len(marginals)
    report["edges"] = [row["entry"] for row in marginals]
    report["per_subject_edge"] = per_subject_edge
    report["subject_summary"] = subject_rows
    ranked = sorted(
        subject_rows,
        key=lambda row: (
            row["max_abs_delta"] is None,
            -(row["max_abs_delta"] or 0.0),
            row["subject"],
        ),
    )
    report["ranking"] = [row["subject"] for row in ranked]
    top = ranked[0] if ranked else None
    report["summary"] = {
        "n_subjects": n_rows,
        "n_edges": len(marginals),
        "n_subject_edge_records": len(per_subject_edge),
        "top_subject_by_max_abs_delta": None if top is None else top["subject"],
        "top_abs_delta": None if top is None else top["max_abs_delta"],
        "top_edge": None if top is None else top["max_abs_delta_edge"],
        "total_abs_delta": _num(
            sum(
                abs(record["delta"])
                for record in per_subject_edge
                if record["delta"] is not None
                and record["status"] != "rank_loss"
            )
        ),
        "n_rank_loss_records": sum(
            1 for record in per_subject_edge if record["status"] == "rank_loss"
        ),
        "ranking_excludes": ["rank_loss"],
    }
    if bool(getattr(result, "harmonized", False)):
        report["limitations"].append(
            "LOO is conditional on the fitted harmonization: ComBat was applied "
            "once upstream and is not refit for each deletion."
        )
    return report
