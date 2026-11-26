"""Descriptive motion (framewise displacement) and QC-FC side-car diagnostics.

This module answers a question that must be asked **before** inference: do the
connectivity values move with head motion, and does that movement depend on
edge length? Everything here is descriptive rank agreement -- Spearman
correlations without p-values, without a null distribution and without a
permutation run -- so it cannot change an inference result and is never used
as a filter. No subject, edge or observation is excluded.

Contract
--------
* **FD** comes only from an explicit phenotype column (``Motion_FD`` or
  ``func_mean_fd`` are auto-detected) or an explicit ``fd=`` vector; an
  explicit ``fd=`` wins over auto-detection. FD is a mean framewise
  displacement per observation in millimetres as declared by the caller. The
  Power vs Jenkinson convention is *not* recorded and is never guessed, and FD
  is never recomputed from ROI time series. Without a source the diagnostic
  reports ``fd_unavailable`` with a reason and fabricates nothing.
* **Alignment** is positional: the FD vector must have one value per row of
  the data's subject axis, in that row order. When ``subject_ids`` are given
  and FD is read from a phenotype frame, the frame's subject column must match
  them exactly (Python equality, no type normalisation, no coercion); a
  mismatch raises instead of silently pairing the wrong rows. Visits are
  separate observations, not independent subjects.
* **Native scale**: the input is described as it was given. ``mean_abs_r`` is
  used only for known Pearson correlations; a known Fisher z is mapped back
  with ``tanh`` and that transform is recorded; unknown or arbitrary units get
  a neutral ``mean_abs_value`` with ``scale: unknown`` and no transform.
* **Denominators**: the NaN share uses every upper-triangle edge
  (``n_rois * (n_rois - 1) / 2``, diagonal excluded); the negative share uses
  the finite edges of that observation. Numerators and denominators are both
  reported.
* **QC-FC**: per-edge Spearman over pairwise-complete observations with at
  least three complete pairs. A constant FD or a constant edge on its complete
  pairs makes that association explicitly ``unavailable`` (counted by cause).
  The group summary reports the distribution of the per-edge associations and
  the subject-level Spearman of mean ``|r|`` against FD.
* **Optional residualisation** runs only when an aligned nuisance design
  matrix (including a non-zero intercept) is passed explicitly. It is a
  **single-equation FC residualisation, not a partial Spearman coefficient**:
  edge values are residualised on the design and then rank-correlated with the
  raw FD (partial Spearman would residualise ranks). Rank and residual df are
  recorded per edge on that edge's complete rows, at least two residual df are
  required, and zero residual variance is reported as unavailable instead of
  being interpreted.
* **Distance** uses known ROI coordinates in matrix order (explicit
  ``coords=`` or ``AtlasInfo.coords``) and a declared space; without
  coordinates the distance part is skipped as ``coordinates_unavailable``.
  Two distinct measures are reported and never mixed: the edge-level
  ``spearman_mean_vs_distance`` (mean edge value vs distance) and the
  ``qc_fc_vs_distance`` (per-edge edge-FD association vs distance) with its
  own n and status. The existing post-inference ``distance_qc`` of
  :mod:`conninfpy.interpret.annotate` is untouched and independent.

Everything is descriptive: no p-values, no causal reading, and the share of
negative edges says nothing about global signal regression.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, cast

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.spatial.distance import pdist

__all__ = [
    "FD_COLUMN_CANDIDATES",
    "FD_UNITS",
    "MIN_COMPLETE_PAIRS",
    "MIN_RESIDUAL_DF",
    "detect_fd_column",
    "resolve_fd",
    "motion_qc_fc",
    "motion_qc_fc_paired",
]

SCHEMA_VERSION = 1
FD_COLUMN_CANDIDATES = ("Motion_FD", "func_mean_fd")
FD_UNITS = "mm"
MIN_COMPLETE_PAIRS = 3
MIN_RESIDUAL_DF = 2

FD_CONVENTION_NOTE = (
    "Framewise-displacement convention (Power vs Jenkinson) is not recorded in "
    "the phenotype column and is never guessed; FD is taken as the "
    "caller-declared mean framewise displacement in millimetres."
)
NO_PVALUES_NOTE = (
    "All associations here are descriptive Spearman rank agreement with no "
    "p-values, no null distribution and no permutation run."
)
NO_GSR_NOTE = (
    "The share of negative edges is a descriptive property of the input matrix; "
    "it is not evidence for or against global signal regression."
)
NO_FILTER_NOTE = (
    "No subject, edge or observation is excluded or weighted by this "
    "diagnostic; inference is unaffected."
)
NATIVE_INPUT_NOTE = (
    "Described on the native input scale before Fisher transformation, "
    "ComBat, harmonisation or any inference."
)
DISTANCE_SEPARATION_NOTE = (
    "The post-inference distance_qc of conninfpy.interpret.annotate is a "
    "separate statistic-versus-distance summary over tested edges and is not "
    "changed or reused here."
)

_SCALE_LABELS = {
    "pearson_r": "pearson_r",
    "r": "pearson_r",
    "correlation": "pearson_r",
    "fisher_z": "fisher_z",
    "z": "fisher_z",
    "unknown": "unknown",
}
_MEAN_LABELS = {"pearson_r": "mean_abs_r", "fisher_z": "mean_abs_r", "unknown": "mean_abs_value"}
_TRANSFORMS = {
    "pearson_r": "none (input is Pearson r)",
    "fisher_z": "tanh (documented inverse Fisher z; mean |tanh(z)| reported as mean_abs_r)",
    "unknown": "none (input scale unknown; no |r| label and no transform)",
}


# ----------------------------------------------------------------------
# Small numeric helpers
# ----------------------------------------------------------------------
def _finite(value: Any) -> Optional[float]:
    """Finite Python float, else ``None`` (``nan``/``inf`` never leak out)."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _finite_list(values: Any) -> List[Optional[float]]:
    return [_finite(v) for v in np.asarray(values, dtype=np.float64).ravel()]


def _describe(values: Any) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None,
                "min": None, "max": None, "n_unique": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_unique": int(np.unique(arr).size),
    }


def _quantiles(values: Any) -> Dict[str, Optional[float]]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    keys = ("min", "q10", "q25", "median", "q75", "q90", "max")
    if arr.size == 0:
        return {key: None for key in keys}
    q = np.quantile(arr, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {key: float(val) for key, val in zip(keys, q, strict=True)}


def _spearman(a: Any, b: Any, *, min_n: int = MIN_COMPLETE_PAIRS) -> Dict[str, Any]:
    """Descriptive Spearman of two vectors; ``None`` when it is not defined."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"Spearman inputs must have equal length; got {x.size} and {y.size}.")
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < min_n:
        return {"rho": None, "n": n, "status": "insufficient_data",
                "reason": f"fewer than {min_n} complete pairs"}
    x, y = x[ok], y[ok]
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return {"rho": None, "n": n, "status": "constant",
                "reason": "one of the vectors is constant on the complete pairs"}
    rho = float(stats.spearmanr(x, y).statistic)
    if not np.isfinite(rho):
        return {"rho": None, "n": n, "status": "unavailable",
                "reason": "Spearman coefficient is not finite"}
    return {"rho": rho, "n": n, "status": "available", "reason": None}


def _canonical_scale(scale: Optional[str]) -> str:
    if scale is None:
        return "unknown"
    key = str(scale).strip().lower()
    if key not in _SCALE_LABELS:
        raise ValueError(
            "scale must be one of 'r'/'pearson_r'/'correlation', 'z'/'fisher_z' "
            f"or 'unknown'/None; got {scale!r}."
        )
    return _SCALE_LABELS[key]


# ----------------------------------------------------------------------
# FD resolution
# ----------------------------------------------------------------------
def _pheno_columns(pheno: Any) -> List[Any]:
    """Column labels of a phenotype frame (``[]`` when there is no frame)."""
    columns = getattr(pheno, "columns", None)
    return [] if columns is None else list(columns)


def detect_fd_column(pheno: Any) -> Optional[str]:
    """The FD column of ``pheno`` using the strict candidate list, else ``None``."""
    columns = _pheno_columns(pheno)
    for name in FD_COLUMN_CANDIDATES:
        if name in columns:
            return name
    return None


def _pheno_subject_column(pheno: Any, explicit: Optional[str] = None) -> Optional[str]:
    columns = _pheno_columns(pheno)
    if explicit is not None:
        if explicit not in columns:
            raise ValueError(f"pheno_subject_column={explicit!r} is not a column of the phenotype frame.")
        return explicit
    for name in ("subject_id", "sub_id", "subject", "participant_id", "FILE_ID"):
        if name in columns:
            return name
    return None


def _pheno_values(pheno: Any, column: str) -> npt.NDArray[np.float64]:
    import pandas as pd

    series = pheno[column]
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def _resolve_fd_vector(value: Any, pheno: Any, n: Optional[int], name: str) -> npt.NDArray[np.float64]:
    """A vector argument: numeric, 1-D and one value per subject-axis row."""
    if pheno is not None and isinstance(value, str):
        columns = _pheno_columns(pheno)
        if value not in columns:
            raise ValueError(f"{name}={value!r} is not a column of the phenotype frame.")
        arr = _pheno_values(pheno, value)
    else:
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric vector or a phenotype column name.") from exc
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional; got shape {arr.shape}.")
    if n is not None and arr.size != n:
        raise ValueError(
            f"{name} has {arr.size} value(s) but the subject axis has {n} row(s); "
            "FD must align positionally with the data's axis 0."
        )
    return arr


def _compare_ids(pheno_ids: Sequence[Any], subject_ids: Sequence[Any]) -> List[int]:
    """Indices whose raw Python values differ; no normalisation or coercion."""
    mismatched: List[int] = []
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a hard dependency
        pd = None
    for idx, (left, right) in enumerate(zip(pheno_ids, subject_ids, strict=False)):
        if pd is not None and (pd.isna(left) or pd.isna(right)):
            mismatched.append(idx)
            continue
        try:
            equal = bool(left == right)
        except (TypeError, ValueError):
            equal = False
        if not equal:
            mismatched.append(idx)
    return mismatched


def resolve_fd(
    fd: Any = None,
    *,
    pheno: Any = None,
    n: Optional[int] = None,
    subject_ids: Optional[Sequence[Any]] = None,
    pheno_subject_column: Optional[str] = None,
    units: str = FD_UNITS,
    source_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve the FD source and describe it honestly.

    Returns a block with ``status`` (``available``/``unavailable``), the
    ``values`` (or ``None``), the ``source`` record, units, the unrecorded
    convention note and the ID-alignment state. An explicit ``fd=`` wins over
    the auto-detected ``Motion_FD``/``func_mean_fd`` column. Without a source
    the status is ``unavailable`` with a reason and nothing is fabricated.
    """
    block: Dict[str, Any] = {
        "status": "unavailable",
        "reason": None,
        "values": None,
        "source": None,
        "units": units,
        "units_source": "declared by the caller",
        "convention": "unknown",
        "convention_note": FD_CONVENTION_NOTE,
        "id_alignment": "not_applicable",
        "id_alignment_note": None,
        "n": int(n) if n is not None else None,
        "n_finite": 0,
        "n_nonfinite": 0,
        "summary": _describe([]),
    }
    if fd is None and pheno is None:
        block["reason"] = (
            "no FD source: pass fd= (a phenotype column name or a per-observation "
            "vector) or a phenotype frame with Motion_FD/func_mean_fd"
        )
        return block
    if isinstance(fd, str) and pheno is None:
        raise ValueError(
            f"fd={fd!r} is a column name, so the phenotype frame must be passed as "
            "pheno=; a column name alone cannot be resolved."
        )
    source: Dict[str, Any]
    if fd is None:
        detected = detect_fd_column(pheno)
        if detected is None:
            columns = [str(c) for c in _pheno_columns(pheno)]
            candidates = [c for c in columns if any(tok in c.lower() for tok in ("motion", "fd", "framewise"))]
            block["reason"] = (
                "no auto-detected FD column: expected exactly one of "
                f"{list(FD_COLUMN_CANDIDATES)} in the phenotype frame"
                + (f"; motion-like columns present (not used): {candidates}" if candidates else "")
            )
            return block
        values = _pheno_values(pheno, detected)
        source = {"kind": "pheno_column", "name": detected,
                  "detail": "auto-detected from the strict candidate list"}
    elif pheno is not None and isinstance(fd, str):
        columns = _pheno_columns(pheno)
        if fd not in columns:
            raise ValueError(f"fd={fd!r} is not a column of the phenotype frame.")
        values = _pheno_values(pheno, fd)
        source = {"kind": "pheno_column", "name": fd, "detail": "explicit column name"}
    else:
        values = _resolve_fd_vector(fd, None, n, "fd")
        source = {"kind": "vector", "name": source_name,
                  "detail": "explicit per-observation vector in subject-axis order"
                            + (f", sliced from phenotype column {source_name!r}"
                               if source_name else "")}

    if n is not None and values.size != n:
        raise ValueError(
            f"FD has {values.size} value(s) but the subject axis has {n} row(s); no "
            "value was trimmed, repeated or filled."
        )

    alignment = "positional"
    alignment_note = (
        "FD is aligned positionally with the data's subject axis (one value per "
        "row, same order as the phenotype rows used to select the observations)."
    )
    if subject_ids is not None:
        wanted = list(subject_ids)
        if len(wanted) != values.size:
            raise ValueError(
                f"subject_ids has {len(wanted)} value(s) but FD describes {values.size} "
                "observation(s); align them before requesting diagnostics."
            )
        if source["kind"] == "pheno_column":
            column = _pheno_subject_column(pheno, pheno_subject_column)
            if column is None:
                alignment = "not_verified"
                alignment_note = (
                    "subject_ids were supplied but the phenotype frame has no "
                    "recognised subject-ID column; FD rows are therefore positional "
                    "and their identity is not verified."
                )
            else:
                pheno_ids = list(pheno[column].to_numpy(dtype=object))
                if len(pheno_ids) != len(wanted):
                    raise ValueError(
                        f"phenotype subject column {column!r} has {len(pheno_ids)} "
                        f"row(s) but subject_ids has {len(wanted)}."
                    )
                mismatched = _compare_ids(pheno_ids, wanted)
                if mismatched:
                    shown = ", ".join(
                        f"row {k}: {pheno_ids[k]!r} != {wanted[k]!r}" for k in mismatched[:5]
                    )
                    raise ValueError(
                        f"FD phenotype rows do not match subject_ids in {len(mismatched)} "
                        f"row(s) ({shown}); IDs are compared exactly, without "
                        "normalisation or type coercion."
                    )
                alignment = "exact_id_match"
                alignment_note = (
                    f"FD phenotype rows were matched exactly by {column!r} against "
                    f"subject_ids ({len(wanted)} observations, no normalisation)."
                )
        else:
            alignment = "positional_vector"
            alignment_note = (
                "subject_ids were supplied with an explicit FD vector; the vector is "
                "assumed to follow that order and was not reordered or matched by value."
            )

    finite = np.isfinite(values)
    block.update({
        "status": "available",
        "values": values,
        "source": source,
        "id_alignment": alignment,
        "id_alignment_note": alignment_note,
        "n": int(values.size),
        "n_finite": int(finite.sum()),
        "n_nonfinite": int((~finite).sum()),
        "summary": _describe(values),
    })
    return block


# ----------------------------------------------------------------------
# Edge-level descriptive machinery
# ----------------------------------------------------------------------
def _edge_matrix(Y: Any) -> npt.NDArray[np.float64]:
    arr = np.asarray(Y, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"Y must be (n_observations, n_rois, n_rois); got {arr.shape}.")
    iu = np.triu_indices(arr.shape[1], 1)
    return arr[:, iu[0], iu[1]]


def _value_map(edges: npt.NDArray[np.float64], scale: str) -> npt.NDArray[np.float64]:
    """Native values used for |r|/sign summaries (tanh only for known Fisher z)."""
    finite = np.isfinite(edges)
    if scale == "fisher_z":
        return np.where(finite, np.tanh(np.where(finite, edges, 0.0)), np.nan)
    return np.where(finite, edges, np.nan)


def _roi_nonfinite(Y: Any) -> Dict[str, Any]:
    """Per-ROI NaN / Inf / finite counts over its non-diagonal edges.

    The denominator is ``n_observations * (n_rois - 1)``: every stored matrix
    entry of that ROI against another ROI and every observation. NaN and Inf
    are counted separately (a correlation backend may emit either) and the
    combined non-finite share is reported alongside; nothing is dropped.
    """
    arr = np.asarray(Y, dtype=np.float64)
    n, n_rois = arr.shape[0], arr.shape[1]
    off = ~np.eye(n_rois, dtype=bool)
    nan_count = np.zeros(n_rois, dtype=np.int64)
    inf_count = np.zeros(n_rois, dtype=np.int64)
    finite_count = np.zeros(n_rois, dtype=np.int64)
    for roi in range(n_rois):
        vals = arr[:, roi, off[roi]]
        nan_count[roi] = int(np.isnan(vals).sum())
        inf_count[roi] = int(np.isinf(vals).sum())
        finite_count[roi] = int(np.isfinite(vals).sum())
    denominator = int(n * (n_rois - 1))
    nonfinite_count = nan_count + inf_count
    share = np.full(n_rois, np.nan, dtype=np.float64)
    if denominator > 0:
        share = nonfinite_count / float(denominator)
    return {
        "roi_index": list(range(n_rois)),
        "roi_id": None,
        "n_observations": int(n),
        "n_edges_per_roi": int(n_rois - 1),
        "denominator": denominator,
        "denominator_note": (
            "denominator = n_observations * (n_rois - 1): all stored non-diagonal "
            "entries incident to the ROI; the diagonal is excluded."
        ),
        "nan_count": [int(v) for v in nan_count],
        "inf_count": [int(v) for v in inf_count],
        "nonfinite_count": [int(v) for v in nonfinite_count],
        "finite_count": [int(v) for v in finite_count],
        "nan_share": _finite_list(np.divide(nan_count, denominator, out=np.full(n_rois, np.nan), where=denominator > 0)),
        "inf_share": _finite_list(np.divide(inf_count, denominator, out=np.full(n_rois, np.nan), where=denominator > 0)),
        "nonfinite_share": _finite_list(share),
        "note": (
            "NaN and Inf are counted separately; a combined non-finite share is "
            "reported as well. This is descriptive and excludes nothing."
        ),
    }


def _subject_statistics(Y: Any, scale: str) -> Dict[str, Any]:
    """Per-observation mean |value|, NaN share and negative share with denominators."""
    edges = _edge_matrix(Y)
    n, m = edges.shape
    values = _value_map(edges, scale)
    finite = np.isfinite(values)
    n_finite = finite.sum(axis=1)
    nan_count = m - n_finite
    nan_share = np.full(n, np.nan, dtype=np.float64)
    if m > 0:
        nan_share = nan_count / float(m)
    mean_abs = np.full(n, np.nan, dtype=np.float64)
    np.divide(
        np.where(finite, np.abs(values), 0.0).sum(axis=1), n_finite,
        out=mean_abs, where=n_finite > 0,
    )
    neg_count = np.where(finite, values < 0.0, False).sum(axis=1)
    neg_share = np.full(n, np.nan, dtype=np.float64)
    np.divide(neg_count, n_finite, out=neg_share, where=n_finite > 0)
    return {
        "n_edges": int(m),
        "n_edges_total": int(m),
        "edges": edges,
        "values": values,
        "mean_abs": mean_abs,
        "nan_count": nan_count,
        "nan_share": nan_share,
        "negative_count": neg_count,
        "negative_share": neg_share,
        "n_finite_edges": n_finite,
    }


def _edge_associations(
    edges: npt.NDArray[np.float64],
    fd: npt.NDArray[np.float64],
    *,
    min_pairs: int = MIN_COMPLETE_PAIRS,
    chunk: int = 512,
) -> Dict[str, Any]:
    """Per-edge Spearman(edge, FD) over pairwise-complete observations.

    Fully vectorised over edge chunks; ties get average ranks (Spearman).
    An edge with fewer than ``min_pairs`` complete pairs, a constant FD or a
    constant edge on its complete pairs gets an explicit status instead of a
    number.
    """
    import pandas as pd

    n, m = edges.shape
    fd = np.asarray(fd, dtype=np.float64).ravel()
    if fd.size != n:
        raise ValueError(f"FD has {fd.size} value(s) but the data has {n} observation(s).")
    rho = np.full(m, np.nan, dtype=np.float64)
    n_pairs = np.zeros(m, dtype=np.int64)
    status = np.empty(m, dtype=object)
    status[:] = "unavailable"
    fd_finite = np.isfinite(fd)
    for start in range(0, m, max(1, int(chunk))):
        stop = min(m, start + max(1, int(chunk)))
        block = edges[:, start:stop]
        mask = np.isfinite(block) & fd_finite[:, None]
        n_j = mask.sum(axis=0)
        # Constant-on-complete-pairs flags without all-NaN warnings.
        with np.errstate(invalid="ignore"):
            e_max = np.where(mask, block, -np.inf).max(axis=0)
            e_min = np.where(mask, block, np.inf).min(axis=0)
            f_max = np.where(mask, fd[:, None], -np.inf).max(axis=0)
            f_min = np.where(mask, fd[:, None], np.inf).min(axis=0)
        const_edge = e_max == e_min
        const_fd = f_max == f_min
        x = np.where(mask, block, np.nan)
        y = np.where(mask, fd[:, None], np.nan)
        # DataFrame.rank keeps NaN and averages ties, which is Spearman's rank.
        rx = pd.DataFrame(x).rank(axis=0, method="average", na_option="keep").to_numpy(dtype=np.float64)
        ry = pd.DataFrame(y).rank(axis=0, method="average", na_option="keep").to_numpy(dtype=np.float64)
        nn = n_j.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            sx = np.nansum(rx, axis=0)
            sy = np.nansum(ry, axis=0)
            sxx = np.nansum(rx * rx, axis=0)
            syy = np.nansum(ry * ry, axis=0)
            sxy = np.nansum(rx * ry, axis=0)
            vx = sxx - sx * sx / nn
            vy = syy - sy * sy / nn
            num = sxy - sx * sy / nn
            den = np.sqrt(vx * vy)
            r = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        enough = n_j >= min_pairs
        block_status = np.where(~enough, "insufficient_pairs",
                                np.where(const_fd, "constant_fd",
                                         np.where(const_edge, "constant_edge",
                                                  np.where(np.isfinite(r), "ok", "unavailable"))))
        seg = slice(start, stop)
        rho[seg] = np.where(block_status == "ok", r, np.nan)
        n_pairs[seg] = n_j
        status[seg] = block_status
    valid = status == "ok"
    counts = {name: int((status == name).sum())
              for name in ("ok", "insufficient_pairs", "constant_fd", "constant_edge", "unavailable")}
    return {
        "rho": rho,
        "n_pairs": n_pairs,
        "status": status,
        "n_edges": int(m),
        "n_valid": int(valid.sum()),
        "counts": counts,
        "min_complete_pairs": int(min_pairs),
        "quantiles": _quantiles(rho[valid]),
        "note": NO_PVALUES_NOTE,
    }


def _residualize(
    edges: npt.NDArray[np.float64],
    nuisance: Any,
    *,
    names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Residualise the raw edge values on an explicit nuisance design.

    This is a **single-equation FC residualisation, not a partial Spearman
    coefficient**: the edge values are residualised on the nuisance design and
    then rank-correlated with the *raw* FD, while a partial Spearman would
    require residualising ranks. The distinction is recorded in the result.

    The matrix must include a non-zero intercept column. Rank and residual df
    are evaluated on the rows actually complete for each edge (the shared
    complete rows when they coincide), at least :data:`MIN_RESIDUAL_DF`
    residual df are required per edge, and zero-variance residual edges are
    reported as unavailable rather than interpreted.
    """
    X = np.asarray(nuisance, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(f"nuisance must be (n_observations, n_columns); got {X.shape}.")
    n = edges.shape[0]
    if X.shape[0] != n:
        raise ValueError(
            f"nuisance has {X.shape[0]} row(s) but the data has {n} observation(s); "
            "pass the aligned design matrix, not a phenotype frame."
        )
    finite_rows = np.isfinite(X).all(axis=1)
    n_used = int(finite_rows.sum())
    rank = int(np.linalg.matrix_rank(X[finite_rows])) if n_used else 0
    residual_df = int(n_used - rank)
    constant_cols = [k for k in range(X.shape[1])
                     if n_used and float(np.ptp(X[finite_rows, k])) == 0.0]
    intercept_present = any(
        float(np.any(X[finite_rows, k] != 0.0)) for k in constant_cols
    )
    columns = list(names) if names is not None else [f"nuisance_{k + 1}" for k in range(X.shape[1])]
    block: Dict[str, Any] = {
        "status": "unavailable",
        "reason": None,
        "interpretation": "fc_only_residualization_not_partial_spearman",
        "partial_spearman": False,
        "interpretation_note": (
            "Edge values are residualised on the nuisance design and then rank-correlated "
            "with the raw FD. This is NOT a partial Spearman coefficient (partial "
            "Spearman would residualise the ranks of edge and FD on the nuisance ranks)."
        ),
        "n_columns": int(X.shape[1]),
        "column_names": [str(c) for c in columns],
        "matrix_rank": rank,
        "residual_df": residual_df,
        "n_rows_used": n_used,
        "n_rows_dropped_nonfinite": int(n - n_used),
        "intercept_present": intercept_present,
        "intercept_columns": [columns[k] for k in constant_cols if float(np.any(X[finite_rows, k] != 0.0))],
        "intercept_note": (
            "A non-zero constant column (intercept) is present."
            if intercept_present
            else "No non-zero constant column was detected; the matrix must include an "
                 "intercept and none is added automatically."
        ),
        "residual_df_min": None,
        "residual_df_max": None,
        "matrix_rank_min": None,
        "matrix_rank_max": None,
        "n_edges_insufficient_df": 0,
        "n_edges_zero_residual_variance": 0,
        "n_edges_residualized": 0,
        "mask_kind": None,
    }
    if not intercept_present:
        block["reason"] = (
            "the nuisance matrix must include a non-zero intercept column; without it "
            "the residualisation does not remove the mean and is not applied"
        )
        return block
    if residual_df < MIN_RESIDUAL_DF:
        block["reason"] = (
            f"residual df = {residual_df} < {MIN_RESIDUAL_DF} on the shared complete "
            "rows: the nuisance design leaves too little residual variation"
        )
        return block

    rows_idx = np.flatnonzero(finite_rows)
    residual = np.full_like(edges, np.nan)
    df_by_edge = np.full(edges.shape[1], np.nan, dtype=np.float64)
    rank_by_edge = np.full(edges.shape[1], np.nan, dtype=np.float64)
    finite_here = np.isfinite(edges[rows_idx])
    full_edges = finite_here.all(axis=0)
    cols_full = np.flatnonzero(full_edges)
    if cols_full.size:
        sub = edges[np.ix_(rows_idx, cols_full)]
        beta, *_ = np.linalg.lstsq(X[rows_idx], sub, rcond=None)
        residual[np.ix_(rows_idx, cols_full)] = sub - X[rows_idx] @ beta
        df_by_edge[cols_full] = residual_df
        rank_by_edge[cols_full] = rank
    partial = np.flatnonzero(~full_edges)
    for j in partial:
        rows_j = finite_here[:, j]
        n_j = int(rows_j.sum())
        if n_j <= 0:
            continue
        cols = rows_idx[rows_j]
        r_j = int(np.linalg.matrix_rank(X[cols]))
        df_j = int(n_j - r_j)
        if df_j < MIN_RESIDUAL_DF:
            continue
        beta, *_ = np.linalg.lstsq(X[cols], edges[cols, j], rcond=None)
        residual[cols, j] = edges[cols, j] - X[cols] @ beta
        df_by_edge[j] = df_j
        rank_by_edge[j] = r_j
    block["mask_kind"] = "shared_complete_rows" if partial.size == 0 else "per_edge_complete_rows"
    block["n_edges_insufficient_df"] = int(np.isnan(df_by_edge).sum())
    usable_df = df_by_edge[np.isfinite(df_by_edge)]
    if usable_df.size:
        block["residual_df_min"] = int(np.nanmin(usable_df))
        block["residual_df_max"] = int(np.nanmax(usable_df))
        ranks = rank_by_edge[np.isfinite(rank_by_edge)]
        block["matrix_rank_min"] = int(np.min(ranks))
        block["matrix_rank_max"] = int(np.max(ranks))
    block["n_edges_residualized"] = int(np.isfinite(residual).any(axis=0).sum())
    block["n_edges_skipped"] = int((~np.isfinite(residual).any(axis=0)).sum())

    finite_res = np.isfinite(residual)
    with np.errstate(invalid="ignore"):
        r_max = np.where(finite_res, residual, -np.inf).max(axis=0)
        r_min = np.where(finite_res, residual, np.inf).min(axis=0)
    zero_var = (finite_res.sum(axis=0) >= 2) & (r_max == r_min)
    block["n_edges_zero_residual_variance"] = int(zero_var.sum())
    interpretable = (finite_res.sum(axis=0) >= 2) & ~zero_var
    if not interpretable.any():
        block["status"] = "unavailable"
        block["reason"] = (
            "no residual edge has both at least two residual df and non-zero residual "
            "variance; the adjusted association is not interpretable"
        )
        return block
    block["status"] = "applied"
    block["n_edges_interpretable"] = int(interpretable.sum())
    block["residuals"] = residual  # internal only; popped before serialisation
    return block


def _resolve_coords(
    coords: Any,
    atlas: Any,
    *,
    n_rois: int,
    space: Optional[str],
) -> Dict[str, Any]:
    """Known ROI coordinates for the matrix order plus their declared space."""
    source: Optional[str] = None
    arr: Optional[npt.NDArray[np.float64]] = None
    if coords is not None:
        arr = np.asarray(coords, dtype=np.float64)
        source = "explicit coords="
    elif atlas is not None and getattr(atlas, "coords", None) is not None:
        arr = np.asarray(atlas.coords, dtype=np.float64)
        source = "AtlasInfo.coords"
    block: Dict[str, Any] = {
        "status": "coordinates_unavailable",
        "reason": None,
        "coordinates_source": source,
        "space": space,
        "space_declared": space is not None,
        "units": "mm",
        "n_rois": int(n_rois),
        "distances": None,
    }
    if arr is None:
        block["reason"] = (
            "no known ROI coordinates: pass coords= (n_rois, 3) in matrix order or an "
            "AtlasInfo that ships centroids; no coordinate is guessed"
        )
        return block
    if arr.shape != (n_rois, 3):
        raise ValueError(
            f"ROI coordinates must have shape ({n_rois}, 3) in connectivity-matrix "
            f"order; got {arr.shape}."
        )
    if not np.isfinite(arr).all():
        block["reason"] = "ROI coordinates contain non-finite values; distances are not computable"
        return block
    if space is None:
        block["space_note"] = (
            "The coordinate space was not declared by the caller. The coordinates are "
            "used as given (millimetre centroids in the atlas's own grid when they come "
            "from AtlasInfo); the template variant is not guessed."
        )
    distances = pdist(arr)  # condensed upper triangle, same order as triu_indices
    block.update({"status": "available", "distances": distances})
    if source == "AtlasInfo.coords":
        block["space_note"] = (
            "Coordinates are AtlasInfo MNI-millimetre centroids in matrix order; the "
            "template variant is the atlas's own and is not re-declared."
            if space is None else block.get("space_note")
        )
    return block


def _distance_association(
    edges: npt.NDArray[np.float64],
    coords_block: Mapping[str, Any],
    *,
    value_label: str,
    per_edge_rho: Optional[npt.NDArray[np.float64]] = None,
    per_edge_status: Optional[npt.NDArray[Any]] = None,
) -> Dict[str, Any]:
    """Two distinct descriptive distance measures (never mixed together).

    ``spearman_mean_vs_distance`` relates the **edge level** (mean edge value
    over observations) to distance; it is not the dependence of the QC-FC
    association on distance. ``qc_fc_vs_distance`` relates the **per-edge
    edge-FD association** to distance and is the descriptive answer to "is
    QC-FC stronger for short or long edges?". Both keep their own n and status.
    """
    block = dict(coords_block)
    block.pop("distances", None)
    block["edge_mean"] = value_label
    block["note"] = DISTANCE_SEPARATION_NOTE
    block["measures_note"] = (
        "spearman_mean_vs_distance: Spearman(mean edge value over observations, distance) "
        "— an edge-level measure, NOT the distance dependence of QC-FC. "
        "qc_fc_vs_distance: Spearman(per-edge Spearman(edge, FD), distance) — the "
        "distance dependence of the edge-level motion association."
    )
    block["spearman_mean_vs_distance_measure"] = (
        "Spearman(mean edge value over observations, distance)"
    )
    if coords_block.get("status") != "available":
        block["qc_fc_vs_distance"] = _qc_fc_distance_association(
            None, per_edge_rho, per_edge_status
        )
        block["qc_fc_vs_distance"]["status"] = "coordinates_unavailable"
        block["qc_fc_vs_distance"]["reason"] = "no known ROI coordinates"
        block["qc_fc_vs_distance"]["n_edges_total"] = (
            None if per_edge_rho is None else int(len(per_edge_rho))
        )
        return block
    distances = np.asarray(coords_block["distances"], dtype=np.float64)
    finite = np.isfinite(edges)
    n_finite = finite.sum(axis=0)
    mean_value = np.full(edges.shape[1], np.nan, dtype=np.float64)
    np.divide(
        np.where(finite, edges, 0.0).sum(axis=0), n_finite,
        out=mean_value, where=n_finite > 0,
    )
    assoc = _spearman(mean_value, distances)
    block.update({
        "n_edges_total": int(edges.shape[1]),
        "n_edges_with_mean": int(np.isfinite(mean_value).sum()),
        "spearman_mean_vs_distance": assoc["rho"],
        "n_edges_used": assoc["n"],
        "association_status": assoc["status"],
        "association_reason": assoc["reason"],
    })
    if assoc["status"] != "available":
        block["status"] = "unavailable"
        block["reason"] = assoc["reason"]
    block["qc_fc_vs_distance"] = _qc_fc_distance_association(
        distances, per_edge_rho, per_edge_status
    )
    return block


def _qc_fc_distance_association(
    distances: Optional[npt.NDArray[np.float64]],
    per_edge_rho: Optional[npt.NDArray[np.float64]],
    per_edge_status: Optional[npt.NDArray[Any]],
) -> Dict[str, Any]:
    """Spearman(per-edge edge-FD association, distance) with its own status."""
    block: Dict[str, Any] = {
        "measure": "Spearman(per-edge Spearman(edge, FD), distance)",
        "method": "spearman",
        "p_values": "none (descriptive)",
        "status": "unavailable",
        "reason": None,
        "rho": None,
        "n_pairs": 0,
        "n_edges_total": None if distances is None else int(np.asarray(distances).size),
        "n_edges_with_association": 0,
    }
    if distances is None:
        block["reason"] = "no known ROI coordinates"
        return block
    if per_edge_rho is None or per_edge_status is None:
        block["reason"] = (
            "per-edge edge-FD associations are unavailable (no FD source or FD constant), "
            "so their distance dependence cannot be described"
        )
        return block
    valid = (np.asarray(per_edge_status, dtype=object) == "ok") & np.isfinite(per_edge_rho)
    block["n_edges_with_association"] = int(valid.sum())
    assoc = _spearman(np.asarray(per_edge_rho)[valid], np.asarray(distances)[valid])
    block.update({"rho": assoc["rho"], "n_pairs": assoc["n"],
                  "status": assoc["status"], "reason": assoc["reason"]})
    return block


# ----------------------------------------------------------------------
# Public diagnostics
# ----------------------------------------------------------------------
def motion_qc_fc(
    Y: Any,
    *,
    fd: Any = None,
    pheno: Any = None,
    subject_ids: Optional[Sequence[Any]] = None,
    condition_labels: Optional[Sequence[Any]] = None,
    condition: Optional[str] = None,
    index_values: Optional[Sequence[Any]] = None,
    scale: Optional[str] = None,
    scale_source: Optional[str] = None,
    nuisance: Any = None,
    nuisance_names: Optional[Sequence[str]] = None,
    coords: Any = None,
    atlas: Any = None,
    space: Optional[str] = None,
    roi_ids: Optional[Sequence[Any]] = None,
    pheno_subject_column: Optional[str] = None,
    fd_units: str = FD_UNITS,
    fd_source_name: Optional[str] = None,
    unit_of_observation: str = "subject",
    input_stage: str = "native input before preprocessing",
) -> Dict[str, Any]:
    """Descriptive motion / QC-FC diagnostics for one observation set.

    Parameters
    ----------
    Y : ndarray of shape (n_observations, n_rois, n_rois)
        Connectivity matrices on their native scale, in observation order.
    fd : str or array-like, optional
        FD source: a phenotype column name or a per-observation vector in
        ``Y``'s axis-0 order. Wins over auto-detection.
    pheno : pandas.DataFrame, optional
        Phenotype frame with the FD column and (optionally) subject IDs.
    subject_ids : sequence, optional
        The dataset's subject IDs for ``Y``'s axis 0; used to verify that FD
        rows belong to the same observations.
    condition_labels : sequence, optional
        Per-observation condition label, recorded in the per-observation table.
    scale : {'r', 'z', 'unknown'} or None
        Known native input scale. ``'r'`` labels the mean ``mean_abs_r``;
        ``'z'`` applies the documented ``tanh`` and labels it ``mean_abs_r``;
        ``None``/``'unknown'`` keeps a neutral ``mean_abs_value`` without a
        transform.
    nuisance : ndarray, optional
        Aligned nuisance design matrix **including an intercept**. Only an
        explicit matrix enables residualisation; without it the diagnostic
        records ``adjustment: none``.
    coords, atlas : array or AtlasInfo, optional
        Known ROI coordinates in matrix order (explicit ``coords`` wins).
    space : str, optional
        Declared coordinate space; never guessed.

    Returns
    -------
    dict
        Descriptive diagnostics with ``status`` (``available``,
        ``fd_unavailable``, ``constant_fd``, ``insufficient_data``,
        ``unavailable``), the explicit ``unavailable`` reasons, the FD source
        provenance, the per-observation table with denominators, the QC-FC
        summary, the adjustment state and the distance association.
    """
    arr = np.asarray(Y, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"Y must be (n_observations, n_rois, n_rois); got {arr.shape}.")
    n, n_rois = arr.shape[0], arr.shape[1]
    canonical = _canonical_scale(scale)
    fd_block = resolve_fd(
        fd, pheno=pheno, n=n, subject_ids=subject_ids,
        pheno_subject_column=pheno_subject_column, units=fd_units,
        source_name=fd_source_name,
    )
    stats_block = _subject_statistics(arr, canonical)
    edges = stats_block["edges"]
    mean_label = _MEAN_LABELS[canonical]

    notes: List[str] = [NO_PVALUES_NOTE, NO_FILTER_NOTE, NATIVE_INPUT_NOTE, NO_GSR_NOTE]
    unavailable: List[Dict[str, Any]] = []

    per_edge_rho: Optional[npt.NDArray[np.float64]] = None
    per_edge_status: Optional[npt.NDArray[Any]] = None
    assoc: Optional[Dict[str, Any]] = None
    qc_fc: Dict[str, Any] = {
        "status": "unavailable",
        "reason": None,
        "method": "spearman (descriptive, no p-values)",
        "n_edges_total": int(stats_block["n_edges_total"]),
        "n_edges_with_association": 0,
        "n_edges_insufficient_pairs": 0,
        "n_edges_constant_fd": 0,
        "n_edges_constant_edge": 0,
        "n_edges_unavailable": 0,
        "min_complete_pairs": MIN_COMPLETE_PAIRS,
        "quantiles": _quantiles([]),
        "per_edge": None,
        "subject_level": None,
    }

    fd_values = fd_block["values"]
    fd_constant = False
    if fd_block["status"] == "available":
        finite_fd = np.asarray(fd_values, dtype=np.float64)
        finite_fd = finite_fd[np.isfinite(finite_fd)]
        fd_constant = finite_fd.size == 0 or np.unique(finite_fd).size < 2
        if fd_constant:
            qc_fc["status"] = "constant_fd"
            qc_fc["reason"] = (
                "FD has no variation (or no finite value) across the observations, so "
                "no per-edge or subject-level association is defined"
            )
        else:
            # Always classify every edge; with fewer than three observations each
            # edge is counted as "insufficient_pairs" instead of silently missing.
            assoc = _edge_associations(edges, np.asarray(fd_values, dtype=np.float64))
            per_edge_rho, per_edge_status = assoc["rho"], assoc["status"]
            counts = assoc["counts"]
            qc_fc.update({
                "n_edges_with_association": counts["ok"],
                "n_edges_insufficient_pairs": counts["insufficient_pairs"],
                "n_edges_constant_fd": counts["constant_fd"],
                "n_edges_constant_edge": counts["constant_edge"],
                "n_edges_unavailable": counts["unavailable"],
                "quantiles": assoc["quantiles"],
            })
            subject_level = _spearman(stats_block["mean_abs"], fd_values)
            qc_fc["subject_level"] = subject_level
            if assoc["n_valid"] > 0:
                qc_fc["status"] = "available"
            else:
                qc_fc["status"] = "insufficient_data"
                qc_fc["reason"] = (
                    f"{n} observation(s); no edge has at least "
                    f"{MIN_COMPLETE_PAIRS} complete pairs with varying FD and edge value"
                )
    else:
        qc_fc["status"] = "fd_unavailable"
        qc_fc["reason"] = fd_block["reason"]
    if qc_fc["status"] != "available":
        unavailable.append({"field": "qc_fc", "status": qc_fc["status"], "reason": qc_fc["reason"]})

    # ---- Optional residualisation (explicit matrix only) ----
    adjustment: Dict[str, Any] = {
        "mode": "none",
        "reason": None,
        "residualization": None,
        "note": (
            "No nuisance design matrix was supplied, so the diagnostic is unadjusted; "
            "no adjustment is claimed."
        ),
    }
    working_edges = edges
    if nuisance is not None:
        res = _residualize(edges, nuisance, names=nuisance_names)
        residuals = res.pop("residuals", None)
        adjustment["residualization"] = res
        if res["status"] == "applied":
            adjustment["mode"] = "fc_residualized_not_partial"
            adjustment["note"] = (
                "Edge values were residualised on the explicit nuisance design and then "
                "rank-correlated with the raw FD. This is an FC-only residualisation, NOT a "
                "partial Spearman coefficient; rank and per-edge residual df are recorded, "
                "and no subject or edge was excluded."
            )
            working_edges = residuals
            if fd_block["status"] == "available" and not fd_constant and n >= MIN_COMPLETE_PAIRS:
                assoc_adj = _edge_associations(
                    working_edges, np.asarray(fd_values, dtype=np.float64)
                )
                per_edge_rho, per_edge_status = assoc_adj["rho"], assoc_adj["status"]
                counts_adj = assoc_adj["counts"]
                qc_fc.update({
                    "n_edges_with_association": counts_adj["ok"],
                    "n_edges_insufficient_pairs": counts_adj["insufficient_pairs"],
                    "n_edges_constant_fd": counts_adj["constant_fd"],
                    "n_edges_constant_edge": counts_adj["constant_edge"],
                    "n_edges_unavailable": counts_adj["unavailable"],
                    "quantiles": assoc_adj["quantiles"],
                    "adjustment": "fc_residualized_not_partial",
                })
                qc_fc["subject_level"] = _spearman(
                    _subject_mean_abs(working_edges), fd_values
                )
                if assoc_adj["n_valid"] > 0:
                    qc_fc["status"] = "available"
                else:
                    qc_fc["status"] = "insufficient_data"
                    qc_fc["reason"] = (
                        "no residualised edge has at least "
                        f"{MIN_COMPLETE_PAIRS} complete pairs with varying FD and residual "
                        "value"
                    )
        else:
            adjustment["mode"] = "unavailable"
            adjustment["reason"] = res["reason"]
            adjustment["note"] = (
                "Residualisation was requested with an explicit nuisance matrix but is "
                "unavailable; the unadjusted numbers are not relabelled as adjusted."
            )
            qc_fc["status"] = "unavailable"
            qc_fc["reason"] = res["reason"]
            unavailable.append({"field": "qc_fc", "status": "unavailable", "reason": res["reason"]})
    # An applied residualisation overrides the unadjusted per-edge summary above.
    if adjustment["mode"] == "none":
        qc_fc["adjustment"] = "none"

    # ---- Distance (known coordinates only) ----
    coords_block = _resolve_coords(coords, atlas, n_rois=n_rois, space=space)
    value_label = (
        "mean over observations of the native input value"
        if adjustment["mode"] != "fc_residualized_not_partial"
        else "mean over observations of the residualised edge value"
    )
    distance = _distance_association(
        working_edges, coords_block, value_label=value_label,
        per_edge_rho=per_edge_rho, per_edge_status=per_edge_status,
    )
    distance["association_is_adjusted"] = adjustment["mode"] == "fc_residualized_not_partial"
    distance["qc_fc_vs_distance"]["is_adjusted"] = adjustment["mode"] == "fc_residualized_not_partial"
    if distance.get("status") != "available":
        unavailable.append({
            "field": "distance",
            "status": distance.get("status"),
            "reason": distance.get("reason"),
        })

    # ---- Overall status ----
    status = "available"
    status_reason: Optional[str] = None
    if fd_block["status"] != "available":
        status = "fd_unavailable"
        status_reason = fd_block["reason"]
    elif fd_constant:
        status = "constant_fd"
        status_reason = qc_fc["reason"]
    elif adjustment["mode"] == "unavailable":
        status = "unavailable"
        status_reason = adjustment["reason"]
    elif n < MIN_COMPLETE_PAIRS or qc_fc["status"] == "insufficient_data":
        status = "insufficient_data"
        status_reason = qc_fc["reason"]

    subjects = {
        "n_observations": int(n),
        "subject_ids": _opaque_list(subject_ids) if subject_ids is not None else None,
        "condition": ([str(c) for c in condition_labels] if condition_labels is not None
                      else ([condition] * n if condition is not None else None)),
        "index_values": _opaque_list(index_values) if index_values is not None else None,
        "fd": (_finite_list(fd_values) if fd_block["status"] == "available"
               else [None] * n),
        mean_label: _finite_list(stats_block["mean_abs"]),
        "nan_count": [int(v) for v in stats_block["nan_count"]],
        "nan_share": _finite_list(stats_block["nan_share"]),
        "nan_denominator": int(stats_block["n_edges_total"]),
        "negative_count": [int(v) for v in stats_block["negative_count"]],
        "negative_share": _finite_list(stats_block["negative_share"]),
        "negative_denominator": [int(v) for v in stats_block["n_finite_edges"]],
        "n_finite_edges": [int(v) for v in stats_block["n_finite_edges"]],
        "denominators_note": (
            "nan_share = nan_count / n_rois*(n_rois-1)/2 (all upper-triangle edges, "
            "diagonal excluded); negative_share = negative_count / finite edges of "
            "that observation."
        ),
    }

    if qc_fc.get("subject_level") is None:
        qc_fc["subject_level"] = _spearman(
            stats_block["mean_abs"],
            fd_values if fd_block["status"] == "available" else np.full(n, np.nan),
        )

    return {
        "kind": "motion_qc_fc",
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "status_reason": status_reason,
        "unavailable": unavailable,
        "unit_of_observation": str(unit_of_observation),
        "condition": condition,
        "n_observations": int(n),
        "n_rois": int(n_rois),
        "n_edges": int(stats_block["n_edges_total"]),
        "roi_nonfinite": _roi_nonfinite(arr),
        "input": {
            "scale": canonical,
            "scale_source": scale_source or (
                "explicit scale=" if scale is not None else "not declared"
            ),
            "mean_label": mean_label,
            "transform": _TRANSFORMS[canonical],
            "stage": input_stage,
            "note": NATIVE_INPUT_NOTE,
        },
        "fd": {k: v for k, v in fd_block.items() if k != "values"},
        "subjects": subjects,
        "qc_fc": qc_fc,
        "adjustment": adjustment,
        "distance": distance,
        "roi_ids": _opaque_list(roi_ids) if roi_ids is not None else None,
        "notes": notes,
        "provenance": {
            "diagnostic": "motion_qc_fc",
            "schema_version": SCHEMA_VERSION,
            "descriptive_only": True,
            "affects_inference": False,
            "fd_units": fd_units,
            "fd_convention": "unknown (not recorded)",
            "input_scale": canonical,
            "scale_source": scale_source or (
                "explicit scale=" if scale is not None else "not declared"
            ),
            "input_stage": input_stage,
            "adjustment": adjustment["mode"],
            "p_values": "none",
            "exclusions": "none",
            "distance_qc_contract": DISTANCE_SEPARATION_NOTE,
        },
    }


def _subject_mean_abs(edges: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Per-observation mean |edge value| over that observation's finite edges."""
    finite = np.isfinite(edges)
    n_finite = finite.sum(axis=1)
    out = np.full(edges.shape[0], np.nan, dtype=np.float64)
    np.divide(
        np.where(finite, np.abs(edges), 0.0).sum(axis=1), n_finite,
        out=out, where=n_finite > 0,
    )
    return out


def _opaque_list(values: Any) -> List[Any]:
    """Opaque IDs/values as plain Python objects (no normalisation)."""
    if isinstance(values, (list, tuple)):
        return list(values)
    return list(np.asarray(values, dtype=object).ravel())


_STATUS_PRIORITY = {
    "fd_unavailable": 4,
    "constant_fd": 3,
    "insufficient_data": 2,
    "unavailable": 1,
    "available": 0,
}


def _aggregate_status(diags: Sequence[Mapping[str, Any]]) -> tuple:
    worst = "available"
    reason: Optional[str] = None
    for diag in diags:
        status = str(diag.get("status") or "available")
        if _STATUS_PRIORITY.get(status, 0) > _STATUS_PRIORITY.get(worst, 0):
            worst = status
            reason = diag.get("status_reason")
        elif reason is None and status != "available":
            reason = diag.get("status_reason")
    return worst, reason


def motion_qc_fc_paired(
    group1: Any,
    group2: Any,
    *,
    fd_group1: Any = None,
    fd_group2: Any = None,
    subject_ids: Optional[Sequence[Any]] = None,
    ids_group1: Optional[Sequence[Any]] = None,
    ids_group2: Optional[Sequence[Any]] = None,
    pool: bool = True,
    condition_a: str = "A",
    condition_b: str = "B",
    id_alignment: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Descriptive motion / QC-FC diagnostics for paired conditions A and B.

    Each condition gets its own diagnostic with FD tied to its own rows (one
    observation per subject and condition). ``ids_group1``/``ids_group2``
    verify the strict A<->B ID bijection and move condition B *and its FD
    vector* into condition A's order before anything is summarised; without
    them the rows are taken positionally. Pooling across conditions happens
    only with that verified identity and is reported as stacked observations
    (a subject contributes one row per condition, so the two rows are not
    independent). The condition difference is deliberately not computed: a
    B - A descriptive difference must not be mixed into the common FC QC.

    Extra keyword arguments go to :func:`motion_qc_fc` (``scale``, ``pheno``,
    ``coords``, ``atlas``, ``space``, ``nuisance``, ...).
    """
    a = np.asarray(group1, dtype=np.float64)
    b = np.asarray(group2, dtype=np.float64)
    if a.ndim != 3 or b.ndim != 3 or a.shape[1:] != b.shape[1:]:
        raise ValueError(
            "paired conditions must have identical (subjects, rois, rois) shapes; "
            f"got {a.shape} and {b.shape}."
        )
    n_a, n_b = a.shape[0], b.shape[0]
    fd_a = None if fd_group1 is None else _resolve_fd_vector(fd_group1, kwargs.get("pheno"), n_a, "fd_group1")
    fd_b = None if fd_group2 is None else _resolve_fd_vector(fd_group2, kwargs.get("pheno"), n_b, "fd_group2")
    ids: Optional[List[Any]] = None
    alignment_source = "positional"
    if ids_group1 is not None or ids_group2 is not None:
        from conninfpy._shared.alignment import IdAlignmentError, paired_id_permutation

        try:
            # One-sided IDs are rejected inside ``paired_id_permutation``;
            # the cast only states that the validator owns the None check.
            perm = paired_id_permutation(
                cast(Sequence[Any], ids_group1), cast(Sequence[Any], ids_group2),
                n_group1=n_a, n_group2=n_b,
                name_group1="ids_group1", name_group2="ids_group2",
            )
        except IdAlignmentError as exc:
            raise ValueError(str(exc)) from exc
        b = b[perm]
        fd_b = None if fd_b is None else fd_b[perm]
        assert ids_group1 is not None  # a successful permutation implies both ID vectors
        ids = list(ids_group1)
        alignment_source = "exact_id_match"
    elif subject_ids is not None:
        wanted = _opaque_list(subject_ids)
        if len(wanted) != n_a or len(wanted) != n_b:
            raise ValueError(
                f"subject_ids has {len(wanted)} value(s) but the conditions have "
                f"{n_a} and {n_b} row(s); rows must already be aligned per subject."
            )
        ids = wanted
        alignment_source = "caller_declared_subject_ids"

    common = dict(kwargs)
    arm_a = motion_qc_fc(
        a, fd=fd_a, subject_ids=ids, condition=condition_a, **common
    )
    arm_b = motion_qc_fc(
        b, fd=fd_b, subject_ids=ids, condition=condition_b, **common
    )

    if id_alignment is not None:
        alignment_source = str(id_alignment)
    pooling: Dict[str, Any] = {
        "status": "unavailable",
        "reason": None,
        "n_subjects": 0 if ids is None else int(len(ids)),
        "id_alignment": alignment_source,
        "note": (
            "Condition A and condition B are summarised separately with FD tied to "
            "their own rows; one observation per subject per condition is used."
        ),
    }
    pooled = None
    if not pool:
        pooling["reason"] = "pooling disabled by the caller"
    elif ids is None:
        pooling["reason"] = (
            "no verified subject identity: pooling the conditions would need the same "
            "subjects on both sides, so the pooled diagnostic is unavailable"
        )
    else:
        pooled_scale = common.get("scale")
        pooled_fd = None
        if fd_a is not None and fd_b is not None:
            pooled_fd = np.concatenate([fd_a, fd_b])
        pooled = motion_qc_fc(
            np.concatenate([a, b], axis=0),
            fd=pooled_fd,
            subject_ids=list(ids) + list(ids),
            condition_labels=[condition_a] * n_a + [condition_b] * n_b,
            **common,
        )
        pooled["provenance"]["pooling"] = "stacked_conditions_verified_ids"
        pooling.update({
            "status": "verified_ids",
            "n_subjects": int(len(ids)),
            "n_observations": int(n_a + n_b),
            "note": (
                "Conditions were stacked only because the subject identity was verified: "
                "each subject contributes one observation per condition, so the two rows "
                "of a subject are not independent (a descriptive limitation, not a test)."
            ),
            "scale": pooled_scale,
        })
        pooled["notes"] = list(pooled.get("notes") or []) + [pooling["note"]]

    arms = [arm_a, arm_b] + ([pooled] if pooled is not None else [])
    status, reason = _aggregate_status(arms)
    return {
        "kind": "motion_qc_fc_paired",
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "status_reason": reason,
        "conditions": {condition_a: arm_a, condition_b: arm_b},
        "pooled": pooled,
        "pooling": pooling,
        "id_alignment": alignment_source,
        "subject_ids": ids,
        "difference": {
            "status": "not_computed",
            "reason": (
                "The descriptive B - A difference is deliberately not computed here: a "
                "condition difference must not be mixed with the common-FC QC, and no "
                "inference is drawn from these summaries."
            ),
        },
        "notes": [NO_PVALUES_NOTE, NO_FILTER_NOTE, NO_GSR_NOTE],
        "provenance": {
            "diagnostic": "motion_qc_fc_paired",
            "schema_version": SCHEMA_VERSION,
            "descriptive_only": True,
            "affects_inference": False,
            "id_alignment": alignment_source,
            "pooling": pooling["status"],
            "p_values": "none",
            "exclusions": "none",
        },
    }
