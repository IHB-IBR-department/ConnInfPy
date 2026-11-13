"""Input preparation shared by every inference entry point.

Connectivity stacks enter the package in exactly one way:

- :func:`check_connectivity` — shape check, finite off-diagonal values,
  zero diagonal. Used by the
  advanced pipelines (``compute_p_val*``, ``nbs_bct``), which expect data
  already on the analysis scale (e.g. Fisher z).
- :func:`prepare_connectivity` — the same check plus a single Fisher r→z
  transform when the input holds correlations. Used by
  :func:`conninfpy.analyze`.
- :func:`infer_input_kind` — tells correlations from Fisher-z values when
  the data make it unambiguous.

The caller's array is never modified; a copy is made only when the
diagonal actually needs zeroing. Missing (NaN) or infinite off-diagonal
values are rejected: a single one would turn every permutation maximum into
NaN, and every edge would then get the smallest possible p-value.
Matrices must be symmetric: the inference pipelines, ComBat, the parametric
baselines and the network summaries treat edges as undirected (upper
triangle), so a directed matrix would be tested inconsistently. The scoring
utilities (``get_tfnbs_score``, ``get_nbs_score``) still accept directed
graphs.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from conninfpy._shared.arrays import fisher_r_to_z

# Subjects scanned per block in infer_input_kind, so large stacks don't
# allocate a full-size |Y| temporary.
_SCAN_BLOCK = 64


def check_connectivity(
    Y: npt.ArrayLike, name: str = "Y", allow_missing: bool = False
) -> npt.NDArray[np.float64]:
    """Validate an ``(n_subjects, N, N)`` stack and zero its diagonal.

    Parameters
    ----------
    Y : array_like of shape (n_subjects, N, N)
        Per-subject connectivity matrices.
    name : str, optional
        Argument name used in error messages.

    Returns
    -------
    np.ndarray of shape (n_subjects, N, N)
        float64 stack with a zero diagonal, exactly symmetric. The input
        itself when it already is all of that; otherwise a new array. Stacks
        that are symmetric up to rounding (e.g. from ``np.corrcoef``, whose
        two triangles can differ in the last bit) get their triangles
        averaged, so every edge has one value: downstream statistics are then
        exactly symmetric, and a map is never mistaken for a directed one.

    Raises
    ------
    ValueError
        If ``Y`` is not a stack of square matrices, or if any off-diagonal
        value is NaN or infinite. The diagonal is ignored (it is zeroed).
    """
    arr = np.asarray(Y, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(
            f"{name} must have shape (n_subjects, N, N); got {arr.shape}."
        )
    _check_symmetric(arr, name)
    if allow_missing:
        # Available-case analysis: NaN marks a missing value; Inf is an error.
        off = ~np.eye(arr.shape[-1], dtype=bool)
        if np.isinf(arr[:, off]).any():
            raise ValueError(f"{name} has infinite off-diagonal values; only NaN marks missing data.")
    else:
        report = nonfinite_report(arr)
        if report["n_values"]:
            raise ValueError(_format_nonfinite(report, name))
    if not np.array_equal(arr, arr.transpose(0, 2, 1), equal_nan=True):
        arr = (arr + arr.transpose(0, 2, 1)) / 2.0   # a new array; x + x halves exactly
    idx = np.arange(arr.shape[-1])
    if np.any(arr[:, idx, idx] != 0):  # NaN != 0, so NaN diagonals are zeroed too
        if isinstance(Y, np.ndarray) and np.shares_memory(arr, Y):
            arr = arr.copy()  # never write into the caller's array
        arr[:, idx, idx] = 0.0
    return arr


def _check_symmetric(arr: npt.NDArray[np.float64], name: str) -> None:
    """Reject directed (asymmetric) stacks, including NaN in one triangle only."""
    scale = 1.0
    for start in range(0, arr.shape[0], _SCAN_BLOCK):
        block = arr[start:start + _SCAN_BLOCK]
        with np.errstate(invalid="ignore"):
            finite = np.abs(block[np.isfinite(block)])
            scale = max(scale, float(finite.max()) if finite.size else 0.0)
    tol = 1e-8 * scale
    for start in range(0, arr.shape[0], _SCAN_BLOCK):
        block = arr[start:start + _SCAN_BLOCK]
        blockT = block.transpose(0, 2, 1)
        nan_mismatch = np.isnan(block) != np.isnan(blockT)
        with np.errstate(invalid="ignore"):
            diff = np.abs(block - blockT)
        bad = nan_mismatch | (np.nan_to_num(diff, nan=0.0) > tol)
        if bad.any():
            s, i, j = np.argwhere(bad)[0]
            raise ValueError(
                f"{name} is not symmetric (subject {start + s}: "
                f"[{i}, {j}] = {block[s, i, j]!r}, [{j}, {i}] = {block[s, j, i]!r}). "
                "Directed connectivity is not supported by the inference "
                "pipelines, which treat edges as undirected; symmetrise the "
                "matrices (e.g. (Y + Y^T) / 2) or analyse each direction "
                "separately."
            )


def nonfinite_report(Y: npt.ArrayLike) -> dict:
    """Where an ``(n_subjects, N, N)`` stack has NaN / infinite off-diagonal values.

    Returns a dict with

    - ``n_values``: number of non-finite off-diagonal entries;
    - ``subjects``: indices of the affected subjects;
    - ``rois``: ``{roi: number of subjects}`` for regions with no finite
      edge in a subject — typically a region whose signal is constant
      (outside the field of view, signal dropout);
    - ``other_values``: non-finite entries outside those regions (scattered
      missing edges).
    """
    arr = np.asarray(Y)
    n_sub, n_roi = arr.shape[0], arr.shape[-1]
    diag = np.eye(n_roi, dtype=bool)
    n_values = 0
    other = 0
    subjects: list = []
    roi_counts = np.zeros(n_roi, dtype=np.int64)
    for start in range(0, n_sub, _SCAN_BLOCK):
        bad = ~np.isfinite(arr[start:start + _SCAN_BLOCK])
        bad[:, diag] = False
        if not bad.any():
            continue
        per_subject = bad.reshape(bad.shape[0], -1).sum(axis=1)
        n_values += int(per_subject.sum())
        subjects.extend(int(start + i) for i in np.flatnonzero(per_subject))
        # A region without signal: every off-diagonal entry of its row or column
        dead = (bad.sum(axis=2) == n_roi - 1) | (bad.sum(axis=1) == n_roi - 1)
        roi_counts += dead.sum(axis=0)
        covered = dead[:, :, None] | dead[:, None, :]
        other += int((bad & ~covered).sum())
    rois = {int(r): int(roi_counts[r]) for r in np.flatnonzero(roi_counts)}
    return {"n_values": n_values, "subjects": subjects, "rois": rois, "other_values": other}


def _format_nonfinite(report: dict, name: str) -> str:
    """Error message for :func:`check_connectivity`."""
    def _short(items, k=10):
        items = list(items)
        return ", ".join(map(str, items[:k])) + (f", ... ({len(items)} in total)" if len(items) > k else "")

    parts = [
        f"{name} has {report['n_values']} missing (NaN) or infinite off-diagonal "
        f"values in {len(report['subjects'])} subject(s): {_short(report['subjects'])}."
    ]
    if report["rois"]:
        rois = sorted(report["rois"].items(), key=lambda kv: (-kv[1], kv[0]))
        parts.append("Regions without signal (ROI: number of subjects): "
                     f"{_short(f'{r}: {c}' for r, c in rois)}.")
    if report["other_values"]:
        parts.append(f"Other missing edges: {report['other_values']}.")
    parts.append(
        "Permutation inference cannot use them: one missing value makes every "
        "permutation maximum NaN and every edge 'significant'. Recompute or "
        "repair the connectivity, or remove the affected subjects or ROIs "
        "before inference."
    )
    return " ".join(parts)


def check_finite_design(X: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Reject NaN / infinite values in a design variable (interest, confounds, design matrix)."""
    arr = np.asarray(X, dtype=np.float64)
    bad = ~np.isfinite(arr)
    if bad.any():
        rows = np.flatnonzero(bad.reshape(arr.shape[0], -1).any(axis=1)) if arr.ndim else np.array([0])
        listed = ", ".join(map(str, rows[:10])) + (f", ... ({len(rows)} in total)" if len(rows) > 10 else "")
        raise ValueError(
            f"{name} has missing (NaN) or infinite values in row(s) {listed}. "
            "Remove those subjects (from the connectivity too) or fill the "
            "values deliberately before inference."
        )
    return arr


def infer_input_kind(*arrays: npt.NDArray[np.float64]) -> Optional[str]:
    """Guess whether stacks hold correlations (``'r'``) or Fisher-z (``'z'``).

    Returns ``'z'`` when any off-diagonal value exceeds 1 in magnitude,
    ``'r'`` when the diagonal is 1, and ``None`` when the data cannot
    tell (all values in [-1, 1] and a diagonal that is not ones — e.g.
    correlations whose diagonal was already zeroed, or weak Fisher-z
    connectivity). Empty rows (all zero or NaN: a region with no usable
    signal in that subject) are ignored when checking the diagonal.
    """
    offdiag_max = 0.0
    diag_is_one = True
    any_one = False
    for arr in arrays:
        arr = np.asarray(arr, dtype=np.float64)
        n_nodes = arr.shape[-1]
        offdiag = ~np.eye(n_nodes, dtype=bool)
        idx = np.arange(n_nodes)
        for start in range(0, arr.shape[0], _SCAN_BLOCK):
            raw = arr[start:start + _SCAN_BLOCK]
            block = np.abs(raw)
            if block[:, offdiag].size:
                offdiag_max = max(offdiag_max, float(np.nanmax(block[:, offdiag])))
            diag = raw[:, idx, idx]
            is_one = np.isclose(diag, 1.0)
            empty_row = np.all((raw == 0) | np.isnan(raw), axis=2)
            diag_is_one = diag_is_one and bool(np.all(is_one | empty_row))
            any_one = any_one or bool(is_one.any())
    if offdiag_max > 1.0:
        return "z"
    if diag_is_one and any_one:
        return "r"
    return None


def make_fisher_entry(*, stage: str, scope: str, report: dict,
                     source: str = "conninfpy.fisher_r_to_z") -> dict:
    """Structured provenance entry for one actually applied Fisher transform.

    ``stage`` names where in the pipeline the transform ran (e.g.
    ``analysis_prepare``, ``analysis_prepare_before_difference``,
    ``app_ingestion_option``, ``app_design_button``); ``scope`` names the
    array it was applied to (``Y`` / ``group1`` / ``group2`` /
    ``dataset_in_memory``); ``report`` is the ``fisher_r_to_z`` report of
    that same call (policy, parameters, counts with explicit units). The
    default ``source`` is legitimate only because this factory is called on
    the actual shared-helper call; entries supplied from outside must state
    their own source (see :func:`normalize_fisher_history`). The returned
    entry deep-copies the report, so later mutation of the caller's dict
    cannot rewrite provenance.
    """
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("a fisher entry needs a nonempty stage")
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("a fisher entry needs a nonempty scope")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("a fisher entry needs a nonempty source")
    if not isinstance(report, dict) or not isinstance(report.get("policy"), str):
        raise ValueError("a fisher entry needs the fisher_r_to_z report of the applied call")
    return {
        "stage": stage,
        "scope": scope,
        "source": source,
        "policy": report["policy"],
        "parameters": deepcopy(report.get("parameters") or {}),
        "counts": deepcopy(report),
    }


def normalize_fisher_history(history: Any) -> "list[dict] | None":
    """Validate a *supplied* Fisher history (list of entries) or ``None``.

    Nothing is invented or overwritten: every supplied entry must carry an
    explicit nonempty ``source`` (upstream work is never silently attributed
    to this library), a nonempty ``stage``/``scope`` and the ``counts``
    report of the call it describes, and its top-level ``policy``/
    ``parameters`` — when present — must AGREE with that report instead of
    contradicting it. A dict is accepted only as a full fisher record and
    passes its schema check (a future-version record cannot sneak through
    the convenience branch). ``None`` stays ``None``.
    """
    if history is None:
        return None
    if isinstance(history, dict):
        return normalize_fisher_record(history)["applied"]
    if not isinstance(history, (list, tuple)):
        raise ValueError("fisher history must be a list of entry dicts or None")
    out = []
    for entry in history:
        if not isinstance(entry, dict):
            raise ValueError("each fisher history entry must be a dict")
        source = entry.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                "a supplied fisher history entry needs an explicit nonempty source; "
                "upstream work is not attributed to this library by default")
        counts = entry.get("counts")
        if not isinstance(counts, dict) or not isinstance(counts.get("policy"), str):
            raise ValueError("each fisher history entry needs the counts report of its call")
        if "policy" in entry and entry["policy"] != counts["policy"]:
            raise ValueError("fisher history entry policy contradicts its counts report")
        if "parameters" in entry and dict(entry["parameters"] or {}) != dict(counts.get("parameters") or {}):
            raise ValueError("fisher history entry parameters contradict its counts report")
        out.append(make_fisher_entry(
            stage=entry.get("stage"), scope=entry.get("scope"),
            report=counts, source=source,
        ))
    return out


def normalize_fisher_record(record: Any) -> "dict | None":
    """Validate a Fisher provenance record (or ``None``) and return a copy.

    ``{'schema_version': 1, 'applied': [entries], 'upstream': [entries] |
    'unknown' | None}``. Future schema versions are rejected rather than
    silently rewritten.
    """
    if record is None:
        return None
    if not isinstance(record, dict):
        raise ValueError("the fisher record must be a dict or None")
    version = record.get("schema_version", 1)
    if version != 1:
        raise ValueError(f"unsupported fisher record schema_version {version!r}; supported: 1")
    unknown_keys = set(record) - {"schema_version", "applied", "upstream"}
    if unknown_keys:
        raise ValueError(f"unknown fisher record key(s): {sorted(unknown_keys)}")
    upstream = record.get("upstream")
    if upstream is not None and upstream != "unknown":
        upstream = normalize_fisher_history(upstream)
    return {"schema_version": 1,
            "applied": normalize_fisher_history(record.get("applied") or []),
            "upstream": upstream}


def prepare_connectivity(
    Y: npt.ArrayLike, input: str, name: str = "Y", allow_missing: bool = False,
    *, stage: str = "analysis_prepare", return_report: bool = False,
) -> "npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], Optional[dict]]":
    """Check ``Y``, zero its diagonal, and apply Fisher r→z once if needed.

    Parameters
    ----------
    Y : array_like of shape (n_subjects, N, N)
        Per-subject connectivity matrices.
    input : {'r', 'z'}
        ``'r'`` for correlations (Fisher r→z is applied), ``'z'`` for data
        already on the analysis scale (left unchanged apart from the
        diagonal).
    name : str, optional
        Argument name used in error messages; also the ``scope`` of the
        captured Fisher entry.
    allow_missing : bool, optional
        Accept NaN / infinite off-diagonal values (``available_case`` paths).
    stage : str, optional
        Provenance stage recorded on the Fisher entry (default
        ``analysis_prepare``; the paired-interest path records
        ``analysis_prepare_before_difference``).
    return_report : bool, optional
        Additionally return the structured Fisher entry of the transform
        this call actually applied (``None`` when ``input='z'``: nothing is
        transformed here). Default False returns only the array.

    Returns
    -------
    np.ndarray of shape (n_subjects, N, N)
        Prepared float64 stack with a zero diagonal (plus the entry dict
        when ``return_report=True``).
    """
    if input not in ("r", "z"):
        raise ValueError(f"input must be 'r' or 'z'; got {input!r}.")
    arr = check_connectivity(Y, name, allow_missing=allow_missing)
    entry: Optional[dict] = None
    if input == "r":
        # Diagonal is already zero, so arctanh never sees r = 1 there. The
        # report comes from the same call that transforms (no extra pass,
        # same numbers, same warnings).
        arr, report = fisher_r_to_z(arr, return_report=True)
        entry = make_fisher_entry(stage=stage, scope=name, report=report)
    return (arr, entry) if return_report else arr
