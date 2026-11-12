"""Capture reporting metadata at the original inference call."""
from __future__ import annotations

import inspect
import json
from copy import deepcopy
from functools import wraps
from typing import Any, Mapping, Optional

import numpy as np


def _off_diagonal(n: int) -> np.ndarray:
    return ~np.eye(n, dtype=bool)


def _mask(value: Any, n: int, name: str) -> np.ndarray:
    arr = np.array(value, dtype=bool, copy=True)
    if arr.shape != (n, n) or not np.array_equal(arr, arr.T):
        raise ValueError(f"{name} must be a symmetric {(n, n)} matrix")
    np.fill_diagonal(arr, False)
    return arr


def _tested_for_map(mask: np.ndarray, p: np.ndarray) -> np.ndarray:
    if p.ndim == 2:
        return mask.copy()
    if p.ndim == 3:
        return np.broadcast_to(mask[:, :, None], p.shape).copy()
    raise ValueError("corrected p maps must be two- or three-dimensional")


def _units_for_scale(scale: str) -> str:
    return {
        "fisher_z": "Fisher z",
        "correlation": "Pearson correlation (r)",
    }.get(scale, "unknown response units")


def _model_identity(design: Mapping[str, Any], contrast_id: Optional[str]) -> tuple[str, str]:
    kind = str(design.get("kind", "unspecified model"))
    columns = ",".join(str(x) for x in design.get("columns", ()))
    model_id = f"{kind}[{columns}]"
    if contrast_id is None:
        interest = design.get("interest")
        if isinstance(interest, Mapping) and interest.get("column"):
            contrast_id = str(interest["column"])
        else:
            contrast_id = json.dumps(design.get("contrast"), ensure_ascii=False, separators=(",", ":"))
    return model_id, contrast_id


def _atlas_metadata(atlas: Any, n: int) -> tuple[Optional[tuple], dict, dict]:
    if atlas is None:
        return None, {}, {}
    if len(atlas) != n:
        raise ValueError(f"atlas has {len(atlas)} ROIs but the result has {n}")
    labels = tuple(str(x) for x in atlas.labels)
    ids = None
    identity = {"atlas_source": deepcopy(getattr(atlas, "source", None))}
    label_ids = getattr(atlas, "label_ids", None)
    if label_ids is not None:
        ids = tuple(np.asarray(label_ids).tolist())
        identity["roi_id_kind"] = "original_atlas_label_id"
    elif len(set(labels)) == n:
        ids = labels
        identity["roi_id_kind"] = "original_roi_label"
    groups = {
        "atlas_label": labels,
        "network": tuple(str(x) for x in atlas.networks),
    }
    if getattr(atlas, "hemisphere", None) is not None:
        groups["hemisphere"] = tuple(str(x) for x in atlas.hemisphere)
    assignment = getattr(atlas, "assignment", None)
    if assignment is not None:
        for column in ("region", "anatomy", "lobe", "division"):
            if column in assignment.columns:
                groups[column] = tuple(str(x) for x in assignment[column])
    if "Brainnetome" in str(getattr(atlas, "source", "")):
        from conninfpy.atlas.resources import read_bna_hierarchy
        hierarchy = read_bna_hierarchy(index="name")
        if all(label in hierarchy.index for label in labels):
            for name, column in (("anatomical", "description"), ("lobe", "lobe"),
                                 ("structure", "structure"), ("yeo7", "yeo7"),
                                 ("yeo17", "yeo17")):
                values = tuple(str(hierarchy.at[label, column]) for label in labels)
                if name in ("yeo7", "yeo17"):
                    values = tuple(value.replace("not applicable", "not_applicable") for value in values)
                groups[name] = values
    return ids, groups, identity


def _identity(arguments: Mapping[str, Any], n: int, design: Mapping[str, Any],
              contrast_id: Optional[str]) -> dict:
    atlas_ids, groups, identity = _atlas_metadata(arguments.get("atlas"), n)
    roi_ids = arguments.get("roi_ids")
    if roi_ids is None:
        roi_ids = atlas_ids if atlas_ids is not None else tuple(range(n))
        identity.setdefault("roi_id_kind", "input_position_zero_based")
    else:
        roi_ids = tuple(roi_ids)
        identity["roi_id_kind"] = "explicit_input_id"
    if len(roi_ids) != n or len(set(roi_ids)) != n or any(x is None for x in roi_ids):
        raise ValueError("roi_ids must contain unique, non-null IDs in input matrix order")
    custom = arguments.get("groupings")
    if custom is not None:
        overlap = set(groups) & set(custom)
        if overlap:
            raise ValueError(f"custom groupings duplicate atlas grouping names: {sorted(overlap)}")
        groups.update({str(k): tuple(v) for k, v in custom.items()})
    if not groups:
        groups = {"input_position": tuple(str(k) for k in range(n))}
    for name, values in groups.items():
        if not name or len(values) != n or any(not isinstance(x, str) or not x for x in values):
            raise ValueError(f"grouping {name!r} needs one nonempty string label per ROI")
    scale = str(design.get("scale", "unknown"))
    units = arguments.get("units") or _units_for_scale(scale)
    if not isinstance(units, str) or not units.strip():
        raise ValueError("units must be a nonempty explicit string")
    model_id, resolved_contrast = _model_identity(design, contrast_id)
    resolved_contrast = arguments.get("contrast_id") or resolved_contrast
    return {
        "roi_ids": tuple(roi_ids),
        "groupings": groups,
        "identity": identity,
        "model_id": model_id,
        "contrast_id": resolved_contrast,
        "units": units,
        "scale": scale,
    }


def _coverage(arguments: Mapping[str, Any], result: Any) -> dict:
    report = getattr(result, "missing_report", None)
    if report is not None:
        keys = ("edge_n", "edge_n_group1", "edge_n_group2", "edge_residual_df")
        return {key: deepcopy(report[key]) for key in keys if key in report}
    group1 = arguments.get("group1")
    if group1 is not None:
        test_type = arguments.get("test_type")
        test_type = getattr(test_type, "value", test_type)
        coverage = {"group1": int(group1.shape[0])}
        group2 = arguments.get("group2")
        if test_type == "two-sample" and group2 is not None:
            coverage["group2"] = int(group2.shape[0])
        return coverage
    design = getattr(result, "design", None) or {}
    return {"observations": int(design["n_subjects"])} if "n_subjects" in design else {}


def _attach(result: Any, arguments: Mapping[str, Any], contrast_id: Optional[str] = None) -> None:
    design = getattr(result, "design", None)
    if not design or "contrast" not in design:
        raise ValueError("a new inference result cannot be snapshotted without design/contrast")
    n = result[next(iter(result.keys()))].shape[0]
    edge_mask = arguments.get("edge_mask")
    eligible = _mask(edge_mask if edge_mask is not None else _off_diagonal(n), n, "eligible")
    report = getattr(result, "missing_report", None)
    estimable = eligible.copy()
    tested = eligible.copy()
    if report is not None:
        if "effect_estimable_mask" in report:
            estimable = _mask(report["effect_estimable_mask"], n, "effect_estimable")
        if "tested_mask" in report:
            tested = _mask(report["tested_mask"], n, "tested")
    is_omnibus = "omnibus" in result
    if is_omnibus:
        estimable = np.zeros((n, n), dtype=bool)
    if np.any(estimable & ~eligible) or np.any(tested & ~eligible):
        raise ValueError("producer masks must be subsets of the eligible family")
    tails = ["omnibus"] if is_omnibus else ["positive", "negative"]
    tested_maps = {tail: _tested_for_map(tested, np.asarray(result[tail])) for tail in tails}
    family = {
        "scope": "explicit_edge_mask" if edge_mask is not None else "all_off_diagonal_edges",
        "n_eligible": int(eligible[np.triu_indices(n, 1)].sum()),
        "method": result.method,
        "correction": result.correction,
        "inference_unit": result.inference_unit,
    }
    result.analysis_provenance = {
        "schema_version": 1,
        "eligible": eligible,
        "estimable": estimable,
        "tested": tested_maps,
        "family": family,
        "coverage": _coverage(arguments, result),
        **_identity(arguments, n, design, contrast_id),
    }


def capture_analysis_provenance(fn: Any) -> Any:
    """Attach reporting provenance after a producer has built its result."""
    sig = inspect.signature(fn)

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        out = fn(*args, **kwargs)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        if isinstance(out, dict) and not hasattr(out, "tail_policy"):
            for name, result in out.items():
                _attach(result, arguments, str(name))
        else:
            _attach(out, arguments)
        return out

    return wrapper


def update_analysis_scale(result: Any, scale: str, units: Optional[str] = None) -> None:
    provenance = getattr(result, "analysis_provenance", None)
    if provenance is not None:
        provenance["scale"] = scale
        provenance["units"] = units or _units_for_scale(scale)


def refresh_model_identity(result: Any) -> None:
    provenance = getattr(result, "analysis_provenance", None)
    design = getattr(result, "design", None)
    if provenance is not None and design is not None:
        model_id, contrast_id = _model_identity(design, provenance.get("contrast_id"))
        provenance["model_id"] = model_id
        provenance["contrast_id"] = contrast_id


def project_analysis_provenance(provenance: Optional[dict], *, param_idx: Optional[int] = None,
                                tested_tail: Optional[str] = None) -> Optional[dict]:
    if provenance is None:
        return None
    projected = deepcopy(provenance)
    if param_idx is not None:
        projected["tested"] = {
            tail: (mask[:, :, param_idx].copy() if mask.ndim == 3 else mask.copy())
            for tail, mask in projected["tested"].items()
        }
    if tested_tail is not None:
        projected["tested"] = {tested_tail: projected["tested"][tested_tail]}
        projected["family"]["tail_policy"] = f"one-sided-{tested_tail}"
    return projected


__all__ = [
    "capture_analysis_provenance",
    "project_analysis_provenance",
    "refresh_model_identity",
    "update_analysis_scale",
]
