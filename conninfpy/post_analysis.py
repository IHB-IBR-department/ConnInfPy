"""Versioned, descriptive summaries of already completed inference.

The bundle contains no observations, null distribution or test statistic. New
producers capture ROI identity, family and validity masks; legacy results need
explicit metadata rather than guessing. Pickle files must come from a trusted
source. Group summaries do not perform inference.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1


def _square(value: Any, n: int, name: str, *, boolean: bool = False) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != (n, n):
        raise ValueError(f"{name} must have shape {(n, n)}; got {arr.shape}")
    if boolean and arr.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean matrix")
    arr = np.array(arr, dtype=bool if boolean else float, copy=True)
    if not np.allclose(arr, arr.T, equal_nan=True):
        raise ValueError(f"{name} must be symmetric")
    return arr


def _mask(value: Any, n: int, name: str) -> np.ndarray:
    arr = _square(value, n, name, boolean=True)
    if np.diag(arr).any():
        raise ValueError(f"{name} must exclude the diagonal")
    return arr


@dataclass
class ResultBundle:
    """A schema-versioned snapshot for undirected, descriptive reporting.

    Construct with :meth:`from_result`. Unknown information is never inferred
    from zero or p=1; producer-captured masks are preferred, with explicit
    metadata supported for legacy and manually constructed results.
    """
    schema_version: int
    roi_ids: tuple
    groupings: dict
    effect: Any
    effect_spec: dict
    eligible: np.ndarray
    estimable: np.ndarray
    tested: dict
    corrected_p: dict
    metadata: dict
    annotation: Any = None

    @classmethod
    def from_result(cls, result: Any, *, roi_ids: Any = None, groupings: Any = None,
                    eligible: Any = None, estimable: Any = None, tested: Any = None,
                    model_id: Any = None, contrast_id: Any = None, units: Any = None,
                    scale: Any = None, family: Any = None, annotation: Any = None) -> "ResultBundle":
        """Snapshot captured analysis metadata, with checked overrides for import.

        New producer results need no mask or family arguments. Legacy results
        require explicit metadata; nothing is reconstructed from p-values,
        zero-filled statistics or absent tails.
        """
        result = getattr(result, "inference", result)
        provenance = getattr(result, "analysis_provenance", None) or {}

        def resolve(name: str, value: Any) -> Any:
            if value is None:
                return deepcopy(provenance.get(name))
            if name not in provenance:
                return value
            saved = provenance[name]
            if name in ("eligible", "estimable"):
                consistent = np.array_equal(np.asarray(value), np.asarray(saved))
            elif name == "tested":
                consistent = (isinstance(value, Mapping) and set(value) == set(saved)
                              and all(np.array_equal(np.asarray(value[k]), np.asarray(saved[k])) for k in saved))
            elif name == "roi_ids":
                consistent = tuple(value) == tuple(saved)
            elif name == "groupings":
                consistent = {str(k): tuple(v) for k, v in value.items()} == saved
            elif name in ("units", "scale") and saved in (None, "unknown", "unknown response units"):
                return value
            else:
                consistent = value == saved
            if not consistent:
                raise ValueError(f"explicit {name} conflicts with the captured analysis")
            return deepcopy(saved)

        roi_ids = resolve("roi_ids", roi_ids)
        groupings = resolve("groupings", groupings)
        eligible = resolve("eligible", eligible)
        estimable = resolve("estimable", estimable)
        tested = resolve("tested", tested)
        model_id = resolve("model_id", model_id)
        contrast_id = resolve("contrast_id", contrast_id)
        units = resolve("units", units)
        scale = resolve("scale", scale)
        family = resolve("family", family)
        required = (roi_ids, groupings, eligible, estimable, tested,
                    model_id, contrast_id, units, scale, family)
        if any(value is None for value in required):
            raise ValueError("legacy result lacks captured analysis metadata; supply it explicitly")
        ids = tuple(roi_ids)
        n = len(ids)
        if n < 2 or len(set(ids)) != n or any(x is None for x in ids):
            raise ValueError("roi_ids must be unique, non-null original IDs")
        if not all(isinstance(x, str) and x.strip() for x in (model_id, contrast_id, units, scale)):
            raise ValueError("model_id, contrast_id, units and scale must be declared")
        if isinstance(family, str) and not family:
            raise ValueError("the original correction family must be declared")
        if isinstance(family, dict):
            family = {**deepcopy(family), "tail_policy": result.tail_policy}
        groups = {str(k): tuple(v) for k, v in groupings.items()}
        if annotation is not None:
            for name, values in annotation.get("groupings", {}).items():
                if name not in groups:
                    groups[name] = tuple(values)
        if not groups or any(not k or len(v) != n or any(not isinstance(x, str) or not x for x in v) for k, v in groups.items()):
            raise ValueError("each named grouping needs one explicit nonempty string label per ROI")
        elig = _mask(eligible, n, "eligible")
        est = _mask(estimable, n, "estimable")
        if np.any(est & ~elig):
            raise ValueError("estimable must be a subset of eligible")
        design = deepcopy(getattr(result, "design", None))
        if not design or "contrast" not in design:
            raise ValueError("legacy result lacks design/contrast; no effect can be reconstructed")
        source_scale = design.get("scale")
        if source_scale not in (None, "unknown", scale):
            raise ValueError("declared scale conflicts with the original design")
        is_omnibus = "omnibus" in result
        effect = None if is_omnibus else result.effect_map()
        if effect is None:
            if est.any():
                raise ValueError("effect unavailable but estimable mask is nonempty")
        else:
            effect = _square(effect, n, "effect")
            if not np.isfinite(effect[est]).all():
                raise ValueError("estimable entries require finite effects")
        tails = ["omnibus"] if is_omnibus else ([result.tested_tail] if getattr(result, "tested_tail", None) else ["positive", "negative"])
        if set(tested) != set(tails):
            raise ValueError(f"tested masks must name the tested tails {tails}")
        p_maps, tests = {}, {}
        for tail in tails:
            p = np.asarray(result[tail], dtype=float)
            m = np.asarray(tested[tail])
            if p.shape[:2] != (n, n) or p.ndim not in (2, 3) or m.shape != p.shape:
                raise ValueError(f"{tail} p and tested shapes must align with ROI axes and grid")
            if m.dtype.kind != "b":
                raise ValueError("tested masks must be boolean")
            p_cells = [p] if p.ndim == 2 else [p[:, :, k] for k in range(p.shape[2])]
            m_cells = [m] if m.ndim == 2 else [m[:, :, k] for k in range(m.shape[2])]
            for k, (pc, mc) in enumerate(zip(p_cells, m_cells)):
                pc = _square(pc, n, f"{tail} p[{k}]")
                mc = _mask(mc, n, f"{tail} tested[{k}]")
                if np.any(mc & ~elig) or not np.isfinite(pc[mc]).all() or np.any((pc[mc] < 0) | (pc[mc] > 1)):
                    raise ValueError("tested entries must be eligible and carry valid corrected p")
            p_maps[tail], tests[tail] = p.copy(), m.copy()
        if len({p.shape for p in p_maps.values()}) != 1:
            raise ValueError("tail maps must share the same parameter grid")
        shape = next(iter(p_maps.values())).shape
        e_grid, h_grid = getattr(result, "e_grid", None), getattr(result, "h_grid", None)
        if len(shape) == 3 and (e_grid is None or h_grid is None or len(e_grid) != shape[2] or len(h_grid) != shape[2]):
            raise ValueError("grid maps require aligned E and H coordinates")
        unit = result.inference_unit
        report = deepcopy(getattr(result, "component_report" if unit == "component" else "block_report", None)) if unit != "edge" else None
        from conninfpy import __version__
        meta = {"model_id": model_id, "contrast_id": contrast_id, "design": design, "software_version": __version__,
                "method": result.method, "correction": result.correction, "family": deepcopy(family),
                "inference_unit": unit, "native_report": report, "tail_policy": result.tail_policy,
                "n_permutations": result.n_permutations, "acceleration": result.acceleration,
                "e_grid": deepcopy(e_grid), "h_grid": deepcopy(h_grid),
                "identity": deepcopy(provenance.get("identity", {"roi_id_kind": "explicit_input_id"})),
                "coverage": deepcopy(provenance.get("coverage")),
                "missing_report": deepcopy(getattr(result, "missing_report", None))}
        spec = {"name": "contrast_beta", "definition": "cᵀβ", "units": units,
                "scale": scale, "standardizer": None, "available": effect is not None}
        if annotation is not None and tuple(annotation.get("roi_ids", ())) != ids:
            raise ValueError("annotation ROI IDs must match the result in matrix order")
        return cls(SCHEMA_VERSION, ids, groups, effect, spec, elig, est, tests, p_maps, meta, deepcopy(annotation))

    def save(self, path: Any) -> None:
        """Save a trusted Python pickle envelope with an explicit schema."""
        self._validate()
        with Path(path).open("wb") as f:
            pickle.dump({"schema_version": SCHEMA_VERSION, "bundle": self}, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _validate(self) -> None:
        n = len(self.roi_ids)
        if self.schema_version != SCHEMA_VERSION or n < 2 or len(set(self.roi_ids)) != n:
            raise ValueError("invalid schema or ROI identities")
        required = {"model_id", "contrast_id", "design", "method", "correction", "family", "inference_unit", "tail_policy", "native_report", "e_grid", "h_grid"}
        if not required.issubset(self.metadata) or self.metadata["inference_unit"] not in ("edge", "component", "block"):
            raise ValueError("incomplete or incompatible inference metadata")
        if (self.effect_spec.get("name") != "contrast_beta" or self.effect_spec.get("definition") != "cᵀβ"
                or self.effect_spec.get("standardizer") is not None or not self.effect_spec.get("units") or not self.effect_spec.get("scale")):
            raise ValueError("incomplete or incompatible effect specification")
        if not self.groupings or any(len(v) != n for v in self.groupings.values()):
            raise ValueError("groupings do not align with ROI identities")
        eligible = _mask(self.eligible, n, "eligible")
        estimable = _mask(self.estimable, n, "estimable")
        if np.any(estimable & ~eligible):
            raise ValueError("estimable is not a subset of eligible")
        if self.effect is None:
            if estimable.any() or self.effect_spec.get("available"):
                raise ValueError("missing effect is marked available or estimable")
        else:
            effect = _square(self.effect, n, "effect")
            if not np.isfinite(effect[estimable]).all():
                raise ValueError("estimable effects must be finite")
        if set(self.tested) != set(self.corrected_p) or not self.tested:
            raise ValueError("tested tails do not match corrected evidence")
        for tail, p in self.corrected_p.items():
            mask = np.asarray(self.tested[tail])
            p = np.asarray(p)
            if mask.shape != p.shape or p.ndim not in (2, 3) or p.shape[:2] != (n, n):
                raise ValueError("p maps and tested masks are not aligned")
            if p.ndim == 3 and (self.metadata["e_grid"] is None or self.metadata["h_grid"] is None
                                or len(self.metadata["e_grid"]) != p.shape[2] or len(self.metadata["h_grid"]) != p.shape[2]):
                raise ValueError("grid metadata is not aligned")
            for k in range(1 if p.ndim == 2 else p.shape[2]):
                pc = p if p.ndim == 2 else p[:, :, k]
                mc = mask if mask.ndim == 2 else mask[:, :, k]
                _square(pc, n, "corrected p")
                mc = _mask(mc, n, "tested")
                if np.any(mc & ~eligible) or not np.isfinite(pc[mc]).all() or np.any((pc[mc] < 0) | (pc[mc] > 1)):
                    raise ValueError("invalid corrected evidence or tested family")
        if self.annotation is not None and tuple(self.annotation.get("roi_ids", ())) != self.roi_ids:
            raise ValueError("annotation IDs do not match result")

    @classmethod
    def load(cls, path: Any) -> "ResultBundle":
        """Load a trusted bundle; unsupported or legacy schemas are rejected."""
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported or missing result bundle schema")
        obj = data.get("bundle")
        if not isinstance(obj, cls) or obj.schema_version != SCHEMA_VERSION:
            raise ValueError("invalid result bundle")
        obj._validate()
        return obj

    def _cell(self, mapping: Mapping[str, Any], tail: str, param_idx: Any) -> np.ndarray:
        if tail not in mapping:
            raise ValueError(f"tail {tail!r} was not saved")
        arr = mapping[tail]
        if arr.ndim == 3:
            if not isinstance(param_idx, (int, np.integer)) or not 0 <= param_idx < arr.shape[2]:
                raise ValueError("select an in-range param_idx for the grid")
            return arr[:, :, param_idx]
        if param_idx is not None:
            raise ValueError("param_idx is only valid for a grid")
        return arr

    def summarize(self, *, grouping: str, measure: str, contrast_id: str, tail: str,
                  param_idx: Any = None, alpha: float = 0.05, zero_band: float = 0.0) -> pd.DataFrame:
        """Summarize unthresholded effects and original corrected detections.

        Counts for component/block methods are projected edge locations, not
        counts of independently significant edges. No new significance is made.
        """
        if grouping not in self.groupings or measure != self.effect_spec["name"] or contrast_id != self.metadata["contrast_id"]:
            raise ValueError("unknown grouping, measure or contrast")
        if not 0 < alpha <= 1 or not np.isfinite(zero_band) or zero_band < 0:
            raise ValueError("invalid alpha or zero_band")
        p, tested = self._cell(self.corrected_p, tail, param_idx), self._cell(self.tested, tail, param_idx)
        labels = self.groupings[grouping]
        unique = list(dict.fromkeys(labels))
        i, j = np.triu_indices(len(labels), 1)
        rows = []
        for k, a in enumerate(unique):
            for b in unique[k:]:
                member = np.array([((labels[x] == a and labels[y] == b) or (labels[x] == b and labels[y] == a)) for x, y in zip(i, j)])
                em = member & self.eligible[i, j]
                vm = member & self.estimable[i, j]
                tm = member & tested[i, j]
                vals = np.array([]) if self.effect is None else self.effect[i[vm], j[vm]]
                count = len(vals)
                row = {"group_a": a, "group_b": b, "n_eligible": int(em.sum()),
                       "n_rois_a": labels.count(a), "n_rois_b": labels.count(b),
                       "n_effect_estimable": int(vm.sum()), "n_tested": int(tm.sum()),
                       "n_detected_locations": int((tm & (p[i, j] <= alpha)).sum()),
                       "mean_effect": float(vals.mean()) if count else np.nan,
                       "median_effect": float(np.median(vals)) if count else np.nan,
                       "mean_abs_effect": float(np.abs(vals).mean()) if count else np.nan,
                       "q25_effect": float(np.quantile(vals, .25)) if count else np.nan,
                       "q75_effect": float(np.quantile(vals, .75)) if count else np.nan,
                       "share_positive": float((vals > zero_band).mean()) if count else np.nan,
                       "share_negative": float((vals < -zero_band).mean()) if count else np.nan,
                       "share_zero": float((np.abs(vals) <= zero_band).mean()) if count else np.nan}
                row["detection_rate"] = row["n_detected_locations"] / row["n_tested"] if row["n_tested"] else np.nan
                rows.append(row)
        df = pd.DataFrame(rows)
        df.attrs.update({"grouping": grouping, "measure": deepcopy(self.effect_spec), "contrast_id": contrast_id,
                         "tail": tail, "param_idx": param_idx, "alpha": alpha, "zero_band": zero_band,
                         "model_id": self.metadata["model_id"], "method": self.metadata["method"],
                         "correction": self.metadata["correction"], "tail_policy": self.metadata["tail_policy"],
                         "inference_unit": self.metadata["inference_unit"], "family": deepcopy(self.metadata["family"]),
                         "note": "descriptive projected locations; no new significance"})
        if param_idx is not None:
            df.attrs["e"] = float(self.metadata["e_grid"][param_idx])
            df.attrs["h"] = float(self.metadata["h_grid"][param_idx])
        return df

    @staticmethod
    def export_summary(summary: pd.DataFrame, path: Any) -> None:
        """Export the selected table and its reporting metadata as a JSON sidecar."""
        import json
        path = Path(path)
        summary.to_csv(path, index=False)
        with path.with_suffix(path.suffix + ".json").open("w", encoding="utf-8") as f:
            json.dump(summary.attrs, f, ensure_ascii=False, indent=2)

    def detected_pairs(self, *, tail: str, param_idx: Any = None, alpha: float = 0.05,
                       shared_terms: bool = False) -> pd.DataFrame:
        """Return corrected detected locations with both endpoint annotations.

        For NBS/cNBS each row is a component/block-supported location and is
        not an individually significant edge. Profile terms are reference
        context, not a claim about the connection's function.
        """
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        p = self._cell(self.corrected_p, tail, param_idx)
        tested = self._cell(self.tested, tail, param_idx)
        report = self.metadata["native_report"]
        rows = []
        for i, j in zip(*np.triu_indices(len(self.roi_ids), 1)):
            if not tested[i, j] or p[i, j] > alpha:
                continue
            row = {"roi_i": self.roi_ids[i], "roi_j": self.roi_ids[j], "p_corrected": float(p[i, j]),
                   "tail": tail, "inference_unit": self.metadata["inference_unit"],
                   "effect": float(self.effect[i, j]) if self.effect is not None and self.estimable[i, j] else np.nan,
                   "native_id": None}
            if report is not None and self.metadata["inference_unit"] == "component":
                labels = report.get("labels", {}).get(tail)
                if labels is not None:
                    row["native_id"] = (tail, int(labels[i, j]))
            elif report is not None and self.metadata["inference_unit"] == "block":
                labels = report.get("node_labels")
                if labels is not None:
                    row["native_id"] = tuple(sorted((labels[i], labels[j]), key=str))
            if self.annotation is not None:
                left, right = self.annotation["endpoints"][i], self.annotation["endpoints"][j]
                row["endpoint_i"] = deepcopy(left)
                row["endpoint_j"] = deepcopy(right)
                if shared_terms:
                    for kind in ("BD", "PC"):
                        li = {t["term"] for t in left["profiles"][kind]["terms"]}
                        rj = {t["term"] for t in right["profiles"][kind]["terms"]}
                        row[f"shared_{kind}_terms"] = sorted(li & rj)
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame.attrs.update({"tail": tail, "param_idx": param_idx, "alpha": alpha,
                            "inference_unit": self.metadata["inference_unit"],
                            "family": deepcopy(self.metadata["family"]),
                            "note": "projected locations; endpoint terms describe reference regions only"})
        return frame

    def native_units(self, *, tail: str, param_idx: Any = None, alpha: float = 0.05) -> pd.DataFrame:
        """List native component/block detections with unique endpoint composition."""
        if self.metadata["inference_unit"] == "edge":
            raise ValueError("edge inference has no component/block units")
        if self.metadata["native_report"] is None:
            raise ValueError("native unit identities unavailable in this legacy result")
        pairs = self.detected_pairs(tail=tail, param_idx=param_idx, alpha=alpha)
        if pairs.empty:
            return pd.DataFrame(columns=["native_id", "n_locations", "roi_ids", "groupings"])
        if pairs.native_id.isna().any():
            raise ValueError("native unit identities unavailable in this legacy result")
        rows = []
        for ident in dict.fromkeys(pairs.native_id):
            subset = pairs[pairs.native_id.map(lambda x: x == ident)]
            members = set(subset.roi_i) | set(subset.roi_j)
            ordered = [x for x in self.roi_ids if x in members]
            composition = {name: {label: sum(1 for k, x in enumerate(self.roi_ids) if x in members and values[k] == label)
                                  for label in dict.fromkeys(values)} for name, values in self.groupings.items()}
            p_values = subset.p_corrected.to_numpy(float)
            if not np.allclose(p_values, p_values[0]):
                raise ValueError("projected locations in a native unit disagree on corrected p")
            row = {"native_id": ident, "p_corrected": float(p_values[0]),
                   "n_locations": len(subset), "roi_ids": ordered, "groupings": composition}
            if self.annotation is not None:
                row["endpoint_annotations"] = [deepcopy(self.annotation["endpoints"][k])
                                               for k, x in enumerate(self.roi_ids) if x in members]
            rows.append(row)
        out = pd.DataFrame(rows)
        out.attrs.update(pairs.attrs)
        return out

    @staticmethod
    def plot_summary(summary: pd.DataFrame, *, value: str = "mean_effect", ax: Any = None) -> Any:
        """Plot the already selected summary, without rebuilding or retesting it."""
        import matplotlib.pyplot as plt
        if value not in summary.columns:
            raise ValueError(f"summary has no {value!r} column")
        groups = list(dict.fromkeys(list(summary.group_a) + list(summary.group_b)))
        matrix = pd.DataFrame(np.nan, index=groups, columns=groups)
        for row in summary.itertuples(index=False):
            x = getattr(row, value)
            matrix.loc[row.group_a, row.group_b] = matrix.loc[row.group_b, row.group_a] = x
        if ax is None:
            _, ax = plt.subplots()
        im = ax.imshow(matrix.to_numpy(float))
        ax.set_xticks(range(len(groups)), groups, rotation=45, ha="right")
        ax.set_yticks(range(len(groups)), groups)
        ax.figure.colorbar(im, ax=ax, label=value)
        return ax


def brainnetome_crosswalk(*, roi_ids: Sequence[Any], coords: Any = None, space: str, units: str,
                         reference: Any = None, support: str = "point", names: Any = None,
                         parcel_image: Any = None, parcel_labels: Any = None, radius_mm: Any = None,
                         networks: Any = None, purity: float = 0.5, min_coverage: float = 0.5,
                         min_weight: float = 0.2, nearest_mm: Any = None,
                         nearest_structure: Any = None) -> dict:
    """Build an original-ID-keyed offline Brainnetome annotation snapshot.

    ``space`` is the actual source template; no registration is performed.
    A point is only a centre estimate. Nearest voxel placement is opt-in and
    requires each source ROI's known structure (``cortex``/``subcortex``);
    fallback across hemispheres or structures is rejected. Parcel fractions
    use every support voxel including the unlabelled fraction. EPI coverage is
    not inferred from atlas overlap.
    """
    from conninfpy.interpret.annotate import annotate_rois
    if units != "mm":
        raise ValueError("coordinates must be declared in mm")
    ids = tuple(roi_ids)
    if len(set(ids)) != len(ids) or any(x is None for x in ids):
        raise ValueError("roi_ids must be unique and non-null")
    if not space or space == "unknown":
        raise ValueError("declare the actual compatible source template")
    if space in ("Talairach", "TAL", "tal"):
        raise ValueError("Talairach must be transformed explicitly before annotation")
    if nearest_mm is not None:
        if support != "point" or not np.isfinite(nearest_mm) or nearest_mm <= 0:
            raise ValueError("nearest_mm must be a positive finite point distance")
        if nearest_structure is None or len(nearest_structure) != len(ids) or any(x not in ("cortex", "subcortex") for x in nearest_structure):
            raise ValueError("nearest requires a known cortex/subcortex structure for every ROI")
    rois = annotate_rois(coords, reference=reference, names=names, space=space, support=support,
                         parcel_image=parcel_image, parcel_labels=parcel_labels, radius_mm=radius_mm,
                         networks=networks, purity=purity, min_coverage=min_coverage,
                         min_weight=min_weight, nearest_mm=nearest_mm)
    if len(rois) != len(ids):
        raise ValueError("one original ROI ID is required per coordinate/support")
    ref = rois.reference
    geometry = {"reference_affine": np.asarray(ref.affine, float).tolist(),
                "reference_shape": list(ref.labels.shape), "parcel_labels": None,
                "parcel_affine": None, "parcel_shape": None}
    if support == "parcel":
        import nibabel as nib
        import hashlib
        image = nib.load(str(parcel_image)) if isinstance(parcel_image, (str, Path)) else parcel_image
        geometry.update({"parcel_labels": [int(x) for x in parcel_labels],
                         "parcel_affine": np.asarray(image.affine, float).tolist(),
                         "parcel_shape": list(image.shape),
                         "parcel_sha256": hashlib.sha256(np.ascontiguousarray(np.asarray(image.dataobj)).tobytes()).hexdigest()})
    endpoints = []
    for pos, (rid, row) in enumerate(zip(ids, rois.table.to_dict("records"))):
        if nearest_mm is not None and "nearest labelled voxel" in row["flags"]:
            expected_hemi = "L" if row["x"] < 0 else "R" if row["x"] > 0 else None
            tissue = "cortex" if row["structure"] == "cortex" else "subcortex" if row["structure"] not in ("", "mixed") else None
            if expected_hemi is None or row["hemisphere"] != expected_hemi or tissue != nearest_structure[pos]:
                raise ValueError(f"nearest placement for {rid!r} crosses hemisphere or tissue; use exact support")
        w = rois.weights[pos]
        coverage = float(w.sum())
        norm = w / coverage if coverage > 0 else w
        profiles = {}
        homologs = []
        for kind in ("BD", "PC"):
            terms, missing = [], False
            for idx in np.flatnonzero(w > 0):
                bid = int(idx) + 1
                source = None if ref.profiles is None else ref.profiles.get(kind)
                if source is None or bid not in source:
                    missing = True
                    continue
                for term, score in source[bid].items():
                    terms.append({"bna_id": bid, "term": str(term), "score": float(score),
                                  "weight": float(norm[idx])})
            if row["level"] == "unassigned":
                status = "unassigned"
            elif missing or ref.profiles is None or kind not in ref.profiles:
                status = "profile_missing"
            elif terms:
                status = "terms"
            else:
                status = "no_source_filtered_terms"
            profiles[kind] = {"status": status, "terms": terms}
        for idx in np.flatnonzero(w > 0):
            bid = int(idx) + 1
            hid = ref.homolog(bid)
            hp = {}
            for kind in ("BD", "PC"):
                source = None if ref.profiles is None else ref.profiles.get(kind)
                if source is None or hid not in source:
                    hp[kind] = {"status": "profile_missing", "terms": {}}
                else:
                    terms = dict(source[hid])
                    hp[kind] = {"status": "terms" if terms else "no_source_filtered_terms", "terms": terms}
            homologs.append({"source_bna_id": bid, "homolog_bna_id": hid,
                             "name": str(ref.regions.at[hid, "name"]), "profiles": hp})
        endpoints.append({"roi_id": rid, "anatomy": row, "bna_overlap": {int(k + 1): float(w[k]) for k in np.flatnonzero(w)},
                          "unknown_fraction": max(0.0, 1.0 - coverage), "profiles": profiles,
                          "homolog_reference": homologs})
    def group_value(row: Mapping[str, Any], column: str) -> str:
        value = str(row.get(column, ""))
        if column == "functional":
            value = value.replace("not applicable", "not_applicable")
        if value:
            return value
        flags = str(row.get("flags", ""))
        if "mixed:" in flags:
            return "mixed"
        if "mostly unlabelled" in flags:
            return "low_coverage"
        return "unknown"

    groupings = {
        name: tuple(group_value(endpoint["anatomy"], column) for endpoint in endpoints)
        for name, column in (("anatomical", "description"), ("gyrus", "gyrus"),
                             ("lobe", "lobe"), ("hemisphere", "hemisphere"),
                             ("structure", "structure"), ("functional", "network"))
    }
    return {"roi_ids": ids, "endpoints": endpoints, "groupings": groupings,
            "space": space, "units": units,
            "settings": deepcopy(rois.settings), "reference": deepcopy(ref.provenance), "geometry": geometry,
            "description": "BrainMap profiles describe endpoint reference regions, not edge function"}
