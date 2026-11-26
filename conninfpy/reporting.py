"""Deterministic provenance-driven Methods boilerplate and BrainNet Viewer export."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


_REFERENCES: Dict[str, str] = {
    "nbs": (
        "Zalesky, A., Fornito, A., & Bullmore, E. T. (2010). Network-based statistic: "
        "identifying differences in brain networks. NeuroImage, 53(4), 1197-1207."
    ),
    "tfnbs": (
        "Baggio, H. C., Abos, A., Segura, B., Campabadal, A., Garcia-Diaz, A., "
        "Uribe, C., Compta, Y., Marti, M. J., Valldeoriola, F., & Junque, C. (2018). "
        "Statistical inference in brain graphs using threshold-free network-based "
        "statistics. Human Brain Mapping, 39(6), 2289-2306."
    ),
    "tfnbs_ext": (
        "Hao, Z., Wang, P., Xia, X., Pan, Y., & Dou, W. (2024). Threshold-free "
        "network-oriented statistics in neuroscience. bioRxiv, 2024-01."
    ),
    "cnbs": (
        "Noble, S., & Scheinost, D. (2020). The constrained network-based statistic: "
        "a new level of inference for neuroimaging. MICCAI / NeuroImage."
    ),
    "freedman_lane": (
        "Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & "
        "Nichols, T. E. (2014). Permutation inference for the general linear model. "
        "NeuroImage, 92, 381-397."
    ),
    "phipson_smyth": (
        "Phipson, B., & Smyth, G. K. (2010). Permutation P-values should never be zero: "
        "calculating exact P-values when permutations are randomly drawn. "
        "Statistical Applications in Genetics and Molecular Biology, 9(1), Article 39."
    ),
    "combat_fortin": (
        "Fortin, J.-P., Cullen, N., Sheline, Y. I., Taylor, W. D., Aselcioglu, I., "
        "Cook, P. A., Adams, P., Cooper, C., Fava, M., McGrath, P. J., McInnis, M., "
        "Phillips, M. L., Trivedi, M. H., Weissman, M. M., & Shinohara, R. T. (2018). "
        "Harmonization of cortical thickness measurements across scanners and sites. "
        "NeuroImage, 167, 104-120."
    ),
    "combat_nygaard": (
        "Nygaard, V., Rødland, E. A., & Hovig, E. (2016). Methods that remove batch "
        "effects while retaining group differences may lead to exaggerated confidence in "
        "downstream analyses. Biostatistics, 17(1), 29-39."
    ),
    "bna246": (
        "Fan, L., Li, H., Zhuo, J., Zhang, Y., Wang, J., Chen, L., Yang, Z., Chu, C., "
        "Xie, S., Laird, A. R., Fox, P. T., Eickhoff, S. B., Yu, C., & Jiang, T. (2016). "
        "The Human Brainnetome Atlas: a new brain atlas based on connectional "
        "architecture. Cerebral Cortex, 26(8), 3508-3526."
    ),
    "suit": (
        "Diedrichsen, J., Balsters, J. H., Flavell, J., Cussans, E., & "
        "Ramnani, N. (2009). A probabilistic MR atlas of the human cerebellum. "
        "NeuroImage, 46(1), 39-46."
    ),
}


def _extract_provenance_fields(
    result_or_provenance: Any,
    *,
    alpha: float,
    result_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize InferenceResult / AnalyzeResult / ResultBundle / provenance dict."""
    from conninfpy import __version__

    info: Dict[str, Any] = {
        "version": __version__,
        "python": None,
        "alpha": alpha,
        "design_kind": None,
        "design": None,
        "n_subjects": None,
        "n_regions": None,
        "missing_policy": None,
        "missing_report": None,
        "fisher": None,
        "scale": None,
        "resolved_strategy": None,
        "harmonized": False,
        "preserve_provided": False,
        "strata_provided": False,
        "combat_diagnostics": None,
        "method": None,
        "inference_unit": None,
        "n_permutations": None,
        "acceleration": None,
        "tail_policy": None,
        "tail_selection": None,
        "tested_tail": None,
        "e_grid": None,
        "h_grid": None,
        "enhancement": None,
        "motion_qc": None,
        "atlas_name": None,
        "atlas_source": None,
    }

    if isinstance(result_or_provenance, Mapping):
        prov = result_or_provenance
        if "results" in prov and isinstance(prov["results"], Mapping):
            info["version"] = prov.get("conninfpy_version") or info["version"]
            info["python"] = prov.get("python")
            if prov.get("alpha") is not None:
                info["alpha"] = float(prov["alpha"])
            data = prov.get("data") or {}
            info["design_kind"] = data.get("design")
            info["n_subjects"] = data.get("n_subjects_used")
            info["n_regions"] = data.get("n_regions")
            cfg = prov.get("config") or {}
            if isinstance(cfg.get("atlas"), str):
                info["atlas_name"] = cfg.get("atlas")
            inf_cfg = cfg.get("inference") or {}
            results_map = prov["results"]
            key = result_name if (result_name and result_name in results_map) else next(iter(results_map), None)
            res_entry = results_map.get(key, {}) if key is not None else {}
        else:
            res_entry = prov
            inf_cfg = {}

        info["method"] = res_entry.get("method")
        info["inference_unit"] = res_entry.get("inference_unit")
        info["n_permutations"] = res_entry.get("n_permutations")
        info["acceleration"] = res_entry.get("acceleration")
        info["tail_policy"] = res_entry.get("tail_policy")
        info["tail_selection"] = res_entry.get("tail_selection")
        info["tested_tail"] = res_entry.get("tested_tail")
        info["resolved_strategy"] = res_entry.get("resolved_strategy")
        info["combat_diagnostics"] = res_entry.get("combat_diagnostics")
        info["harmonized"] = bool(
            info["resolved_strategy"] in ("combat_only", "combat_site_dummies_glm")
            or (
                isinstance(info["combat_diagnostics"], Mapping)
                and info["combat_diagnostics"].get("between_site_variance_ratio_after_over_before") is not None
            )
        )
        info["motion_qc"] = res_entry.get("motion_qc")
        info["fisher"] = res_entry.get("fisher")
        info["design"] = res_entry.get("design")
        if isinstance(info["design"], Mapping):
            info["design_kind"] = info["design_kind"] or info["design"].get("kind")
            info["scale"] = info["design"].get("scale")
            info["enhancement"] = info["design"].get("enhancement")
        if not info["enhancement"] and isinstance(inf_cfg, Mapping):
            enh = {
                k: inf_cfg[k]
                for k in ("E", "H", "start_thres", "n", "threshold", "nbs_stat", "min_cluster_size")
                if k in inf_cfg
            }
            if enh:
                info["enhancement"] = enh
        info["missing_report"] = res_entry.get("missing_report")
        if info["missing_report"] is not None:
            info["missing_policy"] = "available_case"
        return info

    # ResultBundle
    if hasattr(result_or_provenance, "metadata") and hasattr(result_or_provenance, "corrected_p"):
        meta = result_or_provenance.metadata or {}
        info["version"] = meta.get("software_version") or info["version"]
        info["method"] = meta.get("method")
        info["inference_unit"] = meta.get("inference_unit")
        info["n_permutations"] = meta.get("n_permutations")
        info["acceleration"] = meta.get("acceleration")
        info["tail_policy"] = meta.get("tail_policy")
        info["tail_selection"] = meta.get("tail_selection")
        info["tested_tail"] = meta.get("tested_tail")
        info["e_grid"] = meta.get("e_grid")
        info["h_grid"] = meta.get("h_grid")
        info["resolved_strategy"] = meta.get("resolved_strategy")
        info["combat_diagnostics"] = meta.get("combat_diagnostics")
        info["harmonized"] = bool(
            info["resolved_strategy"] in ("combat_only", "combat_site_dummies_glm")
            or info["combat_diagnostics"] is not None
        )
        info["motion_qc"] = meta.get("motion_qc")
        info["fisher"] = meta.get("fisher")
        info["missing_report"] = meta.get("missing_report")
        if info["missing_report"] is not None:
            info["missing_policy"] = "available_case"
        info["design"] = meta.get("design")
        if isinstance(info["design"], Mapping):
            info["design_kind"] = info["design"].get("kind")
            info["scale"] = info["design"].get("scale")
            info["enhancement"] = info["design"].get("enhancement")
            info["n_subjects"] = info["design"].get("n_subjects")
        cov = meta.get("coverage") or {}
        if info["n_subjects"] is None:
            if "observations" in cov:
                info["n_subjects"] = cov["observations"]
            elif "group1" in cov:
                info["n_subjects"] = int(cov["group1"]) + int(cov.get("group2", 0))
        info["n_regions"] = len(getattr(result_or_provenance, "roi_ids", ())) or None
        ident = meta.get("identity") or {}
        info["atlas_source"] = ident.get("atlas_source")
        return info

    # AnalyzeResult or InferenceResult / OmnibusInferenceResult
    outer = result_or_provenance
    inf = getattr(outer, "inference", outer)
    info["method"] = getattr(inf, "method", None)
    info["inference_unit"] = getattr(inf, "inference_unit", None)
    info["n_permutations"] = getattr(inf, "n_permutations", None)
    info["acceleration"] = getattr(inf, "acceleration", None)
    info["tail_policy"] = getattr(inf, "tail_policy", None)
    info["tail_selection"] = getattr(inf, "tail_selection", None)
    info["tested_tail"] = getattr(inf, "tested_tail", None)
    info["e_grid"] = getattr(inf, "e_grid", None)
    info["h_grid"] = getattr(inf, "h_grid", None)
    info["harmonized"] = bool(getattr(inf, "harmonized", False))
    info["preserve_provided"] = bool(getattr(inf, "preserve_provided", False))
    info["strata_provided"] = bool(getattr(inf, "strata_provided", False))
    info["resolved_strategy"] = getattr(outer, "resolved_strategy", None) or getattr(inf, "resolved_strategy", None)
    info["combat_diagnostics"] = (
        getattr(outer, "combat_diagnostics", None)
        if getattr(outer, "combat_diagnostics", None) is not None
        else getattr(inf, "combat_diagnostics", None)
    )
    info["motion_qc"] = getattr(outer, "motion_qc", None) or getattr(inf, "motion_qc", None)
    info["missing_report"] = getattr(inf, "missing_report", None)
    if info["missing_report"] is not None:
        info["missing_policy"] = "available_case"
    info["design"] = getattr(inf, "design", None)
    if isinstance(info["design"], Mapping):
        info["design_kind"] = info["design"].get("kind")
        info["scale"] = info["design"].get("scale")
        info["enhancement"] = info["design"].get("enhancement")
        info["n_subjects"] = info["design"].get("n_subjects")
    prov_attr = getattr(inf, "analysis_provenance", None)
    if isinstance(prov_attr, Mapping):
        info["fisher"] = prov_attr.get("fisher")
        ident = prov_attr.get("identity") or {}
        info["atlas_source"] = ident.get("atlas_source")
        cov = prov_attr.get("coverage") or {}
        if info["n_subjects"] is None:
            if "observations" in cov:
                info["n_subjects"] = cov["observations"]
            elif "group1" in cov:
                info["n_subjects"] = int(cov["group1"]) + int(cov.get("group2", 0))
    for k in ("positive", "omnibus"):
        if isinstance(inf, Mapping) and k in inf:
            arr = np.asarray(inf[k])
            info["n_regions"] = int(arr.shape[0])
            break
    return info


def _format_enhancement_params(info: Mapping[str, Any]) -> str:
    method = str(info.get("method") or "unknown")
    enh = info.get("enhancement")
    parts: List[str] = []
    if isinstance(enh, Mapping):
        if "E" in enh:
            parts.append(f"E={enh['E']}")
        if "H" in enh:
            parts.append(f"H={enh['H']}")
        if "start_thres" in enh:
            parts.append(f"start_thres={enh['start_thres']}")
        if "n" in enh:
            parts.append("n=None (exact)" if enh["n"] is None else f"n={enh['n']}")
        if "threshold" in enh:
            parts.append(f"threshold={enh['threshold']}")
        if "nbs_stat" in enh:
            parts.append(f"nbs_stat={enh['nbs_stat']}")
        if "min_cluster_size" in enh:
            parts.append(f"min_cluster_size={enh['min_cluster_size']}")
    e_grid = info.get("e_grid")
    h_grid = info.get("h_grid")
    if e_grid is not None and h_grid is not None and not parts:
        parts.append(f"grid size={len(e_grid)} (E={list(e_grid)}, H={list(h_grid)})")
    if not parts:
        return method
    return f"{method} ({', '.join(parts)})"


def methods_boilerplate(
    result_or_provenance: Any,
    *,
    alpha: float = 0.05,
    atlas: Optional[Any] = None,
    cerebellum: Optional[str] = None,
    result_name: Optional[str] = None,
) -> str:
    """Generate a deterministic, provenance-driven Methods section in Markdown.

    Uses only fields actually present in the supplied result, bundle, or
    ``provenance.json`` mapping; never invents parameters or runs new tests.
    """
    info = _extract_provenance_fields(
        result_or_provenance, alpha=alpha, result_name=result_name
    )
    refs_used: List[str] = []

    def _cite(key: str) -> None:
        if key in _REFERENCES and key not in refs_used:
            refs_used.append(key)

    lines: List[str] = ["## Statistical Methods", ""]

    # 1. Software & sample/design overview
    sw = f"ConnInfPy {info['version']}"
    if info["python"]:
        sw += f" (Python {info['python']})"

    design_kind = info["design_kind"]
    n_subj = info["n_subjects"]
    n_reg = info["n_regions"]
    design_parts: List[str] = []
    if design_kind:
        design_parts.append(f"a `{design_kind}` design")
    if n_subj is not None and n_reg is not None:
        design_parts.append(f"{n_subj} observations across {n_reg} regions of interest")
    elif n_reg is not None:
        design_parts.append(f"{n_reg} regions of interest")
    elif n_subj is not None:
        design_parts.append(f"{n_subj} observations")

    intro = f"Functional connectivity inference was performed with **{sw}**"
    if design_parts:
        intro += " using " + " on ".join(design_parts) if len(design_parts) == 2 else " using " + design_parts[0]
    intro += "."

    design_obj = info.get("design")
    if isinstance(design_obj, Mapping) and design_obj.get("columns"):
        cols = ", ".join(str(c) for c in design_obj["columns"])
        intro += f" Model columns: `{cols}`."
    lines.append(intro)

    # Missingness
    if info["missing_policy"] == "available_case":
        mr = info.get("missing_report") or {}
        if isinstance(mr, Mapping) and "edge_n_min" in mr:
            n_min = mr["edge_n_min"]
            if mr.get("edge_n_max") is not None:
                n_max = int(mr["edge_n_max"])
            elif mr.get("edge_n") is not None:
                n_max = int(np.max(mr["edge_n"]))
            else:
                n_max = n_min
            lines.append(
                f"Missing edges were handled with available-case estimation "
                f"(`missing=\"available_case\"`), yielding per-edge effective sample sizes "
                f"(`n_effective`) from {n_min} to {n_max}."
            )
        else:
            lines.append(
                "Missing edges were handled with available-case estimation (`missing=\"available_case\"`)."
            )

    # 2. Input scale & Fisher r->z
    fisher = info.get("fisher")
    scale = info.get("scale")
    if fisher or scale:
        f_parts: List[str] = []
        if scale:
            f_parts.append(f"response scale `{scale}`")
        if isinstance(fisher, Mapping):
            applied = fisher.get("applied") or []
            if applied:
                policies = [str(e.get("policy")) for e in applied if isinstance(e, Mapping) and e.get("policy")]
                if policies:
                    f_parts.append(f"Fisher r-to-z transform (policy: {', '.join(policies)})")
            upstream = fisher.get("upstream")
            if upstream:
                f_parts.append(f"upstream Fisher history: `{upstream}`")
        if f_parts:
            lines.append("Input preprocessing recorded " + "; ".join(f_parts) + ".")

    # 3. Harmonization
    strat = info.get("resolved_strategy")
    if info["harmonized"] or strat:
        h_desc: List[str] = []
        if strat:
            h_desc.append(f"harmonization strategy `{strat}`")
        if strat in ("combat_only", "combat_site_dummies_glm") or info["harmonized"]:
            _cite("combat_fortin")
            _cite("combat_nygaard")
            if info["preserve_provided"]:
                h_desc.append("with biological covariates preserved during ComBat")
        if info["strata_provided"]:
            h_desc.append("site-blocked (stratified) permutations")
        cd = info.get("combat_diagnostics")
        if isinstance(cd, Mapping) and cd.get("between_site_variance_ratio_after_over_before") is not None:
            ratio = float(cd["between_site_variance_ratio_after_over_before"])
            h_desc.append(f"between-site variance ratio after/before ComBat = {ratio:.3f}")
        lines.append("Multi-site handling used " + ", ".join(h_desc) + ".")

    # 4. Inference method, permutations, acceleration, tail policy, inference_unit
    method = str(info.get("method") or "unknown")
    if method == "nbs":
        _cite("nbs")
    elif method in ("tfnbs", "ni_tfnbs", "fbc_tfnbs"):
        _cite("tfnbs")
        if method in ("ni_tfnbs", "fbc_tfnbs"):
            _cite("tfnbs_ext")
    elif method == "cnbs":
        _cite("cnbs")

    n_perm = info.get("n_permutations")
    accel = info.get("acceleration")
    tail_policy = info.get("tail_policy") or "unspecified"
    tail_sel = info.get("tail_selection")
    unit = info.get("inference_unit") or "edge"

    inf_line = f"Statistical inference used **{_format_enhancement_params(info)}**"
    if n_perm is not None and int(n_perm) > 0:
        if design_kind == "glm":
            _cite("freedman_lane")
            inf_line += f" with {int(n_perm)} Freedman-Lane permutations"
        else:
            inf_line += f" with {int(n_perm)} permutations"
        if accel is None:
            _cite("phipson_smyth")
            inf_line += " and Monte Carlo empirical Phipson-Smyth (+1) p-values"
        else:
            inf_line += (
                f" and `{accel}` tail-approximation acceleration "
                "(exploratory: parametric tail fit replaces empirical exceedance counts)"
            )
    inf_line += "."
    lines.append(inf_line)

    tail_str = f"Multiple-comparison control used `tail_policy=\"{tail_policy}\"`"
    if tail_sel:
        tail_str += f" (`tail_selection=\"{tail_sel}\"`)"
    tail_str += f" at alpha = {info['alpha']}."
    lines.append(tail_str)

    unit_notes = {
        "edge": "p-values are indexed at the edge level (FWER or FDR under the chosen method's premises; for enhanced operators, strong edge-level FWER requires subset pivotality)",
        "component": "FWER control applies to connected components as a whole and does not support individual-edge localization claims",
        "block": "error control (FWER/FDR) applies to network blocks as a whole and does not support individual-edge localization claims",
    }
    lines.append(
        f"**Inference unit (`inference_unit`):** `{unit}` — {unit_notes.get(unit, unit)}."
    )

    # 5. Atlas, cerebellum, motion QC
    atlas_source = getattr(atlas, "source", None) if atlas is not None else info.get("atlas_source")
    atlas_name = info.get("atlas_name")
    if atlas is not None or atlas_source or atlas_name or n_reg == 246:
        label_str = str(atlas_source or atlas_name or f"{n_reg}-ROI parcellation")
        lines.append(f"Regions were mapped using `{label_str}`.")
        if "brainnetome" in label_str.lower() or "bna" in label_str.lower() or n_reg == 246:
            _cite("bna246")

    if cerebellum is not None and str(cerebellum) != "none":
        _cite("suit")
        lines.append(f"Cerebellar regions were annotated with the SUIT companion atlas (`{cerebellum}`).")

    if info.get("motion_qc") is not None:
        lines.append("Descriptive motion / QC-FC diagnostics were computed alongside inference.")

    # 6. References
    if refs_used:
        lines.extend(["", "### References", ""])
        for key in refs_used:
            lines.append(f"- {_REFERENCES[key]}")

    return "\n".join(lines) + "\n"


def export_brainnet(
    result: Any,
    atlas: Any,
    out_prefix: "str | os.PathLike[str]",
    *,
    alpha: float = 0.05,
    tail: str = "both",
    param_idx: Optional[int] = None,
) -> Dict[str, Path]:
    """Export thresholded connectivity results to BrainNet Viewer ``.node`` and ``.edge`` files."""
    if tail not in ("both", "positive", "negative"):
        raise ValueError(f"tail must be 'both', 'positive', or 'negative'; got {tail!r}.")
    if atlas is None:
        raise ValueError("atlas with valid 3D MNI coords is required for BrainNet Viewer export.")
    coords = getattr(atlas, "coords", None)
    if coords is None:
        raise ValueError("atlas.coords is missing; BrainNet Viewer export requires 3D MNI coordinates.")
    coords_arr = np.asarray(coords, dtype=np.float64)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3 or not np.isfinite(coords_arr).all():
        raise ValueError(
            f"atlas.coords must be a finite (N, 3) array; got shape {coords_arr.shape}."
        )

    inf = getattr(result, "inference", result)
    is_omnibus = "omnibus" in inf
    if is_omnibus:
        p_omn = inf._project_2d(np.asarray(inf["omnibus"], dtype=np.float64), param_idx)
        N = int(p_omn.shape[0])
        sig = np.isfinite(p_omn) & (p_omn <= alpha)
        stat_omn = getattr(inf, "stat_omnibus", None)
        val = (
            np.asarray(stat_omn, dtype=np.float64)
            if stat_omn is not None
            else np.ones((N, N), dtype=np.float64)
        )
        edge_mat = np.where(sig, val, 0.0)
    else:
        p_pos = inf._project_2d(np.asarray(inf["positive"], dtype=np.float64), param_idx)
        p_neg = inf._project_2d(np.asarray(inf["negative"], dtype=np.float64), param_idx)
        N = int(p_pos.shape[0])
        sig_pos = np.isfinite(p_pos) & (p_pos <= alpha)
        sig_neg = np.isfinite(p_neg) & (p_neg <= alpha)
        if tail == "positive":
            sig = sig_pos
        elif tail == "negative":
            sig = sig_neg
        else:
            sig = sig_pos | sig_neg

        stat_signed = getattr(inf, "stat_signed", None)
        if stat_signed is not None:
            val = np.asarray(stat_signed, dtype=np.float64)
        else:
            val = np.where(sig_pos & ~sig_neg, 1.0, np.where(sig_neg & ~sig_pos, -1.0, 0.0))
        edge_mat = np.where(sig, val, 0.0)

    if coords_arr.shape[0] != N:
        raise ValueError(
            f"atlas has {coords_arr.shape[0]} ROIs, but result matrix has {N} ROIs."
        )

    edge_mat = 0.5 * (edge_mat + edge_mat.T)
    np.fill_diagonal(edge_mat, 0.0)
    edge_mat = np.nan_to_num(edge_mat, nan=0.0)

    node_size = np.sum(edge_mat != 0.0, axis=1).astype(int)

    networks = getattr(atlas, "networks", None)
    if networks is not None and len(networks) == N:
        seen: Dict[str, int] = {}
        mod_ids: List[int] = []
        for net in networks:
            key = str(net)
            if key not in seen:
                seen[key] = len(seen) + 1
            mod_ids.append(seen[key])
    else:
        mod_ids = [1] * N

    raw_labels = getattr(atlas, "labels", None)
    labels: List[str] = []
    for i in range(N):
        raw = str(raw_labels[i]).strip() if raw_labels is not None and i < len(raw_labels) else f"ROI_{i}"
        clean = "_".join(raw.split()) or f"ROI_{i}"
        labels.append(clean)

    prefix_path = Path(out_prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    node_path = Path(f"{prefix_path}.node")
    edge_path = Path(f"{prefix_path}.edge")

    node_lines = [
        f"{coords_arr[i, 0]:.6g}\t{coords_arr[i, 1]:.6g}\t{coords_arr[i, 2]:.6g}\t{mod_ids[i]}\t{int(node_size[i])}\t{labels[i]}\n"
        for i in range(N)
    ]
    node_path.write_text("".join(node_lines), encoding="utf-8")
    np.savetxt(edge_path, edge_mat, fmt="%.6g", delimiter="\t")

    return {"node": node_path, "edge": edge_path}


__all__ = ["methods_boilerplate", "export_brainnet"]
