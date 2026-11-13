"""Pure presentation, unit/scale formatting, summary, and figure builders for UI views.

These helpers keep screen rendering and file exports (PNG / CSV) backed by the
exact same prepared figures and DataFrames, without altering inference,
multiple-testing correction, design matrices, or historical results.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from apps.utils.helpers import atlas_has_networks
from conninfpy.plot.matrices import _network_order, plot_null_max_distribution
from conninfpy.plot.networks import plot_network_summary
from conninfpy.results import InferenceResult, OmnibusInferenceResult
from conninfpy.stats._estimates import COUPLING_CLASSES


def unwrap_result(res_wrapper: Any) -> Any:
    """Return the underlying InferenceResult / OmnibusInferenceResult."""
    return getattr(res_wrapper, "inference", res_wrapper)


def get_2d_result(res_wrapper: Any, param_idx: Optional[int] = None) -> Tuple[Any, Optional[int]]:
    """Return a 2D-indexed result (selecting ``param_idx`` when ``is_grid``)."""
    inner = unwrap_result(res_wrapper)
    if getattr(inner, "is_grid", False):
        e_grid = getattr(inner, "e_grid", None)
        n_cells = len(e_grid) if e_grid is not None else 1
        idx = 0 if param_idx is None else int(np.clip(int(param_idx), 0, max(0, n_cells - 1)))
        return inner.select(idx), idx
    return inner, None


def _matrix_node_count(res: Any) -> int:
    inner = unwrap_result(res)
    for key in ("positive", "negative", "omnibus"):
        try:
            arr = np.asarray(inner[key])
            return int(arr.shape[0])
        except Exception:
            continue
    raise ValueError("Cannot determine matrix node count from inference result.")


def _summarize_fisher_entry(entry: Any) -> str:
    counts = entry.get("counts") or {}
    off = counts.get("off_diagonal") or {}
    diag = counts.get("diagonal") or {}
    policy = entry.get("policy")
    scope = entry.get("scope")
    if policy == "unbounded":
        infinity = counts.get("left_at_infinity") or {}
        inf_off = (infinity.get("off_diagonal") or {}).get("n_cells", 0) or 0
        return f"{scope}: unbounded, {int(inf_off)} off-diagonal cell(s) left at ±∞"
    n_off = int(off.get("n_cells_clipped", 0) or 0)
    n_edges = int(off.get("n_unique_undirected_edges_clipped", 0) or 0)
    n_diag = int(diag.get("n_cells_clipped", 0) or 0)
    params = entry.get("parameters") or {}
    if policy == "clip_r":
        bound = f"clip r at ±{params.get('clip_r', 0.9999):g}"
    else:
        bound = f"clip z at ±{params.get('max_z', 5):g}"
    parts = [f"{scope}: {bound}"]
    if n_off or n_diag:
        parts.append(f"{n_off} off-diagonal matrix cell(s) clipped"
                     + (f", {n_edges} unique undirected edge(s)" if n_edges else "")
                     + (f", {n_diag} structural diagonal cell(s)" if n_diag else ""))
    return ", ".join(parts)


def summarize_fisher_record(record: Any) -> Optional[str]:
    """One-line summary of the captured Fisher-transform provenance, or None."""
    if not isinstance(record, dict):
        return None
    applied = record.get("applied") or []
    upstream = record.get("upstream")
    pieces = []
    if applied:
        pieces.append("Applied here (" + "; ".join(_summarize_fisher_entry(e) for e in applied) + ")")
    if upstream == "unknown":
        pieces.append("upstream transform not recorded")
    elif isinstance(upstream, list) and upstream:
        pieces.append("Upstream (" + "; ".join(_summarize_fisher_entry(e) for e in upstream) + ")")
    return " · ".join(pieces) or None


def describe_result_scale_and_stat(res_wrapper: Any, run_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return explicit statistic symbols, response scale, and inference unit labels."""
    inner = unwrap_result(res_wrapper)
    is_omnibus = isinstance(inner, OmnibusInferenceResult) or (
        hasattr(inner, "keys") and "omnibus" in inner and "positive" not in inner
    )
    stat_type = getattr(inner, "stat_type", None)
    if is_omnibus:
        stat_symbol = "F"
        stat_name = "F-statistic (unsigned)"
    elif stat_type == "beta":
        stat_symbol = "β"
        stat_name = "Regression contrast slope (β test statistic)"
    else:
        stat_symbol = "t"
        stat_name = "t-statistic"

    design = getattr(inner, "design", None)
    provenance = getattr(inner, "analysis_provenance", None)
    scale = None
    if isinstance(design, dict) and design.get("scale"):
        scale = str(design.get("scale"))
    elif isinstance(provenance, dict) and provenance.get("scale"):
        scale = str(provenance.get("scale"))
    elif isinstance(run_plan, dict) and run_plan.get("analysis_scale") and design is None and provenance is None:
        # Legacy run_plan without design/provenance scale
        scale = str(run_plan.get("analysis_scale"))

    if scale not in ("fisher_z", "correlation", "difference_fisher_z"):
        scale = "unknown"

    # Known preprocessing facts (three-state): the estimator that produced the
    # matrices, when explicitly recorded; unknown stays unknown.
    facts = None
    for source in (provenance if isinstance(provenance, dict) else None,
                   run_plan if isinstance(run_plan, dict) else None):
        value = source.get("preprocessing") if isinstance(source, dict) else None
        if isinstance(value, dict):
            facts = value
            break
    method_fact = facts.get("connectivity_method") if facts else None
    estimator = method_fact.get("value") if isinstance(method_fact, dict) else None

    fisher_record = None
    for source in (provenance if isinstance(provenance, dict) else None,
                   run_plan if isinstance(run_plan, dict) else None):
        value = source.get("fisher") if isinstance(source, dict) else None
        if isinstance(value, dict):
            fisher_record = value
            break

    scale_labels = {
        "fisher_z": "Fisher z",
        "correlation": "Pearson correlation (r)",
        "difference_fisher_z": "Fisher z difference (B − A)",
        "unknown": "unknown / not recorded",
    }
    effect_units = {
        "fisher_z": "c' β (Δz, Fisher z units)",
        "correlation": "c' β (Δr, correlation units)",
        "difference_fisher_z": "c' β (Fisher z difference units)",
        "unknown": "c' β (unknown response units)",
    }
    scale_note = ""
    if estimator == "partial":
        scale_labels["fisher_z"] = "Fisher z (atanh of partial r)"
        scale_labels["correlation"] = "partial correlation (r)"
        effect_units["fisher_z"] = "c' β (Δz, atanh of partial r)"
        effect_units["correlation"] = "c' β (Δ partial r, correlation units)"
        scale_note = (
            "Fisher z here is atanh(partial r): a scale transform that stabilizes the "
            "variance of partial correlations. It is not the exact standard z of a "
            "marginal Pearson r, and the sign/magnitude of partial correlations does "
            "not by itself carry physiological or causal meaning."
        )
    elif estimator == "spearman":
        scale_labels["fisher_z"] = "Fisher z (atanh of Spearman ρ)"
        scale_labels["correlation"] = "Spearman rank correlation (ρ)"
        effect_units["fisher_z"] = "c' β (Δz, atanh of Spearman ρ)"
        effect_units["correlation"] = "c' β (Δρ, rank correlation units)"
        scale_note = (
            "Fisher z here is atanh(Spearman ρ): a scale transform for variance "
            "stabilization, not the exact standard z of a Pearson correlation."
        )

    method = str(getattr(inner, "method", "") or "").lower()
    unit = getattr(inner, "inference_unit", None)
    if unit not in ("edge", "component", "block"):
        if method == "nbs":
            unit = "component"
        elif method == "cnbs":
            unit = "block"
        else:
            unit = "edge"

    unit_notes = {
        "edge": "Statistical inference unit: Edge (corrected p-value applies to each edge).",
        "component": (
            "Statistical inference unit: Connected Component (NBS FWER p-value applies to the "
            "component as a whole; individual edges inside a component share its p-value and are "
            "not independent edge-level detections)."
        ),
        "block": (
            "Statistical inference unit: Network Block (cNBS p-value tests the mean signed statistic "
            "across all edges of a network pair; individual edges inside a significant block share "
            "the block's p-value and may themselves be zero or opposite in sign)."
        ),
    }

    effect_label = effect_units[scale]
    recorded_units = provenance.get("units") if isinstance(provenance, dict) else None
    default_units = {
        "Fisher z", "Pearson correlation (r)", "unknown response units",
        "Fisher z (atanh of partial r)", "Fisher z (atanh of Spearman ρ)",
        "partial correlation (r)", "Spearman rank correlation (ρ)",
    }
    if isinstance(recorded_units, str) and recorded_units.strip() and recorded_units not in default_units:
        effect_label = f"c' β ({recorded_units})"

    return {
        "is_omnibus": is_omnibus,
        "stat_symbol": stat_symbol,
        "stat_name": stat_name,
        "scale": scale,
        "scale_label": scale_labels[scale],
        "effect_unit_label": effect_label,
        "scale_note": scale_note,
        "fisher": fisher_record,
        "fisher_note": summarize_fisher_record(fisher_record),
        "inference_unit": unit,
        "inference_unit_note": unit_notes[unit],
    }


def resolve_roi_ordering(atlas: Any, n_nodes: int) -> Tuple[np.ndarray, List[int], List[str], bool]:
    """Consistent ROI permutation and network boundaries across display and export."""
    if atlas is not None and len(atlas) == n_nodes and atlas_has_networks(atlas):
        perm, block_starts, block_names = _network_order(atlas)
        return np.asarray(perm, dtype=int), [int(s) for s in block_starts], [str(n) for n in block_names], True
    return np.arange(n_nodes, dtype=int), [], [], False


def _draw_network_guides(
    ax: Any,
    n_nodes: int,
    block_starts: Sequence[int],
    block_names: Sequence[str],
    has_nets: bool,
) -> None:
    """Draw matching network block separator lines and midpoint ticks when usable."""
    if not has_nets or not block_starts:
        ax.set_xlabel("ROI (matrix index)")
        ax.set_ylabel("ROI (matrix index)")
        return
    for s in block_starts[1:]:
        ax.axvline(s - 0.5, color="dimgrey", linewidth=0.6, alpha=0.8)
        ax.axhline(s - 0.5, color="dimgrey", linewidth=0.6, alpha=0.8)
    ends = list(block_starts[1:]) + [n_nodes]
    mids = [0.5 * (s + e - 1) for s, e in zip(block_starts, ends)]
    if len(block_names) <= 24:
        ax.set_xticks(mids)
        ax.set_xticklabels(block_names, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(mids)
        ax.set_yticklabels(block_names, fontsize=7)
    ax.set_xlabel("ROI (grouped by network)")
    ax.set_ylabel("ROI (grouped by network)")


def _apply_sig_contour(ax: Any, sig_mask: np.ndarray) -> None:
    sig = np.asarray(sig_mask, dtype=bool).copy()
    np.fill_diagonal(sig, False)
    if sig.any():
        ax.contour(sig.astype(float), levels=[0.5], colors="black", linewidths=0.5)


def _signed_neglog10_map(res_2d: Any, alpha: float) -> Tuple[np.ndarray, np.ndarray, str, bool]:
    """Return ``(signed_neglog10, sig_mask, colorbar_label, is_unsigned)``."""
    inner = unwrap_result(res_2d)
    is_omnibus = isinstance(inner, OmnibusInferenceResult) or (
        hasattr(inner, "keys") and "omnibus" in inner and "positive" not in inner
    )
    if is_omnibus:
        p_omni = np.asarray(inner["omnibus"], dtype=np.float64)
        sig = np.isfinite(p_omni) & (p_omni <= alpha)
        np.fill_diagonal(sig, False)
        out = np.zeros_like(p_omni)
        out[sig] = -np.log10(np.clip(p_omni[sig], 1e-300, 1.0))
        return out, sig, rf"$-\log_{{10}}(p)$ (α = {alpha:g})", True

    p_pos = np.asarray(inner["positive"], dtype=np.float64)
    p_neg = np.asarray(inner["negative"], dtype=np.float64)
    tested_tail = getattr(inner, "tested_tail", None)
    if tested_tail == "positive":
        sig = np.isfinite(p_pos) & (p_pos <= alpha)
        np.fill_diagonal(sig, False)
        out = np.zeros_like(p_pos)
        out[sig] = -np.log10(np.clip(p_pos[sig], 1e-300, 1.0))
        return out, sig, rf"$-\log_{{10}}(p_{{pos}})$ (α = {alpha:g})", True
    if tested_tail == "negative":
        sig = np.isfinite(p_neg) & (p_neg <= alpha)
        np.fill_diagonal(sig, False)
        out = np.zeros_like(p_neg)
        out[sig] = -np.log10(np.clip(p_neg[sig], 1e-300, 1.0))
        return out, sig, rf"$-\log_{{10}}(p_{{neg}})$ (α = {alpha:g})", True

    p_min = np.fmin(p_pos, p_neg)
    sig = np.isfinite(p_min) & (p_min <= alpha)
    np.fill_diagonal(sig, False)
    out = np.zeros_like(p_min)
    pos_wins = sig & (
        (~np.isfinite(p_neg)) | (p_pos < p_neg) | ((p_pos == p_neg) & (np.asarray(inner.stat_signed) >= 0))
    )
    neg_wins = sig & ~pos_wins
    out[pos_wins] = -np.log10(np.clip(p_pos[pos_wins], 1e-300, 1.0))
    out[neg_wins] = np.log10(np.clip(p_neg[neg_wins], 1e-300, 1.0))  # negative sign
    return out, sig, rf"signed $-\log_{{10}}(p)$ (α = {alpha:g})", False


def build_connectivity_maps_figure(
    res_2d: Any,
    *,
    atlas: Any = None,
    alpha: float = 0.05,
    method_name: str = "",
    run_plan: Optional[Dict[str, Any]] = None,
    effect_mask: Optional[np.ndarray] = None,
) -> Tuple[Optional[plt.Figure], Dict[str, Any]]:
    """Build aligned stat map, signed -log10(p) map, optional native c'beta, and optional truth mask."""
    inner = unwrap_result(res_2d)
    meta = describe_result_scale_and_stat(inner, run_plan=run_plan)
    is_omnibus = meta["is_omnibus"]
    stat_map = getattr(inner, "stat_omnibus", None) if is_omnibus else getattr(inner, "stat_signed", None)
    if stat_map is None:
        return None, {**meta, "has_stat_map": False, "has_effect_map": False, "ordered_by_network": False}

    stat_map = np.asarray(stat_map, dtype=np.float64)
    n_nodes = stat_map.shape[0]
    perm, block_starts, block_names, has_nets = resolve_roi_ordering(atlas, n_nodes)

    native_effect = None
    if not is_omnibus and hasattr(inner, "effect_map"):
        try:
            native_effect = inner.effect_map()
        except Exception:
            native_effect = None
    has_effect_map = native_effect is not None and np.asarray(native_effect).shape == (n_nodes, n_nodes)

    has_truth_mask = (
        effect_mask is not None
        and np.asarray(effect_mask).ndim == 2
        and np.asarray(effect_mask).shape == (n_nodes, n_nodes)
    )

    signed_log_p, sig_mask, logp_cbar_label, logp_unsigned = _signed_neglog10_map(inner, alpha)

    panels: List[str] = ["stat", "logp"]
    if has_effect_map:
        panels.append("effect")
    if has_truth_mask:
        panels.append("truth")

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.8), facecolor="white")
    if n_panels == 1:
        axes = [axes]

    prefix = f"{method_name}: " if method_name else ""
    stat_ord = stat_map[np.ix_(perm, perm)]
    sig_ord = sig_mask[np.ix_(perm, perm)]
    logp_ord = signed_log_p[np.ix_(perm, perm)]

    for ax, kind in zip(axes, panels):
        if kind == "stat":
            finite = stat_ord[np.isfinite(stat_ord)]
            vmax = float(np.quantile(np.abs(finite), 0.98)) if finite.size else 1.0
            if not vmax or vmax <= 0:
                vmax = float(np.nanmax(np.abs(finite))) if finite.size and np.nanmax(np.abs(finite)) > 0 else 1.0
            if is_omnibus:
                im = ax.imshow(stat_ord, cmap="magma", vmin=0.0, vmax=vmax, aspect="equal")
                title = f"{prefix}Observed F-Statistic (unsigned)"
            else:
                im = ax.imshow(stat_ord, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
                title = f"{prefix}Observed {meta['stat_symbol']}-Statistic (not effect size)"
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(meta["stat_symbol"], fontsize=8)
            _apply_sig_contour(ax, sig_ord)
            ax.set_title(title, fontsize=9)
        elif kind == "logp":
            finite_lp = logp_ord[np.isfinite(logp_ord)]
            vmax_lp = float(np.max(np.abs(finite_lp))) if finite_lp.size else 1.0
            if vmax_lp <= 0:
                vmax_lp = 1.0
            if logp_unsigned:
                im = ax.imshow(logp_ord, cmap="inferno", vmin=0.0, vmax=vmax_lp, aspect="equal")
            else:
                im = ax.imshow(logp_ord, cmap="RdBu_r", vmin=-vmax_lp, vmax=vmax_lp, aspect="equal")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(logp_cbar_label, fontsize=8)
            tested_tail = getattr(inner, "tested_tail", None)
            tail_txt = f" ({tested_tail} tail)" if tested_tail else ""
            ax.set_title(f"{prefix}Significant Detections{tail_txt} at α = {alpha:g}", fontsize=9)
        elif kind == "effect":
            eff_ord = np.asarray(native_effect, dtype=np.float64)[np.ix_(perm, perm)]
            finite_eff = eff_ord[np.isfinite(eff_ord)]
            vmax_eff = float(np.quantile(np.abs(finite_eff), 0.98)) if finite_eff.size else 1.0
            if not vmax_eff or vmax_eff <= 0:
                vmax_eff = float(np.nanmax(np.abs(finite_eff))) if finite_eff.size and np.nanmax(np.abs(finite_eff)) > 0 else 1.0
            im = ax.imshow(eff_ord, cmap="RdBu_r", vmin=-vmax_eff, vmax=vmax_eff, aspect="equal")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(meta["effect_unit_label"], fontsize=8)
            _apply_sig_contour(ax, sig_ord)
            ax.set_title(f"{prefix}Native Contrast Effect (c' β)", fontsize=9)
        elif kind == "truth":
            truth_ord = (np.asarray(effect_mask) > 0).astype(float)[np.ix_(perm, perm)]
            np.fill_diagonal(truth_ord, 0.0)
            im = ax.imshow(truth_ord, cmap="Greys", vmin=0.0, vmax=1.0, aspect="equal")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
            cbar.set_label("Planted effect (0 / 1)", fontsize=8)
            ax.set_title("Synthetic Ground-Truth Mask", fontsize=9)

        _draw_network_guides(ax, n_nodes, block_starts, block_names, has_nets)

    fig.tight_layout()
    if meta["inference_unit"] != "edge":
        fig.text(0.5, 0.01, meta["inference_unit_note"], ha="center", fontsize=7, wrap=True)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig, {
        **meta,
        "has_stat_map": True,
        "has_effect_map": has_effect_map,
        "has_truth_mask": has_truth_mask,
        "ordered_by_network": has_nets,
        "network_order": block_names,
        "perm": perm,
        "block_starts": block_starts,
    }


def prepare_network_summary(
    res_2d: Any,
    atlas: Any,
    alpha: float = 0.05,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Return ``(network_summary_df, unavailable_reason)``."""
    inner = unwrap_result(res_2d)
    n_nodes = _matrix_node_count(inner)
    if atlas is None:
        return None, "No atlas metadata loaded."
    if len(atlas) != n_nodes:
        return None, f"Atlas has {len(atlas)} ROIs but the result matrix has {n_nodes} nodes."
    if not atlas_has_networks(atlas):
        return None, "Atlas has no usable network labels (only missing or 'Unknown')."
    df = inner.network_summary(atlas=atlas, alpha=alpha)
    if not df.empty and "n_sig_total" in df.columns:
        df = df.sort_values(by=["n_sig_total", "network_pair"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    elif not df.empty and "n_sig_omnibus" in df.columns:
        df = df.sort_values(by=["n_sig_omnibus", "network_pair"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    return df, None


def build_network_summary_figure(
    res_2d: Any,
    atlas: Any,
    alpha: float = 0.05,
    method_name: str = "",
) -> Optional[plt.Figure]:
    """Build the K x K network-pair summary heatmap using ``plot_network_summary``."""
    inner = unwrap_result(res_2d)
    n_nodes = _matrix_node_count(inner)
    if atlas is None or len(atlas) != n_nodes or not atlas_has_networks(atlas):
        return None
    fig, ax = plt.subplots(figsize=(5.8, 4.6), facecolor="white")
    title = f"{method_name}: Network-Pair Summary (α = {alpha:g})" if method_name else f"Network-Pair Summary (α = {alpha:g})"
    plot_network_summary(inner, atlas=atlas, alpha=alpha, ax=ax, title=title)
    unit = getattr(inner, "inference_unit", "edge")
    fig.text(0.5, 0.01, f"Descriptive edge-membership summary; inference unit: {unit}.\nNo network-level significance test.",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def prepare_node_summary(
    res_2d: Any,
    atlas: Any = None,
    alpha: float = 0.05,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Build the descriptive ROI/node summary in the result's inference unit."""
    inner = unwrap_result(res_2d)
    n_nodes = _matrix_node_count(inner)
    usable_atlas = atlas if (atlas is not None and len(atlas) == n_nodes) else None
    try:
        df = inner.node_summary(atlas=usable_atlas, alpha=alpha, sort="detections")
    except (ValueError, TypeError) as exc:
        return None, str(exc)
    if usable_atlas is not None and not atlas_has_networks(usable_atlas):
        for col in ("network", "n_networks_reached", "participation", "partner_networks"):
            if col in df.columns:
                df = df.drop(columns=[col])
    return df, None


def _resolve_cnbs_label_map(report: Dict[str, Any], atlas: Any, n_nodes: int) -> Dict[Any, str]:
    """Map cNBS block node_labels codes back to human-readable network names when unambiguous."""
    node_labels = report.get("node_labels")
    if node_labels is None or len(node_labels) != n_nodes:
        return {}
    if atlas is not None and len(atlas) == n_nodes and atlas_has_networks(atlas):
        atlas_nets = [str(x) for x in atlas.networks]
        code_to_nets: Dict[Any, set] = {}
        for code, net in zip(node_labels, atlas_nets):
            code_to_nets.setdefault(code, set()).add(net)
        if all(len(nets) == 1 for nets in code_to_nets.values()):
            return {code: next(iter(nets)) for code, nets in code_to_nets.items()}
    return {code: (str(code) if isinstance(code, str) else f"Block {code}") for code in dict.fromkeys(node_labels)}


def prepare_cnbs_blocks_table(
    res_2d: Any,
    atlas: Any = None,
    alpha: float = 0.05,
    direction_labels: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Format ``res.block_report`` for display and CSV export without recomputing."""
    inner = unwrap_result(res_2d)
    report = getattr(inner, "block_report", None)
    if report is None:
        return None, {}
    n_nodes = _matrix_node_count(inner)
    label_map = _resolve_cnbs_label_map(report, atlas, n_nodes)
    tested_tail = report.get("tested_tail") or getattr(inner, "tested_tail", None)
    tail_policy = report.get("tail_policy") or getattr(inner, "tail_policy", "two-sided")
    correction = report.get("correction", "bonferroni")
    blocks = report.get("blocks") or []

    rows = []
    for row in blocks:
        a_raw = row["network_a"]
        b_raw = row["network_b"]
        a_name = label_map.get(a_raw, str(a_raw))
        b_name = label_map.get(b_raw, str(b_raw))
        pair = a_name if a_name == b_name else f"{a_name} — {b_name}"
        rec: Dict[str, Any] = {
            "Block": pair,
            "Network A": a_name,
            "Network B": b_name,
            "Edges": int(row["n_edges"]),
            "Mean Stat": float(row["mean_stat"]),
        }
        if tested_tail == "positive":
            p_val = float(row.get("p_positive", np.nan))
            rec["p (positive, corr)"] = p_val
            rec["p (positive, uncorr)"] = float(row.get("p_positive_uncorrected", np.nan))
            sig = bool(np.isfinite(p_val) and p_val <= alpha)
        elif tested_tail == "negative":
            p_val = float(row.get("p_negative", np.nan))
            rec["p (negative, corr)"] = p_val
            rec["p (negative, uncorr)"] = float(row.get("p_negative_uncorrected", np.nan))
            sig = bool(np.isfinite(p_val) and p_val <= alpha)
        elif "p_two_sided" in row:
            p_val = float(row["p_two_sided"])
            rec["p (two-sided, corr)"] = p_val
            rec["p (positive, corr)"] = float(row.get("p_positive", np.nan))
            rec["p (negative, corr)"] = float(row.get("p_negative", np.nan))
            sig = bool(np.isfinite(p_val) and p_val <= alpha)
        else:
            p_pos = float(row.get("p_positive", np.nan))
            p_neg = float(row.get("p_negative", np.nan))
            rec["p (positive, corr)"] = p_pos
            rec["p (negative, corr)"] = p_neg
            sig = bool((np.isfinite(p_pos) and p_pos <= alpha) or (np.isfinite(p_neg) and p_neg <= alpha))
        rec["Correction"] = correction
        rec["Significant"] = sig
        rows.append(rec)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["Significant", "Mean Stat"], ascending=[False, False], key=lambda s: s.abs() if s.name == "Mean Stat" else s, kind="mergesort").reset_index(drop=True)
    info = {
        "n_blocks": int(report.get("n_blocks", len(blocks))),
        "n_blocks_tested": int(report.get("n_blocks_tested", len(blocks))),
        "correction": correction,
        "tail_policy": tail_policy,
        "tested_tail": tested_tail,
        "statistic": report.get("statistic", "mean signed edge statistic per block"),
        "n_significant": int(df["Significant"].sum()) if not df.empty else 0,
    }
    return df, info


def build_cnbs_block_figure(
    res_2d: Any,
    atlas: Any = None,
    alpha: float = 0.05,
    method_name: str = "",
) -> Optional[plt.Figure]:
    """Build K x K block-level heatmap of ``mean_stat`` with significant blocks marked."""
    inner = unwrap_result(res_2d)
    report = getattr(inner, "block_report", None)
    if report is None or not report.get("blocks"):
        return None
    n_nodes = _matrix_node_count(inner)
    label_map = _resolve_cnbs_label_map(report, atlas, n_nodes)
    node_labels = report.get("node_labels") or []
    ordered_codes = list(dict.fromkeys(node_labels))
    if not ordered_codes:
        ordered_codes = sorted({r["network_a"] for r in report["blocks"]} | {r["network_b"] for r in report["blocks"]})
    code_idx = {c: i for i, c in enumerate(ordered_codes)}
    names = [label_map.get(c, str(c)) for c in ordered_codes]
    K = len(ordered_codes)
    mat = np.full((K, K), np.nan, dtype=float)
    sig = np.zeros((K, K), dtype=bool)

    tested_tail = report.get("tested_tail") or getattr(inner, "tested_tail", None)
    for row in report["blocks"]:
        i = code_idx.get(row["network_a"])
        j = code_idx.get(row["network_b"])
        if i is None or j is None:
            continue
        mat[i, j] = mat[j, i] = float(row["mean_stat"])
        if tested_tail == "positive":
            is_sig = np.isfinite(row.get("p_positive", np.nan)) and row["p_positive"] <= alpha
        elif tested_tail == "negative":
            is_sig = np.isfinite(row.get("p_negative", np.nan)) and row["p_negative"] <= alpha
        elif "p_two_sided" in row:
            is_sig = np.isfinite(row["p_two_sided"]) and row["p_two_sided"] <= alpha
        else:
            is_sig = (
                (np.isfinite(row.get("p_positive", np.nan)) and row["p_positive"] <= alpha)
                or (np.isfinite(row.get("p_negative", np.nan)) and row["p_negative"] <= alpha)
            )
        sig[i, j] = sig[j, i] = bool(is_sig)

    fig, ax = plt.subplots(figsize=(5.8, 4.6), facecolor="white")
    finite = mat[np.isfinite(mat)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("block mean statistic", fontsize=8)
    for i in range(K):
        for j in range(K):
            if np.isfinite(mat[i, j]):
                marker = " *" if sig[i, j] else ""
                ax.text(j, i, f"{mat[i, j]:+.2f}{marker}", ha="center", va="center", fontsize=6.5, color="black")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    prefix = f"{method_name}: " if method_name else ""
    ax.set_title(f"{prefix}cNBS Block Mean Statistic (* p ≤ {alpha:g})", fontsize=9)
    fig.tight_layout()
    return fig


def prepare_predicted_coupling_summary(
    res_2d: Any,
    atlas: Any = None,
    alpha: float = 0.05,
    run_plan: Optional[Dict[str, Any]] = None,
    top_k_edges: int = 50,
) -> Dict[str, Any]:
    """Prepare ``predicted()`` and ``coupling()`` summaries with explicit scale/assumptions."""
    inner = unwrap_result(res_2d)
    meta = describe_result_scale_and_stat(inner, run_plan=run_plan)
    if meta["is_omnibus"]:
        return {
            "available": False,
            "reason": "Predicted profiles and contrast coupling are not defined for non-directional omnibus F-tests.",
            **meta,
        }
    beta = getattr(inner, "beta", None)
    design = getattr(inner, "design", None)
    if beta is None or not isinstance(design, dict) or not design.get("profiles"):
        return {
            "available": False,
            "reason": "Predicted connectivity profiles unavailable (coefficient maps or covariate profiles were not recorded on this result).",
            **meta,
        }

    profiles_dict = dict(design.get("profiles") or {})
    profile_names = list(profiles_dict.keys())
    scale = meta["scale"]
    if scale == "fisher_z":
        scale_explanation = (
            "Model fitted on Fisher-z values. Predicted connectivity for each covariate profile is "
            "converted separately via r = tanh(ẑ_profile), and Δr = r_B − r_A is computed from those "
            "separate predictions (note: tanh(Δz) is NOT Δr). These are back-transformed fitted "
            "profile values, rather than arithmetic group means of observed r."
        )
    elif scale == "correlation":
        scale_explanation = (
            "Model fitted directly on correlation (r) values. Predicted connectivity r̂_A and r̂_B "
            "are shown directly on the correlation scale with Δr = r̂_B − r̂_A. Linear predictions "
            "outside [−1, 1] are extrapolations, not valid bounded correlations."
        )
    elif scale == "difference_fisher_z":
        scale_explanation = (
            "This model predicts paired Fisher-z differences (B − A). Covariate profiles predict "
            "changes, not the original condition A/B connectivity. Values and profile contrasts "
            "remain in Fisher-z difference units; absolute r, Δr and coupling classes cannot be "
            "recovered from this difference-only model."
        )
    else:
        scale_explanation = (
            "Response scale is unknown / not recorded. Predicted profiles A and B and their difference "
            "Δ = pred_B − pred_A are shown in uncalibrated response units (no r / Δr or tanh conversion applied)."
        )

    n_nodes = _matrix_node_count(inner)
    iu, ju = np.triu_indices(n_nodes, k=1)

    prof_rows = []
    pred_maps: Dict[str, np.ndarray] = {}
    for pname in profile_names:
        try:
            pmap = np.asarray(inner.predicted(pname), dtype=np.float64)
            pred_maps[pname] = pmap
            vals = pmap[iu, ju]
            vals = vals[np.isfinite(vals)]
            row = {
                "Profile": str(pname),
                "Mean (analysis scale)": float(np.mean(vals)) if vals.size else np.nan,
                "Median (analysis scale)": float(np.median(vals)) if vals.size else np.nan,
            }
            if scale == "fisher_z":
                rvals = np.tanh(vals)
                row["Mean (r = tanh z)"] = float(np.mean(rvals)) if rvals.size else np.nan
                row["Median (r = tanh z)"] = float(np.median(rvals)) if rvals.size else np.nan
            elif scale == "correlation":
                row["Mean (r)"] = row["Mean (analysis scale)"]
                row["Median (r)"] = row["Median (analysis scale)"]
            prof_rows.append(row)
        except Exception:
            continue
    profile_summaries = pd.DataFrame(prof_rows)

    comparison = design.get("comparison")
    coupling_counts_df = None
    edge_predictions_df = None

    if comparison and len(comparison) == 2 and all(c in pred_maps for c in comparison):
        name_a, name_b = str(comparison[0]), str(comparison[1])
        absolute_connectivity = scale in ("fisher_z", "correlation")
        a_native = pred_maps[name_a]
        b_native = pred_maps[name_b]
        if absolute_connectivity:
            coupling_out = inner.coupling([name_a, name_b])
            delta_arr = np.asarray(coupling_out["delta"], dtype=np.float64)
            classes_arr = np.asarray(coupling_out["classes"], dtype=object)
        else:
            delta_arr = b_native - a_native
            classes_arr = None

        if scale == "fisher_z":
            a_disp = np.tanh(a_native)
            b_disp = np.tanh(b_native)
        else:
            a_disp = a_native
            b_disp = b_native

        p_pos = np.asarray(inner["positive"], dtype=np.float64)
        p_neg = np.asarray(inner["negative"], dtype=np.float64)
        p_min = np.fmin(p_pos, p_neg)
        tested_tail = getattr(inner, "tested_tail", None)
        if tested_tail == "positive":
            sig_triu = np.isfinite(p_pos[iu, ju]) & (p_pos[iu, ju] <= alpha)
        elif tested_tail == "negative":
            sig_triu = np.isfinite(p_neg[iu, ju]) & (p_neg[iu, ju] <= alpha)
        else:
            sig_triu = np.isfinite(p_min[iu, ju]) & (p_min[iu, ju] <= alpha)

        provenance = _provenance_or_none(inner) or {}
        if provenance.get("tested"):
            tested = _tested_edge_mask(inner)[iu, ju]
        else:
            tested = np.isfinite(p_min[iu, ju])
        sig_triu &= tested
        n_sig = int(sig_triu.sum())
        scope_mask = sig_triu if n_sig > 0 else (tested & np.isfinite(delta_arr[iu, ju]))
        scope_label = (f"significant edges (n = {n_sig} at α = {alpha:g})" if n_sig > 0 else
                       "tested edges with finite predictions (no detections at α)")

        count_rows = []
        if classes_arr is not None:
            scoped_classes = classes_arr[iu, ju][scope_mask]
            total_scoped = max(1, len(scoped_classes))
            for cls_name in COUPLING_CLASSES:
                cnt = int(np.sum(scoped_classes == cls_name))
                if cnt > 0:
                    count_rows.append({"Coupling Pattern": cls_name.replace("_", " "),
                                       "Edges": cnt, "Share (%)": round(100.0 * cnt / total_scoped, 1)})
            coupling_counts_df = pd.DataFrame(count_rows)

        sel_idx = np.flatnonzero(scope_mask)
        if n_sig > 0:
            order = sel_idx[np.argsort(p_min[iu, ju][sel_idx], kind="mergesort")]
        else:
            order = sel_idx[np.argsort(-np.abs(delta_arr[iu, ju][sel_idx]), kind="mergesort")]
        order = order[: int(top_k_edges)]

        eff_map = None
        try:
            eff_map = inner.effect_map()
        except Exception:
            eff_map = None

        usable_atlas = atlas if (atlas is not None and len(atlas) == n_nodes) else None
        has_nets = usable_atlas is not None and atlas_has_networks(usable_atlas)
        edge_rows = []
        for k in order:
            i, j = int(iu[k]), int(ju[k])
            rec: Dict[str, Any] = {
                "roi_i": i,
                "roi_j": j,
            }
            source_ids = provenance.get("roi_ids")
            if source_ids is not None and len(source_ids) == n_nodes:
                rec["roi_id_i"] = source_ids[i]
                rec["roi_id_j"] = source_ids[j]
            if usable_atlas is not None:
                rec["roi_i_name"] = str(usable_atlas.labels[i])
                rec["roi_j_name"] = str(usable_atlas.labels[j])
            if has_nets:
                ni, nj = str(usable_atlas.networks[i]), str(usable_atlas.networks[j])
                rec["network_pair"] = f"{ni} — {nj}" if ni <= nj else f"{nj} — {ni}"
            if scale == "fisher_z":
                rec[f"z ({name_a})"] = float(a_native[i, j])
                rec[f"z ({name_b})"] = float(b_native[i, j])
                rec[f"r ({name_a})"] = float(a_disp[i, j])
                rec[f"r ({name_b})"] = float(b_disp[i, j])
                rec["Δr (r_B - r_A)"] = float(delta_arr[i, j])
            elif scale == "correlation":
                rec[f"r ({name_a})"] = float(a_disp[i, j])
                rec[f"r ({name_b})"] = float(b_disp[i, j])
                rec["Δr (r_B - r_A)"] = float(delta_arr[i, j])
            else:
                rec[f"pred ({name_a})"] = float(a_native[i, j])
                rec[f"pred ({name_b})"] = float(b_native[i, j])
                delta_label = "Δ (Fisher z difference units)" if scale == "difference_fisher_z" else "Δ (unknown units)"
                rec[delta_label] = float(delta_arr[i, j])
            if eff_map is not None:
                rec["c' β"] = float(eff_map[i, j])
            if classes_arr is not None:
                rec["coupling_class"] = str(classes_arr[i, j])
            rec["significant"] = bool(sig_triu[k])
            edge_rows.append(rec)
        edge_predictions_df = pd.DataFrame(edge_rows)
    else:
        scope_label = "profiles only (no two-profile comparison defined)"

    coupling_note = None
    if classes_arr is not None:
        facts = None
        for source in (provenance, run_plan if isinstance(run_plan, dict) else None):
            value = source.get("preprocessing") if isinstance(source, dict) else None
            if isinstance(value, dict):
                facts = value
                break
        gsr = None
        estimator_note = None
        if facts:
            gsr_fact = facts.get("gsr")
            gsr = gsr_fact.get("value") if isinstance(gsr_fact, dict) else None
            method_fact = facts.get("connectivity_method")
            if isinstance(method_fact, dict) and method_fact.get("value") == "partial":
                estimator_note = "Classes describe changes in partial correlation, which conditions on the other ROIs."
        if gsr is True:
            gsr_note = "Global-signal regression was applied (recorded); it removes shared variance and can itself induce negative correlations."
        elif gsr is False:
            gsr_note = "Global-signal regression was not applied (recorded); the shared global signal can make anticorrelations common even between unrelated regions."
        else:
            gsr_note = "Whether global-signal regression was applied is unknown for this dataset."
        coupling_note = " ".join(part for part in (
            estimator_note,
            "Coupling classes are descriptive labels on the recorded scale; anticorrelations depend on the preprocessing and do not by themselves imply a physiological or causal relationship.",
            gsr_note,
        ) if part)

    return {
        "available": True,
        "reason": None,
        **meta,
        "scale_explanation": scale_explanation,
        "coupling_note": coupling_note,
        "profiles": profile_names,
        "comparison": tuple(comparison) if comparison and len(comparison) == 2 else None,
        "profile_summaries": profile_summaries,
        "coupling_counts_df": coupling_counts_df,
        "edge_predictions_df": edge_predictions_df,
        "scope_label": scope_label,
    }


def build_null_max_figure(res_2d: Any, method_name: str = "") -> Tuple[Optional[plt.Figure], Dict[str, Any]]:
    """Plot saved ``null_max_dist`` via ``plot_null_max_distribution`` when available."""
    inner = unwrap_result(res_2d)
    null_dist = getattr(inner, "null_max_dist", None)
    if not isinstance(null_dist, dict) or not null_dist:
        return None, {"available": False, "reason": "Null max-distribution (`null_max_dist`) was not saved on this result."}

    is_omnibus = isinstance(inner, OmnibusInferenceResult) or (
        hasattr(inner, "keys") and "omnibus" in inner and "positive" not in inner
    )
    tested_tail = getattr(inner, "tested_tail", None)
    if is_omnibus:
        candidate_tails = ["omnibus"]
    elif tested_tail in ("positive", "negative"):
        candidate_tails = [tested_tail]
    else:
        candidate_tails = ["positive", "negative"]

    tails_to_plot = [
        t for t in candidate_tails
        if t in null_dist and null_dist[t] is not None
        and np.asarray(null_dist[t]).ndim == 1
        and np.isfinite(np.asarray(null_dist[t], dtype=float)).any()
    ]
    if not tails_to_plot:
        return None, {"available": False, "reason": "No permutation null max-distribution arrays recorded for the tested tail(s)."}

    fig, axes = plt.subplots(1, len(tails_to_plot), figsize=(5.5 * len(tails_to_plot), 3.6), facecolor="white")
    if len(tails_to_plot) == 1:
        axes = [axes]

    prefix = f"{method_name}: " if method_name else ""
    # Result statistic maps retain raw t/F/beta. Enhancement methods calibrate
    # a different score, whose observed maximum is not retained on the result.
    # Show the saved null without substituting a raw statistic on that axis.
    raw_stat_null = str(getattr(inner, "method", "") or "").lower() == "tstat"
    observed_note = None if raw_stat_null else (
        "The observed maximum on this operator's score scale was not saved. "
        "Its reference line is omitted; raw t/F values use a different scale."
    )
    for ax, tail in zip(axes, tails_to_plot):
        arr = np.asarray(null_dist[tail], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if tail == "omnibus":
            obs_map = getattr(inner, "stat_omnibus", None)
        elif tail == "positive":
            obs_map = getattr(inner, "stat_positive", None)
        else:
            obs_map = getattr(inner, "stat_negative", None)
        obs_max = None
        if raw_stat_null and obs_map is not None:
            finite_obs = np.asarray(obs_map, dtype=np.float64)
            finite_obs = finite_obs[np.isfinite(finite_obs)]
            if finite_obs.size:
                obs_max = float(np.max(finite_obs))
        plot_null_max_distribution(
            arr,
            observed_max=obs_max,
            ax=ax,
            title=f"{prefix}Null Max Distribution ({tail})",
        )
        unit = getattr(inner, "inference_unit", "edge")
        ax.set_xlabel(f"Maximum {unit} score per permutation")
    fig.tight_layout()
    if observed_note:
        fig.text(0.5, 0.01, observed_note, ha="center", fontsize=7, wrap=True)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig, {"available": True, "tails": tails_to_plot, "observed_max_note": observed_note}


def _provenance_or_none(res_wrapper: Any) -> Optional[Dict[str, Any]]:
    inner = unwrap_result(res_wrapper)
    prov = getattr(inner, "analysis_provenance", None)
    if prov is None:
        prov = getattr(res_wrapper, "analysis_provenance", None)
    return prov if isinstance(prov, dict) else None


def _tested_edge_mask(result: Any) -> np.ndarray:
    """Use recorded tested hypotheses, including a selected grid cell's mask."""
    n = _matrix_node_count(result)
    tested = (_provenance_or_none(result) or {}).get("tested")
    if not isinstance(tested, dict) or not tested:
        raise ValueError("Tested hypothesis masks were not recorded.")
    mask = np.zeros((n, n), dtype=bool)
    for tail, value in tested.items():
        if tail not in result:
            continue
        arr = np.asarray(value, dtype=bool)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.shape != mask.shape:
            raise ValueError("Recorded tested mask does not match the ROI frame.")
        mask |= arr
    np.fill_diagonal(mask, False)
    return mask


def recorded_native_detection_count(result: Any, alpha: float) -> Optional[int]:
    """Count recorded significant components/blocks, never their projected edges."""
    inner = unwrap_result(result)
    unit = getattr(inner, "inference_unit", None)
    if unit == "component":
        report = getattr(inner, "component_report", None)
        if not isinstance(report, dict) or "components" not in report:
            return None
        tail = getattr(inner, "tested_tail", None)
        return sum(
            bool(np.isfinite(row.get("p_fwer", np.nan)) and row["p_fwer"] <= alpha)
            for row in report["components"] if tail is None or row.get("tail") == tail
        )
    if unit == "block":
        table, info = prepare_cnbs_blocks_table(inner, alpha=alpha)
        return int(info["n_significant"]) if table is not None else None
    return None


def _native_unit_detections(result: Any, unit: str, alpha: float) -> Dict[Any, Tuple[bool, np.ndarray]]:
    """Read tested native units; their membership defines identity across runs."""
    n = _matrix_node_count(result)
    iu = np.triu_indices(n, 1)
    tested = _tested_edge_mask(result)
    units: Dict[Any, Tuple[bool, np.ndarray]] = {}
    selected_tail = getattr(result, "tested_tail", None)
    if unit == "component":
        report = getattr(result, "component_report", None)
        if not isinstance(report, dict) or "labels" not in report:
            raise ValueError("Native component memberships were not recorded.")
        for row in report.get("components", []):
            tail = row["tail"]
            if selected_tail is not None and tail != selected_tail:
                continue
            p = float(row["p_fwer"])
            if not np.isfinite(p):
                continue
            labels = np.asarray(report["labels"][tail])
            if labels.shape != (n, n):
                raise ValueError("Component labels do not match the ROI frame.")
            mask = (labels == row["component_id"]) & tested
            members = tuple(np.flatnonzero(mask[iu]).tolist())
            if members:
                units[(tail, members)] = (bool(p <= alpha), mask)
    else:
        report = getattr(result, "block_report", None)
        if not isinstance(report, dict) or report.get("node_labels") is None:
            raise ValueError("Native block memberships were not recorded.")
        labels = np.asarray(report["node_labels"])
        if labels.shape != (n,):
            raise ValueError("Block labels do not match the ROI frame.")
        selected_tail = report.get("tested_tail") or selected_tail
        for row in report.get("blocks", []):
            if selected_tail:
                p = float(row.get(f"p_{selected_tail}", np.nan))
            elif "p_two_sided" in row:
                p = float(row["p_two_sided"])
            else:
                ps = np.asarray([row.get("p_positive", np.nan), row.get("p_negative", np.nan)], dtype=float)
                p = float(np.min(ps[np.isfinite(ps)])) if np.isfinite(ps).any() else np.nan
            if not np.isfinite(p):
                continue
            a = labels == row["network_a"]
            b = labels == row["network_b"]
            mask = ((a[:, None] & b[None, :]) | (b[:, None] & a[None, :])) & tested
            members = tuple(np.flatnonzero(mask[iu]).tolist())
            if members:
                units[members] = (bool(p <= alpha), mask)
    return units


def _overlap_counts(primary: np.ndarray, companion: np.ndarray) -> Dict[str, Any]:
    both = int(np.sum(primary & companion))
    only_p = int(np.sum(primary & ~companion))
    only_c = int(np.sum(~primary & companion))
    union = both + only_p + only_c
    return {
        "both": both, "primary_only": only_p, "companion_only": only_c,
        "neither": int(np.sum(~primary & ~companion)), "union": union,
        "jaccard": float(both / union) if union else 1.0,
    }


def check_companion_compatibility(
    primary_res: Any,
    comp_res: Any,
    *,
    alpha: float = 0.05,
    primary_alpha: Optional[float] = None,
    comp_alpha: Optional[float] = None,
) -> Dict[str, Any]:
    """Check whether primary and companion agreement is well-defined.

    Agreement requires exact alignment on:
    1) error/hypothesis family (correction + tail_policy + omnibus vs tailed),
    2) statistical inference unit (edge vs component vs block),
    3) contrast / model / scale,
    4) tested tail direction,
    5) significance level alpha,
    6) tested / eligible edge mask,
    7) ROI frame (matrix size, roi_ids order when recorded).
    """
    p_in = unwrap_result(primary_res)
    c_in = unwrap_result(comp_res)
    reasons: List[str] = []

    try:
        n_p = _matrix_node_count(p_in)
        n_c = _matrix_node_count(c_in)
        if n_p != n_c:
            reasons.append(f"ROI frame size differs ({n_p} vs {n_c} nodes).")
    except Exception:
        reasons.append("Unable to verify ROI matrix shapes.")
        n_p = n_c = -1

    p_prov = _provenance_or_none(primary_res)
    c_prov = _provenance_or_none(comp_res)
    required_provenance = ("roi_ids", "contrast_id", "model_id", "eligible", "tested", "scale")
    for label, prov in (("Primary", p_prov), ("Companion", c_prov)):
        if not prov or any(prov.get(key) is None for key in required_provenance):
            reasons.append(
                f"{label} model/contrast, tested-mask or ROI-frame provenance is not recorded; "
                "agreement cannot be verified."
            )
    if p_prov and c_prov:
        if p_prov.get("scale") != c_prov.get("scale"):
            reasons.append("Recorded response scales differ between primary and companion.")
        if p_prov.get("roi_ids") != c_prov.get("roi_ids"):
            reasons.append("ROI frame (`roi_ids` / ordering) differs between primary and companion.")
    elif bool(p_prov) != bool(c_prov):
        reasons.append("ROI provenance is recorded on only one result, so ROI frame identity cannot be guaranteed.")

    u_p = getattr(p_in, "inference_unit", None) or ("component" if getattr(p_in, "method", "") == "nbs" else "block" if getattr(p_in, "method", "") == "cnbs" else "edge")
    u_c = getattr(c_in, "inference_unit", None) or ("component" if getattr(c_in, "method", "") == "nbs" else "block" if getattr(c_in, "method", "") == "cnbs" else "edge")
    if u_p != u_c:
        reasons.append(f"Inference unit differs (`{u_p}` vs `{u_c}`); edge-level agreement is not comparable across units.")

    corr_p = getattr(p_in, "correction", None)
    corr_c = getattr(c_in, "correction", None)
    if corr_p != corr_c:
        reasons.append(f"Multiple-testing correction family differs (`{corr_p}` vs `{corr_c}`).")

    tp_p = getattr(p_in, "tail_policy", None)
    tp_c = getattr(c_in, "tail_policy", None)
    if tp_p != tp_c:
        reasons.append(f"Hypothesis tail family differs (`{tp_p}` vs `{tp_c}`).")

    tt_p = getattr(p_in, "tested_tail", None)
    tt_c = getattr(c_in, "tested_tail", None)
    if tt_p != tt_c:
        reasons.append(f"Tested tail direction differs (`{tt_p}` vs `{tt_c}`).")

    if bool(getattr(p_in, "is_grid", False)) != bool(getattr(c_in, "is_grid", False)):
        reasons.append("One result is a parameter-grid sweep while the other is a single parameter set.")

    d_p = getattr(p_in, "design", None) or {}
    d_c = getattr(c_in, "design", None) or {}
    if d_p and d_c:
        if d_p.get("scale") != d_c.get("scale"):
            reasons.append(f"Response scale differs (`{d_p.get('scale')}` vs `{d_c.get('scale')}`).")
        if "contrast" in d_p and "contrast" in d_c:
            contrast_p = np.asarray(d_p["contrast"], dtype=float)
            contrast_c = np.asarray(d_c["contrast"], dtype=float)
            if contrast_p.shape != contrast_c.shape or not np.array_equal(contrast_p, contrast_c):
                reasons.append("Contrast vector differs between primary and companion.")
        if d_p.get("comparison") != d_c.get("comparison"):
            reasons.append("Design comparison profiles differ between primary and companion.")
    if p_prov and c_prov:
        if p_prov.get("contrast_id") != c_prov.get("contrast_id"):
            reasons.append(f"Contrast identity differs (`{p_prov.get('contrast_id')}` vs `{c_prov.get('contrast_id')}`).")
        if p_prov.get("model_id") != c_prov.get("model_id"):
            reasons.append(f"Model identity differs (`{p_prov.get('model_id')}` vs `{c_prov.get('model_id')}`).")

    if primary_alpha is not None and comp_alpha is not None and not np.isclose(primary_alpha, comp_alpha):
        reasons.append(f"Evaluation α differs ({primary_alpha:g} vs {comp_alpha:g}).")

    if n_p == n_c and n_p > 0:
        if p_prov and c_prov:
            if not np.array_equal(np.asarray(p_prov.get("eligible")), np.asarray(c_prov.get("eligible"))):
                reasons.append("Eligible edge mask differs between primary and companion.")
            t_p_prov = p_prov.get("tested") or {}
            t_c_prov = c_prov.get("tested") or {}
            if set(t_p_prov) != set(t_c_prov) or any(
                not np.array_equal(np.asarray(t_p_prov[k]), np.asarray(t_c_prov[k])) for k in t_p_prov
            ):
                reasons.append("Tested edge mask differs between primary and companion.")
        p_2d, _ = get_2d_result(p_in, 0)
        c_2d, _ = get_2d_result(c_in, 0)
        common_keys = [k for k in ("positive", "negative", "omnibus") if k in p_2d and k in c_2d]
        for k in common_keys:
            if not np.array_equal(np.isfinite(p_2d[k]), np.isfinite(c_2d[k])):
                reasons.append(f"Tested/finite edge mask differs on `{k}` p-values.")
                break

    if reasons:
        return {"compatible": False, "reasons": reasons, "summary": None, "categories": None}

    p_2d, _ = get_2d_result(p_in, 0)
    c_2d, _ = get_2d_result(c_in, 0)
    try:
        tested = _tested_edge_mask(p_2d)
        native_p = native_c = None
        if u_p in ("component", "block"):
            report_name = "component_report" if u_p == "component" else "block_report"
            report_p = getattr(p_2d, report_name, None) or {}
            report_c = getattr(c_2d, report_name, None) or {}
            if any(report_p.get(k) != report_c.get(k) for k in ("threshold", "edge_statistic", "correction")):
                raise ValueError("Native-unit inference recipes differ.")
            native_p = _native_unit_detections(p_2d, u_p, alpha)
            native_c = _native_unit_detections(c_2d, u_p, alpha)
            if set(native_p) != set(native_c):
                raise ValueError("Native component/block memberships differ; matching inferential units cannot be established.")
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return {"compatible": False, "reasons": [str(exc)], "summary": None, "categories": None}
    _, sig_p, _, _ = _signed_neglog10_map(p_2d, alpha)
    _, sig_c, _, _ = _signed_neglog10_map(c_2d, alpha)
    iu, ju = np.triu_indices(n_p, k=1)
    if native_p is not None:
        keys = sorted(native_p)
        sp = np.asarray([native_p[k][0] for k in keys], dtype=bool)
        sc = np.asarray([native_c[k][0] for k in keys], dtype=bool)
    else:
        sp = sig_p[iu, ju][tested[iu, ju]]
        sc = sig_c[iu, ju][tested[iu, ju]]
    summary = _overlap_counts(sp, sc)

    cat_mat = np.zeros((n_p, n_p), dtype=int)
    cat_mat[sig_p & ~sig_c] = 1
    cat_mat[~sig_p & sig_c] = 2
    cat_mat[sig_p & sig_c] = 3
    cat_mat[~tested] = -1
    np.fill_diagonal(cat_mat, -1)

    return {
        "compatible": True,
        "reasons": [],
        "summary": summary,
        "inference_unit": u_p,
        "categories": cat_mat,
    }


def build_companion_agreement_figure(
    compat_info: Dict[str, Any],
    *,
    atlas: Any = None,
    primary_name: str = "Primary",
    comp_name: str = "Companion",
) -> Optional[plt.Figure]:
    """Build discrete network-ordered agreement matrix when companion is compatible."""
    if not compat_info.get("compatible") or compat_info.get("categories") is None:
        return None
    cat = np.asarray(compat_info["categories"], dtype=int)
    n_nodes = cat.shape[0]
    perm, block_starts, block_names, has_nets = resolve_roi_ordering(atlas, n_nodes)
    cat_ord = cat[np.ix_(perm, perm)]

    cmap = ListedColormap(["#BDBDBD", "#F3F4F6", "#2563EB", "#F59E0B", "#16A34A"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(5.8, 4.8), facecolor="white")
    im = ax.imshow(cat_ord, cmap=cmap, norm=norm, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[-1, 0, 1, 2, 3])
    cbar.ax.set_yticklabels(["Not tested", "Neither", f"{primary_name} only", f"{comp_name} only", "Both"], fontsize=7)
    _draw_network_guides(ax, n_nodes, block_starts, block_names, has_nets)
    jacc = compat_info["summary"]["jaccard"]
    unit = compat_info.get("inference_unit", "edge")
    ax.set_title(f"{unit.capitalize()} Detection Agreement (Jaccard = {jacc:.2f})", fontsize=9)
    if unit != "edge":
        ax.set_xlabel(f"{unit.capitalize()} memberships projected onto edges; counts are per {unit}", fontsize=7)
    fig.tight_layout()
    return fig


def build_unit_annotation_context(
    result: Any,
    detected_df: pd.DataFrame,
    rois: Any,
    alpha: float = 0.05,
    top_rois_per_unit: int = 5,
) -> Dict[str, Any]:
    """Aggregate annotation context at the result's statistical inference unit.

    Explicitly documents the ROI selection rule so NBS components and cNBS
    blocks are not framed as collections of independent edge findings or
    statistically significant hub nodes.
    """
    inner = unwrap_result(result)
    unit = getattr(inner, "inference_unit", None)
    if unit not in ("edge", "component", "block"):
        method = str(getattr(inner, "method", "") or "").lower()
        unit = "component" if method == "nbs" else "block" if method == "cnbs" else "edge"

    roi_tbl = rois.table if hasattr(rois, "table") else pd.DataFrame()
    roi_by_idx: Dict[int, Any] = {}
    if not roi_tbl.empty:
        if "roi_index" in roi_tbl.columns:
            roi_by_idx = {int(r["roi_index"]): r for _, r in roi_tbl.iterrows()}
        else:
            roi_by_idx = {int(i): r for i, (_, r) in enumerate(roi_tbl.iterrows())}

    if detected_df is None or detected_df.empty:
        return {
            "inference_unit": unit,
            "roi_selection_rule": (
                f"ROI selection rule ({unit} unit): no detections at α = {alpha:g}; 0 ROIs selected."
            ),
            "unit_table": pd.DataFrame(),
        }

    endpoint_rois = sorted(
        set(int(x) for x in detected_df.get("roi_i", pd.Series(dtype=int)).dropna())
        | set(int(x) for x in detected_df.get("roi_j", pd.Series(dtype=int)).dropna())
    )

    def _summarize_roi_indices(indices: Sequence[int], degree_counts: Dict[int, int]) -> Dict[str, str]:
        ranked = sorted(set(int(i) for i in indices), key=lambda idx: (-degree_counts.get(idx, 0), idx))
        top = ranked[:top_rois_per_unit]
        top_labels = []
        lobes, gyri, domains = [], [], []
        for idx in ranked:
            row = roi_by_idx.get(idx)
            if row is None:
                continue
            lobe = str(row.get("lobe", "")).strip()
            gyrus = str(row.get("gyrus", "")).strip()
            if lobe and lobe not in ("nan", "None", "outside", "unassigned"):
                lobes.append(lobe)
            if gyrus and gyrus not in ("nan", "None", "outside", "unassigned"):
                gyri.append(gyrus)
            doms = row.get("domains")
            if isinstance(doms, (list, tuple, set)):
                domains.extend(str(d).strip() for d in doms if str(d).strip())
            elif isinstance(doms, str) and doms.strip():
                domains.extend(d.strip() for d in re.split(r"[;,]", doms) if d.strip())
        for idx in top:
            row = roi_by_idx.get(idx)
            nm = str(row.get("name") or row.get("roi") or f"ROI {idx}") if row is not None else f"ROI {idx}"
            deg = degree_counts.get(idx, 0)
            top_labels.append(f"{nm} ({deg} edges)" if deg else nm)
        lobe_counts = pd.Series(lobes).value_counts().head(4) if lobes else pd.Series(dtype=int)
        gyrus_counts = pd.Series(gyri).value_counts().head(4) if gyri else pd.Series(dtype=int)
        dom_counts = pd.Series(domains).value_counts().head(5) if domains else pd.Series(dtype=int)
        return {
            "Top ROIs (by within-unit degree)": "; ".join(top_labels) if top_labels else "—",
            "Lobes (ROI counts)": "; ".join(f"{k} ({v})" for k, v in lobe_counts.items()) if not lobe_counts.empty else "—",
            "Gyri (ROI counts)": "; ".join(f"{k} ({v})" for k, v in gyrus_counts.items()) if not gyrus_counts.empty else "—",
            "BrainMap domains (ROI counts)": "; ".join(f"{k} ({v})" for k, v in dom_counts.items()) if not dom_counts.empty else "—",
        }

    if unit == "component":
        comp_report = getattr(inner, "component_report", None)
        rows = []
        if comp_report is not None:
            comp_df = inner.component_table(alpha=alpha, significant_only=True)
            for _, crow in comp_df.iterrows():
                cid = int(crow["component_id"])
                tail = str(crow["tail"])
                if "component_id" in detected_df.columns:
                    mask = detected_df["component_id"] == cid
                    if "component_tail" in detected_df.columns:
                        mask = mask & (detected_df["component_tail"] == tail)
                    sub = detected_df[mask]
                else:
                    sub = pd.DataFrame()
                deg: Dict[int, int] = {}
                for _, er in sub.iterrows():
                    ii, jj = int(er["roi_i"]), int(er["roi_j"])
                    deg[ii] = deg.get(ii, 0) + 1
                    deg[jj] = deg.get(jj, 0) + 1
                nodes = [int(n) for n in crow["nodes"]]
                ann_bits = _summarize_roi_indices(nodes, deg)
                rows.append({
                    "Component": f"{tail} #{cid}",
                    "Edges": int(crow["n_edges"]),
                    "ROIs": int(crow["n_nodes"]),
                    "p (FWER)": float(crow["p_fwer"]),
                    **ann_bits,
                })
        rule = (
            f"ROI selection rule (component unit): ROIs are grouped per significant NBS component at α = {alpha:g} "
            f"({len(rows)} component(s), {len(endpoint_rois)} unique ROIs total). Within each component, the top "
            f"{top_rois_per_unit} ROIs are shown descriptively by within-component edge count; degree ordering is "
            "descriptive only and does not test node/hub significance."
        )
        return {"inference_unit": unit, "roi_selection_rule": rule, "unit_table": pd.DataFrame(rows)}

    if unit == "block":
        rows = []
        report = getattr(inner, "block_report", None)
        if report is not None and report.get("node_labels") is not None:
            node_labels = list(report["node_labels"])
            code_to_name: Dict[Any, str] = {}
            if not roi_tbl.empty and "network" in roi_tbl.columns:
                tmp_map: Dict[Any, set] = {}
                for idx_roi, code in enumerate(node_labels):
                    rrow = roi_by_idx.get(idx_roi)
                    if rrow is not None:
                        net_val = str(rrow.get("network", "")).strip()
                        if net_val and net_val.lower() not in ("nan", "none", "unknown", ""):
                            tmp_map.setdefault(code, set()).add(net_val)
                if tmp_map and all(len(v) == 1 for v in tmp_map.values()):
                    code_to_name = {k: next(iter(v)) for k, v in tmp_map.items()}
            tested_tail = report.get("tested_tail") or getattr(inner, "tested_tail", None)
            for brow in report.get("blocks") or []:
                if tested_tail == "positive":
                    p_blk = float(brow.get("p_positive", np.nan))
                elif tested_tail == "negative":
                    p_blk = float(brow.get("p_negative", np.nan))
                elif "p_two_sided" in brow:
                    p_blk = float(brow["p_two_sided"])
                else:
                    p_blk = float(np.nanmin([brow.get("p_positive", np.nan), brow.get("p_negative", np.nan)]))
                if not np.isfinite(p_blk) or p_blk > alpha:
                    continue
                na, nb = brow["network_a"], brow["network_b"]
                nodes_a = {idx for idx, lbl in enumerate(node_labels) if lbl == na}
                nodes_b = {idx for idx, lbl in enumerate(node_labels) if lbl == nb}
                block_nodes = sorted(nodes_a | nodes_b)
                deg: Dict[int, int] = {}
                edge_cnt = 0
                for _, er in detected_df.iterrows():
                    ii, jj = int(er["roi_i"]), int(er["roi_j"])
                    if (ii in nodes_a and jj in nodes_b) or (ii in nodes_b and jj in nodes_a):
                        edge_cnt += 1
                        deg[ii] = deg.get(ii, 0) + 1
                        deg[jj] = deg.get(jj, 0) + 1
                la = code_to_name.get(na, str(na) if isinstance(na, str) else f"Block {na}")
                lb = code_to_name.get(nb, str(nb) if isinstance(nb, str) else f"Block {nb}")
                pair_name = la if la == lb else f"{la} — {lb}"
                ann_bits = _summarize_roi_indices(block_nodes, deg)
                rows.append({
                    "Block (network pair)": pair_name,
                    "Edges in block": int(brow.get("n_edges", edge_cnt)),
                    "ROIs in block": len(block_nodes),
                    "Block p": p_blk,
                    **ann_bits,
                })
        elif "i_network" in detected_df.columns and "j_network" in detected_df.columns:
            pair_series = [
                f"{a} — {b}" if str(a) <= str(b) else f"{b} — {a}"
                for a, b in zip(detected_df["i_network"], detected_df["j_network"])
            ]
            for pair_name, sub in detected_df.assign(_pair=pair_series).groupby("_pair", sort=False):
                deg = {}
                for _, er in sub.iterrows():
                    ii, jj = int(er["roi_i"]), int(er["roi_j"])
                    deg[ii] = deg.get(ii, 0) + 1
                    deg[jj] = deg.get(jj, 0) + 1
                nodes = sorted(deg)
                ann_bits = _summarize_roi_indices(nodes, deg)
                p_vals = np.fmin(
                    pd.to_numeric(sub.get("p_positive", pd.Series([np.nan])), errors="coerce"),
                    pd.to_numeric(sub.get("p_negative", pd.Series([np.nan])), errors="coerce"),
                )
                p_block = float(np.nanmin(p_vals)) if len(p_vals) else np.nan
                rows.append({
                    "Block (network pair)": str(pair_name),
                    "Edges in block": len(sub),
                    "ROIs in block": len(nodes),
                    "Block p": p_block,
                    **ann_bits,
                })
        rule = (
            f"ROI selection rule (block unit): ROIs are aggregated per significant cNBS network block at α = {alpha:g} "
            f"({len(rows)} block(s), {len(endpoint_rois)} unique ROIs total). Within each block, the top "
            f"{top_rois_per_unit} ROIs are listed descriptively by within-block edge participation; block significance "
            "reflects the block mean rather than every constituent edge or ROI."
        )
        return {"inference_unit": unit, "roi_selection_rule": rule, "unit_table": pd.DataFrame(rows)}

    rule = (
        f"ROI selection rule (edge unit): all {len(endpoint_rois)} unique endpoint ROIs belonging to the "
        f"{len(detected_df)} significant edge(s) at α = {alpha:g}, summarized descriptively without node-level significance."
    )
    return {"inference_unit": unit, "roi_selection_rule": rule, "unit_table": pd.DataFrame()}
