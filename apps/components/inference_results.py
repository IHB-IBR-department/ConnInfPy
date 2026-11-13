import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from apps.utils.ui_state import persistent_key
from apps.utils.workspace import require_analysis_step
import pandas as pd
from io import BytesIO

from conninfpy.plot import plot_connectome_graph
from apps.utils.helpers import (
    active_analysis_atlas,
    atlas_for_annotation,
    atlas_has_coords,
    atlas_has_networks,
    current_contrast_name,
    effect_direction_labels,
    result_is_stale,
    safe_filename_part,
    render_help,
)
from apps.utils.result_views import (
    build_cnbs_block_figure,
    build_companion_agreement_figure,
    build_connectivity_maps_figure,
    build_network_summary_figure,
    build_null_max_figure,
    check_companion_compatibility,
    describe_result_scale_and_stat,
    get_2d_result,
    prepare_cnbs_blocks_table,
    prepare_network_summary,
    prepare_node_summary,
    prepare_predicted_coupling_summary,
    recorded_native_detection_count,
    unwrap_result,
)


def _run_alpha():
    """The α fixed at run time (``run_plan``); result views must not re-choose it.

    Older session states without a recorded alpha fall back to 0.05, the
    value those runs were displayed at.
    """
    return (st.session_state.get("run_plan") or {}).get("alpha", 0.05)


def _family_caption(res):
    """One line explaining which family the run's α applies to."""
    policy = res.tail_policy
    if policy == "two-sided":
        return ("Two-sided family: α covers both directions jointly "
                "(FWER over positive and negative together).")
    if policy == "separate-one-sided":
        return ("Separate one-sided families: α applies to each direction's family "
                "separately; the combined-direction FWER is not α (use α/2 per tail).")
    if policy == "omnibus":
        return "Omnibus F family: α applies to the non-directional F test."
    tested = getattr(res, "tested_tail", None)
    selection = getattr(res, "tail_selection", None)
    origin = ("fixed before the run" if selection == "prespecified"
              else "selected after the run (post hoc)" if selection == "post-hoc"
              else "selection origin not recorded")
    return f"One-sided family ({tested}, {origin}): α applies to the tested direction only."


def _is_omnibus_result(res):
    inner = unwrap_result(res)
    return hasattr(inner, "keys") and "omnibus" in inner and "positive" not in inner


def _select_grid_slice(res_wrapper, slot):
    """If ``res`` is a parameter-grid result, let the user pick one (E, H) slice."""
    inner = unwrap_result(res_wrapper)
    if not getattr(inner, "is_grid", False):
        return inner, None
    raw_e = getattr(inner, "e_grid", None)
    raw_h = getattr(inner, "h_grid", None)
    e_grid = list(raw_e) if raw_e is not None else []
    h_grid = list(raw_h) if raw_h is not None else []
    n_cells = len(e_grid)
    if n_cells <= 1:
        return get_2d_result(inner, 0)
    options = list(range(n_cells))
    idx = st.selectbox(
        "TFNBS parameter-grid cell (E, H)",
        options,
        index=0,
        format_func=lambda k: f"Cell {k}: E={e_grid[k]:g}, H={h_grid[k]:g}" if k < len(h_grid) else f"Cell {k}",
        key=persistent_key(f"results:grid_idx:{slot}", scope="results", options=options),
    )
    return get_2d_result(inner, idx)


def _extract_result_diagnostics_summary(res_wrapper, run_plan=None):
    """Extract flags, backend-resolved strategy, combat_diagnostics, and design_diagnostics.

    ``resolved_strategy`` always reflects the actual backend execution decision
    recorded on the result / ``combat_diagnostics`` (or ``'unknown / not recorded'``
    for legacy results lacking backend markers). Requested ``run_plan`` arguments
    are reported separately and are never substituted as the resolved strategy.
    """
    from conninfpy._compat import TailResult

    inner = getattr(res_wrapper, "inference", res_wrapper)
    meta = getattr(res_wrapper, "metadata", None)
    if isinstance(res_wrapper, dict) and not isinstance(res_wrapper, TailResult):
        flags = list(res_wrapper.get("flags") or [])
        combat_diag = res_wrapper.get("combat_diagnostics")
        design_diag = res_wrapper.get("design_diagnostics")
        explicit_resolved = res_wrapper.get("resolved_strategy")
    elif isinstance(meta, dict):
        flags = list(meta.get("flags") or [])
        combat_diag = meta.get("combat_diagnostics")
        design_diag = meta.get("design_diagnostics")
        explicit_resolved = meta.get("resolved_strategy")
    else:
        flags = list(
            getattr(res_wrapper, "flags", None)
            if getattr(res_wrapper, "flags", None) is not None
            else (getattr(inner, "flags", None) or [])
        )
        combat_diag = getattr(res_wrapper, "combat_diagnostics", None)
        if combat_diag is None:
            combat_diag = getattr(inner, "combat_diagnostics", None)
        design_diag = getattr(res_wrapper, "design_diagnostics", None)
        if design_diag is None:
            design_diag = getattr(inner, "design_diagnostics", None)
        explicit_resolved = getattr(res_wrapper, "resolved_strategy", None)
        if explicit_resolved is None:
            explicit_resolved = getattr(inner, "resolved_strategy", None)

    plan = run_plan or {}
    if isinstance(explicit_resolved, str) and explicit_resolved.strip():
        resolved_strategy = explicit_resolved.strip()
    elif isinstance(combat_diag, dict) and combat_diag.get("strategy"):
        resolved_strategy = str(combat_diag.get("strategy"))
    elif getattr(inner, "harmonized", None) is True:
        resolved_strategy = "combat (strategy not recorded)"
    else:
        resolved_strategy = "unknown / not recorded"

    return {
        "resolved_strategy": str(resolved_strategy),
        "site_recipe": plan.get("site_recipe"),
        "effective_site_recipe": plan.get("effective_site_recipe"),
        "harmonize_requested": plan.get("harmonize"),
        "flags": flags,
        "combat_diagnostics": combat_diag,
        "design_diagnostics": design_diag,
    }


def _format_inference_df_display(design_diag):
    """Format inference-level df cleanly without conflating Welch/available_case with n - rank."""
    kind = design_diag.get("inference_df_kind")
    inf_df = design_diag.get("inference_df")
    if kind == "shared_design_residual_df" and inf_df is not None:
        return str(inf_df)
    if kind in ("varying_welch_satterthwaite", "available_case_welch_satterthwaite"):
        return "Per-edge (Welch)"
    if kind == "available_case_per_edge":
        ac = design_diag.get("available_case")
        if isinstance(ac, dict) and ac.get("edge_residual_df_min") is not None and ac.get("edge_residual_df_max") is not None:
            lo, hi = ac["edge_residual_df_min"], ac["edge_residual_df_max"]
            return f"{lo}" if lo == hi else f"{lo}–{hi} (per edge)"
        return "Per-edge (available-case)"
    return "See scope note"


def _render_analyze_decisions_and_diagnostics(res_wrapper, run_plan=None, method_name=""):
    """Render analyze() flags, resolved strategy, combat_diagnostics, and design_diagnostics."""
    info = _extract_result_diagnostics_summary(res_wrapper, run_plan=run_plan)
    flags = info["flags"]
    combat_diag = info["combat_diagnostics"]
    design_diag = info["design_diagnostics"]

    has_any = bool(
        flags
        or combat_diag is not None
        or design_diag is not None
        or info["harmonize_requested"] is not None
        or info["effective_site_recipe"] is not None
        or info["resolved_strategy"] not in ("none", "unknown / not recorded")
    )
    if not has_any:
        return

    title_suffix = f" ({method_name})" if method_name else ""
    with st.expander(f"🔎 Analyze Decisions, Harmonization & Design Diagnostics{title_suffix}", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Resolved Strategy", info["resolved_strategy"])
        if info["harmonize_requested"] is not None:
            c2.metric("Requested Harmonize", str(info["harmonize_requested"]))
        elif info["effective_site_recipe"]:
            c2.metric("Requested Site Recipe", str(info["effective_site_recipe"]))
        else:
            c2.metric("Requested Harmonize", "not recorded")

        if isinstance(combat_diag, dict) and combat_diag.get("between_site_variance_ratio_after_over_before") is not None:
            c3.metric(
                "Between-Site Var Ratio (After/Before)",
                f"{combat_diag['between_site_variance_ratio_after_over_before']:.3f}",
            )
        elif design_diag is not None:
            c3.metric(
                "Design Rank / Columns",
                f"{design_diag.get('matrix_rank', '?')} / {design_diag.get('n_columns', '?')}",
            )
        if info["effective_site_recipe"] and info["harmonize_requested"] is not None:
            st.caption(f"Requested App Site Recipe: `{info['effective_site_recipe']}`")

        if flags:
            st.markdown("**Analyze Decision Flags (`flags`):**")
            for fl in flags:
                st.info(str(fl))

        if isinstance(combat_diag, dict):
            st.markdown("**ComBat / Site Adjustment Diagnostics (`combat_diagnostics`):**")
            cd_rows = [
                {"Metric": str(k), "Value": str(v)}
                for k, v in combat_diag.items()
                if not isinstance(v, (dict, list))
            ]
            if cd_rows:
                st.dataframe(pd.DataFrame(cd_rows), hide_index=True, width="stretch")

        if isinstance(design_diag, dict):
            st.markdown("**Design Diagnostics (`design_diagnostics`):**")
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("Observations", design_diag.get("n_observations", "—"))
            d2.metric(
                "Design df (n − rank)",
                design_diag.get("design_residual_df", design_diag.get("residual_df", "—")),
            )
            d3.metric("Inference df", _format_inference_df_display(design_diag))
            d4.metric("Rank Deficient", "Yes" if design_diag.get("rank_deficient") else "No")
            vmax = design_diag.get("vif_max")
            d5.metric("Max VIF", f"{vmax:.2f}" if isinstance(vmax, (int, float)) else "—")

            st.caption(
                f"Design: `{design_diag.get('design_kind')}` | "
                f"Unit: `{design_diag.get('unit_of_observation')}` | "
                f"Scope: `{design_diag.get('scope')}`"
            )
            if design_diag.get("design_note"):
                st.caption(design_diag["design_note"])
            if design_diag.get("inference_df_note"):
                st.caption(design_diag["inference_df_note"])
            if design_diag.get("scope_note"):
                st.caption(design_diag["scope_note"])

            cond_raw = design_diag.get("condition_number_XtX")
            cond_std = design_diag.get("condition_number_standardized")
            cond_raw_str = "∞" if design_diag.get("condition_number_XtX_infinite") else (f"{cond_raw:.2f}" if isinstance(cond_raw, (int, float)) else "—")
            cond_std_str = f"{cond_std:.2f}" if isinstance(cond_std, (int, float)) else "—"
            st.caption(
                f"Raw cond(X'X) = {cond_raw_str} | Standardized cond = {cond_std_str} — "
                + str(design_diag.get("condition_number_note", ""))
            )

            col_summaries = design_diag.get("column_summaries") or []
            if col_summaries:
                st.dataframe(pd.DataFrame(col_summaries), hide_index=True, width="stretch")

            ss = design_diag.get("site_summary")
            if isinstance(ss, dict):
                if ss.get("site_by_interest_table"):
                    st.markdown("**Site × Group / Factor Counts:**")
                    tbl_df = pd.DataFrame.from_dict(ss["site_by_interest_table"], orient="index").reset_index().rename(columns={"index": "Site"})
                    st.dataframe(tbl_df, hide_index=True, width="stretch")
                elif ss.get("site_by_interest_stats"):
                    st.markdown("**Site × Predictor Stats:**")
                    st_df = pd.DataFrame.from_dict(ss["site_by_interest_stats"], orient="index").reset_index().rename(columns={"index": "Site"})
                    st.dataframe(st_df, hide_index=True, width="stretch")
                    if ss.get("descriptive_r2_site_on_interest") is not None:
                        st.caption(f"Descriptive between-site R² on predictor: {ss['descriptive_r2_site_on_interest']:.3f}")

            ac = design_diag.get("available_case")
            if isinstance(ac, dict):
                df_part = ""
                if ac.get("edge_residual_df_min") is not None and ac.get("edge_residual_df_max") is not None:
                    df_part = f"; per-edge residual df = {ac.get('edge_residual_df_min')}–{ac.get('edge_residual_df_max')}"
                st.caption(
                    f"Available-case edge coverage: {ac.get('n_edges_tested')} tested, "
                    f"{ac.get('n_edges_untested')} untested; "
                    f"per-edge n = {ac.get('edge_n_min')}–{ac.get('edge_n_max')} (median {ac.get('edge_n_median')}){df_part}."
                )

            for dfl in design_diag.get("flags") or []:
                st.warning(f"Design diagnostic: {dfl}")


def _render_one_sentence_summary(res, edges_df, method_name, direction_labels):
    """Render the single-sentence plain-language result summary."""
    alpha = _run_alpha()
    meta = describe_result_scale_and_stat(res, st.session_state.get("run_plan"))
    n_perms = getattr(res, "n_permutations", 100)
    if meta["inference_unit"] in ("component", "block"):
        count = recorded_native_detection_count(res, alpha)
        count_text = str(count) if count is not None else "not recorded"
        st.info(
            f"📝 **Summary:** The {method_name} run detected **{count_text}** significant "
            f"{meta['inference_unit']}s at α = {alpha:g} using {n_perms} permutations. "
            f"{meta['inference_unit_note']}"
        )
        return
    if meta["is_omnibus"]:
        nsig = res.n_significant(alpha).get("omnibus", 0)
        st.info(
            f"📝 **Summary:** The omnibus F-test ({method_name}) detected **{nsig}** significant edges "
            f"(non-directional, α = {alpha:g}) using {n_perms} permutations. "
            f"{meta['inference_unit_note']}"
        )
        return

    pos_count = 0
    neg_count = 0
    if edges_df is not None and not edges_df.empty and "t_signed" in edges_df.columns:
        pos_count = len(edges_df[edges_df["t_signed"] > 0])
        neg_count = len(edges_df[edges_df["t_signed"] < 0])
    else:
        nsig = res.n_significant(alpha)
        pos_count = int(nsig.get("positive", 0))
        neg_count = int(nsig.get("negative", 0))

    tested = getattr(res, "tested_tail", None)
    if tested is not None:
        count = pos_count if tested == "positive" else neg_count
        st.info(
            f"📝 **Summary:** The one-sided {method_name} test of **{direction_labels[tested]}** "
            f"detected **{count}** significant edges using {n_perms} permutations. "
            f"The opposite direction was not tested. {meta['inference_unit_note']}"
        )
        return
    st.info(
        f"📝 **Summary:** The {method_name} run detected **{pos_count}** significant edges for **{direction_labels['positive']}** "
        f"and **{neg_count}** edges for **{direction_labels['negative']}** using {n_perms} permutations. "
        f"{meta['inference_unit_note']}"
    )


def _shown_tail_counts(res, alpha):
    """Per-tail counts with an untested direction labelled 'not tested'."""
    nsig = res.n_significant(alpha)
    if _is_omnibus_result(res):
        return {"omnibus": nsig.get("omnibus", 0)}
    tested = getattr(res, "tested_tail", None)
    return {
        tail: ("not tested" if tested is not None and tail != tested else nsig[tail])
        for tail in ("positive", "negative")
    }


def _render_summary_metrics(res, edges_df, method_name, direction_labels):
    """Render the headline metric row for one result."""
    st.markdown(f"##### 📊 {method_name} Summary")
    alpha = _run_alpha()
    meta = describe_result_scale_and_stat(res, st.session_state.get("run_plan"))
    shown = _shown_tail_counts(res, alpha)
    if meta["inference_unit"] in ("component", "block"):
        col1, col2, col3 = st.columns(3)
        count = recorded_native_detection_count(res, alpha)
        col1.metric(f"Significant {meta['inference_unit'].capitalize()}s", count if count is not None else "not recorded")
        col2.metric("Statistic / Scale", f"{meta['stat_symbol']} / {meta['scale_label']}")
        col3.metric("Permutations Completed", res.n_permutations)
    elif meta["is_omnibus"]:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Omnibus F Detections", shown["omnibus"], help=f"Edges significant on omnibus F-test at α = {alpha:g}.")
        with col2:
            st.metric("Statistic / Scale", f"{meta['stat_symbol']} / {meta['scale_label']}")
        with col3:
            st.metric("Permutations Completed", res.n_permutations)
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(direction_labels["positive"], shown["positive"], help=f"Edges significant in the positive statistical tail at α = {alpha:g}.")
        with col2:
            st.metric(direction_labels["negative"], shown["negative"], help=f"Edges significant in the negative statistical tail at α = {alpha:g}.")
        with col3:
            st.metric("Permutations Completed", res.n_permutations)
    st.caption(
        f"Detections at α = {alpha:g}. {_family_caption(res)} "
        f"Test statistic: `{meta['stat_name']}` | Input scale: `{meta['scale_label']}` | Unit: `{meta['inference_unit']}`."
    )


def _render_ground_truth_metrics(edges_df, effect_mask, method_name):
    """Render precision/recall against the planted effect mask, if any."""
    if effect_mask is not None:
        st.markdown(f"##### 🎯 Ground Truth Comparison ({method_name})")
        st.caption("Descriptive edge-membership coverage against the saved truth mask; these ratios do not estimate component/block FWER or node significance.")
        true_mask = (effect_mask > 0)
        true_mask_triu = np.triu(true_mask, 1)

        n_nodes = true_mask.shape[0]
        possible_edges = int(n_nodes * (n_nodes - 1) / 2)

        n_true_pos_edges = int(np.sum(true_mask_triu))
        n_true_neg_edges = possible_edges - n_true_pos_edges

        detected_edges_set = set()
        if edges_df is not None and not edges_df.empty:
            for _, row in edges_df.iterrows():
                i = int(row["roi_i"])
                j = int(row["roi_j"])
                u, v = min(i, j), max(i, j)
                detected_edges_set.add((u, v))

        tp_count = 0
        fp_count = 0
        for u, v in detected_edges_set:
            if true_mask_triu[u, v]:
                tp_count += 1
            else:
                fp_count += 1

        fn_count = n_true_pos_edges - tp_count

        tp_rate = (tp_count / n_true_pos_edges) if n_true_pos_edges > 0 else 0.0
        fp_rate = (fp_count / n_true_neg_edges) if n_true_neg_edges > 0 else 0.0
        fn_rate = (fn_count / n_true_pos_edges) if n_true_pos_edges > 0 else 0.0

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Sensitivity / Power (TPR)", f"{tp_rate*100:.1f}%", help=f"True Positives: {tp_count} detected out of {n_true_pos_edges} active ground-truth edges")
        with m_col2:
            st.metric("Type I Error Rate (FPR)", f"{fp_rate*100:.1f}%", help=f"False Positives: {fp_count} detected out of {n_true_neg_edges} null ground-truth edges")
        with m_col3:
            st.metric("Type II Error Rate (FNR)", f"{fn_rate*100:.1f}%", help=f"False Negatives: {fn_count} missed out of {n_true_pos_edges} active ground-truth edges")

        if len(detected_edges_set) > 0:
            fdr = fp_count / len(detected_edges_set)
            st.markdown(f"**False Discovery Proportion (FDP):** `{fdr*100:.1f}%` (false edge memberships among selected edge memberships in this saved example)")


def _download_figure_button(fig, label, filename, key):
    """Render a figure download button with a unique widget key."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    st.download_button(
        label,
        data=buffer.getvalue(),
        file_name=filename,
        mime="image/png",
        key=key,
    )


def _render_heatmaps(res, base_atlas, method_name, contrast_name="contrast", slot="primary", effect_mask=None):
    """Render aligned statistic, signed -log10(p), native contrast effect, and optional truth mask."""
    st.markdown(f"##### 🗺️ Connectivity Maps ({method_name})")
    alpha = _run_alpha()
    atlas = _active_atlas(base_atlas)
    fig, info = build_connectivity_maps_figure(
        res,
        atlas=atlas,
        alpha=alpha,
        method_name=method_name,
        run_plan=st.session_state.get("run_plan"),
        effect_mask=effect_mask,
    )
    if fig is None:
        st.info("Statistic matrix unavailable on this result.")
        return

    order_desc = (
        f"grouped by atlas networks ({', '.join(info['network_order'])})"
        if info["ordered_by_network"]
        else "matrix index order (no usable atlas network labels)"
    )
    effect_note = (
        f" Native contrast effect (`c' β`) is shown separately in `{info['effect_unit_label']}`."
        if info["has_effect_map"]
        else " Native contrast effect (`c' β`) unavailable (model coefficients / 1D contrast not recorded)."
    )
    st.caption(
        f"Statistic panel shows `{info['stat_name']}` (test statistic, not standardized effect size); "
        f"input scale: `{info['scale_label']}`. ROI ordering: {order_desc}.{effect_note}"
    )
    st.pyplot(fig)
    _download_figure_button(
        fig,
        f"Download {method_name} Connectivity Maps PNG",
        f"connectivity_maps_{safe_filename_part(contrast_name)}_{safe_filename_part(method_name)}.png",
        key=f"heatmaps_download_{slot}_{safe_filename_part(method_name)}",
    )
    plt.close(fig)


def _render_network_and_node_summaries(res, base_atlas, method_name, contrast_name, slot):
    """Render network-pair summary (above edge table) and descriptive node summary."""
    alpha = _run_alpha()
    atlas = _active_atlas(base_atlas)
    meta = describe_result_scale_and_stat(res, st.session_state.get("run_plan"))

    st.markdown(f"##### 🌐 Network-Pair Summary ({method_name})")
    net_df, net_reason = prepare_network_summary(res, atlas=atlas, alpha=alpha)
    if net_df is None:
        st.info(f"Network-pair summary unavailable: {net_reason}")
    else:
        st.caption(
            f"Descriptive aggregation of significant detections at α = {alpha:g} by atlas network pair "
            f"(same ROI network labels used for matrix ordering). Inference unit remains `{meta['inference_unit']}`."
        )
        fig = build_network_summary_figure(res, atlas=atlas, alpha=alpha, method_name=method_name)
        if fig is not None:
            st.pyplot(fig)
            _download_figure_button(
                fig,
                f"Download {method_name} Network Summary PNG",
                f"network_summary_{safe_filename_part(contrast_name)}_{safe_filename_part(method_name)}.png",
                key=f"network_summary_png_{slot}_{safe_filename_part(method_name)}",
            )
            plt.close(fig)
        st.dataframe(net_df, width="stretch", hide_index=True)
        st.download_button(
            f"Download {method_name} Network Summary CSV",
            data=net_df.to_csv(index=False).encode("utf-8"),
            file_name=f"network_summary_{safe_filename_part(contrast_name)}_{safe_filename_part(method_name)}.csv",
            mime="text/csv",
            key=f"network_summary_csv_{slot}_{safe_filename_part(method_name)}",
        )

    st.markdown(f"##### 🧩 Node Summary ({method_name}, descriptive)")
    node_df, node_reason = prepare_node_summary(res, atlas=atlas, alpha=alpha)
    if node_df is None:
        st.info(f"Node summary unavailable: {node_reason}")
    else:
        st.caption(
            f"Descriptive per-ROI summary in the `{meta['inference_unit']}` inference unit at α = {alpha:g}. "
            "Counts reflect how many detected units touch each ROI and do NOT constitute node-level significance tests or hub p-values."
        )
        st.dataframe(node_df, width="stretch", hide_index=True)
        st.download_button(
            f"Download {method_name} Node Summary CSV",
            data=node_df.to_csv(index=False).encode("utf-8"),
            file_name=f"node_summary_{safe_filename_part(contrast_name)}_{safe_filename_part(method_name)}.csv",
            mime="text/csv",
            key=f"node_summary_csv_{slot}_{safe_filename_part(method_name)}",
        )


def _render_blocks_table(res, base_atlas, method_name, contrast_name, direction_labels, slot):
    """cNBS only: block-level visual matrix, table from ``block_report``, and CSV download."""
    report = getattr(res, "block_report", None)
    if report is None:
        return
    st.markdown(f"##### 🧱 cNBS Network Blocks ({method_name})")
    alpha = _run_alpha()
    atlas = _active_atlas(base_atlas)
    df, info = prepare_cnbs_blocks_table(res, atlas=atlas, alpha=alpha, direction_labels=direction_labels)
    if df is None or df.empty:
        st.info("No network-block statistics recorded on this cNBS result.")
        return
    st.caption(
        f"cNBS tests the mean signed edge statistic within each network block across {info['n_blocks_tested']} tested "
        f"block(s) using `{info['correction']}` correction ({info['n_significant']} significant at α = {alpha:g}). "
        "Every edge inside a block shares the block's p-value."
    )
    fig = build_cnbs_block_figure(res, atlas=atlas, alpha=alpha, method_name=method_name)
    if fig is not None:
        st.pyplot(fig)
        _download_figure_button(
            fig,
            f"Download {method_name} cNBS Blocks PNG",
            f"cnbs_blocks_{safe_filename_part(contrast_name)}_{safe_filename_part(method_name)}.png",
            key=f"cnbs_blocks_png_{slot}_{safe_filename_part(method_name)}",
        )
        plt.close(fig)
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button(
        f"Download {method_name} Blocks CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"cnbs_blocks_{safe_filename_part(contrast_name)}_{safe_filename_part(method_name)}.csv",
        mime="text/csv",
        key=f"cnbs_blocks_csv_{slot}_{safe_filename_part(method_name)}",
    )


def _render_predicted_and_coupling(res, base_atlas, method_name, contrast_name, slot):
    """Render covariate profile predictions (``predicted()``) and ``coupling()``."""
    st.markdown(f"##### 🧭 Predicted Profiles & Coupling ({method_name})")
    alpha = _run_alpha()
    atlas = _active_atlas(base_atlas)
    info = prepare_predicted_coupling_summary(
        res, atlas=atlas, alpha=alpha, run_plan=st.session_state.get("run_plan")
    )
    if not info["available"]:
        st.info(info["reason"])
        return

    st.caption(info["scale_explanation"])
    if info.get("scale_note"):
        st.caption(info["scale_note"])
    if info.get("fisher_note"):
        st.caption(f"Fisher transform: {info['fisher_note']}")
    if info.get("coupling_note"):
        st.info(info["coupling_note"])
    if info["profile_summaries"] is not None and not info["profile_summaries"].empty:
        st.markdown("**Covariate Profiles (whole-connectome summary):**")
        st.dataframe(info["profile_summaries"], width="stretch", hide_index=True)

    if info["comparison"] is not None:
        name_a, name_b = info["comparison"]
        description = ("Coupling pattern breakdown" if info["coupling_counts_df"] is not None
                       else "Model profile comparison")
        st.caption(
            f"{description} for **{name_a} → {name_b}** over {info['scope_label']} "
            "(descriptive effect classification only; not a physiological or causal claim)."
        )
        if info["coupling_counts_df"] is not None and not info["coupling_counts_df"].empty:
            st.dataframe(info["coupling_counts_df"], width="stretch", hide_index=True)
        if info["edge_predictions_df"] is not None and not info["edge_predictions_df"].empty:
            st.dataframe(info["edge_predictions_df"], width="stretch", hide_index=True)
            st.download_button(
                f"Download {method_name} Predicted Profiles CSV",
                data=info["edge_predictions_df"].to_csv(index=False).encode("utf-8"),
                file_name=f"predicted_coupling_{safe_filename_part(contrast_name)}_{safe_filename_part(method_name)}.csv",
                mime="text/csv",
                key=f"predicted_coupling_csv_{slot}_{safe_filename_part(method_name)}",
            )


def _render_null_max_distribution(res, method_name, contrast_name, slot):
    """Render ``plot_null_max_distribution`` when ``null_max_dist`` was saved."""
    st.markdown(f"##### 📈 Null Max-Statistic Distribution ({method_name})")
    fig, info = build_null_max_figure(res, method_name=method_name)
    if fig is None:
        st.caption(info["reason"])
        return
    st.caption(
        "Empirical permutation null distribution of the family-wise maximum statistic "
        f"for tested tail(s): {', '.join(info['tails'])}."
    )
    if info.get("observed_max_note"):
        st.caption(info["observed_max_note"])
    st.pyplot(fig)
    _download_figure_button(
        fig,
        f"Download {method_name} Null Max Distribution PNG",
        f"null_max_{safe_filename_part(contrast_name)}_{safe_filename_part(method_name)}.png",
        key=f"null_max_png_{slot}_{safe_filename_part(method_name)}",
    )
    plt.close(fig)


def _active_atlas(base_atlas):
    """Atlas to annotate results with, honoring any workspace override."""
    return active_analysis_atlas(base_atlas)


def _filter_edges_for_tail(edges_df, atlas, tail, alpha, top_n, rank_by, selected_networks, network_filter_mode):
    """Filter the edge table to one tail / network / top-N selection."""
    if edges_df is None or edges_df.empty:
        return edges_df.copy() if edges_df is not None else pd.DataFrame()

    p_col = "p_positive" if tail == "positive" else "p_negative"
    df = edges_df.copy()
    df = df[df[p_col] <= alpha]

    if selected_networks and atlas_has_networks(atlas):
        networks = np.asarray(atlas.networks, dtype=object)
        roi_i_net = df["roi_i"].astype(int).map(lambda idx: networks[idx])
        roi_j_net = df["roi_j"].astype(int).map(lambda idx: networks[idx])
        selected = set(selected_networks)
        if network_filter_mode == "Both endpoints":
            keep = roi_i_net.isin(selected) & roi_j_net.isin(selected)
        else:
            keep = roi_i_net.isin(selected) | roi_j_net.isin(selected)
        df = df[keep]

    if df.empty:
        return df

    if rank_by == "|t|":
        df = df.assign(_rank_value=df["t_signed"].abs())
        df = df.sort_values("_rank_value", ascending=False)
    else:
        df = df.sort_values(p_col, ascending=True)

    return df.head(int(top_n)).drop(columns=[c for c in ["_rank_value"] if c in df.columns])


def _render_brain_connectome_graphs(edges_df, base_atlas, method_name, direction_labels, res):
    """Render glass-brain connectome graphs of the significant edges."""
    atlas = _active_atlas(base_atlas)
    st.markdown(f"##### 🧠 Brain-Space Effect Graphs ({method_name})")

    meta = describe_result_scale_and_stat(res, st.session_state.get("run_plan"))
    if meta["inference_unit"] in ("component", "block"):
        st.caption(
            f"Note ({meta['inference_unit']} inference unit): edges plotted below belong to significant "
            f"{meta['inference_unit']}s and carry their {meta['inference_unit']}-level p-value rather than "
            "independent edge-level significance."
        )
    if meta["is_omnibus"]:
        st.info("3D directional glass-brain plotting is defined for directional t/β contrasts; omnibus F-test detections are shown in the connectivity matrices, network summary, and edge table.")
        return

    if atlas is None:
        st.info("Brain graph plotting requires atlas metadata with ROI coordinates.")
        return
    if not atlas_has_coords(atlas):
        st.info("Brain graph plotting requires complete atlas x/y/z coordinates.")
        return
    tested_tail = getattr(res, "tested_tail", None)
    if tested_tail in ("positive", "negative"):
        other_tail = "negative" if tested_tail == "positive" else "positive"
        st.caption(f"Only {direction_labels[tested_tail]} was tested. To test {direction_labels[other_tail]}, change the hypothesis direction in Design and run a new analysis.")
    if edges_df is None or edges_df.empty:
        st.info("No significant edges available to plot.")
        return

    network_options = list(dict.fromkeys(str(n) for n in atlas.networks)) if atlas_has_networks(atlas) else []
    alpha_key = f"{method_name}_graph_alpha"
    direction_key = f"{method_name}_graph_direction"
    result_key = f"_{method_name}_graph_result_id"
    run_alpha_key = f"_{method_name}_graph_run_alpha"
    run_alpha = float(_run_alpha())
    if (
        st.session_state.get(result_key) != id(res)
        or st.session_state.get(run_alpha_key) != run_alpha
    ):
        st.session_state.pop(alpha_key, None)
        st.session_state.pop(direction_key, None)
        st.session_state[result_key] = id(res)
        st.session_state[run_alpha_key] = run_alpha
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 1])
    with ctrl1:
        alpha = st.slider("Graph p-value cutoff", 0.001, 0.100, run_alpha, 0.001,
                          help="Defaults to the α of the run; raise or lower it only to show more or fewer edges.",
                          key=persistent_key(alpha_key, scope='results'))
        top_n = st.slider("Top edges per direction", 10, 500, 100, 10, key=persistent_key(f"{method_name}_graph_top_n", scope='results'))
    with ctrl2:
        display_mode = st.selectbox("Brain view", ["z", "ortho", "x", "y"], index=0, key=persistent_key(f"{method_name}_graph_display", scope='results'))
        weight_by = st.selectbox("Edge weight", ["-log10(p)", "|t|"], index=0, key=persistent_key(f"{method_name}_graph_weight", scope='results'))
    with ctrl3:
        color_options = ["Network", "Degree"]
        if getattr(atlas, "hemisphere", None) is not None:
            color_options.insert(1, "Hemisphere")
        color_by = st.selectbox("Node color", color_options, index=0, key=persistent_key(f"{method_name}_graph_color", scope='results', options=color_options))
        rank_by = st.selectbox("Top edge ranking", ["p-value", "|t|"], index=0, key=persistent_key(f"{method_name}_graph_rank", scope='results'))
    with ctrl4:
        selected_networks = []
        network_filter_mode = "Any endpoint"
        if network_options:
            selected_networks = st.multiselect("Networks", network_options, default=[], key=persistent_key(f"{method_name}_graph_networks", scope='results', options=network_options, multiple=True))
            network_filter_mode = st.radio("Network filter", ["Any endpoint", "Both endpoints"], horizontal=True, key=persistent_key(f"{method_name}_graph_network_mode", scope='results'))
        else:
            st.caption("No network labels available for filtering.")

    tails = [tested_tail] if tested_tail in ("positive", "negative") else ["positive", "negative"]
    filtered = {
        tail: _filter_edges_for_tail(edges_df, atlas, tail, alpha, top_n, rank_by, selected_networks, network_filter_mode)
        for tail in tails
    }
    default_tail = max(tails, key=lambda tail: len(filtered[tail]))
    tail = st.radio(
        "Effect direction", tails, index=tails.index(default_tail), horizontal=True,
        format_func=lambda value: f"{direction_labels[value]} ({len(filtered[value])} edges)",
        key=persistent_key(direction_key, scope="results", options=tails),
    )
    if filtered[tail].empty:
        st.info(f"No edges for {direction_labels[tail]} pass the current graph filters.")
        st.caption("Try raising the p-value cutoff or clearing network filters.")
        return

    fig = plot_connectome_graph(
        edges_df=edges_df, atlas=atlas, tail=tail, alpha=alpha, top_n=top_n,
        display_mode=display_mode, color_by=color_by, weight_by=weight_by,
        rank_by=rank_by, selected_networks=selected_networks,
        network_filter_mode=network_filter_mode, title=direction_labels[f"{tail}_title"],
    )
    st.pyplot(fig, width="stretch")
    _download_figure_button(fig, f"Download {direction_labels[tail]} graph PNG", f"{safe_filename_part(direction_labels[tail])}_connectome_{safe_filename_part(method_name)}.png", f"{method_name}_{tail}_graph_download")
    plt.close(fig)


def _render_table_and_download(edges_df, contrast_name, file_suffix, res=None):
    """Render the edge table with a CSV download button."""
    st.markdown("##### 📋 Significant Edges Table")
    if res is not None:
        meta = describe_result_scale_and_stat(res, st.session_state.get("run_plan"))
        if meta["inference_unit"] in ("component", "block"):
            st.caption(
                f"Unit caveat ({meta['inference_unit']}): each listed edge inherits its {meta['inference_unit']}-level "
                "p-value; rows are member edges of detected units, not separately corrected edge findings."
            )
    if edges_df is None or edges_df.empty:
        st.info("No significant edges detected at this α.")
        return
    st.dataframe(edges_df, width="stretch")

    csv_data = edges_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        f"Download {file_suffix} Significant Edges CSV",
        data=csv_data,
        file_name=f"significant_edges_{safe_filename_part(contrast_name)}_{file_suffix}.csv",
        mime="text/csv",
        key=f"edges_download_{safe_filename_part(file_suffix)}",
    )


def _components_for_display(table, direction_labels):
    """The component table with reader-facing headers and one-line cells."""
    out = pd.DataFrame({
        "Direction": table["tail"].map(lambda tail: direction_labels.get(tail, tail)),
        "Component": table["component_id"],
        "Edges": table["n_edges"],
        "ROIs": table["n_nodes"],
        f"Statistic ({table.attrs['statistic_type']})": table["statistic"],
        "p (FWER)": table["p_fwer"],
        "Significant": table["significant"],
    })
    if "roi_names" in table:
        out["ROI names"] = table["roi_names"].map("; ".join)
    else:
        out["ROI indices"] = table["nodes"].map(lambda nodes: ", ".join(str(n) for n in nodes))
    if "networks" in table:
        out["Networks (ROIs)"] = table["networks"].map("; ".join)
        out["Network pairs (edges)"] = table["network_pairs"].map("; ".join)
    return out


def _render_components_table(res, base_atlas, method_name, contrast_name, direction_labels, slot):
    """NBS only: the observed components with their FWER p-values and a CSV download."""
    report = getattr(res, "component_report", None)
    if report is None:
        return
    st.markdown(f"##### 🧩 NBS Components ({method_name})")
    alpha = _run_alpha()
    n_nodes = next(iter(report["labels"].values())).shape[0]
    atlas = atlas_for_annotation(_active_atlas(base_atlas), n_nodes, warn=False)
    table = res.component_table(atlas=atlas, alpha=alpha)
    threshold = report["threshold"]
    if table.empty:
        st.info(f"No edge exceeds the NBS threshold ({threshold:g}), so there is no component to test.")
        return
    shown = _components_for_display(table, direction_labels)
    n_sig = int(table["significant"].sum())
    st.caption(
        f"A component is a connected set of edges above the threshold ({threshold:g}). Its p-value is "
        f"FWER-corrected: it comes from the permutation distribution of the largest component statistic "
        f"({report['stat_type']}: " + ("number of edges" if report["stat_type"] == "extent"
                                       else "sum of statistic minus threshold") + "); every edge of a component "
        f"has that p-value. All observed components are listed; {n_sig} of {len(table)} have p ≤ {alpha:g}. "
        "The Component column is the component_id of the edge table below."
    )
    st.dataframe(shown, width="stretch", hide_index=True)
    st.download_button(
        f"Download {method_name} Components CSV",
        data=shown.to_csv(index=False).encode("utf-8"),
        file_name=f"nbs_components_{safe_filename_part(contrast_name)}_{method_name}.csv",
        mime="text/csv",
        key=f"components_download_{slot}",
    )


def _render_comparative_view(primary_res, primary_edges, comp_res, comp_edges, primary_name, comp_name, base_atlas, effect_mask, direction_labels, contrast_name="contrast"):
    """Render the side-by-side two-method comparison view with explicit compatibility gating."""
    st.markdown("#### 🔄 Side-by-Side Comparison")
    alpha = _run_alpha()
    p_2d, _ = get_2d_result(primary_res, 0)
    c_2d, _ = get_2d_result(comp_res, 0)
    st.caption(f"Detections and maps at the run's α = {alpha:g}. {_family_caption(p_2d)}")

    # 1. Compare summary metrics in two columns
    col_p, col_c = st.columns(2)
    with col_p:
        st.markdown(f"**Primary Analysis ({primary_name})**")
        shown_p = _shown_tail_counts(p_2d, alpha)
        if _is_omnibus_result(p_2d):
            st.write(f"- **Omnibus F:** `{shown_p['omnibus']}`")
        else:
            st.write(f"- **{direction_labels['positive']}:** `{shown_p['positive']}`")
            st.write(f"- **{direction_labels['negative']}:** `{shown_p['negative']}`")
        if getattr(p_2d, "inference_unit", "edge") != "edge":
            st.caption(f"Native {p_2d.inference_unit} detections: {recorded_native_detection_count(p_2d, alpha)}; tail counts above describe projected edge membership.")
        st.write(f"- **Permutations:** `{p_2d.n_permutations}`")
    with col_c:
        st.markdown(f"**Sensitivity Companion ({comp_name})**")
        shown_c = _shown_tail_counts(c_2d, alpha)
        if _is_omnibus_result(c_2d):
            st.write(f"- **Omnibus F:** `{shown_c['omnibus']}`")
        else:
            st.write(f"- **{direction_labels['positive']}:** `{shown_c['positive']}`")
            st.write(f"- **{direction_labels['negative']}:** `{shown_c['negative']}`")
        if getattr(c_2d, "inference_unit", "edge") != "edge":
            st.caption(f"Native {c_2d.inference_unit} detections: {recorded_native_detection_count(c_2d, alpha)}; tail counts above describe projected edge membership.")
        st.write(f"- **Permutations:** `{c_2d.n_permutations}`")

    # 2. Strict companion agreement check (family / unit / contrast / tail / alpha / mask / ROI frame)
    st.divider()
    st.markdown("##### 🤝 Companion Agreement")
    compat = check_companion_compatibility(primary_res, comp_res, alpha=alpha)
    atlas = _active_atlas(base_atlas)
    if compat["compatible"]:
        sm = compat["summary"]
        a1, a2, a3, a4 = st.columns(4)
        unit = compat.get("inference_unit", "edge")
        a1.metric(f"Both Significant ({unit}s)", sm["both"])
        a2.metric(f"{primary_name} Only ({unit}s)", sm["primary_only"])
        a3.metric(f"{comp_name} Only ({unit}s)", sm["companion_only"])
        a4.metric("Jaccard Overlap", f"{sm['jaccard']:.2f}")
        st.caption(
            f"Agreement computed on compatible family, inference unit, contrast, tested tail, α = {alpha:g}, "
            "tested mask, and ROI frame. Counts and Jaccard use matching tested inferential units; "
            "two-sided edge overlap counts any detection rather than agreement of effect direction."
        )
        if unit != "edge":
            st.caption(f"The matrix projects {unit} memberships onto edges; these are not independent edge findings.")
        agree_fig = build_companion_agreement_figure(
            compat, atlas=atlas, primary_name=primary_name, comp_name=comp_name
        )
        if agree_fig is not None:
            st.pyplot(agree_fig)
            _download_figure_button(
                agree_fig,
                "Download Companion Agreement PNG",
                f"companion_agreement_{safe_filename_part(contrast_name)}_{safe_filename_part(primary_name)}_{safe_filename_part(comp_name)}.png",
                key="companion_agreement_png",
            )
            plt.close(agree_fig)
    else:
        st.warning(
            "**Companion agreement omitted (incompatible runs):**\n"
            + "\n".join(f"- {r}" for r in compat["reasons"])
        )

    # 3. Compare ground truth side-by-side if synthetic (no recomputation)
    if effect_mask is not None:
        st.divider()
        st.markdown("##### 🎯 Ground Truth Comparison")

        true_mask = (effect_mask > 0)
        true_mask_triu = np.triu(true_mask, 1)
        n_nodes = true_mask.shape[0]
        possible_edges = int(n_nodes * (n_nodes - 1) / 2)
        n_true_pos_edges = int(np.sum(true_mask_triu))
        n_true_neg_edges = possible_edges - n_true_pos_edges

        def _get_metrics(edges_df):
            detected = set()
            if edges_df is not None and not edges_df.empty:
                for _, row in edges_df.iterrows():
                    u, v = min(int(row['roi_i']), int(row['roi_j'])), max(int(row['roi_i']), int(row['roi_j']))
                    detected.add((u, v))
            tp = sum(1 for u, v in detected if true_mask_triu[u, v])
            fp = len(detected) - tp
            fn = n_true_pos_edges - tp
            tpr = tp / n_true_pos_edges if n_true_pos_edges > 0 else 0.0
            fpr = fp / n_true_neg_edges if n_true_neg_edges > 0 else 0.0
            fnr = fn / n_true_pos_edges if n_true_pos_edges > 0 else 0.0
            fdr = fp / len(detected) if len(detected) > 0 else 0.0
            return tp, fp, fn, tpr, fpr, fnr, fdr

        tp_p, fp_p, fn_p, tpr_p, fpr_p, fnr_p, fdr_p = _get_metrics(primary_edges)
        tp_c, fp_c, fn_c, tpr_c, fpr_c, fnr_c, fdr_c = _get_metrics(comp_edges)

        col_m_p, col_m_c = st.columns(2)
        with col_m_p:
            st.markdown(f"**{primary_name} Performance**")
            st.metric("Sensitivity / Power (TPR)", f"{tpr_p*100:.1f}%", help=f"{tp_p} / {n_true_pos_edges}")
            st.metric("Type I Error Rate (FPR)", f"{fpr_p*100:.1f}%", help=f"{fp_p} / {n_true_neg_edges}")
            st.metric("False Discovery Proportion (FDP)", f"{fdr_p*100:.1f}%", help=f"{fp_p} / {len(primary_edges) if primary_edges is not None else 0}; descriptive edge membership")
        with col_m_c:
            st.markdown(f"**{comp_name} Performance**")
            st.metric("Sensitivity / Power (TPR)", f"{tpr_c*100:.1f}%", help=f"{tp_c} / {n_true_pos_edges}")
            st.metric("Type I Error Rate (FPR)", f"{fpr_c*100:.1f}%", help=f"{fp_c} / {n_true_neg_edges}")
            st.metric("False Discovery Proportion (FDP)", f"{fdr_c*100:.1f}%", help=f"{fp_c} / {len(comp_edges) if comp_edges is not None else 0}; descriptive edge membership")

    # 4. Aligned connectivity maps side-by-side
    st.divider()
    _render_heatmaps(p_2d, base_atlas, primary_name, contrast_name=contrast_name, slot="compare_primary", effect_mask=effect_mask)
    _render_heatmaps(c_2d, base_atlas, comp_name, contrast_name=contrast_name, slot="compare_comp", effect_mask=None)


def _render_single_run_view(res_wrapper, edges_df, method_name, base_atlas, effect_mask, contrast_name, direction_labels, slot):
    """Render all sections for a single run consistently."""
    res, param_idx = _select_grid_slice(res_wrapper, slot)
    if (param_idx is not None and param_idx > 0) or edges_df is None:
        edges_df = res.significant_edges(atlas=_active_atlas(base_atlas), alpha=_run_alpha())
    _render_one_sentence_summary(res, edges_df, method_name, direction_labels)
    _render_summary_metrics(res, edges_df, method_name, direction_labels)
    _render_analyze_decisions_and_diagnostics(res_wrapper, st.session_state.get("run_plan"), method_name)
    _render_ground_truth_metrics(edges_df, effect_mask, method_name)
    _render_heatmaps(res, base_atlas, method_name, contrast_name=contrast_name, slot=slot, effect_mask=effect_mask)
    _render_null_max_distribution(res, method_name, contrast_name, slot)
    _render_predicted_and_coupling(res, base_atlas, method_name, contrast_name, slot)
    _render_network_and_node_summaries(res, base_atlas, method_name, contrast_name, slot)
    _render_brain_connectome_graphs(edges_df, base_atlas, method_name, direction_labels, res)
    _render_components_table(res, base_atlas, method_name, contrast_name, direction_labels, slot)
    _render_blocks_table(res, base_atlas, method_name, contrast_name, direction_labels, slot)
    _render_table_and_download(edges_df, contrast_name, method_name, res=res)


def _render_archive_export(base_atlas, effect_mask):
    """One download button: the full self-contained analysis archive (ZIP).

    Pure export of the finished primary run: the trusted bundle, evidence
    arrays, tables, figures, deterministic methods and an honest CLI replay
    configuration. No preprocessing, inference, ComBat, community detection
    or simulation is (re)run; current display widgets are never written back
    into the captured analysis facts.
    """
    from apps.utils.archive_export import archive_filename, build_analysis_archive
    from apps.utils.helpers import PROJECT_ROOT

    run_plan = st.session_state.get("run_plan") or {}
    result = st.session_state.inference_result
    dataset_info = st.session_state.get("dataset_info") or {}
    inner = unwrap_result(result)
    n_nodes = np.asarray(inner["omnibus"] if "omnibus" in inner else inner["positive"]).shape[0]
    atlas = atlas_for_annotation(active_analysis_atlas(base_atlas), n_nodes, warn=False)
    grid_cell = st.session_state.get("results:grid_idx:primary")
    # Build once per finished run / display cell; the page can rerun freely.
    memo = (
        "_archive_zip", id(result), str(run_plan.get("alpha")), grid_cell,
        id(effect_mask), str(dataset_info.get("loaded_at")),
    )
    if st.session_state.get(memo) is None:
        zip_bytes, _manifest = build_analysis_archive(
            result,
            run_plan,
            dataset_info,
            atlas=atlas,
            effect_mask=effect_mask,
            grid_cell=grid_cell,
            project_root=PROJECT_ROOT,
        )
        st.session_state[memo] = zip_bytes
    else:
        zip_bytes = st.session_state[memo]
    st.download_button(
        "📦 Download Full Analysis Archive (ZIP)",
        data=zip_bytes,
        file_name=archive_filename(
            current_contrast_name(),
            str(run_plan.get("method") or "analysis"),
            run_plan.get("started_at"),
        ),
        mime="application/zip",
        help=(
            "Self-contained archive of the finished run: result bundle, evidence "
            "arrays, tables, figures, deterministic methods.md and a CLI replay "
            "analysis.yaml with its replay status. Contains no subject data; "
            "bundle.pkl is a trusted-only pickle."
        ),
        key="archive_export_download",
        type="primary",
    )


def render_inference_results_view(base_atlas):
    """Render the results step (summaries, heatmaps, tables, exports)."""
    col_t, col_h = st.columns([0.8, 0.2])
    with col_t:
        st.markdown("### Inference Results")
    with col_h:
        render_help("inference_results")

    is_stale = result_is_stale()

    if is_stale:
        st.warning("⚠️ **Stale Results Detected:** The dataset configuration, parcellation, or preprocessing settings have changed in Tab 1 since these inference results were generated. Please re-run the inference in Tab 2 to obtain up-to-date results.")

    if not require_analysis_step(key="results"):
        return

    _render_archive_export(base_atlas, st.session_state.get("_synthetic_effect_mask"))

    companion_res_wrapper = st.session_state.get("companion_inference_result")
    primary_method = st.session_state.run_plan.get("method", "Primary").upper()
    effect_mask = st.session_state.get("_synthetic_effect_mask")
    contrast_name = current_contrast_name()
    direction_labels = effect_direction_labels(st.session_state.get("run_plan"))

    if companion_res_wrapper is not None:
        companion_method = st.session_state.get("companion_method", "Companion").upper()

        view_mode = st.radio(
            "Select Result View Mode:",
            [f"Primary Run ({primary_method})", f"Sensitivity Companion ({companion_method})", "Side-by-Side Comparison"],
            horizontal=True,
            key=persistent_key('results:select_result_view_mode', scope='results', options=[f'Primary Run ({primary_method})', f'Sensitivity Companion ({companion_method})', 'Side-by-Side Comparison']),
        )
        st.write("")

        if view_mode == f"Primary Run ({primary_method})":
            _render_single_run_view(
                st.session_state.inference_result,
                st.session_state.edges_df,
                primary_method,
                base_atlas,
                effect_mask,
                contrast_name,
                direction_labels,
                "primary",
            )
        elif view_mode == f"Sensitivity Companion ({companion_method})":
            _render_single_run_view(
                companion_res_wrapper,
                st.session_state.companion_edges_df,
                companion_method,
                base_atlas,
                effect_mask,
                contrast_name,
                direction_labels,
                "companion",
            )
        else:
            primary_res = st.session_state.inference_result
            primary_edges = st.session_state.edges_df
            comp_res = companion_res_wrapper
            comp_edges = st.session_state.companion_edges_df

            _render_comparative_view(
                primary_res,
                primary_edges,
                comp_res,
                comp_edges,
                primary_method,
                companion_method,
                base_atlas,
                effect_mask,
                direction_labels,
                contrast_name=contrast_name,
            )
            _render_analyze_decisions_and_diagnostics(primary_res, st.session_state.get("run_plan"), primary_method)

            p_2d, _ = get_2d_result(primary_res, 0)
            c_2d, _ = get_2d_result(comp_res, 0)
            tab_p, tab_c = st.tabs([f"Primary Details ({primary_method})", f"Companion Details ({companion_method})"])
            with tab_p:
                _render_network_and_node_summaries(p_2d, base_atlas, primary_method, contrast_name, "compare_primary")
                _render_components_table(p_2d, base_atlas, primary_method, contrast_name, direction_labels, 'primary')
                _render_blocks_table(p_2d, base_atlas, primary_method, contrast_name, direction_labels, 'primary')
                _render_table_and_download(primary_edges, contrast_name, primary_method, res=p_2d)
            with tab_c:
                _render_network_and_node_summaries(c_2d, base_atlas, companion_method, contrast_name, "compare_comp")
                _render_components_table(c_2d, base_atlas, companion_method, contrast_name, direction_labels, 'companion')
                _render_blocks_table(c_2d, base_atlas, companion_method, contrast_name, direction_labels, 'companion')
                _render_table_and_download(comp_edges, contrast_name, companion_method, res=c_2d)
    else:
        _render_single_run_view(
            st.session_state.inference_result,
            st.session_state.edges_df,
            primary_method,
            base_atlas,
            effect_mask,
            contrast_name,
            direction_labels,
            "primary",
        )
