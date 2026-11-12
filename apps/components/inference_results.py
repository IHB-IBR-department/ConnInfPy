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
    atlas_has_coords,
    atlas_has_networks,
    current_contrast_name,
    effect_direction_labels,
    result_is_stale,
    safe_filename_part,
    render_help,
)

def _render_one_sentence_summary(res, edges_df, method_name, direction_labels):
    """Render the single-sentence plain-language result summary."""
    pos_count = 0
    neg_count = 0
    if edges_df is not None and not edges_df.empty:
        pos_count = len(edges_df[edges_df["t_signed"] > 0])
        neg_count = len(edges_df[edges_df["t_signed"] < 0])
    
    n_perms = getattr(res, "n_permutations", 100)
    tested = getattr(res, "tested_tail", None)
    if tested is not None:
        count = pos_count if tested == "positive" else neg_count
        st.info(
            f"📝 **Summary:** The one-sided {method_name} test of **{direction_labels[tested]}** "
            f"detected **{count}** significant edges using {n_perms} permutations. "
            "The opposite direction was not tested."
        )
        return
    st.info(
        f"📝 **Summary:** The {method_name} run detected **{pos_count}** significant edges for **{direction_labels['positive']}** "
        f"and **{neg_count}** edges for **{direction_labels['negative']}** using {n_perms} permutations."
    )

def _render_summary_metrics(res, edges_df, method_name, direction_labels):
    """Render the headline metric row for one result."""
    st.markdown(f"##### 📊 {method_name} Summary")
    nsig = res.n_significant(0.05)
    tested = getattr(res, "tested_tail", None)
    shown = {
        tail: ("not tested" if tested is not None and tail != tested else nsig[tail])
        for tail in ("positive", "negative")
    }
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(direction_labels["positive"], shown["positive"], help="Edges significant in the positive statistical tail.")
    with col2:
        st.metric(direction_labels["negative"], shown["negative"], help="Edges significant in the negative statistical tail.")
    with col3:
        st.metric("Permutations Completed", res.n_permutations)
    if tested is not None:
        st.caption(
            f"Prespecified one-sided test ({res.tail_policy}): the {tested} direction has its own "
            "max-statistic null at the nominal alpha; the opposite direction was not tested."
        )

def _render_ground_truth_metrics(edges_df, effect_mask, method_name):
    """Render precision/recall against the planted effect mask, if any."""
    if effect_mask is not None:
        st.markdown(f"##### 🎯 Ground Truth Comparison ({method_name})")
        true_mask = (effect_mask > 0)
        true_mask_triu = np.triu(true_mask, 1)
        
        n_nodes = true_mask.shape[0]
        possible_edges = int(n_nodes * (n_nodes - 1) / 2)
        
        n_true_pos_edges = int(np.sum(true_mask_triu))
        n_true_neg_edges = possible_edges - n_true_pos_edges
        
        detected_edges_set = set()
        if edges_df is not None and not edges_df.empty:
            for _, row in edges_df.iterrows():
                i = int(row['roi_i'])
                j = int(row['roi_j'])
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
            st.markdown(f"**False Discovery Rate (FDR):** `{fdr*100:.1f}%` (proportion of false positives among all detected edges)")

def _render_heatmaps(res, base_atlas, method_name):
    """Render the positive/negative p-value heatmaps."""
    st.markdown(f"##### 🗺️ Connectivity Maps ({method_name})")
    
    stat_map = res.stat_signed
    p_pos = res["positive"]
    p_neg = res["negative"]
    
    if stat_map is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Heatmap 1: Observed statistics
        im1 = ax1.imshow(stat_map, cmap="RdBu_r", aspect="auto")
        fig.colorbar(im1, ax=ax1)
        ax1.set_title("Observed Effect Size (Stat Map)")
        
        # Heatmap 2: -log10 thresholded p-values
        min_p = np.minimum(p_pos, p_neg)
        log_p = -np.log10(min_p + 1e-10)
        log_p[min_p > 0.05] = 0.0  # Hide non-significant
        
        im2 = ax2.imshow(log_p, cmap="inferno", aspect="auto")
        fig.colorbar(im2, ax=ax2)
        ax2.set_title("Significant Edges (-log10 p-value)")
        
        st.pyplot(fig)
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

def _render_brain_connectome_graphs(edges_df, base_atlas, method_name, direction_labels, res):
    """Render glass-brain connectome graphs of the significant edges."""
    atlas = _active_atlas(base_atlas)
    st.markdown(f"##### 🧠 Brain-Space Effect Graphs ({method_name})")

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
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 1])
    with ctrl1:
        alpha = st.slider("Graph p-value cutoff", 0.001, 0.100, 0.050, 0.001, key=persistent_key(f"{method_name}_graph_alpha", scope='results'))
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
    direction_key = f"{method_name}_graph_direction"
    result_key = f"_{method_name}_graph_result_id"
    if st.session_state.get(result_key) != id(res):
        st.session_state.pop(direction_key, None)
        st.session_state[result_key] = id(res)
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

def _render_table_and_download(edges_df, contrast_name, file_suffix):
    """Render the edge table with a CSV download button."""
    st.markdown("##### 📋 Significant Edges Table")
    st.dataframe(edges_df, width="stretch")
    
    csv_data = edges_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        f"Download {file_suffix} Significant Edges CSV",
        data=csv_data,
        file_name=f"significant_edges_{safe_filename_part(contrast_name)}_{file_suffix}.csv",
        mime="text/csv"
    )

def _render_comparative_view(primary_res, primary_edges, comp_res, comp_edges, primary_name, comp_name, base_atlas, effect_mask, direction_labels):
    """Render the side-by-side two-method comparison view."""
    st.markdown("#### 🔄 Side-by-Side Comparison")
    
    # 1. Compare summary metrics in two columns
    col_p, col_c = st.columns(2)
    with col_p:
        st.markdown(f"**Primary Analysis ({primary_name})**")
        nsig_p = primary_res.n_significant(0.05)
        st.write(f"- **{direction_labels['positive']}:** `{nsig_p['positive']}`")
        st.write(f"- **{direction_labels['negative']}:** `{nsig_p['negative']}`")
        st.write(f"- **Permutations:** `{primary_res.n_permutations}`")
    with col_c:
        st.markdown(f"**Sensitivity Companion ({comp_name})**")
        nsig_c = comp_res.n_significant(0.05)
        st.write(f"- **{direction_labels['positive']}:** `{nsig_c['positive']}`")
        st.write(f"- **{direction_labels['negative']}:** `{nsig_c['negative']}`")
        st.write(f"- **Permutations:** `{comp_res.n_permutations}`")
        
    # 2. Compare ground truth side-by-side if synthetic
    if effect_mask is not None:
        st.divider()
        st.markdown("##### 🎯 Ground Truth Comparison")
        
        true_mask = (effect_mask > 0)
        true_mask_triu = np.triu(true_mask, 1)
        n_nodes = true_mask.shape[0]
        possible_edges = int(n_nodes * (n_nodes - 1) / 2)
        n_true_pos_edges = int(np.sum(true_mask_triu))
        n_true_neg_edges = possible_edges - n_true_pos_edges
        
        # Helper to compute TPR, FPR, FNR, FDR
        def _get_metrics(edges_df):
            """True-positive / detected-edge metrics vs the planted mask."""
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
            st.metric("False Discovery Rate (FDR)", f"{fdr_p*100:.1f}%", help=f"{fp_p} / {len(primary_edges) if primary_edges is not None else 0}")
        with col_m_c:
            st.markdown(f"**{comp_name} Performance**")
            st.metric("Sensitivity / Power (TPR)", f"{tpr_c*100:.1f}%", help=f"{tp_c} / {n_true_pos_edges}")
            st.metric("Type I Error Rate (FPR)", f"{fpr_c*100:.1f}%", help=f"{fp_c} / {n_true_neg_edges}")
            st.metric("False Discovery Rate (FDR)", f"{fdr_c*100:.1f}%", help=f"{fp_c} / {len(comp_edges) if comp_edges is not None else 0}")

    # 3. Compare heatmaps side-by-side
    st.divider()
    st.markdown("##### 🗺️ Edge Significance Maps Comparison (-log10 p-value)")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Primary Map
    p_pos_p = primary_res["positive"]
    p_neg_p = primary_res["negative"]
    min_p_p = np.minimum(p_pos_p, p_neg_p)
    log_p_p = -np.log10(min_p_p + 1e-10)
    log_p_p[min_p_p > 0.05] = 0.0
    im1 = ax1.imshow(log_p_p, cmap="inferno", aspect="auto")
    fig.colorbar(im1, ax=ax1)
    ax1.set_title(f"{primary_name} Significant Edges")
    
    # Companion Map
    p_pos_c = comp_res["positive"]
    p_neg_c = comp_res["negative"]
    min_p_c = np.minimum(p_pos_c, p_neg_c)
    log_p_c = -np.log10(min_p_c + 1e-10)
    log_p_c[min_p_c > 0.05] = 0.0
    im2 = ax2.imshow(log_p_c, cmap="inferno", aspect="auto")
    fig.colorbar(im2, ax=ax2)
    ax2.set_title(f"{comp_name} Significant Edges")
    
    st.pyplot(fig)
    plt.close(fig)

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
    else:
        # Determine if companion result is available
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
                res_wrapper = st.session_state.inference_result
                res = getattr(res_wrapper, "inference", res_wrapper)
                edges_df = st.session_state.edges_df
                
                _render_one_sentence_summary(res, edges_df, primary_method, direction_labels)
                _render_summary_metrics(res, edges_df, primary_method, direction_labels)
                _render_ground_truth_metrics(edges_df, effect_mask, primary_method)
                _render_heatmaps(res, base_atlas, primary_method)
                _render_brain_connectome_graphs(edges_df, base_atlas, primary_method, direction_labels, res)
                _render_table_and_download(edges_df, contrast_name, primary_method)
                
            elif view_mode == f"Sensitivity Companion ({companion_method})":
                res = getattr(companion_res_wrapper, "inference", companion_res_wrapper)
                edges_df = st.session_state.companion_edges_df
                
                _render_one_sentence_summary(res, edges_df, companion_method, direction_labels)
                _render_summary_metrics(res, edges_df, companion_method, direction_labels)
                _render_ground_truth_metrics(edges_df, effect_mask, companion_method)
                _render_heatmaps(res, base_atlas, companion_method)
                _render_brain_connectome_graphs(edges_df, base_atlas, companion_method, direction_labels, res)
                _render_table_and_download(edges_df, contrast_name, companion_method)
                
            else: # Side-by-Side
                primary_res = getattr(st.session_state.inference_result, "inference", st.session_state.inference_result)
                primary_edges = st.session_state.edges_df
                comp_res = getattr(companion_res_wrapper, "inference", companion_res_wrapper)
                comp_edges = st.session_state.companion_edges_df
                
                _render_comparative_view(primary_res, primary_edges, comp_res, comp_edges, primary_method, companion_method, base_atlas, effect_mask, direction_labels)
                
                # Show tables in tabs
                tab_p, tab_c = st.tabs([f"Primary Edges ({primary_method})", f"Companion Edges ({companion_method})"])
                with tab_p:
                    _render_table_and_download(primary_edges, contrast_name, primary_method)
                with tab_c:
                    _render_table_and_download(comp_edges, contrast_name, companion_method)
        else:
            # Traditional single run view
            res_wrapper = st.session_state.inference_result
            res = getattr(res_wrapper, "inference", res_wrapper)
            edges_df = st.session_state.edges_df
            
            _render_one_sentence_summary(res, edges_df, primary_method, direction_labels)
            _render_summary_metrics(res, edges_df, primary_method, direction_labels)
            _render_ground_truth_metrics(edges_df, effect_mask, primary_method)
            _render_heatmaps(res, base_atlas, primary_method)
            _render_brain_connectome_graphs(edges_df, base_atlas, primary_method, direction_labels, res)
            _render_table_and_download(edges_df, contrast_name, primary_method)
