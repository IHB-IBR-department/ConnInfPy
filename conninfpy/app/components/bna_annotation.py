"""Brainnetome annotation step: where the detected connections are."""
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from conninfpy.app.utils.ui_state import persistent_key
from conninfpy.app.utils.workspace import require_analysis_step, render_data_table

from conninfpy.app.utils.helpers import (
    active_analysis_atlas,
    atlas_has_coords,
    current_contrast_name,
    render_help,
    result_is_stale,
    safe_filename_part,
)
from conninfpy.app.utils.result_views import build_unit_annotation_context, get_2d_result, unwrap_result
from conninfpy.interpret.annotate import (aggregate_connections, annotate_bna_regions, annotate_rois,
                                          bna_region_ids, summarize_annotation)
from conninfpy.atlas.brainnetome import BrainnetomeReference
from conninfpy.interpret.evidence import build_annotation_evidence, describe_coverage, summarize_terms
from conninfpy.plot import plot_connectogram


# Unknown variant first: an undeclared space is flagged approximate, never assumed exact.
_SPACES = {
    "MNI152": "MNI152, variant unknown",
    "MNI152NLin6Asym": "FSL MNI152 (MNI152NLin6Asym)",
    "MNI152NLin2009cAsym": "fMRIPrep MNI152NLin2009cAsym",
}

# No numeric accuracy is promised: the offset between template grids is not
# quantified by the atlas, so every other MNI variant is only flagged approximate.
_SPACE_HELP = (
    "FSL MNI152 is the atlas's own grid (exact); other MNI variants differ from that grid by an "
    "unquantified amount and every ROI is flagged approximate. Ignored for Brainnetome regions."
)


@st.cache_resource(show_spinner=False)
def _reference():
    """Brainnetome reference (bundled atlas and BrainMap profiles; local maps if present)."""
    return BrainnetomeReference.load()


def render_bna_annotation_view(base_atlas):
    """Annotate the ROIs and detected connections with the Brainnetome atlas."""
    col_t, col_h = st.columns([0.8, 0.2])
    with col_t:
        st.markdown("### Brainnetome Annotation")
    with col_h:
        render_help("bna_annotation")

    if result_is_stale():
        st.error("The dataset configuration has changed. Re-run inference in Tab 2 before annotating.")
        return
    if not require_analysis_step(key="annotation"):
        return
    from conninfpy.app.utils.workspace import render_restored_workspace_notice
    render_restored_workspace_notice()
    result = unwrap_result(st.session_state.inference_result)
    if hasattr(result, "keys") and "omnibus" in result and "positive" not in result:
        st.info("Brainnetome directional connection annotation and connectogram require directional t/β contrasts (not yet defined for non-directional omnibus F-tests).")
        return

    atlas = active_analysis_atlas(base_atlas)
    n_nodes = len(result["positive"])
    if atlas is None or not atlas_has_coords(atlas) or len(atlas) != n_nodes:
        st.warning(
            "Annotation needs ROI coordinates (or the parcellation image) in matrix order, "
            f"for the {n_nodes} ROIs of the result. Load atlas metadata with coordinates."
        )
        return

    has_image = getattr(atlas, "image", None) is not None and getattr(atlas, "label_ids", None) is not None
    supports = (["parcel"] if has_image else []) + ["sphere", "point"]
    saved = st.session_state.get("annotation") or {}
    saved_support = saved.get("support", supports[0])
    summary_slot = st.container()
    settings_panel = st.expander("Annotation settings", expanded=not bool(st.session_state.get("annotation")))
    with settings_panel, st.form("bna_annotation_form"):
        support = st.radio(
            "ROI signal source", supports, horizontal=True,
            index=supports.index(saved_support) if saved_support in supports else 0,
            format_func={"parcel": "Atlas parcels", "sphere": "Spheres around ROI centres", "point": "ROI centre coordinates"}.get,
            help="parcel: the label image the signals were extracted with; sphere: a sphere around each "
                 "coordinate; point: the coordinate alone (flagged as a point estimate).",
            key=persistent_key('annotation:what_produced_each_roi_s_signal', scope='annotation', options=supports),
        )
        c1, c2, c3 = st.columns(3)
        radius = c1.number_input("Sphere radius (mm)", 2.0, 20.0, saved.get("radius", 6.0), 0.5, help="Used only for spheres around ROI centres.", key=persistent_key('annotation:sphere_radius_mm', scope='annotation'))
        purity = c2.slider("Minimum label purity", 0.3, 0.9, saved.get("purity", 0.5), 0.05,
                           help="A level (subregion, gyrus, lobe) is assigned when it holds this share of "
                                "the ROI's labelled tissue; otherwise the ROI is 'mixed'.", key=persistent_key('annotation:purity_for_a_label', scope='annotation'))
        run_alpha = float((st.session_state.get("run_plan") or {}).get("alpha", 0.05))
        ann_res_key = "_annotation_result_id"
        ann_run_alpha_key = "_annotation_run_alpha"
        if (
            st.session_state.get(ann_res_key) != id(result)
            or st.session_state.get(ann_run_alpha_key) != run_alpha
        ):
            if not saved:
                st.session_state.pop("annotation:alpha", None)
            st.session_state[ann_res_key] = id(result)
            st.session_state[ann_run_alpha_key] = run_alpha
        alpha = c3.number_input("Detection threshold (α)", 0.001, 0.2, float(saved.get("alpha", run_alpha)), 0.005, format="%.3f", help="Select detections using the corrected p-values already computed by inference, at the α of the run by default. This does not change the correction.", key=persistent_key('annotation:alpha', scope='annotation'))
        space = st.selectbox(
            "Template the signals were extracted in", list(_SPACES), format_func=_SPACES.get,
            index=list(_SPACES).index(saved.get("space", "MNI152")),
            help=_SPACE_HELP,
            key=persistent_key('annotation:template_the_signals_were_extracted_in', scope='annotation', options=list(_SPACES)),
        )
        c4, c5 = st.columns(2)
        levels = ["lobe", "gyrus", "subregion", "network"]
        level = c4.selectbox("Summary level", levels, index=levels.index(saved.get("level", "lobe")), format_func=str.capitalize, key=persistent_key('annotation:summary_level', scope='annotation'))
        param_idx = None
        if getattr(result, "is_grid", False):
            param_idx = int(c5.number_input("(E, H) cell", 0, len(result.e_grid) - 1, saved.get("param_idx") or 0, key=persistent_key('annotation:e_h_cell', scope='annotation')))
        submitted = st.form_submit_button("Annotate", type="primary")

    if submitted:
        with st.spinner("Annotating ROIs and connections ..."):
            ref = _reference()
            # Network labels follow the atlas: Brainnetome regions take the Brainnetome
            # division, other atlases their own labels (Schaefer: Yeo-7) when they have them.
            bna_ids = bna_region_ids(atlas.labels, reference=ref)
            own = list(getattr(atlas, "networks", None) or [])
            networks = own if own and own != list(atlas.labels) else None   # labels-as-networks = no division
            common = dict(reference=ref, names=list(atlas.labels), purity=purity, networks=networks, space=space)
            if bna_ids is not None:
                rois = annotate_bna_regions(bna_ids, reference=ref, names=list(atlas.labels))
            elif support == "parcel":
                rois = annotate_rois(support="parcel", parcel_image=atlas.image,
                                     parcel_labels=list(atlas.label_ids), **common)
            else:
                rois = annotate_rois(atlas.coords, support=support,
                                     radius_mm=radius if support == "sphere" else None, **common)
            tested_tail = getattr(result, "tested_tail", None)
            tail_arg = tested_tail if tested_tail in ("positive", "negative") else "both"
            summary = summarize_annotation(result, rois, alpha=alpha, tail=tail_arg, level=level, param_idx=param_idx)
            edges = summary["edges"]
            if tested_tail == "positive":
                detected = edges[edges["p_positive"] <= alpha]
            elif tested_tail == "negative":
                detected = edges[edges["p_negative"] <= alpha]
            else:
                detected = edges[(edges["p_positive"] <= alpha) | (edges["p_negative"] <= alpha)]
            evidence = build_annotation_evidence(
                detected, rois, contrast_name=current_contrast_name(),
                source_atlas=getattr(atlas, "source", None) or "user atlas",
                inference_unit=getattr(result, "inference_unit", "edge"), summary=summary,
            )
        st.session_state.annotation = {"rois": rois, "summary": summary, "detected": detected,
                                       "support": support, "level": level, "alpha": alpha,
                                       "radius": radius, "purity": purity, "space": space,
                                       "param_idx": param_idx}
        st.session_state.evidence_packet = evidence if evidence["rois"] else None
        st.rerun()

    ann = st.session_state.get("annotation")
    if not ann:
        return
    rois, summary, detected = ann["rois"], ann["summary"], ann["detected"]
    meta = summary["meta"]
    result_2d, _ = get_2d_result(result, ann.get("param_idx"))
    unit_ctx = build_unit_annotation_context(result_2d, detected, rois, alpha=ann["alpha"])
    with summary_slot:
        endpoint_rois = set(detected.get("i_roi", [])) | set(detected.get("j_roi", []))
        c1, c2, c3 = st.columns(3)
        c1.metric("Detected connections", len(detected))
        c2.metric("Endpoint ROIs", len(endpoint_rois))
        c3.metric("Detection threshold", f"{ann['alpha']:.3f}")
        st.caption(
            f"Inference unit: **{meta['inference_unit']}**. Descriptive annotation; corrected p-values "
            "are unchanged. No enrichment test is performed."
        )
        st.caption(unit_ctx["roi_selection_rule"])
        packet = st.session_state.evidence_packet
        restriction = ((packet or {}).get("query") or {}).get("roi_restriction") or {}
        if restriction.get("statement"):
            st.caption(restriction["statement"])
        if packet is not None:
            st.info(describe_coverage(st.session_state.evidence_packet)["report_sentence"])
        elif detected.empty:
            st.info("No detected connections at this threshold. The analysis is complete.")
        else:
            st.info("The detected connections have no usable annotation evidence. Review ROI coverage and atlas metadata.")
        exports = st.columns(2)
        exports[0].download_button("Connections · CSV", detected.to_csv(index=False),
                                   file_name=f"bna_connections_{safe_filename_part(current_contrast_name())}.csv", mime="text/csv")
        exports[1].download_button("ROI annotation · CSV", rois.table.to_csv(index=False),
                                   file_name=f"bna_rois_{safe_filename_part(current_contrast_name())}.csv", mime="text/csv")
        if not unit_ctx["unit_table"].empty:
            st.markdown(f"**Aggregated {meta['inference_unit'].capitalize()} Annotation Context**")
            render_data_table(unit_ctx["unit_table"], width="stretch", hide_index=True)
            st.download_button(
                f"{meta['inference_unit'].capitalize()} context · CSV",
                unit_ctx["unit_table"].to_csv(index=False),
                file_name=f"bna_{meta['inference_unit']}_context_{safe_filename_part(current_contrast_name())}.csv",
                mime="text/csv",
                key="bna_unit_context_csv",
            )

    tab_c, tab_r, tab_rel, tab_dom, tab_qc = st.tabs(
        ["Detected connections", "ROIs", "Relations", "BrainMap domains", "Distance QC"])
    with tab_c:
        if meta["inference_unit"] in ("component", "block"):
            st.caption(
                f"Member edges belonging to detected {meta['inference_unit']}s (shown descriptively; "
                "not independent edge-level tests)."
            )
        cols = ["i_roi", "j_roi", "i_bna", "j_bna", "t_signed", "p_positive", "p_negative",
                "hemispheric_relation", "anatomical_relation", "circuit", "distance_mm",
                "shared_support", "sc", "network_relation", "shared_domains"]
        render_data_table(detected[[c for c in cols if c in detected.columns]], width="stretch", hide_index=True)
    with tab_r:
        render_data_table(rois.table, width="stretch", hide_index=True)
    with tab_rel:
        st.markdown("Detections next to the tested opportunities (shares of each).")
        render_data_table(summary["relations"], width="stretch", hide_index=True)
        st.markdown(f"**Groups at the {ann['level']} level** (ROIs not pure at that level are 'mixed').")
        render_data_table(summary["regions"], width="stretch", hide_index=True)
    with tab_dom:
        st.markdown("Unique ROIs per BrainMap domain (forward-inference literature context; "
                    "memberships overlap).")
        render_data_table(summary["domains"], width="stretch", hide_index=True)
        if st.session_state.evidence_packet is not None:
            terms = pd.DataFrame(summarize_terms(st.session_state.evidence_packet)["terms"])
            if not terms.empty:
                render_data_table(terms, width="stretch", hide_index=True)
    with tab_qc:
        st.markdown("A contrast that covaries with head motion often shows short-range increases and "
                    "long-range decreases; overlapping spheres share voxels.")
        render_data_table(summary["distance_qc"], width="stretch", hide_index=True)

    st.markdown("#### Connectogram")
    cc1, cc2, cc3 = st.columns(3)
    tested_tail = getattr(result, "tested_tail", None)
    tail_options = [tested_tail] if tested_tail else ["positive", "negative"]
    if st.session_state.get("bna_connectogram_tail") not in (None, *tail_options):
        st.session_state.pop("bna_connectogram_tail")   # left over from an earlier run
    tail = cc1.selectbox("Tail", tail_options, key=persistent_key("bna_connectogram_tail", scope='annotation', options=tail_options),
                         help="Draw one connectogram per tail; both tails at the same alpha are two figures. "
                              "After a one-sided test only the tested direction is available.")
    conn_level = cc2.selectbox("Level", ["gyrus", "subregion", "lobe"], key=persistent_key("bna_connectogram_level", scope='annotation'))
    assign = cc3.selectbox("Assignment", ["dominant", "weighted"], key=persistent_key("bna_connectogram_assign", scope='annotation'),
                           help="dominant: each ROI sits in the unit holding most of it (whole chords); "
                                "weighted: spread over the units it covers (fractional chords).")
    conn_alpha = ann.get("alpha", 0.05)
    conn_param_idx = ann.get("param_idx")
    agg = aggregate_connections(result, rois, tail=tail, level=conn_level, alpha=conn_alpha,
                                assign=assign, param_idx=conn_param_idx)
    cov = agg.attrs["coverage"]
    fig, ax = plt.subplots(figsize=(9, 9), facecolor="white")
    plot_connectogram(result, rois, tail=tail, level=conn_level, alpha=conn_alpha, assign=assign,
                      param_idx=conn_param_idx, ax=ax)
    st.pyplot(fig, width="stretch")
    st.caption(
        f"{cov['n_placed']} of {cov['n_rois']} ROIs placed ({len(cov['unplaced_no_support'])} outside the "
        f"atlas, {len(cov['unplaced_low_coverage'])} below min_coverage). Detected ({tail} tail): "
        f"{cov['detected_placed']} of {cov['detected_total']} with both endpoints placed; tested: "
        f"{cov['tested_placed']} of {cov['tested_total']}. Descriptive only; correction unchanged."
    )
    png, pdf = BytesIO(), BytesIO()
    fig.savefig(png, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, format="pdf", bbox_inches="tight", facecolor="white")
    stem = f"connectogram_{safe_filename_part(current_contrast_name())}_{conn_level}_{assign}_{tail}"
    dc1, dc2 = st.columns(2)
    dc1.download_button("Download connectogram (PNG)", data=png.getvalue(), file_name=f"{stem}.png",
                        mime="image/png", key="bna_connectogram_png")
    dc2.download_button("Download connectogram (PDF)", data=pdf.getvalue(), file_name=f"{stem}.pdf",
                        mime="application/pdf", key="bna_connectogram_pdf")
    plt.close(fig)
