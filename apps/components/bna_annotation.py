"""Brainnetome annotation step: where the detected connections are."""
import pandas as pd
import streamlit as st

from apps.utils.helpers import (
    active_analysis_atlas,
    atlas_has_coords,
    current_contrast_name,
    render_help,
    result_is_stale,
    safe_filename_part,
)
from conninfpy.interpret.annotate import annotate_bna_regions, annotate_rois, bna_region_ids, summarize_annotation
from conninfpy.brainnetome import BrainnetomeReference
from conninfpy.interpret.evidence import build_annotation_evidence, describe_coverage, summarize_terms


# Unknown variant first: an undeclared space is flagged approximate, never assumed exact.
_SPACES = {
    "MNI152": "MNI152, variant unknown",
    "MNI152NLin6Asym": "FSL MNI152 (MNI152NLin6Asym)",
    "MNI152NLin2009cAsym": "fMRIPrep MNI152NLin2009cAsym",
}


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
    result = st.session_state.get("inference_result")
    if result is None:
        st.warning("Run inference first (Tab 2).")
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
    with st.form("bna_annotation_form"):
        support = st.radio(
            "What produced each ROI's signal?", supports, horizontal=True,
            help="parcel: the label image the signals were extracted with; sphere: a sphere around each "
                 "coordinate; point: the coordinate alone (flagged as a point estimate).",
        )
        c1, c2, c3 = st.columns(3)
        radius = c1.number_input("Sphere radius (mm)", 2.0, 20.0, 6.0, 0.5)
        purity = c2.slider("Purity for a label", 0.3, 0.9, 0.5, 0.05,
                           help="A level (subregion, gyrus, lobe) is assigned when it holds this share of "
                                "the ROI's labelled tissue; otherwise the ROI is 'mixed'.")
        alpha = c3.number_input("Alpha", 0.001, 0.2, 0.05, 0.005, format="%.3f")
        space = st.selectbox(
            "Template the signals were extracted in", list(_SPACES), format_func=_SPACES.get,
            help="FSL MNI152 is the atlas's own grid (exact); other MNI variants differ by about 1-3 mm and "
                 "every ROI is flagged approximate. Ignored for Brainnetome regions.",
        )
        c4, c5 = st.columns(2)
        level = c4.selectbox("Summary level", ["lobe", "gyrus", "subregion", "network"])
        param_idx = None
        if getattr(result, "is_grid", False):
            param_idx = int(c5.number_input("(E, H) cell", 0, len(result.e_grid) - 1, 0))
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
            summary = summarize_annotation(result, rois, alpha=alpha, level=level, param_idx=param_idx)
            edges = summary["edges"]
            detected = edges[(edges["p_positive"] <= alpha) | (edges["p_negative"] <= alpha)]
            evidence = build_annotation_evidence(
                detected, rois, contrast_name=current_contrast_name(),
                source_atlas=getattr(atlas, "source", None) or "user atlas",
                inference_unit=getattr(result, "inference_unit", "edge"), summary=summary,
            )
        st.session_state.annotation = {"rois": rois, "summary": summary, "detected": detected,
                                       "support": support, "level": level}
        st.session_state.evidence_packet = evidence if evidence["rois"] else None
        st.session_state.narrative_text = None
        st.session_state.narrative_results = None

    ann = st.session_state.get("annotation")
    if not ann:
        return
    rois, summary, detected = ann["rois"], ann["summary"], ann["detected"]
    meta = summary["meta"]
    st.caption(
        f"Inference unit: **{meta['inference_unit']}**. Descriptive annotation: the p-values and their "
        "correction are unchanged, and no enrichment test is made."
    )
    if st.session_state.evidence_packet is not None:
        st.info(describe_coverage(st.session_state.evidence_packet)["report_sentence"])
    else:
        st.warning("No detected connections at this alpha.")

    tab_c, tab_r, tab_rel, tab_dom, tab_qc = st.tabs(
        ["Detected connections", "ROIs", "Relations", "BrainMap domains", "Distance QC"])
    with tab_c:
        cols = ["i_roi", "j_roi", "i_bna", "j_bna", "t_signed", "p_positive", "p_negative",
                "hemispheric_relation", "anatomical_relation", "circuit", "distance_mm",
                "shared_support", "sc", "network_relation", "shared_domains"]
        st.dataframe(detected[[c for c in cols if c in detected.columns]], width="stretch", hide_index=True)
        st.download_button("Download annotated connections (CSV)", detected.to_csv(index=False),
                           file_name=f"bna_connections_{safe_filename_part(current_contrast_name())}.csv")
    with tab_r:
        st.dataframe(rois.table, width="stretch", hide_index=True)
        st.download_button("Download ROI annotation (CSV)", rois.table.to_csv(index=False),
                           file_name=f"bna_rois_{safe_filename_part(current_contrast_name())}.csv")
    with tab_rel:
        st.markdown("Detections next to the tested opportunities (shares of each).")
        st.dataframe(summary["relations"], width="stretch", hide_index=True)
        st.markdown(f"**Groups at the {ann['level']} level** (ROIs not pure at that level are 'mixed').")
        st.dataframe(summary["regions"], width="stretch", hide_index=True)
    with tab_dom:
        st.markdown("Unique ROIs per BrainMap domain (forward-inference literature context; "
                    "memberships overlap).")
        st.dataframe(summary["domains"], width="stretch", hide_index=True)
        if st.session_state.evidence_packet is not None:
            terms = pd.DataFrame(summarize_terms(st.session_state.evidence_packet)["terms"])
            if not terms.empty:
                st.dataframe(terms, width="stretch", hide_index=True)
    with tab_qc:
        st.markdown("A contrast that covaries with head motion often shows short-range increases and "
                    "long-range decreases; overlapping spheres share voxels.")
        st.dataframe(summary["distance_qc"], width="stretch", hide_index=True)
