"""Guided connectivity analysis workspace."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conninfpy.atlas import AtlasInfo
from apps.components.data_ingestion import render_data_ingestion_view
from apps.components.design_inference import render_design_inference_view
from apps.components.inference_results import render_inference_results_view
from apps.components.bna_annotation import render_bna_annotation_view
from apps.components.workspace_docs import render_workspace_docs_view
from apps.utils.helpers import atlas_has_coords, atlas_has_networks, load_custom_atlas_csv
from apps.utils.ui_state import persistent_file_uploader, persistent_key, retain_widget_state
from apps.utils.workspace import (
    DATA_TAB, DESIGN_TAB, RESULTS_TAB, ANNOTATION_TAB, DOCS_TAB,
    render_analysis_context, render_progress, render_workspace_banner,
)

st.set_page_config(page_title="ConnInfPy · Connectivity analysis", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
retain_widget_state()

tabs_list = [DATA_TAB, DESIGN_TAB, RESULTS_TAB, ANNOTATION_TAB, DOCS_TAB]
for key in (
    "connectivity_data", "connectivity_data_kind", "pheno_df", "inference_result", "edges_df",
    "companion_inference_result", "companion_edges_df", "companion_method", "annotation",
    "evidence_packet", "sub_atlas", "dataset_atlas", "roi_indices",
):
    st.session_state.setdefault(key, None)
if st.session_state.get("next_tab") is not None:
    st.session_state.active_tab = st.session_state.pop("next_tab")
if st.session_state.get("active_tab") not in tabs_list:
    st.session_state.active_tab = DATA_TAB

st.markdown("""
<style>
    [data-testid="stMainBlockContainer"] { padding-top: 3.5rem; padding-bottom: 2rem; }
    .workspace-banner { width: 100%; margin-bottom: .2rem; }
    /* Show the artwork bounds (140, 235)-(2010, 500) of the 2172x724 source. */
    .workspace-banner-art { position: relative; width: 100%; aspect-ratio: 1870 / 265;
        overflow: hidden; }
    .workspace-banner-art img { position: absolute; display: block;
        width: 116.149733%; max-width: none !important; height: auto;
        left: -7.486631%; top: -88.679245%; }
    .workspace-wordmark { display: none; font-size: 1.55rem; font-weight: 700; }
    .analysis-context { display: flex; flex-wrap: wrap; gap: .35rem .9rem;
        font-size: .8rem; opacity: .72; margin-bottom: .1rem; }
    .analysis-context span + span { border-left: 1px solid rgba(128,128,128,.3); padding-left: .9rem; }
    .st-key-workspace_navigation { padding-bottom: .35rem; border-bottom: 1px solid rgba(128,128,128,.2); }
    .custom-card { padding: 1rem; border-radius: .5rem; margin-bottom: 1rem; }
    :is(.st-key-results_view, .st-key-annotation_view) [data-testid="stDownloadButton"] button:not(:disabled) {
        background: #167345; color: white; border: 1px solid #167345;
    }
    :is(.st-key-results_view, .st-key-annotation_view) [data-testid="stDownloadButton"] button:not(:disabled):hover {
        background: #105a35; color: white; border-color: #105a35;
    }
    :is(.st-key-results_view, .st-key-annotation_view) [data-testid="stDownloadButton"] button:focus-visible {
        outline: 3px solid #85bfa0; outline-offset: 2px;
    }
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding-top: 4rem; }
        .analysis-context { gap: .35rem .6rem; }
        .analysis-context span + span { padding-left: .6rem; }
        .workspace-banner-art { display: none; }
        .workspace-wordmark { display: block; }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_atlas(name):
    if name == "Schaefer-100 Yeo-7":
        return AtlasInfo.schaefer_100_yeo7()
    if name == "Schaefer-200 Yeo-7":
        return AtlasInfo.schaefer_200_yeo7()
    if name == "Schaefer-400 Yeo-7":
        return AtlasInfo.schaefer_400_yeo7()
    return AtlasInfo.bna_246()


st.sidebar.markdown("### Workspace")
if st.sidebar.button("Help & documentation", key="nav_docs_btn", width="stretch"):
    st.session_state.next_tab = DOCS_TAB
    st.rerun()
st.sidebar.markdown("### Atlas metadata")
st.sidebar.caption("Used for the next dataset load. The active analysis keeps its loaded atlas.")
atlas_mode = st.sidebar.radio("Atlas Metadata", ["No atlas metadata", "Bundled atlas", "Custom atlas CSV"], index=1, key="atlas_mode")
base_atlas = None
atlas_choice = atlas_mode
palette = None
if atlas_mode == "Bundled atlas":
    atlas_choice = st.sidebar.selectbox("Reference Parcellation", ["Schaefer-100 Yeo-7", "Schaefer-200 Yeo-7", "Schaefer-400 Yeo-7", "BNA-246"], key=persistent_key("atlas_choice", scope="atlas"))
    base_atlas = get_atlas(atlas_choice)
    palette = {
        "Schaefer-100 Yeo-7": "apps/assets/yeo7_schaefer100_sidebar_palette.png",
        "Schaefer-200 Yeo-7": "apps/assets/yeo7_schaefer200_sidebar_palette.png",
        "Schaefer-400 Yeo-7": "apps/assets/yeo7_schaefer400_sidebar_palette.png",
        "BNA-246": "apps/assets/bna246_sidebar_palette.png",
    }[atlas_choice]
elif atlas_mode == "Custom atlas CSV":
    custom_source = st.sidebar.radio("Custom atlas source", ["Path", "Upload"], horizontal=True, key=persistent_key("custom_atlas_source", scope="atlas"))
    try:
        if custom_source == "Path":
            path = st.sidebar.text_input("Atlas CSV path", key=persistent_key("custom_atlas_path", scope="atlas"))
            if path:
                base_atlas = load_custom_atlas_csv(path, source=f"Custom atlas: {path}")
        else:
            with st.sidebar:
                uploaded = persistent_file_uploader("Upload atlas CSV", type=["csv"], key="custom_atlas_upload")
            if uploaded is not None:
                uploaded.seek(0)
                base_atlas = load_custom_atlas_csv(uploaded, source=f"Custom atlas: {uploaded.name}")
        if base_atlas is not None:
            atlas_choice = f"Custom atlas ({len(base_atlas)} ROIs)"
    except Exception as exc:
        st.sidebar.error(f"Could not load custom atlas: {exc}")
        base_atlas = None

if base_atlas is None:
    st.sidebar.caption("Atlas metadata is optional for statistics. Network-aware methods need network labels; annotation needs ROI coordinates or a parcellation image.")
else:
    capabilities = [f"{len(base_atlas)} ROIs", "network labels" if atlas_has_networks(base_atlas) else "no network labels", "coordinates" if atlas_has_coords(base_atlas) else "no coordinates"]
    st.sidebar.caption(" · ".join(capabilities))
if palette:
    with st.sidebar.expander("Preview atlas and network colours"):
        st.image(palette, width="stretch")

st.session_state.active_atlas_signature = "|".join([
    atlas_mode, atlas_choice, str(len(base_atlas)) if base_atlas is not None else "0",
    str(getattr(base_atlas, "source", "")) if base_atlas is not None else "",
])

# Adopt provenance for older live sessions without guessing their data source.
if st.session_state.connectivity_data is not None:
    st.session_state.setdefault("loaded_reference_atlas", base_atlas)
    st.session_state.setdefault("loaded_atlas_signature", st.session_state.active_atlas_signature)

render_workspace_banner()
render_analysis_context()
with st.container(key="workspace_navigation"):
    render_progress(tabs_list[:4])

active_tab = st.session_state.active_tab
if active_tab == DATA_TAB:
    render_data_ingestion_view(base_atlas, atlas_choice, tabs_list)
elif active_tab == DESIGN_TAB:
    render_design_inference_view(base_atlas, tabs_list)
elif active_tab == RESULTS_TAB:
    with st.container(key="results_view"):
        render_inference_results_view(base_atlas)
elif active_tab == ANNOTATION_TAB:
    with st.container(key="annotation_view"):
        render_bna_annotation_view(base_atlas)
else:
    render_workspace_docs_view()
