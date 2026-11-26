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
from apps.utils.atlas_wizard import (
    custom_atlas_csv_template,
    inspect_atlas_readiness,
    load_custom_atlas_nifti,
)
from apps.utils.helpers import PROJECT_ROOT, atlas_has_coords, atlas_has_networks, load_custom_atlas_csv
from apps.utils.ui_state import persistent_file_uploader, persistent_key, retain_widget_state
from apps.utils.workspace import (
    DATA_TAB, DESIGN_TAB, RESULTS_TAB, ANNOTATION_TAB, DOCS_TAB,
    render_analysis_context, render_progress, render_workspace_banner,
)
from apps.utils.workspace_io import (
    WorkspaceArchiveError,
    apply_pending_workspace_restore,
    build_workspace_archive,
    inspect_workspace_archive,
    workspace_filename,
)

st.set_page_config(page_title="ConnInfPy · Connectivity analysis", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
try:
    apply_pending_workspace_restore(st.session_state, project_root=PROJECT_ROOT)
except WorkspaceArchiveError as _ws_restore_exc:
    st.session_state["_workspace_restore_error"] = str(_ws_restore_exc)
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

if st.session_state.get("_workspace_restore_error"):
    st.sidebar.error(st.session_state.pop("_workspace_restore_error"))
if st.session_state.get("_workspace_restore_banner"):
    st.sidebar.success(st.session_state.pop("_workspace_restore_banner"))

with st.sidebar.expander("Save & restore workspace", expanded=False):
    has_saveable = (
        st.session_state.get("dataset_info") is not None
        or st.session_state.get("connectivity_data") is not None
        or st.session_state.get("inference_result") is not None
    )
    if has_saveable:
        include_data = st.checkbox(
            "Include subject data",
            value=False,
            key="workspace_include_data",
            help="Off by default: saves results, atlas, annotation, annotation evidence, and UI state without subject-level matrices.",
        )
        reg = st.session_state.get("_workspace_widget_keys") or {}
        ui_sig = tuple(sorted((str(k), repr(st.session_state.get(k))) for k in reg))
        ws_memo = (
            "_workspace_zip_cache",
            id(st.session_state.get("dataset_info")),
            id(st.session_state.get("connectivity_data")),
            id(st.session_state.get("inference_result")),
            id(st.session_state.get("companion_inference_result")),
            id(st.session_state.get("annotation")),
            id(st.session_state.get("evidence_packet")),
            str(st.session_state.get("active_tab") or ""),
            bool(include_data),
            ui_sig,
        )
        if st.session_state.get("_workspace_zip_memo_key") != ws_memo:
            ws_bytes, _ = build_workspace_archive(
                st.session_state,
                include_data=bool(include_data),
                project_root=PROJECT_ROOT,
            )
            st.session_state["_workspace_zip_memo_key"] = ws_memo
            st.session_state["_workspace_zip_bytes"] = ws_bytes
        else:
            ws_bytes = st.session_state["_workspace_zip_bytes"]
        st.download_button(
            "Save workspace (.zip)",
            data=ws_bytes,
            file_name=workspace_filename(st.session_state),
            mime="application/zip",
            key="workspace_save_download",
            width="stretch",
            help=(
                "Pickle-free workspace archive (results, atlas, annotation, evidence and UI "
                "state, plus subject data when enabled). It never contains bundle.pkl; the "
                "trusted bundle is only in the results-page Full Analysis Archive."
            ),
        )
    else:
        st.caption("Load data or results to save a workspace.")

    uploaded_ws = st.file_uploader(
        "Restore workspace (.zip)",
        type=["zip"],
        key="workspace_restore_upload",
        help="Accepts ConnInfPy workspace archives or unified analysis archives with a safe workspace/ layer.",
    )
    if uploaded_ws is not None:
        raw_ws_bytes = uploaded_ws.getvalue()
        try:
            insp = inspect_workspace_archive(raw_ws_bytes, project_root=PROJECT_ROOT)
            src_status = insp.source_verification["overall"]
            parts = [insp.summary.get("name") or "Workspace"]
            if insp.summary.get("method"):
                parts.append(str(insp.summary["method"]).upper())
            parts.append("with data" if insp.has_data else "results only")
            st.caption(f"{' · '.join(parts)} · source: {src_status}")
            if insp.legacy_unknowns:
                st.info("; ".join(insp.legacy_unknowns))
            if st.button("Restore workspace", key="workspace_apply_restore_btn", type="primary", width="stretch"):
                st.session_state["_pending_workspace_restore"] = raw_ws_bytes
                st.rerun()
        except WorkspaceArchiveError as exc:
            st.error(str(exc))
st.sidebar.markdown("### Atlas metadata")
st.sidebar.caption("Used for the next dataset load. The active analysis keeps its loaded atlas.")
if st.session_state.get("atlas_mode") == "Custom atlas CSV":
    st.session_state["atlas_mode"] = "Custom atlas"
atlas_mode = st.sidebar.radio("Atlas Metadata", ["No atlas metadata", "Bundled atlas", "Custom atlas"], index=1, key="atlas_mode")
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
elif atlas_mode == "Custom atlas":
    custom_format = st.sidebar.radio(
        "Custom atlas format",
        ["ROI Table (CSV)", "3D Label Mask (NIfTI + optional LUT)"],
        key=persistent_key("custom_atlas_format", scope="atlas"),
    )
    custom_source = st.sidebar.radio("Custom atlas source", ["Path", "Upload"], horizontal=True, key=persistent_key("custom_atlas_source", scope="atlas"))
    try:
        if custom_format == "ROI Table (CSV)":
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
            st.sidebar.download_button(
                "Download CSV template",
                data=custom_atlas_csv_template(),
                file_name="custom_atlas_template.csv",
                mime="text/csv",
                key="custom_atlas_template_download",
                width="stretch",
            )
        else:
            if custom_source == "Path":
                nifti_path = st.sidebar.text_input("3D label NIfTI path (.nii / .nii.gz)", key=persistent_key("custom_atlas_nifti_path", scope="atlas"))
                lut_path = st.sidebar.text_input("Optional LUT path (.csv / .tsv / .txt)", key=persistent_key("custom_atlas_lut_path", scope="atlas"))
                if nifti_path:
                    base_atlas = load_custom_atlas_nifti(
                        nifti_path,
                        lut_path_or_buffer=lut_path or None,
                        source=f"Custom NIfTI atlas: {nifti_path}",
                    )
            else:
                with st.sidebar:
                    nifti_upload = persistent_file_uploader("Upload 3D label NIfTI", type=["nii", "gz"], key="custom_atlas_nifti_upload")
                    lut_upload = persistent_file_uploader("Upload optional LUT", type=["csv", "tsv", "txt", "lut"], key="custom_atlas_lut_upload")
                if nifti_upload is not None:
                    base_atlas = load_custom_atlas_nifti(
                        nifti_upload,
                        lut_path_or_buffer=lut_upload,
                        source=f"Custom NIfTI atlas: {nifti_upload.name}",
                    )
        if base_atlas is not None:
            atlas_choice = f"Custom atlas ({len(base_atlas)} ROIs)"
    except Exception as exc:
        st.sidebar.error(f"Could not load custom atlas: {exc}")
        base_atlas = None
    with st.sidebar.expander("Custom atlas format guide"):
        st.markdown(
            "- **CSV table**: required `name`, `network`; optional `x, y, z` (MNI mm), `hemisphere`, `id`/`roi_id`. Row order must match matrix node order `0..N-1`.\n"
            "- **3D NIfTI mask**: 3D integer labels (`0 = background`), ROI order = ascending label IDs (matching nilearn `NiftiLabelsMasker`), optional LUT (`id, name[, network]` or FreeSurfer LUT). 4D probabilistic maps are not label masks — convert to 3D integer labels or supply an MNI coordinate CSV.\n"
            "- **What each provides**: `name` → edge tables; `network` → `network_summary` + `cNBS`/`NI-TFNBS`/`FBC-TFNBS`; `x, y, z` → Brainnetome `sphere` annotation; 3D NIfTI → exact Brainnetome + SUIT `parcel` annotation."
        )

if base_atlas is None:
    st.sidebar.caption("Atlas metadata is optional for statistics. Network-aware methods need network labels; annotation needs ROI coordinates or a parcellation image.")
else:
    capabilities = [f"{len(base_atlas)} ROIs", "network labels" if atlas_has_networks(base_atlas) else "no network labels", "coordinates" if atlas_has_coords(base_atlas) else "no coordinates"]
    st.sidebar.caption(" · ".join(capabilities))
    if atlas_mode == "Custom atlas":
        conn_data = st.session_state.get("connectivity_data")
        n_nodes = int(conn_data.shape[1]) if conn_data is not None and getattr(conn_data, "ndim", 0) >= 2 else None
        readiness = inspect_atlas_readiness(base_atlas, n_nodes=n_nodes)
        with st.sidebar.expander("Atlas readiness checklist", expanded=any(item.status == "error" for item in readiness)):
            for item in readiness:
                icon = {"ok": "✓", "warning": "!", "error": "✕"}[item.status]
                st.markdown(f"**{icon} {item.label}** — {item.detail}")
                if item.guidance and item.status != "ok":
                    st.caption(item.guidance)
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
