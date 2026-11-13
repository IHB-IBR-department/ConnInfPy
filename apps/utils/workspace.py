"""Presentation and navigation shared by the analysis steps."""

from datetime import datetime, timezone
import base64
from html import escape
from pathlib import Path

import streamlit as st


DATA_TAB = "1. Data Ingestion & Preprocessing"
DESIGN_TAB = "2. Design & Inference"
RESULTS_TAB = "3. Inference Results"
ANNOTATION_TAB = "4. Brainnetome Annotation"
DOCS_TAB = "6. Workspace Documentation"


@st.cache_data(show_spinner=False)
def _workflow_banner_data():
    path = Path(__file__).resolve().parents[1] / "assets" / "conninf_workflow_banner_v2.png"
    return base64.b64encode(path.read_bytes()).decode("ascii")


def render_workspace_banner():
    alt = (
        "ConnInfPy: time series, connectivity matrices, model and contrast, parallel "
        "TFNBS(+), NBS(+), t-stat (max perm), BH-FDR, and diffuse-block cNBS/FBC methods, then a connectogram."
    )
    st.markdown(
        '<header class="workspace-banner">'
        '<div class="workspace-banner-art">'
        f'<img src="data:image/png;base64,{_workflow_banner_data()}" alt="{alt}"></div>'
        '<span class="workspace-wordmark">ConnInfPy</span></header>',
        unsafe_allow_html=True,
    )


def go_to_step(label, tab, *, key, primary=True):
    if st.button(label, key=key, type="primary" if primary else "secondary"):
        st.session_state.next_tab = tab
        st.rerun()


def require_analysis_step(*, annotation=False, key):
    """Guide an empty view to its earliest missing prerequisite."""
    if st.session_state.get("connectivity_data") is None:
        st.info("Load a dataset to begin your analysis.")
        go_to_step("Load data", DATA_TAB, key=f"{key}_data")
        return False
    if st.session_state.get("inference_result") is None:
        st.info("Your dataset is ready. Configure and run an analysis to see results.")
        go_to_step("Configure analysis", DESIGN_TAB, key=f"{key}_design")
        return False
    if annotation and st.session_state.get("evidence_packet") is None:
        if st.session_state.get("annotation") is not None:
            detected = st.session_state.annotation["detected"]
            st.info("There are no detected connections at the selected threshold." if detected.empty else "The detected connections have no usable annotation evidence. Review atlas coverage.")
            go_to_step("Review annotation", ANNOTATION_TAB, key=f"{key}_annotation")
        else:
            st.info("Annotate the detected connections before preparing a report.")
            go_to_step("Annotate connections", ANNOTATION_TAB, key=f"{key}_annotation")
        return False
    return True


def remember_dataset(*, name, source, atlas, mode, **details):
    """Capture provenance only after a successful explicit load/generation."""
    data = st.session_state.connectivity_data
    pheno = st.session_state.get("pheno_df")
    subjects = None
    if pheno is not None:
        id_col = next((c for c in ("subject_id", "subject", "participant_id") if c in pheno), None)
        if id_col:
            subjects = int(pheno[id_col].nunique())
    st.session_state.dataset_info = {
        "name": str(name), "source": str(source), "mode": mode,
        "observations": int(data.shape[0]), "subjects": subjects,
        "n_rois": int(data.shape[1]), "atlas": str(atlas),
        "data_kind": st.session_state.get("connectivity_data_kind"),
        "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **details,
    }


def atlas_display_name(atlas, fallback):
    """A short name from the loaded atlas's provenance, without long citations."""
    source = str(getattr(atlas, "source", None) or fallback)
    if "schaefer" in source.lower():
        return f"Schaefer-{len(atlas)}"
    if "brainnetome" in source.lower() or "bna" in source.lower():
        return f"Brainnetome-{len(atlas)}"
    if "/" in source:
        source = Path(source).stem
    return source.split(";")[0]


def dataset_summary():
    """Describe loaded arrays, including sessions started before this UI update."""
    data = st.session_state.get("connectivity_data")
    if data is None:
        return None
    info = dict(st.session_state.get("dataset_info") or {})
    info.setdefault("name", "Loaded session dataset")
    info.setdefault("source", "Source details unavailable for this earlier session")
    info.setdefault("observations", int(data.shape[0]))
    info.setdefault("n_rois", int(data.shape[1]))
    info.setdefault("atlas", "Dataset metadata")
    atlas = st.session_state.get("dataset_atlas")
    if atlas is None:
        atlas = st.session_state.get("loaded_reference_atlas")
    if atlas is not None:
        info["atlas"] = atlas_display_name(atlas, info["atlas"])
    return info


def render_dataset_card():
    info = dataset_summary()
    if info is None:
        return
    st.markdown(f"**Active dataset: {info['name']}**")
    cols = st.columns(3)
    cols[0].metric("Observations", info["observations"])
    cols[1].metric("ROIs", info["n_rois"])
    cols[2].metric("Subjects", info.get("subjects") or "—")
    st.caption(f"{info['source']} · {info['atlas']} · Validated")
    record = st.session_state.get("dataset_preprocessing")
    from apps.utils.helpers import describe_per_site_facts, describe_preprocessing_facts
    with st.expander("Preprocessing facts (known / unknown)", expanded=False):
        for label, value, source in describe_preprocessing_facts(record):
            st.markdown(f"- **{label}:** {value} — *{source}*")
        per_site = describe_per_site_facts(record)
        if per_site:
            st.markdown("Per-site facts (combined dataset; no common value is claimed):")
            for site, value, source in per_site:
                st.markdown(f"- **Site `{site}` GSR:** {value} — *{source}*")
        st.caption(
            "Only facts with an explicit recorded source are stated; everything else is "
            "unknown and is never inferred from the connectivity values."
        )


def render_data_table(frame, **kwargs):
    """Readable labels and numeric formats; original values stay in exports."""
    names = {
        "i_roi": "Source ROI", "j_roi": "Target ROI", "i_bna": "Source Brainnetome region",
        "j_bna": "Target Brainnetome region", "t_signed": "Signed t statistic",
        "p_positive": "Corrected p · positive", "p_negative": "Corrected p · negative",
        "distance_mm": "Distance (mm)", "domain": "BrainMap domain", "term": "BrainMap term",
        "n_detected_endpoint_rois": "Detected endpoint ROIs",
        "share_detected_endpoint_rois": "Share of detected endpoint ROIs",
        "n_all_rois": "All ROIs", "share_all_rois": "Share of all ROIs",
        "n_rois": "ROIs", "max_likelihood_ratio": "Maximum likelihood ratio",
    }
    config = {}
    for column in frame.columns:
        label = names.get(column, str(column).replace("_", " ").capitalize())
        if str(column).startswith("share_"):
            config[column] = st.column_config.NumberColumn(label, format="percent")
        elif str(column).startswith("p_"):
            config[column] = st.column_config.NumberColumn(label, format="%.4g")
        else:
            config[column] = label
    st.dataframe(frame, column_config=config, **kwargs)


def render_analysis_context():
    info = dataset_summary()
    if info is None:
        st.caption("Start by loading a demo, importing a manifest, or creating a synthetic dataset.")
        return
    roi_idx = st.session_state.get("roi_indices")
    n_rois = len(roi_idx) if roi_idx is not None else info["n_rois"]
    pieces = [info["name"], f"{info['observations']} observations", f"{n_rois} ROIs", info["atlas"]]
    plan = st.session_state.get("run_plan") or {}
    if plan:
        if plan.get("question_type") == "Group Difference":
            pieces.append(f"{plan.get('target_group')} − {plan.get('reference_group')}")
        elif plan.get("question_type") == "Paired Condition":
            pieces.append(f"{plan.get('target_condition')} − {plan.get('baseline_condition')}")
        elif plan.get("interest_var"):
            pieces.append(str(plan["interest_var"]))
        if plan.get("method"):
            pieces.append(str(plan["method"]).upper())
        if plan.get("alpha") is not None:
            pieces.append(f"α = {plan['alpha']:g}")
    st.markdown('<div class="analysis-context">' + "<span>" + "</span><span>".join(escape(p) for p in pieces) + "</span></div>", unsafe_allow_html=True)


def render_progress(tabs):
    state = st.session_state
    task = state.get("inference_task")
    running = task is not None and getattr(task, "status", "") == "running"
    loaded = state.get("connectivity_data") is not None
    complete = state.get("inference_result") is not None
    edges = state.get("edges_df")
    annotation = state.get("annotation")
    statuses = [
        "Loaded" if loaded else "No data",
        "Running" if running else "Completed" if complete else "Ready" if loaded else "Needs data",
        (f"{len(edges)} detected connections" if edges is not None and not edges.empty else "No detections") if complete else "Needs analysis",
        ("Completed · no detections" if annotation is not None and annotation["detected"].empty else "Completed") if annotation is not None else "Ready" if complete else "Needs results",
    ]
    finished = [loaded, complete, complete, annotation is not None]
    styles = []
    for index, (tab, done) in enumerate(zip(tabs, finished), start=1):
        selector = f".st-key-workspace_navigation .st-key-nav_step_{index} button"
        if done:
            styles.append(f"{selector} {{ background: #edf8f0; border-color: #8dc5a0; color: #166534; }}")
        elif running and index == 2:
            styles.append(f"{selector} {{ background: #fff6dd; border-color: #d5b65a; color: #795800; }}")
        if state.active_tab == tab:
            styles.append(f"{selector} {{ border-color: #376fa6; box-shadow: inset 0 0 0 1px #376fa6; font-weight: 650; }}")
    # Style-only HTML uses Streamlit's event container, adding no visible row.
    st.html("<style>" + "\n".join(styles) + "</style>")
    for index, (col, title, tab, status) in enumerate(zip(st.columns(4), ["Data", "Design", "Results", "Annotation"], tabs, statuses), start=1):
        with col:
            label = f"{'✓ ' if finished[index - 1] else ''}{index}. {title}"
            if st.button(label, key=f"nav_step_{index}", width="stretch", help=status):
                state.next_tab = tab
                st.rerun()
