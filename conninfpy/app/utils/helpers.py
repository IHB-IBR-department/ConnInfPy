import os
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from conninfpy.atlas import AtlasInfo
from conninfpy.loaders import (
    AbideSchaeferLoader,
    NumpyLoader,
    CSVDirectoryLoader,
    NiftiDirectoryLoader,
    FmriprepDerivativesLoader,
    TimeseriesDirectoryLoader,
    ConnectivityMatrixLoader,
    ConditionTimeseriesArrayLoader,
    ManifestLoader,
)
# Lab-specific loaders live with the app; importing them also registers
# them for manifests (e.g. datasets/open_close.yaml).
from conninfpy.app.utils.lab_loaders import (
    ChinaCloseCloseLoader,
    CustomPreparedZerssenLoader,
    MultiSiteOpenCloseLoader,
    OpenCloseLoader,
    StressTimeseriesLoader,
    ZerssenNiftiLoader,
)

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent.parent

LOADER_CLASSES = {
    "AbideSchaeferLoader": AbideSchaeferLoader,
    "OpenCloseLoader": OpenCloseLoader,
    "MultiSiteOpenCloseLoader": MultiSiteOpenCloseLoader,
    "StressTimeseriesLoader": StressTimeseriesLoader,
    "ZerssenNiftiLoader": ZerssenNiftiLoader,
    "NumpyLoader": NumpyLoader,
    "CSVDirectoryLoader": CSVDirectoryLoader,
    "NiftiDirectoryLoader": NiftiDirectoryLoader,
    "FmriprepDerivativesLoader": FmriprepDerivativesLoader,
    "TimeseriesDirectoryLoader": TimeseriesDirectoryLoader,
    "ConnectivityMatrixLoader": ConnectivityMatrixLoader,
    "ConditionTimeseriesArrayLoader": ConditionTimeseriesArrayLoader,
    "CustomPreparedZerssenLoader": CustomPreparedZerssenLoader,
    "ManifestLoader": ManifestLoader,
    "ChinaCloseCloseLoader": ChinaCloseCloseLoader
}

def local_dataset_templates_enabled():
    """Whether local-only custom/beta built-in templates should be exposed."""
    return os.getenv("CONNINFPY_ENABLE_LOCAL_TEMPLATES", "").lower() in {"1", "true", "yes"}


def load_custom_datasets():
    """Load user-defined dataset manifests from the workspace."""
    require_local_paths()
    if not local_dataset_templates_enabled():
        return {}
    path = os.path.expanduser("~/.conninfpy/custom_datasets.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def safe_filename_part(text):
    """Return a compact filesystem-safe slug for generated downloads."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return slug or "connectivity_analysis"


def fisher_clip_note(report):
    """One-line Fisher clipping summary with explicit count units.

    Separates observed off-diagonal saturation from the structural
    r = 1 diagonal; matrix cells and unique undirected edges are counted
    and labelled as such (see ``fisher_r_to_z(..., return_report=True)``).
    Under the unbounded policy nothing is clipped: the note reports the
    values left at ±infinity instead, never as "clipped".
    """
    if not isinstance(report, dict):
        return None
    off = report.get("off_diagonal") or {}
    diag = report.get("diagonal") or {}
    n_off = int(off.get("n_cells_clipped", 0) or 0)
    n_diag = int(diag.get("n_cells_clipped", 0) or 0)
    unbounded = report.get("policy") == "unbounded"
    if unbounded:
        infinity = report.get("left_at_infinity") or {}
        inf_diag = (infinity.get("diagonal") or {}).get("n_cells", 0) or 0
        inf_off = (infinity.get("off_diagonal") or {}).get("n_cells", 0) or 0
        if not inf_off and not inf_diag:
            return None
        parts = [f"Fisher r-to-z: {int(inf_off)} off-diagonal matrix cell(s) at r = ±1 left as ±infinity (unbounded policy)"]
        n_edges = int((infinity.get("off_diagonal") or {}).get("n_unique_undirected_edges", 0) or 0)
        if n_edges:
            parts.append(f"{n_edges} unique undirected edge(s) affected")
        if inf_diag:
            parts.append(f"{int(inf_diag)} diagonal cell(s) (r = 1 by construction, not observed edge saturation)")
        return "; ".join(parts) + "."
    if not n_off and not n_diag:
        return None
    params = report.get("parameters") or {}
    if report.get("policy") == "clip_r":
        bound = f"|r| > {params.get('clip_r', 0.9999):g}"
    elif report.get("policy") == "clip_z":
        bound = f"|z| > {params.get('max_z', 5):g}"
    else:
        bound = "unbounded"
    n_obs = report.get("n_observations")
    obs = f" across {n_obs} observation matrices" if n_obs else ""
    parts = [f"Fisher r-to-z: {n_off} off-diagonal matrix cell(s){obs} at {bound} were clipped"]
    n_edges = int(off.get("n_unique_undirected_edges_clipped", 0) or 0)
    if n_edges:
        parts.append(f"{n_edges} unique undirected edge(s) affected")
    if n_diag:
        parts.append(f"{n_diag} diagonal cell(s) (r = 1 by construction, not observed edge saturation)")
    return "; ".join(parts) + "."


_SOURCE_LABELS = {
    "computed_by_conninfpy": "computed by the connectivity backend",
    "loader_denoising_pipeline": "applied by the loader denoising pipeline",
    "input_file_name_token": "declared by the input file name",
    "input_manifest_declared": "declared by the dataset manifest",
    "app_rank_correlation_pipeline": "ranks computed at load time in the app",
}


def _source_label(source):
    source = str(source or "")
    return _SOURCE_LABELS.get(source, source)


def describe_per_site_facts(record):
    """Per-site GSR lines ``[(site, value, source)]`` from ``per_site``, or [].

    Shows the heterogeneous context a combined dataset carries (each site's
    own known value and source) without claiming a common value: the global
    record stays unknown where the sites disagree.
    """
    per_site = (record or {}).get("per_site")
    if not isinstance(per_site, dict):
        return []
    rows = []
    for site in sorted(per_site):
        gsr = (per_site[site] or {}).get("gsr")
        if isinstance(gsr, dict) and isinstance(gsr.get("value"), bool):
            rows.append((site, "yes" if gsr["value"] else "no", _source_label(gsr.get("source"))))
        else:
            rows.append((site, "unknown", "no explicit source recorded"))
    return rows


def describe_preprocessing_facts(record):
    """Rows ``[(label, value, source)]`` for the dataset card; unknown stays unknown.

    Only known facts are stated with their source; everything else is
    explicitly unknown — never guessed from the connectivity values
    (e.g. GSR is never inferred from negative edges).
    """
    from conninfpy.preprocessing import normalize_preprocessing_record

    record = normalize_preprocessing_record(record) if record else {}

    def known(key):
        fact = record.get(key)
        return (fact.get("value"), _source_label(fact.get("source"))) if isinstance(fact, dict) else (None, None)

    rows = []
    gsr, gsr_source = known("gsr")
    rows.append(("Global-signal regression (GSR)",
                 "yes" if gsr is True else "no" if gsr is False else "unknown",
                 gsr_source or "no explicit source recorded"))
    method, method_source = known("connectivity_method")
    method_names = {"pearson": "Pearson correlation", "partial": "partial correlation",
                    "spearman": "Spearman rank correlation"}
    rows.append(("Connectivity estimator", method_names.get(method, "unknown"),
                 method_source or "no explicit source recorded"))
    standardize, std_source = known("standardize")
    rows.append(("Connectivity standardization",
                 "on" if standardize is True else "off" if standardize is False else "unknown",
                 std_source or "no explicit source recorded"))
    shrinkage_fact = record.get("shrinkage")
    if isinstance(shrinkage_fact, dict):
        value = shrinkage_fact.get("value")
        if value is None:
            text = "none"
        elif value == "ledoit_wolf":
            text = "Ledoit–Wolf (intensity estimated per observation)"
        elif value == "oas":
            text = "OAS (intensity estimated per observation)"
        else:
            text = f"fixed intensity δ = {float(value):g}"
        rows.append(("Covariance shrinkage", text, _source_label(shrinkage_fact.get("source"))))
    else:
        rows.append(("Covariance shrinkage", "unknown", "no explicit source recorded"))
    tr, tr_source = known("tr_seconds")
    rows.append(("Repetition time (TR)",
                 f"{float(tr):g} s" if isinstance(tr, (int, float)) else "unknown",
                 tr_source or "no explicit source recorded (frame counts are not TRs)"))
    return rows

def current_contrast_name():
    """Build a stable label for exports and evidence packets."""
    plan = st.session_state.get("run_plan") or {}
    question = plan.get("question_type") or "connectivity_analysis"
    method = plan.get("method")
    return f"{question}_{method}" if method else question

def clear_downstream_results():
    """Clear outputs and completed background tasks invalidated upstream."""
    for key in (
        "inference_result",
        "edges_df",
        "companion_inference_result",
        "companion_edges_df",
        "companion_method",
        "annotation",
        "evidence_packet",
        "run_plan",
    ):
        st.session_state[key] = None

    # Task objects retain their result payload after completion. Leaving one in
    # session state lets Tab 2 republish an old result after a dataset reload.
    # A still-running thread cannot be safely killed, but removing its task
    # handle prevents its stale payload from being promoted into this session.
    for task_key in ("inference_task",):
        task = st.session_state.get(task_key)
        if task is not None and getattr(task, "status", None) == "running":
            task.status = "invalidated"
        st.session_state.pop(task_key, None)


def active_analysis_atlas(base_atlas=None):
    """Use the loaded ROI/dataset/reference atlas before draft sidebar metadata."""
    if st.session_state.get("sub_atlas") is not None:
        return st.session_state.sub_atlas
    if st.session_state.get("dataset_atlas") is not None:
        return st.session_state.dataset_atlas
    if "loaded_reference_atlas" in st.session_state and (
        st.session_state.get("connectivity_data") is not None
        or st.session_state.get("inference_result") is not None
    ):
        return st.session_state.loaded_reference_atlas
    return base_atlas


def atlas_for_annotation(atlas, n_nodes, *, warn=True):
    """Return atlas only when it matches the result dimensionality."""
    if atlas is None or len(atlas) == n_nodes:
        return atlas
    if warn:
        st.warning(
            f"Atlas metadata has {len(atlas)} ROIs, but the inference result has {n_nodes} nodes. "
            "The significant-edge table will be shown without atlas annotations."
        )
    return None


def data_is_fisher_z():
    """Use explicit ingestion metadata instead of value-range heuristics."""
    return st.session_state.get("connectivity_data_kind") == "fisher_z"


def result_is_stale(run_plan=None):
    """True when a result belongs to a different loaded dataset or atlas."""
    plan = run_plan if run_plan is not None else st.session_state.get("run_plan")
    if not plan:
        return False
    current_hash = st.session_state.get("loaded_settings_hash")
    if plan.get("loaded_settings_hash") != current_hash:
        return True
    current_atlas_sig = st.session_state.get("loaded_atlas_signature", st.session_state.get("active_atlas_signature"))
    if plan.get("active_atlas_signature") != current_atlas_sig:
        return True
    return False


def effect_direction_labels(run_plan=None):
    """Return reader-facing labels for positive/negative statistical tails.

    After a prespecified one-sided run (``run_plan['tested_tail']``) the
    direction that was not tested is marked "(not tested)".
    """
    plan = run_plan if run_plan is not None else (st.session_state.get("run_plan") or {})
    labels = _direction_labels_for_question(plan)
    tested = plan.get("tested_tail")
    if tested in ("positive", "negative"):
        other = "negative" if tested == "positive" else "positive"
        labels[other] = f"{labels[other]} (not tested)"
        labels[f"{other}_title"] = f"{labels[f'{other}_title']} (not tested)"
    return labels


def _direction_labels_for_question(plan):
    """Positive/negative tail labels for the design's question type."""
    question = plan.get("question_type")

    if question == "Group Difference":
        target = plan.get("target_group")
        reference = plan.get("reference_group")
        if target is None or reference is None:
            return {
                "positive": "Positive group effect",
                "negative": "Negative group effect",
                "positive_title": "Positive group effect",
                "negative_title": "Negative group effect",
            }
        
        target_str = str(target)
        reference_str = str(reference)
        if target_str.isdigit() or (target_str.startswith("Group") and target_str[5:].strip().isdigit()):
            target_lbl = f"Group {target_str}" if target_str.isdigit() else target_str
        else:
            target_lbl = target_str
            
        if reference_str.isdigit() or (reference_str.startswith("Group") and reference_str[5:].strip().isdigit()):
            reference_lbl = f"Group {reference_str}" if reference_str.isdigit() else reference_str
        else:
            reference_lbl = reference_str
            
        return {
            "positive": f"{target_lbl} > {reference_lbl}",
            "negative": f"{reference_lbl} > {target_lbl}",
            "positive_title": f"Higher connectivity in {target_lbl}",
            "negative_title": f"Higher connectivity in {reference_lbl}",
        }

    if question == "Paired Condition":
        target = plan.get("target_condition")
        baseline = plan.get("baseline_condition")
        if target is None or baseline is None:
            return {
                "positive": "Positive paired effect",
                "negative": "Negative paired effect",
                "positive_title": "Positive paired effect",
                "negative_title": "Negative paired effect",
            }
        return {
            "positive": f"{target} > {baseline}",
            "negative": f"{baseline} > {target}",
            "positive_title": f"{target} > {baseline}",
            "negative_title": f"{baseline} > {target}",
        }

    predictor = plan.get("interest_var") or plan.get("predictor") or "predictor"
    if question == "Continuous Predictor":
        return {
            "positive": f"Positive association with {predictor}",
            "negative": f"Negative association with {predictor}",
            "positive_title": f"Positive association with {predictor}",
            "negative_title": f"Negative association with {predictor}",
        }

    return {
        "positive": "Positive effect",
        "negative": "Negative effect",
        "positive_title": "Positive effect",
        "negative_title": "Negative effect",
    }

def is_public_mode() -> bool:
    """Whether this deployment disables access to user-selected server paths."""
    return os.environ.get("CONNINFPY_PUBLIC_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def require_local_paths():
    if is_public_mode():
        raise ValueError("Server filesystem paths are disabled in public mode.")


def resolve_project_path(user_path):
    """Resolve a user-entered path while keeping writes inside the project."""
    require_local_paths()
    candidate = Path(user_path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Save path must stay inside the project: {root}")
    return resolved

def atlas_has_networks(atlas):
    """True when atlas has usable, non-placeholder network labels."""
    if atlas is None or getattr(atlas, "networks", None) is None:
        return False
    labels = [str(x).strip() for x in atlas.networks if str(x).strip()]
    return bool(labels) and set(labels) != {"Unknown"}

def atlas_has_coords(atlas):
    """True when atlas has finite MNI-like coordinates for all ROIs."""
    if atlas is None or getattr(atlas, "coords", None) is None:
        return False
    coords = np.asarray(atlas.coords)
    return coords.ndim == 2 and coords.shape[1] == 3 and np.isfinite(coords).all()

def atlas_label(atlas, fallback="No atlas metadata"):
    """Display label for an ROI index under an atlas."""
    if atlas is None:
        return fallback
    return getattr(atlas, "source", None) or f"Custom atlas ({len(atlas)} ROIs)"

def atlas_from_dataframe(df, *, source="Custom atlas"):
    """Build AtlasInfo from a flexible custom atlas CSV dataframe."""
    column_map = {str(c).strip().lower(): c for c in df.columns}
    label_col = next(
        (column_map[c] for c in ("name", "label", "roi", "roi_name", "region", "parcel") if c in column_map),
        None,
    )
    if label_col is None:
        raise ValueError("Custom atlas CSV must include a label column such as name, label, roi, or roi_name.")

    network_col = next(
        (column_map[c] for c in ("network", "system", "module", "community", "lobe") if c in column_map),
        None,
    )
    hemi_col = next(
        (column_map[c] for c in ("hemisphere", "hemi", "side") if c in column_map),
        None,
    )

    labels = df[label_col].astype(str).str.strip().tolist()
    if any(not label for label in labels):
        raise ValueError("Custom atlas labels must be non-empty.")

    if network_col is None:
        networks = ["Unknown"] * len(labels)
    else:
        networks = df[network_col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown").tolist()

    coords = None
    coord_cols = []
    for names in (("x", "y", "z"), ("mni_x", "mni_y", "mni_z")):
        if all(name in column_map for name in names):
            coord_cols = [column_map[name] for name in names]
            break
    if coord_cols:
        coords_df = df[coord_cols].apply(pd.to_numeric, errors="coerce")
        if not coords_df.isna().all().all():
            if coords_df.isna().any().any():
                raise ValueError("Custom atlas coordinate columns must be complete numeric x/y/z values or omitted.")
            coords = coords_df.to_numpy(dtype=float)

    hemisphere = None
    if hemi_col is not None:
        hemisphere = df[hemi_col].fillna("").astype(str).str.strip().tolist()
        if not any(hemisphere):
            hemisphere = None

    return AtlasInfo(
        labels=labels,
        networks=networks,
        coords=coords,
        hemisphere=hemisphere,
        source=source,
    )

def load_custom_atlas_csv(path_or_buffer, *, source="Custom atlas"):
    """Load a user-supplied atlas CSV from the workspace."""
    if isinstance(path_or_buffer, (str, os.PathLike)):
        require_local_paths()
    df = pd.read_csv(path_or_buffer)
    if df.empty:
        raise ValueError("Custom atlas CSV has no rows.")
    return atlas_from_dataframe(df, source=source)

def make_sub_atlas(atlas, roi_idx):
    """Create an AtlasInfo restricted to selected ROI indices."""
    if atlas is None:
        return None
    coords = atlas.coords[roi_idx] if atlas.coords is not None else None
    hemisphere = [atlas.hemisphere[i] for i in roi_idx] if atlas.hemisphere is not None else None
    return AtlasInfo(
        labels=[atlas.labels[i] for i in roi_idx],
        networks=[atlas.networks[i] for i in roi_idx],
        coords=coords,
        hemisphere=hemisphere,
        source=f"{atlas.source} (Subnetwork of {len(roi_idx)} ROIs)",
    )


HELP_TEXT = {
    "synthetic_dataset": """
**Synthetic connectivity data**

Generate artificial participants with connectivity matrices and a known pattern of affected connections. Use these data to explore the workflow and compare detected connections with the simulated ground truth.

- **Simulation paradigm** chooses a two-group comparison, a two-group comparison with a confound, or a continuous variable with a confound.
- For two-group simulations, set group sizes independently. **Group variability** adds independent noise to each subject and edge. Its variance is a percentage of the approximate original Fisher-z sampling variance: 0% adds nothing, 100% adds approximately the original variance again. The reference is `1 / (T − 3)` for Gaussian time series (T = 30), not each edge's observed variance; a confound can add further variability.
- **Target scenario** controls where the signal is placed in the network. **Signal of interest** controls its strength; `0` means no signal of interest. A confound can still produce an effect when its strength is nonzero.
- In both two-group paradigms, the signal is introduced in **Group 2** (`group = 1`) relative to **Group 1** (`group = 0`); the observed difference also reflects sampling variation and any confound.
- In confound-enabled paradigms, choose its strength, its target correlation with the variable of interest, and its topology. The observed correlation can differ from the target.
- The selected atlas determines the number of regions and, when available, network labels. Without an atlas, choose the number of regions and synthetic modules.
- **Random seed** makes a dataset reproducible. A different seed changes the simulated data and may also change the affected connections in random scenarios.

Choose Fisher-z output or correlation-scale output (r). Confound effects and variability adjustments are made on the Fisher-z scale before the optional conversion to r. These edgewise adjustments can produce matrices that are not valid correlation matrices, even though r values remain between −1 and 1. Synthetic data illustrate controlled scenarios; they do not represent real participants or establish how a method will perform on a real dataset.
""",
    "data_source": """
**Data Ingestion Methods**
- **Generate Synthetic Dataset**: Simulates groups/conditions and connectivity changes for testing and debugging.
- **Built-in / Standard Dataset Structures**: Pre-configured pipelines for standard cohorts like `ABIDE-mini` and `OpenClose`.
- **Dataset Manifest (data.yaml)**: Imports custom, full, or private local datasets using portable YAML manifest configurations.
""",
    "built_in_demo": """
**Built-in Demo Datasets**
- **ABIDE-mini**: Curated 5-site subset of ABIDE (204 subjects) preserving complete phenotypic columns and group balances.
- **OpenClose**: Paired Eyes-Open vs. Eyes-Closed resting-state dataset using a 182-ROI unified Schaefer-200 mask.
- **NBS-Schizophrenia**: The example of the MATLAB NBS toolbox (Zalesky et al., 2010): 12 people with schizophrenia and 15 healthy controls, 74 AAL regions. Its published result is known, so it serves as a reference check.
*Note: Demo datasets are for trying out workflow features and diagnostic tests, not definitive scientific analysis. Full ABIDE, stress, and local/private datasets should be imported from a manifest.*
""",
    "openclose_multisite": """
**Open-Close Paired Demo**

- **Both sites (IHB + China)** loads the paired Eyes-Open and Eyes-Closed observations from both cohorts and adds a `site` column.
- Every subject has both conditions at the same site. Use **Paired Condition** inference with `subject_id` and `condition`.
- This is a useful demonstration that the within-subject Open-Close contrast is less sensitive to between-site offsets than a between-subject comparison. The site column remains available for design inspection and sensitivity checks.
""",
    "china_close_close": """
**China Close-Close Test-Retest Example**

This is a paired null/control comparison of two Eyes-Closed runs from the same China participants. It is useful for checking method specificity: a valid Open-Close result should not be reproduced strongly when comparing repeated closed-eyes runs. One participant with an incomplete second run is excluded automatically.
""",
    "nbs_schizophrenia": """
**NBS Schizophrenia Example (reference check)**

The example data of the MATLAB NBS toolbox v1.2 (Zalesky, Fornito & Bullmore, 2010, *NeuroImage* 53:1197-1207): resting-state wavelet correlations for 12 people with schizophrenia (`SZ`) and 15 healthy controls (`HC`) in 74 AAL regions. The dataset brings its own AAL-74 atlas (MNI coordinates, no networks), so the sidebar atlas is not used; network-aware methods are unavailable.

**Published result.** NBS tests one direction, controls > patients (contrast `[-1,1]`), with a pooled-variance t on the raw correlations, tau = 3.1: one component of 40 edges with weaker connectivity in schizophrenia, p ≈ 0.04 (NBS v1.2: extent 0.0398, intensity 0.0462). ConnInfPy gives the same edge t-statistics (to 1e-14), the same 40 edges and p within Monte-Carlo error for both component statistics (`tests/test_nbs_schizophrenia_reference.py`).

To reproduce it in the app:
1. **Group Difference**, group column `group`, reference `SZ`, target `HC`, no covariates.
2. **Group variances: Equal (Student's t)**, the default, as in NBS. Welch's t gives 28 instead of 40 edges above tau here: on the r scale the patients' correlations are more variable and the groups are unbalanced (12 vs 15).
3. **Hypothesis direction: One-sided: HC > SZ.**
4. Method **Classical connected component (NBS)**, tau = 3.1, statistic `extent` (or `intensity`), 5000 permutations.
5. Keep the raw correlations: the app analyses data on the scale they were loaded with, so do not press **Apply Fisher-z now**.

Expected: one component of 40 edges, HC > SZ, p ≈ 0.04. The direction matters: with the same settings a two-sided test gives the same component with p ≈ 0.07, not significant at 0.05.

| Scale | Direction | Edges above tau | Component p |
|---|---|---|---|
| r | one-sided (NBS) | 40 | ≈ 0.04 |
| r | two-sided | 40 | ≈ 0.07 |
| Fisher z | one-sided | 39 | ≈ 0.04 |
| Fisher z | two-sided | 39 | ≈ 0.08 |

Data: NBS v1.2 example, GNU GPL v3 or later; see `datasets/demo/NBS_schizophrenia/README.md`.
""",
    "manifest_import": """
**Manifest Ingestion**
- A `data.yaml` file declares `schema_version`, `name`, a supported `loader`, and its file `paths`. Add `params` for loader options, `preprocessing` for requested connectivity construction, and `checks` for expected dimensions.
- Known preprocessing facts (GSR yes/no, estimator, standardize, shrinkage, TR) are carried from loaders and manifests into the analysis record; `preprocessing.applied` in the manifest declares what was done upstream (a declaration — the record keeps it separate from what this pipeline computed), while anything undeclared stays **unknown** and is never inferred from the connectivity values.
- Relative paths are resolved from the manifest's own folder. Use `BUILTIN:<file>` for packaged atlas assets, for example `BUILTIN:bna246.csv`.
- For ready connectivity matrices, choose the matching loader and include atlas metadata whenever ROI labels, networks, coordinates, or brain plots are needed.
- **Prepared Zerssen example:** use `loader: CustomPreparedZerssenLoader`; under `paths`, provide `prepared_npz`, `set84_bnt_ids`, and `atlas_metadata: BUILTIN:bna246.csv`. Set `params.matrix_key` to select the analysis array. `connectivity_pearson_z`, `connectivity_tangent`, and `connectivity_tangent_hc_ref` use the full 246-ROI BNA atlas; `set84m_pearson_z` and `set84w_pearson_z` use the ordered 84-ROI BNA subset defined by BNT IDs. Set `data_kind: fisher_z` for the Pearson-z arrays and `data_kind: correlation` for tangent arrays. Update `checks.expected_rois` to match the selected 84- or 246-ROI matrix.
- The dataset-provided atlas takes precedence after loading. The sidebar atlas is then only a reference for datasets without their own metadata.
- See `examples/fmriprep_nilearn/manifest_template.yaml` (or `datasets/abide_mini.yaml`) for a working template; replace its paths with your own.
""",
    "source_type": """
**Source File Types**
- **fMRIPrep derivatives / preprocessed NIfTIs**: Volumetric neuroimaging directories that require anatomical ROI atlas extraction.
- **Timeseries directories / arrays**: Denoised timeseries matrices requiring connectivity correlation construction.
- **Connectivity matrices**: Ready-to-inference Fisher-z or Pearson correlation matrices.
""",
    "dataset_preview": """
**Dataset Preview Metrics**
- **Subjects**: Number of unique participants.
- **Observations**: Total runs/sessions (observations may exceed subjects for paired/repeated designs).
- **ROIs**: Parcellation node count (must align with the reference atlas).
- **File size/count**: Helps verify if files are correct and identify unintended full-dataset imports.
""",
    "source_validation": """
**Source Validation Checks**
- **Errors**: Structural issues (non-square matrices, NaNs, missing folders, phenotype mismatch) that block loading.
- **Warnings**: Non-blocking alerts (e.g. low-coverage ROIs, minor group imbalances) indicating potential limitations in interpretation.
""",
    "preprocessing": """
**Ingestion Preprocessing**
- **Correlation Estimator**: Converts timeseries data to connectivity matrices using Pearson or Spearman correlation.
- **Fisher r-to-z**: Standard transformation recommended before running statistical inference.
- **Diagonal Zeroing**: Diagonal elements are set to zero before Fisher z-transform to prevent infinite values.
""",
    "inference_ready": """
**Inference-Ready Checks**
- Enforces that data is a 3D connectivity tensor of shape `(observations, ROIs, ROIs)`.
- Verified symmetric, finite, and aligned with phenotype rows before running statistics.
""",
    "analysis_setup": """
**Analysis Design Binding**
- **Group comparison**: Compares discrete groups (e.g., patient vs. control) using a group column.
- **Paired comparison**: Evaluates within-subject changes across paired conditions (e.g., open vs. close).
- **Continuous predictor**: Models correlation against continuous behavioral scales (e.g. ADOS) with optional confound regression.
- **Site harmonization**: Adjusts for batch effects in multi-site designs using ComBat or ComBat-Apply strategies.
""",
    "run_plan": """
**Run Plan Review**
- **Permutation Budget**: Selects the statistical precision. Larger budgets (e.g., 5000+ permutations) are required for paper-grade inference.
- **GLM / Contrast**: Confirms interest predictors, covariates, and the contrast matrix.
- **Reloading**: If settings were changed after importing, remember to reload the dataset to refresh stats matrices.
""",
    "inference_results": """
**Inference Results**
- Displays the statistical significance of network edges after family-wise error rate (FWER) correction.
- Interactive network graphs show the enhanced subnetworks.
- Node summary lists anatomical regions sorted by degree burden.
""",
    "bna_annotation": """
**Brainnetome Annotation**
Describes the ROIs and the detected connections with the Brainnetome atlas (BNA-246; Fan et al., 2016).

- **Support**: each ROI is described by the tissue it sampled (its parcel, or a sphere around its coordinate), as shares of Brainnetome regions. A label is given at the finest level (subregion, gyrus, lobe) holding at least the purity threshold; otherwise the ROI is *mixed*. Cerebellum, brainstem and white matter are *unassigned* (not covered by BNA-246).
- **Template**: the atlas is on the FSL MNI152 grid. Choose the template the signals were extracted in; other MNI variants (e.g. fMRIPrep's MNI152NLin2009cAsym, or an unknown variant) are flagged approximate.
- **Connections**: hemispheric relation (homotopic or not), anatomical level, circuit class (cortico-cortical, thalamo-cortical, ...), distance, sphere overlap, population structural connectivity (when the Brainnetome tractography maps are available) and network relation (the atlas's own network labels, e.g. Schaefer Yeo-7; the Brainnetome division for Brainnetome regions).
- **BrainMap domains**: published forward-inference associations (which tasks activate a region in the literature). They are not the function of a region or of a connection; a region without terms has too few BrainMap studies.
- **Summaries** put the detections next to the tested opportunities; nothing is tested and the correction is unchanged.
- **Distance QC**: a contrast that covaries with head motion often shows short-range increases and long-range decreases.
""",
    "workspace_documentation": """
**Workspace Documentation**
- Indexes and displays active research notes, design plans, and implementation logs from the wiki.
"""
}


def render_help(key: str):
    """Render the collapsible help popover for a UI section."""
    if key in HELP_TEXT:
        with st.popover("ℹ️ Help Info", width="content"):
            st.markdown(HELP_TEXT[key])


def list_manifest_files() -> list[str]:
    """List all YAML manifest files directly under the datasets/ folder of the project root."""
    require_local_paths()
    datasets_dir = PROJECT_ROOT / "datasets"
    if not datasets_dir.exists():
        return []
    yamls = list(datasets_dir.glob("*.yaml")) + list(datasets_dir.glob("*.yml"))
    rel_yamls = []
    for y in yamls:
        try:
            rel_path = y.relative_to(PROJECT_ROOT).as_posix()
            rel_yamls.append(rel_path)
        except Exception:
            rel_yamls.append(y.as_posix())
    return sorted(rel_yamls)


def align_atlas_coordinates(target_atlas, reference_atlas):
    """Align and copy coordinates from reference_atlas to target_atlas by matching ROI names."""
    if target_atlas is None or reference_atlas is None:
        return target_atlas
    if getattr(reference_atlas, "coords", None) is None:
        return target_atlas
        
    ref_coords_map = {}
    for label, coord in zip(reference_atlas.labels, reference_atlas.coords, strict=True):
        ref_coords_map[label.strip().lower()] = coord
        
    new_coords = []
    has_any = False
    for label in target_atlas.labels:
        clean = label.strip().lower()
        if clean in ref_coords_map:
            new_coords.append(ref_coords_map[clean])
            has_any = True
        else:
            new_coords.append([np.nan, np.nan, np.nan])
            
    if has_any:
        target_atlas.coords = np.array(new_coords)
    return target_atlas
