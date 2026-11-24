import hashlib
import io

import numpy as np
import pandas as pd
import streamlit as st
from apps.utils.ui_state import persistent_key, reset_widget_scopes
from apps.utils.workspace import DATA_TAB, go_to_step
import threading
import traceback
from conninfpy.atlas import AtlasInfo
from conninfpy import InferenceCancelled, ProgressSession, analyze
from conninfpy.harmonize.diagnostics import analyze_design_diagnostics
from conninfpy.defaults import (
    DEFAULT_ACCELERATION,
    DEFAULT_EXTENT_EXPONENT,
    DEFAULT_HEIGHT_EXPONENT,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_NBS_STAT,
    DEFAULT_NBS_THRESHOLD,
    DEFAULT_N_PERMUTATIONS,
    DEFAULT_N_THRESHOLDS_PERMUTATION,
    DEFAULT_START_THRESHOLD,
)
from apps.utils.helpers import (
    active_analysis_atlas,
    atlas_for_annotation,
    atlas_has_networks,
    clear_downstream_results,
    data_is_fisher_z,
    effect_direction_labels,
    fisher_clip_note,
    render_help,
)

class InferenceTask:
    """Background inference job with progress reporting.

    NETSTAT-27: ``status`` also takes ``cancelled`` (a cooperative cancel at
    a permutation chunk boundary, or a job superseded by a newer run). A
    cancelled task never carries a result: ``result``/``edges``/``run_plan``
    stay ``None`` so nothing can be published or exported from an
    incomplete null distribution.
    """

    def __init__(self):
        """Initialize an idle task with empty result slots."""
        # idle, running, success, failed, cancelled
        self.status = "idle"
        self.progress_message = ""
        self.result = None
        self.edges = None
        self.comp_result = None
        self.comp_edges = None
        self.run_plan = None
        self.error = None
        # Parent-side progress / cooperative cancellation (NETSTAT-27).
        self.cancel_event = threading.Event()
        self.cancel_reason = None
        self.generation = 0
        self.progress_done = 0
        self.progress_total = 0
        self.progress_stage = None
        self.launch_signature = None
        self.finished_generation = None

    # -- progress / cancel (called from the background thread) -----------
    def _on_progress(self, done, total):
        """Record parent-side ``(done, total)`` permutation progress."""
        try:
            self.progress_done = int(done)
            self.progress_total = int(total)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass

    def request_cancel(self, reason=None):
        """Ask the running pipeline to stop at the next chunk boundary."""
        if reason:
            self.cancel_reason = reason
        self.cancel_event.set()


class _StaleJob(Exception):
    """This background job was superseded; its payload must never publish."""


def _worker_plan(use_mp, n_processes, n_permutations):
    """Requested vs effective worker facts for the run plan (honest unknowns)."""
    from conninfpy.stats._workers import describe_worker_policy

    try:
        return describe_worker_policy(
            use_mp=bool(use_mp),
            n_processes=n_processes,
            n_permutations=n_permutations,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return {"request_error": f"{type(exc).__name__}: {exc}"}


def _normalize_fact(value):
    """Normalize one launch fact into a comparable, hashable plain value."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, np.generic):
        return _normalize_fact(value.item())
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(np.asarray(value))
        if arr.size <= 200_000:
            digest = hashlib.sha256(arr.tobytes()).hexdigest()
        else:  # pragma: no cover - very large arrays stay a shape/type fact
            digest = f"large:{float(np.nansum(arr))}"
        return ("ndarray", arr.dtype.str, tuple(arr.shape), digest)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _normalize_fact(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_fact(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_normalize_fact(v) for v in value), key=repr))
    return f"{type(value).__name__}:{value!r}"


def _run_signature(**facts):
    """Comparable signature of the complete scientific launch/design snapshot.

    Every fact that changes the design or the execution of a run must be
    passed in: group/condition levels, method and all its parameters, grid
    values, declared primary cell, edge mask / ROI restriction, condition
    confounds, site recipe, columns, tail/variance/missing policy, seed,
    permutation count, worker settings and the data/atlas identity. Values are
    normalized (nested dicts, sequences, scalars, ndarray digests) so the same
    facts always produce the same signature.
    """
    return {str(key): _normalize_fact(value) for key, value in facts.items()}


def _task_cancel_requested(task):
    """True when this task has a parent-side cancel token that was set."""
    event = getattr(task, "cancel_event", None)
    return event is not None and bool(event.is_set())


def _invalidate_superseded_task(task, signature):
    """Cancel a running job whose loaded data/design no longer matches."""
    if getattr(task, "status", "") != "running":
        return
    launch = getattr(task, "launch_signature", None)
    if launch and launch != signature:
        task.request_cancel(
            "the loaded data or design changed while this run was active"
        )


ABIDE_DEFAULT_CONFOUNDS = ("Age", "Sex", "Motion_FD")
NON_COVARIATE_NAMES = {
    "subject",
    "subject_id",
    "sub",
    "subj",
    "sub_id",
    "id",
    "file_id",
    "session",
    "site",
    "group",
    "group_interest",
    "dx",
    "diagnosis",
    "class",
    "label",
}


def _numeric_covariate_candidates(pheno_df, exclude=()):
    """Numeric pheno columns usable as covariates, minus excluded names."""
    excluded = {str(c).lower() for c in exclude if c is not None}
    candidates = []
    for col in pheno_df.columns:
        col_key = str(col).lower()
        if col_key in excluded or col_key in NON_COVARIATE_NAMES:
            continue
        if pd.api.types.is_numeric_dtype(pheno_df[col]):
            candidates.append(col)
    return candidates


def _default_abide_confound_vars(candidates):
    """ABIDE default confound columns that exist among ``candidates``."""
    return [col for col in ABIDE_DEFAULT_CONFOUNDS if col in candidates]


def _resolve_site_strategy(recipe_choice, test_type, confound_vars):
    """Return (effective_recipe, harmonize_arg, note) for app-driven site handling."""
    if recipe_choice == "ComBat + site-aware GLM":
        if test_type != "glm":
            return (
                "Site-stratified permutation",
                None,
                "ComBat + site-aware GLM requires a GLM design. This run will pass sites for exchangeability/strata and skip ComBat.",
            )
        if not confound_vars:
            return (
                "Site-aware GLM",
                "site_dummies_glm",
                "ComBat + site-aware GLM requires at least one nuisance covariate to preserve. With no covariates selected, this run uses site dummies in the GLM and skips ComBat.",
            )
        return recipe_choice, "combat_site_dummies_glm", None

    if recipe_choice == "ComBat-only":
        if test_type != "glm":
            return (
                "Site-stratified permutation",
                None,
                "ComBat-only requires a GLM design with nuisance covariates. This run will pass sites for exchangeability/strata and skip ComBat.",
            )
        if not confound_vars:
            return (
                "No site handling",
                None,
                "ComBat-only requires at least one nuisance covariate to preserve. With no covariates selected, ComBat is skipped.",
            )
        return recipe_choice, "combat_only", None

    if recipe_choice == "Site-aware GLM":
        if test_type != "glm":
            return (
                "Site-stratified permutation",
                None,
                "Site dummies require a GLM design. This run will pass sites for exchangeability/strata instead.",
            )
        return recipe_choice, "site_dummies_glm", None

    return recipe_choice, None, None


def _format_runtime(seconds):
    """Human-readable wall-clock string (sec / min / h)."""
    seconds = max(1.0, float(seconds))
    if seconds < 90.0:
        return f"{int(round(seconds))} sec"
    if seconds < 3600.0:
        mins = int(round(seconds / 60.0))
        return f"{mins} min"
    hours = seconds / 3600.0
    return f"{hours:.1f} h"


def _method_plan_parameters(method, method_kwargs):
    """Keep only human-readable settings relevant to the selected method."""
    if method in ("tfnbs", "ni_tfnbs", "fbc_tfnbs"):
        keys = ("e", "h", "n", "start_thres", "min_cluster_size", "normalization")
        labels = {
            "e": "E",
            "h": "H",
            "min_cluster_size": "minimum block size",
            "n": "integration steps",
            "start_thres": "start threshold",
            "normalization": "normalization",
        }
    elif method == "nbs":
        keys = ("threshold", "nbs_stat")
        labels = {"threshold": "tau", "nbs_stat": "component statistic"}
    else:
        return {}
    result = {
        labels[key]: method_kwargs[key]
        for key in keys
        if key in method_kwargs
    }
    if method in ("tfnbs", "ni_tfnbs", "fbc_tfnbs"):
        result["integration"] = "exact" if method_kwargs.get("n") is None else "right-grid"
    return result


def _build_prelaunch_design_qc_summary(
    pheno_df,
    connectivity_data,
    *,
    question_choice,
    test_type,
    group_col=None,
    ref_group=None,
    target_group=None,
    interest_var=None,
    subject_col=None,
    condition_col=None,
    baseline_val=None,
    target_val=None,
    confound_vars=None,
    site_col=None,
    harmonization_choice=None,
    roi_indices=None,
):
    """Build pre-launch sample/group, site cross-tab, QC, and rank/VIF summary without mutating data."""
    confound_vars = list(confound_vars or [])
    summary = {
        "n_total_rows": 0 if pheno_df is None else int(len(pheno_df)),
        "n_complete_rows": 0,
        "n_dropped_na": 0,
        "na_counts": {},
        "sample_counts": {},
        "interest_stats": None,
        "site_table": None,
        "site_summary_kind": None,
        "site_notes": [],
        "qc_rows": [],
        "connectivity_qc": None,
        "design_preview": None,
        "warnings": [],
    }
    if pheno_df is None or len(pheno_df) == 0:
        return summary

    req_cols = []
    if test_type == "glm":
        if question_choice == "Group Difference" and group_col:
            req_cols.append(group_col)
        elif interest_var:
            req_cols.append(interest_var)
        req_cols.extend(confound_vars)
    elif test_type == "paired":
        for c in (condition_col, subject_col):
            if c:
                req_cols.append(c)
    elif test_type == "two-sample" and group_col:
        req_cols.append(group_col)
    if site_col:
        req_cols.append(site_col)
    req_cols = [c for c in dict.fromkeys(req_cols) if c in pheno_df.columns]

    if req_cols:
        na_counts = {c: int(pheno_df[c].isna().sum()) for c in req_cols if int(pheno_df[c].isna().sum()) > 0}
        summary["na_counts"] = na_counts
        valid_mask = pheno_df[req_cols].notna().all(axis=1).to_numpy()
    else:
        valid_mask = np.ones(len(pheno_df), dtype=bool)

    summary["n_complete_rows"] = int(valid_mask.sum())
    summary["n_dropped_na"] = int(len(pheno_df) - valid_mask.sum())
    if summary["n_dropped_na"] > 0:
        summary["warnings"].append(
            f"{summary['n_dropped_na']} subject(s) have NaN in selected model columns and will be dropped before inference."
        )

    df_valid = pheno_df.loc[valid_mask].reset_index(drop=True)
    Y_active = None
    if connectivity_data is not None:
        Y_active = connectivity_data
        if roi_indices is not None:
            Y_active = Y_active[:, roi_indices][:, :, roi_indices]
        Y_valid = Y_active[valid_mask]
    else:
        Y_valid = None

    active_df = df_valid
    active_Y = Y_valid
    sites_vec = None
    interest_vec = None
    group_labels = None

    if question_choice == "Group Difference" and group_col and group_col in df_valid.columns:
        g_mask = df_valid[group_col].isin([ref_group, target_group]).to_numpy()
        active_df = df_valid.loc[g_mask].reset_index(drop=True)
        if Y_valid is not None:
            active_Y = Y_valid[g_mask]
        n_ref = int((active_df[group_col] == ref_group).sum())
        n_target = int((active_df[group_col] == target_group).sum())
        summary["sample_counts"] = {
            f"Reference ({ref_group})": n_ref,
            f"Target ({target_group})": n_target,
            "Total in Contrast": int(len(active_df)),
        }
        group_labels = [str(v) for v in active_df[group_col].tolist()]
        interest_vec = (active_df[group_col] == target_group).to_numpy(dtype=np.float64)
        if n_ref < 2 or n_target < 2:
            summary["warnings"].append("At least one group has fewer than 2 observations.")
        if site_col and site_col in active_df.columns:
            sites_vec = active_df[site_col].astype(str).tolist()
            ct = pd.crosstab(active_df[site_col].astype(str), active_df[group_col].astype(str))
            for g in (str(ref_group), str(target_group)):
                if g not in ct.columns:
                    ct[g] = 0
            cols_order = [str(ref_group)] if str(ref_group) == str(target_group) else [str(ref_group), str(target_group)]
            ct = ct[cols_order]
            ct["Total"] = ct.sum(axis=1)
            summary["site_table"] = ct.reset_index().rename(columns={active_df[site_col].name or "row_0": "Site", "row_0": "Site"})
            if summary["site_table"].columns[0] != "Site":
                summary["site_table"] = summary["site_table"].rename(columns={summary["site_table"].columns[0]: "Site"})
            summary["site_summary_kind"] = "site_by_group"
            zero_cells = int((ct[cols_order] == 0).sum().sum())
            if zero_cells > 0:
                summary["site_notes"].append(f"{zero_cells} empty site × group cell(s) detected.")

    elif question_choice == "Continuous Predictor" and interest_var and interest_var in df_valid.columns:
        vals = pd.to_numeric(df_valid[interest_var], errors="coerce").to_numpy(dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        summary["sample_counts"] = {"Subjects": int(len(df_valid))}
        if finite.size:
            summary["interest_stats"] = {
                "predictor": interest_var,
                "n": int(finite.size),
                "mean": float(np.mean(finite)),
                "sd": float(np.std(finite)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "n_unique": int(np.unique(finite).size),
            }
        interest_vec = vals
        if site_col and site_col in df_valid.columns:
            sites_vec = df_valid[site_col].astype(str).tolist()
            site_df = pd.DataFrame({"Site": sites_vec, "val": vals}).dropna()
            if not site_df.empty:
                grp = site_df.groupby("Site")["val"].agg(
                    n="count",
                    mean="mean",
                    sd=lambda x: float(np.std(x, ddof=0)),
                    min="min",
                    max="max",
                ).reset_index()
                summary["site_table"] = grp
                summary["site_summary_kind"] = "site_by_interest"
                ss_tot = float(np.sum((site_df["val"] - site_df["val"].mean()) ** 2))
                ss_within = float(site_df.groupby("Site")["val"].apply(lambda s: np.sum((s - s.mean()) ** 2)).sum())
                if ss_tot > 1e-12:
                    r2 = max(0.0, min(1.0, 1.0 - ss_within / ss_tot))
                    summary["site_notes"].append(f"Descriptive between-site variance share in {interest_var}: R² = {r2:.3f}.")

    elif question_choice == "Paired Condition" and subject_col and condition_col and subject_col in df_valid.columns and condition_col in df_valid.columns:
        df_base = df_valid[df_valid[condition_col] == baseline_val]
        df_targ = df_valid[df_valid[condition_col] == target_val]
        base_subs = set(df_base[subject_col])
        targ_subs = set(df_targ[subject_col])
        common = sorted(base_subs & targ_subs)
        unpaired = len((base_subs | targ_subs) - set(common))
        summary["sample_counts"] = {
            f"Baseline ({baseline_val}) rows": int(len(df_base)),
            f"Target ({target_val}) rows": int(len(df_targ)),
            "Paired Subjects": int(len(common)),
            "Unpaired Excluded": int(unpaired),
        }
        if unpaired > 0:
            summary["warnings"].append(
                f"{unpaired} subject(s) lack one of the two paired conditions; a paired run "
                "is blocked until both conditions contain exactly the same subjects (no "
                "silent exclusion, no re-sorting and no positional matching)."
            )
        if site_col and site_col in df_valid.columns and common:
            idx_b = [df_base[df_base[subject_col] == s].index[0] for s in common]
            idx_t = [df_targ[df_targ[subject_col] == s].index[0] for s in common]
            b_sites = df_valid.loc[idx_b, site_col].astype(str).tolist()
            t_sites = df_valid.loc[idx_t, site_col].astype(str).tolist()
            mismatch = [s for s, bs, ts in zip(common, b_sites, t_sites) if bs != ts]
            if mismatch:
                summary["warnings"].append(
                    "Site mismatch between paired conditions for: " + ", ".join(map(str, mismatch[:5]))
                )
            site_counts = pd.Series(b_sites).value_counts().rename_axis("Site").reset_index(name="Paired Subjects")
            summary["site_table"] = site_counts
            summary["site_summary_kind"] = "paired_sites"
        if Y_valid is not None and common:
            idx_all = sorted(
                {df_base[df_base[subject_col] == s].index[0] for s in common}
                | {df_targ[df_targ[subject_col] == s].index[0] for s in common}
            )
            active_Y = Y_valid[idx_all]
            active_df = df_valid.loc[idx_all].reset_index(drop=True)

    # Available QC from pheno + connectivity
    if active_Y is not None and active_Y.ndim == 3 and active_Y.shape[0] > 0:
        n_sub, n_roi, _ = active_Y.shape
        iu = np.triu_indices(n_roi, k=1)
        upper = active_Y[:, iu[0], iu[1]]
        nonfinite = ~np.isfinite(upper)
        subj_missing = int(nonfinite.any(axis=1).sum())
        edge_missing = int(nonfinite.any(axis=0).sum())
        pct_missing = float(100.0 * nonfinite.mean()) if nonfinite.size else 0.0
        summary["connectivity_qc"] = {
            "n_subjects": int(n_sub),
            "n_rois": int(n_roi),
            "subjects_with_nan": subj_missing,
            "edges_with_nan": edge_missing,
            "pct_missing_entries": pct_missing,
        }
        if subj_missing > 0:
            summary["warnings"].append(
                f"Connectivity contains NaN in {subj_missing}/{n_sub} subject(s) ({edge_missing} edge(s)); configure missing='available_case' or inspect data if unintended."
            )

    qc_candidates = []
    for col in active_df.columns:
        low = str(col).lower()
        if pd.api.types.is_numeric_dtype(active_df[col]) and (
            low in ("n_valid_tr", "delta_s")
            or any(tok in low for tok in ("motion", "framewise", "mean_fd", "dvars"))
            or low == "fd"
        ):
            qc_candidates.append(col)
    for col in dict.fromkeys(qc_candidates):
        vals = pd.to_numeric(active_df[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if vals.size:
            row = {
                "Metric": str(col),
                "n": int(vals.size),
                "Mean": round(float(np.mean(vals)), 4),
                "Median": round(float(np.median(vals)), 4),
                "Min": round(float(np.min(vals)), 4),
                "Max": round(float(np.max(vals)), 4),
            }
            if str(col) == "n_valid_tr" and int((vals < 50).sum()) > 0:
                summary["warnings"].append(f"{int((vals < 50).sum())} subject(s) have n_valid_tr < 50.")
            summary["qc_rows"].append(row)

    # Pre-launch design preview (rank / VIF / balance) for GLM paths
    if test_type == "glm" and interest_vec is not None and len(active_df) >= 2:
        try:
            blocks = [np.ones((len(active_df), 1), dtype=np.float64)]
            col_names = ["intercept"]
            roles = {"intercept": "intercept"}
            if confound_vars:
                c_mat = active_df[confound_vars].to_numpy(dtype=np.float64)
                blocks.append(c_mat)
                for cv in confound_vars:
                    col_names.append(str(cv))
                    roles[str(cv)] = "confound"
            if sites_vec is not None and harmonization_choice in ("combat_site_dummies_glm", "site_dummies_glm"):
                s_arr = np.asarray(sites_vec)
                uniq, inv = np.unique(s_arr, return_inverse=True)
                if len(uniq) >= 2:
                    s_dum = np.zeros((len(s_arr), len(uniq) - 1), dtype=np.float64)
                    for j in range(1, len(uniq)):
                        s_dum[inv == j, j - 1] = 1.0
                    blocks.append(s_dum)
                    for u in uniq[1:]:
                        nm = f"site[{u}]"
                        col_names.append(nm)
                        roles[nm] = "site_dummy"
            iv_col = interest_vec.reshape(-1, 1)
            blocks.append(iv_col)
            target_nm = str(group_col if question_choice == "Group Difference" else interest_var)
            if target_nm in col_names:
                target_nm = f"interest:{target_nm}"
            col_names.append(target_nm)
            roles[target_nm] = "interest"
            X_prev = np.hstack(blocks)
            c_prev = np.zeros(X_prev.shape[1], dtype=np.float64)
            c_prev[-1] = 1.0
            summary["design_preview"] = analyze_design_diagnostics(
                design_kind="glm_prelaunch_preview",
                unit_of_observation="subject",
                X=X_prev,
                names=col_names,
                column_roles=roles,
                contrast=c_prev,
                focus_contrast=target_nm,
                sites=sites_vec,
                group_labels=group_labels,
                interest_name=target_nm,
                interest_values=interest_vec,
            )
        except Exception:
            summary["design_preview"] = None

    return summary


def _render_prelaunch_design_qc_summary(summary):
    """Render pre-launch sample sizes, site cross-tabs, QC metrics, and rank/VIF warnings."""
    with st.container(border=True):
        st.markdown("**Pre-Launch Sample, Site & QC Check:**")
        if summary["sample_counts"]:
            cols = st.columns(min(len(summary["sample_counts"]), 4))
            for idx, (label, val) in enumerate(summary["sample_counts"].items()):
                cols[idx % len(cols)].metric(label, val)
        if summary["interest_stats"]:
            ist = summary["interest_stats"]
            st.caption(
                f"Predictor `{ist['predictor']}` (n={ist['n']}): "
                f"mean = {ist['mean']:.3g}, SD = {ist['sd']:.3g}, "
                f"range = [{ist['min']:.3g}, {ist['max']:.3g}] ({ist['n_unique']} unique values)"
            )
        if summary["site_table"] is not None:
            st.caption(
                "Site × Group Breakdown:" if summary["site_summary_kind"] == "site_by_group"
                else "Site × Predictor Breakdown:" if summary["site_summary_kind"] == "site_by_interest"
                else "Paired Subjects by Site:"
            )
            st.dataframe(summary["site_table"], hide_index=True, width="stretch")
            for note in summary["site_notes"]:
                st.caption(note)
        cqc = summary.get("connectivity_qc")
        if cqc is not None:
            st.caption(
                f"Connectivity QC ({cqc['n_subjects']} × {cqc['n_rois']}×{cqc['n_rois']}): "
                f"{cqc['subjects_with_nan']} subject(s) with NaN, "
                f"{cqc['edges_with_nan']} edge(s) with NaN ({cqc['pct_missing_entries']:.2f}% missing entries)."
            )
        if summary["qc_rows"]:
            st.caption("Available Phenotype / Signal QC:")
            st.dataframe(pd.DataFrame(summary["qc_rows"]), hide_index=True, width="stretch")
        dp = summary.get("design_preview")
        if dp is not None:
            rank_txt = f"rank {dp['matrix_rank']}/{dp['n_columns']} (df={dp['residual_df']})"
            vif_txt = f"max VIF = {dp['vif_max']:.2f} ({dp['vif_max_col']})" if dp.get("vif_max") is not None else "VIF n/a"
            st.caption(f"GLM Design Preview: {rank_txt} | {vif_txt}")
            for fl in dp.get("flags", []):
                st.warning(f"Design check: {fl}")
        for w in summary.get("warnings", []):
            st.warning(w)


def _format_method_plan_parameters(method, method_kwargs):
    """One-line ``label=value`` summary of method parameters."""
    params = _method_plan_parameters(method, method_kwargs)
    return ", ".join(f"{label}={value}" for label, value in params.items()) or "Default settings"


def _estimate_runtime_range(
    *,
    n_subjects,
    n_nodes,
    n_permutations,
    test_type,
    method,
    use_mp,
    run_sensitivity=False,
):
    """Conservative Streamlit-facing runtime estimate for permutation runs.

    This is deliberately a range. Wall time depends heavily on BLAS, CPU
    topology, first-run compilation/cache effects, and multiprocessing startup.
    """
    n_edges = n_nodes * (n_nodes - 1) // 2
    work_units = max(1, n_permutations) * max(1, n_edges) * max(1, n_subjects)

    if test_type == "glm":
        seconds = work_units * 1.05e-7
    else:
        seconds = work_units * 3.0e-8

    method_factor = {
        "tstat": 0.75,
        "bh_fdr": 0.85,
        "nbs": 1.35,
        "tfnbs": 2.4,
        "ni_tfnbs": 3.0,
        "fbc_tfnbs": 3.2,
        "cnbs": 2.2,
    }.get(method, 2.0)
    seconds *= method_factor

    if use_mp:
        import os
        cores = os.cpu_count() or 1
        # Permutation work parallelizes, but Streamlit/process startup and
        # memory movement make near-linear core scaling unrealistic.
        speedup = min(3.0, max(1.0, cores ** 0.45))
        seconds = seconds / speedup + 20.0

    if run_sensitivity:
        seconds *= 1.75

    lower = seconds * 0.6
    upper = seconds * 1.8
    return f"{_format_runtime(lower)} - {_format_runtime(upper)}"


MOTION_QC_WIDGET = "design:motion_qc_sidecar"
MOTION_QC_FD_WIDGET = "design:motion_qc_fd_column"


def _motion_qc_fd_candidates(pheno_df):
    """Numeric phenotype columns that can carry mean framewise displacement."""
    from conninfpy.interpret.motion_qc import FD_COLUMN_CANDIDATES

    out = []
    columns = getattr(pheno_df, "columns", None)
    for col in ([] if columns is None else list(columns)):
        if not pd.api.types.is_numeric_dtype(pheno_df[col]):
            continue
        name = str(col)
        if name in FD_COLUMN_CANDIDATES or any(tok in name.lower() for tok in ("motion", "fd", "framewise")):
            out.append(name)
    return out


def _motion_qc_fd_values(pheno_df, column):
    """One FD value per phenotype row, NaN where the column is not numeric."""
    return pd.to_numeric(pheno_df[column], errors="coerce").to_numpy(dtype=np.float64)


def _motion_qc_subject_ids(pheno_df, explicit=None):
    """Subject IDs in row order, only from an explicitly named/recognised column."""
    if pheno_df is None:
        return None
    column = explicit
    if column is None:
        column = next(
            (c for c in ("subject_id", "subject", "sub_id", "participant_id") if c in pheno_df.columns),
            None,
        )
    if column is None or column not in pheno_df.columns:
        return None
    return list(pheno_df[column].to_numpy(dtype=object))


def _motion_qc_preview(spec, test_type, Y, group1, group2, *, atlas=None, scale=None):
    """Compute the descriptive side-car BEFORE the inference thread starts."""
    from conninfpy.interpret.motion_qc import motion_qc_fc, motion_qc_fc_paired

    common = dict(
        pheno=spec.get("pheno"),
        fd_units=spec.get("fd_units", "mm"),
        fd_source_name=spec.get("fd_source_name"),
        scale=scale,
        scale_source="loaded data kind (native scale)",
        atlas=atlas,
        input_stage="native input before inference (pre-launch preview)",
    )
    if test_type == "paired" and group1 is not None and group2 is not None:
        return motion_qc_fc_paired(
            group1, group2,
            fd_group1=spec.get("fd_group1"),
            fd_group2=spec.get("fd_group2"),
            subject_ids=spec.get("subject_ids"),
            pool=spec.get("pool", True),
            condition_a=spec.get("condition_a", "A"),
            condition_b=spec.get("condition_b", "B"),
            **common,
        )
    if test_type == "two-sample" and group2 is not None:
        data = np.concatenate([group1, group2], axis=0)
    else:
        data = Y if Y is not None else group1
    return motion_qc_fc(
        data,
        fd=spec.get("fd"),
        subject_ids=spec.get("subject_ids"),
        condition_labels=spec.get("condition_labels"),
        condition=spec.get("condition"),
        index_values=spec.get("index_values"),
        **common,
    )


class PairedDesignError(ValueError):
    """The two conditions of a paired contrast are not a one-to-one subject match."""


def _short_list(values, limit=8):
    """A short, readable enumeration of values for error messages."""
    values = list(values)
    items = [str(v) for v in values[:limit]]
    suffix = ", ..." if len(values) > limit else ""
    return ", ".join(items) + suffix


def _blank_subject_id_mask(series):
    """True where a subject ID is missing or empty (never filled in or guessed)."""
    text = series.map(lambda v: "" if v is None else str(v).strip())
    return text.isin(("", "nan", "NaN", "None", "<NA>", "NaT"))


def _paired_design_violations(
    pheno_df, condition_col, subject_col, baseline_val, target_val
):
    """Every reason the two conditions are not a strict one-to-one subject match.

    Nothing is repaired here: the launch and the pre-launch preview both call
    this so the reported paired n is the analysed paired n.
    """
    problems = []
    for column, label in ((condition_col, "condition"), (subject_col, "subject")):
        if column not in pheno_df.columns:
            return [f"the {label} column {column!r} is not in the phenotype table"]
    frames = {}
    for value in (baseline_val, target_val):
        frame = pheno_df[pheno_df[condition_col] == value]
        label = str(value)
        frames[label] = frame
        if frame.empty:
            problems.append(f"condition {label!r} has no rows")
            continue
        blanks = _blank_subject_id_mask(frame[subject_col])
        if bool(blanks.any()):
            problems.append(
                f"condition {label!r} has {int(blanks.sum())} row(s) with an empty subject ID"
            )
        ids = frame.loc[~blanks, subject_col].astype(str)
        duplicates = sorted(set(ids[ids.duplicated(keep=False)]))
        if duplicates:
            problems.append(
                f"condition {label!r} has duplicate subject IDs ({_short_list(duplicates)})"
            )
    if problems:
        return problems
    a_ids = frames[str(baseline_val)][subject_col].astype(str).tolist()
    b_ids = frames[str(target_val)][subject_col].astype(str).tolist()
    b_set, a_set = set(b_ids), set(a_ids)
    only_a = [s for s in a_ids if s not in b_set]
    only_b = [s for s in b_ids if s not in a_set]
    if only_a:
        problems.append(
            f"subject(s) measured only in {str(baseline_val)!r}: {_short_list(only_a)}"
        )
    if only_b:
        problems.append(
            f"subject(s) measured only in {str(target_val)!r}: {_short_list(only_b)}"
        )
    if not problems and len(a_ids) != len(b_ids):
        problems.append(
            f"the two conditions have different numbers of rows ({len(a_ids)} vs {len(b_ids)})"
        )
    return problems


def _edge_mask_metadata(mask, source_name=None):
    """Facts of a validated tested-family mask; the array itself never enters run_plan.

    Raises ``ValueError`` for anything that is not a non-empty square 2-D array,
    so a malformed stored/uploaded mask is reported instead of raising an
    ``IndexError`` deep in a caption or the run-plan table.
    """
    m = np.asarray(mask)
    if m.ndim != 2 or int(m.shape[0]) != int(m.shape[1]) or int(m.shape[0]) == 0:
        shape = tuple(int(s) for s in m.shape)
        raise ValueError(
            "an edge family mask must be a square 2-D (ROI x ROI) array; "
            f"got shape {shape}"
        )
    m = m.astype(bool)
    return {
        "purpose": "edge_family_mask",
        "sha256": hashlib.sha256(np.ascontiguousarray(m).tobytes()).hexdigest(),
        "shape": [int(m.shape[0]), int(m.shape[1])],
        "n_nodes": int(m.shape[0]),
        "eligible_edges": int(np.triu(m, 1).sum()),
        "source_filename": None if source_name is None else str(source_name),
    }


def _parse_float_list(text):
    """Parse a comma-separated float list; returns ``(values, error)``."""
    parts = [p.strip() for p in str(text or "").replace(";", ",").split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return [], "enter at least one value"
    values = []
    for part in parts:
        try:
            values.append(float(part))
        except (TypeError, ValueError):
            return [], f"{part!r} is not a number"
    return values, None


def _validate_eh_grid(e_values, h_values, e_bounds, h_bounds):
    """Grid validation mirroring the scalar widget ranges and the engine's pairing."""
    errors = []
    if len(e_values) != len(h_values):
        errors.append(
            f"E and H lists must have the same length (E={len(e_values)}, H={len(h_values)})"
        )
    for name, values, (low, high) in (("E", e_values, e_bounds), ("H", h_values, h_bounds)):
        out = [v for v in values if not (low <= v <= high)]
        if out:
            errors.append(f"{name} values must lie within {low:g}-{high:g}; got {_short_list(out)}")
    return errors


def _resolve_primary_cell(result, primary_cell):
    """Resolve the declared primary (E, H) cell to the result's grid index.

    The cell was declared before the run. When the result carries its own
    ``e_grid``/``h_grid`` the declaration must name EXACTLY ONE cell of that
    grid: a cell that is absent, ambiguous (two identical (E, H) cells) or
    contradicted by the recorded index raises ``ValueError`` instead of
    silently reporting a different cell. Without a grid the declaration cannot
    be honoured at all and also raises.
    """
    if not primary_cell:
        return None, None
    record = dict(primary_cell)
    if not bool(getattr(result, "is_grid", False)):
        raise ValueError(
            "a primary (E, H) cell was declared before the run, but the completed "
            "result carries no (E, H) parameter grid; the declared primary cell "
            "cannot be reported"
        )
    e_grid = getattr(result, "e_grid", None)
    h_grid = getattr(result, "h_grid", None)
    if e_grid is None or h_grid is None or len(e_grid) != len(h_grid) or len(e_grid) == 0:
        raise ValueError(
            "the completed result carries an (E, H) grid but not its E/H axis labels; "
            "the declared primary cell cannot be resolved against it"
        )
    e_decl, h_decl = record.get("e"), record.get("h")
    if e_decl is None or h_decl is None:
        raise ValueError(
            "the run plan declares a primary cell without its (E, H) coordinates; "
            "the cell cannot be verified against the completed grid"
        )
    e_decl, h_decl = float(e_decl), float(h_decl)
    matches = [
        k for k in range(len(e_grid))
        if np.isclose(float(e_grid[k]), e_decl) and np.isclose(float(h_grid[k]), h_decl)
    ]
    if not matches:
        raise ValueError(
            f"the declared primary cell E={e_decl:g}, H={h_decl:g} is not a cell of the "
            f"completed (E, H) grid (E={[float(x) for x in e_grid]}, "
            f"H={[float(x) for x in h_grid]}); no cell can be reported as primary"
        )
    if len(matches) > 1:
        raise ValueError(
            f"the declared primary cell E={e_decl:g}, H={h_decl:g} matches {len(matches)} "
            "cells of the completed grid; the declaration is ambiguous"
        )
    idx = matches[0]
    declared_idx = record.get("param_idx")
    if declared_idx is not None:
        try:
            declared_idx = int(declared_idx)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"the recorded primary grid index {record.get('param_idx')!r} is not an integer"
            ) from exc
        if declared_idx != idx:
            raise ValueError(
                f"the run plan records primary grid index {declared_idx}, but the declared "
                f"cell E={e_decl:g}, H={h_decl:g} is index {idx}; the declaration contradicts "
                "itself"
            )
    record["param_idx"] = int(idx)
    record["resolved_from"] = "result_e_h_grid"
    return int(idx), record


def _build_design_arrays(
    Y,
    pheno_df,
    *,
    test_type,
    question_choice,
    group_col=None,
    ref_group=None,
    target_group=None,
    interest_var=None,
    confound_vars=None,
    site_col=None,
    condition_col=None,
    subject_col=None,
    baseline_val=None,
    target_val=None,
    paired_confound_cols=None,
    require_paired_bijection=False,
):
    """Slice connectivity + phenotype exactly as the launch does (no widgets).

    Shared by the launch handler and the pre-launch diagnostic preview so the
    side-car always describes the arrays that will actually be analysed: the
    same NaN filtering, the same group/level selection, condition A/B rows in
    condition A's order and ``group1`` followed by ``group2`` for the
    two-sample path. It does not run or alter any inference.

    Paired contrasts: with ``require_paired_bijection`` (the launch and the
    pre-launch preview always pass it) the two conditions must be a strict
    one-to-one subject match. Condition A (baseline) rows keep phenotype
    order, condition B is matched by subject ID into the same order, and any
    mismatch raises :class:`PairedDesignError` naming the subjects — no
    sorting, no intersection and no first-duplicate pick. Without the flag the
    historical permissive slicing (sorted intersection) is kept for direct
    callers. ``paired_confound_cols`` builds the condition-specific confound
    matrices in the same row order as ``group1``/``group2``; the difference
    ``B - A`` is computed package-side, never here.
    """
    confound_vars = list(confound_vars or [])
    paired_confound_cols = list(paired_confound_cols or [])
    index_before_filter = list(pheno_df.index)
    req_cols = []
    if test_type == "glm":
        if question_choice == "Group Difference":
            req_cols = [group_col] + confound_vars
        else:
            req_cols = [interest_var] + confound_vars
    elif test_type == "paired":
        req_cols = [condition_col, subject_col]
    elif test_type == "two-sample":
        req_cols = [group_col]
    if site_col:
        req_cols.append(site_col)
    req_cols = [c for c in req_cols if c in pheno_df.columns]

    n_dropped = 0
    if req_cols:
        valid_mask = pheno_df[req_cols].notna().all(axis=1)
        n_dropped = int(len(pheno_df) - valid_mask.sum())
        visible_index = [index_before_filter[k] for k, ok in enumerate(valid_mask.to_numpy()) if ok]
        Y = Y[valid_mask.to_numpy()]
        pheno_df = pheno_df.loc[valid_mask].reset_index(drop=True)
    else:
        visible_index = list(pheno_df.index)

    # Condition-specific confound columns: a NaN in either condition's value
    # excludes that subject's pair (both rows) before the run — a pre-run
    # exclusion reported to the user, never a silent half-pair drop.
    n_pairs_dropped_confound = 0
    if test_type == "paired" and paired_confound_cols and subject_col in pheno_df.columns:
        present_cols = [c for c in paired_confound_cols if c in pheno_df.columns]
        if present_cols:
            bad_rows = pheno_df[present_cols].isna().any(axis=1)
            if bool(bad_rows.any()):
                bad_subjects = set(pheno_df.loc[bad_rows, subject_col].tolist())
                keep_mask = ~pheno_df[subject_col].isin(bad_subjects).to_numpy()
                n_pairs_dropped_confound = len(bad_subjects)
                visible_index = [visible_index[k] for k, ok in enumerate(keep_mask) if ok]
                Y = Y[keep_mask]
                pheno_df = pheno_df.loc[keep_mask].reset_index(drop=True)

    interest = None
    confounds = None
    confounds_group1 = None
    confounds_group2 = None
    ids_group1 = None
    ids_group2 = None
    group1 = None
    group2 = None
    sites = list(pheno_df[site_col].values) if site_col else None
    analysis_df = None
    index_values = None
    idx_baseline = None
    idx_target = None
    paired_subjects = None

    if test_type == "glm":
        if question_choice == "Group Difference":
            group_mask = pheno_df[group_col].isin([ref_group, target_group])
            Y = Y[group_mask.to_numpy()]
            pheno_df_filtered = pheno_df.loc[group_mask].reset_index(drop=True)
            if site_col:
                sites = list(pheno_df_filtered[site_col].values)
            interest = (pheno_df_filtered[group_col] == target_group).values.astype(np.float64)
            if confound_vars:
                confounds = pheno_df_filtered[confound_vars].values.astype(np.float64)
            analysis_df = pheno_df_filtered
            index_values = [
                visible_index[k] for k, keep in enumerate(group_mask.to_numpy()) if keep
            ]
        else:
            interest = pheno_df[interest_var].values.astype(np.float64)
            if confound_vars:
                confounds = pheno_df[confound_vars].values.astype(np.float64)
            analysis_df = pheno_df
            index_values = visible_index

    elif test_type == "paired":
        df_baseline = pheno_df[pheno_df[condition_col] == baseline_val]
        df_target = pheno_df[pheno_df[condition_col] == target_val]
        if require_paired_bijection:
            problems = _paired_design_violations(
                pheno_df, condition_col, subject_col, baseline_val, target_val
            )
            if problems:
                raise PairedDesignError(
                    "Paired design must be a one-to-one subject match between the two "
                    "conditions; nothing is dropped, re-sorted or matched by row "
                    "position. " + " | ".join(problems)
                )
            # Condition A keeps phenotype order; condition B follows it by ID.
            idx_baseline = [int(i) for i in df_baseline.index]
            target_by_subject = dict(
                zip(df_target[subject_col].astype(str), df_target.index)
            )
            idx_target = [
                int(target_by_subject[str(s)]) for s in df_baseline[subject_col].astype(str)
            ]
            paired_subjects = list(df_baseline[subject_col])
        else:
            common_subjects = sorted(
                list(set(df_baseline[subject_col]).intersection(set(df_target[subject_col])))
            )
            idx_baseline = [df_baseline[df_baseline[subject_col] == s].index[0] for s in common_subjects]
            idx_target = [df_target[df_target[subject_col] == s].index[0] for s in common_subjects]
            paired_subjects = common_subjects
        group1 = Y[idx_baseline]
        group2 = Y[idx_target]
        if paired_confound_cols:
            # Same row order as group1/group2; the B - A difference stays package-side.
            confounds_group1 = pheno_df.loc[idx_baseline, paired_confound_cols].to_numpy(
                dtype=np.float64
            )
            confounds_group2 = pheno_df.loc[idx_target, paired_confound_cols].to_numpy(
                dtype=np.float64
            )
        ids_group1 = list(paired_subjects)
        # group2 is in condition A's row order, so its alignment IDs are the same.
        ids_group2 = list(paired_subjects)
        if site_col:
            baseline_sites = pheno_df.loc[idx_baseline, site_col].astype(str).tolist()
            target_sites = pheno_df.loc[idx_target, site_col].astype(str).tolist()
            mismatched_sites = [
                subject
                for subject, baseline_site, target_site in zip(
                    common_subjects, baseline_sites, target_sites
                )
                if baseline_site != target_site
            ]
            if mismatched_sites:
                raise ValueError(
                    "Paired site handling requires one stable site per subject. "
                    "Site labels differ between conditions for: "
                    + ", ".join(map(str, mismatched_sites[:5]))
                    + ("..." if len(mismatched_sites) > 5 else "")
                )
            # One site label per paired subject, in the same order as group1/group2.
            sites = baseline_sites

    elif test_type == "two-sample":
        g1_mask = pheno_df[group_col] == ref_group
        g2_mask = pheno_df[group_col] == target_group
        group1 = Y[g1_mask.to_numpy()]
        group2 = Y[g2_mask.to_numpy()]
        analysis_df = pd.concat([pheno_df.loc[g1_mask], pheno_df.loc[g2_mask]]).reset_index(drop=True)
        index_values = (
            [visible_index[k] for k, keep in enumerate(g1_mask.to_numpy()) if keep]
            + [visible_index[k] for k, keep in enumerate(g2_mask.to_numpy()) if keep]
        )
        if site_col:
            sites = (
                list(pheno_df.loc[g1_mask, site_col].values)
                + list(pheno_df.loc[g2_mask, site_col].values)
            )

    return {
        "Y": Y,
        "pheno_filtered": pheno_df,
        "interest": interest,
        "confounds": confounds,
        "group1": group1,
        "group2": group2,
        "sites": sites,
        "analysis_df": analysis_df,
        "index_values": index_values,
        "n_dropped": n_dropped,
        "idx_baseline": idx_baseline,
        "idx_target": idx_target,
        "paired_subjects": paired_subjects,
        "ids_group1": ids_group1,
        "ids_group2": ids_group2,
        "confounds_group1": confounds_group1,
        "confounds_group2": confounds_group2,
        "condition_confound_cols": list(paired_confound_cols) if confounds_group1 is not None else [],
        "n_pairs_dropped_confound": n_pairs_dropped_confound,
    }


def _motion_qc_spec_from_arrays(
    arrays,
    *,
    test_type,
    question_choice,
    fd_col,
    subject_col,
    group_col=None,
    ref_group=None,
    target_group=None,
    baseline_val=None,
    target_val=None,
):
    """The descriptive side-car spec for the exact arrays the analysis uses."""
    pheno_filtered = arrays["pheno_filtered"]
    spec = {
        "pheno": pheno_filtered,
        "fd_units": "mm",
        "pool": True,
        "fd_source_name": fd_col,
    }
    if test_type == "paired":
        if fd_col is not None:
            fd_all = _motion_qc_fd_values(pheno_filtered, fd_col)
            spec["fd_group1"] = fd_all[np.asarray(arrays["idx_baseline"], dtype=int)]
            spec["fd_group2"] = fd_all[np.asarray(arrays["idx_target"], dtype=int)]
        spec["subject_ids"] = list(arrays["paired_subjects"])
        spec["condition_a"] = str(baseline_val)
        spec["condition_b"] = str(target_val)
        return spec

    frame = arrays["analysis_df"]
    if fd_col is not None:
        spec["fd"] = _motion_qc_fd_values(frame, fd_col)
    ids = _motion_qc_subject_ids(frame, subject_col)
    if ids is not None:
        spec["subject_ids"] = ids
    if arrays["index_values"] is not None:
        spec["index_values"] = list(arrays["index_values"])
    if test_type == "two-sample" and group_col is not None:
        # group1 rows then group2 rows, exactly the concatenation order of the
        # analysed arrays; the labels are not inferred from the FD vector.
        spec["condition_labels"] = (
            [str(ref_group)] * int(arrays["group1"].shape[0])
            + [str(target_group)] * int(arrays["group2"].shape[0])
        )
    elif test_type == "glm" and question_choice == "Group Difference" and group_col in frame.columns:
        spec["condition_labels"] = [str(v) for v in frame[group_col].to_numpy()]
    return spec


def _motion_qc_preview_key(*parts):
    """Stable, comparable signature for the cached pre-launch preview."""
    return tuple("" if p is None else str(p) for p in parts)


def _run_inference_background(
    task,
    Y, interest, confounds, group1, group2, test_type, sites, harmonization_choice,
    operator_choice, n_perms, seed_val, use_mp, acceleration_choice, op_kwargs,
    run_sensitivity, comp_method, comp_kwargs, sc_net_labels, use_sc_prior, atlas,
    annotation_atlas, alpha, question_choice, loaded_settings_hash, recipe_choice,
    effective_recipe_choice, group_col, ref_group, target_group, interest_var,
    subject_col, condition_col, baseline_val, target_val, confound_vars, site_col,
    active_atlas_signature, data_kind, tested_tail=None, var_equal=None,
    preprocessing=None, fisher_history=None, roi_indices=None, motion_qc=None,
    ids_group1=None, ids_group2=None, confounds_group1=None, confounds_group2=None,
    missing_policy="raise", min_coverage=0.8, edge_mask_plan=None,
    primary_cell=None, design_facts=None, n_processes=None,
):
    """Worker thread body: run the inference and record status.

    The data are analysed on the scale they were loaded with: raw
    correlations stay correlations (``r_to_z=False``); Fisher z is applied
    only by the explicit ingestion option or the Design-step button.
    ``tested_tail`` (``'positive'`` / ``'negative'``) is the prespecified
    one-sided hypothesis chosen in the design step: it is declared to
    ``analyze()`` *before* the permutation run, which uses per-tail nulls
    and marks the restriction ``prespecified`` in provenance; the other
    direction is reported as not tested. ``None`` is the two-sided default
    (one joint family; the single alpha covers both directions).
    ``var_equal`` (two-sample only) chooses Student's (True) or Welch's
    (False) t in the two-sample permutation test.

    ``roi_indices`` is the ROI subset the caller selected before subsetting
    the arrays; it is recorded as a launch fact (metadata only). The run
    plan captures the actual launch arguments of this execution — started
    time, resolved acceleration, multiprocessing, the operator kwargs
    actually passed (including the network-label vector used by
    network-informed methods) and the enhancement backend — so exports can
    report them without rerunning anything.

    ``ids_group1``/``ids_group2`` and ``confounds_group1``/``confounds_group2``
    are the paired design's subject IDs (condition A order) and its
    condition-specific confound rows; the ``B - A`` difference is computed by
    ``analyze()``, never here. ``missing_policy``/``min_coverage`` are the
    executed missing-connectivity policy. ``edge_mask_plan`` is the metadata
    of the validated tested-family mask (the array travels in ``op_kwargs``).
    ``primary_cell`` is the (E, H) cell declared before the run; on a grid
    result the completion table is the declared cell, not cell 0.
    """
    from datetime import datetime, timezone

    # Launch facts with defined fallbacks: the engine default is complete-case.
    missing_policy = "raise" if missing_policy in (None, "") else str(missing_policy)
    min_coverage = 0.8 if min_coverage is None else float(min_coverage)

    started_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Local import so the worker is self-contained when loaded through the
    # AST helper used by the app tests (module-level names are not visible).
    from conninfpy import InferenceCancelled, ProgressSession

    class _StaleJob(Exception):
        """This job was superseded; its payload must never be published."""

    generation = getattr(task, "generation", 0)
    # Parent-side only: the callback records progress on the task object and
    # the cancel predicate is evaluated here, never pickled into a worker.
    # Both are looked up defensively so a bare task-like object (tests, older
    # restored state) still runs without progress or cancellation.
    _progress_cb = getattr(task, "_on_progress", None)
    _cancel_event = getattr(task, "cancel_event", None)

    def _cancel_requested():
        return _cancel_event is not None and _cancel_event.is_set()

    def _resource_plan_facts(use_mp_flag, workers_requested, n_planned):
        """Requested vs engine-resolved worker facts (honest unknowns)."""
        try:
            from conninfpy.stats._workers import describe_worker_policy

            return describe_worker_policy(
                use_mp=bool(use_mp_flag),
                n_processes=workers_requested,
                n_permutations=n_planned,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return {"request_error": f"{type(exc).__name__}: {exc}"}

    def _stale() -> bool:
        """True when this worker no longer owns the task state."""
        return getattr(task, "generation", generation) != generation

    def _write(attr, value):
        """Set a task attribute only for the generation captured at launch.

        Every mutation of task state goes through here (or through an explicit
        ``_stale()`` check) so a superseded worker can never overwrite the
        newer generation's status, result, run plan or progress.
        """
        if not _stale():
            setattr(task, attr, value)

    def _record_progress(done, total):
        if _stale():
            return  # a superseded loop reports nothing to the new generation
        if _progress_cb is not None:
            _progress_cb(done, total)
        stage = session.stage
        if stage is not None:
            _write("progress_stage", stage)

    session = ProgressSession(
        callback=_record_progress if _progress_cb is not None else None,
        cancel=_cancel_requested,
    )

    def _ensure_current():
        """Stop before publishing anything if this job was cancelled/superseded."""
        if _cancel_requested():
            raise InferenceCancelled("cancelled at a permutation chunk boundary")
        if getattr(task, "generation", generation) != generation:
            raise _StaleJob("superseded by a newer inference run")

    try:
        input_kind = {"correlation": "r", "fisher_z": "z"}.get(data_kind)
        # Only the two-sample t-test has the choice; analyze() rejects it elsewhere.
        variance_kwargs = (
            {"var_equal": bool(var_equal)}
            if test_type == "two-sample" and var_equal is not None
            else {}
        )
        t_statistic = (
            ("student" if var_equal else "welch") if variance_kwargs
            else ("glm" if test_type == "glm" else None)
        )
        _write(
            "progress_message",
            f"Executing primary method ({operator_choice.upper()}) permutation loops...",
        )
        from conninfpy import analyze
        from conninfpy.defaults import (
            DEFAULT_ACCELERATION,
            DEFAULT_EXTENT_EXPONENT,
            DEFAULT_HEIGHT_EXPONENT,
            DEFAULT_MIN_CLUSTER_SIZE,
            DEFAULT_NBS_STAT,
            DEFAULT_NBS_THRESHOLD,
            DEFAULT_N_PERMUTATIONS,
            DEFAULT_N_THRESHOLDS_PERMUTATION,
            DEFAULT_START_THRESHOLD,
        )
        effective_n_perms = DEFAULT_N_PERMUTATIONS if n_perms is None else n_perms
        if run_sensitivity and comp_method and effective_n_perms:
            # Announce the second permutation loop before the first one runs,
            # so the reported total is exact from the first callback.
            session.expect(
                int(effective_n_perms),
                f"Sensitivity companion ({str(comp_method).upper()}) planned",
            )
        effective_acceleration = (
            DEFAULT_ACCELERATION
            if acceleration_choice in (None, "none")
            else acceleration_choice
        )
        primary_run_kwargs = dict(op_kwargs or {})
        if operator_choice in ("tfnbs", "ni_tfnbs", "fbc_tfnbs"):
            primary_run_kwargs.setdefault("e", DEFAULT_EXTENT_EXPONENT)
            primary_run_kwargs.setdefault("h", DEFAULT_HEIGHT_EXPONENT)
            primary_run_kwargs.setdefault("n", DEFAULT_N_THRESHOLDS_PERMUTATION)
            primary_run_kwargs.setdefault("start_thres", DEFAULT_START_THRESHOLD)
            if operator_choice == "fbc_tfnbs":
                primary_run_kwargs.setdefault("min_cluster_size", DEFAULT_MIN_CLUSTER_SIZE)
        elif operator_choice == "nbs":
            primary_run_kwargs.setdefault("threshold", DEFAULT_NBS_THRESHOLD)
            primary_run_kwargs.setdefault("nbs_stat", DEFAULT_NBS_STAT)
        if operator_choice in ("ni_tfnbs", "fbc_tfnbs", "cnbs") and "net_labels" not in primary_run_kwargs:
            if use_sc_prior and sc_net_labels is not None:
                primary_run_kwargs["net_labels"] = sc_net_labels
            elif atlas is not None:
                primary_run_kwargs["net_labels"] = atlas.network_index()
        # The actual network labels used by this run, captured from the
        # arguments that are about to be passed (never recomputed at export
        # time and never replaced by the current UI selection).
        used_net_labels = primary_run_kwargs.get("net_labels")
        net_labels_capture = None
        if used_net_labels is not None:
            label_values = used_net_labels.tolist() if hasattr(used_net_labels, "tolist") else used_net_labels
            net_labels_capture = {
                "source": "sc_prior" if (use_sc_prior and sc_net_labels is not None) else "atlas_networks",
                "labels": [int(x) for x in label_values],
            }
        res = analyze(
            Y=Y if test_type == "glm" else None,
            interest=interest,
            confounds=confounds,
            group1=group1,
            group2=group2,
            test_type=test_type,
            sites=sites,
            harmonize=harmonization_choice,
            input=input_kind,
            atlas=atlas,
            r_to_z=False,
            method=operator_choice,
            n_permutations=effective_n_perms,
            rng=seed_val,
            verbose=False,
            use_mp=use_mp,
            tested_tail=tested_tail,
            acceleration=effective_acceleration,
            preprocessing=preprocessing,
            fisher_history=fisher_history if input_kind == "z" else None,
            motion_qc=motion_qc,
            ids_group1=ids_group1,
            ids_group2=ids_group2,
            confounds_group1=confounds_group1,
            confounds_group2=confounds_group2,
            missing=missing_policy,
            min_coverage=min_coverage,
            progress=session,
            cancel=_cancel_requested,
            n_processes=n_processes,
            **variance_kwargs,
            **primary_run_kwargs
        )
        _ensure_current()
        # ``analyze`` returns an AnalyzeResult wrapper; the grid facts and the
        # p-value maps live on the wrapped inference result.
        primary_inner = getattr(res, "inference", res)
        primary_idx, primary_record = (
            _resolve_primary_cell(primary_inner, primary_cell) if primary_cell else (None, None)
        )
        if getattr(primary_inner, "is_grid", False) and primary_idx is None:
            raise ValueError(
                "the analysis returned an (E, H) grid but no primary cell was declared "
                "before the run; a grid cell must be declared to report a completion table"
            )
        edges = res.significant_edges(
            atlas=annotation_atlas,
            alpha=alpha,
            **({"param_idx": primary_idx}
               if getattr(primary_inner, "is_grid", False) else {}),
        )
        
        comp_res = None
        comp_edges = None
        if run_sensitivity and comp_method:
            _write(
                "progress_message",
                f"Executing sensitivity companion ({comp_method.upper()}) permutation loops...",
            )
            comp_run_kwargs = dict(comp_kwargs or {})
            if comp_method in ("tfnbs", "ni_tfnbs", "fbc_tfnbs"):
                comp_run_kwargs.setdefault("e", DEFAULT_EXTENT_EXPONENT)
                comp_run_kwargs.setdefault("h", DEFAULT_HEIGHT_EXPONENT)
                comp_run_kwargs.setdefault("n", DEFAULT_N_THRESHOLDS_PERMUTATION)
                comp_run_kwargs.setdefault("start_thres", DEFAULT_START_THRESHOLD)
                if comp_method == "fbc_tfnbs":
                    comp_run_kwargs.setdefault("min_cluster_size", DEFAULT_MIN_CLUSTER_SIZE)
                if comp_method in ("ni_tfnbs", "fbc_tfnbs") and "net_labels" not in comp_run_kwargs:
                    if use_sc_prior and sc_net_labels is not None:
                        comp_run_kwargs["net_labels"] = sc_net_labels
                    elif atlas is not None:
                        comp_run_kwargs["net_labels"] = atlas.network_index()
            elif comp_method == "cnbs" and "net_labels" not in comp_run_kwargs:
                if use_sc_prior and sc_net_labels is not None:
                    comp_run_kwargs["net_labels"] = sc_net_labels
                elif atlas is not None:
                    comp_run_kwargs["net_labels"] = atlas.network_index()
            elif comp_method == "nbs":
                comp_run_kwargs.setdefault("threshold", DEFAULT_NBS_THRESHOLD)
                comp_run_kwargs.setdefault("nbs_stat", DEFAULT_NBS_STAT)
                
            comp_res = analyze(
                Y=Y if test_type == "glm" else None,
                interest=interest,
                confounds=confounds,
                group1=group1,
                group2=group2,
                test_type=test_type,
                sites=sites,
                harmonize=harmonization_choice,
                input=input_kind,
                atlas=atlas,
                r_to_z=False,
                method=comp_method,
                n_permutations=effective_n_perms,
                rng=seed_val,
                verbose=False,
                use_mp=use_mp,
                tested_tail=tested_tail,
                acceleration=effective_acceleration,
                preprocessing=preprocessing,
                fisher_history=fisher_history if input_kind == "z" else None,
                progress=session,
                cancel=_cancel_requested,
                n_processes=n_processes,
                **variance_kwargs,
                **comp_run_kwargs
            )
            _ensure_current()
            comp_edges = comp_res.significant_edges(
                atlas=annotation_atlas,
                alpha=alpha,
                **({"param_idx": 0}
                   if getattr(getattr(comp_res, "inference", comp_res), "is_grid", False)
                   else {}),
            )
            
        session.finalize()
        _ensure_current()
        _write("result", res)
        _write("edges", edges)
        _write("comp_result", comp_res)
        _write("comp_edges", comp_edges)
        
        run_plan = {
            "question_type": question_choice,
            "alpha": alpha,
            "loaded_settings_hash": loaded_settings_hash,
            "design_family": test_type,
            "method": operator_choice,
            "n_permutations": effective_n_perms,
            "site_recipe": recipe_choice,
            "effective_site_recipe": effective_recipe_choice,
            "harmonize": harmonization_choice,
            "group_col": group_col,
            "reference_group": ref_group,
            "target_group": target_group,
            "interest_var": interest_var,
            "subject_col": subject_col,
            "condition_col": condition_col,
            "baseline_condition": baseline_val,
            "target_condition": target_val,
            "confound_vars": confound_vars,
            "site_col": site_col,
            "active_atlas_signature": active_atlas_signature,
            "data_kind": data_kind,
            "input": input_kind,
            # The scale the inference ran on: the loaded one, never converted here.
            "analysis_scale": {"r": "correlation", "z": "fisher_z"}.get(input_kind),
            "preprocessing": preprocessing,
            # NETSTAT-28: descriptive side-car opted in before this run (the
            # computed diagnostic itself rides on the result, not here).
            "motion_qc_enabled": motion_qc is not None,
            # Upstream Fisher transforms captured at load/button time (the
            # full record — applied here + upstream — lives on the result and
            # its provenance/exports).
            "fisher_history": fisher_history,
            "fisher": getattr(res, "fisher", None),
            "tail_policy": f"one-sided-{tested_tail}" if tested_tail else "two-sided",
            "tested_tail": tested_tail,
            "tail_selection": "prespecified" if tested_tail else None,
            "two_tailed": tested_tail is None,
            # 'student' / 'welch' (two-sample test), 'glm' (pooled residual variance).
            "t_statistic": t_statistic,
            "seed": seed_val,
            "method_parameters": _method_plan_parameters(operator_choice, primary_run_kwargs),
            "companion_method": comp_method if run_sensitivity else None,
            "companion_method_parameters": (
                _method_plan_parameters(comp_method, comp_run_kwargs)
                if run_sensitivity and comp_method else {}
            ),
            # ---- actual launch facts of this execution (metadata only) ----
            "started_at": started_at_utc,
            "wall_time_s": getattr(getattr(res, "inference", res), "wall_time_s", None),
            "use_mp": bool(use_mp),
            # NETSTAT-27 resource facts: requested vs engine-resolved workers.
            # The BLAS library and which enhancement backend ran are not
            # measured, so they stay explicitly unknown.
            "n_processes_requested": n_processes,
            "resource_plan": _resource_plan_facts(use_mp, n_processes, effective_n_perms),
            "progress_final": session.snapshot(),
            "generation": generation,
            "acceleration": effective_acceleration,
            "var_equal": var_equal,
            # The executed missing-connectivity policy and its coverage floor;
            # missing design fields are excluded earlier and never appear here.
            "missing_policy": missing_policy,
            "min_coverage": (
                float(min_coverage) if missing_policy == "available_case" else None
            ),
            # Tested-family mask metadata of this run (never the array).
            "edge_mask": edge_mask_plan,
            # Declared-before-the-run primary (E, H) cell of a sensitivity grid.
            "primary_cell": primary_record,
            "r_to_z": False,
            "roi_indices": list(roi_indices) if roi_indices is not None else None,
            "net_labels": net_labels_capture,
            "operator_kwargs": {
                key: (value.tolist() if hasattr(value, "tolist") else value)
                for key, value in primary_run_kwargs.items()
                if key != "edge_mask"  # recorded as metadata (sha256), not as an array
            },
            # Installation fact only: whether the numba module was present in
            # the run environment. Which enhancement backend actually
            # executed is not recorded by the kernels (no runtime dispatcher
            # metadata exists), so it is never guessed: methods without
            # enhancement kernels are marked not-applicable, everything else
            # stays unknown until a genuine runtime record exists.
            "numba_module_found": (
                __import__("importlib.util", fromlist=["util"]).find_spec("numba") is not None
            ),
            "enhancement_backend": (
                "not-applicable"
                if str(operator_choice).lower() in ("tstat", "bonferroni", "bh_fdr", "bh_fdr_perm", "nbs")
                else None
            ),
            "package_version": __import__("conninfpy").__version__,
            "python_version": __import__("platform").python_version(),
        }
        if isinstance(design_facts, dict):
            # Launch facts of the sliced design (paired alignment, pre-run
            # exclusions); never inferred later from the current widgets.
            run_plan.update(
                {str(k): v for k, v in design_facts.items() if str(k) not in run_plan}
            )
        if primary_record is not None:
            run_plan["primary_cell"] = primary_record
        _write("run_plan", run_plan)
        _write("finished_generation", generation)
        _write("status", "success")
        _write("progress_message", "✅ Completed!")
    except (InferenceCancelled, _StaleJob) as exc:
        # Nothing computed from an incomplete null distribution may survive.
        session.finalize()
        if _stale():
            # Superseded: a newer generation owns this task now. Drop the
            # payload silently instead of writing over its state.
            return
        for attr in ("result", "edges", "comp_result", "comp_edges", "run_plan"):
            _write(attr, None)
        _write("error", None)
        _write("finished_generation", generation)
        _write("status", "cancelled")
        if not getattr(task, "cancel_reason", None):
            _write(
                "cancel_reason",
                "superseded by a newer run"
                if isinstance(exc, _StaleJob)
                else "cancelled by the user",
            )
        _write(
            "progress_message",
            "⏹ Cancelled at a permutation chunk boundary — no p-values, "
            "no result and no export were produced.",
        )
    except Exception as e:
        if _stale():
            return  # never write an old failure into a newer generation
        _write("error", f"{e}\n{traceback.format_exc()}")
        _write("status", "failed")
        _write("progress_message", "❌ Failed")


def render_design_inference_view(base_atlas, tabs_list):
    """Render the design & inference step (model spec, run, live status)."""
    col_t, col_h = st.columns([0.8, 0.2])
    with col_t:
        st.markdown("### Design & Inference")
    with col_h:
        render_help("analysis_setup")
    
    if st.session_state.connectivity_data is None or st.session_state.pheno_df is None:
        if st.session_state.get("inference_result") is not None:
            from apps.utils.workspace import render_restored_workspace_notice

            render_restored_workspace_notice()
            st.info(
                "This restored workspace contains finished results without subject-level data. "
                "Load the dataset in Tab 1 to configure or re-run inference."
            )
        else:
            st.info("Load a dataset to configure your analysis.")
        go_to_step("Load data", DATA_TAB, key="design_load_data")
    else:
        with st.container(key="design_actions"):
            run_action, results_action = st.columns(2)
            status_area = st.container()
        # 1. Advisory Classifier Heuristic
        recomm = "Group Difference"
        pheno_df = st.session_state.pheno_df
        cols = list(pheno_df.columns)
        
        # Check for repeated subjects
        subj_repeats = False
        for c in cols:
            if c.lower() in ("subject", "subject_id", "sub", "id", "subj"):
                if pheno_df[c].duplicated().any():
                    subj_repeats = True
                    break
        
        # Check for condition columns
        cond_col_candidates = [c for c in cols if c.lower() in ("condition", "state", "task", "session")]
        if subj_repeats and cond_col_candidates:
            recomm = "Paired Condition"
        else:
            group_col_candidates = [c for c in cols if c.lower() in ("group", "dx", "diagnosis", "class", "label")]
            if group_col_candidates:
                recomm = "Group Difference"
            else:
                numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(pheno_df[c])]
                if len(numeric_cols) > 0:
                    recomm = "Continuous Predictor"
                    
        # 2. Reorganized Layout (55% Left, 45% Right)
        col_left, col_right = st.columns([0.55, 0.45])
        
        # Define bindings and settings variables in outer scope
        subject_col = None
        condition_col = None
        baseline_val = None
        target_val = None
        group_col = None
        ref_group = None
        target_group = None
        adjust_covariates = False
        interest_var = None
        confound_vars = []
        site_col = None
        tested_tail = None
        var_equal = None  # two-sample only: True Student, False Welch
        recipe_choice = "No site handling"
        harmonization_choice = "none"
        effective_recipe_choice = "No site handling"
        site_strategy_note = None
        operator_choice = "tfnbs"
        n_perms = DEFAULT_N_PERMUTATIONS
        seed_val = 42
        use_mp = True
        n_processes = None
        acceleration_choice = "none" if DEFAULT_ACCELERATION is None else DEFAULT_ACCELERATION
        op_kwargs = {}
        comp_kwargs = {}
        paired_confound_cols = []
        missing_policy = "raise"
        min_coverage = 0.8
        grid_e_values = None
        grid_h_values = None
        primary_cell_decl = None
        edge_mask_plan = None
        edge_mask_plan_error = None
        edge_mask_array = None
        
        with col_left:
            st.markdown("#### 🔍 1. Research Question & Design")
            
            is_synthetic = st.session_state.get("_synthetic_scenario_name") is not None
            
            is_abide = False
            if st.session_state.pheno_df is not None:
                cols_set = set(st.session_state.pheno_df.columns)
                if "Diagnosis" in cols_set and "Age" in cols_set and "Motion_FD" in cols_set and "ADOS_TOTAL" in cols_set:
                    is_abide = True
            
            if is_synthetic:
                cols = list(st.session_state.pheno_df.columns)
                if "group" in cols:
                    valid_choices = ["Group Difference"]
                else:
                    valid_choices = ["Continuous Predictor"]
                
                default_idx = valid_choices.index(recomm) if recomm in valid_choices else 0
            elif is_abide:
                valid_choices = ["Group Difference", "Continuous Predictor"]
                default_idx = valid_choices.index(recomm) if recomm in valid_choices else 0
            else:
                valid_choices = ["Group Difference", "Paired Condition", "Continuous Predictor"]
                default_idx = valid_choices.index(recomm) if recomm in valid_choices else 0
                
            question_choice = st.selectbox(
                "Select your analysis question:",
                valid_choices,
                format_func=lambda x: f"{x} (💡 Recommended)" if x == recomm else x,
                index=default_idx,
                key=persistent_key('design:select_your_analysis_question', scope='design', options=valid_choices),
            )
            
            if question_choice == "Paired Condition":
                st.markdown("##### Design Binding (Paired Condition)")
                sub_candidates = [c for c in cols if c.lower() in ("subject", "subject_id", "sub", "id", "subj")]
                cond_candidates = [c for c in cols if c.lower() in ("condition", "state", "task", "session")]
                
                subject_col = st.selectbox(
                    "Subject ID Column (Required)",
                    cols,
                    index=cols.index(sub_candidates[0]) if sub_candidates else 0,
                    key=persistent_key('design:subject_id_column_required', scope='design', options=cols),
                )
                condition_col = st.selectbox(
                    "Condition Column (Required)",
                    cols,
                    index=cols.index(cond_candidates[0]) if cond_candidates else 0,
                    key=persistent_key('design:condition_column_required', scope='design', options=cols),
                )
                
                unique_conds = list(pheno_df[condition_col].unique())
                baseline_val = st.selectbox(
                    "Baseline Condition",
                    unique_conds,
                    index=0,
                    key=persistent_key('design:baseline_condition', scope='design', options=unique_conds),
                )
                target_val = st.selectbox(
                    "Target Condition",
                    unique_conds,
                    index=1 if len(unique_conds) > 1 else 0,
                    key=persistent_key('design:target_condition', scope='design', options=unique_conds),
                )

                if not is_synthetic:
                    site_candidates = [
                        column for column in cols
                        if column not in (subject_col, condition_col)
                    ]
                    site_options = [None] + site_candidates
                    default_site = next(
                        (column for column in site_candidates if str(column).lower() == "site"),
                        None,
                    )
                    site_col = st.selectbox(
                        "Acquisition Site Column (Optional)",
                        site_options,
                        index=site_options.index(default_site) if default_site else 0,
                        key=persistent_key('design:acquisition_site_column_optional', scope='design', options=site_options),
                    )
                
                paired_confound_candidates = [
                    column for column in cols
                    if column not in (subject_col, condition_col, site_col)
                    and pd.api.types.is_numeric_dtype(pheno_df[column])
                ]
                paired_confound_cols = sorted(st.multiselect(
                    "Condition-specific confounds (value may differ between conditions)",
                    paired_confound_candidates,
                    default=[],
                    key=persistent_key(
                        'design:paired_condition_specific_confounds',
                        scope='design', options=paired_confound_candidates, multiple=True,
                    ),
                    help=(
                        "Columns measured once per condition (e.g. condition-level mean "
                        "motion). The within-subject difference is tested while "
                        "partialling out that subject's B - A confound difference. The "
                        "difference is computed inside the analysis engine, never here, "
                        "and a subject-constant shared table is never substituted for it."
                    ),
                ))
                if paired_confound_cols:
                    st.caption(
                        "B - A of the selected columns is computed by the engine. Both "
                        "condition rows of a subject must carry a value; a missing value "
                        "excludes that subject's pair before the run with a warning."
                    )

                st.info(f"**Contrast Direction:**\n• **Positive effect:** {target_val} > {baseline_val}\n• **Negative effect:** {baseline_val} > {target_val}\n\n*Uses within-subject sign-flip permutation (stratum-invariant by construction). Subject-constant additive site effects cancel in differences, though arbitrary site × condition interactions are not controlled.*")
                test_type = "paired"
                
            elif question_choice == "Group Difference":
                st.markdown("##### Design Binding (Group Difference)")
                group_candidates = [c for c in cols if c.lower() in ("group", "dx", "diagnosis", "class", "label")]
                group_col = st.selectbox(
                    "Group Column",
                    cols,
                    index=cols.index(group_candidates[0]) if group_candidates else 0,
                    key=persistent_key('design:group_column', scope='design', options=cols),
                )
                if is_synthetic and group_col == "group":
                    st.caption("Synthetic data: generator Group 1 = 0; generator Group 2 = 1. The signal, if nonzero, was introduced in Group 2 relative to Group 1.")
                unique_groups = list(pheno_df[group_col].unique())
                ref_group = st.selectbox(
                    "Reference Group (Baseline)",
                    unique_groups,
                    index=0,
                    key=persistent_key('design:reference_group_baseline', scope='design', options=unique_groups),
                )
                target_group = st.selectbox(
                    "Target Group (Contrast)",
                    unique_groups,
                    index=1 if len(unique_groups) > 1 else 0,
                    key=persistent_key('design:target_group_contrast', scope='design', options=unique_groups),
                )
                
                other_cols = [c for c in cols if c not in (group_col, "subject_id", "subject", "sub_id")]
                
                if not is_synthetic:
                    adjust_covariates = st.checkbox("Adjust for nuisance covariates (e.g. age, sex, motion) or scanner site differences", value=is_abide, key=persistent_key('design:adjust_for_nuisance_covariates_e_g_age_sex_motion_or_scanner_site_diff', scope='design'))
                    if adjust_covariates:
                        confound_candidates = _numeric_covariate_candidates(pheno_df, exclude=[group_col])
                        default_confound_vars = _default_abide_confound_vars(confound_candidates) if is_abide else []
                        confound_vars = st.multiselect(
                            "Nuisance Covariates",
                            confound_candidates,
                            default=default_confound_vars,
                            key=persistent_key('design:nuisance_covariates_2', scope='design', options=confound_candidates, multiple=True),
                        )
                        site_candidates = [c for c in cols if c not in [group_col] + confound_vars]
                        site_options = [None] + site_candidates
                        default_site_col = next((c for c in site_candidates if str(c).lower() == "site"), None)
                        site_default = site_options.index(default_site_col) if default_site_col is not None else 0
                        site_col = st.selectbox(
                            "Acquisition Site Column (Optional)",
                            site_options,
                            index=site_default,
                            key=persistent_key('design:acquisition_site_column_optional_2', scope='design', options=site_options),
                        )
                        test_type = "glm"
                    else:
                        site_col = st.selectbox("Acquisition Site Column (Optional)", [None] + [c for c in cols if c != group_col], key=persistent_key('design:acquisition_site_column_optional_3', scope='design', options=[None] + [c for c in cols if c != group_col]))
                        test_type = "two-sample"
                else:
                    if other_cols:
                        confound_vars = st.multiselect("Nuisance Covariates", other_cols, default=other_cols, key=persistent_key('design:nuisance_covariates_3', scope='design', options=other_cols, multiple=True))
                        test_type = "glm"
                    else:
                        test_type = "two-sample"
                    site_col = None

                if test_type == "two-sample":
                    variance_choice = st.radio(
                        "Group variances",
                        ["Equal: Student's t (pooled variance)", "Unequal: Welch's t"],
                        index=0,
                        horizontal=True,
                        help=(
                            "Student's t pools the two group variances (as in the MATLAB NBS "
                            "toolbox); Welch's t keeps them separate. With equal group sizes "
                            "they coincide. The permutation is the same group-label shuffle "
                            "in both cases; it is exact when the groups are exchangeable under "
                            "the null. With unequal sizes and variances neither is exact."
                        ),
                        key=persistent_key('design:group_variances', scope='design'),
                    )
                    var_equal = variance_choice.startswith("Equal")
                else:
                    st.caption(
                        "With covariates the group effect is tested in the GLM, which uses a "
                        "pooled (Student) residual variance."
                    )

            elif question_choice == "Continuous Predictor":
                st.markdown("##### Design Binding (Continuous Predictor)")
                interest_candidates = _numeric_covariate_candidates(pheno_df)
                if not interest_candidates:
                    st.warning("Continuous Predictor requires at least one numeric phenotype column.")
                    return
                interest_var = st.selectbox(
                    "Predictor of Interest",
                    interest_candidates,
                    index=0 if interest_candidates else 0,
                    key=persistent_key('design:predictor_of_interest', scope='design', options=interest_candidates),
                )
                confound_candidates = _numeric_covariate_candidates(
                    pheno_df,
                    exclude=[interest_var],
                )
                confound_vars = st.multiselect("Nuisance Covariates", confound_candidates, key=persistent_key('design:nuisance_covariates', scope='design', options=confound_candidates, multiple=True))
                
                if not is_synthetic:
                    site_col = st.selectbox("Acquisition Site Column (Optional)", [None] + [c for c in cols if c not in [interest_var] + confound_vars and c.lower() not in ("subject", "subject_id", "sub", "id", "subj", "session")], key=persistent_key('design:acquisition_site_column_optional_4', scope='design', options=[None] + [c for c in cols if c not in [interest_var] + confound_vars and c.lower() not in ('subject', 'subject_id', 'sub', 'id', 'subj', 'session')]))
                else:
                    site_col = None
                test_type = "glm"

            # Hypothesis direction: two-sided by default; a one-sided test
            # must be prespecified (e.g. controls > patients in the NBS example).
            direction_labels_now = effect_direction_labels({
                "question_type": question_choice,
                "target_group": target_group, "reference_group": ref_group,
                "target_condition": target_val, "baseline_condition": baseline_val,
                "interest_var": interest_var,
            })
            tail_options = {
                "Two-sided (either direction)": None,
                f"One-sided: {direction_labels_now['positive']}": "positive",
                f"One-sided: {direction_labels_now['negative']}": "negative",
            }
            tail_choice = st.radio(
                "Hypothesis direction",
                list(tail_options),
                index=0,
                help=(
                    "Two-sided tests both directions against one joint null. Choose a "
                    "one-sided test only for a direction fixed before seeing the data: "
                    "that direction gets its own null at the nominal alpha and the other "
                    "direction is not tested (shown as no detections)."
                ),
                key=persistent_key('design:hypothesis_direction', scope='design', options=list(tail_options)),
            )
            tested_tail = tail_options[tail_choice]

            is_synthetic = st.session_state.get("_synthetic_scenario_name") is not None

            if not is_synthetic:
                st.markdown("#### 🌐 2. Site & Exchangeability Handling")
                if question_choice == "Paired Condition":
                    if site_col is not None:
                        recipe_choice = "Paired within-subject + site blocks"
                        effective_recipe_choice = recipe_choice
                        site_strategy_note = (
                            "Site labels are verified to be constant within each subject and recorded in provenance. "
                            "Within-subject sign-flip permutations are stratum-invariant by construction (unlike "
                            "between-subject paired-interest GLMs, where strata permutes subjects within site). "
                            "Subject-constant additive site effects cancel in differences, though arbitrary "
                            "site × condition interactions are not automatically controlled."
                        )
                        st.write(
                            "• **Strategy Selected:** `Paired within-subject + site blocks` "
                            "(site is verified per subject and recorded for provenance; sign-flip is stratum-invariant and additive site effects cancel)"
                        )
                    else:
                        recipe_choice = "Paired within-subject"
                        effective_recipe_choice = recipe_choice
                        st.write("• **Strategy Selected:** `Paired within-subject` (Subject-constant site effects cancel in differences; no ComBat needed)")
                    harmonization_choice = None
                elif site_col is None:
                    recipe_choice = "No site handling"
                    st.write("• **Strategy Selected:** `No site handling` (Single-site or pre-harmonized dataset)")
                    harmonization_choice = None
                else:
                    recipe_choice = st.selectbox(
                        "Select site handling recipe:",
                        ["ComBat + site-aware GLM", "Site-aware GLM", "ComBat-only", "No site handling"],
                        index=0,
                        key=persistent_key('design:select_site_handling_recipe', scope='design'),
                    )
                    if recipe_choice == "ComBat + site-aware GLM":
                        harmonization_choice = "combat_site_dummies_glm"
                    elif recipe_choice == "Site-aware GLM":
                        harmonization_choice = "site_dummies_glm"
                    elif recipe_choice == "ComBat-only":
                        harmonization_choice = "combat_only"
                    else:
                        harmonization_choice = None

                    effective_recipe_choice, harmonization_choice, site_strategy_note = _resolve_site_strategy(
                        recipe_choice,
                        test_type,
                        confound_vars,
                    )
            else:
                recipe_choice = "No site handling"
                harmonization_choice = None
                effective_recipe_choice = recipe_choice

            if site_col is None:
                effective_recipe_choice = recipe_choice
                site_strategy_note = None
                    
            profile_title = "#### 🧠 2. Inference Method Profile" if is_synthetic else "#### 🧠 3. Inference Method Profile"
            st.markdown(profile_title)
            profile_choice = st.selectbox(
                "Select method profile:",
                [
                    "Balanced default (TFNBS)",
                    "Conservative network-aware (NI-TFNBS)",
                    "Diffuse block effect (FBC-TFNBS/cNBS)",
                    "Classical connected component (NBS)",
                    "Diagnostic baseline (t-stat max-permutation)",
                    "Fast edge-wise screen (BH-FDR)"
                ],
                index=0,
                key=persistent_key('design:select_method_profile', scope='design'),
            )
            
            if "FBC-TFNBS" in profile_choice:
                sub_method = st.radio("Choose specific block method:", ["fbc_tfnbs", "cnbs"], horizontal=True, key=persistent_key('design:choose_specific_block_method', scope='design'))
                operator_choice = sub_method
            elif "NI-TFNBS" in profile_choice:
                operator_choice = "ni_tfnbs"
            elif "TFNBS" in profile_choice:
                operator_choice = "tfnbs"
            elif "NBS" in profile_choice:
                operator_choice = "nbs"
            elif "t-stat" in profile_choice:
                operator_choice = "tstat"
            else:
                operator_choice = "bh_fdr"
                
            # Make the active enhancement operator highly visible
            if operator_choice == "tfnbs":
                st.info("⚡ **Active Enhancement Operator:** `TFNBS` (Threshold-Free Network-Based Statistics)\n\n*Combines edge-level strength with topological support from neighbouring edges. Ideal for broad, cluster-like effects without needing a hard threshold.*")
            elif operator_choice == "ni_tfnbs":
                st.info("⚡ **Active Enhancement Operator:** `NI-TFNBS` (Network-Informed TFNBS)\n\n*Weights each suprathreshold edge by the activation density of its predefined network block, using fixed eligible block capacity. No connected-component clustering is used.*")
            elif operator_choice == "fbc_tfnbs":
                st.info("⚡ **Active Enhancement Operator:** `FBC-TFNBS` (Focus Block Cluster TFNBS)\n\n*Specially optimized for dense, block-like/modular network alterations (within or between modules).*")
            elif operator_choice == "cnbs":
                st.info("⚡ **Active Enhancement Operator:** `cNBS` (Constrained NBS)\n\n*Averages the signed t over every edge of each network block and tests each block against its own permutation null (Bonferroni across blocks). The inference unit is the block.*")
            elif operator_choice == "nbs":
                st.info("⚡ **Active Enhancement Operator:** `NBS` (Network-Based Statistics)\n\n*Classic cluster-based method requiring a user-defined primary edge threshold.*")
            elif operator_choice == "tstat":
                st.warning("⚠️ **Active Method:** `t-stat Max-Permutation` (Unenhanced)\n\n*Massive univariate testing with family-wise error rate control. No network-level topological enhancement is applied.*")
            elif operator_choice == "bh_fdr":
                st.warning("⚠️ **Active Method:** `FDR Correction` (Unenhanced)\n\n*Edge-level Benjamini-Hochberg False Discovery Rate correction. No network/topological enhancement is applied.*")

            sc_prior_path = None
            integration_options = ["exact", "right-grid"]
            default_integration = "exact" if DEFAULT_N_THRESHOLDS_PERMUTATION is None else "right-grid"
            default_n_grid = 50 if DEFAULT_N_THRESHOLDS_PERMUTATION is None else int(DEFAULT_N_THRESHOLDS_PERMUTATION)
            nbs_stat_options = ["extent", "intensity"]
            with st.expander("Method Parameters", expanded=False):
                if operator_choice in ("tfnbs", "ni_tfnbs", "fbc_tfnbs"):
                    e_exp = st.number_input("Extent exponent (E)", 0.1, 2.0, float(DEFAULT_EXTENT_EXPONENT), 0.05, key=persistent_key('design:extent_exponent_e', scope='design'))
                    h_exp = st.number_input("Height exponent (H)", 1.0, 5.0, float(DEFAULT_HEIGHT_EXPONENT), 0.1, key=persistent_key('design:height_exponent_h', scope='design'))
                    integration = st.selectbox("Integration", integration_options, index=integration_options.index(default_integration), key=persistent_key('design:integration_mode', scope='design', options=integration_options))
                    n_steps = st.number_input("Integration steps (n)", 1, 10000, default_n_grid, 1, key=persistent_key('design:integration_steps_n', scope='design')) if integration == "right-grid" else DEFAULT_N_THRESHOLDS_PERMUTATION
                    start = st.number_input("Start threshold (t)", 0.0, 100.0, float(DEFAULT_START_THRESHOLD), 0.05, key=persistent_key('design:start_threshold', scope='design'))
                    op_kwargs["e"] = e_exp
                    op_kwargs["h"] = h_exp
                    op_kwargs["n"] = n_steps
                    op_kwargs["start_thres"] = start

                    grid_enabled = st.checkbox(
                        "E/H sensitivity grid (descriptive)",
                        value=False,
                        key=persistent_key('design:eh_sensitivity_grid', scope='design'),
                        help=(
                            "Evaluate several (E, H) cells in one run. The engine pairs "
                            "the two lists element-wise, so they must have the same "
                            "length. Only the cell you declare as primary BEFORE the run "
                            "carries the confirmatory claim; every other cell is "
                            "descriptive sensitivity."
                        ),
                    )
                    if grid_enabled:
                        grid_e_text = st.text_input(
                            "Extent exponents (E), comma-separated",
                            value=f"{float(e_exp):g}, {min(2.0, float(e_exp) + 0.5):g}",
                            key=persistent_key('design:eh_grid_e_values', scope='design'),
                        )
                        grid_h_text = st.text_input(
                            "Height exponents (H), comma-separated",
                            value=f"{float(h_exp):g}, {min(5.0, float(h_exp) + 1.0):g}",
                            key=persistent_key('design:eh_grid_h_values', scope='design'),
                        )
                        e_values, e_err = _parse_float_list(grid_e_text)
                        h_values, h_err = _parse_float_list(grid_h_text)
                        grid_errors = [err for err in (e_err, h_err) if err]
                        if not grid_errors:
                            grid_errors = _validate_eh_grid(
                                e_values, h_values, (0.1, 2.0), (1.0, 5.0)
                            )
                        for err in grid_errors:
                            st.error(f"❌ **E/H grid:** {err}")
                        if not grid_errors:
                            cell_labels = [
                                f"Cell {k}: E={e:g}, H={h:g}"
                                for k, (e, h) in enumerate(zip(e_values, h_values))
                            ]
                            default_primary = next(
                                (
                                    k for k, (e, h) in enumerate(zip(e_values, h_values))
                                    if np.isclose(e, float(e_exp)) and np.isclose(h, float(h_exp))
                                ),
                                0,
                            )
                            primary_label = st.selectbox(
                                "Declared primary cell (fixed before the run)",
                                cell_labels,
                                index=default_primary,
                                key=persistent_key(
                                    'design:eh_grid_primary_cell', scope='design',
                                    options=cell_labels,
                                ),
                            )
                            primary_idx = cell_labels.index(primary_label)
                            op_kwargs["e"] = [float(v) for v in e_values]
                            op_kwargs["h"] = [float(v) for v in h_values]
                            grid_e_values = [float(v) for v in e_values]
                            grid_h_values = [float(v) for v in h_values]
                            primary_cell_decl = {
                                "e": float(e_values[primary_idx]),
                                "h": float(h_values[primary_idx]),
                                "param_idx": int(primary_idx),
                                "selection": "declared_before_run",
                            }
                            st.caption(
                                "Descriptive sensitivity only: the grid does not create a "
                                "corrected family-wise conclusion. Only the declared primary "
                                "cell carries the confirmatory claim; the other cells are "
                                "sensitivity analyses and a cell chosen after seeing the "
                                "results cannot be reported as the primary result. No "
                                "p-based automatic selection is performed."
                            )
                        st.caption("The companion method keeps its own scalar E/H.")

                    if operator_choice == "fbc_tfnbs":
                        m_min = st.number_input("Minimum block size (m_min)", 1, 100, int(DEFAULT_MIN_CLUSTER_SIZE), 1, key=persistent_key('design:minimum_block_size_m_min', scope='design'))
                        op_kwargs["min_cluster_size"] = m_min
                    elif operator_choice == "ni_tfnbs":
                        ni_norm = st.selectbox("Normalization", ["sqrt", "none"], index=0, key=persistent_key('design:normalization', scope='design'))
                        op_kwargs["normalization"] = ni_norm

                elif operator_choice == "nbs":
                    tau = st.number_input("Cluster-forming threshold (tau)", 0.0, 10.0, float(DEFAULT_NBS_THRESHOLD), 0.1, key=persistent_key('design:cluster_forming_threshold_tau', scope='design'))
                    op_kwargs["threshold"] = tau
                    nbs_stat = st.selectbox("NBS statistic", nbs_stat_options, index=nbs_stat_options.index(DEFAULT_NBS_STAT), key=persistent_key('design:nbs_statistic', scope='design', options=nbs_stat_options))
                    op_kwargs["nbs_stat"] = nbs_stat

                if operator_choice in ("ni_tfnbs", "fbc_tfnbs", "cnbs"):
                    sc_prior_path = st.text_input(
                        "SC Prior Matrix (.npy) (Overrides Atlas)",
                        value="",
                        help="Supply an empirical structural connectivity matrix to derive data-driven Louvain communities instead of using fixed atlas networks.",
                        key=persistent_key('design:sc_prior_matrix_npy_overrides_atlas', scope='design'),
                    )

                st.markdown("**Tested edge family (optional)**")
                st.caption(
                    "A boolean N×N mask restricts which edges enter the tested family: "
                    "the permutation max-statistic is then calibrated on those edges "
                    "only. It is not an ROI subset and not an SC community label vector."
                )
                edge_mask_upload = st.file_uploader(
                    "Edge family mask (.npy)", type=["npy"], key="design:edge_family_mask_upload"
                )
                if edge_mask_upload is not None:
                    try:
                        candidate_mask = np.load(
                            io.BytesIO(edge_mask_upload.getvalue()), allow_pickle=False
                        )
                    except Exception as exc:
                        st.error(
                            f"❌ **Edge mask:** the file is not a readable .npy array ({exc})."
                        )
                    else:
                        st.session_state["design:edge_mask_array"] = np.asarray(candidate_mask)
                        st.session_state["design:edge_mask_source"] = getattr(
                            edge_mask_upload, "name", "upload.npy"
                        )
                        persistent_key("design:edge_mask_array", scope="design")
                        persistent_key("design:edge_mask_source", scope="design")
                retained_mask = st.session_state.get("design:edge_mask_array")
                if retained_mask is not None:
                    retained_arr = np.asarray(retained_mask)
                    try:
                        retained_facts = _edge_mask_metadata(
                            retained_arr, st.session_state.get("design:edge_mask_source")
                        )
                    except ValueError as exc:
                        st.error(
                            f"❌ **Edge family mask:** {exc}. The stored mask is not a "
                            "tested-family mask; the run is blocked until it is replaced "
                            "or removed."
                        )
                    else:
                        st.caption(
                            "Retained mask "
                            f"`{st.session_state.get('design:edge_mask_source')}` — shape "
                            f"{retained_facts['shape'][0]}×{retained_facts['shape'][1]}, "
                            f"{retained_facts['eligible_edges']} eligible edges, sha256 "
                            f"{retained_facts['sha256'][:16]}…"
                        )
                    if st.button("Remove retained edge mask", key="design:edge_mask_clear"):
                        st.session_state.pop("design:edge_mask_array", None)
                        st.session_state.pop("design:edge_mask_source", None)
                        st.rerun()

            st.markdown("##### 🧩 Missing Connectivity Values")
            st.caption(
                "Missing values in required design fields (subject, condition, group, site, "
                "selected covariates) are a different matter: those rows are excluded "
                "before the run with a warning, as the review panel shows. This control "
                "only concerns missing (NaN) connectivity values in the matrices."
            )
            if test_type in ("paired", "two-sample"):
                missing_choice = st.radio(
                    "Missing connectivity policy",
                    [
                        "Complete-case (stop on any NaN)",
                        "Available-case (test each edge on its available subjects)",
                    ],
                    index=0,
                    horizontal=True,
                    key=persistent_key(
                        'design:missing_connectivity_policy', scope='design',
                        options=[
                            "Complete-case (stop on any NaN)",
                            "Available-case (test each edge on its available subjects)",
                        ],
                    ),
                    help=(
                        "Complete-case is the engine default (missing='raise'): a single "
                        "missing connectivity value stops the run. Available-case fits "
                        "each edge on the subjects that have a finite value there, so the "
                        "tested n varies per edge; it does not tolerate missing design "
                        "fields and adds no new statistical correction."
                    ),
                )
                missing_policy = (
                    "available_case" if missing_choice.startswith("Available") else "raise"
                )
                if missing_policy == "available_case":
                    min_coverage = float(st.number_input(
                        "Minimum edge coverage (share of subjects with a finite value)",
                        min_value=0.5,
                        max_value=1.0,
                        value=0.8,
                        step=0.05,
                        key=persistent_key('design:minimum_edge_coverage', scope='design'),
                        help=(
                            "Engine default 0.8. Edges below this coverage are not tested "
                            "at all (their p-values are NaN and they are reported as "
                            "untested); the coverage rule is a reporting decision, not an "
                            "additional statistical correction."
                        ),
                    ))
            else:
                st.caption(
                    "GLM design: missing connectivity stops the run (engine default "
                    "`missing='raise'`). No policy switch is offered for this path."
                )

            # Option to add sensitivity companion
            st.markdown("##### 🔄 Sensitivity Baseline Comparison")
            run_sensitivity = st.checkbox("Include Sensitivity Companion Analysis", value=False, help="Run both the chosen method and a companion baseline method sequentially to compare results in Tab 3.", key=persistent_key('design:include_sensitivity_companion_analysis', scope='design'))
            
            comp_method = None
            if run_sensitivity:
                all_methods = {
                    "tfnbs": "TFNBS (Threshold-Free NBS)",
                    "ni_tfnbs": "NI-TFNBS (Network-Informed TFNBS)",
                    "fbc_tfnbs": "FBC-TFNBS (Louvain Prior TFNBS)",
                    "cnbs": "CNBS (Constrained NBS)",
                    "nbs": "NBS (Classic Network-Based Statistic)",
                    "tstat": "TSTAT (Max-t statistic max-permutation)",
                    "bh_fdr": "FDR (Benjamini-Hochberg False Discovery Rate)"
                }
                comp_options = [k for k in all_methods.keys() if k != operator_choice]
                companion_map = {
                    "tfnbs": "tstat",
                    "ni_tfnbs": "tstat",
                    "fbc_tfnbs": "tfnbs",
                    "cnbs": "tfnbs",
                    "nbs": "tfnbs",
                    "tstat": "tfnbs",
                    "bh_fdr": "tfnbs"
                }
                default_comp = companion_map.get(operator_choice, comp_options[0])
                default_idx = comp_options.index(default_comp) if default_comp in comp_options else 0
                
                comp_method = st.selectbox(
                    "Select Companion Baseline Method:",
                    comp_options,
                    index=default_idx,
                    format_func=lambda x: all_methods[x],
                    help="The baseline method to run sequentially alongside your primary method for comparison.",
                    key=persistent_key('design:select_companion_baseline_method', scope='design', options=comp_options),
                )

                with st.expander("Companion Method Parameters", expanded=False):
                    if comp_method in ("tfnbs", "ni_tfnbs", "fbc_tfnbs"):
                        comp_e = st.number_input("Companion Extent exponent (E)", 0.1, 2.0, float(DEFAULT_EXTENT_EXPONENT), 0.05, key=persistent_key("comp_e", scope='design'))
                        comp_h = st.number_input("Companion Height exponent (H)", 1.0, 5.0, float(DEFAULT_HEIGHT_EXPONENT), 0.1, key=persistent_key("comp_h", scope='design'))
                        comp_integration = st.selectbox("Companion integration", integration_options, index=integration_options.index(default_integration), key=persistent_key("comp_integration", scope='design', options=integration_options))
                        comp_n = st.number_input("Companion Integration steps (n)", 1, 10000, default_n_grid, 1, key=persistent_key("comp_n", scope='design')) if comp_integration == "right-grid" else DEFAULT_N_THRESHOLDS_PERMUTATION
                        comp_start = st.number_input("Companion start threshold (t)", 0.0, 100.0, float(DEFAULT_START_THRESHOLD), 0.05, key=persistent_key("comp_start", scope='design'))
                        comp_kwargs["e"] = comp_e
                        comp_kwargs["h"] = comp_h
                        comp_kwargs["n"] = comp_n
                        comp_kwargs["start_thres"] = comp_start
                        if comp_method == "fbc_tfnbs":
                            comp_m_min = st.number_input("Companion Minimum block size (m_min)", 1, 100, int(DEFAULT_MIN_CLUSTER_SIZE), 1, key=persistent_key("comp_m_min", scope='design'))
                            comp_kwargs["min_cluster_size"] = comp_m_min
                        elif comp_method == "ni_tfnbs":
                            comp_ni_norm = st.selectbox("Companion Normalization", ["sqrt", "none"], index=0, key=persistent_key("comp_ni_norm", scope='design'))
                            comp_kwargs["normalization"] = comp_ni_norm
                    elif comp_method == "nbs":
                        comp_tau = st.number_input("Companion Cluster-forming threshold (tau)", 0.0, 10.0, float(DEFAULT_NBS_THRESHOLD), 0.1, key=persistent_key("comp_tau", scope='design'))
                        comp_kwargs["threshold"] = comp_tau
                        comp_nbs_stat = st.selectbox("Companion NBS statistic", nbs_stat_options, index=nbs_stat_options.index(DEFAULT_NBS_STAT), key=persistent_key("comp_nbs_stat", scope='design', options=nbs_stat_options))
                        comp_kwargs["nbs_stat"] = comp_nbs_stat
                
            st.write("")
            
            budget_title = "#### ⏳ 3. Permutation Budget" if is_synthetic else "#### ⏳ 4. Permutation Budget"
            st.markdown(budget_title)
            budget_choice = st.radio(
                "Permutation Budget Preset:",
                [
                    "Quick check (100 permutations)",
                    "Exploratory (1000 permutations)",
                    "Manuscript (5000 permutations)",
                    "Custom"
                ],
                index=1,
                horizontal=True,
                key=persistent_key('design:permutation_budget_preset', scope='design'),
            )
            budget_map = {
                "Quick check (100 permutations)": 100,
                "Exploratory (1000 permutations)": DEFAULT_N_PERMUTATIONS,
                "Manuscript (5000 permutations)": 5000
            }
            if budget_choice == "Custom":
                n_perms = st.number_input("Number of Permutations (B)", 10, 10000, int(DEFAULT_N_PERMUTATIONS), 100, key=persistent_key('design:number_of_permutations_b', scope='design'))
            else:
                n_perms = budget_map[budget_choice]

            alpha_options = [0.001, 0.01, 0.05, 0.1]
            alpha = st.selectbox(
                "Significance level (α)",
                alpha_options,
                index=2,
                format_func=lambda a: f"α = {a:g}",
                help=(
                    "The alpha every result view of this run uses: tables, counters, "
                    "heatmaps, graphs and annotation. With the two-sided default, one "
                    "alpha covers both directions jointly (FWER). A prespecified "
                    "one-sided run applies alpha to the tested direction only."
                ),
                key=persistent_key('design:significance_level_alpha', scope='design', options=alpha_options),
            )
            if tested_tail is None:
                st.caption(
                    "Two-sided family: α is shared by both directions — detections in "
                    "either direction are FWER-controlled at α jointly."
                )
            else:
                st.caption(
                    f"One-sided family ({tested_tail}): α applies to the tested direction "
                    "only; the other direction is not tested."
                )
                
            accel_options = ["none", "gpd", "gamma"]
            default_accel = "none" if DEFAULT_ACCELERATION is None else DEFAULT_ACCELERATION
            with st.expander("Execution Options", expanded=False):
                acceleration_choice = st.selectbox("Tail Acceleration", accel_options, index=accel_options.index(default_accel), key=persistent_key('design:tail_acceleration', scope='design', options=accel_options))
                use_mp = st.checkbox("Parallel Execution (Multiprocessing)", value=True, key=persistent_key('design:parallel_execution_multiprocessing', scope='design'))
                worker_options = ["auto", "1", "2", "4", "8"]
                workers_choice = st.selectbox(
                    "Worker Processes",
                    worker_options,
                    index=0,
                    key=persistent_key('design:worker_processes', scope='design', options=worker_options),
                    help=(
                        "auto = available CPUs, capped at the number of planned "
                        "permutations. Each pool worker uses one BLAS/OpenMP "
                        "thread. The run plan records the requested value and the "
                        "engine-resolved worker count."
                    ),
                )
                n_processes = None if workers_choice == "auto" else int(workers_choice)
                st.caption(
                    "Resource measurements come only from the small bounded local "
                    "benchmark recorded in the NETSTAT-27 evidence note "
                    "(<=24 observations, <=64 permutations, <=2 workers); they are "
                    "not universal limits. For large runs use the CLI "
                    "(`conninfpy run`) and set n_processes explicitly."
                )
                seed_val = st.number_input("Random Seed", 1, 10000, 42, key=persistent_key('design:random_seed', scope='design'))
                
        with col_right:
            st.markdown("#### 📋 Review & Execute")
            
            is_fisher_z = data_is_fisher_z()
                    
            with st.container(border=True):
                if is_fisher_z:
                    st.success("✅ **Input scale:** Fisher-z connectivity")
                    clip_note = st.session_state.get("last_fisher_clip_note")
                    if clip_note:
                        st.warning(clip_note)
                else:
                    st.warning(
                        "⚠️ **Input scale: raw correlation.** Inference runs on the correlations "
                        "as loaded; no Fisher z is applied automatically. Fisher r-to-z is usually "
                        "recommended (it stabilises the variance); apply it here to analyse z values."
                    )
                    if st.button("⚡ Apply Fisher-z now", key="apply_fisher_z_btn"):
                        # The historical in-app policy, now explicit and shared:
                        # clip the correlations at ±0.9999 before arctanh
                        # (z bounded by atanh(0.9999) ≈ 4.95); the API's
                        # default clip-z-at-±5 policy is a different policy.
                        # Invalid correlations (|r| > 1) are rejected by the
                        # shared helper; the dataset, its kind and metadata
                        # stay untouched when that happens.
                        try:
                            from conninfpy import fisher_r_to_z
                            from conninfpy._prepare import make_fisher_entry
                            z, fisher_report = fisher_r_to_z(
                                st.session_state.connectivity_data,
                                clip_r=0.9999, return_report=True,
                            )
                        except ValueError as exc:
                            st.error(f"Fisher r-to-z not applied: {exc}")
                        else:
                            st.session_state.connectivity_data = z
                            st.session_state.connectivity_data_kind = "fisher_z"
                            clear_downstream_results()
                            st.session_state["last_fisher_clip_note"] = fisher_clip_note(fisher_report)
                            # Structured upstream provenance of THIS transform;
                            # analyze() will carry it into the result without
                            # transforming or counting again.
                            st.session_state["dataset_fisher_history"] = [make_fisher_entry(
                                stage="app_design_button", scope="dataset_in_memory",
                                report=fisher_report)]
                            st.success("Fisher r-to-z transform applied to dataset in-memory!")
                            st.rerun()
                        
                resolution = 1.0 / (n_perms + 1)
                
                # Compute a conservative, range-based runtime estimate.
                est_str = "N/A"
                if st.session_state.connectivity_data is not None:
                    N_sub = st.session_state.connectivity_data.shape[0]
                    V_nodes = st.session_state.connectivity_data.shape[1]
                    est_str = _estimate_runtime_range(
                        n_subjects=N_sub,
                        n_nodes=V_nodes,
                        n_permutations=n_perms,
                        test_type=test_type,
                        method=operator_choice,
                        use_mp=use_mp,
                        run_sensitivity=run_sensitivity,
                    )
                
                prelaunch_summary = _build_prelaunch_design_qc_summary(
                    pheno_df,
                    st.session_state.connectivity_data,
                    question_choice=question_choice,
                    test_type=test_type,
                    group_col=group_col,
                    ref_group=ref_group,
                    target_group=target_group,
                    interest_var=interest_var,
                    subject_col=subject_col,
                    condition_col=condition_col,
                    baseline_val=baseline_val,
                    target_val=target_val,
                    confound_vars=confound_vars,
                    site_col=site_col,
                    harmonization_choice=harmonization_choice,
                    roi_indices=st.session_state.get("roi_indices"),
                )
                _render_prelaunch_design_qc_summary(prelaunch_summary)

                from conninfpy.interpret.motion_qc import (
                    FD_COLUMN_CANDIDATES,
                    detect_fd_column,
                )

                motion_qc_enabled = st.checkbox(
                    "Add descriptive motion / QC–FC side-car (computed before inference)",
                    value=False,
                    key=persistent_key(MOTION_QC_WIDGET, scope='design'),
                    help=(
                        "Opt-in, descriptive only: Spearman associations of edge values with "
                        "mean framewise displacement (per edge and per subject), NaN/negative "
                        "edge shares with denominators, and (with known atlas coordinates) a "
                        "distance association. No p-values, no subjects/edges excluded, and the "
                        "inference result is bit-identical with it on or off."
                    ),
                )
                motion_qc_fd_col = None
                if motion_qc_enabled:
                    auto_label = "auto-detect (" + " / ".join(FD_COLUMN_CANDIDATES) + ")"
                    fd_options = [auto_label] + _motion_qc_fd_candidates(pheno_df)
                    fd_choice = st.selectbox(
                        "FD column (mean framewise displacement, mm)",
                        fd_options,
                        index=0,
                        key=persistent_key(MOTION_QC_FD_WIDGET, scope='design', options=fd_options),
                    )
                    motion_qc_fd_col = None if fd_choice == auto_label else fd_choice
                    if detect_fd_column(pheno_df) is None and motion_qc_fd_col is None:
                        st.caption(
                            "No Motion_FD / func_mean_fd column found: the diagnostic will be "
                            "saved with an explicit `fd_unavailable` reason and no FD is invented."
                        )
                    else:
                        st.caption(
                            "FD is taken from the phenotype column in millimetres; the Power vs "
                            "Jenkinson convention is not guessed and no FD is recomputed from ROI "
                            "time series."
                        )

                if motion_qc_enabled:
                    preview_key = _motion_qc_preview_key(
                        test_type, question_choice, motion_qc_fd_col,
                        group_col, ref_group, target_group, interest_var,
                        condition_col, baseline_val, target_val, subject_col,
                        st.session_state.get("loaded_settings_hash"),
                        tuple(st.session_state.get("roi_indices") or ()),
                        tuple(paired_confound_cols or ()),
                    )
                    preview_col1, preview_col2 = st.columns([1, 1])
                    compute_preview = preview_col1.button(
                        "Compute motion / QC–FC preview (no inference)",
                        key="motion_qc_preview_btn",
                        width="stretch",
                    )
                    if compute_preview:
                        try:
                            from conninfpy.interpret.report import motion_qc_markdown

                            Y_prev = st.session_state.connectivity_data.copy()
                            atlas_prev = active_analysis_atlas(base_atlas)
                            if st.session_state.get("roi_indices") is not None:
                                roi_prev = st.session_state.roi_indices
                                Y_prev = Y_prev[:, roi_prev][:, :, roi_prev]
                                atlas_prev = st.session_state.sub_atlas
                            prev_arrays = _build_design_arrays(
                                Y_prev, st.session_state.pheno_df,
                                test_type=test_type,
                                question_choice=question_choice,
                                group_col=group_col,
                                ref_group=ref_group,
                                target_group=target_group,
                                interest_var=interest_var,
                                confound_vars=confound_vars,
                                site_col=site_col,
                                condition_col=condition_col,
                                subject_col=subject_col,
                                baseline_val=baseline_val,
                                target_val=target_val,
                                paired_confound_cols=paired_confound_cols,
                                require_paired_bijection=(test_type == "paired"),
                            )
                            if prev_arrays["n_dropped"] > 0:
                                st.warning(
                                    f"⚠️ **Missing Data Excluded:** {prev_arrays['n_dropped']} "
                                    "subject(s) would be dropped for the selected model variables."
                                )
                            if prev_arrays.get("n_pairs_dropped_confound"):
                                st.warning(
                                    "⚠️ **Pairs excluded for missing condition-specific "
                                    f"confounds:** {prev_arrays['n_pairs_dropped_confound']} "
                                    "subject pair(s) would be excluded."
                                )
                            prev_fd_col = motion_qc_fd_col or detect_fd_column(prev_arrays["pheno_filtered"])
                            prev_spec = _motion_qc_spec_from_arrays(
                                prev_arrays,
                                test_type=test_type,
                                question_choice=question_choice,
                                fd_col=prev_fd_col,
                                subject_col=subject_col,
                                group_col=group_col,
                                ref_group=ref_group,
                                target_group=target_group,
                                baseline_val=baseline_val,
                                target_val=target_val,
                            )
                            prev_scale = {"correlation": "r", "fisher_z": "z"}.get(
                                st.session_state.get("connectivity_data_kind")
                            )
                            prev_preview = _motion_qc_preview(
                                prev_spec, test_type,
                                prev_arrays["Y"] if test_type != "two-sample" else None,
                                prev_arrays["group1"], prev_arrays["group2"],
                                atlas=atlas_prev, scale=prev_scale,
                            )
                            st.session_state["motion_qc_preview"] = prev_preview
                            st.session_state["motion_qc_preview_key"] = preview_key
                        except Exception as exc:
                            st.warning(f"Motion / QC–FC preview could not be computed: {exc}")

                    cached_preview = st.session_state.get("motion_qc_preview")
                    cached_key = st.session_state.get("motion_qc_preview_key")
                    if cached_preview is not None and cached_key == preview_key:
                        from conninfpy.interpret.report import motion_qc_markdown

                        with st.expander(
                            "Motion / QC–FC preview (descriptive; computed before any inference)",
                            expanded=True,
                        ):
                            st.caption(
                                "Computed from the rows this design would analyse, without "
                                "running inference; the same diagnostic is saved with the result."
                            )
                            st.markdown(motion_qc_markdown(cached_preview))
                    elif cached_preview is not None:
                        st.caption(
                            "A stored motion/QC–FC preview belongs to a different design; "
                            "compute it again to see the current one."
                        )

                st.markdown("**Structured Run Plan:**")
                # Retained mask facts (validated later against the analysis ROIs;
                # the metadata here only describes the stored array and a
                # malformed mask must not crash the page).
                if st.session_state.get("design:edge_mask_array") is not None:
                    try:
                        edge_mask_plan = _edge_mask_metadata(
                            st.session_state["design:edge_mask_array"],
                            st.session_state.get("design:edge_mask_source"),
                        )
                    except ValueError as exc:
                        edge_mask_plan_error = str(exc)
                        edge_mask_plan = None
                if edge_mask_plan_error is not None:
                    edge_mask_summary = f"invalid retained mask — {edge_mask_plan_error}"
                elif edge_mask_plan:
                    edge_mask_summary = (
                        f"edge family mask: {edge_mask_plan['eligible_edges']} eligible "
                        f"edges, sha256 {edge_mask_plan['sha256'][:12]}…"
                    )
                else:
                    edge_mask_summary = "all edges"
                method_parameters = _format_method_plan_parameters(operator_choice, op_kwargs)
                plan_md = f"""
| Field | Value |
|---|---|
| **Question** | {question_choice} |
| **Statistical Path** | {test_type.upper() if not (question_choice == 'Group Difference' and adjust_covariates) else 'GLM (Promoted)'} |
| **t statistic** | {("Student (pooled variance)" if var_equal else "Welch (separate variances)") if test_type == "two-sample" else ("GLM (pooled residual variance)" if test_type == "glm" else "one-sample / paired")} |
| **Hypothesis Direction** | {"two-sided — one joint α-family" if tested_tail is None else f"one-sided ({tested_tail}), fixed before the run"} |
| **Significance Level** | α = {alpha:g} |
| **Method Profile** | {profile_choice.split(' ')[0]} ({operator_choice}) |
| **Method Parameters** | {method_parameters} |
| **Atlas Metadata** | {'None' if active_analysis_atlas(base_atlas) is None else f"{len(active_analysis_atlas(base_atlas))} ROIs"} |
| **Site Strategy** | {effective_recipe_choice} |
| **Missing Connectivity** | {(f"available-case, minimum edge coverage {min_coverage:.0%}" if missing_policy == "available_case" else "complete-case (stop on any NaN)")} |
| **Tested Edge Family** | {edge_mask_summary} |
| **E/H Grid** | {(f"{len(grid_e_values)} cells, declared primary E={primary_cell_decl['e']:g}, H={primary_cell_decl['h']:g} (cell {primary_cell_decl['param_idx']})" if primary_cell_decl else "single scalar (E, H)")} |
| **Condition Confounds** | {(', '.join(paired_confound_cols) + ' (B - A computed by the engine)' if paired_confound_cols else "none")} |
| **Permutations** | {n_perms} |
| **Parallel** | {("engine-resolved workers (auto, capped at permutations)" if use_mp and n_processes is None else (f"{n_processes} workers (requested)" if use_mp else "single process (requested)"))} |
| **Rough Runtime** | `{est_str}` |
| **P-value Resolution** | ~{resolution:.4f} |
"""
                st.markdown(plan_md)
                st.caption("Runtime is a rough range; it does not account for the number of integration steps. First runs, multiprocessing startup, and TFNBS scoring can move it substantially.")
                if site_strategy_note:
                    st.info(site_strategy_note)
                
                res_color = "green" if n_perms >= 1000 else "orange"
                st.markdown(f"<span style='color:{res_color}; font-weight:bold;'>p-value resolution is about 1 / ({n_perms} + 1) = {resolution:.4f}</span>", unsafe_allow_html=True)
                
                has_blocking_errors = False
                
                if question_choice == "Paired Condition" and subject_col is None:
                    st.error("❌ **Subject Column Required:** A subject ID column must be selected for paired condition contrasts.")
                    has_blocking_errors = True

                if test_type == "paired" and subject_col and condition_col:
                    precheck_cols = [c for c in (condition_col, subject_col, site_col) if c]
                    frame_precheck = pheno_df.dropna(subset=precheck_cols)
                    paired_problems = _paired_design_violations(
                        frame_precheck, condition_col, subject_col, baseline_val, target_val
                    )
                    if paired_problems:
                        st.error(
                            "❌ **Paired design is not a one-to-one subject match:** "
                            + " | ".join(paired_problems)
                            + " Nothing is dropped or matched by row position; fix the "
                            "subject IDs before running."
                        )
                        has_blocking_errors = True
                    else:
                        n_verified_pairs = int(
                            (frame_precheck[condition_col] == baseline_val).sum()
                        )
                        st.caption(
                            f"Paired design verified: {n_verified_pairs} subject(s) measured "
                            f"in both {baseline_val!r} and {target_val!r}; condition A order is "
                            "preserved and condition B is matched by subject ID."
                        )
                    
                atlas = active_analysis_atlas(base_atlas)
                data_nodes = (
                    st.session_state.connectivity_data.shape[1]
                    if st.session_state.connectivity_data is not None else None
                )
                analysis_nodes = (
                    len(st.session_state.roi_indices)
                    if st.session_state.roi_indices is not None else data_nodes
                )
                if atlas is not None and st.session_state.connectivity_data is not None:
                    if len(atlas) != analysis_nodes:
                        st.error(
                            f"❌ **Atlas/Data Mismatch:** Active data has {analysis_nodes} analysis ROIs, "
                            f"but atlas metadata has {len(atlas)} rows."
                        )
                        has_blocking_errors = True

                # Optional tested-family mask: validated with the engine's own
                # mask semantics, and blocked for combinations the engine rejects
                # (no silent fallback to the full family or to complete-case).
                if st.session_state.get("design:edge_mask_array") is not None:
                    from conninfpy.stats.pairwise import _check_edge_mask

                    try:
                        validated_mask = _check_edge_mask(
                            np.asarray(st.session_state["design:edge_mask_array"]),
                            int(analysis_nodes) if analysis_nodes else 0,
                        )
                    except Exception as exc:
                        st.error(
                            "❌ **Edge family mask:** "
                            f"{exc} The mask must be a boolean symmetric N×N array for the "
                            f"{analysis_nodes} analysis ROIs."
                        )
                        has_blocking_errors = True
                    else:
                        edge_mask_plan = _edge_mask_metadata(
                            validated_mask, st.session_state.get("design:edge_mask_source")
                        )
                        if operator_choice not in ("tfnbs", "ni_tfnbs", "nbs", "tstat"):
                            st.error(
                                "❌ **Edge family mask:** the engine supports a tested-edge "
                                f"family only for TFNBS, NI-TFNBS, NBS and t-stat; not for `{operator_choice}`."
                            )
                            has_blocking_errors = True
                        elif missing_policy == "available_case":
                            st.error(
                                "❌ **Edge family mask + available-case:** the engine rejects "
                                "a tested-family mask combined with available-case missing "
                                "handling. Choose complete-case or run without the mask."
                            )
                            has_blocking_errors = True
                        elif test_type == "glm" or (
                            test_type == "paired" and paired_confound_cols
                        ):
                            st.error(
                                "❌ **Edge family mask + GLM design:** the GLM route "
                                "(covariates, or paired condition-specific confounds) has no "
                                "edge-mask support in the engine. Run the mask with the "
                                "paired or two-sample contrast, or without covariates."
                            )
                            has_blocking_errors = True
                        else:
                            edge_mask_array = validated_mask
                            st.info(
                                "✅ **Tested edge family:** "
                                f"{edge_mask_plan['eligible_edges']} of "
                                f"{int(analysis_nodes) * (int(analysis_nodes) - 1) // 2} edges "
                                "are eligible; the family-wise correction applies to them only."
                            )

                use_sc_prior = False
                sc_net_labels = None
                
                if operator_choice in ("ni_tfnbs", "fbc_tfnbs", "cnbs"):
                    if sc_prior_path:
                        try:
                            sc = np.load(sc_prior_path)
                            import networkx as nx
                            W_pos = np.where(sc > 0, sc, 0.0)
                            G = nx.from_numpy_array(W_pos)
                            communities = nx.community.louvain_communities(G, weight="weight", seed=seed_val)
                            sc_net_labels = np.empty(sc.shape[0], dtype=np.int_)
                            for idx, comm in enumerate(communities):
                                for node in comm:
                                    sc_net_labels[node] = idx
                            use_sc_prior = True
                        except Exception as e:
                            st.error(f"❌ **SC Prior Error:** Failed to process SC matrix. {e}")
                            has_blocking_errors = True
                            
                    if not use_sc_prior:
                        has_networks = atlas_has_networks(atlas)
                        if not has_networks:
                            st.error("❌ **Network Labels Required:** Selected network-informed method requires an atlas with valid network labels, or a valid SC Prior Matrix.")
                            has_blocking_errors = True

                if run_sensitivity and comp_method in ("ni_tfnbs", "fbc_tfnbs", "cnbs"):
                    if not use_sc_prior and not atlas_has_networks(atlas):
                        st.error("❌ **Companion Method Requires Networks:** Choose a non-network-informed companion, provide atlas networks, or provide an SC Prior Matrix.")
                        has_blocking_errors = True
                        
                if site_col is not None and recipe_choice == "No site handling":
                    st.warning("⚠️ **Uncorrected Sites:** Scanner site variable exists, but no site handling was chosen. Results may contain site confounds.")
                    
                run_enabled = not has_blocking_errors
                btn_label = "🚀 Run Multi-Method Inference (Primary + Companion)" if run_sensitivity else "🚀 Run Connectivity Inference"
                
                if "inference_task" not in st.session_state:
                    st.session_state.inference_task = InferenceTask()
                    
                task = st.session_state.inference_task
                # NETSTAT-27: a running job belongs to the data/design it was
                # launched on. Comparing the recorded signature on every
                # rerun cancels a superseded thread cooperatively instead of
                # letting it publish into a session that has moved on.
                _signature_data = st.session_state.get("connectivity_data")
                _signature_mask = st.session_state.get("design:edge_mask_array")
                job_signature = _run_signature(
                    # -- data + atlas identity -----------------------------
                    loaded_settings_hash=st.session_state.get("loaded_settings_hash"),
                    atlas_signature=st.session_state.get(
                        "loaded_atlas_signature",
                        st.session_state.get("active_atlas_signature"),
                    ),
                    data_id=id(_signature_data),
                    data_shape=getattr(_signature_data, "shape", None),
                    data_dtype=getattr(_signature_data, "dtype", None),
                    roi_indices=st.session_state.get("roi_indices"),
                    data_kind=st.session_state.get("connectivity_data_kind"),
                    # -- design family / question / tail -------------------
                    test_type=test_type,
                    question_choice=question_choice,
                    group_col=group_col,
                    reference_group=ref_group,
                    target_group=target_group,
                    interest_var=interest_var,
                    condition_col=condition_col,
                    subject_col=subject_col,
                    baseline_condition=baseline_val,
                    target_condition=target_val,
                    confounds=tuple(confound_vars or ()),
                    condition_confounds=tuple(paired_confound_cols or ()),
                    tested_tail=tested_tail,
                    var_equal=var_equal,
                    two_tailed=(tested_tail is None),
                    # -- method + every execution parameter ----------------
                    method=operator_choice,
                    acceleration=acceleration_choice,
                    harmonization=harmonization_choice,
                    site_recipe=recipe_choice,
                    op_kwargs=op_kwargs,
                    run_sensitivity=bool(run_sensitivity),
                    comp_method=comp_method,
                    comp_kwargs=comp_kwargs,
                    # -- grid / declared cell / tested family --------------
                    grid_e=grid_e_values,
                    grid_h=grid_h_values,
                    primary_cell=primary_cell_decl,
                    edge_mask=_signature_mask,
                    edge_mask_plan=edge_mask_plan,
                    missing_policy=missing_policy,
                    min_coverage=min_coverage,
                    # -- sampling / resources ------------------------------
                    n_permutations=n_perms,
                    seed=seed_val,
                    use_mp=bool(use_mp),
                    n_processes=n_processes,
                    motion_qc_enabled=bool(motion_qc_enabled),
                    motion_qc_fd_col=motion_qc_fd_col,
                )
                _invalidate_superseded_task(task, job_signature)

                # Check status and draw corresponding elements
                if task.status == "idle":
                    if not run_enabled:
                        status_area.warning("Review the highlighted design or atlas settings below before running inference.")
                    if run_action.button(btn_label, type="primary", disabled=not run_enabled, key="run_inference_btn", width="stretch"):
                        try:
                            Y = st.session_state.connectivity_data.copy()
                            atlas = active_analysis_atlas(base_atlas)
                            if st.session_state.roi_indices is not None:
                                roi_idx = st.session_state.roi_indices
                                Y = Y[:, roi_idx][:, :, roi_idx]
                                atlas = st.session_state.sub_atlas
                                
                            pheno_df = st.session_state.pheno_df

                            arrays = _build_design_arrays(
                                Y, pheno_df,
                                test_type=test_type,
                                question_choice=question_choice,
                                group_col=group_col,
                                ref_group=ref_group,
                                target_group=target_group,
                                interest_var=interest_var,
                                confound_vars=confound_vars,
                                site_col=site_col,
                                condition_col=condition_col,
                                subject_col=subject_col,
                                baseline_val=baseline_val,
                                target_val=target_val,
                                paired_confound_cols=paired_confound_cols,
                                require_paired_bijection=(test_type == "paired"),
                            )
                            if arrays["n_dropped"] > 0:
                                st.warning(f"⚠️ **Missing Data Excluded:** Automatically dropped {arrays['n_dropped']} subject(s) due to missing (`NaN`) values in selected model variables.")
                            if arrays.get("n_pairs_dropped_confound"):
                                st.warning(
                                    "⚠️ **Pairs excluded for missing condition-specific confounds:** "
                                    f"{arrays['n_pairs_dropped_confound']} subject pair(s) have no "
                                    "value for the selected condition-specific confound column(s) in "
                                    "one of the two conditions, so both rows were excluded before "
                                    "the run."
                                )
                            Y = arrays["Y"]
                            pheno_df = arrays["pheno_filtered"]
                            interest = arrays["interest"]
                            confounds = arrays["confounds"]
                            group1 = arrays["group1"]
                            group2 = arrays["group2"]
                            sites = arrays["sites"]
                            ids_group1 = arrays["ids_group1"]
                            ids_group2 = arrays["ids_group2"]
                            confounds_group1 = arrays["confounds_group1"]
                            confounds_group2 = arrays["confounds_group2"]
                            if edge_mask_array is not None:
                                op_kwargs["edge_mask"] = edge_mask_array
                            design_facts = {
                                "n_paired_subjects": (
                                    int(group1.shape[0]) if test_type == "paired" else None
                                ),
                                "paired_subject_order": (
                                    "condition A (baseline) phenotype order; condition B "
                                    "matched one-to-one by subject ID"
                                    if test_type == "paired" else None
                                ),
                                "condition_confound_cols": (
                                    list(arrays.get("condition_confound_cols") or []) or None
                                ),
                                "n_pairs_dropped_confound": int(
                                    arrays.get("n_pairs_dropped_confound") or 0
                                ),
                                "missing_design_rows_dropped": int(arrays["n_dropped"]),
                                "sensitivity_grid": (
                                    {
                                        "e": [float(v) for v in grid_e_values],
                                        "h": [float(v) for v in grid_h_values],
                                        "n_cells": int(len(grid_e_values)),
                                    }
                                    if (grid_e_values and grid_h_values) else None
                                ),
                            }

                            if operator_choice in ("ni_tfnbs", "fbc_tfnbs", "cnbs"):
                                if use_sc_prior:
                                    op_kwargs["net_labels"] = sc_net_labels
                                else:
                                    op_kwargs["net_labels"] = atlas.network_index()

                            annotation_atlas = atlas_for_annotation(atlas, Y.shape[1])

                            # ---- NETSTAT-28: descriptive side-car, computed BEFORE inference ----
                            motion_qc_spec = None
                            if motion_qc_enabled:
                                try:
                                    fd_col = motion_qc_fd_col or detect_fd_column(pheno_df)
                                    spec = _motion_qc_spec_from_arrays(
                                        arrays,
                                        test_type=test_type,
                                        question_choice=question_choice,
                                        fd_col=fd_col,
                                        subject_col=subject_col,
                                        group_col=group_col,
                                        ref_group=ref_group,
                                        target_group=target_group,
                                        baseline_val=baseline_val,
                                        target_val=target_val,
                                    )
                                    native_scale = {"correlation": "r", "fisher_z": "z"}.get(
                                        st.session_state.get("connectivity_data_kind")
                                    )
                                    preview = _motion_qc_preview(
                                        spec, test_type, Y if test_type != "two-sample" else None,
                                        group1, group2, atlas=atlas, scale=native_scale,
                                    )
                                    motion_qc_spec = spec
                                    st.session_state["motion_qc_preview"] = preview
                                    # Same signature as the preview control, so the cached
                                    # expander shows this run's diagnostic (not a stale one).
                                    st.session_state["motion_qc_preview_key"] = preview_key
                                except Exception as exc:
                                    st.warning(
                                        "Motion / QC–FC side-car could not be computed; the analysis "
                                        f"will run without it: {exc}"
                                    )
                                    motion_qc_spec = None
                                    st.session_state.pop("motion_qc_preview", None)
                                    st.session_state.pop("motion_qc_preview_key", None)

                            # NETSTAT-27: every launch gets its OWN task object
                            # and generation. An older background thread holds
                            # (and can only mutate) the previous object, and
                            # even that is generation-guarded in the worker.
                            previous_task = task
                            if getattr(previous_task, "status", "") == "running":
                                previous_task.request_cancel("superseded by a new run")
                            task = InferenceTask()
                            st.session_state.inference_task = task
                            st.session_state["inference_generation"] = (
                                int(st.session_state.get("inference_generation", 0)) + 1
                            )
                            task.generation = int(st.session_state["inference_generation"])
                            task.launch_signature = job_signature
                            task.status = "running"
                            task.progress_message = "Starting background inference thread..."
                            
                            # Reset in-session results and downstream views/widgets
                            st.session_state.inference_result = None
                            st.session_state.edges_df = None
                            st.session_state.companion_inference_result = None
                            st.session_state.companion_edges_df = None
                            st.session_state.companion_method = None
                            st.session_state.run_plan = None
                            st.session_state.annotation = None
                            st.session_state.evidence_packet = None
                            reset_widget_scopes("results", "annotation")
                            
                            # Start thread
                            t = threading.Thread(
                                target=_run_inference_background,
                                kwargs=dict(
                                    task=task,
                                    Y=Y, interest=interest, confounds=confounds, group1=group1, group2=group2,
                                    test_type=test_type, sites=sites, harmonization_choice=harmonization_choice,
                                    operator_choice=operator_choice, n_perms=n_perms, seed_val=seed_val,
                                    use_mp=use_mp, acceleration_choice=acceleration_choice, op_kwargs=op_kwargs,
                                    run_sensitivity=run_sensitivity, comp_method=comp_method, comp_kwargs=comp_kwargs,
                                    sc_net_labels=sc_net_labels, use_sc_prior=use_sc_prior, atlas=atlas,
                                    annotation_atlas=annotation_atlas, alpha=alpha, question_choice=question_choice,
                                    loaded_settings_hash=st.session_state.get("loaded_settings_hash"),
                                    recipe_choice=recipe_choice, effective_recipe_choice=effective_recipe_choice,
                                    group_col=group_col, ref_group=ref_group, target_group=target_group,
                                    interest_var=interest_var, subject_col=subject_col, condition_col=condition_col,
                                    baseline_val=baseline_val, target_val=target_val, confound_vars=confound_vars,
                                    site_col=site_col, active_atlas_signature=st.session_state.get("loaded_atlas_signature", st.session_state.get("active_atlas_signature")),
                                    data_kind=st.session_state.get("connectivity_data_kind"),
                                    tested_tail=tested_tail,
                                    var_equal=var_equal,
                                    preprocessing=st.session_state.get("dataset_preprocessing"),
                                    fisher_history=st.session_state.get("dataset_fisher_history"),
                                    roi_indices=st.session_state.get("roi_indices"),
                                    motion_qc=motion_qc_spec,
                                    ids_group1=ids_group1 if test_type == "paired" else None,
                                    ids_group2=ids_group2 if test_type == "paired" else None,
                                    confounds_group1=confounds_group1,
                                    confounds_group2=confounds_group2,
                                    missing_policy=missing_policy,
                                    min_coverage=min_coverage,
                                    edge_mask_plan=edge_mask_plan,
                                    primary_cell=primary_cell_decl,
                                    design_facts=design_facts,
                                    n_processes=n_processes,
                                )
                            )
                            t.daemon = True
                            t.start()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to start background thread: {e}")
                            st.exception(e)
                            
                elif task.status == "running":
                    run_action.button(
                        "⏳ Inference running…",
                        disabled=True,
                        key="run_inference_running",
                        width="stretch",
                    )
                    if run_action.button(
                        "⏹ Cancel run",
                        key="cancel_inference_btn",
                        width="stretch",
                        help="Stop at the next permutation chunk boundary. No partial result is produced.",
                    ):
                        task.request_cancel("cancelled by the user")
                        st.rerun()

                    def render_running_status():
                        """Poll the thread state without restarting its inference work."""
                        current_task = st.session_state.inference_task
                        if current_task.status != "running":
                            # A full rerun promotes the completed result into
                            # session state and navigates to the results tab.
                            if hasattr(st, "fragment"):
                                st.rerun(scope="app")
                            st.rerun()

                        st.info(f"⏳ **Status:** {current_task.progress_message}")
                        done = int(getattr(current_task, "progress_done", 0) or 0)
                        total = int(getattr(current_task, "progress_total", 0) or 0)
                        stage = getattr(current_task, "progress_stage", None)
                        if total > 0:
                            # Exact parent-side count of folded permutations.
                            st.progress(
                                min(1.0, done / total),
                                text=(
                                    f"Completed permutations: {done} of {total}"
                                    + (f" — stage: {stage}" if stage else "")
                                ),
                            )
                        elif _task_cancel_requested(current_task):
                            st.warning("Cancelling at the next permutation chunk boundary…")
                        else:
                            st.caption(
                                "Completed permutations: 0 — no permutation loop has "
                                "started yet (preparing data/design)."
                            )
                        st.markdown(f"""
                        **Estimated remaining time:** `{est_str}`

                        *   Permutations are running in the background; the count above
                            advances at chunk boundaries, so it is a lower bound
                            between updates.
                        *   You can switch to other tabs (such as Workspace Documentation) or browse around the workspace.
                        *   This status checks automatically every 2 seconds.
                        """)

                    # Fragment reruns are scoped to this status area, so
                    # they do not re-create the background worker thread.
                    with status_area:
                        st.fragment(run_every=2.0)(render_running_status)()
                elif task.status in ("cancelled", "invalidated"):
                    status_area.warning(
                        "⏹ **Inference cancelled.** "
                        f"Reason: {task.cancel_reason or 'cancelled by the user'}. "
                        "The run stopped at a permutation chunk boundary: no p-values, "
                        "no edges and no analysis result were produced from the "
                        "incomplete null distribution, so nothing can be saved or "
                        "exported for this run. The previous completed result (if any) "
                        "was cleared when this run started and is not restored."
                    )
                    if run_action.button(
                        "🚀 Configure & Run New Inference",
                        key="run_inf_new_button",
                        width="stretch",
                    ):
                        task.status = "idle"
                        task.cancel_reason = None
                        st.rerun()
                else:
                    # success or failed
                    status_area.info(f"Inference run complete. Current status: **{task.status.upper()}**")
                    if run_action.button("🚀 Configure & Run New Inference", key="run_inf_new_button", width="stretch"):
                        task.status = "idle"
                        st.rerun()
                        
                # Handle finished background task results
                if task.status == "success":
                    # Publish only the task that is still current and whose
                    # payload was finished for its own generation: a stale or
                    # cancelled thread can never overwrite newer session state.
                    current = st.session_state.get("inference_task") is task
                    task_generation = int(getattr(task, "generation", 0))
                    same_generation = (
                        int(getattr(task, "finished_generation", task_generation))
                        == task_generation
                        and int(st.session_state.get("inference_generation", task_generation))
                        == task_generation
                    )
                    if (
                        current
                        and same_generation
                        and not _task_cancel_requested(task)
                        and st.session_state.inference_result is None
                        and getattr(task, "result", None) is not None
                    ):
                        st.session_state.inference_result = task.result
                        st.session_state.edges_df = task.edges
                        st.session_state.companion_inference_result = task.comp_result
                        st.session_state.companion_edges_df = task.comp_edges
                        st.session_state.companion_method = (task.run_plan or {}).get("companion_method")
                        st.session_state.run_plan = task.run_plan
                        st.success("🎉 Connectivity inference completed successfully!")
                        st.session_state.next_tab = tabs_list[2] # Redirect to Inference Results
                        st.rerun()
                elif task.status == "failed":
                    st.error("❌ Inference failed!")
                    st.text_area("Error Traceback", value=task.error or "Unknown error", height=150, key=persistent_key('design:error_traceback', scope='design'))
                    if st.button("Reset & Try Again", key="reset_failed_inf_task"):
                        task.status = "idle"
                        st.rerun()

                if st.session_state.inference_result is not None and task.status != "running":
                    st.html("""<style>
                    .st-key-design_actions .st-key-design_continue_results button {
                        background: #167345; color: white; border: 1px solid #167345;
                    }
                    .st-key-design_actions .st-key-design_continue_results button:hover {
                        background: #105a35; color: white; border-color: #105a35;
                    }
                    </style>""")
                    if results_action.button("Continue to Results →", key="design_continue_results", width="stretch"):
                        st.session_state.next_tab = tabs_list[2]
                        st.rerun()
