import numpy as np
import pandas as pd
import streamlit as st
from apps.utils.ui_state import persistent_key, reset_widget_scopes
from apps.utils.workspace import DATA_TAB, go_to_step
import threading
import traceback
from conninfpy.atlas import AtlasInfo
from conninfpy import analyze
from apps.utils.helpers import (
    active_analysis_atlas,
    atlas_for_annotation,
    atlas_has_networks,
    clear_downstream_results,
    data_is_fisher_z,
    effect_direction_labels,
    render_help,
)

class InferenceTask:
    """Background inference job with progress reporting."""
    def __init__(self):
        """Initialize an idle task with empty result slots."""
        self.status = "idle"  # idle, running, success, failed
        self.progress_message = ""
        self.result = None
        self.edges = None
        self.comp_result = None
        self.comp_edges = None
        self.run_plan = None
        self.error = None


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


def _run_inference_background(
    task,
    Y, interest, confounds, group1, group2, test_type, sites, harmonization_choice,
    operator_choice, n_perms, seed_val, use_mp, acceleration_choice, op_kwargs,
    run_sensitivity, comp_method, comp_kwargs, sc_net_labels, use_sc_prior, atlas,
    annotation_atlas, alpha, question_choice, loaded_settings_hash, recipe_choice,
    effective_recipe_choice, group_col, ref_group, target_group, interest_var,
    subject_col, condition_col, baseline_val, target_val, confound_vars, site_col,
    active_atlas_signature, data_kind, tested_tail=None, var_equal=None
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
    """
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
        task.progress_message = f"Executing primary method ({operator_choice.upper()}) permutation loops..."
        from conninfpy import analyze
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
            n_permutations=n_perms,
            rng=seed_val,
            verbose=False,
            use_mp=use_mp,
            tested_tail=tested_tail,
            acceleration=None if acceleration_choice == "none" else acceleration_choice,
            **variance_kwargs,
            **op_kwargs
        )
        edges = res.significant_edges(atlas=annotation_atlas, alpha=alpha)
        
        comp_res = None
        comp_edges = None
        if run_sensitivity and comp_method:
            task.progress_message = f"Executing sensitivity companion ({comp_method.upper()}) permutation loops..."
            comp_run_kwargs = dict(comp_kwargs)
            if comp_method in ("tfnbs", "ni_tfnbs", "fbc_tfnbs"):
                comp_run_kwargs.setdefault("e", 0.4)
                comp_run_kwargs.setdefault("h", 3.0)
                comp_run_kwargs.setdefault("n", None)
                comp_run_kwargs.setdefault("start_thres", 1.65)
                if comp_method in ("ni_tfnbs", "fbc_tfnbs"):
                    comp_run_kwargs["net_labels"] = sc_net_labels if use_sc_prior else atlas.network_index()
            elif comp_method == "cnbs":
                comp_run_kwargs["net_labels"] = sc_net_labels if use_sc_prior else atlas.network_index()
            elif comp_method == "nbs":
                comp_run_kwargs.setdefault("threshold", 3.0)
                comp_run_kwargs.setdefault("nbs_stat", "extent")
                
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
                n_permutations=n_perms,
                rng=seed_val,
                verbose=False,
                use_mp=use_mp,
                tested_tail=tested_tail,
                acceleration=None if acceleration_choice == "none" else acceleration_choice,
                **variance_kwargs,
                **comp_run_kwargs
            )
            comp_edges = comp_res.significant_edges(atlas=annotation_atlas, alpha=alpha)
            
        task.result = res
        task.edges = edges
        task.comp_result = comp_res
        task.comp_edges = comp_edges
        
        task.run_plan = {
            "question_type": question_choice,
            "alpha": alpha,
            "loaded_settings_hash": loaded_settings_hash,
            "design_family": test_type,
            "method": operator_choice,
            "n_permutations": n_perms,
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
            "tail_policy": f"one-sided-{tested_tail}" if tested_tail else "two-sided",
            "tested_tail": tested_tail,
            "tail_selection": "prespecified" if tested_tail else None,
            "two_tailed": tested_tail is None,
            # 'student' / 'welch' (two-sample test), 'glm' (pooled residual variance).
            "t_statistic": t_statistic,
            "seed": seed_val,
            "method_parameters": _method_plan_parameters(operator_choice, op_kwargs),
            "companion_method": comp_method if run_sensitivity else None,
            "companion_method_parameters": (
                _method_plan_parameters(comp_method, comp_run_kwargs)
                if run_sensitivity and comp_method else {}
            ),
        }
        task.status = "success"
        task.progress_message = "✅ Completed!"
    except Exception as e:
        task.error = f"{e}\n{traceback.format_exc()}"
        task.status = "failed"
        task.progress_message = "❌ Failed"


def render_design_inference_view(base_atlas, tabs_list):
    """Render the design & inference step (model spec, run, live status)."""
    col_t, col_h = st.columns([0.8, 0.2])
    with col_t:
        st.markdown("### Design & Inference")
    with col_h:
        render_help("analysis_setup")
    
    if st.session_state.connectivity_data is None or st.session_state.pheno_df is None:
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
        n_perms = 1000
        seed_val = 42
        use_mp = True
        acceleration_choice = "gpd"
        op_kwargs = {}
        comp_kwargs = {}
        
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
                
                st.info(f"**Contrast Direction:**\n• **Positive effect:** {target_val} > {baseline_val}\n• **Negative effect:** {baseline_val} > {target_val}\n\n*Uses sign-flip permutation within-subject. Subject-constant site effects cancel in differences.*")
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
                            "Site labels are checked to be constant within each subject and "
                            "passed as paired exchangeability/provenance blocks. ComBat is not "
                            "applied because additive site effects cancel in within-subject differences."
                        )
                        st.write(
                            "• **Strategy Selected:** `Paired within-subject + site blocks` "
                            "(site is retained for validation and exchangeability provenance; no ComBat needed)"
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
            
            if "TFNBS" in profile_choice:
                if "NI-TFNBS" in profile_choice:
                    operator_choice = "ni_tfnbs"
                else:
                    operator_choice = "tfnbs"
            elif "FBC-TFNBS" in profile_choice:
                sub_method = st.radio("Choose specific block method:", ["fbc_tfnbs", "cnbs"], horizontal=True, key=persistent_key('design:choose_specific_block_method', scope='design'))
                operator_choice = sub_method
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
            with st.expander("Method Parameters", expanded=False):
                if operator_choice in ("tfnbs", "ni_tfnbs", "fbc_tfnbs"):
                    e_exp = st.number_input("Extent exponent (E)", 0.1, 2.0, 0.4, 0.05, key=persistent_key('design:extent_exponent_e', scope='design'))
                    h_exp = st.number_input("Height exponent (H)", 1.0, 5.0, 3.0, 0.1, key=persistent_key('design:height_exponent_h', scope='design'))
                    integration = st.selectbox("Integration", ["exact", "right-grid"], key=persistent_key('design:integration_mode', scope='design'))
                    n_steps = st.number_input("Integration steps (n)", 1, 10000, 50, 1, key=persistent_key('design:integration_steps_n', scope='design')) if integration == "right-grid" else None
                    start = st.number_input("Start threshold (t)", 0.0, 100.0, 1.65, 0.05, key=persistent_key('design:start_threshold', scope='design'))
                    op_kwargs["e"] = e_exp
                    op_kwargs["h"] = h_exp
                    op_kwargs["n"] = n_steps
                    op_kwargs["start_thres"] = start

                    if operator_choice == "fbc_tfnbs":
                        m_min = st.number_input("Minimum block size (m_min)", 5, 100, 20, key=persistent_key('design:minimum_block_size_m_min', scope='design'))
                        op_kwargs["min_cluster_size"] = m_min
                    elif operator_choice == "ni_tfnbs":
                        ni_norm = st.selectbox("Normalization", ["sqrt", "none"], index=0, key=persistent_key('design:normalization', scope='design'))
                        op_kwargs["normalization"] = ni_norm

                elif operator_choice == "nbs":
                    tau = st.number_input("Cluster-forming threshold (tau)", 1.0, 10.0, 3.0, 0.1, key=persistent_key('design:cluster_forming_threshold_tau', scope='design'))
                    op_kwargs["threshold"] = tau
                    nbs_stat = st.selectbox("NBS statistic", ["extent", "intensity"], index=0, key=persistent_key('design:nbs_statistic', scope='design'))
                    op_kwargs["nbs_stat"] = nbs_stat

                if operator_choice in ("ni_tfnbs", "fbc_tfnbs", "cnbs"):
                    sc_prior_path = st.text_input(
                        "SC Prior Matrix (.npy) (Overrides Atlas)",
                        value="",
                        help="Supply an empirical structural connectivity matrix to derive data-driven Louvain communities instead of using fixed atlas networks.",
                        key=persistent_key('design:sc_prior_matrix_npy_overrides_atlas', scope='design'),
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
                        comp_e = st.number_input("Companion Extent exponent (E)", 0.1, 2.0, 0.4, 0.05, key=persistent_key("comp_e", scope='design'))
                        comp_h = st.number_input("Companion Height exponent (H)", 1.0, 5.0, 3.0, 0.1, key=persistent_key("comp_h", scope='design'))
                        comp_integration = st.selectbox("Companion integration", ["exact", "right-grid"], key=persistent_key("comp_integration", scope='design'))
                        comp_n = st.number_input("Companion Integration steps (n)", 1, 10000, 50, 1, key=persistent_key("comp_n", scope='design')) if comp_integration == "right-grid" else None
                        comp_start = st.number_input("Companion start threshold (t)", 0.0, 100.0, 1.65, 0.05, key=persistent_key("comp_start", scope='design'))
                        comp_kwargs["e"] = comp_e
                        comp_kwargs["h"] = comp_h
                        comp_kwargs["n"] = comp_n
                        comp_kwargs["start_thres"] = comp_start
                        if comp_method == "fbc_tfnbs":
                            comp_m_min = st.number_input("Companion Minimum block size (m_min)", 5, 100, 20, key=persistent_key("comp_m_min", scope='design'))
                            comp_kwargs["min_cluster_size"] = comp_m_min
                        elif comp_method == "ni_tfnbs":
                            comp_ni_norm = st.selectbox("Companion Normalization", ["sqrt", "none"], index=0, key=persistent_key("comp_ni_norm", scope='design'))
                            comp_kwargs["normalization"] = comp_ni_norm
                    elif comp_method == "nbs":
                        comp_tau = st.number_input("Companion Cluster-forming threshold (tau)", 1.0, 10.0, 3.0, 0.1, key=persistent_key("comp_tau", scope='design'))
                        comp_kwargs["threshold"] = comp_tau
                        comp_nbs_stat = st.selectbox("Companion NBS statistic", ["extent", "intensity"], index=0, key=persistent_key("comp_nbs_stat", scope='design'))
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
                "Exploratory (1000 permutations)": 1000,
                "Manuscript (5000 permutations)": 5000
            }
            if budget_choice == "Custom":
                n_perms = st.number_input("Number of Permutations (B)", 10, 10000, 1000, 100, key=persistent_key('design:number_of_permutations_b', scope='design'))
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
                
            with st.expander("Execution Options", expanded=False):
                acceleration_choice = st.selectbox("Tail Acceleration", ["gpd", "gamma", "none"], index=0, key=persistent_key('design:tail_acceleration', scope='design'))
                use_mp = st.checkbox("Parallel Execution (Multiprocessing)", value=True, key=persistent_key('design:parallel_execution_multiprocessing', scope='design'))
                seed_val = st.number_input("Random Seed", 1, 10000, 42, key=persistent_key('design:random_seed', scope='design'))
                
        with col_right:
            st.markdown("#### 📋 Review & Execute")
            
            is_fisher_z = data_is_fisher_z()
                    
            with st.container(border=True):
                if is_fisher_z:
                    st.success("✅ **Input scale:** Fisher-z connectivity")
                else:
                    st.warning(
                        "⚠️ **Input scale: raw correlation.** Inference runs on the correlations "
                        "as loaded; no Fisher z is applied automatically. Fisher r-to-z is usually "
                        "recommended (it stabilises the variance); apply it here to analyse z values."
                    )
                    if st.button("⚡ Apply Fisher-z now", key="apply_fisher_z_btn"):
                        Y_clipped = np.clip(st.session_state.connectivity_data, -0.9999, 0.9999)
                        st.session_state.connectivity_data = np.arctanh(Y_clipped)
                        st.session_state.connectivity_data_kind = "fisher_z"
                        clear_downstream_results()
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
                
                st.markdown("**Structured Run Plan:**")
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
| **Permutations** | {n_perms} |
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
                    
                atlas = active_analysis_atlas(base_atlas)
                if atlas is not None and st.session_state.connectivity_data is not None:
                    data_nodes = st.session_state.connectivity_data.shape[1]
                    analysis_nodes = len(st.session_state.roi_indices) if st.session_state.roi_indices is not None else data_nodes
                    if len(atlas) != analysis_nodes:
                        st.error(
                            f"❌ **Atlas/Data Mismatch:** Active data has {analysis_nodes} analysis ROIs, "
                            f"but atlas metadata has {len(atlas)} rows."
                        )
                        has_blocking_errors = True

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
                            
                            # --- NaN Handling ---
                            req_cols = []
                            if test_type == "glm":
                                if question_choice == "Group Difference":
                                    req_cols = [group_col]
                                    if confound_vars: req_cols.extend(confound_vars)
                                else:
                                    req_cols = [interest_var]
                                    if confound_vars: req_cols.extend(confound_vars)
                            elif test_type == "paired":
                                req_cols = [condition_col, subject_col]
                            elif test_type == "two-sample":
                                req_cols = [group_col]
                            
                            if site_col:
                                req_cols.append(site_col)
                                
                            req_cols = [c for c in req_cols if c in pheno_df.columns]
                            
                            if req_cols:
                                valid_mask = pheno_df[req_cols].notna().all(axis=1)
                                num_dropped = len(pheno_df) - valid_mask.sum()
                                if num_dropped > 0:
                                    st.warning(f"⚠️ **Missing Data Excluded:** Automatically dropped {num_dropped} subject(s) due to missing (`NaN`) values in selected model variables.")
                                Y = Y[valid_mask.to_numpy()]
                                pheno_df = pheno_df.loc[valid_mask].reset_index(drop=True)
                            
                            interest = None
                            confounds = None
                            group1 = None
                            group2 = None
                            sites = None
                            
                            if site_col:
                                sites = list(pheno_df[site_col].values)
                                
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
                                else:
                                    interest = pheno_df[interest_var].values.astype(np.float64)
                                    if confound_vars:
                                        confounds = pheno_df[confound_vars].values.astype(np.float64)
                                        
                            elif test_type == "paired":
                                df_baseline = pheno_df[pheno_df[condition_col] == baseline_val]
                                df_target = pheno_df[pheno_df[condition_col] == target_val]
                                common_subjects = sorted(list(set(df_baseline[subject_col]).intersection(set(df_target[subject_col]))))
                                
                                idx_baseline = [df_baseline[df_baseline[subject_col] == s].index[0] for s in common_subjects]
                                idx_target = [df_target[df_target[subject_col] == s].index[0] for s in common_subjects]
                                
                                group1 = Y[idx_baseline]
                                group2 = Y[idx_target]

                                if site_col:
                                    baseline_sites = pheno_df.loc[idx_baseline, site_col].astype(str).tolist()
                                    target_sites = pheno_df.loc[idx_target, site_col].astype(str).tolist()
                                    mismatched_sites = [
                                        subject for subject, baseline_site, target_site in zip(
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
                                g1_mask = (pheno_df[group_col] == ref_group)
                                g2_mask = (pheno_df[group_col] == target_group)
                                group1 = Y[g1_mask.to_numpy()]
                                group2 = Y[g2_mask.to_numpy()]
                                
                            if operator_choice in ("ni_tfnbs", "fbc_tfnbs", "cnbs"):
                                if use_sc_prior:
                                    op_kwargs["net_labels"] = sc_net_labels
                                else:
                                    op_kwargs["net_labels"] = atlas.network_index()

                            annotation_atlas = atlas_for_annotation(atlas, Y.shape[1])
                            
                            task.status = "running"
                            task.progress_message = "Starting background inference thread..."
                            task.result = None
                            task.edges = None
                            task.comp_result = None
                            task.comp_edges = None
                            task.error = None
                            
                            # Reset in-session results and downstream views/widgets
                            st.session_state.inference_result = None
                            st.session_state.edges_df = None
                            st.session_state.companion_inference_result = None
                            st.session_state.companion_edges_df = None
                            st.session_state.companion_method = None
                            st.session_state.run_plan = None
                            st.session_state.annotation = None
                            st.session_state.evidence_packet = None
                            st.session_state.narrative_text = None
                            st.session_state.narrative_results = None
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
                                )
                            )
                            t.daemon = True
                            t.start()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to start background thread: {e}")
                            st.exception(e)
                            
                elif task.status == "running":
                    run_action.button("⏳ Inference running…", disabled=True, key="run_inference_running", width="stretch")
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
                        st.markdown(f"""
                        **Estimated remaining time:** `{est_str}`

                        *   Permutations are running in the background.
                        *   You can switch to other tabs (such as Workspace Documentation) or browse around the workspace.
                        *   This status checks automatically every 2 seconds.
                        """)

                    # Fragment reruns are scoped to this status area, so
                    # they do not re-create the background worker thread.
                    with status_area:
                        st.fragment(run_every=2.0)(render_running_status)()
                else:
                    # success or failed
                    status_area.info(f"Inference run complete. Current status: **{task.status.upper()}**")
                    if run_action.button("🚀 Configure & Run New Inference", key="run_inf_new_button", width="stretch"):
                        task.status = "idle"
                        st.rerun()
                        
                # Handle finished background task results
                if task.status == "success":
                    if st.session_state.inference_result is None:
                        st.session_state.inference_result = task.result
                        st.session_state.edges_df = task.edges
                        st.session_state.companion_inference_result = task.comp_result
                        st.session_state.companion_edges_df = task.comp_edges
                        st.session_state.companion_method = task.run_plan.get("companion_method")
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
