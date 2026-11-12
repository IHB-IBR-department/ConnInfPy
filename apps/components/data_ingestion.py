import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from apps.utils.ui_state import persistent_key, persistent_file_uploader, reset_widget_scopes
from apps.utils.workspace import atlas_display_name, remember_dataset, render_dataset_card

from conninfpy.atlas import AtlasInfo, get_bna_246_nifti_path
from conninfpy.loaders import (
    validate_loaded_dataset,
    NumpyLoader,
    CSVDirectoryLoader,
    NiftiDirectoryLoader,
    AbideSchaeferLoader,
)
from conninfpy.loaders.manifest import drop_subjects_with_missing
from apps.utils.lab_loaders import (
    ChinaCloseCloseLoader,
    MultiSiteOpenCloseLoader,
    OpenCloseLoader,
    StressTimeseriesLoader,
    ZerssenNiftiLoader,
)
from conninfpy.topologies import TopologyDatasetGenerator, list_scenarios

from apps.utils.helpers import (
    LOADER_CLASSES,
    atlas_has_networks,
    clear_downstream_results,
    load_custom_datasets,
    local_dataset_templates_enabled,
    make_sub_atlas,
    resolve_project_path,
    render_help,
    list_manifest_files,
    align_atlas_coordinates,
)

def _add_group_variance(values, added_percent, rng, *, time_points=30):
    """Add a percentage of the approximate Fisher-z sampling variance per edge."""
    if added_percent == 0.0:
        return values
    # For correlations from independent Gaussian time points, Var(atanh(r))
    # is approximately 1 / (T - 3), independently of the correlation strength.
    added_variance = (added_percent / 100.0) / (time_points - 3)
    rows, cols = np.triu_indices(values.shape[1], k=1)
    noise = rng.normal(0.0, np.sqrt(added_variance), size=(len(values), len(rows)))
    result = values.copy()
    result[:, rows, cols] += noise
    result[:, cols, rows] += noise
    return result


def render_data_ingestion_view(base_atlas, atlas_choice, tabs_list):
    """Render the data-ingestion step (manifest upload, preview, validation)."""
    col_t, col_h = st.columns([0.8, 0.2])
    with col_t:
        st.markdown("### Preprocessing & Data Ingestion")
    with col_h:
        render_help("data_source")

    st.caption("Settings apply when you load or generate a dataset. Your active dataset and results are kept while you edit.")

    # Ingestion type selector - simplified options
    ingest_mode = st.radio(
        "Select Ingestion Method",
        [
            "Generate Synthetic Dataset",
            "Built-in data",
            "Dataset Manifest (data.yaml)"
        ],
        horizontal=True,
        key=persistent_key('data:select_ingestion_method', scope='data'),
    )
    with st.container(key="data_actions"):
        load_action, next_action = st.columns(2)

    loader_instance = None
    preview = None
    drop_missing_subjects = False
    ts_preprocess_required = False
    corr_type = "Pearson correlation"
    apply_fisher = False

    # ----------------- LAYOUT GENERATION -----------------
    if ingest_mode == "Generate Synthetic Dataset":
        # Wrap BOTH generator and inspector inside a single bordered container
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.markdown("##### Synthetic Dataset Generator", width="content")
                render_help("synthetic_dataset")
            col_left, col_right = st.columns([0.35, 0.65])
            
            with col_left:
                sim_type = st.radio("Simulation Paradigm", ["Two-Group Contrast", "Two-Group Contrast with Confound", "Continuous Covariate GLM"], key=persistent_key('data:simulation_paradigm', scope='data'))
                
                scenarios = list_scenarios()
                selected_scenario = st.selectbox("Select Target Scenario", scenarios, format_func=lambda name: name.replace("_", " ").capitalize(), key=persistent_key('data:select_target_scenario', scope='data', options=scenarios))

                if base_atlas is None:
                    manual_n_nodes = st.number_input("Number of ROIs / Nodes", 10, 500, 100, 10, key=persistent_key('data:number_of_rois_nodes', scope='data'))
                    manual_n_modules = st.number_input("Synthetic modules", 1, 50, 7, 1, key=persistent_key('data:synthetic_modules', scope='data'))
                else:
                    manual_n_nodes = len(base_atlas)
                    manual_n_modules = len(set(base_atlas.networks)) if atlas_has_networks(base_atlas) else 1
                
                if sim_type in ("Two-Group Contrast", "Two-Group Contrast with Confound"):
                    n_group1 = st.number_input("Subjects in group 1", 10, 200, 30, 5, key=persistent_key('data:subjects_group_1', scope='data'))
                    n_group2 = st.number_input("Subjects in group 2", 10, 200, 30, 5, key=persistent_key('data:subjects_group_2', scope='data'))
                    with st.expander("Group variability", expanded=False):
                        st.caption("Added variance as a percentage of the approximate original Fisher-z sampling variance. 0% adds no noise; 100% adds approximately the original variance again.")
                        variance_g1 = st.number_input("Group 1 added variance (%)", 0, 500, 0, 10, key=persistent_key('data:variance_percent_group_1', scope='data'))
                        variance_g2 = st.number_input("Group 2 added variance (%)", 0, 500, 0, 10, key=persistent_key('data:variance_percent_group_2', scope='data'))
                        st.caption("The reference is 1 / (T − 3), an approximation for Gaussian time series (T = 30), not the observed variance of each edge.")
                        st.caption("Edgewise noise can produce matrices that are not valid correlation matrices.")

                if sim_type == "Two-Group Contrast":
                    effect_sz = st.slider("Signal of Interest (0.0 = Null)", 0.0, 1.0, 0.25, 0.05, key=persistent_key('data:signal_of_interest_0_0_null', scope='data'))
                    st.caption("Signal is introduced in Group 2 (group = 1), relative to Group 1 (group = 0), on the selected connections. Observed differences may vary.")
                elif sim_type == "Two-Group Contrast with Confound":
                    effect_sz = st.slider("Signal of Interest (0.0 = Null)", 0.0, 1.0, 0.25, 0.05, key=persistent_key('data:signal_of_interest_0_0_null_2', scope='data'))
                    st.caption("Signal is introduced in Group 2 (group = 1), relative to Group 1 (group = 0), on the selected connections. The confound can also affect group differences.")
                    confound_effect_sz = st.slider("Confound Effect Size", 0.0, 1.0, 0.15, 0.05, key=persistent_key('data:confound_effect_size', scope='data'))
                    interest_confound_corr = st.slider("Correlation (Signal vs Confound)", -0.9, 0.9, 0.3, 0.1, key=persistent_key('data:correlation_signal_vs_confound', scope='data'))
                    confound_scenario = st.selectbox("Confound Topology Scenario", scenarios, index=scenarios.index("between_modules_dense") if "between_modules_dense" in scenarios else 0, key=persistent_key('data:confound_topology_scenario', scope='data', options=scenarios))
                else:
                    n_subjects = st.number_input("Total Subjects", 20, 400, 60, 10, key=persistent_key('data:total_subjects', scope='data'))
                    effect_sz = st.slider("Signal of Interest (0.0 = Null)", 0.0, 1.0, 0.20, 0.05, key=persistent_key('data:signal_of_interest_0_0_null_3', scope='data'))
                    confound_effect_sz = st.slider("Confound Effect Size", 0.0, 1.0, 0.15, 0.05, key=persistent_key('data:confound_effect_size_2', scope='data'))
                    interest_confound_corr = st.slider("Correlation (Signal vs Confound)", -0.9, 0.9, 0.3, 0.1, key=persistent_key('data:correlation_signal_vs_confound_2', scope='data'))
                    confound_scenario = st.selectbox("Confound Topology Scenario", scenarios, index=scenarios.index("between_modules_dense") if "between_modules_dense" in scenarios else 0, key=persistent_key('data:confound_topology_scenario_2', scope='data', options=scenarios))

                synthetic_fisher = st.checkbox("Apply Fisher r-to-z transform", value=True, help="Output Fisher-z matrices when selected; otherwise output correlations (r).", key=persistent_key('data:synthetic_fisher', scope='data'))
                synthetic_seed = st.number_input(
                    "Random seed", min_value=0, max_value=999999, value=42, step=1,
                    help="The same seed and settings reproduce the same dataset. A different seed changes the simulated data and may change the affected connections.",
                    key=persistent_key('data:synthetic_seed', scope='data'),
                )

                synthetic_settings = [st.session_state.get("active_atlas_signature", atlas_choice), ingest_mode, sim_type, selected_scenario, manual_n_nodes, manual_n_modules]
                if sim_type != "Continuous Covariate GLM":
                    synthetic_settings.extend([n_group1, n_group2, variance_g1, variance_g2])
                else:
                    synthetic_settings.append(n_subjects)
                synthetic_settings.append(effect_sz)
                if sim_type != "Two-Group Contrast":
                    synthetic_settings.extend([confound_effect_sz, interest_confound_corr, confound_scenario])
                synthetic_settings.append(synthetic_fisher)
                # Default 42 preserves signatures for datasets created before this control existed.
                if synthetic_seed != 42:
                    synthetic_settings.append(f"seed={synthetic_seed}")
                draft_signature = "|".join(str(part) for part in [*synthetic_settings, corr_type, apply_fisher])
                loaded_synthetic = (
                    st.session_state.connectivity_data is not None
                    and (st.session_state.get("dataset_info") or {}).get("mode") == ingest_mode
                )
                if loaded_synthetic:
                    generate_label = "↻ Generate again" if st.session_state.get("loaded_settings_hash") == draft_signature else "↻ Generate with new settings"
                else:
                    generate_label = "Generate Synthetic Dataset"
                
                if load_action.button(generate_label, type="primary", width="stretch", key="generate_dataset_btn", help="Generate a dataset with the current settings. A successful generation replaces the active dataset and clears its analysis results."):
                    # Use atlas dimensions when metadata is active; otherwise use explicit synthetic controls.
                    n_nodes = int(manual_n_nodes)
                    n_modules = int(manual_n_modules)
                    
                    gen = TopologyDatasetGenerator(n_nodes=n_nodes, n_modules=max(1, n_modules), seed=synthetic_seed)
                    noise_rng_g1 = np.random.default_rng(synthetic_seed + 1)
                    noise_rng_g2 = np.random.default_rng(synthetic_seed + 2)
                    
                    with st.spinner(f"Generating {sim_type} dataset..."):
                        if sim_type == "Two-Group Contrast":
                            ds = gen.generate(
                                selected_scenario,
                                effect_size=effect_sz,
                                n_samples_g1=n_group1,
                                n_samples_g2=n_group2
                            )
                            g1_z, g2_z = ds.fisher_z()
                            g1_z = _add_group_variance(g1_z, variance_g1, noise_rng_g1)
                            g2_z = _add_group_variance(g2_z, variance_g2, noise_rng_g2)
                            Y_z = np.concatenate([g1_z, g2_z], axis=0)
                            n_total = len(Y_z)
                            pheno = pd.DataFrame({
                                "subject_id": [f"sub_{i:02d}" for i in range(n_total)],
                                "group": [0]*len(ds.group1) + [1]*len(ds.group2)
                            })
                            effect_mask = ds.effect_mask
                            
                            confound_weights_val = None
                            empirical_corr = None
                        elif sim_type == "Two-Group Contrast with Confound":
                            base_ds = gen.generate(
                                selected_scenario,
                                effect_size=0.0,
                                n_samples_g1=n_group1,
                                n_samples_g2=n_group2,
                                time_points=30
                            )
                            g1_z, g2_z = base_ds.fisher_z()
                            g1_z = _add_group_variance(g1_z, variance_g1, noise_rng_g1)
                            g2_z = _add_group_variance(g2_z, variance_g2, noise_rng_g2)
                            Y_z = np.concatenate([g1_z, g2_z], axis=0)
                            n_total = len(Y_z)
                            
                            group_interest = np.concatenate([np.zeros(n_group1), np.ones(n_group2)])
                            signal_weights = np.abs(base_ds.effect_mask)
                            
                            if confound_scenario == selected_scenario:
                                confound_weights = signal_weights.copy()
                            else:
                                confound_ds = gen.generate(
                                    confound_scenario,
                                    effect_size=0.0,
                                    n_samples=n_total,
                                    time_points=30
                                )
                                confound_weights = np.abs(confound_ds.effect_mask)
                                
                            rng = np.random.default_rng(synthetic_seed)
                            def _zscore(x):
                                """Population z-score with a tiny std floor to avoid division by zero."""
                                return (x - x.mean()) / max(x.std(ddof=0), 1e-12)
                            
                            z_group = _zscore(group_interest)
                            confound_noise = _zscore(rng.standard_normal(n_total))
                            rho = float(np.clip(interest_confound_corr, -0.99, 0.99))
                            confound = _zscore(rho * z_group + np.sqrt(1.0 - rho**2) * confound_noise)
                            
                            if effect_sz != 0.0:
                                Y_z += effect_sz * group_interest[:, None, None] * signal_weights[None, :, :]
                            if confound_effect_sz != 0.0:
                                Y_z += confound_effect_sz * confound[:, None, None] * confound_weights[None, :, :]
                                
                            diag = np.arange(n_nodes)
                            Y_z[:, diag, diag] = 0.0
                            
                            pheno = pd.DataFrame({
                                "subject_id": [f"sub_{i:02d}" for i in range(n_total)],
                                "group": group_interest.astype(int),
                                "confounds": confound
                            })
                            effect_mask = base_ds.effect_mask
                            
                            confound_weights_val = confound_weights
                            empirical_corr = np.corrcoef(group_interest, confound)[0, 1]
                        else:
                            # Continuous Covariate GLM (matches glm_validation.py logic)
                            base_ds = gen.generate(
                                selected_scenario,
                                effect_size=0.0,
                                n_samples=n_subjects,
                                time_points=30
                            )
                            Y_z, _ = base_ds.fisher_z()
                            signal_weights = np.abs(base_ds.effect_mask)
                            
                            if confound_scenario == selected_scenario:
                                confound_weights = signal_weights.copy()
                            else:
                                confound_ds = gen.generate(
                                    confound_scenario,
                                    effect_size=0.0,
                                    n_samples=n_subjects,
                                    time_points=30
                                )
                                confound_weights = np.abs(confound_ds.effect_mask)
                            
                            rng = np.random.default_rng(synthetic_seed)
                            
                            def _zscore(x):
                                """Population z-score with a tiny std floor to avoid division by zero."""
                                return (x - x.mean()) / max(x.std(ddof=0), 1e-12)
                                
                            interest = _zscore(rng.standard_normal(n_subjects))
                            confound_noise = _zscore(rng.standard_normal(n_subjects))
                            rho = float(np.clip(interest_confound_corr, -0.99, 0.99))
                            confound = _zscore(rho * interest + np.sqrt(1.0 - rho**2) * confound_noise)
                            
                            if effect_sz != 0.0:
                                Y_z += effect_sz * interest[:, None, None] * signal_weights[None, :, :]
                            if confound_effect_sz != 0.0:
                                Y_z += confound_effect_sz * confound[:, None, None] * confound_weights[None, :, :]
                                
                            diag = np.arange(n_nodes)
                            Y_z[:, diag, diag] = 0.0
                            
                            n_total = n_subjects
                            pheno = pd.DataFrame({
                                "subject_id": [f"sub_{i:02d}" for i in range(n_total)],
                                "interest": interest,
                                "confounds": confound
                            })
                            effect_mask = base_ds.effect_mask
                            
                            confound_weights_val = confound_weights
                            empirical_corr = np.corrcoef(interest, confound)[0, 1]
                        
                        clear_downstream_results()
                        st.session_state.connectivity_data = Y_z if synthetic_fisher else np.tanh(Y_z)
                        st.session_state.connectivity_data_kind = "fisher_z" if synthetic_fisher else "correlation"
                        st.session_state.pheno_df = pheno
                        st.session_state["loaded_settings_hash"] = st.session_state.get("current_settings_hash")
                        st.session_state["_just_loaded_data"] = True
                        st.session_state.dataset_atlas = None
                        st.session_state.sub_atlas = None
                        st.session_state["_synthetic_effect_mask"] = effect_mask
                        st.session_state["_synthetic_scenario_name"] = selected_scenario
                        st.session_state["_synthetic_confound_name"] = confound_scenario if sim_type in ("Continuous Covariate GLM", "Two-Group Contrast with Confound") else None
                        st.session_state["_synthetic_confound_mask"] = confound_weights_val
                        st.session_state["_empirical_corr"] = empirical_corr
                        st.session_state["dataset_load_notes"] = []
                        st.session_state.loaded_reference_atlas = base_atlas
                        st.session_state.loaded_atlas_signature = st.session_state.active_atlas_signature
                        remember_dataset(
                            name=selected_scenario.replace("_", " ").capitalize(),
                            source="Synthetic simulation", atlas=atlas_choice, mode=ingest_mode,
                            scenario=selected_scenario, fisher_z=synthetic_fisher, seed=synthetic_seed,
                            group_added_variance_percent=(variance_g1, variance_g2) if sim_type != "Continuous Covariate GLM" else None,
                        )
                        reset_widget_scopes("design", "results", "annotation")
                        
                    st.success(f"Synthetic dataset generated successfully with {selected_scenario} topology.")
                    st.rerun()

            with col_right:
                st.markdown("##### Dataset Inspector")
                if st.session_state.connectivity_data is not None:
                    # Active dataset in session state (e.g. synthetic)
                    n_obs = st.session_state.connectivity_data.shape[0]
                    n_rois = st.session_state.connectivity_data.shape[1]
                    
                    render_dataset_card()
                    
                    # Show the synthetic preview immediately after generation.
                    with st.expander("🔍 Inspect connectivity matrices & relationships", expanded=st.session_state.get("_synthetic_scenario_name") is not None):
                        if st.session_state.get("_synthetic_effect_mask") is not None:
                            overall_mean = np.mean(st.session_state.connectivity_data, axis=0)
                            mask = st.session_state["_synthetic_effect_mask"] > 0
                            mask_triu = np.triu(mask, 1)
                            non_mask_triu = np.triu(~mask, 1)
                            if np.any(mask_triu):
                                mean_in_sig = np.mean(overall_mean[mask_triu])
                                mean_out_sig = np.mean(overall_mean[non_mask_triu])
                                st.write(f"- **Mean Conn (Signal Edges):** `{mean_in_sig:.4f}`")
                                st.write(f"- **Mean Conn (Other Edges):** `{mean_out_sig:.4f}`")
                                
                            if st.session_state.get("_synthetic_confound_mask") is not None:
                                c_mask = st.session_state["_synthetic_confound_mask"] > 0
                                c_mask_triu = np.triu(c_mask, 1)
                                if np.any(c_mask_triu):
                                    mean_in_c = np.mean(overall_mean[c_mask_triu])
                                    st.write(f"- **Mean Conn (Confound Edges):** `{mean_in_c:.4f}`")
                                    
                        pheno_df = st.session_state.pheno_df
                        if pheno_df is not None and "group" in pheno_df.columns:
                            unique_groups = sorted(pheno_df["group"].unique())
                            if len(unique_groups) == 2:
                                g0, g1 = unique_groups
                                g0_idx = np.where(pheno_df["group"] == g0)[0]
                                g1_idx = np.where(pheno_df["group"] == g1)[0]
                                
                                mean_g0 = np.mean(st.session_state.connectivity_data[g0_idx], axis=0)
                                mean_g1 = np.mean(st.session_state.connectivity_data[g1_idx], axis=0)
                                
                                matrix_scale = "Fisher-z" if st.session_state.get("connectivity_data_kind") == "fisher_z" else "correlation r"
                                st.markdown(f"**Mean Connectivity Matrices ({matrix_scale})**")
                                fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
                                
                                v_max = max(np.max(np.abs(mean_g0)), np.max(np.abs(mean_g1)))
                                v_max = max(0.5, min(v_max, 2.0))
                                
                                im0 = ax0.imshow(mean_g0, cmap="RdBu_r", vmin=-v_max, vmax=v_max)
                                ax0.set_title(f"Group {g0} Mean")
                                fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
                                
                                im1 = ax1.imshow(mean_g1, cmap="RdBu_r", vmin=-v_max, vmax=v_max)
                                ax1.set_title(f"Group {g1} Mean")
                                fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)
                                
                                # Confound scatter plot for Two-Group Contrast with Confound
                                if st.session_state.get("_synthetic_confound_mask") is not None and "confounds" in pheno_df.columns:
                                    c_mask = st.session_state["_synthetic_confound_mask"] > 0
                                    c_mask_triu = np.triu(c_mask, 1)
                                    if np.any(c_mask_triu):
                                        sub_means_c = np.mean(st.session_state.connectivity_data[:, c_mask_triu], axis=1)
                                        st.markdown("**Confound Relationship**")
                                        fig2, ax_c = plt.subplots(figsize=(4, 3.5))
                                        ax_c.scatter(pheno_df["confounds"], sub_means_c, alpha=0.7, c='darkorange', edgecolors='none')
                                        ax_c.set_xlabel("Confound (x)")
                                        ax_c.set_ylabel("Mean Connectivity (Confound Mask)")
                                        ax_c.set_title("Confound vs. Brain Connectivity")
                                        z_c = np.polyfit(pheno_df["confounds"], sub_means_c, 1)
                                        p_c = np.poly1d(z_c)
                                        x_c_sort = np.sort(pheno_df["confounds"])
                                        ax_c.plot(x_c_sort, p_c(x_c_sort), "r--", alpha=0.8)
                                        ax_c.grid(True, alpha=0.3)
                                        plt.tight_layout()
                                        st.pyplot(fig2)
                                        plt.close(fig2)
                            else:
                                st.markdown("**Example Correlation Matrix (Subject 0)**")
                                fig, ax = plt.subplots(figsize=(4, 4))
                                v_max = np.max(np.abs(st.session_state.connectivity_data[0]))
                                v_max = max(0.5, min(v_max, 2.0))
                                im = ax.imshow(st.session_state.connectivity_data[0], cmap="RdBu_r", vmin=-v_max, vmax=v_max)
                                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                                ax.set_title(f"Subject 0 ({'Fisher-z' if st.session_state.get('connectivity_data_kind') == 'fisher_z' else 'correlation r'})")
                                st.pyplot(fig)
                                plt.close(fig)
                        elif pheno_df is not None and "interest" in pheno_df.columns:
                            st.markdown("**Covariate Relationships**")
                            # Plot Covariate Scatter Plots
                            if st.session_state.get("_synthetic_effect_mask") is not None:
                                mask = st.session_state["_synthetic_effect_mask"] > 0
                                mask_triu = np.triu(mask, 1)
                                
                                if np.any(mask_triu):
                                    # Mean connectivity in signal mask per subject
                                    sub_means_sig = np.mean(st.session_state.connectivity_data[:, mask_triu], axis=1)
                                    
                                    c_mask_triu = None
                                    if st.session_state.get("_synthetic_confound_mask") is not None:
                                        c_mask = st.session_state["_synthetic_confound_mask"] > 0
                                        c_mask_triu = np.triu(c_mask, 1)
                                    
                                    has_confound = c_mask_triu is not None and np.any(c_mask_triu)
                                    n_cols = 2 if has_confound else 1
                                    fig2, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 3.5), sharey=True, sharex=True)
                                    if n_cols == 1:
                                        axes = [axes]
                                        
                                    ax_s = axes[0]
                                    ax_s.scatter(pheno_df["interest"], sub_means_sig, alpha=0.7, c='royalblue', edgecolors='none')
                                    ax_s.set_xlabel("Signal of Interest (x)")
                                    ax_s.set_ylabel("Mean Connectivity (Signal Mask)")
                                    ax_s.set_title("Behavioral Covariate vs. Brain Connectivity")
                                    z_s = np.polyfit(pheno_df["interest"], sub_means_sig, 1)
                                    p_s = np.poly1d(z_s)
                                    x_sort = np.sort(pheno_df["interest"])
                                    ax_s.plot(x_sort, p_s(x_sort), "r--", alpha=0.8)
                                    ax_s.grid(True, alpha=0.3)
                                    
                                    if has_confound:
                                        sub_means_c = np.mean(st.session_state.connectivity_data[:, c_mask_triu], axis=1)
                                        ax_c = axes[1]
                                        ax_c.scatter(pheno_df["confounds"], sub_means_c, alpha=0.7, c='darkorange', edgecolors='none')
                                        ax_c.set_xlabel("Confound (x)")
                                        ax_c.set_title("Confound vs. Brain Connectivity")
                                        z_c = np.polyfit(pheno_df["confounds"], sub_means_c, 1)
                                        p_c = np.poly1d(z_c)
                                        x_c_sort = np.sort(pheno_df["confounds"])
                                        ax_c.plot(x_c_sort, p_c(x_c_sort), "r--", alpha=0.8)
                                        ax_c.grid(True, alpha=0.3)
                                            
                                    plt.tight_layout()
                                    st.pyplot(fig2)
                                    plt.close(fig2)
                        else:
                            st.markdown("**Example Correlation Matrix (Subject 0)**")
                            fig, ax = plt.subplots(figsize=(4, 4))
                            v_max = np.max(np.abs(st.session_state.connectivity_data[0]))
                            v_max = max(0.5, min(v_max, 2.0))
                            im = ax.imshow(st.session_state.connectivity_data[0], cmap="RdBu_r", vmin=-v_max, vmax=v_max)
                            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            ax.set_title(f"Subject 0 ({'Fisher-z' if st.session_state.get('connectivity_data_kind') == 'fisher_z' else 'correlation r'})")
                            st.pyplot(fig)
                            plt.close(fig)
                else:
                    st.info("No synthetic dataset generated yet. Configure settings on the left and click Generate.")

    else:
        # Standard File/Directory Import or Built-in Templates
        col_left, col_right = st.columns([0.45, 0.55])
        
        with col_left:
            # ----------------- INGESTION MODE: BUILT-IN STRUCTURES -----------------
            if ingest_mode == "Built-in data":
                col_st, col_sh = st.columns([0.7, 0.3])
                with col_st:
                    st.markdown("##### Standard Dataset Templates")
                with col_sh:
                    render_help("built_in_demo")
                
                custom_ds = load_custom_datasets()
                builtin_templates = ["ABIDE-mini", "OpenClose"]
                template_options = builtin_templates
                
                dataset_choice = st.selectbox(
                    "Select Template Structure",
                    template_options,
                    key=persistent_key('data:select_template_structure', scope='data', options=template_options),
                )
                
                if dataset_choice == "ABIDE-mini":
                    data_path = st.text_input(
                        "Pickled Dict Path (.npy)",
                        value="datasets/demo/ABIDE_mini/abide_mini.npy",
                        key=persistent_key('data:pickled_dict_path_npy', scope='data'),
                    )
                    pheno_path = st.text_input(
                        "Optional Phenotypic CSV Path (optional)",
                        value="datasets/demo/ABIDE_mini/ABIDE1_Phenotypic_mini.csv",
                        key=persistent_key('data:optional_phenotypic_csv_path_optional', scope='data'),
                    )
                    if data_path:
                        loader_instance = AbideSchaeferLoader(data_path, pheno_csv_path=pheno_path if pheno_path else None)
                        # Same policy as datasets/abide_mini.yaml (params.missing: drop_subjects):
                        # a subject with a region without signal has undefined correlations.
                        drop_missing_subjects = True
                        st.caption("Subjects with missing connectivity (a region without signal) are "
                                   "removed on loading and listed, as in `datasets/abide_mini.yaml`.")

                elif dataset_choice == "OpenClose":
                    cohort_col, cohort_help_col = st.columns([0.7, 0.3])
                    with cohort_col:
                        cohort = st.selectbox(
                            "Cohort",
                            ["Both sites (IHB + China)", "IHB", "China"],
                            key=persistent_key('data:cohort', scope='data'),
                        )
                    with cohort_help_col:
                        render_help("openclose_multisite")
                    
                    if cohort == "Both sites (IHB + China)":
                        st.caption(
                            "Paired Open-Close observations are retained within each site; "
                            "the loaded phenotype table includes `site` for IHB and China."
                        )
                        ihb_open_path = st.text_input(
                            "IHB Open Condition Path (.npy)",
                            value="datasets/open_close/ihb_open_Schaefer200_strategy-4_GSR.npy",
                            key=persistent_key('data:ihb_open_condition_path_npy', scope='data'),
                        )
                        ihb_close_path = st.text_input(
                            "IHB Close Condition Path (.npy)",
                            value="datasets/open_close/ihb_close_Schaefer200_strategy-4_GSR.npy",
                            key=persistent_key('data:ihb_close_condition_path_npy', scope='data'),
                        )
                        ihb_sub_list = st.text_input(
                            "IHB Subject Order file (.txt)",
                            value="datasets/open_close/subject_order_ihb.txt",
                            key=persistent_key('data:ihb_subject_order_file_txt', scope='data'),
                        )
                        china_open_path = st.text_input(
                            "China Open Condition Path (.npy)",
                            value="datasets/open_close/china_open_Schaefer200_strategy-4_GSR.npy",
                            key=persistent_key('data:china_open_condition_path_npy', scope='data'),
                        )
                        china_close_path = st.text_input(
                            "China Close Condition Path (.npy)",
                            value="datasets/open_close/china_close_Schaefer200_strategy-4_GSR.npy",
                            key=persistent_key('data:china_close_condition_path_npy', scope='data'),
                        )
                        china_sub_list = st.text_input(
                            "China Subject Order file (.txt)",
                            value="datasets/open_close/subject_order_china.txt",
                            key=persistent_key('data:china_subject_order_file_txt', scope='data'),
                        )
                        atlas = "datasets/open_close/schaefer_labels.csv"
                        loader_instance = MultiSiteOpenCloseLoader(
                            {
                                "IHB": {
                                    "open_path": ihb_open_path,
                                    "close_path": ihb_close_path,
                                    "subject_list_path": ihb_sub_list,
                                },
                                "China": {
                                    "open_path": china_open_path,
                                    "close_path": china_close_path,
                                    "subject_list_path": china_sub_list,
                                },
                            },
                            atlas=atlas,
                            drop_missing_rois=True,
                        )
                    elif cohort == "IHB":
                        open_path = st.text_input(
                            "Open Condition Path (.npy)",
                            value="datasets/open_close/ihb_open_Schaefer200_strategy-4_GSR.npy",
                            key=persistent_key('data:open_condition_path_npy', scope='data'),
                        )
                        close_path = st.text_input(
                            "Close Condition Path (.npy)",
                            value="datasets/open_close/ihb_close_Schaefer200_strategy-4_GSR.npy",
                            key=persistent_key('data:close_condition_path_npy_2', scope='data'),
                        )
                        sub_list = st.text_input(
                            "Subject Order file (.txt)",
                            value="datasets/open_close/subject_order_ihb.txt",
                            key=persistent_key('data:subject_order_file_txt_2', scope='data'),
                        )
                    else:
                        open_path = st.text_input(
                            "Open Condition Path (.npy)",
                            value="datasets/open_close/china_open_Schaefer200_strategy-4_GSR.npy",
                            key=persistent_key('data:open_condition_path_npy_2', scope='data'),
                        )
                        close_path = st.text_input(
                            "Close Condition Path (.npy)",
                            value="datasets/open_close/china_close_Schaefer200_strategy-4_GSR.npy",
                            key=persistent_key('data:close_condition_path_npy_3', scope='data'),
                        )
                        sub_list = st.text_input(
                            "Subject Order file (.txt)",
                            value="datasets/open_close/subject_order_china.txt",
                            key=persistent_key('data:subject_order_file_txt_3', scope='data'),
                        )
                        
                    if cohort != "Both sites (IHB + China)":
                        atlas = "datasets/open_close/schaefer_labels.csv"
                    if cohort != "Both sites (IHB + China)" and open_path and close_path:
                        loader_instance = OpenCloseLoader(
                            open_path,
                            close_path,
                            subject_list_path=sub_list if sub_list else None,
                            atlas=atlas,
                            drop_missing_rois=True
                        )
                        
                elif dataset_choice == "China-CloseClose":
                    close_close_col, close_close_help_col = st.columns([0.7, 0.3])
                    with close_close_col:
                        st.caption("China test-retest null example")
                    with close_close_help_col:
                        render_help("china_close_close")
                    close_path = st.text_input(
                        "Close Condition Path (.npy)",
                        value="datasets/open_close/china_close_Schaefer200_strategy-4_GSR.npy",
                        key=persistent_key('data:close_condition_path_npy', scope='data'),
                    )
                    sub_list = st.text_input(
                        "Subject Order file (.txt)",
                        value="datasets/open_close/subject_order_china.txt",
                        key=persistent_key('data:subject_order_file_txt', scope='data'),
                    )
                    atlas = "datasets/open_close/schaefer_labels.csv"
                    
                    if close_path:
                        loader_instance = ChinaCloseCloseLoader(
                            close_path,
                            subject_list_path=sub_list if sub_list else None,
                            atlas=atlas,
                            drop_missing_rois=True
                        )

                elif dataset_choice == "NBS-Schizophrenia":
                    nbs_col, nbs_help_col = st.columns([0.7, 0.3])
                    with nbs_col:
                        st.caption("MATLAB NBS example (Zalesky et al., 2010): 12 SZ vs 15 HC, AAL-74")
                    with nbs_help_col:
                        render_help("nbs_schizophrenia")
                    data_path = st.text_input(
                        "Connectivity matrices (.npy)",
                        value="datasets/demo/NBS_schizophrenia/nbs_schizophrenia.npy",
                        key=persistent_key('data:connectivity_matrices_npy', scope='data'),
                    )
                    pheno_path = st.text_input(
                        "Phenotype CSV",
                        value="datasets/demo/NBS_schizophrenia/pheno.csv",
                        key=persistent_key('data:phenotype_csv', scope='data'),
                    )
                    atlas = st.text_input(
                        "Atlas CSV",
                        value="datasets/demo/NBS_schizophrenia/aal74_atlas.csv",
                        key=persistent_key('data:atlas_csv', scope='data'),
                    )
                    if data_path:
                        # Raw wavelet correlations, analysed as loaded unless Fisher z is applied explicitly.
                        loader_instance = NumpyLoader(
                            data_path,
                            pheno_path=pheno_path or None,
                            data_kind="correlation",
                            atlas_metadata=atlas or None,
                        )

                elif dataset_choice in custom_ds:
                    cfg = custom_ds[dataset_choice]
                    loader_name = cfg.get("loader")
                    params = cfg.get("params", {})
                    
                    st.info(f"Custom registered template using `{loader_name}`")
                    
                    edited_params = {}
                    for key, val in params.items():
                        edited_params[key] = st.text_input(f"{key}", value=str(val), key=persistent_key(f'data:custom:{dataset_choice}:{key}', scope='data'))
                    
                    # Convert parameter values to python types (int, bool, None, str)
                    final_params = {}
                    for k, v in edited_params.items():
                        if v.lower() == "none":
                            final_params[k] = None
                        elif v.lower() == "true":
                            final_params[k] = True
                        elif v.lower() == "false":
                            final_params[k] = False
                        else:
                            try:
                                final_params[k] = int(v)
                            except ValueError:
                                final_params[k] = v
                    
                    if loader_name in LOADER_CLASSES:
                        try:
                            loader_instance = LOADER_CLASSES[loader_name](**final_params)
                            if loader_name in {"StressTimeseriesLoader", "ZerssenNiftiLoader", "CSVDirectoryLoader", "NiftiDirectoryLoader"} or (loader_name == "NumpyLoader" and final_params.get("data_kind") == "timeseries"):
                                ts_preprocess_required = True
                        except Exception as e:
                            st.error(f"Error instantiating custom loader: {e}")
                    else:
                        st.error(f"Loader class `{loader_name}` is not recognized.")

            # ----------------- INGESTION MODE: DATASET MANIFEST -----------------
            elif ingest_mode == "Dataset Manifest (data.yaml)":
                col_mt, col_mh = st.columns([0.7, 0.3])
                with col_mt:
                    st.markdown("##### Import from manifest")
                with col_mh:
                    render_help("manifest_import")
                manifest_options = list_manifest_files()
                if manifest_options:
                    default_idx = 0
                    if "datasets/abide_mini.yaml" in manifest_options:
                        default_idx = manifest_options.index("datasets/abide_mini.yaml")
                    manifest_file = st.selectbox(
                        "Select Ingestion Manifest File (.yaml)",
                        manifest_options,
                        index=default_idx,
                        key=persistent_key('data:select_ingestion_manifest_file_yaml', scope='data', options=manifest_options),
                    )
                else:
                    manifest_file = st.text_input(
                        "Path to data.yaml Manifest File",
                        value="datasets/abide_mini.yaml",
                        key=persistent_key('data:path_to_data_yaml_manifest_file', scope='data'),
                    )
                if manifest_file:
                    try:
                        from conninfpy.loaders.manifest import ManifestLoader
                        loader_instance = ManifestLoader(manifest_file)
                        
                        # Pre-populate preprocessing checkboxes from manifest
                        manifest = loader_instance.manifest
                        requested = manifest.preprocessing.get("requested", {})
                        if manifest.params.get("connectivity_kind") == "partial_correlation":
                            st.session_state["corr_type_pre"] = "Partial correlation"
                        elif manifest.params.get("connectivity_kind") == "correlation":
                            st.session_state["corr_type_pre"] = "Pearson correlation"

                        if "correlation" in requested:
                            val = requested["correlation"]
                            if val == "pearson":
                                st.session_state["corr_type_pre"] = "Pearson correlation"
                            elif val == "spearman":
                                st.session_state["corr_type_pre"] = "Spearman correlation"
                                
                        if "fisher_z" in requested:
                            st.session_state["apply_fisher_pre"] = requested["fisher_z"]
                            
                        if requested.get("extract_timeseries", False) or requested.get("correlation", None) is not None:
                            ts_preprocess_required = True
                        if manifest.loader in {"NiftiDirectoryLoader", "CSVDirectoryLoader", "FmriprepDerivativesLoader", "TimeseriesDirectoryLoader", "StressTimeseriesLoader", "ZerssenNiftiLoader"}:
                            ts_preprocess_required = True

                        delegated_loader = loader_instance.target_loader
                        matrix_options = getattr(delegated_loader, "matrix_options", lambda: [])()
                        if matrix_options:
                            option_by_key = {option["key"]: option for option in matrix_options}
                            default_key = delegated_loader.matrix_key
                            if default_key not in option_by_key:
                                default_key = next(iter(option_by_key))
                            selected_key = st.selectbox(
                                "Connectivity matrix",
                                list(option_by_key),
                                index=list(option_by_key).index(default_key),
                                format_func=lambda key: str(option_by_key[key]["label"]),
                                key=persistent_key(f"manifest_matrix_{manifest_file}", scope='data', options=list(option_by_key)),
                            )
                            selected = loader_instance.set_runtime_matrix_key(selected_key)
                            st.caption(
                                "Session override: "
                                f"`{selected['data_kind']}` input; validation expects "
                                f"{selected['n_rois']} ROIs. The YAML file is unchanged."
                            )
                            
                        st.success("✅ Manifest parsed successfully.")
                    except Exception as e:
                        st.error(f"Error parsing manifest: {e}")



        # Right column: Ingestion Preview summary, Validation report, cache status
        with col_right:
            with st.container(border=True):
                st.markdown("##### Dataset Inspector")
                if loader_instance is not None:
                    is_active_loaded_dataset = st.session_state.connectivity_data is not None
                    if is_active_loaded_dataset:
                        render_dataset_card()
                        for note in st.session_state.get("dataset_load_notes") or []:
                            st.info(note)
                        st.divider()

                    with st.expander("Preview of the next load", expanded=not is_active_loaded_dataset):
                        preview = loader_instance.preview()
                    
                        delegated_loader = getattr(loader_instance, "target_loader", None)
                        if delegated_loader is not None:
                            st.markdown(
                                f"**Loader:** `{loader_instance.name}` "
                                f"-> `{delegated_loader.name}`"
                            )
                        else:
                            st.markdown(f"**Loader:** `{loader_instance.name}`")
                        st.write(f"- **Observations:** {preview.n_observations if preview.n_observations else 'Unknown'}")
                        st.write(f"- **Subjects:** {preview.n_subjects if preview.n_subjects else 'Unknown'}")
                        st.write(f"- **ROIs:** {preview.n_rois if preview.n_rois else 'Unknown'}")
                        st.write(f"- **Data Type:** `{preview.data_kind_guess}`")
                        if preview.atlas_guess:
                            st.write(f"- **Assumed Atlas:** `{preview.atlas_guess}`")
                        if preview.conditions:
                            st.write(f"- **Conditions/Groups:** `{preview.conditions}`")

                        st.write(f"- **Sample Subject IDs:** `{preview.subject_ids_sample}`")
                        st.write(f"- **File Count:** {preview.file_count if preview.file_count else 1}")
                        if preview.file_sizes_mb:
                            st.write(f"- **Total Size:** {preview.file_sizes_mb:.2f} MB")
                        # A manifest may bring atlas metadata for a subset that is unrelated
                        # to the optional reference atlas selected in the sidebar.
                        manifest_atlas_path = getattr(loader_instance, "resolved_paths", {}).get("atlas_metadata")
                        loader_atlas_path = getattr(getattr(loader_instance, "target_loader", None), "atlas_metadata", None)
                        dataset_owns_atlas = bool(manifest_atlas_path or loader_atlas_path)
                        if dataset_owns_atlas and preview.n_rois:
                            atlas_detail = ""
                            if getattr(loader_instance, "manifest", None) is not None:
                                atlas_detail = loader_instance.manifest.metadata.get("atlas_description", "")
                            st.info(
                                f"This dataset supplies its own atlas metadata for {preview.n_rois} ROIs"
                                f" ({atlas_detail}). " if atlas_detail else
                                f"This dataset supplies its own atlas metadata for {preview.n_rois} ROIs. "
                            )
                            st.caption(
                                "It will become the active atlas after loading; the sidebar parcellation is only a reference."
                            )
                        # Check atlas choice matching only when the source does not define its own atlas.
                        elif base_atlas is not None and preview.n_rois and preview.n_rois != len(base_atlas):
                            if (preview.n_rois == 182 and len(base_atlas) == 200) or (preview.n_rois == 84 and len(base_atlas) == 246):
                                st.info(f"💡 Note: Dataset has {preview.n_rois} ROIs (subset of {atlas_choice}). Parcellation metadata will be filtered automatically.")
                            else:
                                st.warning(f"Selected dataset expects {preview.n_rois} ROIs, but reference parcellation in sidebar is {atlas_choice} ({len(base_atlas)} ROIs).")
                    
                        if preview.warnings:
                            for w in preview.warnings:
                                st.warning(w)
                elif st.session_state.connectivity_data is not None:
                    # Active dataset in session state (e.g. synthetic)
                    n_obs = st.session_state.connectivity_data.shape[0]
                    n_rois = st.session_state.connectivity_data.shape[1]
                    
                    st.markdown("**Dataset Type:** `Loaded Session Data`")
                    st.write(f"- **Observations:** {n_obs}")
                    st.write(f"- **ROIs:** {n_rois}")
                    st.write(f"- **Atlas Metadata:** `{atlas_choice}`")
                    
                    group_col_name = next((c for c in ["group", "group_interest", "dx", "diagnosis"] if c in st.session_state.pheno_df.columns), None)
                    if group_col_name:
                        unique_groups = st.session_state.pheno_df[group_col_name].unique().tolist()
                        st.write(f"- **Conditions/Groups:** `{unique_groups}`")
                    
                    st.success("✅ Dataset in-session, verified, and ready for inference.")
                else:
                    st.info("No data source loaded. Configure ingestion settings on the left to preview.")
                    
                if st.session_state.pheno_df is not None:
                    st.markdown("---")
                    st.markdown("**Phenotypic Variables (Active Dataset):**")
                    
                    # If current selections don't match loaded settings, alert the user
                    is_stale = st.session_state.get("loaded_settings_hash") != st.session_state.get("current_settings_hash")
                    if is_stale:
                        st.caption("*(Showing active loaded dataset characteristics. Ingest current settings to update)*")
                        
                    with st.expander("Phenotypic variable completeness"):
                        pheno_df = st.session_state.pheno_df
                        st.write(f"- **Total Variables:** {pheno_df.shape[1]}")
                        st.markdown("**Phenotypic Variables Summary:**")
                    
                        summary_lines = []
                        site_col_detected = next((c for c in ["site", "SITE_ID", "Site", "site_id"] if c in pheno_df.columns), None)
                    
                        for col in pheno_df.columns:
                            non_null = pheno_df[col].notna().sum()
                            pct = (non_null / len(pheno_df)) * 100
                        
                            line = f"- `{col}`: {non_null}/{len(pheno_df)} ({pct:.1f}%)"
                        
                            if site_col_detected and col not in (site_col_detected, "subject_id"):
                                n_sites = pheno_df[site_col_detected].nunique()
                                n_full_sites = 0
                                for site, grp in pheno_df.groupby(site_col_detected):
                                    if len(grp) > 0 and grp[col].notna().sum() == len(grp):
                                        n_full_sites += 1
                                line += f" | 100% complete in {n_full_sites}/{n_sites} sites"
                            
                            summary_lines.append(line)
                        st.markdown("\n".join(summary_lines))

    # Draft subnetwork: commit only after a successful explicit load.
    roi_idx = None
    roi_selection_valid = True
    if ingest_mode != "Generate Synthetic Dataset":
        with col_left, st.expander("ROI selection", expanded=False):
            roi_input_method = st.radio(
                "ROI Selection Input Method",
                ["Whole Brain (Default)", "Manual List (Comma-separated)", "Upload ROI List File (.csv, .txt)"],
                horizontal=True,
                index=0,
                key=persistent_key('data:roi_selection_input_method', scope='data'),
            )

            roi_idx = None

            if roi_input_method == "Whole Brain (Default)":
                pass
            elif roi_input_method == "Manual List (Comma-separated)":
                roi_indices_input = st.text_input(
                    "ROI Indices (comma-separated, e.g. 0,1,2,10,11 to restrict analysis to a subset of nodes)",
                    value="",
                    key=persistent_key('data:roi_indices_comma_separated_e_g_0_1_2_10_11_to_restrict_analysis_to_a_', scope='data'),
                )
                if roi_indices_input:
                    try:
                        roi_idx = [int(x.strip()) for x in roi_indices_input.split(",") if x.strip()]
                    except ValueError:
                        st.error("Please enter valid comma-separated integers.")
                        roi_selection_valid = False
            else:
                roi_file = persistent_file_uploader("Upload ROI List File (.csv, .txt)", type=["csv", "txt"], key='data:upload_roi_list_file_csv_txt')
                if roi_file:
                    try:
                        roi_file.seek(0)
                        content = roi_file.read().decode('utf-8')
                        # Split by commas, newlines, semicolons, tabs
                        raw_items = [x.strip() for x in re.split(r'[,\n;\r\t]+', content) if x.strip()]

                        parsed_idx = []
                        missing_labels = []
                        for item in raw_items:
                            # Skip typical CSV headers if present
                            if item.lower() in {"roi", "rois", "roi_index", "roi_indices", "label", "labels", "roi_name", "roi_names"}:
                                continue
                            try:
                                parsed_idx.append(int(item))
                            except ValueError:
                                # Try matching with atlas labels when metadata exists
                                found = False
                                clean_item = item.replace('"', '').replace("'", "").strip().lower()
                                if base_atlas is not None:
                                    for idx_lbl, lbl in enumerate(base_atlas.labels):
                                        if lbl.lower().strip() == clean_item:
                                            parsed_idx.append(idx_lbl)
                                            found = True
                                            break
                                if not found:
                                    missing_labels.append(item)

                        if missing_labels:
                            st.warning(f"Could not find matching ROIs in the selected atlas for these entries: {missing_labels}")

                        if parsed_idx:
                            roi_idx = sorted(list(set(parsed_idx)))
                    except Exception as e:
                        st.error(f"Error parsing ROI file: {e}")
                        roi_selection_valid = False

            if roi_idx:
                n_available_rois = (
                    int(preview.n_rois) if preview is not None and preview.n_rois
                    else st.session_state.connectivity_data.shape[1] if st.session_state.connectivity_data is not None
                    else len(base_atlas) if base_atlas is not None else 0
                )
                invalid = [r for r in roi_idx if r < 0 or r >= n_available_rois]
                if invalid:
                    st.error(f"Indices {invalid} are out of range for the active data ({n_available_rois} ROIs).")
                    roi_selection_valid = False
                else:
                    st.caption(f"Next load will use a subnetwork of {len(roi_idx)} ROIs.")


    # ----------------- DISPLAY PREPROCESSING OPTIONS & TRIGGER -----------------
    if loader_instance is not None:
        with col_left:
            st.divider()
            # Timeseries connectivity options if timeseries is loaded
            if ts_preprocess_required or (preview and preview.data_kind_guess == "timeseries") or hasattr(getattr(loader_instance, "target_loader", loader_instance), "connectivity_kind"):
                st.markdown("##### 🛠️ Connectivity Construction Options")
                preprocess_source = dataset_choice if ingest_mode == "Built-in data" else manifest_file
                conn_col1, conn_col2 = st.columns(2)
                with conn_col1:
                    default_corr = st.session_state.get("corr_type_pre", "Pearson correlation")
                    choices = ["Pearson correlation", "Partial correlation", "Spearman correlation"]
                    idx = choices.index(default_corr) if default_corr in choices else 0
                    corr_type = st.selectbox("Correlation Estimator", choices, index=idx, key=persistent_key(f'data:correlation:{ingest_mode}:{preprocess_source}', scope='data'))
                with conn_col2:
                    default_fisher = st.session_state.get("apply_fisher_pre", True)
                    apply_fisher = st.checkbox("Fisher r-to-z transform", value=default_fisher, help="Convert correlations to Fisher-z values before analysis.", key=persistent_key(f'data:fisher:{ingest_mode}:{preprocess_source}', scope='data'))
                
                default_shrinkage = getattr(getattr(loader_instance, "manifest", None), "params", {}).get("shrinkage", "auto")
                shrinkage_options = ["auto", "none", "ledoit_wolf", "oas", "fixed"]
                default_label = ("none" if default_shrinkage is None else "fixed" if isinstance(default_shrinkage, (int, float)) else default_shrinkage)
                default_label = default_label if default_label in shrinkage_options else "auto"
                shrinkage_choice = st.selectbox(
                    "Covariance shrinkage", shrinkage_options, index=shrinkage_options.index(default_label),
                    help="Auto: none for Pearson, Ledoit-Wolf for partial correlation. Estimated shrinkage can vary across subjects.",
                    key=persistent_key(f'data:shrinkage:{ingest_mode}:{preprocess_source}', scope='data'))
                shrinkage_value = (st.number_input("Common shrinkage intensity", min_value=0.0, max_value=1.0, value=float(default_shrinkage) if isinstance(default_shrinkage, (int, float)) else 0.1, step=0.01,
                    key=persistent_key(f'data:shrinkage_value:{ingest_mode}:{preprocess_source}', scope='data')) if shrinkage_choice == "fixed" else None)
                if corr_type == "Spearman correlation" and shrinkage_choice != "auto":
                    st.caption("Spearman uses ranks and no covariance shrinkage.")
                st.caption("Compare valid-frame counts and shrinkage by group before inference; subject-wise shrinkage can induce group differences.")

                if "corr_type_pre" in st.session_state or "apply_fisher_pre" in st.session_state:
                    st.caption("ℹ️ Preprocessing options were pre-selected by the manifest.")
            else:
                st.markdown("##### 🛠️ Ingestion Preprocessing")
                st.info("Input data is already precomputed connectivity matrices. No correlation calculation required.")
                apply_fisher = st.checkbox("Fisher r-to-z transform", value=False,
                    key=persistent_key(f'data:fisher_precomputed:{ingest_mode}', scope='data'))

            # Check cache status beforehand to inform the user
            is_cached = False
            if hasattr(loader_instance, "preview"):
                try:
                    p = loader_instance.preview()
                    if any("[Cached]" in w for w in p.warnings):
                        is_cached = True
                except Exception:
                    pass

            if load_action.button("Load, Preprocess & Validate Data", type="primary", width="stretch", disabled=not roi_selection_valid, key="load_dataset_btn", help="Load the selected source. A successful load replaces the active dataset and clears its analysis results."):
                spinner_msg = "Loading dataset from cache..." if is_cached else "Executing loader & validation pipeline (extracting and caching)..."
                with st.spinner(spinner_msg):
                    try:
                        target = getattr(loader_instance, "target_loader", loader_instance)
                        is_timeseries_source = ts_preprocess_required or (preview and preview.data_kind_guess == "timeseries") or hasattr(target, "connectivity_kind")
                        if is_timeseries_source and hasattr(target, "connectivity_kind"):
                            if corr_type == "Spearman correlation":
                                raise ValueError("Spearman is available for raw time-series loaders; this loader constructs connectivity internally.")
                            target.connectivity_kind = "partial_correlation" if corr_type == "Partial correlation" else "correlation"
                            target.shrinkage = ("auto" if shrinkage_choice == "auto" else None if shrinkage_choice == "none" else shrinkage_value if shrinkage_choice == "fixed" else shrinkage_choice)
                        # Manifest conversion is deferred here so UI settings also apply to raw loaders.
                        if is_timeseries_source and hasattr(loader_instance, "manifest") and not hasattr(target, "connectivity_kind"):
                            loader_instance.manifest.params.pop("connectivity_kind", None)
                            loader_instance.manifest.params.pop("shrinkage", None)
                        loaded = loader_instance.load()
                        if drop_missing_subjects:
                            loaded = drop_subjects_with_missing(loaded)
                        load_notes = list(loaded.warnings)

                        # 1. Source Validation
                        report = validate_loaded_dataset(loaded, check_inference_ready=False)
                        if not report.ok:
                            st.error(f"Source validation failed: {report.errors}")
                        else:
                            # 2. Connectivity Preprocessing
                            Y = loaded.data
                            if loaded.data_kind == "timeseries":
                                from conninfpy.connectivity import connectivity_from_dataset
                                if corr_type == "Spearman correlation":
                                    from scipy.stats import rankdata
                                    ranked = []
                                    for ts in loaded.data:
                                        ts = np.asarray(ts, dtype=float)
                                        ranked.append(rankdata(ts[np.isfinite(ts).all(axis=1)], axis=0))
                                    loaded.data = ranked
                                    opts = {"shrinkage": None}
                                    kind = "correlation"
                                else:
                                    kind = "partial_correlation" if corr_type == "Partial correlation" else "correlation"
                                    opts = {} if shrinkage_choice == "auto" else {"shrinkage": None if shrinkage_choice == "none" else shrinkage_value if shrinkage_choice == "fixed" else shrinkage_choice}
                                manifest_params = getattr(getattr(loader_instance, "manifest", None), "params", {})
                                for key in ("min_valid_tr", "group_column"):
                                    if key in manifest_params:
                                        opts[key] = manifest_params[key]
                                if "connectivity_standardize" in manifest_params:
                                    opts["standardize"] = manifest_params["connectivity_standardize"]
                                connectivity_from_dataset(loaded, kind=kind, **opts)
                                if corr_type == "Spearman correlation":
                                    loaded.provenance["connectivity"]["kind"] = "spearman"
                                load_notes = list(loaded.warnings)
                                Y = loaded.data

                            # Apply Fisher z if checked
                            if loaded.data_kind == "correlation":
                                # Always zero out the diagonal before Fisher z-transform to prevent infinity values
                                for i in range(loaded.data.shape[0]):
                                    np.fill_diagonal(loaded.data[i], 0.0)
                                    
                                if apply_fisher:
                                    from conninfpy.utils import fisher_r_to_z
                                    loaded.data = fisher_r_to_z(loaded.data)
                                    loaded.data_kind = "fisher_z"
                                
                            # 3. Final Inference-Ready Validation
                            final_report = validate_loaded_dataset(loaded, check_inference_ready=True)
                            if not final_report.ok:
                                st.error(f"Inference validation failed: {final_report.errors}")
                            else:
                                if roi_idx and any(r < 0 or r >= loaded.data.shape[1] for r in roi_idx):
                                    st.error(f"The loaded dataset has {loaded.data.shape[1]} ROIs. Update the ROI selection before replacing the active dataset.")
                                    return
                                st.session_state.connectivity_data = loaded.data
                                st.session_state.connectivity_data_kind = loaded.data_kind
                                st.session_state.pheno_df = loaded.pheno
                                st.session_state["dataset_load_notes"] = load_notes
                                st.session_state["loaded_settings_hash"] = st.session_state.get("current_settings_hash")
                                st.session_state["_just_loaded_data"] = True

                                if loaded.atlas is not None:
                                    if base_atlas is not None:
                                        loaded.atlas = align_atlas_coordinates(loaded.atlas, base_atlas)
                                    st.session_state.dataset_atlas = loaded.atlas
                                    st.session_state.sub_atlas = loaded.atlas
                                else:
                                    st.session_state.dataset_atlas = None
                                    st.session_state.sub_atlas = None

                                st.session_state.loaded_reference_atlas = base_atlas
                                st.session_state.loaded_atlas_signature = st.session_state.active_atlas_signature
                                loaded_atlas = loaded.atlas if loaded.atlas is not None else base_atlas
                                loaded_atlas_label = (
                                    atlas_display_name(loaded_atlas, atlas_choice)
                                    if loaded_atlas is not None else "No atlas metadata"
                                )
                                remember_dataset(
                                    name=dataset_choice if ingest_mode == "Built-in data" else Path(manifest_file).stem,
                                    source="Built-in dataset" if ingest_mode == "Built-in data" else str(manifest_file),
                                    atlas=loaded_atlas_label, mode=ingest_mode,
                                    loader_provenance=loaded.provenance,
                                )
                                reset_widget_scopes("design", "results", "annotation")
                                    
                                clear_downstream_results()
                                # A data reload starts a new analysis lineage.
                                # Do not honor an old completion redirect to Results.
                                st.session_state.next_tab = None
                                st.session_state.active_tab = tabs_list[0]
                                
                                # Clear synthetic dataset state
                                st.session_state["_synthetic_scenario_name"] = None
                                st.session_state["_synthetic_effect_mask"] = None
                                st.session_state["_synthetic_confound_mask"] = None
                                st.session_state["_synthetic_confound_name"] = None
                                st.session_state["_empirical_corr"] = None
                                
                                if is_cached:
                                    st.success("✅ Dataset successfully loaded from local cache!")
                                else:
                                    st.success("✅ Dataset successfully ingested, preprocessed, cached, and validated!")
                                st.rerun()
                                
                    except Exception as e:
                        st.error(f"Loading failed: {e}")
                        st.exception(e)



    # Display data summaries if populated
    if st.session_state.connectivity_data is not None:
        N_nodes = st.session_state.connectivity_data.shape[1]
        st.success(f"✅ Loaded Connectivity Matrix: {st.session_state.connectivity_data.shape[0]} observations, {N_nodes} x {N_nodes} ROIs.")
        
        with st.expander("🔍 Inspect connectivity matrices", expanded=False):
            st.markdown("Average connectivity across all loaded observations. Self-connections are zeroed out internally.")
            
            # If synthetic dataset, plot both the mean and the ground-truth mask
            if "_synthetic_effect_mask" in st.session_state and st.session_state._synthetic_effect_mask is not None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Mean Conn
                mean_conn = np.mean(st.session_state.connectivity_data, axis=0)
                vmax = np.percentile(np.abs(mean_conn), 99)
                cax1 = ax1.imshow(mean_conn, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
                ax1.set_title("Group Mean Connectivity")
                ax1.set_xlabel("ROI Index")
                ax1.set_ylabel("ROI Index")
                fig.colorbar(cax1, ax=ax1, label="Correlation (or Fisher-z)")
                
                # Ground Truth Mask
                mask = st.session_state._synthetic_effect_mask
                cax2 = ax2.imshow(mask, cmap='viridis', aspect='auto')
                ax2.set_title("Topological Effect Mask")
                ax2.set_xlabel("ROI Index")
                ax2.set_ylabel("ROI Index")
                fig.colorbar(cax2, ax=ax2, label="Effect Weight")
                
                st.pyplot(fig)
                plt.close(fig)
            else:
                mean_conn = np.mean(st.session_state.connectivity_data, axis=0)
                fig, ax = plt.subplots(figsize=(7, 6))
                vmax = np.percentile(np.abs(mean_conn), 99)
                cax = ax.imshow(mean_conn, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
                ax.set_title("Group Mean Connectivity")
                ax.set_xlabel("ROI Index")
                ax.set_ylabel("ROI Index")
                fig.colorbar(cax, ax=ax, label="Correlation (or Fisher-z)")
                st.pyplot(fig)
                plt.close(fig)
        
    if st.session_state.pheno_df is not None:
        st.markdown("**Phenotypic Data Preview**")
        st.dataframe(st.session_state.pheno_df.head(), width="stretch")

    # Compute settings hash at the end of Tab 1 to track all modifications (atlas, source, preprocessing, subnetworks)
    settings_parts = [st.session_state.get("active_atlas_signature", atlas_choice), ingest_mode]
    if ingest_mode == "Built-in data":
        if "dataset_choice" in locals(): settings_parts.append(dataset_choice)
        if "cohort" in locals(): settings_parts.append(cohort)
        if "data_path" in locals(): settings_parts.append(data_path)
        if "pheno_path" in locals(): settings_parts.append(pheno_path)
        source_fields = ("open_path", "close_path", "sub_list", "ihb_open_path", "ihb_close_path", "ihb_sub_list", "china_open_path", "china_close_path", "china_sub_list", "atlas", "final_params")
        source_values = locals()
        settings_parts.extend(f"{field}={source_values[field]}" for field in source_fields if field in source_values)
    elif ingest_mode == "Dataset Manifest (data.yaml)":
        # A different manifest or connectivity matrix is a different dataset.
        if "manifest_file" in locals(): settings_parts.append(manifest_file)
        if "selected_key" in locals(): settings_parts.append(selected_key)
    elif ingest_mode == "Generate Synthetic Dataset":
        settings_parts = synthetic_settings

    # Preprocessing options
    if "corr_type" in locals(): settings_parts.append(corr_type)
    if "apply_fisher" in locals(): settings_parts.append(apply_fisher)
    if "shrinkage_choice" in locals(): settings_parts.extend([shrinkage_choice, shrinkage_value])

    # Subnetwork selection indices
    if roi_idx is not None:
        settings_parts.append(tuple(roi_idx))
        
    current_settings_hash = "|".join(str(p) for p in settings_parts)
    st.session_state["current_settings_hash"] = current_settings_hash
    
    if os.getenv("CONNINFPY_STREAMLIT_DEBUG"):
        st.sidebar.markdown("### 🛠️ Hash Debug")
        st.sidebar.write("Loaded:", st.session_state.get("loaded_settings_hash"))
        st.sidebar.write("Current:", current_settings_hash)
        st.sidebar.write("Parts:", settings_parts)

    if st.session_state.pop("_just_loaded_data", False):
        st.session_state["loaded_settings_hash"] = current_settings_hash
        st.session_state.roi_indices = roi_idx if roi_selection_valid else None
        active_base = st.session_state.get("dataset_atlas")
        if active_base is None:
            active_base = st.session_state.get("loaded_reference_atlas")
        st.session_state.sub_atlas = make_sub_atlas(active_base, roi_idx) if roi_idx and roi_selection_valid else st.session_state.get("dataset_atlas")
    elif st.session_state.connectivity_data is not None and st.session_state.get("loaded_settings_hash") != current_settings_hash:
        st.info("You have changed the settings for the next load. The active dataset and its results are unchanged.")

    if loader_instance is None and ingest_mode != "Generate Synthetic Dataset":
        load_action.button("Load, Preprocess & Validate Data", disabled=True, width="stretch", key="load_dataset_btn")
    if st.session_state.connectivity_data is not None and st.session_state.get("loaded_settings_hash") == current_settings_hash:
        # Once the selected data are ready, guide the user to the next step.
        st.html("""<style>
        .st-key-data_actions .st-key-load_dataset_btn button {
            background: white; color: #31333f; border: 1px solid #bfc5cc;
        }
        .st-key-data_actions .st-key-generate_dataset_btn button {
            background: white; color: #245d8c; border: 2px solid #4681ad;
        }
        .st-key-data_actions .st-key-generate_dataset_btn button:hover {
            background: #eff7fc; color: #17486d; border-color: #245d8c;
        }
        .st-key-data_actions .st-key-btn_continue_to_design button {
            background: #167345; color: white; border: 1px solid #167345;
        }
        .st-key-data_actions .st-key-btn_continue_to_design button:hover {
            background: #105a35; color: white; border-color: #105a35;
        }
        </style>""")
        if next_action.button("Continue to Design & Inference →", key="btn_continue_to_design", width="stretch"):
            st.session_state.next_tab = tabs_list[1]
            st.rerun()
