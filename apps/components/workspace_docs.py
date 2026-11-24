import streamlit as st
from apps.utils.helpers import render_help

def render_workspace_docs_view():
    """Render the workspace documentation viewer."""
    col_t, col_h = st.columns([0.8, 0.2])
    with col_t:
        st.markdown("### Help & Documentation")
    with col_h:
        render_help("workspace_documentation")
    
    st.markdown("""
    #### Experimental Designs
    - **Two-sample comparison:** permutation t-test of independent groups; Student's t (pooled variance, the app default and the NBS toolbox choice) or Welch's t (separate variances).
    - **Paired contrast:** Subject-wise difference contrast. The two conditions must be a strict one-to-one subject match: condition A (baseline) keeps the phenotype row order, condition B is matched by subject ID, and empty IDs, duplicate IDs within a condition, a subject measured in only one condition or different numbers of rows block the run and name the subjects. Nothing is dropped, sorted or matched by row position.
    - **Condition-specific confounds (paired):** columns whose value may differ between the two conditions (e.g. condition-level motion). The per-subject difference B − A is computed by the analysis engine — never in this interface — and is never replaced by one shared, subject-constant table. A subject missing a value in either condition is excluded before the run, with a warning.
    - **GLM:** Evaluate variable of interest while regressing out nuisance covariates.

    #### Missing Design Fields vs Missing Connectivity
    - **Missing design fields** (subject, condition, group, site, selected covariates) are a data-entry problem: those rows are excluded before the run and the count is reported.
    - **Missing connectivity** is a separate policy: *Complete-case* stops on any NaN (engine default `missing='raise'`); *Available-case* tests each edge on the subjects that have a finite value there, with a coverage floor `min_coverage` (default 0.8). Available-case does not tolerate missing design fields, adds no statistical correction, and makes the tested n vary per edge — the results page reports tested/untested edges, the coverage floor, the per-edge n range and the affected subjects, and never calls that one complete-case sample. Combinations the engine rejects are blocked before the run.

    #### Tested Edge Family (edge mask) and E/H Sensitivity Grid
    - An optional boolean, symmetric `N × N` `.npy` **edge family mask** restricts which edges enter the tested family (TFNBS, NI-TFNBS, NBS, t-stat). It is not an ROI exclusion subset and not an SC community label vector; the run plan records its purpose, SHA-256, shape and eligible-edge count, and the array is bundled with the archive and saved workspace.
    - The TFNBS family can evaluate an **(E, H) sensitivity grid** (E and H lists of equal length, paired element-wise). The **primary cell is declared before the run**; only it carries the confirmatory claim. Other cells are descriptive sensitivity: the grid creates no corrected family-wise conclusion for a cell chosen after seeing the data, there is no p-based automatic best-cell choice, and browsing cells does not change the declared primary in the run plan or in an export.
    - Not offered in the guided interface: arbitrary F / multi-row contrasts, interaction terms and EEG-style data. Use the Python API (`compute_p_val_glm(..., stat_type='fstat')` or `analyze(..., interactions=[...])`); `docs/source/usage.rst` has a worked example.
    
    #### Enhancement Operators Quick-Reference
    - **tstat:** Max-stat FWER on raw edge statistics.
    - **bh_fdr / bonferroni:** Standard edge-wise multiple comparison correction.
    - **nbs:** Fixed-threshold connected-component mass (Zalesky et al., 2010).
    - **tfnbs:** Threshold-free connected-component integration (Baggio et al., 2018; Hao et al., 2024).
    - **cnbs:** Constrained NBS (Noble & Scheinost, 2020): mean signed t per network block, tested per block (Bonferroni across blocks); the inference unit is the block.
    - **ni_tfnbs / fbc_tfnbs:** Block prior informed network threshold-free statistics.
    
    #### Scientific Caveats on BrainMap Profiles
    Each Brainnetome region carries the Behavioral Domain and Paradigm Class profiles published with the atlas (BrainMap forward inference, FDR < 0.05; Fan et al., 2016; Laird et al., 2005, 2011). A term such as `'Cognition.Memory.Working'` means that tasks of that kind activate the region in the literature. It is context for the region, **not** a function of the region or of a detected connection, and it adds no test: the p-values and the inference unit are unchanged. Report the statistical result first and treat the profiles as literature context.
    """)
