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
    - **Paired contrast:** Subject-wise difference contrast.
    - **GLM:** Evaluate variable of interest while regressing out nuisance covariates.
    
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
