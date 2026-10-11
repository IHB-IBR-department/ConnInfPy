# ConnInfPy Examples & Validation

This directory contains the tutorials, paper validation pipelines, and performance benchmarks for the `conninfpy` package.

## Audience Routing

- **New to ConnInfPy?** Start with the **[notebooks/](./notebooks/)** for interactive tutorials on t-tests, GLM inference, and enhancement methods.
- **Evaluating Paper Claims?** See the validation folders below. We recommend following the **Evidence Hierarchy** order.
- **Performance Tuning?** See **[benchmarks/](./benchmarks/)** for acceleration and GLM timing tests.

## Evidence Hierarchy (Validation Folders)

The validation of `conninfpy` is organized by the strength of the statistical design, moving from controlled simulations to observational multi-site data:

1.  **[simulations/](./simulations/)**: **Null Calibration (FPR)** and Power. The foundation of belief; demonstrates that the package controls Type I error under null data.
2.  **[abide/](./abide/)**: **Positive Controls (Age, Motion)**. Demonstrates that the pipeline reliably detects strong, well-replicated developmental and artifactual signals in real fMRI data.
3.  **[open_close/](./open_close/)**: **Paired Within-Subject (Open vs. Closed Eyes)**. The strongest causal-adjacent real-data design, utilizing exact sign-flip permutations.
4.  **[abide/](./abide/)**: **Observational Case-Control (ASD vs. HC)**. The final stress test involving multi-site harmonization (ComBat Strategy D) and stratified permutations.

---

## Directory Overview

One folder per dataset or purpose:

| Folder | Contents |
|---|---|
| `abide/` | ABIDE I multi-site GLM validation (age, diagnosis, motion, within-site replication); `configs/` (CLI analysis files); `brainnetome_interpretation.py` |
| `open_close/` | Open-Close (eyes closed vs open; IHB and Beijing) paired validation: loader, paired TFNBS, retest, ML transfer (`ml/`), `brainnetome_interpretation.py`, and the experiments `atlas_stability/` (Schaefer vs native Brainnetome), `sc_inference/` (structural-connectivity mask and NI-TFNBS blocks), `prediction/` (edge-model feature reproducibility) |
| `simulations/` | FPR calibration, power analysis, sensitivity audits, structural-prior simulations |
| `benchmarks/` | Scalability, timing and external-validation benchmarks |
| `notebooks/` | Jupyter tutorials for getting started |

`_bna_interpretation.py` holds the steps shared by the two `brainnetome_interpretation.py`
scripts. Every folder writes regenerable outputs to its own `results/` (git-ignored, except
the paper plots).
