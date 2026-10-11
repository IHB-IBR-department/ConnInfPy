# ConnInfPy examples

Runnable examples are organized by purpose. Install the package in the active Python environment (`python -m pip install -e .`). Run the module commands below from the repository root; direct script paths also resolve their defaults independently of the working directory.

| Folder | Purpose | Start here |
|---|---|---|
| [notebooks/](notebooks/) | Interactive tutorials | Choose a notebook for paired tests, GLM or enhancement |
| [open_close/](open_close/) | Paired state contrasts, atlas stability and interpretation; separate existing ML studies | Its workflow table |
| [abide/](abide/) | Observational multi-site age and diagnosis contrasts | Data preparation, then age/ or diagnosis/ |
| [simulations/](simulations/) | Calibration, power and sensitivity under specified generative models | Explicit YAML configurations |
| [benchmarks/performance/](benchmarks/performance/) | Timing and scaling | Benchmark CLI help |
| [benchmarks/external_validation/](benchmarks/external_validation/) | Agreement with external implementations and bounded validation studies | Optional dependencies listed in each module |

Each study's `_shared/` contains data adapters and support functions. `examples/_shared/` contains repository paths and small artifact utilities. These are study code, separate from the installed `conninfpy` library.

Generated results remain ignored. Shared publication plots stay at `abide/results/plots/` and `open_close/results/plots/`; benchmark outputs stay at `benchmarks/res_benchmarks/`. Saved configurations and source hashes describe the run that created them and are retained verbatim.

Simulation calibration is conditional on the generative model and tested settings. Paired Open-Close comparisons improve control of between-person variation; they do not establish causality. ABIDE diagnosis contrasts remain observational. Method agreement, bootstrap stability and cross-cohort replication measure different properties.

The maintained paper reproduction entry point is `bash paper/reproduce_figures.sh`. Full research runs can take hours. Inspect `--help` and use bounded synthetic checks for code validation.
