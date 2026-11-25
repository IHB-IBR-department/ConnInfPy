# ConnInfPy examples

Runnable examples are organized into two layers: **user-facing tutorials/recipes** for learning the package API, and **paper/validation pipelines** for reproducing the manuscript studies. Install the package in the active Python environment (`uv pip install -e .` or `python -m pip install -e .`). Run the module commands below from the repository root; direct script paths also resolve their defaults independently of the working directory.

## User examples and recipes

Start here when learning `conninfpy` for your own fMRI connectome data:

| Location | Purpose | Start here |
|---|---|---|
| [notebooks/](notebooks/) | Interactive tutorials (01–12) | `01_quickstart_ttest.ipynb` through `12_v2_new_features.ipynb` |
| [plot_coupling_classes.py](plot_coupling_classes.py) | Standalone coupling-change matrix & homotopy recipe | `python examples/plot_coupling_classes.py` |
| [post_analysis_bundle.py](post_analysis_bundle.py) | Save, reload, and summarize a `ResultBundle` | `python examples/post_analysis_bundle.py` |

## Paper and validation pipelines

Research scripts used to reproduce calibration, power, empirical case studies, and timing benchmarks:

| Folder | Purpose | Start here |
|---|---|---|
| [open_close/](open_close/) | Paired state contrasts, atlas stability and interpretation; separate existing ML studies | Its workflow table |
| [abide/](abide/) | Observational multi-site age and diagnosis contrasts | Data preparation, then `age/` or `diagnosis/` |
| [simulations/](simulations/) | Calibration, power and sensitivity under specified generative models | Explicit YAML configurations |
| [benchmarks/performance/](benchmarks/performance/) | Timing and scaling | Benchmark CLI help |
| [benchmarks/external_validation/](benchmarks/external_validation/) | Agreement with external implementations and bounded validation studies | Optional dependencies listed in each module |

Each study's `_shared/` contains data adapters and support functions. `examples/_shared/` contains repository paths and small artifact utilities. These are study code, separate from the installed `conninfpy` library.

Each study has one `results/` folder with a subfolder per analysis, named like the code folder that writes it; Brainnetome interpretation goes inside the analysis it describes (e.g. `abide/results/diagnosis/brainnetome/`). The study READMEs show the layout. Generated results remain ignored; manuscript figures are built into `paper/figures/` by `paper/build_fig1_pipeline.py` and `paper/build_manuscript_figures.py`. Saved configurations and source hashes describe the run that created them and are retained verbatim.

Simulation calibration is conditional on the generative model and tested settings. Paired Open-Close comparisons improve control of between-person variation; they do not establish causality. ABIDE diagnosis contrasts remain observational. Method agreement, bootstrap stability and cross-cohort replication measure different properties.

The maintained paper reproduction entry point is `bash paper/reproduce_figures.sh`. Full research runs can take hours. Inspect `--help` and use bounded synthetic checks for code validation.
