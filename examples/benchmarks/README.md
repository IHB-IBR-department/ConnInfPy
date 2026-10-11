# Performance and external validation

| Location | Purpose | Entry points |
|---|---|---|
| `performance/` | Timing, GLM execution, precomputed sums and tail acceleration | `timing_benchmark.py`, `benchmark_glm.py`, `benchmark_precompsum.py`, `benchmark_acceleration.py` |
| `external_validation/` | ComBat equivalence/calibration, full-pipeline and structural-prior checks, PALM comparisons | `validation_*.py` |
| `_shared/data.py` | Synthetic data adapters used across benchmarks | Imported support module |
| `res_benchmarks/` | Existing CSV/PNG output location | `performance/plot_results.py` reads it |

```bash
python -m examples.benchmarks.performance.timing_benchmark --help
python -m examples.benchmarks.performance.benchmark_precompsum --help
python -m examples.benchmarks.performance.plot_results --help
```

Performance defaults resolve to this repository; an explicit output argument overrides them. External-validation modules describe their optional dependencies and execution cost (for example PALM/MATLAB, neuroCombat or ENIGMA). They are not run by importing the modules. Timing depends on hardware, thread settings, graph size and permutation budget; correctness comparisons and speed measurements answer separate questions.
