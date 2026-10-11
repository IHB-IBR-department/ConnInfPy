"""
Null FWER Calibration Test for TFNBS Methods.

Validates that statistical methods maintain nominal family-wise error
rate (FWER) under the null hypothesis (no true effect).

Protocol:
  1. Generate N null datasets (effect_size=0)
  2. Run each method with full permutation testing
  3. Check whether any edge is significant in each directional tail
  4. Verify directional FWER <= alpha within Monte-Carlo uncertainty

ConnInfPy reports separate positive and negative p-value maps. The
primary calibration target here is therefore directional FWER:

    P(any false-positive edge in one tail | H0) <= alpha

If a user wants one undirected/two-sided family across both tails using
the existing per-tail maps, they should threshold each tail at alpha / 2.
This script reports that Bonferroni-over-tails diagnostic separately.
The unadjusted "either tail at alpha" value is retained only as a
diagnostic and is expected to approach <= 2*alpha, not alpha.

Supports sweeping over multiple sample sizes to verify calibration
holds across different sample sizes (like power_analysis.py).

Methods tested:
  tstat, tfnbs, ni_tfnbs, fbc_tfnbs, nbs_extent_2.0, nbs_intensity_2.0,
  nbs_extent_3.0, nbs_intensity_3.0, cnbs, bonferroni, bh_fdr

Usage:
    python fpr_calibration.py --config fpr_config_local.yaml
    python fpr_calibration.py --config fpr_config.yaml --resume
    python fpr_calibration.py --methods tstat tfnbs cnbs --n-null 100

Output:
    <output_dir>/
        fpr_results.csv          # All null-run results (combined)
        fpr_summary.csv          # Per-method (per-n_samples) FWER summary
        fpr_summary.txt          # Human-readable summary
        checkpoint.csv           # For --resume (removed on success)
        per_method/
            fpr_results_<method>.csv         # Single n_samples mode
            fpr_results_<method>_n<N>.csv    # Multi n_samples mode

Plotting (separate script):
    python plot_fpr_results.py <output_dir>
"""

from __future__ import annotations


from pathlib import Path as _Path
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from examples._shared.paths import ROOT, config_path
import os

# Pin each worker to 1 BLAS/LAPACK thread (must be before numpy import).
_SINGLE_THREAD = "1"
os.environ["OMP_NUM_THREADS"] = _SINGLE_THREAD
os.environ["OPENBLAS_NUM_THREADS"] = _SINGLE_THREAD
os.environ["MKL_NUM_THREADS"] = _SINGLE_THREAD
os.environ["VECLIB_MAXIMUM_THREADS"] = _SINGLE_THREAD
os.environ["NUMBA_NUM_THREADS"] = _SINGLE_THREAD

import argparse
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Add repo root to path for imports

from conninfpy.stats.pairwise import compute_p_val, get_available_cores
from conninfpy.stats._workers import _limit_worker_threads
from conninfpy.sim.datasets import ModularDatasetGenerator
from examples.simulations._shared.calibration import clopper_pearson_ci, _ci_not_liberal, _binom_p_above, simultaneous_calibration_test, compute_fpr_summary


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FPRResult:
    """Result from a single null simulation."""

    method: str
    null_run_id: int
    n_samples: int             # Samples per group used in this run
    n_edges: int               # Total edges tested (upper triangle)
    # Positive direction (g2 > g1)
    any_significant_pos: bool  # Any p <= alpha in positive tail (directional FWER)
    n_significant_pos: int     # Count of significant edges
    fpr_pos: float             # n_significant_pos / n_edges (edge-wise FPR)
    min_p_pos: float
    # Negative direction (g1 > g2)
    any_significant_neg: bool
    n_significant_neg: int
    fpr_neg: float             # n_significant_neg / n_edges
    min_p_neg: float
    # Combined diagnostic: either direction at unadjusted alpha.
    # This is not a two-sided FWER-alpha test; use alpha/2 per tail for that.
    any_significant: bool      # Either direction has any significant at alpha
    n_significant_total: int   # Total significant edges (both directions)
    fpr_total: float           # Combined edge-wise FPR
    elapsed_time: float


@dataclass
class FPRConfig:
    """Configuration for null FWER calibration run."""

    # Methods to test
    methods: List[str]

    # Null simulation parameters
    n_null: int = 100
    n_permutations: int = 500
    alpha: float = 0.05

    # Data generation parameters
    n_nodes: int = 60
    n_modules: int = 4
    n_samples: int = 20
    sample_sizes: Optional[List[int]] = None  # Sweep over multiple sample sizes
    intra_corr: float = 0.3
    inter_corr: float = 0.05
    noise_level: float = 0.05
    time_points: int = 30       # same as the power simulations

    # Method-specific parameters
    tfnbs_e: float = 0.4
    tfnbs_h: float = 3.0
    tfnbs_n: int = 10
    tfnbs_start_thres: float = 1.65
    nbs_threshold: float = 2.0
    fbc_min_cluster: int = 3

    # Execution options
    seed: int = 42
    use_mp: bool = False
    n_jobs: int = -1            # -1 = use all cores for parallel null runs
    checkpoint_every: int = 10  # Save checkpoint every N runs per method

    # Output
    output_dir: Path = (ROOT / "examples/simulations/results/fpr_calibration")

    @property
    def effective_sample_sizes(self) -> List[int]:
        """Return sample_sizes list if set, else [n_samples]."""
        if self.sample_sizes is not None:
            return self.sample_sizes
        return [self.n_samples]

    @classmethod
    def from_yaml(cls, path: Path) -> "FPRConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Handle output_dir as Path
        if "output_dir" in data:
            data["output_dir"] = config_path(data["output_dir"], path)

        return cls(**data)

    def get_method_kwargs(self, method: str) -> dict:
        """Get method-specific kwargs."""
        nbs_kwargs = self._parse_nbs_method(method)
        if nbs_kwargs is not None:
            return nbs_kwargs

        configs = {
            "tstat": {},
            "tfnbs": {
                "e": self.tfnbs_e,
                "h": self.tfnbs_h,
                "n": self.tfnbs_n,
                "start_thres": self.tfnbs_start_thres,
            },
            "ni_tfnbs": {
                "e": self.tfnbs_e,
                "h": self.tfnbs_h,
                "n": self.tfnbs_n,
                "start_thres": self.tfnbs_start_thres,
            },
            "fbc_tfnbs": {
                "e": self.tfnbs_e,
                "h": self.tfnbs_h,
                "n": self.tfnbs_n,
                "start_thres": self.tfnbs_start_thres,
                "min_cluster_size": self.fbc_min_cluster,
            },
            "cnbs": {},
            "bonferroni": {},
            "bh_fdr": {},
        }
        return configs.get(method, {})

    def get_compute_method(self, method: str) -> str:
        """Map display method name to compute_p_val method name."""
        if self._parse_nbs_method(method) is not None:
            return "nbs"
        return method

    def _parse_nbs_method(self, method: str) -> Optional[dict]:
        """Parse NBS method aliases with optional threshold suffix.

        Supported names:
        - nbs_extent, nbs_intensity: use ``self.nbs_threshold``
        - nbs_extent_2.0, nbs_extent_3.0, nbs_intensity_2.0, ...
        """
        prefixes = {
            "nbs_extent": "extent",
            "nbs_intensity": "intensity",
        }
        for prefix, stat in prefixes.items():
            if method == prefix:
                return {"threshold": self.nbs_threshold, "nbs_stat": stat}
            marker = prefix + "_"
            if method.startswith(marker):
                threshold_text = method[len(marker):]
                try:
                    threshold = float(threshold_text)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid NBS threshold suffix in method {method!r}."
                    ) from exc
                return {"threshold": threshold, "nbs_stat": stat}
        return None


# =============================================================================
# NULL DATA GENERATION
# =============================================================================

def generate_null_dataset(
    n_nodes: int = 60,
    n_modules: int = 4,
    n_samples: int = 20,
    intra_corr: float = 0.3,
    inter_corr: float = 0.05,
    noise_level: float = 0.05,
    time_points: int = 30,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a null dataset: both groups share one population correlation."""

    generator = ModularDatasetGenerator(
        N=n_nodes,  # ModularDatasetGenerator uses 'N' for number of nodes
        n_modules=n_modules,
        intra_corr=intra_corr,
        inter_corr=inter_corr,
        noise_level=noise_level,
        seed=seed,
    )

    # effect size = 0 (i.e. no true difference)
    # effect_mask must be provided but effect_size=0 means no effect
    effect_mask = np.zeros((n_nodes, n_nodes))
    group1, group2, net_labels = generator.generate_data(
        effect_mask=effect_mask,
        effect_size=0.0,  # NULL hypothesis
        n_samples_g1=n_samples,
        n_samples_g2=n_samples,
        time_points=time_points,
    )

    return group1, group2, net_labels


# =============================================================================
# SINGLE NULL TEST
# =============================================================================

def run_single_null_test(
    method: str,
    null_run_id: int,
    n_samples: int,
    config: FPRConfig,
) -> FPRResult:
    """Run a single null hypothesis test for one method."""

    start_time = time.time()

    # Generate null data with unique seed (incorporate n_samples to avoid
    # identical datasets across different sample sizes)
    seed = config.seed + null_run_id * 1000 + n_samples
    group1, group2, net_labels = generate_null_dataset(
        n_nodes=config.n_nodes,
        n_modules=config.n_modules,
        n_samples=n_samples,
        intra_corr=config.intra_corr,
        inter_corr=config.inter_corr,
        noise_level=config.noise_level,
        time_points=config.time_points,
        seed=seed,
    )

    # Apply Fisher z-transform
    group1_z = np.arctanh(np.clip(group1, -0.999, 0.999))
    group2_z = np.arctanh(np.clip(group2, -0.999, 0.999))

    # Prepare method kwargs
    kwargs = config.get_method_kwargs(method)
    compute_method = config.get_compute_method(method)

    if compute_method in ["cnbs", "ni_tfnbs", "fbc_tfnbs"]:
        kwargs["net_labels"] = net_labels

    # Number of edges in upper triangle
    n_edges = config.n_nodes * (config.n_nodes - 1) // 2

    # Compute p-values (an error stops the run: recording it as "nothing
    # significant" would bias the FWER estimate downwards)
    p_vals = compute_p_val(
        group1_z, group2_z,
        test_type="two-sample",
        method=compute_method,
        n_permutations=config.n_permutations,
        use_mp=config.use_mp,
        random_state=seed,
        **kwargs,
    )

    p_pos = np.array(p_vals["g2>g1"])
    p_neg = np.array(p_vals["g1>g2"])

    # Get upper triangle only (symmetric matrix)
    triu_idx = np.triu_indices(config.n_nodes, k=1)
    p_pos_upper = p_pos[triu_idx]
    p_neg_upper = p_neg[triu_idx]

    # Positive direction metrics
    sig_mask_pos = p_pos_upper <= config.alpha
    any_sig_pos = bool(np.any(sig_mask_pos))
    n_sig_pos = int(np.sum(sig_mask_pos))
    fpr_pos = n_sig_pos / n_edges
    min_p_pos = float(np.min(p_pos_upper))

    # Negative direction metrics
    sig_mask_neg = p_neg_upper <= config.alpha
    any_sig_neg = bool(np.any(sig_mask_neg))
    n_sig_neg = int(np.sum(sig_mask_neg))
    fpr_neg = n_sig_neg / n_edges
    min_p_neg = float(np.min(p_neg_upper))

    # Combined metrics (both directions)
    n_sig_total = n_sig_pos + n_sig_neg
    fpr_total = n_sig_total / (2 * n_edges)  # Denominator is 2*n_edges for two-sided

    elapsed = time.time() - start_time

    return FPRResult(
        method=method,
        null_run_id=null_run_id,
        n_samples=n_samples,
        n_edges=n_edges,
        any_significant_pos=any_sig_pos,
        n_significant_pos=n_sig_pos,
        fpr_pos=fpr_pos,
        min_p_pos=min_p_pos,
        any_significant_neg=any_sig_neg,
        n_significant_neg=n_sig_neg,
        fpr_neg=fpr_neg,
        min_p_neg=min_p_neg,
        any_significant=any_sig_pos or any_sig_neg,
        n_significant_total=n_sig_total,
        fpr_total=fpr_total,
        elapsed_time=elapsed,
    )


def _run_null_test_wrapper(args: tuple) -> FPRResult:
    """Wrapper for parallel execution."""
    method, null_run_id, n_samples, config = args
    return run_single_null_test(method, null_run_id, n_samples, config)


# =============================================================================
# CHECKPOINTING
# =============================================================================

def save_checkpoint(results: List[FPRResult], checkpoint_path: Path) -> None:
    """Save intermediate results to checkpoint file."""
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(checkpoint_path, index=False)


def load_checkpoint(
    checkpoint_path: Path,
    default_n_samples: int = 0,
) -> Tuple[List[FPRResult], set]:
    """Load results from checkpoint and return completed (method, n_samples, run_id) keys."""
    if not checkpoint_path.exists():
        return [], set()

    df = pd.read_csv(checkpoint_path)
    results = []
    completed = set()

    # Handle old checkpoints without n_samples column
    has_n_samples = "n_samples" in df.columns

    for _, row in df.iterrows():
        n_samples = int(row["n_samples"]) if has_n_samples else default_n_samples
        result = FPRResult(
            method=row["method"],
            null_run_id=int(row["null_run_id"]),
            n_samples=n_samples,
            n_edges=int(row["n_edges"]),
            any_significant_pos=bool(row["any_significant_pos"]),
            n_significant_pos=int(row["n_significant_pos"]),
            fpr_pos=float(row["fpr_pos"]),
            min_p_pos=float(row["min_p_pos"]),
            any_significant_neg=bool(row["any_significant_neg"]),
            n_significant_neg=int(row["n_significant_neg"]),
            fpr_neg=float(row["fpr_neg"]),
            min_p_neg=float(row["min_p_neg"]),
            any_significant=bool(row["any_significant"]),
            n_significant_total=int(row["n_significant_total"]),
            fpr_total=float(row["fpr_total"]),
            elapsed_time=float(row["elapsed_time"]),
        )
        results.append(result)
        completed.add((row["method"], n_samples, int(row["null_run_id"])))

    return results, completed


# =============================================================================
# MAIN CALIBRATION RUNNER
# =============================================================================

def run_fpr_calibration(
    config: FPRConfig,
    resume: bool = False,
) -> pd.DataFrame:
    """Run full null-FWER calibration with optional parallelization/checkpointing."""

    checkpoint_path = config.output_dir / "checkpoint.csv"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    sample_sizes = config.effective_sample_sizes

    # Load checkpoint if resuming
    if resume:
        # A completed run has removed its checkpoint; resume from its results
        finished = config.output_dir / "fpr_results.csv"
        source = checkpoint_path if checkpoint_path.exists() else finished
        results, completed = load_checkpoint(
            source, default_n_samples=sample_sizes[0]
        )
        print(f"Resuming from checkpoint: {len(completed)} runs already completed")
    else:
        results = []
        completed = set()

    # Build task list
    tasks = []
    for method in config.methods:
        for n_samples in sample_sizes:
            for null_id in range(config.n_null):
                if (method, n_samples, null_id) not in completed:
                    tasks.append((method, null_id, n_samples, config))

    if not tasks:
        print("All runs already completed!")
        return pd.DataFrame([asdict(r) for r in results])

    total_runs = len(tasks)
    n_cores = min(config.n_jobs if config.n_jobs > 0 else get_available_cores(), total_runs)
    use_parallel = n_cores > 1

    print(f"=== Null FWER Calibration Test ===")
    print(f"Methods: {config.methods}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Null datasets per (method, n_samples): {config.n_null}")
    print(f"Permutations per test: {config.n_permutations}")
    print(f"Alpha: {config.alpha}")
    print(f"Parallel: {'yes' if use_parallel else 'no'} ({n_cores} cores)")
    print(f"Remaining runs: {total_runs}")
    print(f"Expected directional FWER: <= {config.alpha:.3f} (Monte-Carlo half-width ~"
          f"{1.96 * np.sqrt(config.alpha * (1-config.alpha) / config.n_null):.3f})")
    print(f"Two-sided diagnostic: threshold each tail at alpha/2 = {config.alpha / 2:.3f}")
    print()

    if use_parallel:
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_cores, initializer=_limit_worker_threads) as pool:
            pbar = tqdm(
                pool.imap_unordered(_run_null_test_wrapper, tasks),
                total=total_runs,
                desc="FWER calibration",
                unit="run",
                miniters=1,
            )
            for i, result in enumerate(pbar, 1):
                results.append(result)
                pbar.set_postfix_str(
                    f"{result.method} n={result.n_samples} #{result.null_run_id}"
                )

                if i % config.checkpoint_every == 0 or i == total_runs:
                    save_checkpoint(results, checkpoint_path)
    else:
        pbar = tqdm(tasks, desc="FWER calibration", unit="run", miniters=1)
        for i, task in enumerate(pbar, 1):
            result = _run_null_test_wrapper(task)
            results.append(result)
            pbar.set_postfix_str(
                f"{result.method} n={result.n_samples} #{result.null_run_id}"
            )

            if i % config.checkpoint_every == 0 or i == total_runs:
                save_checkpoint(results, checkpoint_path)

    # Final checkpoint save
    save_checkpoint(results, checkpoint_path)

    return pd.DataFrame([asdict(r) for r in results])


# =============================================================================
# STATISTICAL SUMMARY
# =============================================================================











# =============================================================================
# PER-METHOD CSV SAVING
# =============================================================================

def save_per_method(df: pd.DataFrame, output_dir: Path) -> None:
    """Save one CSV per method (n_samples stays as a column)."""
    method_dir = output_dir / "per_method"
    method_dir.mkdir(parents=True, exist_ok=True)
    for method, mdf in df.groupby("method"):
        path = method_dir / f"fpr_results_{method}.csv"
        mdf.to_csv(path, index=False)
    print(f"  Saved {df['method'].nunique()} per-method files to {method_dir}/")


# =============================================================================
# CLI params
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Null FWER calibration test for TFNBS methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config", type=Path,
        help="YAML configuration file",
    )
    parser.add_argument(
        "--n-null", type=int, default=None,
        help="Number of null datasets (overrides config)",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=None,
        help="Permutations per test (overrides config)",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Methods to test (overrides config)",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Significance threshold (overrides config)",
    )
    parser.add_argument(
        "--n-nodes", type=int, default=None,
        help="Number of nodes (overrides config)",
    )
    parser.add_argument(
        "--n-modules", type=int, default=None,
        help="Number of modules (overrides config)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Samples per group (overrides config, single value)",
    )
    parser.add_argument(
        "--sample-sizes", nargs="+", type=int, default=None,
        help="Sweep over multiple sample sizes (overrides config)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Base random seed (overrides config)",
    )
    parser.add_argument(
        "--use-mp", action="store_true",
        help="Use multiprocessing for permutations within each test",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=None,
        help="Number of parallel jobs for null runs (-1=all cores, 1=sequential)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if available",
    )

    args = parser.parse_args(argv)

    # Load config from file or use defaults
    if args.config:
        config = FPRConfig.from_yaml(args.config)
    else:
        # Default configuration
        config = FPRConfig(
            methods=["tstat", "tfnbs", "ni_tfnbs", "fbc_tfnbs",
                     "nbs_extent_2.0", "nbs_intensity_2.0",
                     "nbs_extent_3.0", "nbs_intensity_3.0", "cnbs",
                     "bonferroni", "bh_fdr"],
        )

    # Override config with CLI arguments
    if args.n_null is not None:
        config.n_null = args.n_null
    if args.n_permutations is not None:
        config.n_permutations = args.n_permutations
    if args.methods is not None:
        config.methods = args.methods
    if args.alpha is not None:
        config.alpha = args.alpha
    if args.n_nodes is not None:
        config.n_nodes = args.n_nodes
    if args.n_modules is not None:
        config.n_modules = args.n_modules
    if args.n_samples is not None:
        config.n_samples = args.n_samples
    if args.sample_sizes is not None:
        config.sample_sizes = args.sample_sizes
    if args.seed is not None:
        config.seed = args.seed
    if args.use_mp:
        config.use_mp = True
    if args.n_jobs is not None:
        config.n_jobs = args.n_jobs
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Run calibration
    df = run_fpr_calibration(config, resume=args.resume)

    # Save results
    df.to_csv(config.output_dir / "fpr_results.csv", index=False)
    print(f"\nSaved results to {config.output_dir / 'fpr_results.csv'}")

    # Save per-method CSVs
    save_per_method(df, config.output_dir)

    # Compute and save summary
    summary_df = compute_fpr_summary(df, alpha=config.alpha)
    summary_df.to_csv(config.output_dir / "fpr_summary.csv", index=False)

    # Print summary
    sample_sizes = config.effective_sample_sizes
    multi_n = len(sample_sizes) > 1

    print("\n" + "=" * 80)
    print("FWER CALIBRATION SUMMARY")
    print("=" * 80)
    print(f"Alpha = {config.alpha}")
    print(f"Two-sided per-tail threshold = {config.alpha / 2:.4f}")
    print(f"N edges = {summary_df['n_edges'].iloc[0]}")
    print(f"Sample sizes = {sample_sizes}")
    print()

    if multi_n:
        header = (f"{'Method':<15} {'n_samples':>9} {'Max-tail':>9} "
                  f"{'95% CI':>20} {'2-sided':>9} {'Either@a':>9} {'Status':>12}")
    else:
        header = (f"{'Method':<15} {'Max-tail':>9} "
                  f"{'95% CI':>20} {'2-sided':>9} {'Either@a':>9} {'Status':>12}")
    print(header)
    print("-" * 80)

    for _, row in summary_df.iterrows():
        fwer_ok = row["directional_fwer_valid"] and row["fwer_two_sided_bonf_valid"]
        status = "OK" if fwer_ok else "CI > alpha"
        ci_str = f"[{row['fwer_ci_low']:.3f}, {row['fwer_ci_high']:.3f}]"
        if multi_n:
            print(f"  {row['method']:<13} {int(row['n_samples']):>9} "
                  f"{row['fwer_max_tail']:>9.4f} {ci_str:>20} "
                  f"{row['fwer_two_sided_bonf']:>9.4f} "
                  f"{row['fwer_either_at_alpha']:>9.4f} {status:>12}")
        else:
            print(f"  {row['method']:<13} {row['fwer_max_tail']:>9.4f} {ci_str:>20} "
                  f"{row['fwer_two_sided_bonf']:>9.4f} "
                  f"{row['fwer_either_at_alpha']:>9.4f} {status:>12}")

    test = simultaneous_calibration_test(summary_df, alpha=config.alpha)
    n_ci_above = int((~(summary_df["directional_fwer_valid"] & summary_df["fwer_two_sided_bonf_valid"])).sum())
    verdict_lines = [
        f"Cells whose 95% CI lies above alpha: {n_ci_above} of {len(summary_df)} "
        "(descriptive; some are expected by chance)",
    ]
    for name, label in (("directional", "Directional FWER (each tail at alpha)"),
                        ("two_sided", "Two-sided FWER (alpha/2 per tail)")):
        t = test[name]
        verdict_lines.append(
            f"{label}: smallest exact binomial p = {t['min_p']:.4f}, Bonferroni over "
            f"{t['n_tests']} tests = {t['p_bonferroni']:.3f} -> {'PASS' if t['pass'] else 'FAIL'}")
    verdict_lines.append(f"OVERALL (simultaneous test): {'PASS' if test['pass'] else 'FAIL'}")

    # Save text summary
    with open(config.output_dir / "fpr_summary.txt", "w") as f:
        f.write("FWER Calibration Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write("Parameters:\n")
        f.write(f"  n_null_runs = {config.n_null}\n")
        f.write(f"  n_permutations = {config.n_permutations}\n")
        f.write(f"  alpha = {config.alpha}\n")
        f.write(f"  two_sided_alpha_per_tail = {config.alpha / 2:.6f}\n")
        f.write(f"  n_nodes = {config.n_nodes}\n")
        f.write(f"  n_edges = {summary_df['n_edges'].iloc[0]}\n")
        f.write(f"  sample_sizes = {sample_sizes}\n")
        f.write(f"  time_points = {config.time_points}\n")
        f.write(f"  tfnbs: e = {config.tfnbs_e}, h = {config.tfnbs_h}, "
                f"n = {config.tfnbs_n}, start_thres = {config.tfnbs_start_thres}\n\n")

        f.write("Metrics Explained:\n")
        f.write("  Directional FWER: P(any FP in one tail | H0). This is the\n")
        f.write("        primary ConnInfPy target because positive and negative tails\n")
        f.write("        use separate max-statistic nulls.\n")
        f.write("  Max-tail FWER: max(FWER positive, FWER negative); should be <= alpha.\n")
        f.write("  2-sided Bonferroni FWER: P(any FP in either tail at alpha/2 per tail);\n")
        f.write("        should be <= alpha for undirected/two-sided claims.\n")
        f.write("  Either@alpha: P(any FP in either tail at unadjusted alpha); diagnostic\n")
        f.write("        only, expected upper bound is <= 2*alpha, not alpha.\n")
        f.write("  Edge FPR: Mean proportion of FP edges; descriptive and usually << alpha\n")
        f.write("        for FWER-controlling methods.\n\n")

        if multi_n:
            hdr = (f"{'Method':<15} {'n_samples':>9} {'Max-tail':>9} "
                   f"{'95% CI':>22} {'2-sided':>9} {'Either@a':>9} {'Cal.':>10}\n")
        else:
            hdr = (f"{'Method':<15} {'Max-tail':>9} "
                   f"{'95% CI':>22} {'2-sided':>9} {'Either@a':>9} {'Cal.':>10}\n")
        f.write(hdr)
        f.write("-" * 80 + "\n")
        for _, row in summary_df.iterrows():
            status = "OK" if (row["directional_fwer_valid"] and row["fwer_two_sided_bonf_valid"]) else "CI>alpha"
            ci_str = f"[{row['fwer_ci_low']:.4f}, {row['fwer_ci_high']:.4f}]"
            if multi_n:
                f.write(f"  {row['method']:<13} {int(row['n_samples']):>9} "
                        f"{row['fwer_max_tail']:>9.4f} {ci_str:>22} "
                        f"{row['fwer_two_sided_bonf']:>9.4f} "
                        f"{row['fwer_either_at_alpha']:>9.4f} {status:>10}\n")
            else:
                f.write(f"  {row['method']:<13} {row['fwer_max_tail']:>9.4f} {ci_str:>22} "
                        f"{row['fwer_two_sided_bonf']:>9.4f} "
                        f"{row['fwer_either_at_alpha']:>9.4f} {status:>10}\n")
        f.write("\n" + "\n".join(verdict_lines) + "\n")

    print()
    print("=" * 80)
    print("\n".join(verdict_lines))
    print("=" * 80)

    # The run is complete (results saved); the verdict is a result, not an error
    checkpoint_path = config.output_dir / "checkpoint.csv"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Removed checkpoint file (run completed)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
