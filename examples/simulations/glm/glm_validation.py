"""
GLM validation on synthetic connectivity data.

This script adds two GLM-specific validation arms for the toolbox paper:

1. Null FWER calibration for a continuous tested regressor, optionally with a
   nuisance covariate effect included in the data and adjusted in the GLM.
2. Power analysis for a continuous covariate with effects planted in known
   topological edge sets.

The goal is methodological validation of the GLM API, not a new neuroscience
claim. Outputs mirror the existing simulation validation CSV style so the same
calibration logic and downstream plotting conventions can be reused.

Usage:
    python examples/simulations/glm/glm_validation.py \
        --config examples/simulations/configs/glm_config_quick.yaml

    python examples/simulations/glm/glm_validation.py \
        --config examples/simulations/configs/glm_config_1000.yaml \
        --mode calibration

    ConnInfPy 2.0.0 calibration (exact TFNBS integration, one two-sided family):

    python examples/simulations/glm/glm_validation.py \
        --config examples/simulations/configs/glm_config_2_0_1000.yaml --resume

``--resume`` skips every (method, n_samples, null_run_id) cell and every
(method, scenario, effect_size, repeat_id) power cell already present in the
per-run checkpoint CSV (or in the finished results CSV); ``tfnbs_n: null`` in
YAML or ``--tfnbs-n none`` selects exact integration and ``--one-tailed`` the
legacy per-tail families (default ``two_tailed=True``: the 2.0.0 contract).
"""

from __future__ import annotations


from pathlib import Path as _Path
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from examples._shared.paths import ROOT, config_path
import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SINGLE_THREAD = "1"
os.environ.setdefault("OMP_NUM_THREADS", _SINGLE_THREAD)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _SINGLE_THREAD)
os.environ.setdefault("MKL_NUM_THREADS", _SINGLE_THREAD)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _SINGLE_THREAD)
os.environ.setdefault("NUMBA_NUM_THREADS", _SINGLE_THREAD)

import numpy as np
import pandas as pd

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from tqdm import tqdm

from conninfpy.sim import topologies as topo
from conninfpy.stats.glm import compute_p_val_glm
from conninfpy.stats.pairwise import get_available_cores
from conninfpy.stats._workers import _limit_worker_threads

SIM_ROOT = Path(__file__).resolve().parents[1]
from examples.simulations._shared.calibration import compute_fpr_summary, clopper_pearson_ci  # noqa: E402


GLM_METHODS = [
    "tstat",
    "tfnbs",
    "ni_tfnbs",
    "fbc_tfnbs",
    "nbs_extent_2.0",
    "nbs_intensity_2.0",
    "nbs_extent_3.0",
    "nbs_intensity_3.0",
    "cnbs",
]


def _parse_tfnbs_n(text) -> Optional[int]:
    """``none`` / ``null`` / ``exact`` -> None (exact integration); else a grid size."""
    if str(text).strip().lower() in {"none", "null", "exact"}:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid --tfnbs-n {text!r}: use an integer grid size or 'none'"
        ) from exc


def _git_sha() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=str(ROOT)
        ).stdout.strip()
    except Exception:
        return None


@dataclass
class GLMConfig:
    """Configuration for GLM validation."""

    methods: List[str]
    mode: str = "all"

    # Null calibration
    n_null: int = 100
    sample_sizes: List[int] = None

    # Power
    n_repeats: int = 20
    n_subjects: int = 60
    effect_sizes: List[float] = None
    scenarios: List[str] = None

    # Shared inference
    n_permutations: int = 500
    alpha: float = 0.05
    n_jobs: int = 1
    use_mp: bool = False
    seed: int = 42

    # Synthetic network
    n_nodes: int = 60
    n_modules: int = 4
    time_points: int = 30
    intra_corr: float = 0.3
    inter_corr: float = 0.05
    uniform_corr: float = 0.15
    noise_level: float = 0.05

    # GLM covariates
    interest_confound_corr: float = 0.3
    confound_effect_size: float = 0.15
    confound_scenario: str = "between_modules_dense"

    # Method parameters (ConnInfPy 2.0.0 defaults: e=0.4, h=3.0, exact integration)
    tfnbs_e: float = 0.4
    tfnbs_h: float = 3.0
    tfnbs_n: Optional[int] = None   # None = exact integration (2.0.0); int = legacy grid size
    tfnbs_start_thres: float = 1.65
    nbs_threshold: float = 2.0
    fbc_min_cluster: int = 3

    # Directional family: True = one two-sided family against the joint max-null
    # (ConnInfPy 2.0.0 default); False = legacy per-tail nulls, two one-sided families.
    two_tailed: bool = True

    # Execution
    resume: bool = False
    checkpoint_every: int = 100

    output_dir: Path = (ROOT / "examples/simulations/results/glm/glm_validation")

    def __post_init__(self) -> None:
        if self.sample_sizes is None:
            self.sample_sizes = [30, 60, 100]
        if self.effect_sizes is None:
            self.effect_sizes = [0.10, 0.20, 0.30]
        if self.scenarios is None:
            self.scenarios = ["within_module_dense", "hub", "chain"]


@dataclass
class GLMCalibrationResult:
    method: str
    null_run_id: int
    n_samples: int
    n_edges: int
    any_significant_pos: bool
    n_significant_pos: int
    fpr_pos: float
    min_p_pos: float
    any_significant_neg: bool
    n_significant_neg: int
    fpr_neg: float
    min_p_neg: float
    any_significant: bool
    n_significant_total: int
    fpr_total: float
    # Joint two-sided family: any edge of EITHER sign with min(p_pos, p_neg) <= alpha.
    min_p_joint: float
    any_significant_joint: bool
    two_tailed: bool
    elapsed_time: float


@dataclass
class GLMPowerResult:
    method: str
    method_params: str
    scenario: str
    effect_size: float
    n_subjects: int
    repeat_id: int
    n_true: int
    n_null: int
    TP: int
    FP: int
    FN: int
    TN: int
    TPR: float
    FPR: float
    precision: float
    FDR: float
    elapsed_time: float


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / max(x.std(ddof=0), 1e-12)


def _stable_int_token(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _load_config(path: Optional[Path]) -> GLMConfig:
    if path is None:
        return GLMConfig(methods=list(GLM_METHODS))
    if not HAS_YAML:
        raise RuntimeError("PyYAML is required to read YAML configs.")
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if "methods" not in data:
        data["methods"] = list(GLM_METHODS)
    if "output_dir" in data:
        data["output_dir"] = config_path(data["output_dir"], path)
    # ``tfnbs_n: null`` (exact integration) or a quoted grid size both load cleanly.
    if isinstance(data.get("tfnbs_n"), str):
        data["tfnbs_n"] = _parse_tfnbs_n(data["tfnbs_n"])
    return GLMConfig(**data)


def _parse_nbs_method(method: str, fallback_threshold: float) -> Optional[dict]:
    prefixes = {
        "nbs_extent": "extent",
        "nbs_intensity": "intensity",
    }
    for prefix, stat in prefixes.items():
        if method == prefix:
            return {"threshold": fallback_threshold, "nbs_stat": stat}
        marker = prefix + "_"
        if method.startswith(marker):
            return {"threshold": float(method[len(marker):]), "nbs_stat": stat}
    return None


def _method_call(method: str, cfg: GLMConfig, net_labels: np.ndarray) -> Tuple[str, Dict[str, Any]]:
    nbs_kwargs = _parse_nbs_method(method, cfg.nbs_threshold)
    if nbs_kwargs is not None:
        return "nbs", nbs_kwargs

    kwargs: Dict[str, Any] = {}
    if method in {"tfnbs", "ni_tfnbs", "fbc_tfnbs"}:
        kwargs.update({
            "e": cfg.tfnbs_e,
            "h": cfg.tfnbs_h,
            "n": cfg.tfnbs_n,
            "start_thres": cfg.tfnbs_start_thres,
        })
    if method in {"cnbs", "ni_tfnbs", "fbc_tfnbs"}:
        kwargs["net_labels"] = net_labels
    if method == "fbc_tfnbs":
        kwargs["min_cluster_size"] = cfg.fbc_min_cluster
    return method, kwargs


def _method_params_label(method: str, kwargs: Dict[str, Any]) -> str:
    if not kwargs:
        return ""
    keys = {
        "tfnbs": ("e", "h", "n", "start_thres"),
        "ni_tfnbs": ("e", "h", "n", "start_thres"),
        "fbc_tfnbs": ("e", "h", "n", "start_thres", "min_cluster_size"),
        "nbs_extent_2.0": ("threshold",),
        "nbs_intensity_2.0": ("threshold",),
        "nbs_extent_3.0": ("threshold",),
        "nbs_intensity_3.0": ("threshold",),
    }.get(method, tuple(kwargs.keys()))
    return ",".join(f"{key}={kwargs[key]}" for key in keys if key in kwargs)


def _make_topology_generator(cfg: GLMConfig, seed: int) -> topo.TopologyDatasetGenerator:
    return topo.TopologyDatasetGenerator(
        n_nodes=cfg.n_nodes,
        n_modules=cfg.n_modules,
        intra_corr=cfg.intra_corr,
        inter_corr=cfg.inter_corr,
        uniform_corr=cfg.uniform_corr,
        noise_level=cfg.noise_level,
        seed=seed,
    )


def generate_glm_dataset(
    cfg: GLMConfig,
    *,
    scenario: str,
    effect_size: float,
    n_subjects: int,
    seed: int,
) -> dict:
    """Generate Fisher-z connectivity data for a continuous-covariate GLM."""
    rng = np.random.default_rng(seed)
    generator = _make_topology_generator(cfg, seed)

    base = generator.generate(
        scenario,
        effect_size=0.0,
        n_samples=n_subjects,
        time_points=cfg.time_points,
    )
    Y, _ = base.fisher_z()
    signal_weights = np.abs(base.effect_mask)
    signal_mask = signal_weights > 0

    if cfg.confound_scenario == scenario:
        confound_weights = signal_weights.copy()
    else:
        confound_ds = generator.generate(
            cfg.confound_scenario,
            effect_size=0.0,
            n_samples=n_subjects,
            time_points=cfg.time_points,
        )
        confound_weights = np.abs(confound_ds.effect_mask)

    interest = _zscore(rng.standard_normal(n_subjects))
    confound_noise = _zscore(rng.standard_normal(n_subjects))
    rho = float(np.clip(cfg.interest_confound_corr, -0.99, 0.99))
    confound = _zscore(rho * interest + np.sqrt(1.0 - rho**2) * confound_noise)
    confounds = confound.reshape(-1, 1)

    if effect_size != 0.0:
        Y += effect_size * interest[:, None, None] * signal_weights[None, :, :]
    if cfg.confound_effect_size != 0.0:
        Y += cfg.confound_effect_size * confound[:, None, None] * confound_weights[None, :, :]

    diag = np.arange(cfg.n_nodes)
    Y[:, diag, diag] = 0.0
    return {
        "Y": Y,
        "interest": interest,
        "confounds": confounds,
        "net_labels": base.net_labels,
        "signal_mask": signal_mask,
    }


def _run_glm(
    method: str,
    cfg: GLMConfig,
    data: dict,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float, str]:
    method_key, kwargs = _method_call(method, cfg, data["net_labels"])
    method_params = _method_params_label(method, kwargs)
    start = time.time()
    result = compute_p_val_glm(
        data["Y"],
        interest=data["interest"],
        confounds=data["confounds"],
        method=method_key,
        n_permutations=cfg.n_permutations,
        two_tailed=cfg.two_tailed,
        use_mp=cfg.use_mp,
        rng=seed,
        **kwargs,
    )
    elapsed = time.time() - start
    return np.asarray(result["positive"]), np.asarray(result["negative"]), elapsed, method_params


def run_single_calibration_task(task: tuple) -> GLMCalibrationResult:
    method, n_subjects, run_id, cfg = task
    seed = cfg.seed + 10_000 * run_id + 137 * n_subjects
    data = generate_glm_dataset(
        cfg,
        scenario=cfg.scenarios[0],
        effect_size=0.0,
        n_subjects=n_subjects,
        seed=seed,
    )
    n_edges = cfg.n_nodes * (cfg.n_nodes - 1) // 2
    triu_idx = np.triu_indices(cfg.n_nodes, k=1)

    try:
        p_pos, p_neg, elapsed, _ = _run_glm(method, cfg, data, seed)
        p_pos_upper = p_pos[triu_idx]
        p_neg_upper = p_neg[triu_idx]

        sig_pos = p_pos_upper <= cfg.alpha
        sig_neg = p_neg_upper <= cfg.alpha
        n_sig_pos = int(sig_pos.sum())
        n_sig_neg = int(sig_neg.sum())
        min_p_pos = float(np.min(p_pos_upper))
        min_p_neg = float(np.min(p_neg_upper))
        # Joint two-sided family: any edge of EITHER sign significant.
        min_p_joint = float(np.min(np.minimum(p_pos_upper, p_neg_upper)))
    except Exception as exc:
        # Recording a failure as "nothing significant" would bias FWER down
        raise RuntimeError(f"GLM calibration {method} run {run_id} failed") from exc

    n_sig_total = n_sig_pos + n_sig_neg
    return GLMCalibrationResult(
        method=method,
        null_run_id=run_id,
        n_samples=n_subjects,
        n_edges=n_edges,
        any_significant_pos=bool(n_sig_pos > 0),
        n_significant_pos=n_sig_pos,
        fpr_pos=n_sig_pos / n_edges,
        min_p_pos=min_p_pos,
        any_significant_neg=bool(n_sig_neg > 0),
        n_significant_neg=n_sig_neg,
        fpr_neg=n_sig_neg / n_edges,
        min_p_neg=min_p_neg,
        any_significant=bool(n_sig_total > 0),
        n_significant_total=n_sig_total,
        fpr_total=n_sig_total / (2 * n_edges),
        min_p_joint=min_p_joint,
        any_significant_joint=bool(min_p_joint <= cfg.alpha),
        two_tailed=bool(cfg.two_tailed),
        elapsed_time=elapsed,
    )


def run_single_power_task(task: tuple) -> GLMPowerResult:
    method, scenario, effect_size, repeat_id, cfg = task
    seed = cfg.seed + 50_000 * repeat_id + _stable_int_token(scenario) % 10_000
    data = generate_glm_dataset(
        cfg,
        scenario=scenario,
        effect_size=effect_size,
        n_subjects=cfg.n_subjects,
        seed=seed,
    )
    triu_idx = np.triu_indices(cfg.n_nodes, k=1)
    signal_upper = data["signal_mask"][triu_idx]
    n_true = int(signal_upper.sum())
    n_null = int((~signal_upper).sum())

    try:
        p_pos, _, elapsed, method_params = _run_glm(method, cfg, data, seed)
        sig_upper = p_pos[triu_idx] <= cfg.alpha
        TP = int(np.sum(sig_upper & signal_upper))
        FP = int(np.sum(sig_upper & ~signal_upper))
        FN = int(np.sum(~sig_upper & signal_upper))
        TN = int(np.sum(~sig_upper & ~signal_upper))
    except Exception as exc:
        raise RuntimeError(f"GLM power {method} {scenario} repeat {repeat_id} failed") from exc

    TPR = TP / n_true if n_true else 0.0
    FPR = FP / n_null if n_null else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    FDR = FP / (TP + FP) if (TP + FP) else 0.0

    return GLMPowerResult(
        method=method,
        method_params=method_params,
        scenario=scenario,
        effect_size=effect_size,
        n_subjects=cfg.n_subjects,
        repeat_id=repeat_id,
        n_true=n_true,
        n_null=n_null,
        TP=TP,
        FP=FP,
        FN=FN,
        TN=TN,
        TPR=TPR,
        FPR=FPR,
        precision=precision,
        FDR=FDR,
        elapsed_time=elapsed,
    )


def _run_tasks(tasks: List[tuple], worker, cfg: GLMConfig, desc: str, on_result=None) -> list:
    """Run ``worker`` over ``tasks`` (serial or spawn pool); ``on_result`` sees every result."""
    if not tasks:
        return []
    if cfg.n_jobs == 1 or len(tasks) <= 1:
        results = []
        for task in tqdm(tasks, desc=desc, unit="run"):
            result = worker(task)
            results.append(result)
            if on_result is not None:
                on_result(result)
        return results

    n_jobs = get_available_cores() if cfg.n_jobs < 0 else cfg.n_jobs
    n_jobs = min(n_jobs, len(tasks))
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(processes=n_jobs, initializer=_limit_worker_threads) as pool:
        for result in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks), desc=desc, unit="run"):
            results.append(result)
            if on_result is not None:
                on_result(result)
    return results


# Protocol fingerprint = every generator / model / inference parameter of GLMConfig that is not
# part of the cell key (methods, n_null, sample_sizes, n_repeats, effect_sizes, scenarios may be
# extended under the same contract) and not an execution/location flag (output_dir, resume,
# checkpoint_every, use_mp, n_jobs do not change results), plus the git sha. Stored with a sha256
# over its own fields so a tampered / truncated protocol file is rejected too.
PROTOCOL_EXCLUDE = frozenset({
    "methods", "n_null", "sample_sizes", "n_repeats", "effect_sizes", "scenarios",
    "output_dir", "resume", "checkpoint_every", "use_mp", "n_jobs",
})
PROTOCOL_KEYS = tuple(
    f.name for f in dataclass_fields(GLMConfig) if f.name not in PROTOCOL_EXCLUDE
) + ("git_sha",)

_STATE_FILES = (
    "glm_fwer_results.csv", "glm_fwer_summary.csv", "glm_power_results.csv", "glm_power_summary.csv",
    "glm_protocol.json", "glm_run_metadata.json",
)
_STATE_DIRS = ("glm_fwer_checkpoint_parts", "glm_power_checkpoint_parts", "per_scenario")


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV via a temp file in the same directory + os.replace (never a torn file)."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_write_json(obj: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def _sha256_protocol(fields_: dict) -> str:
    return hashlib.sha256(json.dumps(fields_, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _protocol_fingerprint(cfg: GLMConfig) -> dict:
    fp = {key: getattr(cfg, key) for key in PROTOCOL_KEYS if key != "git_sha"}
    fp["git_sha"] = _git_sha()
    fp["protocol_sha256"] = _sha256_protocol(fp)
    return fp


def _protocol_intact(existing: dict) -> Tuple[bool, List[str]]:
    """Required fields present and the stored sha256 matches the stored fields."""
    missing = [k for k in PROTOCOL_KEYS if k not in existing]
    if missing:
        return False, [f"missing fields {missing}"]
    recomputed = _sha256_protocol({k: existing[k] for k in PROTOCOL_KEYS})
    if existing.get("protocol_sha256") != recomputed:
        return False, [f"protocol_sha256 stored={existing.get('protocol_sha256')!r} recomputed={recomputed!r}"]
    return True, []


def _clear_output_state(out_dir: Path) -> None:
    """``--overwrite``: remove every results / checkpoint / protocol file so the run starts from nothing."""
    removed = []
    for name in _STATE_FILES:
        p = out_dir / name
        if p.exists():
            p.unlink()
            removed.append(name)
    for name in _STATE_DIRS:
        p = out_dir / name
        if p.exists():
            shutil.rmtree(p)
            removed.append(name + "/")
    for p in out_dir.glob("*.tmp"):
        p.unlink()
        removed.append(p.name)
    print(f"--overwrite: removed {', '.join(removed) if removed else 'nothing'} in {out_dir}")


def _check_protocol(cfg: GLMConfig, out_dir: Path, overwrite: bool) -> dict:
    """Fail fast (exit != 0) if ``out_dir`` already holds state from a different protocol.

    Runs before anything is written: existing CSV/JSON files are only read. State means
    a results CSV or a checkpoint part; a lone protocol/metadata file from an aborted
    start is not state and is simply replaced. With ``overwrite`` the mismatch is reported
    but tolerated (the caller then clears the directory).
    """
    current = _protocol_fingerprint(cfg)
    protocol_path = out_dir / "glm_protocol.json"
    state_paths = [
        p for p in (
            out_dir / "glm_fwer_results.csv", out_dir / "glm_power_results.csv",
            out_dir / "glm_fwer_checkpoint_parts", out_dir / "glm_power_checkpoint_parts",
        ) if p.exists()
    ]
    if not state_paths or not protocol_path.exists():
        if state_paths and not protocol_path.exists() and not overwrite:
            raise SystemExit(
                f"{out_dir} holds results/checkpoints without glm_protocol.json "
                f"(written before protocol fingerprints existed); use a new --output-dir "
                f"or --overwrite to recompute from scratch."
            )
        return current

    existing = json.loads(protocol_path.read_text())
    intact, problems = _protocol_intact(existing)
    if not intact:
        if overwrite:
            print(f"--overwrite: ignoring damaged protocol file ({'; '.join(problems)})")
            return current
        raise SystemExit(
            "Refusing to continue: glm_protocol.json is incomplete or its sha256 does not match "
            f"({' ; '.join(problems)}). Nothing was modified. Use a new --output-dir or --overwrite."
        )
    diffs = [
        f"{key}: stored={existing.get(key)!r} current={current[key]!r}"
        for key in PROTOCOL_KEYS if existing.get(key) != current[key]
    ]
    if diffs:
        if overwrite:
            print(f"--overwrite: replacing results made under a different protocol ({len(diffs)} field(s) differ)")
            return current
        raise SystemExit(
            "Refusing to continue: the protocol fingerprint in "
            f"{protocol_path} differs from this invocation ({'--resume' if cfg.resume else 'fresh run into a used directory'}):\n  "
            + "\n  ".join(diffs)
            + "\nNothing was modified. Use a new --output-dir, or --overwrite to discard the "
              "existing results and checkpoints deliberately."
        )
    return current


class _Checkpoint:
    """Chunked, crash-safe per-run progress.

    Progress lives in ``<parts_dir>/part_<seq>.csv``: every ``every`` results a new
    part file is written atomically (temp file + ``os.replace``), so each flush costs
    O(chunk) instead of rewriting the whole growing table (O(N^2) in total before).
    A torn temp file is never read. On ``resume`` all parts (or, for a finished run,
    the results CSV) are loaded once and every task whose key is present is skipped;
    ``finish`` writes the final CSV atomically and removes the parts directory.
    """

    def __init__(self, parts_dir: Path, final_csv: Path, key_cols: List[str], every: int, resume: bool):
        self.parts_dir, self.final_csv, self.key_cols, self.every = parts_dir, final_csv, key_cols, every
        self.done = pd.DataFrame()
        parts = sorted(parts_dir.glob("part_*.csv")) if parts_dir.exists() else []
        if resume:
            # A finished run (results CSV) that was extended and then interrupted leaves both;
            # load everything and de-duplicate by cell key (later parts win).
            frames = []
            if final_csv.exists():
                frames.append(pd.read_csv(final_csv, float_precision="round_trip"))
            frames += [pd.read_csv(p, float_precision="round_trip") for p in parts]
            if frames:
                self.done = (
                    pd.concat(frames, ignore_index=True)
                    .drop_duplicates(subset=key_cols, keep="last")
                    .reset_index(drop=True)
                )
                src = f"{len(parts)} checkpoint part(s)" + (" + results file" if len(frames) > len(parts) else "")
                print(f"Resuming: {len(self.done)} completed runs loaded from {src}")
        else:
            shutil.rmtree(parts_dir, ignore_errors=True)   # fresh run: stale parts are dropped
        self.done_keys = (
            set(map(tuple, self.done[key_cols].to_numpy().tolist())) if not self.done.empty else set()
        )
        self.new: List[dict] = []
        # Continue the part numbering after any existing part, so two consecutive
        # interrupts never overwrite an earlier part.
        self.seq = max((int(m.group(1)) for p in parts if (m := re.match(r"part_(\d+)\.csv$", p.name))), default=0)
        parts_dir.mkdir(parents=True, exist_ok=True)

    def is_done(self, key: tuple) -> bool:
        return key in self.done_keys

    def __call__(self, result) -> None:
        self.new.append(vars(result))
        if len(self.new) >= self.every:
            self.flush()

    def flush(self) -> None:
        if not self.new:
            return
        self.seq += 1
        chunk = pd.DataFrame(self.new)
        _atomic_write_csv(chunk, self.parts_dir / f"part_{self.seq:05d}.csv")
        # The chunk also stays in the in-memory table the final CSV is assembled from;
        # only the on-disk progress is chunked, so a crash loses at most ``every`` rows
        # and finish() never depends on re-reading the parts it is about to delete.
        self.done = pd.concat([d for d in (self.done, chunk) if not d.empty], ignore_index=True)
        self.new = []

    def frame(self) -> pd.DataFrame:
        parts = [d for d in (self.done, pd.DataFrame(self.new)) if not d.empty]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def finish(self, sort_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Write the final results CSV (done + new rows, sorted) atomically; drop the parts."""
        self.flush()
        df = self.frame()
        if sort_cols:
            df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        _atomic_write_csv(df, self.final_csv)
        shutil.rmtree(self.parts_dir, ignore_errors=True)
        return df


def _task_key_calibration(task: tuple) -> tuple:
    method, n_subjects, run_id, _ = task
    return (method, int(n_subjects), int(run_id))


def _task_key_power(task: tuple) -> tuple:
    method, scenario, effect_size, repeat_id, _ = task
    return (method, scenario, float(effect_size), int(repeat_id))


def _add_joint_two_sided(summary: pd.DataFrame, df: pd.DataFrame, cfg: GLMConfig) -> pd.DataFrame:
    """Add the joint two-sided empirical FWER and the resolved run parameters.

    ``fwer_two_sided_joint`` is the share of null runs with any edge of EITHER
    sign having ``min(p_pos, p_neg) <= alpha``. With ``two_tailed=True`` both
    p-maps are corrected against the joint max-null, so this is the FWER of the
    single two-sided family that the 2.0.0 contract holds at alpha. With
    ``two_tailed=False`` it equals the per-tail "either tail at alpha"
    diagnostic (``fwer_either_at_alpha``, expected <= 2*alpha). The per-tail
    ``fwer_pos`` / ``fwer_neg`` / ``fwer_max_tail`` columns stay one-directional
    diagnostics and ``fwer_two_sided_bonf`` the Bonferroni-over-tails one.
    """
    from scipy import stats

    key = ["method", "n_samples"] if "n_samples" in summary.columns else ["method"]
    joint = (
        df.groupby(key)["any_significant_joint"]
        .agg(n_runs_with_fp_joint="sum", fwer_two_sided_joint="mean")
        .reset_index()
    )
    summary = summary.drop(columns=[c for c in ("n_runs_with_fp_joint", "fwer_two_sided_joint") if c in summary])
    summary = summary.merge(joint, on=key, how="left")
    k = summary["n_runs_with_fp_joint"].astype(int).to_numpy()
    n = summary["n_null_runs"].astype(int).to_numpy()
    cis = [clopper_pearson_ci(int(ki), int(ni)) for ki, ni in zip(k, n)]
    summary["fwer_two_sided_joint_ci_low"] = [lo for lo, _ in cis]
    summary["fwer_two_sided_joint_ci_high"] = [hi for _, hi in cis]
    summary["fwer_two_sided_joint_binom_p"] = [float(stats.binom.sf(ki - 1, ni, cfg.alpha)) for ki, ni in zip(k, n)]
    summary["two_tailed"] = bool(cfg.two_tailed)
    summary["tfnbs_n"] = cfg.tfnbs_n
    summary["n_permutations"] = cfg.n_permutations
    return summary


def run_calibration(cfg: GLMConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _Checkpoint(
        out_dir / "glm_fwer_checkpoint_parts", out_dir / "glm_fwer_results.csv",
        key_cols=["method", "n_samples", "null_run_id"],
        every=cfg.checkpoint_every, resume=cfg.resume,
    )
    tasks = [
        task for task in itertools.product(cfg.methods, cfg.sample_sizes, range(cfg.n_null), [cfg])
        if not ckpt.is_done(_task_key_calibration(task))
    ]
    print("=== GLM Null FWER Calibration ===")
    print(f"Methods: {cfg.methods}")
    print(f"Subject counts: {cfg.sample_sizes}")
    print(f"Null repeats per cell: {cfg.n_null}")
    print(f"Permutations: {cfg.n_permutations}")
    print(f"TFNBS: e={cfg.tfnbs_e}, h={cfg.tfnbs_h}, n={cfg.tfnbs_n}, start_thres={cfg.tfnbs_start_thres}")
    print(f"Family: {'joint max-null, one two-sided family (two_tailed=True)' if cfg.two_tailed else 'per-tail nulls, two one-sided families (two_tailed=False)'}")
    print(f"Runs remaining: {len(tasks)}")
    if not tasks:
        print("All runs already completed!")
    _run_tasks(tasks, run_single_calibration_task, cfg, "glm-fwer", on_result=ckpt)

    df = ckpt.finish(sort_cols=["method", "n_samples", "null_run_id"])
    summary = _add_joint_two_sided(compute_fpr_summary(df, alpha=cfg.alpha), df, cfg)
    _atomic_write_csv(summary, out_dir / "glm_fwer_summary.csv")
    return df, summary


def run_power(cfg: GLMConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _Checkpoint(
        out_dir / "glm_power_checkpoint_parts", out_dir / "glm_power_results.csv",
        key_cols=["method", "scenario", "effect_size", "repeat_id"],
        every=cfg.checkpoint_every, resume=cfg.resume,
    )
    tasks = [
        task for task in itertools.product(cfg.methods, cfg.scenarios, cfg.effect_sizes, range(cfg.n_repeats), [cfg])
        if not ckpt.is_done(_task_key_power(task))
    ]
    print("=== GLM Continuous-Covariate Power ===")
    print(f"Methods: {cfg.methods}")
    print(f"Scenarios: {cfg.scenarios}")
    print(f"Effect sizes: {cfg.effect_sizes}")
    print(f"Repeats per cell: {cfg.n_repeats}")
    print(f"Subjects: {cfg.n_subjects}")
    print(f"Permutations: {cfg.n_permutations}")
    print(f"TFNBS: e={cfg.tfnbs_e}, h={cfg.tfnbs_h}, n={cfg.tfnbs_n}, two_tailed={cfg.two_tailed}")
    print(f"Runs remaining: {len(tasks)}")
    if not tasks:
        print("All runs already completed!")
    _run_tasks(tasks, run_single_power_task, cfg, "glm-power", on_result=ckpt)

    df = ckpt.finish(sort_cols=["method", "scenario", "effect_size", "repeat_id"])
    group_cols = ["method", "method_params", "scenario", "effect_size", "n_subjects"]
    summary = (
        df.groupby(group_cols)
        .agg(
            TPR_mean=("TPR", "mean"),
            TPR_std=("TPR", "std"),
            FDR_mean=("FDR", "mean"),
            FDR_std=("FDR", "std"),
            FPR_mean=("FPR", "mean"),
            precision_mean=("precision", "mean"),
            elapsed_time_mean=("elapsed_time", "mean"),
            n_repeats=("repeat_id", "count"),
        )
        .reset_index()
    )

    _atomic_write_csv(summary, out_dir / "glm_power_summary.csv")
    per_scenario = out_dir / "per_scenario"
    per_scenario.mkdir(exist_ok=True)
    for scenario, sdf in df.groupby("scenario"):
        _atomic_write_csv(sdf, per_scenario / f"glm_power_results_{scenario}.csv")
    return df, summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run GLM null calibration and continuous-covariate power validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config path.")
    parser.add_argument(
        "--mode",
        choices=["calibration", "power", "all"],
        default=None,
        help="Override config mode.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    parser.add_argument("--methods", nargs="+", default=None, help="Override config methods.")
    parser.add_argument("--n-null", type=int, default=None, help="Override null repeats per cell.")
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=None,
                        help="Override calibration sample sizes.")
    parser.add_argument("--n-permutations", type=int, default=None, help="Override permutations per test.")
    parser.add_argument("--tfnbs-n", type=_parse_tfnbs_n, default=argparse.SUPPRESS, metavar="N|none",
                        help="TFNBS threshold grid size; 'none' = exact integration (ConnInfPy 2.0.0 default).")
    tails = parser.add_mutually_exclusive_group()
    tails.add_argument("--two-tailed", dest="two_tailed", action="store_true", default=argparse.SUPPRESS,
                       help="One two-sided family against the joint max-null (ConnInfPy 2.0.0 default).")
    tails.add_argument("--one-tailed", dest="two_tailed", action="store_false",
                       help="Legacy per-tail nulls (two one-sided families, each at alpha).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cells already present in the checkpoint parts or finished results CSV.")
    parser.add_argument("--checkpoint-every", type=int, default=None, metavar="N",
                        help="Rows per atomic checkpoint part file (default: config / 100).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Discard existing results/checkpoints in --output-dir even if their protocol "
                             "fingerprint differs (fresh run into a used directory).")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    if args.mode is not None:
        cfg.mode = args.mode
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.methods is not None:
        cfg.methods = list(args.methods)
    if args.n_null is not None:
        cfg.n_null = args.n_null
    if args.sample_sizes is not None:
        cfg.sample_sizes = list(args.sample_sizes)
    if args.n_permutations is not None:
        cfg.n_permutations = args.n_permutations
    # SUPPRESS keeps the config value unless the flag is given; ``--tfnbs-n none``
    # then resolves to None (exact integration) and does override the config.
    if hasattr(args, "tfnbs_n"):
        cfg.tfnbs_n = args.tfnbs_n
    if hasattr(args, "two_tailed"):
        cfg.two_tailed = bool(args.two_tailed)
    if args.resume:
        cfg.resume = True
    if args.checkpoint_every is not None:
        cfg.checkpoint_every = max(1, int(args.checkpoint_every))

    # Fail fast: the GLM pipeline has no parametric bonferroni / bh_fdr baselines
    # (those live in compute_p_val / fpr_calibration); a mid-run crash after hours
    # would only be recoverable through --resume.
    unsupported = [
        m for m in cfg.methods
        if m not in GLM_METHODS and _parse_nbs_method(m, cfg.nbs_threshold) is None
    ]
    if unsupported:
        raise SystemExit(
            f"Unsupported GLM methods {unsupported}: the GLM pipeline has no parametric "
            f"bonferroni/bh_fdr (use fpr_calibration.py for those). Supported: "
            f"{GLM_METHODS} plus nbs_extent_<thr> / nbs_intensity_<thr>."
        )

    # Protocol check BEFORE any write: an existing results/checkpoint made under a
    # different contract (seed, alpha, perms, two_tailed, TFNBS cell, data generator,
    # git sha) makes --resume refuse with exit != 0 and leaves every file untouched.
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = _check_protocol(cfg, cfg.output_dir, overwrite=args.overwrite)
    if args.overwrite:
        # Explicit fresh start: clear every results / checkpoint / protocol file and never
        # load or merge old raw rows, even when the YAML sets resume: true.
        _clear_output_state(cfg.output_dir)
        cfg.resume = False
    meta = {
        "script": "examples/simulations/glm/glm_validation.py",
        "config": str(args.config) if args.config else None,
        "git_sha": protocol["git_sha"],
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol": protocol,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": cfg.mode, "methods": cfg.methods,
        "n_null": cfg.n_null, "sample_sizes": cfg.sample_sizes, "n_permutations": cfg.n_permutations,
        "alpha": cfg.alpha, "seed": cfg.seed, "n_jobs": cfg.n_jobs, "use_mp": cfg.use_mp,
        "tfnbs_e": cfg.tfnbs_e, "tfnbs_h": cfg.tfnbs_h, "tfnbs_n": cfg.tfnbs_n,
        "tfnbs_start_thres": cfg.tfnbs_start_thres, "two_tailed": cfg.two_tailed,
        "resume": cfg.resume, "overwrite": args.overwrite, "checkpoint_every": cfg.checkpoint_every,
    }
    _atomic_write_json(protocol, cfg.output_dir / "glm_protocol.json")
    _atomic_write_json(meta, cfg.output_dir / "glm_run_metadata.json")

    if cfg.mode in {"calibration", "all"}:
        _, summary = run_calibration(cfg)
        cols = [c for c in ("method", "n_samples", "fwer_max_tail", "fwer_two_sided_bonf",
                            "fwer_two_sided_joint", "fwer_two_sided_joint_ci_low",
                            "fwer_two_sided_joint_ci_high") if c in summary]
        print(summary[cols].to_string(index=False))
    if cfg.mode in {"power", "all"}:
        _, summary = run_power(cfg)
        print(summary.head(20).to_string(index=False))

    print(f"\nGLM validation outputs saved to {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
