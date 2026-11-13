"""Command line: ``conninfpy run analysis.yaml`` and ``conninfpy info``.

An analysis file names a data manifest, the design (which phenotype
columns hold the predictor, confounds, sites or groups) and the inference
settings. ``run`` loads the data, calls :func:`conninfpy.analyze`, and
writes three files to the output directory:

- ``edges.csv`` — significant edges (atlas-annotated when an atlas is known)
- ``pvals.npz`` — ``positive`` / ``negative`` p-value maps, ``stat_signed``,
  and the coefficient maps of the tested model: ``beta`` (p, N, N) with
  ``beta_columns``, ``effect`` (the contrast c' beta) and the predicted
  connectivity at the two compared profiles (``fc[<profile>]``)
- ``provenance.json`` — package version, the analysis file, data and design
  summary, resolved tail/correction/native-unit policy, the significance
  level ``alpha``, warnings (``flags``), and run time

Both directions are reported, so ``inference.two_tailed`` defaults to True
(one two-sided family; a single alpha covers both directions). Set it
explicitly to False for the legacy separate one-sided families (each at the
nominal alpha; the combined-direction FWER is then not the nominal alpha).
A directional hypothesis fixed before the run is declared with
``inference.tested_tail: positive`` (or ``negative``): that direction is
tested against its own null and the other is reported as not tested.

Example analysis file: ``examples/abide/configs/abide_mini_age.yaml``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

DESIGNS = ("glm", "two-sample", "paired", "one-sample")
_TOP_LEVEL_KEYS = {"data", "input", "design", "inference", "atlas", "alpha", "output", "connectivity"}
_KIND_TO_INPUT = {"correlation": "r", "fisher_z": "z"}
_BUILTIN_ATLASES = ("schaefer_100_yeo7", "schaefer_200_yeo7", "schaefer_400_yeo7", "bna_246")


class ConfigError(ValueError):
    """A problem in the analysis file, reported without a traceback."""


# ---------------------------------------------------------------------------
# Analysis file
# ---------------------------------------------------------------------------

def load_config(path: "str | os.PathLike[str]") -> Dict[str, Any]:
    """Read and minimally validate an analysis file."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: expected a YAML mapping at the top level.")
    unknown = set(cfg) - _TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(
            f"{path}: unknown keys {sorted(unknown)}; allowed: {sorted(_TOP_LEVEL_KEYS)}."
        )
    for key in ("data", "design"):
        if key not in cfg:
            raise ConfigError(f"{path}: missing required key '{key}'.")
    design_type = cfg["design"].get("type")
    if design_type not in DESIGNS:
        raise ConfigError(f"design.type must be one of {DESIGNS}; got {design_type!r}.")
    return cfg


def _columns(pheno: pd.DataFrame, names: Sequence[str], what: str) -> pd.DataFrame:
    missing = [n for n in names if n not in pheno.columns]
    if missing:
        raise ConfigError(
            f"{what}: column(s) {missing} not in the phenotype table; "
            f"available: {list(pheno.columns)}."
        )
    return pheno[list(names)]


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _numeric(frame: pd.DataFrame) -> np.ndarray:
    """Numeric columns as-is; text columns one-hot encoded (first level dropped)."""
    return pd.get_dummies(frame, drop_first=True, dtype=float).to_numpy(float)


# ---------------------------------------------------------------------------
# Data → analyze() arguments
# ---------------------------------------------------------------------------

def _connectivity(dataset: Any, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Optional[str]]:
    """Construct connectivity when needed and retain observation diagnostics."""
    if dataset.data_kind == "timeseries":
        from conninfpy.connectivity import connectivity_from_dataset
        opts = dict(options or {})
        kind = opts.pop("kind", "correlation")
        if not hasattr(dataset, "pheno"):
            # Retain the lightweight internal conversion contract for callers
            # that supply only data and data_kind.
            from conninfpy.connectivity import compute_connectivity
            opts.pop("group_column", None)
            return compute_connectivity(dataset.data, kind=kind, **opts), "r"
        connectivity_from_dataset(dataset, kind=kind, **opts)
    elif options:
        raise ConfigError("connectivity options require raw timeseries data; configure the manifest for loaders that construct matrices.")
    return np.asarray(dataset.data, dtype=np.float64), _KIND_TO_INPUT.get(dataset.data_kind)


def build_analysis(dataset: Any, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Turn a loaded dataset and the ``design`` block into ``analyze()`` kwargs.

    Rows with a missing value in any column the design uses are dropped and
    counted in the returned summary.
    """
    design = cfg["design"]
    kind = design["type"]
    options = dict(cfg.get("connectivity") or {})
    if design.get("group") and "group_column" not in options:
        options["group_column"] = design["group"]
    Y, scale = _connectivity(dataset, options if dataset.data_kind == "timeseries" else cfg.get("connectivity"))
    # Three-state preprocessing facts (gsr / estimator / standardize /
    # shrinkage / tr_seconds, each with a source) recorded by the loaders and
    # the connectivity backend; kept unknown where no explicit source exists.
    preprocessing_record = dataset.provenance.get("preprocessing")
    pheno = dataset.pheno.reset_index(drop=True)
    if design.get("group") and "n_valid_tr" in pheno:
        from conninfpy.connectivity import warn_group_diagnostics
        warn_group_diagnostics(dataset, design["group"])
    if len(pheno) != Y.shape[0]:
        raise ConfigError(
            f"phenotype table has {len(pheno)} rows but the data has {Y.shape[0]} subjects."
        )
    input_kind = cfg.get("input") or scale
    confound_cols = _as_list(design.get("confounds"))
    site_col = design.get("sites")
    kwargs: Dict[str, Any] = {"input": input_kind}
    if preprocessing_record:
        kwargs["preprocessing"] = preprocessing_record
    encoding_frame = None
    encoding_columns: List[str] = []
    if design.get("harmonize") is not None:
        kwargs["harmonize"] = design["harmonize"]

    if kind == "glm":
        interest_cols = _as_list(design.get("interest"))
        if not interest_cols:
            raise ConfigError("design.interest is required for type 'glm'.")
        used = interest_cols + confound_cols + _as_list(site_col)
        keep = _columns(pheno, used, "design").notna().all(axis=1).to_numpy()
        p = pheno[keep]
        encoding_frame = p
        encoding_columns = interest_cols + confound_cols
        interest = {}
        for c in interest_cols:
            values = _numeric(_columns(p, [c], "design.interest"))
            if values.shape[1] != 1:
                raise ConfigError(
                    f"design.interest {c!r} has more than two categories; code it "
                    "numerically, or use one indicator column per level."
                )
            interest[c] = values.ravel()
        kwargs["Y"] = Y[keep]
        interactions = design.get("interactions")
        kwargs["interest"] = interest if len(interest) > 1 or interactions is not None else interest[interest_cols[0]]
        if interactions is not None:
            kwargs["interactions"] = interactions
        if len(interest) == 1 and interactions is None:
            kwargs["contrast_id"] = interest_cols[0]
        if confound_cols:
            kwargs["confounds"] = _numeric(p[confound_cols])
        if site_col:
            kwargs["sites"] = p[site_col].to_numpy()
        n_used = int(keep.sum())

    elif kind in ("two-sample", "one-sample"):
        if kind == "two-sample":
            group_col = design.get("group")
            levels = design.get("levels")
            if not group_col or not levels or len(levels) != 2:
                raise ConfigError("type 'two-sample' needs design.group and two design.levels.")
            used = [group_col] + _as_list(site_col)
            keep = _columns(pheno, used, "design").notna().all(axis=1).to_numpy()
            keep = keep & pheno[group_col].isin(levels).to_numpy()  # not in place: pandas 3 arrays can be read-only
            for level in levels:
                if not (pheno[group_col] == level).any():
                    raise ConfigError(f"design.levels: {level!r} not found in column {group_col!r}.")
            g1 = keep & (pheno[group_col] == levels[0]).to_numpy()
            g2 = keep & (pheno[group_col] == levels[1]).to_numpy()
            kwargs.update(group1=Y[g1], group2=Y[g2], test_type="two-sample")
            if site_col:
                kwargs["sites"] = np.concatenate([pheno.loc[g1, site_col], pheno.loc[g2, site_col]])
            n_used = int(g1.sum() + g2.sum())
        else:
            kwargs.update(group1=Y, test_type="one-sample")
            n_used = Y.shape[0]

    else:  # paired
        cond_col = design.get("condition") or getattr(dataset, "condition_column", None)
        levels = design.get("levels")
        subj_col = design.get("subject", "subject_id")
        if not cond_col or not levels or len(levels) != 2 or levels[0] == levels[1]:
            raise ConfigError("type 'paired' needs design.condition and two design.levels.")
        interest_cols = _as_list(design.get("interest"))
        subject_confounds = _as_list(design.get("subject_confounds"))
        if set(confound_cols) & set(subject_confounds):
            raise ConfigError("a confound cannot be both condition-specific and subject-level")
        used = [cond_col, subj_col] + confound_cols + interest_cols + subject_confounds + _as_list(site_col)
        _columns(pheno, used, "design")
        if pheno[used].isna().any().any():
            raise ConfigError("paired design columns cannot contain missing values")
        observed = set(pheno[cond_col].unique())
        if observed != set(levels):
            raise ConfigError(f"paired condition levels {levels!r} do not match observed levels {sorted(observed, key=str)!r}")
        rows = {lvl: pheno.index[pheno[cond_col] == lvl] for lvl in levels}
        for lvl in levels:
            if pheno.loc[rows[lvl], subj_col].duplicated().any():
                raise ConfigError(
                    f"subject column {subj_col!r} repeats within condition {lvl!r}; "
                    "each subject needs exactly one row per condition."
                )
        by_subj = {lvl: pd.Series(rows[lvl], index=pheno.loc[rows[lvl], subj_col]) for lvl in levels}
        subjects = by_subj[levels[0]].index.intersection(by_subj[levels[1]].index)
        if len(subjects) == 0 or len(subjects) != len(by_subj[levels[0]]) or len(subjects) != len(by_subj[levels[1]]):
            raise ConfigError(f"each subject must have both conditions {levels}; missing pair detected")
        ia = by_subj[levels[0]].loc[subjects].to_numpy()
        ib = by_subj[levels[1]].loc[subjects].to_numpy()
        encoding_frame = pheno
        encoding_columns = interest_cols + confound_cols + subject_confounds
        kwargs.update(group1=Y[ia], group2=Y[ib], test_type="paired")
        for col in interest_cols + subject_confounds + _as_list(site_col):
            if not np.array_equal(pheno.loc[ia, col].to_numpy(), pheno.loc[ib, col].to_numpy()):
                raise ConfigError(f"paired subject-level column {col!r} must be constant across conditions")
        if interest_cols:
            interest = {}
            for c in interest_cols:
                values = _numeric(pheno.loc[ia, [c]])
                if values.shape[1] != 1:
                    raise ConfigError(f"design.interest {c!r} must be numeric or binary categorical")
                interest[c] = values.ravel()
            interactions = design.get("interactions")
            kwargs["interest"] = interest if len(interest) > 1 or interactions is not None else interest[interest_cols[0]]
            if interactions is not None:
                kwargs["interactions"] = interactions
            elif len(interest) == 1:
                kwargs["contrast_id"] = interest_cols[0]
            if subject_confounds:
                kwargs["confounds"] = _numeric(pheno.loc[ia, subject_confounds])
        elif subject_confounds or design.get("interactions") is not None:
            raise ConfigError("subject_confounds and interactions require paired design.interest")
        if confound_cols:
            for col in confound_cols:
                if not pd.api.types.is_numeric_dtype(pheno[col]):
                    if set(pheno.loc[ia, col]) != set(pheno.loc[ib, col]):
                        raise ConfigError(f"condition-specific categorical confound {col!r} has incompatible levels")
            # Fit one encoding on both conditions so the same columns are
            # subtracted even if rows are shuffled.
            combined = _numeric(pd.concat([pheno.loc[ia, confound_cols], pheno.loc[ib, confound_cols]]))
            kwargs["confounds_group1"] = combined[:len(ia)]
            kwargs["confounds_group2"] = combined[len(ia):]
        if site_col:
            kwargs["sites"] = pheno.loc[ia, site_col].to_numpy()
        n_used = len(subjects)

    categorical_coding = {}
    if encoding_frame is not None:
        for col in dict.fromkeys(encoding_columns):
            series = encoding_frame[col].dropna()
            if not pd.api.types.is_numeric_dtype(series):
                levels_found = pd.Categorical(series).categories.tolist()
                categorical_coding[col] = {"levels": levels_found, "reference": levels_found[0] if levels_found else None}
    summary = {
        "design": kind,
        "n_subjects_in_data": int(Y.shape[0]),
        "n_subjects_used": n_used,
        "n_regions": int(Y.shape[-1]),
        "input": input_kind,
        "categorical_coding": categorical_coding,
        "connectivity": dataset.provenance.get("connectivity"),
        "preprocessing": dataset.provenance.get("preprocessing"),
        "connectivity_diagnostics": dataset.pheno[[c for c in ("n_valid_tr", "delta_s") if c in dataset.pheno]].to_dict(orient="list"),
        "loader_warnings": dataset.warnings,
    }
    return kwargs, summary


def _atlas(spec: Any, dataset: Any, base: Path, n_regions: int) -> Any:
    from conninfpy.atlas.metadata import AtlasInfo

    if spec is None:
        atlas = getattr(dataset, "atlas", None)
    elif spec in _BUILTIN_ATLASES:
        atlas = getattr(AtlasInfo, spec)()
    else:
        atlas = AtlasInfo.from_csv(str((base / spec).resolve()))
    if atlas is not None and len(atlas) != n_regions:
        raise ConfigError(f"atlas has {len(atlas)} regions but the data has {n_regions}.")
    return atlas


# ---------------------------------------------------------------------------
# run / info
# ---------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist() if obj.size <= 1000 else f"<array shape={obj.shape}>"
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def run(
    config_path: "str | os.PathLike[str]",
    out_dir: "str | os.PathLike[str] | None" = None,
    *,
    n_permutations: Optional[int] = None,
    use_mp: Optional[bool] = None,
) -> Dict[str, Path]:
    """Run one analysis file; return the paths written, keyed by name."""
    from conninfpy import __version__
    from conninfpy.analysis import analyze
    from conninfpy.loaders import load

    config_path = Path(config_path).resolve()
    cfg = load_config(config_path)
    base = config_path.parent
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    dataset = load(base / cfg["data"])
    kwargs, summary = build_analysis(dataset, cfg)
    inference = dict(cfg.get("inference") or {})
    # CSV/report output includes both signs: declare one two-sided family
    # (one alpha over both directions) unless the file says otherwise.
    # A prespecified one-sided analysis declares inference.tested_tail
    # instead; forcing two_tailed=True there would contradict it.
    if "tested_tail" not in inference:
        inference.setdefault("two_tailed", True)
    if n_permutations is not None:
        inference["n_permutations"] = n_permutations
    if use_mp is not None:
        inference["use_mp"] = use_mp
    atlas = _atlas(cfg.get("atlas"), dataset, base, summary["n_regions"])
    alpha = float(cfg.get("alpha", 0.05))

    t0 = time.perf_counter()
    result = analyze(**kwargs, **inference, atlas=atlas)
    wall = time.perf_counter() - t0

    out = Path(out_dir) if out_dir else base / cfg.get("output", "conninfpy_results")
    out.mkdir(parents=True, exist_ok=True)
    results = result if isinstance(result, dict) else {"": result}
    written: Dict[str, Path] = {}
    per_result: Dict[str, Any] = {}
    for name, res in results.items():
        suffix = f"_{name}" if name else ""
        edges_path = out / f"edges{suffix}.csv"
        res.to_csv(edges_path, atlas=atlas, alpha=alpha)
        inf = res.inference
        if getattr(inf, "design", None) is not None and cfg["design"]["type"] in ("glm", "paired"):
            # Keep the original field roles and categorical reference levels in
            # the self-contained bundle, without retaining phenotype rows.
            inf.design["input_design"] = dict(cfg["design"])
            inf.design["categorical_coding"] = summary["categorical_coding"]
        pvals_path = out / f"pvals{suffix}.npz"
        arrays = {"positive": inf.positive, "negative": inf.negative}
        if getattr(inf, "stat_signed", None) is not None:
            arrays["stat_signed"] = inf.stat_signed
        design = getattr(inf, "design", None)
        if getattr(inf, "beta", None) is not None and design is not None:
            arrays["beta"] = inf.beta                                  # (p, N, N), response scale
            arrays["beta_columns"] = np.array(design["columns"])
            eff = inf.effect_map()
            if eff is not None:
                arrays["effect"] = eff                                 # c' beta
            for label in design.get("comparison") or []:
                arrays[f"fc[{label}]"] = inf.predicted(label)
        np.savez_compressed(pvals_path, **arrays)
        written[f"edges{suffix}"] = edges_path
        written[f"pvals{suffix}"] = pvals_path
        bundle_path = out / f"bundle{suffix}.pkl"
        res.to_bundle().save(bundle_path)
        written[f"bundle{suffix}"] = bundle_path
        per_result[name or "result"] = {
            "n_significant": inf.n_significant(alpha),
            "method": inf.method,
            "tail_policy": inf.tail_policy,
            "tested_tail": getattr(inf, "tested_tail", None),
            "tail_selection": getattr(inf, "tail_selection", None),
            "correction": inf.correction,
            "inference_unit": inf.inference_unit,
            "n_permutations": inf.n_permutations,
            "acceleration": inf.acceleration,
            "flags": list(res.flags),
            "resolved_strategy": getattr(res, "resolved_strategy", None),
            "combat_diagnostics": res.combat_diagnostics,
            "design_diagnostics": res.design_diagnostics,
            "fisher": (inf.analysis_provenance.get("fisher")
                       if isinstance(getattr(inf, "analysis_provenance", None), dict) else None),
            "design": None if design is None else {
                k: design.get(k) for k in ("kind", "columns", "contrast", "column_means", "profiles",
                                           "comparison", "interest", "scale", "estimable_columns",
                                           "enhancement")
            },
        }

    provenance = {
        "conninfpy_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "started_utc": started,
        "wall_time_s": round(wall, 3),
        "analysis_file": str(config_path),
        "config": cfg,
        "data": {"manifest": str((base / cfg["data"]).resolve()), **summary},
        "alpha": alpha,
        "results": per_result,
    }
    prov_path = out / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2, default=_jsonable), encoding="utf-8")
    written["provenance"] = prov_path

    print(f"conninfpy {__version__} — {summary['design']} on {summary['n_subjects_used']} "
          f"subjects × {summary['n_regions']} regions ({wall:.1f} s)")
    for name, info in per_result.items():
        label = "" if name == "result" else f"[{name}] "
        sig = info["n_significant"]
        print(f"  {label}significant at α={alpha}: positive={sig['positive']}, "
              f"negative={sig['negative']}")
    print(f"  results written to {out}")
    return written


def info() -> None:
    """Print versions, the TFNBS backend and the CPU / BLAS setup."""
    from conninfpy import __version__

    lines = [f"conninfpy  {__version__}", f"python     {platform.python_version()}",
             f"numpy      {np.__version__}"]
    try:
        import numba
        lines.append(f"numba      {numba.__version__} (JIT TFNBS backend)")
    except ImportError:
        lines.append("numba      not installed (SciPy TFNBS backend)")
    lines.append(f"cpu cores  {os.cpu_count()}")
    from conninfpy.stats._engine import get_available_cores

    lines.append(f"cpu usable {get_available_cores()} (default worker budget)")
    lines.append("workers    1 BLAS/OpenMP thread per process; parent settings below")
    try:
        from threadpoolctl import threadpool_info
        for pool in threadpool_info():
            lines.append(f"blas       {pool.get('internal_api')} {pool.get('version')} "
                         f"({pool.get('num_threads')} threads)")
    except ImportError:
        pass
    print("\n".join(lines))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="conninfpy",
        description="Permutation-based inference on brain connectivity networks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run an analysis file (YAML)")
    p_run.add_argument("config", help="analysis file, e.g. examples/abide/configs/abide_mini_age.yaml")
    p_run.add_argument("--out", help="output directory (default: 'output' in the file)")
    p_run.add_argument("--n-permutations", type=int, help="override inference.n_permutations")
    p_run.add_argument("--no-mp", action="store_true", help="run in a single process")
    sub.add_parser("info", help="show versions and the compute backend")
    args = parser.parse_args(argv)

    if args.command == "info":
        info()
        return 0
    try:
        run(args.config, args.out, n_permutations=args.n_permutations,
            use_mp=False if args.no_mp else None)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"conninfpy: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
