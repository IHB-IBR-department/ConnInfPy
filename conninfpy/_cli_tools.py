"""Helper implementations for ``conninfpy`` CLI subcommands: qc, init, interpret, influence, selftest, app."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


def _find_app_script() -> Optional[Path]:
    """Locate ``apps/streamlit_app.py`` from the installed package or repo checkout."""
    spec = importlib.util.find_spec("apps.streamlit_app")
    if spec is not None and spec.origin is not None:
        p = Path(spec.origin)
        if p.is_file():
            return p
    repo_candidate = Path(__file__).resolve().parents[1] / "apps" / "streamlit_app.py"
    if repo_candidate.is_file():
        return repo_candidate
    return None


def launch_app(extra_args: Optional[Sequence[str]] = None) -> int:
    """Launch the ConnInfPy Streamlit workspace application."""
    if importlib.util.find_spec("streamlit") is None:
        print(
            "conninfpy: streamlit is not installed. Install optional UI dependencies with:\n"
            '  uv pip install "conninfpy[app]"   (or: pip install "conninfpy[app]")',
            file=sys.stderr,
        )
        return 1
    app_script = _find_app_script()
    if app_script is None:
        print(
            "conninfpy: could not locate apps/streamlit_app.py in the installed package or repository.",
            file=sys.stderr,
        )
        return 1
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_script), *(extra_args or ())]
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def _format_model_summary(
    cfg: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    *,
    harmonize_strategy: Optional[str] = None,
) -> Optional[str]:
    """Build a concise, human-readable model summary line for ``qc`` and ``dry-run``."""
    from conninfpy.analysis import _resolve_strategy
    from conninfpy.cli import _as_list

    design = cfg["design"]
    kind = design["type"]
    if kind not in ("glm", "paired"):
        return None

    interest_cols = _as_list(design.get("interest"))
    confound_cols = _as_list(design.get("confounds"))
    site_col = design.get("sites")
    harmonize_opt = kwargs.get("harmonize", "auto")

    if harmonize_strategy is None and kind == "glm":
        harmonize_strategy = _resolve_strategy(
            harmonize_opt,
            glm_mode=True,
            ttest_mode=False,
            sites_provided=bool(site_col),
            confounds_provided=bool(confound_cols),
        )

    input_kind = str(kwargs.get("input") or "r")
    r_to_z = bool(kwargs.get("r_to_z", True))
    scale = "correlation" if (input_kind == "r" and not r_to_z) else "fisher_z"

    if kind == "glm":
        pred_str = ", ".join(interest_cols) if len(interest_cols) > 1 else (interest_cols[0] if interest_cols else "none")
        conf_str = f"[{', '.join(confound_cols)}]" if confound_cols else "none"
        if site_col:
            site_str = f"{site_col} ({harmonize_strategy or harmonize_opt})"
        else:
            site_str = "none"
        return f"predictor={pred_str} | confounds={conf_str} | sites={site_str} | scale={scale}"

    # paired
    parts: List[str] = []
    if interest_cols:
        pred_str = ", ".join(interest_cols) if len(interest_cols) > 1 else interest_cols[0]
        parts.append(f"predictor={pred_str}")
    subj_confs = _as_list(design.get("subject_confounds"))
    all_confs = confound_cols + subj_confs
    if all_confs:
        parts.append(f"confounds=[{', '.join(all_confs)}]")
    if site_col:
        parts.append(f"sites={site_col} (stratified)")
    parts.append(f"scale={scale}")
    return " | ".join(parts)


def _design_matrix_for_qc(
    dataset: Any, cfg: Mapping[str, Any], kwargs: Mapping[str, Any]
) -> tuple[Optional[np.ndarray], List[str], Dict[str, Any]]:
    """Construct the design matrix and extra balance diagnostics for ``qc``."""
    from conninfpy.analysis import (
        COMBAT_SITE_DUMMIES_GLM,
        SITE_DUMMIES_GLM,
        _resolve_strategy,
    )
    from conninfpy.cli import _as_list, _columns, _numeric

    design = cfg["design"]
    kind = design["type"]
    pheno = dataset.pheno.reset_index(drop=True)
    extra: Dict[str, Any] = {}
    site_col = design.get("sites")
    if site_col and site_col in pheno.columns:
        counts = pheno[site_col].dropna().astype(str).value_counts().to_dict()
        extra["site_counts"] = {str(k): int(v) for k, v in counts.items()}
        small_sites = [k for k, v in counts.items() if v < 2]
        if small_sites:
            extra.setdefault("warnings", []).append(
                f"Site(s) with < 2 observations: {small_sites}."
            )

    if kind == "two-sample":
        extra["group_counts"] = {
            str(design["levels"][0]): int(kwargs["group1"].shape[0]),
            str(design["levels"][1]): int(kwargs["group2"].shape[0]),
        }
        return None, [], extra

    if kind == "one-sample":
        extra["n_observations"] = int(kwargs["group1"].shape[0])
        return None, [], extra

    if kind == "glm":
        interest_cols = _as_list(design.get("interest"))
        confound_cols = _as_list(design.get("confounds"))
        used = interest_cols + confound_cols + _as_list(site_col)
        keep = _columns(pheno, used, "design").notna().all(axis=1).to_numpy()
        p = pheno[keep]
        blocks = [np.ones((len(p), 1), dtype=np.float64)]
        names = ["intercept"]
        for c in interest_cols:
            arr = _numeric(_columns(p, [c], "design.interest"))
            blocks.append(arr)
            names.append(c)
        for c in confound_cols:
            arr = _numeric(_columns(p, [c], "design.confounds"))
            blocks.append(arr)
            if arr.shape[1] == 1:
                names.append(c)
            else:
                names.extend([f"{c}_{k}" for k in range(arr.shape[1])])
        harmonize_opt = kwargs.get("harmonize", "auto")
        strategy = _resolve_strategy(
            harmonize_opt,
            glm_mode=True,
            ttest_mode=False,
            sites_provided=bool(site_col),
            confounds_provided=bool(confound_cols),
        )
        if strategy is not None:
            extra["harmonize_strategy"] = strategy
        if site_col and strategy in (COMBAT_SITE_DUMMIES_GLM, SITE_DUMMIES_GLM):
            s_arr = pd.get_dummies(p[site_col].astype(str), drop_first=True, dtype=float)
            if s_arr.shape[1] > 0:
                blocks.append(s_arr.to_numpy(dtype=np.float64))
                names.extend([f"site[{col}]" for col in s_arr.columns])
        return np.hstack(blocks), names, extra

    # paired
    extra["n_pairs"] = int(kwargs["group1"].shape[0])
    interest_obj = kwargs.get("interest")
    if interest_obj is not None:
        n_pairs = int(kwargs["group1"].shape[0])
        blocks = [np.ones((n_pairs, 1), dtype=np.float64)]
        names = ["intercept"]
        if isinstance(interest_obj, Mapping):
            for k, v in interest_obj.items():
                blocks.append(np.asarray(v, dtype=np.float64).reshape(-1, 1))
                names.append(str(k))
        else:
            blocks.append(np.asarray(interest_obj, dtype=np.float64).reshape(-1, 1))
            names.append(str(_as_list(design.get("interest"))[0]))
        if kwargs.get("confounds") is not None:
            c_arr = np.asarray(kwargs["confounds"], dtype=np.float64)
            if c_arr.ndim == 1:
                c_arr = c_arr[:, None]
            blocks.append(c_arr)
            names.extend([f"subject_confound_{i}" for i in range(c_arr.shape[1])])
        return np.hstack(blocks), names, extra
    return None, [], extra


def run_qc(
    config_path: "str | os.PathLike[str]",
    *,
    motion_qc: bool = False,
    as_json: bool = False,
    out_path: "str | os.PathLike[str] | None" = None,
) -> tuple[int, Dict[str, Any]]:
    """Inspect dataset, design matrix conditioning, and optional motion/QC-FC."""
    from conninfpy.analysis import (
        _motion_qc_paired_diagnostics,
        _motion_qc_single_diagnostics,
    )
    from conninfpy.cli import (
        _as_list,
        _atlas,
        _cnbs_bonferroni_warning,
        _jsonable,
        _motion_qc_spec,
        _resolve_cli_net_labels,
        build_analysis,
        load_config,
    )
    from conninfpy.harmonize.diagnostics import design_diagnostics
    from conninfpy.interpret.motion_qc import detect_fd_column
    from conninfpy.loaders import load
    from conninfpy.loaders.validation import validate_loaded_dataset

    cfg_path = Path(config_path).resolve()
    cfg = load_config(cfg_path)
    base = cfg_path.parent
    dataset = load(base / cfg["data"])
    val_report = validate_loaded_dataset(dataset, check_inference_ready=False)

    kwargs, summary = build_analysis(dataset, cfg)
    atlas = _atlas(
        cfg.get("atlas"),
        dataset,
        base,
        summary["n_regions"],
        lut_spec=cfg.get("atlas_lut"),
    )

    warnings_list: List[str] = list(val_report.warnings) + list(summary.get("loader_warnings") or [])
    errors_list: List[str] = list(val_report.errors)

    qc_inference = dict(cfg.get("inference") or {})
    qc_alpha = float(cfg.get("alpha", 0.05))
    try:
        qc_nl, _ = _resolve_cli_net_labels(
            qc_inference, atlas, cfg, base, int(summary["n_regions"])
        )
        cnbs_warn = _cnbs_bonferroni_warning(qc_inference, qc_nl, qc_alpha)
        if cnbs_warn is not None:
            warnings_list.append(cnbs_warn)
    except Exception as exc:
        errors_list.append(str(exc))

    if summary["design"] == "paired":
        n_dropped = int(summary["n_dropped"])
        if n_dropped > 0:
            warnings_list.append(
                f"{n_dropped} of {summary['n_scans_in_data']} scan(s) dropped due to missing design values or incomplete pairs."
            )
    else:
        n_dropped = int(summary["n_subjects_in_data"] - summary["n_subjects_used"])
        if n_dropped > 0:
            warnings_list.append(
                f"{n_dropped} of {summary['n_subjects_in_data']} observation(s) dropped due to missing design values or level filtering."
            )

    pheno = dataset.pheno
    design_cfg = cfg["design"]
    design_cols: List[str] = []
    for key in ("interest", "confounds", "group", "condition", "sites", "subject_confounds"):
        for col in _as_list(design_cfg.get(key)):
            if col not in design_cols:
                design_cols.append(col)
    if design_cfg["type"] == "paired" and not design_cfg.get("condition"):
        cond_col = getattr(dataset, "condition_column", None)
        if cond_col and cond_col not in design_cols:
            design_cols.append(cond_col)
    missing_by_column: Dict[str, int] = {
        col: int(pheno[col].isna().sum())
        for col in design_cols
        if col in pheno.columns
    }

    tr_summary = None
    if "n_valid_tr" in pheno.columns:
        tr_vals = pd.to_numeric(pheno["n_valid_tr"], errors="coerce").dropna()
        if len(tr_vals) > 0:
            tr_min = int(tr_vals.min())
            tr_summary = {
                "min": tr_min,
                "median": float(tr_vals.median()),
                "max": int(tr_vals.max()),
            }
            if tr_min < 50:
                warnings_list.append(f"Minimum n_valid_tr is {tr_min} (< 50 valid frames).")

    X, col_names, balance = _design_matrix_for_qc(dataset, cfg, kwargs)
    warnings_list.extend(balance.pop("warnings", []))
    harmonize_strategy = balance.pop("harmonize_strategy", None)
    model_summary = _format_model_summary(cfg, kwargs, harmonize_strategy=harmonize_strategy)

    design_diag = None
    rank_deficient = False
    if X is not None:
        dd = design_diagnostics(X, names=col_names)
        rank_deficient = bool(dd["rank_deficient"])
        design_diag = {
            "n_rows": int(X.shape[0]),
            "n_cols": int(X.shape[1]),
            "columns": col_names,
            "matrix_rank": int(dd["matrix_rank"]),
            "rank_deficient": rank_deficient,
            "residual_df": int(dd["residual_df"]),
            "condition_number": float(dd["condition_number"]),
            "condition_number_standardized": float(dd["condition_number_standardized"]),
            "vif_max": float(dd["vif_max"]),
            "vif_max_col": str(dd["vif_max_col"]),
            "flags": list(dd["flags"]),
        }
        warnings_list.extend(dd["flags"])
        if rank_deficient:
            errors_list.append(
                f"Design matrix is rank-deficient (rank {dd['matrix_rank']} < {X.shape[1]} columns)."
            )

    # Motion QC: run if opted in or FD column auto-detected
    fd_col = detect_fd_column(pheno)
    mqc_cfg_present = "diagnostics" in cfg and (cfg.get("diagnostics") or {}).get("motion_qc") is not None
    motion_report = None
    if motion_qc or mqc_cfg_present or fd_col is not None:
        spec = _motion_qc_spec(dataset, cfg, kwargs=kwargs, enabled=True)
        if spec is not None:
            input_kind = str(kwargs.get("input") or "r")
            if cfg["design"]["type"] == "paired":
                motion_report = _motion_qc_paired_diagnostics(
                    spec, kwargs["group1"], kwargs["group2"], input_kind, atlas
                )
            elif cfg["design"]["type"] == "two-sample":
                data_arr = np.concatenate([kwargs["group1"], kwargs["group2"]], axis=0)
                motion_report = _motion_qc_single_diagnostics(spec, data_arr, input_kind, atlas)
            elif cfg["design"]["type"] == "one-sample":
                motion_report = _motion_qc_single_diagnostics(spec, kwargs["group1"], input_kind, atlas)
            else:
                motion_report = _motion_qc_single_diagnostics(spec, kwargs["Y"], input_kind, atlas)

    ok = len(errors_list) == 0
    report: Dict[str, Any] = {
        "ok": ok,
        "config": str(cfg_path),
        "data_kind": dataset.data_kind,
        "n_subjects_in_data": int(summary["n_subjects_in_data"]),
        "n_subjects_used": int(summary["n_subjects_used"]),
        "n_dropped": n_dropped,
        "missing_by_column": missing_by_column,
        "n_regions": int(summary["n_regions"]),
        "atlas": getattr(atlas, "name", None) if atlas is not None else None,
        "design": summary["design"],
        "model_summary": model_summary,
        "balance": balance,
        "design_diagnostics": design_diag,
        "n_valid_tr": tr_summary,
        "motion_qc": motion_report,
        "warnings": warnings_list,
        "errors": errors_list,
    }
    if summary["design"] == "paired":
        report["n_scans_in_data"] = int(summary["n_scans_in_data"])
        report["n_pairs_used"] = int(summary["n_pairs_used"])
        report["n_scans_used"] = int(summary["n_scans_used"])
        report["contrast_direction"] = summary["contrast_direction"]

    if out_path is not None:
        Path(out_path).write_text(json.dumps(report, indent=2, default=_jsonable), encoding="utf-8")

    if as_json:
        print(json.dumps(report, indent=2, default=_jsonable))
    else:
        status_txt = "PASS" if ok else "FAIL"
        if summary["design"] == "paired":
            print(
                f"conninfpy qc [{status_txt}] — {summary['design']} | "
                f"{summary['n_scans_in_data']} scans -> {summary['n_pairs_used']} complete pairs "
                f"({n_dropped} dropped) × {summary['n_regions']} regions"
            )
            print(f"  contrast: {summary['contrast_direction']}")
        else:
            print(
                f"conninfpy qc [{status_txt}] — {summary['design']} | "
                f"{summary['n_subjects_used']}/{summary['n_subjects_in_data']} subjects × {summary['n_regions']} regions"
            )
        if model_summary is not None:
            print(f"  model: {model_summary}")
        if missing_by_column:
            miss_str = ", ".join(f"{col}={cnt}" for col, cnt in missing_by_column.items())
            print(f"  missing_by_column: {miss_str}")
        if balance:
            print(f"  balance: {json.dumps(balance)}")
        if design_diag is not None:
            print(
                f"  design: rank={design_diag['matrix_rank']}/{design_diag['n_cols']}, "
                f"df={design_diag['residual_df']}, cond_std={design_diag['condition_number_standardized']:.2f}, "
                f"max_vif={design_diag['vif_max']:.2f} ({design_diag['vif_max_col']})"
            )
        if tr_summary is not None:
            print(f"  n_valid_tr: min={tr_summary['min']}, median={tr_summary['median']:.1f}, max={tr_summary['max']}")
        if motion_report is not None:
            print(f"  motion_qc: status={motion_report.get('status')} (fd_col={fd_col or 'explicit'})")
        for w in warnings_list:
            print(f"  WARNING: {w}")
        for e in errors_list:
            print(f"  ERROR: {e}")

    return (0 if ok else 1), report


def init_config(
    manifest_path: "str | os.PathLike[str]",
    *,
    out_path: "str | os.PathLike[str] | None" = None,
    design_type: Optional[str] = None,
    force: bool = False,
) -> str:
    """Generate a commented ``analysis.yaml`` template from a dataset manifest."""
    from conninfpy.interpret.motion_qc import detect_fd_column
    from conninfpy.loaders import load

    m_path = Path(manifest_path).resolve()
    dataset = load(m_path)
    pheno = dataset.pheno
    fd_col = detect_fd_column(pheno)
    if dataset.data_kind == "timeseries" and isinstance(dataset.data, list) and dataset.data:
        n_regions = int(np.asarray(dataset.data[0]).shape[-1])
    else:
        n_regions = int(np.asarray(dataset.data).shape[-1])

    if out_path is not None:
        out_p = Path(out_path).resolve()
        if out_p.exists() and not force:
            raise FileExistsError(f"{out_p} already exists; pass --force to overwrite.")
        try:
            data_ref = os.path.relpath(m_path, out_p.parent)
        except ValueError:
            data_ref = str(m_path)
    else:
        data_ref = str(manifest_path)

    # Infer default design when not explicitly requested
    cond_col = getattr(dataset, "condition_column", None)
    if design_type is None:
        if cond_col and cond_col in pheno.columns and pheno[cond_col].dropna().nunique() == 2:
            design_type = "paired"
        elif any(c.lower() in ("group", "diagnosis", "dx", "dx_group") and pheno[c].dropna().nunique() == 2 for c in pheno.columns):
            design_type = "two-sample"
        else:
            design_type = "glm"

    lines = [
        f"# ConnInfPy 2.0 analysis configuration generated from {m_path.name}",
        f"# Dataset: kind={dataset.data_kind}, subjects={len(pheno)}, regions={n_regions}",
        "# Phenotype columns:",
    ]
    for col in pheno.columns:
        s = pheno[col]
        n_miss = int(s.isna().sum())
        n_uniq = int(s.dropna().nunique())
        dtype_kind = "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical"
        extra = ""
        if n_uniq <= 6:
            vals = s.dropna().unique().tolist()
            extra = f", levels={vals}"
        lines.append(f"#   - {col} ({dtype_kind}, unique={n_uniq}, missing={n_miss}{extra})")

    lines.append("")
    lines.append(f"data: {data_ref}")
    lines.append("# atlas: bna_246        # or schaefer_100_yeo7, path/to/atlas.csv, path/to/atlas.nii.gz")
    lines.append("# atlas_lut: null       # optional LUT file (CSV/TSV/FreeSurfer txt) for NIfTI atlas")
    if dataset.data_kind == "timeseries":
        lines.append("connectivity:")
        lines.append("  kind: correlation  # or partial_correlation (with optional shrinkage: ledoit_wolf / oas)")
        lines.append("  fisher:")
        lines.append("    policy: clip_r")
        lines.append("    clip_r: 0.9999")
    elif dataset.data_kind == "correlation":
        lines.append("input: r  # or add connectivity.fisher to convert correlation -> Fisher z")

    # Suggest columns
    numeric_cols = [
        c for c in pheno.columns
        if pd.api.types.is_numeric_dtype(pheno[c])
        and c.lower() not in ("subject_id", "sub_id", "id", "participant_id")
        and c != fd_col
    ]
    binary_cols = [c for c in pheno.columns if pheno[c].dropna().nunique() == 2 and c != cond_col]
    site_guess = next((c for c in pheno.columns if c.lower() in ("site", "site_id", "center", "scanner")), None)
    subj_guess = next((c for c in pheno.columns if c.lower() in ("subject_id", "sub_id", "participant_id", "id")), "subject_id")

    lines.append("")
    lines.append("design:")
    lines.append(f"  type: {design_type}")
    lines.append("  # Note: for two-sample with confounds, use type: glm + interest_reference")
    if design_type == "glm":
        interest_guess = next((c for c in numeric_cols if c.lower() in ("age", "score", "symptom")), numeric_cols[0] if numeric_cols else "PREDICTOR_COL")
        lines.append(f"  interest: {interest_guess}")
        if site_guess:
            lines.append(f"  sites: {site_guess}")
        lines.append("  # confounds: [sex, fd]")
    elif design_type == "two-sample":
        grp = next((c for c in binary_cols if c.lower() in ("group", "diagnosis", "dx", "dx_group")), binary_cols[0] if binary_cols else "GROUP_COL")
        lvls = pheno[grp].dropna().unique().tolist()[:2] if grp in pheno.columns else ["control", "patient"]
        lines.append(f"  group: {grp}")
        lines.append(f"  levels: [{lvls[0]}, {lvls[1]}]")
        if site_guess:
            lines.append(f"  sites: {site_guess}")
    elif design_type == "paired":
        c_col = cond_col or next((c for c in binary_cols if "cond" in c.lower() or "state" in c.lower()), "CONDITION_COL")
        lvls = pheno[c_col].dropna().unique().tolist()[:2] if c_col in pheno.columns else ["cond_a", "cond_b"]
        lines.append(f"  condition: {c_col}")
        lines.append(f"  levels: [{lvls[0]}, {lvls[1]}]")
        lines.append(f"  subject: {subj_guess}")
        if fd_col:
            lines.append(f"  # confounds: [{fd_col}]")

    lines.extend([
        "",
        "inference:",
        "  method: tfnbs          # tstat, nbs, tfnbs, cnbs, ni_tfnbs, fbc_tfnbs, bonferroni, bh_fdr, bh_fdr_perm",
        "  n_permutations: 1000   # 5000 recommended for publication",
        "  two_tailed: true",
        "  # missing: available_case",
        "  # net_labels: atlas    # for block methods (cnbs)",
        "  # acceleration: null   # empirical p-values by default",
        "",
        "alpha: 0.05",
        "output: conninfpy_results",
        "",
        "# interpret:",
        "#   cerebellum: anatom",
        "#   methods: true",
    ])
    if fd_col:
        lines.extend([
            "",
            "diagnostics:",
            "  motion_qc:",
            f"    fd: {fd_col}",
        ])

    yaml_text = "\n".join(lines) + "\n"
    if out_path is not None:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(yaml_text, encoding="utf-8")
        print(f"Wrote {out_p}")
    else:
        print(yaml_text, end="")
    return yaml_text


def _rel_manifest_path(target: Path, base_dir: Optional[Path]) -> str:
    if base_dir is None:
        return str(target)
    try:
        return os.path.relpath(target, base_dir)
    except ValueError:
        return str(target)


def _infer_matrix_kind(mat: np.ndarray) -> str:
    finite = mat[np.isfinite(mat)]
    is_unit_diag = bool(
        mat.ndim == 2
        and mat.shape[0] == mat.shape[1]
        and np.allclose(np.diag(mat), 1.0, atol=1e-3, equal_nan=True)
    )
    in_unit_range = bool(
        finite.size > 0
        and float(np.min(finite)) >= -1.0 - 1e-6
        and float(np.max(finite)) <= 1.0 + 1e-6
    )
    return "correlation" if (is_unit_diag or in_unit_range) else "fisher_z"


def _inspect_tabular_sample(sample_file: Path, sep: str) -> tuple[np.ndarray, bool]:
    """Read a sample .tsv/.csv matrix or timeseries file, detecting whether row 0 is a header."""
    with open(sample_file, encoding="utf-8") as fh:
        first_line = fh.readline().strip()
    has_header = False
    if first_line:
        tokens = [t.strip() for t in first_line.split(sep)]
        for tok in tokens:
            if not tok:
                continue
            try:
                float(tok)
            except ValueError:
                has_header = True
                break
    df = pd.read_csv(sample_file, sep=sep, header=0 if has_header else None)
    arr = df.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    return arr, has_header


def init_manifest(
    *,
    data_dir: "str | os.PathLike[str] | None" = None,
    data_file: "str | os.PathLike[str] | None" = None,
    pheno_path: "str | os.PathLike[str] | None" = None,
    out_path: "str | os.PathLike[str] | None" = None,
    kind: str = "auto",
    atlas: Optional[str] = None,
    id_col: Optional[str] = None,
    name: Optional[str] = None,
    force: bool = False,
) -> str:
    """Generate a v1 ``DatasetManifest`` YAML from a data directory or 3D NumPy file."""
    from conninfpy.loaders.builtins import _match_subject_id_column, _read_pheno_table

    if (data_dir is None) == (data_file is None):
        raise ValueError("Provide exactly one of --data-dir or --data-file.")

    out_p: Optional[Path] = None
    base_dir: Optional[Path] = None
    if out_path is not None:
        out_p = Path(out_path).resolve()
        if out_p.exists() and not force:
            raise FileExistsError(f"{out_p} already exists; pass --force to overwrite.")
        base_dir = out_p.parent

    pheno_p: Optional[Path] = None
    detected_id_col: Optional[str] = None
    n_pheno_rows: Optional[int] = None
    if pheno_path is not None:
        pheno_p = Path(pheno_path).resolve()
        if not pheno_p.is_file():
            raise FileNotFoundError(f"Phenotype file not found: {pheno_p}")
        pheno_df = _read_pheno_table(pheno_p)
        n_pheno_rows = len(pheno_df)
        if id_col is not None:
            if id_col not in pheno_df.columns:
                raise ValueError(
                    f"--id-col {id_col!r} not found in {pheno_p.name}; "
                    f"available: {list(pheno_df.columns)}."
                )
            detected_id_col = id_col
        else:
            detected_id_col = _match_subject_id_column(pheno_df)

    atlas_ref: Optional[str] = None
    if atlas is not None:
        if atlas in ("bna_246", "bna246.csv"):
            atlas_ref = "BUILTIN:bna246.csv"
        elif Path(atlas).exists():
            atlas_ref = _rel_manifest_path(Path(atlas).resolve(), base_dir)
        else:
            atlas_ref = atlas

    paths: Dict[str, Any] = {}
    params: Dict[str, Any] = {}
    checks: Dict[str, Any] = {}
    n_subjects_found: Optional[int] = None
    n_rois_found: Optional[int] = None
    dataset_name = name

    if data_file is not None:
        df_p = Path(data_file).resolve()
        if not df_p.is_file():
            raise FileNotFoundError(f"Data file not found: {df_p}")
        dataset_name = dataset_name or df_p.stem
        loader_name = "NumpyLoader"
        loaded = np.load(df_p, allow_pickle=True)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            first_key = loaded.files[0]
            arr = np.asarray(loaded[first_key])
        else:
            arr = np.asarray(loaded)
        if arr.ndim != 3:
            raise ValueError(f"Expected a 3D array in {df_p}; got shape {arr.shape}.")
        n_subjects_found = int(arr.shape[0])
        n_rois_found = int(arr.shape[2])
        if kind == "auto":
            data_kind = "timeseries" if arr.shape[1] != arr.shape[2] else _infer_matrix_kind(arr[0])
        else:
            data_kind = kind
        paths["data_path"] = _rel_manifest_path(df_p, base_dir)
        if pheno_p is not None:
            paths["pheno_path"] = _rel_manifest_path(pheno_p, base_dir)
        if atlas_ref is not None:
            paths["atlas_metadata"] = atlas_ref
        params["data_kind"] = data_kind
    else:
        assert data_dir is not None
        dd_p = Path(data_dir).resolve()
        if not dd_p.is_dir():
            raise NotADirectoryError(f"Data directory not found: {dd_p}")
        dataset_name = dataset_name or dd_p.name

        bold_files = sorted(dd_p.rglob("*desc-preproc_bold.nii.gz"))
        if bold_files:
            loader_name = "FmriprepDerivativesLoader"
            n_subjects_found = len(bold_files)
            paths["derivatives_dir"] = _rel_manifest_path(dd_p, base_dir)
            if pheno_p is not None:
                paths["pheno_path"] = _rel_manifest_path(pheno_p, base_dir)
            if atlas_ref is not None:
                paths["atlas_image_path"] = atlas_ref
        else:
            pheno_abs = pheno_p.resolve() if pheno_p is not None else None
            chosen_ext: Optional[str] = None
            matched_files: List[Path] = []
            for ext in (".npy", ".tsv", ".csv"):
                cands = sorted(dd_p.glob(f"*{ext}"))
                if not cands:
                    cands = sorted(dd_p.rglob(f"*{ext}"))
                cands = [
                    p for p in cands
                    if (pheno_abs is None or p.resolve() != pheno_abs)
                    and p.name.lower() not in ("participants.tsv", "participants.csv")
                ]
                if cands:
                    chosen_ext = ext
                    matched_files = cands
                    break

            if not matched_files or chosen_ext is None:
                raise FileNotFoundError(
                    f"No supported data files (.npy, .tsv, .csv, *desc-preproc_bold.nii.gz) found in {dd_p}."
                )

            n_subjects_found = len(matched_files)
            sample = matched_files[0]
            sep: Optional[str] = None
            has_header: Optional[bool] = None
            if chosen_ext == ".npy":
                arr = np.asarray(np.load(sample), dtype=np.float64)
            else:
                sep = "\t" if chosen_ext == ".tsv" else ","
                arr, has_header = _inspect_tabular_sample(sample, sep=sep)

            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array in sample file {sample}; got shape {arr.shape}.")

            n_rois_found = int(arr.shape[1])
            is_square = arr.shape[0] == arr.shape[1]
            is_sym = is_square and bool(np.allclose(arr, arr.T, atol=1e-4, equal_nan=True))

            if kind == "timeseries" or (kind == "auto" and not (is_square and is_sym)):
                loader_name = "TimeseriesDirectoryLoader"
                paths["dir_path"] = _rel_manifest_path(dd_p, base_dir)
                if pheno_p is not None:
                    paths["pheno_path"] = _rel_manifest_path(pheno_p, base_dir)
                if atlas_ref is not None:
                    paths["atlas"] = atlas_ref
                params["pattern"] = f"*{chosen_ext}"
                if sep is not None:
                    params["sep"] = sep
                if has_header is not None:
                    params["header"] = bool(has_header)
            else:
                loader_name = "ConnectivityMatrixLoader"
                data_kind = _infer_matrix_kind(arr) if kind == "auto" else kind
                paths["dir_path"] = _rel_manifest_path(dd_p, base_dir)
                if pheno_p is not None:
                    paths["pheno_path"] = _rel_manifest_path(pheno_p, base_dir)
                if atlas_ref is not None:
                    paths["atlas"] = atlas_ref
                params["pattern"] = f"*{chosen_ext}"
                params["data_kind"] = data_kind
                if sep is not None:
                    params["sep"] = sep
                if has_header is not None:
                    params["header"] = bool(has_header)

    if n_rois_found is not None:
        checks["expected_rois"] = n_rois_found

    manifest_dict: Dict[str, Any] = {
        "schema_version": 1,
        "name": dataset_name,
        "loader": loader_name,
        "paths": paths,
    }
    if params:
        manifest_dict["params"] = params
    if checks:
        manifest_dict["checks"] = checks

    header_lines = [
        f"# ConnInfPy dataset manifest (loader={loader_name})",
        f"# Detected: subjects={n_subjects_found}, rois={n_rois_found}"
        + (f", pheno_rows={n_pheno_rows}, id_col={detected_id_col}" if pheno_p is not None else ""),
    ]
    yaml_text = "\n".join(header_lines) + "\n" + yaml.safe_dump(manifest_dict, sort_keys=False)

    if out_p is not None:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(yaml_text, encoding="utf-8")
        print(f"Wrote {out_p}")
    else:
        print(yaml_text, end="")
    return yaml_text


def _is_trivial_roi_ids(roi_ids: Sequence[Any]) -> bool:
    """Return True if ``roi_ids`` are default 0..N-1, 1..N, or generic ``ROI_i``/``region_i`` placeholders."""
    n = len(roi_ids)
    if n == 0:
        return True
    str_ids = [str(x) for x in roi_ids]
    return any(
        str_ids == [fmt.format(i=i) for i in range(start, start + n)]
        for start in (0, 1)
        for fmt in ("{i}", "ROI_{i}", "region_{i}")
    )


def _validate_atlas_roi_ids(atlas: Any, bundle_roi_ids: Optional[Sequence[Any]]) -> None:
    """Ensure a resolved atlas agrees with non-trivial ``bundle.roi_ids``."""
    if bundle_roi_ids is None or _is_trivial_roi_ids(bundle_roi_ids):
        return
    str_ids = [str(x) for x in bundle_roi_ids]
    atlas_labels = [str(x) for x in (getattr(atlas, "labels", None) or ())]
    if atlas_labels and str_ids == atlas_labels:
        return
    atlas_roi_ids = getattr(atlas, "roi_ids", None)
    if atlas_roi_ids is not None and str_ids == [str(x) for x in atlas_roi_ids]:
        return
    atlas_label_ids = getattr(atlas, "label_ids", None)
    if atlas_label_ids is not None and str_ids == [str(x) for x in atlas_label_ids]:
        return
    raise ValueError(
        f"Atlas ROI labels do not match bundle roi_ids "
        f"(bundle[:3]={str_ids[:3]!r} vs atlas[:3]={atlas_labels[:3]!r}); "
        f"refusing to attach mismatched atlas metadata."
    )


def _resolve_atlas_for_interpret(
    atlas_spec: Optional[str],
    provenance_path: Optional[Path],
    n_regions: int,
    atlas_lut: Optional[str] = None,
    bundle_roi_ids: Optional[Sequence[Any]] = None,
) -> Any:
    from conninfpy.atlas.metadata import AtlasInfo
    from conninfpy.cli import _BUILTIN_ATLASES, _is_nifti_path

    def _load_file_atlas(path: Path, lut: Optional[Path]) -> Any:
        if _is_nifti_path(str(path)):
            return AtlasInfo.from_nifti(str(path), lut=str(lut) if lut is not None else None)
        return AtlasInfo.from_csv(str(path))

    atlas = None
    if atlas_spec is not None:
        if atlas_spec in _BUILTIN_ATLASES:
            atlas = getattr(AtlasInfo, atlas_spec)()
        else:
            lut_path = Path(atlas_lut) if atlas_lut is not None else None
            atlas = _load_file_atlas(Path(atlas_spec), lut_path)
        if len(atlas) != n_regions:
            raise ValueError(f"atlas has {len(atlas)} regions but the data has {n_regions}.")
        _validate_atlas_roi_ids(atlas, bundle_roi_ids)
        return atlas

    cfg_atlas = None
    if provenance_path is not None and provenance_path.is_file():
        try:
            prov = json.loads(provenance_path.read_text(encoding="utf-8"))
            cfg = prov.get("config") or {}
            cfg_atlas = cfg.get("atlas")
            cfg_lut = atlas_lut or cfg.get("atlas_lut")
            if isinstance(cfg_atlas, str):
                if cfg_atlas in _BUILTIN_ATLASES:
                    atlas = getattr(AtlasInfo, cfg_atlas)()
                else:
                    base_candidates = []
                    if prov.get("analysis_file"):
                        base_candidates.append(Path(prov["analysis_file"]).parent)
                    base_candidates.append(Path(provenance_path).parent)
                    for base_dir in base_candidates:
                        cand = base_dir / cfg_atlas
                        if cand.is_file():
                            lut_path = None
                            if cfg_lut is not None:
                                lut_cand = Path(cfg_lut)
                                lut_path = lut_cand if lut_cand.is_absolute() else (base_dir / lut_cand)
                            atlas = _load_file_atlas(cand, lut_path)
                            break
                if atlas is not None:
                    if len(atlas) != n_regions:
                        raise ValueError(f"atlas has {len(atlas)} regions but the data has {n_regions}.")
                    _validate_atlas_roi_ids(atlas, bundle_roi_ids)
                    return atlas
        except ValueError:
            raise
        except Exception:
            pass

    if isinstance(cfg_atlas, str):
        msg = (
            f"conninfpy interpret: atlas {cfg_atlas!r} from provenance.json was not found; "
            f"returning None without guessing an atlas by n_regions={n_regions} (pass --atlas to supply it)."
        )
    else:
        msg = (
            f"conninfpy interpret: no atlas specified or resolved from provenance.json; "
            f"returning None without guessing an atlas by n_regions={n_regions} (pass --atlas to supply it)."
        )
    warnings.warn(msg, UserWarning, stacklevel=2)
    print(msg, file=sys.stderr)
    return None


def _reconstruct_inference_result(bundle: Any, pvals_path: Optional[Path]) -> Any:
    """Build an ``InferenceResult`` from ``pvals.npz`` + ``ResultBundle`` for summaries/plots."""
    from conninfpy.results import InferenceResult

    e_grid = None
    h_grid = None
    if pvals_path is not None and pvals_path.is_file():
        with np.load(pvals_path, allow_pickle=False) as z:
            pos = np.array(z["positive"]) if "positive" in z else np.array(bundle.corrected_p["positive"])
            neg = np.array(z["negative"]) if "negative" in z else np.array(bundle.corrected_p["negative"])
            stat_signed = np.array(z["stat_signed"]) if "stat_signed" in z else None
            beta = np.array(z["beta"]) if "beta" in z else None
            if "e_grid" in z:
                e_grid = np.array(z["e_grid"], dtype=np.float64)
            if "h_grid" in z:
                h_grid = np.array(z["h_grid"], dtype=np.float64)
    else:
        pos = np.array(bundle.corrected_p["positive"])
        neg = np.array(bundle.corrected_p["negative"])
        stat_signed = None
        beta = None

    meta = bundle.metadata
    if e_grid is None and meta.get("e_grid") is not None:
        e_grid = np.asarray(meta["e_grid"], dtype=np.float64)
    if h_grid is None and meta.get("h_grid") is not None:
        h_grid = np.asarray(meta["h_grid"], dtype=np.float64)

    stat_pos = np.maximum(stat_signed, 0.0) if stat_signed is not None else None
    stat_neg = np.maximum(-stat_signed, 0.0) if stat_signed is not None else None
    native = meta.get("native_report")
    unit = meta.get("inference_unit", "edge")
    return InferenceResult(
        pos,
        neg,
        method=str(meta.get("method", "unknown")),
        n_permutations=int(meta.get("n_permutations", 0)),
        acceleration=meta.get("acceleration"),
        stat_positive=stat_pos,
        stat_negative=stat_neg,
        two_tailed=(meta.get("tail_policy") == "two-sided"),
        tested_tail=meta.get("tested_tail"),
        tail_selection=meta.get("tail_selection"),
        beta=beta,
        design=meta.get("design"),
        e_grid=e_grid,
        h_grid=h_grid,
        analysis_provenance={
            "estimable": bundle.estimable,
            "tested": bundle.tested,
        },
        missing_report=meta.get("missing_report"),
        harmonized=bool(
            meta.get("resolved_strategy") in ("combat_only", "combat_site_dummies_glm")
        ),
        combat_diagnostics=meta.get("combat_diagnostics"),
        component_report=native if unit == "component" else None,
        block_report=native if unit == "block" else None,
    )


def interpret_outputs(
    target: "str | os.PathLike[str]",
    *,
    atlas_spec: Optional[str] = None,
    atlas_lut: Optional[str] = None,
    cerebellum: Optional[str] = None,
    alpha: Optional[float] = None,
    out_dir: "str | os.PathLike[str] | None" = None,
    no_plots: bool = False,
    support: Optional[str] = None,
    radius_mm: float = 6.0,
    parcel_nifti: "str | os.PathLike[str] | None" = None,
    space: Optional[str] = None,
    fig_format: str = "png",
    dpi: int = 300,
    methods: bool = False,
    export_brainnet: bool = False,
    brainnet_render: bool = False,
    brainnet_bin: Optional[str] = None,
    param_idx: Optional[int] = None,
) -> Dict[str, Path]:
    """Export post-analysis summaries, node/network tables, BNA annotations, and figures."""
    from conninfpy.atlas.brainnetome import BrainnetomeReference
    from conninfpy.interpret.annotate import (
        annotate_bna_regions,
        annotate_rois,
        bna_region_ids,
        summarize_annotation,
    )
    from conninfpy.post_analysis import ResultBundle

    if fig_format not in ("png", "pdf", "svg"):
        raise ValueError(f"format must be 'png', 'pdf', or 'svg'; got {fig_format!r}.")
    if dpi <= 0:
        raise ValueError(f"dpi must be positive; got {dpi}.")
    if support is not None and support not in ("point", "sphere", "parcel"):
        raise ValueError(f"support must be 'point', 'sphere', or 'parcel'; got {support!r}.")
    if support == "parcel" and parcel_nifti is None:
        raise ValueError("--support parcel requires --parcel-nifti <path>.")

    target_path = Path(target).resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Target does not exist: {target_path}")

    if target_path.is_file():
        bundle_paths = [target_path]
        base_dir = target_path.parent
        prov_path = base_dir / "provenance.json"
        prov_data_early: Optional[Dict[str, Any]] = None
        if prov_path.is_file():
            try:
                prov_data_early = json.loads(prov_path.read_text(encoding="utf-8"))
            except Exception:
                prov_data_early = None
    else:
        base_dir = target_path
        prov_path = base_dir / "provenance.json"
        prov_data_early = None
        if prov_path.is_file():
            try:
                prov_data_early = json.loads(prov_path.read_text(encoding="utf-8"))
            except Exception:
                prov_data_early = None
        prov_bundles = prov_data_early.get("bundles") if isinstance(prov_data_early, Mapping) else None
        if isinstance(prov_bundles, Mapping) and len(prov_bundles) > 0:
            bundle_names = [str(v) for v in prov_bundles.values()]
        elif isinstance(prov_bundles, Sequence) and not isinstance(prov_bundles, (str, bytes)) and len(prov_bundles) > 0:
            bundle_names = [str(b) for b in prov_bundles]
        else:
            bundle_names = None

        if bundle_names is not None:
            bundle_paths = [base_dir / b for b in bundle_names]
            missing = [str(p) for p in bundle_paths if not p.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Bundle file(s) listed in {prov_path} not found: {missing}"
                )
        else:
            bundle_paths = sorted(target_path.glob("bundle*.pkl"))
            if not bundle_paths:
                raise FileNotFoundError(f"No bundle*.pkl files found in {target_path}")

    eff_alpha = alpha
    if eff_alpha is None and isinstance(prov_data_early, Mapping):
        try:
            eff_alpha = float(prov_data_early.get("alpha", 0.05))
        except Exception:
            eff_alpha = 0.05
    if eff_alpha is None:
        eff_alpha = 0.05

    dest = Path(out_dir).resolve() if out_dir else base_dir / "interpretation"
    dest.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    cb_arg = None if cerebellum in (None, "none") else cerebellum
    ref = None

    for bp in bundle_paths:
        stem = bp.stem  # "bundle" or "bundle_<contrast>"
        suffix = stem[len("bundle"):]
        bundle = ResultBundle.load(bp)
        n_regions = len(bundle.roi_ids)
        atlas = _resolve_atlas_for_interpret(
            atlas_spec,
            prov_path if prov_path.is_file() else None,
            n_regions,
            atlas_lut=atlas_lut,
            bundle_roi_ids=bundle.roi_ids,
        )
        pvals_path = base_dir / f"pvals{suffix}.npz"
        inf_res = _reconstruct_inference_result(bundle, pvals_path if pvals_path.is_file() else None)
        is_grid = getattr(inf_res, "is_grid", False)
        if is_grid:
            k_grid = int(inf_res["positive"].shape[2])
            if param_idx is None:
                eff_param_idx: Optional[int] = 0
                e_grid = getattr(inf_res, "e_grid", None)
                h_grid = getattr(inf_res, "h_grid", None)
                e_str = f"{float(e_grid[0]):.4g}" if e_grid is not None and len(e_grid) > 0 else "unknown"
                h_str = f"{float(h_grid[0]):.4g}" if h_grid is not None and len(h_grid) > 0 else "unknown"
                grid_msg = (
                    f"conninfpy interpret: {bp.name} holds an (E, H) grid ({k_grid} cells); "
                    f"defaulting to cell idx=0 (E={e_str}, H={h_str}). "
                    f"Pass --param-idx <0..{k_grid - 1}> to select another cell."
                )
                warnings.warn(grid_msg, UserWarning, stacklevel=2)
                print(grid_msg, file=sys.stderr)
            else:
                if not (0 <= int(param_idx) < k_grid):
                    raise ValueError(
                        f"--param-idx={param_idx} is out of range for grid of size {k_grid} "
                        f"(valid: 0..{k_grid - 1})."
                    )
                eff_param_idx = int(param_idx)
        else:
            if param_idx is not None and int(param_idx) != 0:
                raise ValueError(
                    f"--param-idx={param_idx} is only valid for an (E, H) grid result."
                )
            eff_param_idx = None

        # 1. Bundle summaries for available groupings & tails
        contrast_id = bundle.metadata.get("contrast_id")
        measure_name = (bundle.effect_spec or {}).get("name")
        for tail in sorted(bundle.corrected_p.keys()):
            pairs_df = bundle.detected_pairs(tail=tail, param_idx=eff_param_idx, alpha=eff_alpha)
            pairs_path = dest / f"detected_pairs{suffix}_{tail}.csv"
            pairs_df.to_csv(pairs_path, index=False)
            written[f"detected_pairs{suffix}_{tail}"] = pairs_path

            for grp_name in bundle.groupings.keys():
                if measure_name and contrast_id:
                    s_df = bundle.summarize(
                        grouping=grp_name,
                        measure=measure_name,
                        contrast_id=contrast_id,
                        tail=tail,
                        param_idx=eff_param_idx,
                        alpha=eff_alpha,
                    )
                    s_path = dest / f"bundle_summary{suffix}_{grp_name}_{tail}.csv"
                    ResultBundle.export_summary(s_df, s_path)
                    written[f"bundle_summary{suffix}_{grp_name}_{tail}"] = s_path

        # 2. Node / Network / Component summaries
        try:
            node_df = inf_res.node_summary(atlas=atlas, alpha=eff_alpha, param_idx=eff_param_idx)
            if atlas is None and not _is_trivial_roi_ids(bundle.roi_ids) and "roi_id" not in node_df.columns:
                node_df.insert(1, "roi_id", [bundle.roi_ids[int(r)] for r in node_df["roi"]])
            node_path = dest / f"node_summary{suffix}.csv"
            node_df.to_csv(node_path, index=False)
            written[f"node_summary{suffix}"] = node_path
        except ValueError:
            pass

        if atlas is not None and getattr(atlas, "networks", None) is not None:
            net_df = inf_res.network_summary(atlas, alpha=eff_alpha, param_idx=eff_param_idx)
            net_path = dest / f"network_summary{suffix}.csv"
            net_df.to_csv(net_path, index=False)
            written[f"network_summary{suffix}"] = net_path

        if inf_res.inference_unit == "component" and getattr(inf_res, "component_report", None) is not None:
            comp_df = inf_res.component_table(atlas=atlas, alpha=eff_alpha)
            comp_path = dest / f"component_table{suffix}.csv"
            comp_df.to_csv(comp_path, index=False)
            written[f"component_table{suffix}"] = comp_path

        # 3. Brainnetome (+ optional SUIT cerebellum) annotation
        rois = None
        res_mm = 1 if cb_arg is not None else 2
        bna_ids = (
            bna_region_ids(atlas.labels)
            if atlas is not None and getattr(atlas, "labels", None) is not None
            else None
        )
        if bna_ids is not None:
            if ref is None:
                ref = BrainnetomeReference.load(profiles=False, resolution_mm=res_mm, cerebellum=cb_arg)
            rois = annotate_bna_regions(bna_ids, reference=ref, names=atlas.labels)
        else:
            eff_support = support or ("parcel" if parcel_nifti is not None else "sphere")
            eff_space = space or (getattr(atlas, "space", None) if atlas is not None else None) or "MNI152"
            coords_arr = None
            if atlas is not None and getattr(atlas, "coords", None) is not None:
                cand = np.asarray(atlas.coords, dtype=float)
                if cand.ndim == 2 and cand.shape[1] == 3 and np.isfinite(cand).all():
                    coords_arr = cand
            if eff_support == "parcel" and parcel_nifti is not None:
                if ref is None:
                    ref = BrainnetomeReference.load(profiles=False, resolution_mm=res_mm, cerebellum=cb_arg)
                if all(isinstance(x, (int, np.integer)) and int(x) > 0 for x in bundle.roi_ids):
                    p_labels = [int(x) for x in bundle.roi_ids]
                else:
                    p_labels = list(range(1, n_regions + 1))
                rois = annotate_rois(
                    coords=coords_arr,
                    reference=ref,
                    names=getattr(atlas, "labels", None) if atlas is not None else None,
                    networks=getattr(atlas, "networks", None) if atlas is not None else None,
                    space=eff_space,
                    support="parcel",
                    parcel_image=str(Path(parcel_nifti).resolve()),
                    parcel_labels=p_labels,
                )
            elif coords_arr is not None and eff_support in ("point", "sphere"):
                if ref is None:
                    ref = BrainnetomeReference.load(profiles=False, resolution_mm=res_mm, cerebellum=cb_arg)
                rois = annotate_rois(
                    coords_arr,
                    reference=ref,
                    names=getattr(atlas, "labels", None),
                    networks=getattr(atlas, "networks", None),
                    space=eff_space,
                    support=eff_support,
                    radius_mm=radius_mm if eff_support == "sphere" else None,
                )

        if rois is not None:
            ann_edges = inf_res.annotated_edges(rois, alpha=eff_alpha, param_idx=eff_param_idx)
            ann_path = dest / f"annotated_edges{suffix}.csv"
            ann_edges.to_csv(ann_path, index=False)
            written[f"annotated_edges{suffix}"] = ann_path

            ann_sum = summarize_annotation(inf_res, rois, alpha=eff_alpha, param_idx=eff_param_idx)
            for key in ("relations", "regions", "domains", "distance_qc"):
                if key in ann_sum and isinstance(ann_sum[key], pd.DataFrame):
                    kp = dest / f"annotation_{key}{suffix}.csv"
                    ann_sum[key].to_csv(kp, index=False)
                    written[f"annotation_{key}{suffix}"] = kp

        # 4. Plots
        if not no_plots:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from conninfpy.plot import plot_connectogram, plot_coupling_classes, summary_figure

            plot_res = inf_res.select(eff_param_idx) if is_grid else inf_res
            fig = summary_figure(plot_res, atlas=atlas, alpha=eff_alpha)
            fig_path = dest / f"summary_figure{suffix}.{fig_format}"
            fig.savefig(fig_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            written[f"summary_figure{suffix}"] = fig_path

            design_dict = getattr(inf_res, "design", None) or {}
            if inf_res.beta is not None and design_dict.get("comparison"):
                fig_c, ax_c = plt.subplots(figsize=(6.5, 5.5))
                plot_coupling_classes(plot_res, atlas=atlas, alpha=eff_alpha, ax=ax_c)
                c_path = dest / f"coupling_classes{suffix}.{fig_format}"
                fig_c.savefig(c_path, dpi=dpi, bbox_inches="tight", facecolor="white")
                plt.close(fig_c)
                written[f"coupling_classes{suffix}"] = c_path

            if rois is not None:
                nsig = inf_res.n_significant(eff_alpha, param_idx=eff_param_idx)
                for tail in ("positive", "negative"):
                    if nsig.get(tail, 0) > 0:
                        fig_g, ax_g = plt.subplots(figsize=(7.5, 7.5))
                        plot_connectogram(inf_res, rois, tail=tail, alpha=eff_alpha, param_idx=eff_param_idx, ax=ax_g)
                        g_path = dest / f"connectogram{suffix}_{tail}.{fig_format}"
                        fig_g.savefig(g_path, dpi=dpi, bbox_inches="tight", facecolor="white")
                        plt.close(fig_g)
                        written[f"connectogram{suffix}_{tail}"] = g_path

        # 5. Deterministic Methods boilerplate & BrainNet Viewer export
        if methods:
            from conninfpy.reporting import methods_boilerplate

            prov_data = None
            if prov_path.is_file():
                try:
                    prov_data = json.loads(prov_path.read_text(encoding="utf-8"))
                except Exception:
                    prov_data = None
            result_key = suffix.lstrip("_") or "result"
            source_obj = prov_data if prov_data is not None else bundle
            text = methods_boilerplate(
                source_obj,
                alpha=eff_alpha,
                atlas=atlas,
                cerebellum=cb_arg,
                result_name=result_key,
            )
            methods_path = dest / f"methods_boilerplate{suffix}.md"
            methods_path.write_text(text, encoding="utf-8")
            written[f"methods_boilerplate{suffix}"] = methods_path

        if export_brainnet or brainnet_render:
            from conninfpy.reporting import (
                export_brainnet as _export_brainnet,
                render_brainnet as _render_brainnet,
            )

            try:
                bn_paths = _export_brainnet(
                    inf_res,
                    atlas,
                    dest / f"brainnet{suffix}",
                    alpha=eff_alpha,
                    param_idx=eff_param_idx,
                    write_scene=True,
                )
                written[f"brainnet_node{suffix}"] = bn_paths["node"]
                written[f"brainnet_edge{suffix}"] = bn_paths["edge"]
                written[f"brainnet_scene{suffix}"] = bn_paths["scene"]
                if brainnet_render:
                    try:
                        png_path = _render_brainnet(
                            bn_paths["scene"],
                            dest / f"brainnet{suffix}.png",
                            executable=brainnet_bin,
                        )
                        written[f"brainnet_png{suffix}"] = png_path
                    except (FileNotFoundError, RuntimeError) as exc:
                        print(
                            f"conninfpy interpret: skipping BrainNet render for {bp.name}: {exc}",
                            file=sys.stderr,
                        )
            except ValueError as exc:
                print(
                    f"conninfpy interpret: skipping BrainNet export for {bp.name}: {exc}",
                    file=sys.stderr,
                )

    print(f"conninfpy interpret — wrote {len(written)} artifact(s) to {dest}")
    return written


def collect_info() -> Dict[str, Any]:
    """Collect read-only environment, dependency, backend, and resource info.

    Import-conservative: deliberately does not import ``numba`` or optional
    heavy packages (uses ``importlib.util.find_spec`` and distribution
    metadata only).
    """
    import platform
    from importlib import metadata, util

    from conninfpy import __version__
    from conninfpy.cli import _numba_cache_line
    from conninfpy.stats._engine import get_available_cores
    from conninfpy.stats._workers import describe_worker_policy

    dep_names = ("numpy", "scipy", "pandas", "numba", "nibabel", "nilearn", "streamlit")
    deps: Dict[str, Optional[str]] = {}
    for pkg in dep_names:
        try:
            spec = util.find_spec(pkg)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            deps[pkg] = None
        else:
            try:
                deps[pkg] = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                deps[pkg] = "installed"

    numba_installed = deps["numba"] is not None
    policy = describe_worker_policy(use_mp=True, n_processes=None, n_permutations=None)

    atlas_dir = Path(__file__).resolve().parent / "atlas_data"
    bna_ok = (atlas_dir / "bna246.csv").is_file() and (atlas_dir / "BN_Atlas_246_1mm.nii.gz").is_file()
    suit_ok = (atlas_dir / "atl-Anatom.tsv").is_file() and (
        atlas_dir / "atl-Anatom_space-MNI_dseg.nii.gz"
    ).is_file()

    return {
        "conninfpy_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "os": platform.system(),
        "dependencies": deps,
        "numba": {
            "installed": numba_installed,
            "version": deps["numba"],
            "imported": False,
            "cache": _numba_cache_line(numba_installed),
            "cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
            "disable_jit": os.environ.get("NUMBA_DISABLE_JIT"),
        },
        "cpu": {
            "cores": os.cpu_count(),
            "usable": get_available_cores(),
            "worker_start_method": policy["worker_start_method"],
        },
        "resources": {
            "bna_246": bna_ok,
            "suit_cerebellum": suit_ok,
        },
    }


def _find_nbs_schizophrenia_demo() -> Optional[Path]:
    """Return the NBS schizophrenia manifest path if the demo dataset is present."""
    repo_root = Path(__file__).resolve().parents[1]
    manifest = repo_root / "datasets" / "nbs_schizophrenia.yaml"
    demo_dir = repo_root / "datasets" / "demo" / "NBS_schizophrenia"
    if (
        manifest.is_file()
        and (demo_dir / "nbs_schizophrenia.npy").is_file()
        and (demo_dir / "pheno.csv").is_file()
    ):
        return manifest
    return None


def _run_synthetic_nbs_fallback() -> tuple[bool, str]:
    """Run a deterministic synthetic NBS check when demo files are absent."""
    from conninfpy import analyze
    from conninfpy.sim import TopologyDatasetGenerator

    gen = TopologyDatasetGenerator(n_nodes=20, n_modules=2, seed=42)
    ds = gen.generate("chain", effect_size=0.5, n_samples=12, time_points=80)
    res = analyze(
        group1=ds.group1,
        group2=ds.group2,
        test_type="two-sample",
        input="r",
        method="nbs",
        threshold=2.5,
        n_permutations=200,
        use_mp=False,
        rng=42,
    )
    inf = res.inference
    iu = np.triu_indices(20, 1)
    n_sig = int(np.sum(np.asarray(inf["positive"])[iu] <= 0.05))
    ok = n_sig > 0 and bool(np.isfinite(inf["positive"]).all())
    return ok, f"synthetic NBS fallback n_sig_pos={n_sig}"


def run_selftest(*, as_json: bool = False, reference: bool = False) -> tuple[int, Dict[str, Any]]:
    """Run a fast, zero-external-data functional check of TFNBS, MP, and bundled atlases."""
    t_start = time.perf_counter()
    checks: Dict[str, Dict[str, Any]] = {}
    backend_info: Dict[str, Any] = {"has_numba": False, "actual_backend": "unknown"}

    # 1. Synthetic dataset generation
    t0 = time.perf_counter()
    ds = None
    try:
        from conninfpy.sim import TopologyDatasetGenerator

        gen = TopologyDatasetGenerator(n_nodes=20, n_modules=2, seed=0)
        ds = gen.generate("chain", effect_size=0.3, n_samples=8, time_points=60)
        ok_shape = ds.group1.shape == (8, 20, 20) and ds.group2.shape == (8, 20, 20)
        checks["synthetic_data"] = {
            "status": "PASS" if ok_shape else "FAIL",
            "wall_time_s": round(time.perf_counter() - t0, 3),
            "detail": f"shape={ds.group1.shape}",
        }
    except Exception as exc:
        checks["synthetic_data"] = {
            "status": "FAIL",
            "wall_time_s": round(time.perf_counter() - t0, 3),
            "detail": f"{type(exc).__name__}: {exc}",
        }

    # 2. Exact TFNBS + 2-worker MP + threadpoolctl
    t0 = time.perf_counter()
    if ds is not None:
        try:
            from conninfpy.analysis import AnalyzeResult, analyze
            from conninfpy.enhance._backend import HAS_NUMBA, _enhance_call_scope

            backend_info["has_numba"] = bool(HAS_NUMBA)
            with _enhance_call_scope() as ctx:
                res = analyze(
                    group1=ds.group1,
                    group2=ds.group2,
                    test_type="two-sample",
                    input="r",
                    method="tfnbs",
                    n_permutations=20,
                    n_processes=2,
                    use_mp=True,
                    rng=0,
                )
                actual = ctx.get("actual_backend") or ("numba" if HAS_NUMBA else "scipy")
            backend_info["actual_backend"] = str(actual)
            try:
                import threadpoolctl  # type: ignore[import-untyped]  # noqa: F401
                tp_avail = True
            except ImportError:
                tp_avail = False
            backend_info["threadpoolctl"] = tp_avail
            assert isinstance(res, AnalyzeResult)
            inf = res.inference
            pos_map = np.asarray(inf["positive"], dtype=float)
            neg_map = np.asarray(inf["negative"], dtype=float)
            ok_tfnbs = (
                pos_map.shape == (20, 20)
                and bool(np.isfinite(pos_map).all())
                and bool(np.isfinite(neg_map).all())
            )
            checks["tfnbs_mp"] = {
                "status": "PASS" if ok_tfnbs else "FAIL",
                "wall_time_s": round(time.perf_counter() - t0, 3),
                "detail": f"backend={actual}, n_processes=2, threadpoolctl={tp_avail}",
            }
        except Exception as exc:
            checks["tfnbs_mp"] = {
                "status": "FAIL",
                "wall_time_s": round(time.perf_counter() - t0, 3),
                "detail": f"{type(exc).__name__}: {exc}",
            }
    else:
        checks["tfnbs_mp"] = {
            "status": "FAIL",
            "wall_time_s": 0.0,
            "detail": "skipped because synthetic_data failed",
        }

    # 3. Bundled BNA-246 + SUIT cerebellum reference
    t0 = time.perf_counter()
    try:
        from conninfpy.atlas.brainnetome import BrainnetomeReference

        ref = BrainnetomeReference.load(profiles=False, resolution_mm=1, cerebellum="anatom")
        n_reg = len(ref.regions)
        cb_n = int(ref.provenance.get("cerebellum_n_regions", 0))
        ok_atlas = n_reg >= 280 and cb_n == 34 and ref.label_image is not None
        checks["bundled_atlas"] = {
            "status": "PASS" if ok_atlas else "FAIL",
            "wall_time_s": round(time.perf_counter() - t0, 3),
            "detail": f"regions={n_reg}, cerebellum_regions={cb_n}",
        }
    except Exception as exc:
        checks["bundled_atlas"] = {
            "status": "FAIL",
            "wall_time_s": round(time.perf_counter() - t0, 3),
            "detail": f"{type(exc).__name__}: {exc}",
        }

    if reference:
        # 4a. Reference NBS check (NBS_schizophrenia 40-edge component, or synthetic fallback skip)
        t0 = time.perf_counter()
        manifest_path = _find_nbs_schizophrenia_demo()
        if manifest_path is not None:
            try:
                from conninfpy import compute_p_val_glm
                from conninfpy.loaders.manifest import ManifestLoader

                demo_ds = ManifestLoader(str(manifest_path)).load()
                Y_demo = np.array(demo_ds.data, dtype=np.float64)
                diag_idx = np.arange(Y_demo.shape[1])
                Y_demo[:, diag_idx, diag_idx] = 0.0
                hc = (demo_ds.pheno["group"] == "HC").to_numpy(dtype=np.float64)
                nbs_res = compute_p_val_glm(
                    Y_demo,
                    interest=hc,
                    method="nbs",
                    threshold=3.1,
                    nbs_stat="extent",
                    n_permutations=500,
                    acceleration=None,
                    rng=1,
                    use_mp=False,
                    two_tailed=False,
                )
                iu_demo = np.triu_indices(Y_demo.shape[1], 1)
                sig_edges = {
                    (int(i), int(j))
                    for i, j in zip(*iu_demo)
                    if nbs_res.positive[i, j] < 0.05
                }
                n_sig_edges = len(sig_edges)
                comp_tbl = nbs_res.component_table()
                ref_json = Path(__file__).resolve().parents[1] / "tests" / "data" / "nbs_schizophrenia_matlab_reference.json"
                exact_edge_set_ok: Optional[bool] = None
                if ref_json.is_file():
                    ref_data = json.loads(ref_json.read_text(encoding="utf-8"))
                    expected_edges = {
                        (int(i), int(j))
                        for component in ref_data["extent"]["edges"]
                        for i, j in np.asarray(component).reshape(-1, 2)
                    }
                    exact_edge_set_ok = sig_edges == expected_edges
                ok_nbs = (
                    n_sig_edges == 40
                    and len(comp_tbl) == 1
                    and int(comp_tbl.iloc[0]["n_edges"]) == 40
                    and (exact_edge_set_ok is None or exact_edge_set_ok)
                )
                edge_set_note = (
                    f", exact_edge_set_match={exact_edge_set_ok}"
                    if exact_edge_set_ok is not None
                    else ""
                )
                checks["reference_nbs_schizophrenia"] = {
                    "status": "PASS" if ok_nbs else "FAIL",
                    "wall_time_s": round(time.perf_counter() - t0, 3),
                    "detail": f"n_sig_edges={n_sig_edges} (expected 40), components={len(comp_tbl)}{edge_set_note}",
                }
            except Exception as exc:
                checks["reference_nbs_schizophrenia"] = {
                    "status": "FAIL",
                    "wall_time_s": round(time.perf_counter() - t0, 3),
                    "detail": f"{type(exc).__name__}: {exc}",
                }
        else:
            try:
                ok_fb, fb_detail = _run_synthetic_nbs_fallback()
                checks["reference_nbs_schizophrenia"] = {
                    "status": "SKIP" if ok_fb else "FAIL",
                    "wall_time_s": round(time.perf_counter() - t0, 3),
                    "detail": f"datasets/demo/NBS_schizophrenia not found; {fb_detail}",
                }
            except Exception as exc:
                checks["reference_nbs_schizophrenia"] = {
                    "status": "FAIL",
                    "wall_time_s": round(time.perf_counter() - t0, 3),
                    "detail": f"datasets/demo/NBS_schizophrenia not found; fallback failed: {type(exc).__name__}: {exc}",
                }

        # 4b. Deterministic fixed-seed TFNBS reproducibility check
        t0 = time.perf_counter()
        try:
            from conninfpy.analysis import analyze
            from conninfpy.sim import TopologyDatasetGenerator

            ref_gen = TopologyDatasetGenerator(n_nodes=20, n_modules=2, seed=42)
            ref_ds = ref_gen.generate("chain", effect_size=0.4, n_samples=12, time_points=80)
            r1 = analyze(
                group1=ref_ds.group1,
                group2=ref_ds.group2,
                test_type="two-sample",
                input="r",
                method="tfnbs",
                n_permutations=100,
                use_mp=False,
                rng=42,
            )
            r2 = analyze(
                group1=ref_ds.group1,
                group2=ref_ds.group2,
                test_type="two-sample",
                input="r",
                method="tfnbs",
                n_permutations=100,
                use_mp=False,
                rng=42,
            )
            iu_ref = np.triu_indices(20, 1)
            p1_pos = np.asarray(r1.inference["positive"], dtype=float)
            p2_pos = np.asarray(r2.inference["positive"], dtype=float)
            p1_neg = np.asarray(r1.inference["negative"], dtype=float)
            p2_neg = np.asarray(r2.inference["negative"], dtype=float)
            n_sig_pos_1 = int(np.sum(p1_pos[iu_ref] <= 0.05))
            n_sig_pos_2 = int(np.sum(p2_pos[iu_ref] <= 0.05))
            n_sig_neg_1 = int(np.sum(p1_neg[iu_ref] <= 0.05))
            n_sig_neg_2 = int(np.sum(p2_neg[iu_ref] <= 0.05))
            reproducible = (
                n_sig_pos_1 > 0
                and n_sig_pos_1 == n_sig_pos_2
                and n_sig_neg_1 == n_sig_neg_2
                and bool(np.array_equal(p1_pos, p2_pos))
                and bool(np.array_equal(p1_neg, p2_neg))
            )
            checks["reference_tfnbs_reproducibility"] = {
                "status": "PASS" if reproducible else "FAIL",
                "wall_time_s": round(time.perf_counter() - t0, 3),
                "detail": (
                    f"rng=42, n_sig_pos={n_sig_pos_1}/{n_sig_pos_2}, "
                    f"n_sig_neg={n_sig_neg_1}/{n_sig_neg_2}, "
                    f"exact_match={bool(np.array_equal(p1_pos, p2_pos) and np.array_equal(p1_neg, p2_neg))}"
                ),
            }
        except Exception as exc:
            checks["reference_tfnbs_reproducibility"] = {
                "status": "FAIL",
                "wall_time_s": round(time.perf_counter() - t0, 3),
                "detail": f"{type(exc).__name__}: {exc}",
            }

    all_pass = all(c["status"] in ("PASS", "SKIP") for c in checks.values())
    has_skip = any(c["status"] == "SKIP" for c in checks.values())
    overall_status = "FAIL" if not all_pass else ("SKIPPED" if has_skip else "PASS")
    overall_text = "FAIL" if not all_pass else ("PASS (reference skipped)" if has_skip else "PASS")
    total_wall = round(time.perf_counter() - t_start, 3)
    report: Dict[str, Any] = {
        "ok": all_pass,
        "status": overall_status,
        "skipped": [k for k, c in checks.items() if c["status"] == "SKIP"],
        "wall_time_s": total_wall,
        "backend": backend_info,
        "checks": checks,
    }
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"conninfpy selftest — {overall_text} ({total_wall:.2f} s, backend={backend_info['actual_backend']})")
        for name, item in checks.items():
            print(f"  [{item['status']}] {name} ({item['wall_time_s']:.2f} s): {item['detail']}")
    return (0 if all_pass else 1), report


def _parse_edge_args(
    edges_spec: Optional[str],
    edges_file: "str | os.PathLike[str] | None",
) -> List[Tuple[int, int]]:
    """Parse explicit 0-based ROI index pairs from --edges and/or --edges-file."""
    if not edges_spec and not edges_file:
        raise ValueError(
            "Explicit edges are required: supply --edges 'i:j,...' or --edges-file <csv>."
        )
    pairs: List[Tuple[int, int]] = []
    if edges_spec:
        for raw in edges_spec.split(","):
            token = raw.strip()
            if not token:
                continue
            sep = ":" if ":" in token else ("-" if "-" in token else None)
            if sep is None:
                raise ValueError(f"Invalid edge token {token!r}; expected 'i:j' or 'i-j'.")
            parts = token.split(sep)
            if len(parts) != 2:
                raise ValueError(f"Invalid edge token {token!r}; expected two integers.")
            pairs.append((int(parts[0].strip()), int(parts[1].strip())))
    if edges_file:
        df = pd.read_csv(edges_file)
        if {"roi_i", "roi_j"}.issubset(df.columns):
            col_i, col_j = "roi_i", "roi_j"
        elif {"i", "j"}.issubset(df.columns):
            col_i, col_j = "i", "j"
        else:
            raise ValueError(
                f"{edges_file}: expected columns ('roi_i', 'roi_j') or ('i', 'j'); "
                f"got {list(df.columns)}."
            )
        for row_i, row_j in zip(df[col_i], df[col_j]):
            pairs.append((int(row_i), int(row_j)))
    if not pairs:
        raise ValueError("No edge pairs found in --edges / --edges-file.")
    return pairs


def _build_influence_inputs(
    cfg: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    design_kind: str,
    resolved_strategy: Optional[str],
) -> Dict[str, Any]:
    """Reconstruct the tested inputs expected by ``subject_influence`` from ``build_analysis``."""
    from conninfpy._prepare import prepare_connectivity
    from conninfpy._shared.alignment import paired_id_permutation
    from conninfpy.stats._design import build_design_matrix

    inf_cfg = cfg.get("inference") or {}
    prep_kind = str(kwargs.get("input") or "r")
    if inf_cfg.get("analysis_scale") == "correlation":
        prep_kind = "z"
    allow_missing = inf_cfg.get("missing", "raise") == "available_case"

    if design_kind == "one-sample":
        g1 = np.asarray(prepare_connectivity(kwargs["group1"], prep_kind, "group1", allow_missing=allow_missing), dtype=np.float64)
        return {"group1": g1}
    if design_kind == "two-sample":
        var_equal = bool(inf_cfg.get("var_equal", False))
        g1 = np.asarray(prepare_connectivity(kwargs["group1"], prep_kind, "group1", allow_missing=allow_missing), dtype=np.float64)
        g2 = np.asarray(prepare_connectivity(kwargs["group2"], prep_kind, "group2", allow_missing=allow_missing), dtype=np.float64)
        return {
            "group1": g1,
            "group2": g2,
            "var_equal": var_equal,
        }
    if design_kind == "paired":
        g1 = np.asarray(prepare_connectivity(kwargs["group1"], prep_kind, "group1", allow_missing=allow_missing), dtype=np.float64)
        g2 = np.asarray(prepare_connectivity(kwargs["group2"], prep_kind, "group2", allow_missing=allow_missing), dtype=np.float64)
        out: Dict[str, Any] = {
            "group1": g1,
            "group2": g2,
        }
        if kwargs.get("ids_group1") is not None and kwargs.get("ids_group2") is not None:
            out["ids_group1"] = kwargs["ids_group1"]
            out["ids_group2"] = kwargs["ids_group2"]
        return out

    # design_kind == "glm"
    if cfg["design"]["type"] == "paired":
        g1 = np.asarray(prepare_connectivity(kwargs["group1"], prep_kind, "group1", allow_missing=allow_missing), dtype=np.float64)
        g2 = np.asarray(prepare_connectivity(kwargs["group2"], prep_kind, "group2", allow_missing=allow_missing), dtype=np.float64)
        if kwargs.get("ids_group1") is not None and kwargs.get("ids_group2") is not None:
            perm = paired_id_permutation(kwargs["ids_group1"], kwargs["ids_group2"])
            g2 = g2[perm]
        else:
            perm = None
        Y_diff = g2 - g1
        if "confounds_group1" in kwargs and "confounds_group2" in kwargs:
            ca = np.asarray(kwargs["confounds_group1"], dtype=np.float64)
            cb = np.asarray(kwargs["confounds_group2"], dtype=np.float64)
            if perm is not None:
                cb = cb[perm]
            if ca.ndim == 1:
                ca = ca[:, None]
                cb = cb[:, None]
            X = np.column_stack([np.ones((g1.shape[0], 1), dtype=np.float64), cb - ca])
            return {"Y": Y_diff, "design_matrix": X}
        X, _ = build_design_matrix(kwargs["interest"], kwargs.get("confounds"))
        return {"Y": Y_diff, "design_matrix": X}

    Y = np.asarray(prepare_connectivity(kwargs["Y"], prep_kind, "Y", allow_missing=allow_missing), dtype=np.float64)
    confounds = kwargs.get("confounds")
    sites = kwargs.get("sites")
    if resolved_strategy in ("combat_only", "combat_site_dummies_glm") and sites is not None:
        from conninfpy.harmonize import combat_harmonize

        preserve = None
        if confounds is not None:
            c_arr = np.asarray(confounds, dtype=np.float64)
            preserve = c_arr[:, None] if c_arr.ndim == 1 else c_arr
        Y = combat_harmonize(Y, sites=np.asarray(sites).tolist(), preserve=preserve).Y_adjusted
    if resolved_strategy in ("combat_site_dummies_glm", "site_dummies_glm") and sites is not None:
        from conninfpy.analysis import _site_dummies

        site_dum = _site_dummies(np.asarray(sites))
        if site_dum.shape[1] > 0:
            if confounds is None:
                confounds = site_dum
            else:
                c_arr = np.asarray(confounds, dtype=np.float64)
                if c_arr.ndim == 1:
                    c_arr = c_arr[:, None]
                confounds = np.hstack([c_arr, site_dum])
    X, _ = build_design_matrix(kwargs["interest"], confounds)
    return {"Y": Y, "design_matrix": X}


def run_influence(
    target: "str | os.PathLike[str]",
    *,
    config_path: "str | os.PathLike[str]",
    edges_spec: Optional[str] = None,
    edges_file: "str | os.PathLike[str] | None" = None,
    out_dir: "str | os.PathLike[str] | None" = None,
    as_json: bool = False,
) -> Dict[str, Any]:
    """Compute descriptive leave-one-out subject influence on explicitly chosen edges."""
    from conninfpy.cli import build_analysis, load_config
    from conninfpy.influence import subject_influence
    from conninfpy.loaders import load
    from conninfpy.post_analysis import ResultBundle

    pairs = _parse_edge_args(edges_spec, edges_file)
    target_path = Path(target).resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Target does not exist: {target_path}")
    if target_path.is_file():
        bundle_paths = [target_path]
        base_dir = target_path.parent
    else:
        bundle_paths = sorted(target_path.glob("bundle*.pkl"))
        base_dir = target_path
        if not bundle_paths:
            raise FileNotFoundError(f"No bundle*.pkl files found in {target_path}")

    cfg_path = Path(config_path).resolve()
    cfg = load_config(cfg_path)
    dataset = load(cfg_path.parent / cfg["data"])
    kwargs, _ = build_analysis(dataset, cfg)

    prov_path = base_dir / "provenance.json"
    prov_results: Dict[str, Any] = {}
    if prov_path.is_file():
        try:
            prov_results = json.loads(prov_path.read_text(encoding="utf-8")).get("results") or {}
        except Exception:
            prov_results = {}

    dest = Path(out_dir).resolve() if out_dir else base_dir / "influence"
    dest.mkdir(parents=True, exist_ok=True)

    reports: Dict[str, Any] = {}
    for bp in bundle_paths:
        suffix = bp.stem[len("bundle"):]
        res_key = suffix.lstrip("_") or "result"
        res_prov = prov_results.get(res_key) or {}
        resolved_strategy = res_prov.get("resolved_strategy")

        bundle = ResultBundle.load(bp)
        pvals_path = base_dir / f"pvals{suffix}.npz"
        inf_res = _reconstruct_inference_result(bundle, pvals_path if pvals_path.is_file() else None)
        if resolved_strategy in ("combat_only", "combat_site_dummies_glm"):
            inf_res.harmonized = True

        design_kind = str((inf_res.design or {}).get("kind") or cfg["design"]["type"])
        infl_kwargs = _build_influence_inputs(cfg, kwargs, design_kind, resolved_strategy)
        report = subject_influence(inf_res, edges=pairs, **infl_kwargs)

        json_path = dest / f"subject_influence{suffix}.json"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        csv_path = dest / f"subject_influence{suffix}.csv"
        rows = report.get("per_subject_edge") or []
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        reports[res_key] = report

    payload = next(iter(reports.values())) if len(reports) == 1 else reports
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        for name, rep in reports.items():
            unavail = rep.get("unavailable") or []
            if unavail:
                reason = "; ".join(str(u.get("reason") or u) for u in unavail)
                print(f"conninfpy influence [{name}] — status=unavailable ({reason})")
            else:
                print(
                    f"conninfpy influence [{name}] — status=ok "
                    f"(n_edges={rep.get('n_edges')}, n_subjects={rep.get('n_subjects')})"
                )
        print(f"  wrote influence outputs to {dest}")
    return payload
