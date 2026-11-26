"""Helper implementations for ``conninfpy`` CLI subcommands: qc, init, interpret, influence, selftest, app."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
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


def _design_matrix_for_qc(
    dataset: Any, cfg: Mapping[str, Any], kwargs: Mapping[str, Any]
) -> tuple[Optional[np.ndarray], List[str], Dict[str, Any]]:
    """Construct the design matrix and extra balance diagnostics for ``qc``."""
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
        if site_col:
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
        _atlas,
        _jsonable,
        _motion_qc_spec,
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
    atlas = _atlas(cfg.get("atlas"), dataset, base, summary["n_regions"])

    warnings_list: List[str] = list(val_report.warnings) + list(summary.get("loader_warnings") or [])
    errors_list: List[str] = list(val_report.errors)

    n_dropped = int(summary["n_subjects_in_data"] - summary["n_subjects_used"])
    if n_dropped > 0:
        warnings_list.append(
            f"{n_dropped} of {summary['n_subjects_in_data']} observation(s) dropped due to missing design values or level filtering."
        )

    pheno = dataset.pheno
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
        "n_regions": int(summary["n_regions"]),
        "atlas": getattr(atlas, "name", None) if atlas is not None else None,
        "design": summary["design"],
        "balance": balance,
        "design_diagnostics": design_diag,
        "n_valid_tr": tr_summary,
        "motion_qc": motion_report,
        "warnings": warnings_list,
        "errors": errors_list,
    }

    if out_path is not None:
        Path(out_path).write_text(json.dumps(report, indent=2, default=_jsonable), encoding="utf-8")

    if as_json:
        print(json.dumps(report, indent=2, default=_jsonable))
    else:
        status_txt = "PASS" if ok else "FAIL"
        print(
            f"conninfpy qc [{status_txt}] — {summary['design']} | "
            f"{summary['n_subjects_used']}/{summary['n_subjects_in_data']} subjects × {summary['n_regions']} regions"
        )
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
        "  method: tfnbs          # tstat, nbs, tfnbs, cnbs, ni_tfnbs, fbc_tfnbs, bh_fdr_perm",
        "  n_permutations: 1000",
        "  two_tailed: true",
        "  # acceleration: null   # empirical p-values by default",
        "",
        "alpha: 0.05",
        "output: conninfpy_results",
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


def _resolve_atlas_for_interpret(
    atlas_spec: Optional[str],
    provenance_path: Optional[Path],
    n_regions: int,
) -> Any:
    from conninfpy.atlas.metadata import AtlasInfo

    if atlas_spec is not None:
        if atlas_spec in ("schaefer_100_yeo7", "schaefer_200_yeo7", "schaefer_400_yeo7", "bna_246"):
            return getattr(AtlasInfo, atlas_spec)()
        return AtlasInfo.from_csv(atlas_spec)

    if provenance_path is not None and provenance_path.is_file():
        try:
            prov = json.loads(provenance_path.read_text(encoding="utf-8"))
            cfg_atlas = (prov.get("config") or {}).get("atlas")
            if isinstance(cfg_atlas, str):
                if cfg_atlas in ("schaefer_100_yeo7", "schaefer_200_yeo7", "schaefer_400_yeo7", "bna_246"):
                    return getattr(AtlasInfo, cfg_atlas)()
                candidates = []
                if prov.get("analysis_file"):
                    candidates.append(Path(prov["analysis_file"]).parent / cfg_atlas)
                candidates.append(Path(provenance_path).parent / cfg_atlas)
                for cand in candidates:
                    if cand.is_file():
                        return AtlasInfo.from_csv(str(cand))
        except Exception:
            pass

    by_size = {
        100: "schaefer_100_yeo7",
        200: "schaefer_200_yeo7",
        400: "schaefer_400_yeo7",
        246: "bna_246",
    }
    if n_regions in by_size:
        return getattr(AtlasInfo, by_size[n_regions])()
    return None


def _reconstruct_inference_result(bundle: Any, pvals_path: Optional[Path]) -> Any:
    """Build an ``InferenceResult`` from ``pvals.npz`` + ``ResultBundle`` for summaries/plots."""
    from conninfpy.results import InferenceResult

    if pvals_path is not None and pvals_path.is_file():
        with np.load(pvals_path, allow_pickle=False) as z:
            pos = np.array(z["positive"]) if "positive" in z else np.array(bundle.corrected_p["positive"])
            neg = np.array(z["negative"]) if "negative" in z else np.array(bundle.corrected_p["negative"])
            stat_signed = np.array(z["stat_signed"]) if "stat_signed" in z else None
            beta = np.array(z["beta"]) if "beta" in z else None
    else:
        pos = np.array(bundle.corrected_p["positive"])
        neg = np.array(bundle.corrected_p["negative"])
        stat_signed = None
        beta = None

    stat_pos = np.maximum(stat_signed, 0.0) if stat_signed is not None else None
    stat_neg = np.maximum(-stat_signed, 0.0) if stat_signed is not None else None
    meta = bundle.metadata
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
        analysis_provenance={
            "estimable": bundle.estimable,
            "tested": bundle.tested,
        },
        component_report=native if unit == "component" else None,
        block_report=native if unit == "block" else None,
    )


def interpret_outputs(
    target: "str | os.PathLike[str]",
    *,
    atlas_spec: Optional[str] = None,
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
    else:
        bundle_paths = sorted(target_path.glob("bundle*.pkl"))
        base_dir = target_path
        if not bundle_paths:
            raise FileNotFoundError(f"No bundle*.pkl files found in {target_path}")

    prov_path = base_dir / "provenance.json"
    eff_alpha = alpha
    if eff_alpha is None and prov_path.is_file():
        try:
            eff_alpha = float(json.loads(prov_path.read_text(encoding="utf-8")).get("alpha", 0.05))
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
        atlas = _resolve_atlas_for_interpret(atlas_spec, prov_path if prov_path.is_file() else None, n_regions)
        pvals_path = base_dir / f"pvals{suffix}.npz"
        inf_res = _reconstruct_inference_result(bundle, pvals_path if pvals_path.is_file() else None)
        is_grid = getattr(inf_res, "is_grid", False)
        param_idx = 0 if is_grid else None

        # 1. Bundle summaries for available groupings & tails
        contrast_id = bundle.metadata.get("contrast_id")
        measure_name = (bundle.effect_spec or {}).get("name")
        for tail in sorted(bundle.corrected_p.keys()):
            pairs_df = bundle.detected_pairs(tail=tail, param_idx=param_idx, alpha=eff_alpha)
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
                        param_idx=param_idx,
                        alpha=eff_alpha,
                    )
                    s_path = dest / f"bundle_summary{suffix}_{grp_name}_{tail}.csv"
                    ResultBundle.export_summary(s_df, s_path)
                    written[f"bundle_summary{suffix}_{grp_name}_{tail}"] = s_path

        # 2. Node / Network / Component summaries
        try:
            node_df = inf_res.node_summary(atlas=atlas, alpha=eff_alpha, param_idx=param_idx)
            node_path = dest / f"node_summary{suffix}.csv"
            node_df.to_csv(node_path, index=False)
            written[f"node_summary{suffix}"] = node_path
        except ValueError:
            pass

        if atlas is not None and getattr(atlas, "networks", None) is not None:
            net_df = inf_res.network_summary(atlas, alpha=eff_alpha, param_idx=param_idx)
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
            ann_edges = inf_res.annotated_edges(rois, alpha=eff_alpha, param_idx=param_idx)
            ann_path = dest / f"annotated_edges{suffix}.csv"
            ann_edges.to_csv(ann_path, index=False)
            written[f"annotated_edges{suffix}"] = ann_path

            ann_sum = summarize_annotation(inf_res, rois, alpha=eff_alpha, param_idx=param_idx)
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

            fig = summary_figure(inf_res, atlas=atlas, alpha=eff_alpha)
            fig_path = dest / f"summary_figure{suffix}.{fig_format}"
            fig.savefig(fig_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            written[f"summary_figure{suffix}"] = fig_path

            design_dict = getattr(inf_res, "design", None) or {}
            if inf_res.beta is not None and design_dict.get("comparison"):
                fig_c, ax_c = plt.subplots(figsize=(6.5, 5.5))
                plot_coupling_classes(inf_res, atlas=atlas, alpha=eff_alpha, ax=ax_c)
                c_path = dest / f"coupling_classes{suffix}.{fig_format}"
                fig_c.savefig(c_path, dpi=dpi, bbox_inches="tight", facecolor="white")
                plt.close(fig_c)
                written[f"coupling_classes{suffix}"] = c_path

            if rois is not None:
                nsig = inf_res.n_significant(eff_alpha, param_idx=param_idx)
                for tail in ("positive", "negative"):
                    if nsig.get(tail, 0) > 0:
                        fig_g, ax_g = plt.subplots(figsize=(7.5, 7.5))
                        plot_connectogram(inf_res, rois, tail=tail, alpha=eff_alpha, param_idx=param_idx, ax=ax_g)
                        g_path = dest / f"connectogram{suffix}_{tail}.{fig_format}"
                        fig_g.savefig(g_path, dpi=dpi, bbox_inches="tight", facecolor="white")
                        plt.close(fig_g)
                        written[f"connectogram{suffix}_{tail}"] = g_path

    print(f"conninfpy interpret — wrote {len(written)} artifact(s) to {dest}")
    return written


def run_selftest(*, as_json: bool = False) -> tuple[int, Dict[str, Any]]:
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

    all_pass = all(c["status"] == "PASS" for c in checks.values())
    total_wall = round(time.perf_counter() - t_start, 3)
    report: Dict[str, Any] = {
        "ok": all_pass,
        "wall_time_s": total_wall,
        "backend": backend_info,
        "checks": checks,
    }
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"conninfpy selftest — {'PASS' if all_pass else 'FAIL'} ({total_wall:.2f} s, backend={backend_info['actual_backend']})")
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
