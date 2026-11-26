"""Design diagnostics and network-level detection counts."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import numpy.typing as npt


VIF_CONVENTION_NOTE = (
    "Constant/intercept columns are assigned VIF = 1.0 by convention; "
    "non-constant columns use 1 / (1 - R^2_j) from regressing column j "
    "on all remaining columns plus an intercept."
)

CONDITION_NUMBER_NOTE = (
    "condition_number (cond(X'X)) and condition_number_X (cond(X)) are "
    "computed on the raw design matrix and depend on predictor units and "
    "centering; uncentered or large-scale covariates can inflate raw "
    "cond(X'X) without true collinearity or confounding. Check "
    "condition_number_standardized, VIF, pairwise correlations, and "
    "matrix_rank alongside raw cond(X'X)."
)


def _json_finite(val: float) -> Optional[float]:
    """Return finite Python float or None for non-finite values."""
    v = float(val)
    return v if np.isfinite(v) else None


def _standardized_condition_number(X: npt.NDArray[np.float64]) -> float:
    """Condition number of X_std'X_std after z-scoring non-constant columns."""
    n, p = X.shape
    if n == 0 or p == 0:
        return 1.0
    X_std = np.zeros_like(X, dtype=np.float64)
    for j in range(p):
        col = X[:, j]
        sd = float(np.std(col))
        if sd > 0.0:
            X_std[:, j] = (col - float(np.mean(col))) / sd
        else:
            # Keep constant columns scaled to unit norm / 1.0 so intercept stays represented.
            X_std[:, j] = np.where(col == 0.0, 0.0, 1.0)
    try:
        return float(np.linalg.cond(X_std.T @ X_std))
    except np.linalg.LinAlgError:
        return float("inf")


def compute_vif(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Variance inflation factor per design-matrix column.

    ``VIF_j = 1 / (1 - R²_j)`` where ``R²_j`` is from regressing column j on
    all other columns. ``VIF > 5`` is a common concern threshold; ``> 10``
    is strong collinearity (Kutner et al. 2004).

    Constant columns (e.g. intercept) get ``VIF = 1`` by convention.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Design matrix.

    Returns
    -------
    ndarray of shape (p,)
        VIF per column.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    vif = np.ones(p, dtype=np.float64)
    for j in range(p):
        x_j = X[:, j]
        if np.var(x_j) == 0:
            # Constant column (e.g. intercept) → R² undefined; report 1
            vif[j] = 1.0
            continue
        others = np.delete(X, j, axis=1)
        if others.shape[1] == 0:
            vif[j] = 1.0
            continue
        # Add intercept to the "others" block to get the right R²
        others_full = np.hstack([np.ones((n, 1)), others])
        beta, *_ = np.linalg.lstsq(others_full, x_j, rcond=None)
        pred = others_full @ beta
        ss_res = float(np.sum((x_j - pred) ** 2))
        ss_tot = float(np.sum((x_j - x_j.mean()) ** 2))
        r_sq = 1.0 - ss_res / max(ss_tot, 1e-12)
        r_sq = min(max(r_sq, 0.0), 1.0 - 1e-12)
        vif[j] = 1.0 / (1.0 - r_sq)
    return vif


def design_diagnostics(
    X: npt.NDArray[np.float64],
    names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Diagnostic report for a GLM design matrix.

    Returns a dict with:
      - ``condition_number``: condition number of ``X'X`` (> 30 is caution,
        > 100 is strong concern; raw scale/centering dependent)
      - ``condition_number_X``: condition number of ``X``
      - ``condition_number_standardized``: condition number after z-scoring
        non-constant columns
      - ``matrix_rank``, ``rank_deficient``, ``nullity``, ``residual_df``
      - ``vif``: VIF per column (constant columns = 1.0 by convention)
      - ``vif_max``, ``vif_max_col``: most collinear column
      - ``correlation``: pairwise Pearson correlation matrix (p, p)
      - ``flags``: list of plain-English warnings when thresholds are crossed

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Design matrix (should include intercept if used in the GLM).
    names : sequence of str, optional
        Column names for reporting. Defaults to ``['x0', 'x1', ...]``.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    if names is None:
        names = [f"x{i}" for i in range(p)]
    elif len(names) != p:
        raise ValueError(f"names length ({len(names)}) must match ncols ({p}).")

    rank = int(np.linalg.matrix_rank(X))
    nullity = int(p - rank)
    rank_deficient = bool(rank < p)
    residual_df = int(max(0, n - rank))

    XtX = X.T @ X
    try:
        cond = float(np.linalg.cond(XtX))
    except np.linalg.LinAlgError:
        cond = np.inf
    try:
        cond_x = float(np.linalg.cond(X))
    except np.linalg.LinAlgError:
        cond_x = np.inf
    cond_std = _standardized_condition_number(X)

    vif = compute_vif(X)
    vif_argmax = int(np.argmax(vif))
    constant_cols = [names[j] for j in range(p) if np.var(X[:, j]) == 0]

    # Pairwise correlation (intercept / constant → NaN row/col; report as 0)
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(X, rowvar=False)
    corr = np.atleast_2d(np.nan_to_num(corr, nan=0.0))

    flags: List[str] = []
    if rank_deficient:
        flags.append(
            f"Rank-deficient design matrix: matrix rank {rank} < {p} columns "
            f"(nullity = {nullity}); some linear combinations are unidentifiable."
        )
    if cond > 100:
        flags.append(
            f"Raw cond(X'X) = {cond:.1f} > 100 (standardized cond = {cond_std:.1f}): "
            "reflects predictor scale/centering and/or collinearity."
        )
    elif cond > 30:
        flags.append(
            f"Raw cond(X'X) = {cond:.1f} > 30 (standardized cond = {cond_std:.1f}): "
            "moderate raw conditioning sensitivity."
        )
    if np.max(vif) > 10:
        flags.append(
            f"VIF for '{names[vif_argmax]}' = {vif[vif_argmax]:.1f} > 10: "
            "strong collinearity."
        )
    elif np.max(vif) > 5:
        flags.append(
            f"VIF for '{names[vif_argmax]}' = {vif[vif_argmax]:.1f} > 5: "
            "moderate collinearity."
        )

    return {
        "n_observations": int(n),
        "n_columns": int(p),
        "matrix_rank": rank,
        "rank_deficient": rank_deficient,
        "nullity": nullity,
        "residual_df": residual_df,
        "condition_number": cond,
        "condition_number_X": cond_x,
        "condition_number_standardized": cond_std,
        "condition_number_note": CONDITION_NUMBER_NOTE,
        "vif": vif,
        "vif_max": float(vif[vif_argmax]),
        "vif_max_col": names[vif_argmax],
        "vif_convention": VIF_CONVENTION_NOTE,
        "constant_columns": constant_cols,
        "correlation": corr,
        "names": list(names),
        "flags": flags,
    }


def _classify_factor_values(
    values: npt.NDArray[np.float64],
    role: str,
) -> str:
    """Classify a numeric design column as intercept, binary, categorical_coded, or continuous."""
    if role == "intercept":
        return "intercept"
    if role in ("site_dummy", "group_indicator"):
        return "binary" if len(np.unique(values[np.isfinite(values)])) <= 2 else "categorical_coded"
    finite = values[np.isfinite(values)]
    uniq = np.unique(finite)
    if uniq.size <= 2:
        return "binary"
    if (
        uniq.size <= 6
        and finite.size >= 2 * uniq.size
        and np.allclose(uniq, np.round(uniq))
    ):
        return "categorical_coded"
    return "continuous"


def _format_level_key(val: float) -> str:
    if np.isfinite(val) and float(val).is_integer():
        return str(int(round(float(val))))
    return f"{float(val):g}"


def _summarize_available_case(missing_report: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract JSON-safe available-case missingness diagnostics from missing_report."""
    if not missing_report:
        return None
    edge_n = missing_report.get("edge_n")
    edge_n_max = int(np.max(edge_n)) if edge_n is not None and np.size(edge_n) else None
    summary: Dict[str, Any] = {
        "policy": missing_report.get("policy", "available_case"),
        "min_coverage": _json_finite(missing_report["min_coverage"])
        if missing_report.get("min_coverage") is not None
        else None,
        "n_edges_tested": int(missing_report.get("n_edges_tested", 0)),
        "n_edges_untested": int(missing_report.get("n_edges_untested", 0)),
        "edge_n_min": int(missing_report.get("edge_n_min", 0)),
        "edge_n_median": _json_finite(missing_report.get("edge_n_median", 0.0)),
        "edge_n_max": edge_n_max,
        "missing_by_group_p": _json_finite(missing_report["missing_by_group_p"])
        if missing_report.get("missing_by_group_p") is not None
        else None,
        "missing_by_interest_p": _json_finite(missing_report["missing_by_interest_p"])
        if missing_report.get("missing_by_interest_p") is not None
        else None,
    }
    if missing_report.get("untested_reasons") is not None:
        summary["untested_reasons"] = {
            str(k): int(v) for k, v in missing_report["untested_reasons"].items()
        }
    df_mat = missing_report.get("edge_residual_df")
    if df_mat is not None:
        df_arr = np.asarray(df_mat, dtype=np.float64)
        finite_df = df_arr[np.isfinite(df_arr)]
        if finite_df.size:
            summary["edge_residual_df_min"] = int(np.min(finite_df))
            summary["edge_residual_df_max"] = int(np.max(finite_df))
    return summary


def analyze_design_diagnostics(
    *,
    design_kind: str,
    unit_of_observation: str,
    X: npt.NDArray[np.float64],
    names: Sequence[str],
    column_roles: Optional[Mapping[str, str]] = None,
    contrast: Optional[Any] = None,
    focus_contrast: Optional[str] = None,
    sites: Optional[Sequence[Any]] = None,
    strata: Optional[Sequence[Any]] = None,
    group_labels: Optional[Sequence[Any]] = None,
    interest_name: Optional[str] = None,
    interest_values: Optional[npt.NDArray[np.float64]] = None,
    missing_report: Optional[Mapping[str, Any]] = None,
    var_equal: Optional[bool] = None,
    design_note: Optional[str] = None,
    strata_note: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a descriptive, JSON-serializable design diagnostic dictionary
    for :class:`~conninfpy.analysis.AnalyzeResult`.

    Inspects the actual design matrix ``X``, column roles, continuous and
    categorical predictors/confounds, site cross-tabulations or site-by-predictor
    spread, rank deficiency / collinearity, and ``available_case`` scope.
    """
    from conninfpy.stats._design import contrast_is_estimable

    X_arr = np.asarray(X, dtype=np.float64)
    col_names = list(names)
    roles = dict(column_roles or {})
    base = design_diagnostics(X_arr, names=col_names)

    n_obs = base["n_observations"]
    p = base["n_columns"]
    rank = base["matrix_rank"]
    rank_deficient = base["rank_deficient"]
    nullity = base["nullity"]
    residual_df = base["residual_df"]

    vif_arr = np.asarray(base["vif"], dtype=np.float64)
    vif_map = {nm: _json_finite(vif_arr[i]) for i, nm in enumerate(col_names)}

    corr_arr = np.asarray(base["correlation"], dtype=np.float64)
    high_corr_pairs: List[Dict[str, Any]] = []
    for i in range(p):
        for j in range(i + 1, p):
            if roles.get(col_names[i]) == "intercept" or roles.get(col_names[j]) == "intercept":
                continue
            r_val = float(corr_arr[i, j])
            if np.isfinite(r_val) and abs(r_val) >= 0.7:
                high_corr_pairs.append(
                    {"col_a": col_names[i], "col_b": col_names[j], "r": round(r_val, 4)}
                )

    # Contrast estimability
    estimable_report: Optional[Any] = None
    has_non_estimable = False
    if isinstance(contrast, Mapping):
        estimable_report = {
            str(k): bool(contrast_is_estimable(np.asarray(v, dtype=np.float64), X_arr))
            for k, v in contrast.items()
        }
        if focus_contrast is not None and focus_contrast in estimable_report:
            has_non_estimable = not estimable_report[focus_contrast]
        else:
            has_non_estimable = any(not ok for ok in estimable_report.values())
    elif contrast is not None:
        c_arr = np.asarray(contrast, dtype=np.float64)
        estimable_report = bool(contrast_is_estimable(c_arr, X_arr))
        has_non_estimable = not estimable_report

    # Column-wise continuous & categorical factor summaries
    column_summaries: List[Dict[str, Any]] = []
    categorical_factors: Dict[str, Dict[str, Any]] = {}
    continuous_factors: Dict[str, Dict[str, Any]] = {}
    non_intercept_constant_cols: List[str] = []

    for j, nm in enumerate(col_names):
        col = X_arr[:, j]
        role = roles.get(nm, "intercept" if nm == "intercept" else "predictor")
        kind = _classify_factor_values(col, role)
        zero_var = bool(np.var(col) == 0.0)
        if zero_var and role != "intercept":
            non_intercept_constant_cols.append(nm)

        finite = col[np.isfinite(col)]
        entry: Dict[str, Any] = {
            "name": nm,
            "role": role,
            "factor_type": kind,
            "vif": vif_map[nm],
            "zero_variance": zero_var,
        }
        if kind in ("binary", "categorical_coded"):
            uniq, counts = np.unique(finite, return_counts=True)
            level_counts = {
                _format_level_key(float(u)): int(c) for u, c in zip(uniq, counts, strict=True)
            }
            entry["n_levels"] = int(uniq.size)
            entry["level_counts"] = level_counts
            categorical_factors[nm] = {
                "role": role,
                "factor_type": kind,
                "n_levels": int(uniq.size),
                "level_counts": level_counts,
                "zero_variance": zero_var,
                "vif": vif_map[nm],
            }
        elif kind == "continuous":
            stats_dict = {
                "role": role,
                "n": int(finite.size),
                "mean": _json_finite(np.mean(finite)) if finite.size else None,
                "std": _json_finite(np.std(finite)) if finite.size else None,
                "min": _json_finite(np.min(finite)) if finite.size else None,
                "max": _json_finite(np.max(finite)) if finite.size else None,
                "median": _json_finite(np.median(finite)) if finite.size else None,
                "iqr": _json_finite(
                    np.percentile(finite, 75) - np.percentile(finite, 25)
                )
                if finite.size
                else None,
                "n_unique": int(np.unique(finite).size),
                "zero_variance": zero_var,
                "vif": vif_map[nm],
            }
            entry.update(stats_dict)
            continuous_factors[nm] = stats_dict
        column_summaries.append(entry)

    # Sample sizes
    sample_sizes: Dict[str, Any] = {"total": int(n_obs)}
    if group_labels is not None:
        g_arr = [str(g) for g in group_labels]
        g_counts: Dict[str, int] = {}
        for g in g_arr:
            g_counts[g] = g_counts.get(g, 0) + 1
        sample_sizes["groups"] = g_counts
        if "group" not in categorical_factors:
            categorical_factors["group"] = {
                "role": "group",
                "factor_type": "binary" if len(g_counts) <= 2 else "categorical_coded",
                "n_levels": len(g_counts),
                "level_counts": g_counts,
                "zero_variance": len(g_counts) <= 1,
                "vif": vif_map.get("group", 1.0),
            }
    elif unit_of_observation == "subject_pair_difference":
        sample_sizes["paired_subjects"] = int(n_obs)

    # Site breakdown & cross-summaries
    site_summary: Optional[Dict[str, Any]] = None
    if sites is not None:
        site_strs = [str(s) for s in sites]
        site_order = sorted(set(site_strs))
        site_counts = {s: site_strs.count(s) for s in site_order}
        sample_sizes["by_site"] = site_counts
        site_summary = {
            "n_sites": len(site_order),
            "site_counts": site_counts,
            "single_subject_sites": [s for s, c in site_counts.items() if c < 2],
        }
        # Site × group or binary/categorical interest crosstab
        cat_labels_for_site: Optional[List[str]] = None
        if group_labels is not None:
            cat_labels_for_site = [str(g) for g in group_labels]
        elif interest_values is not None:
            iv = np.asarray(interest_values, dtype=np.float64)
            if _classify_factor_values(iv, "interest") in ("binary", "categorical_coded"):
                cat_labels_for_site = [_format_level_key(float(v)) for v in iv]

        if cat_labels_for_site is not None:
            levels = sorted(set(cat_labels_for_site))
            table: Dict[str, Dict[str, int]] = {}
            empty_cells: List[List[str]] = []
            small_cells: List[Dict[str, Any]] = []
            single_level_sites: List[str] = []
            for s in site_order:
                row_counts = {lv: 0 for lv in levels}
                for st_val, lv_val in zip(site_strs, cat_labels_for_site, strict=False):
                    if st_val == s:
                        row_counts[lv_val] += 1
                table[s] = row_counts
                nonzero_levels = sum(1 for c in row_counts.values() if c > 0)
                if len(levels) >= 2 and nonzero_levels <= 1:
                    single_level_sites.append(s)
                for lv, c in row_counts.items():
                    if c == 0:
                        empty_cells.append([s, lv])
                    elif c < 2:
                        small_cells.append({"site": s, "level": lv, "count": c})
            site_summary["site_by_interest_table"] = table
            site_summary["empty_cells"] = empty_cells
            site_summary["small_cells"] = small_cells
            site_summary["perfectly_confounded_with_site"] = bool(
                len(levels) >= 2 and len(single_level_sites) == len(site_order)
            )
        elif interest_values is not None:
            iv = np.asarray(interest_values, dtype=np.float64)
            per_site_stats: Dict[str, Dict[str, Any]] = {}
            ss_within = 0.0
            for s in site_order:
                vals_s = iv[np.array([sv == s for sv in site_strs])]
                vals_s = vals_s[np.isfinite(vals_s)]
                per_site_stats[s] = {
                    "n": int(vals_s.size),
                    "mean": _json_finite(np.mean(vals_s)) if vals_s.size else None,
                    "std": _json_finite(np.std(vals_s)) if vals_s.size else None,
                    "min": _json_finite(np.min(vals_s)) if vals_s.size else None,
                    "max": _json_finite(np.max(vals_s)) if vals_s.size else None,
                }
                if vals_s.size > 0:
                    ss_within += float(np.sum((vals_s - np.mean(vals_s)) ** 2))
            finite_iv = iv[np.isfinite(iv)]
            ss_total = float(np.sum((finite_iv - np.mean(finite_iv)) ** 2)) if finite_iv.size else 0.0
            site_r2 = (
                max(0.0, min(1.0, 1.0 - ss_within / ss_total))
                if ss_total > 1e-12
                else None
            )
            site_summary["site_by_interest_stats"] = per_site_stats
            site_summary["descriptive_r2_site_on_interest"] = _json_finite(site_r2) if site_r2 is not None else None

    strata_summary: Optional[Dict[str, Any]] = None
    if strata is not None:
        strata_strs = [str(s) for s in strata]
        st_order = sorted(set(strata_strs))
        strata_summary = {
            "n_strata": len(st_order),
            "strata_counts": {s: strata_strs.count(s) for s in st_order},
            "matches_sites": (
                sites is not None and [str(s) for s in sites] == strata_strs
            ),
            "note": strata_note,
        }

    ac_summary = _summarize_available_case(missing_report)
    design_residual_df = int(residual_df)
    is_welch_two_sample = design_kind == "two-sample" and var_equal is False
    if ac_summary is not None:
        scope = "global_design_with_available_case_edge_patterns"
        if is_welch_two_sample:
            inference_df_kind = "available_case_welch_satterthwaite"
            inference_df: Optional[int] = None
            inference_df_note = (
                f"Design-matrix residual df = n - rank ({design_residual_df}) characterizes the "
                "full-sample indicator matrix only; under missing='available_case' with Welch's "
                "two-sample t-test, each edge uses its own valid subject counts and group "
                "variances via varying per-edge Welch-Satterthwaite degrees of freedom."
            )
            scope_note = (
                "Rank, VIF, factor summaries, and design residual df (n - rank) are computed "
                "globally across all observations; under missing='available_case' with Welch's "
                "two-sample t-test, edges are tested on edge-specific valid subject subsets with "
                "varying per-edge Welch-Satterthwaite degrees of freedom."
            )
        else:
            inference_df_kind = "available_case_per_edge"
            inference_df = None
            inference_df_note = (
                f"Design-matrix residual df = n - rank ({design_residual_df}) applies only to the "
                "complete-sample design matrix; under missing='available_case', individual edges "
                "are fitted on edge-specific valid subject subsets with lower per-edge residual df."
            )
            scope_note = (
                "Rank, VIF, and factor summaries are computed globally across all "
                "observations; under missing='available_case', individual edges are "
                "fitted on edge-specific subject subsets that may have lower per-edge "
                "residual df or reduced rank."
            )
    else:
        scope = "global_complete_design"
        if is_welch_two_sample:
            inference_df_kind = "varying_welch_satterthwaite"
            inference_df = None
            inference_df_note = (
                f"Design-matrix residual df = n - rank ({design_residual_df}) characterizes the "
                "indicator matrix [intercept, group] only; Welch's two-sample inference uses "
                "varying per-edge Welch-Satterthwaite degrees of freedom determined by each "
                "edge's group variances (not a shared value across edges)."
            )
            scope_note = (
                f"Design summaries (rank, VIF, sample counts, and design residual df = n - rank = "
                f"{design_residual_df}) are computed globally across complete observations; however, "
                "two-sample Welch inference does not share common inference residual df across "
                "edges — Welch-Satterthwaite degrees of freedom depend on per-edge group variances."
            )
        else:
            inference_df_kind = "shared_design_residual_df"
            inference_df = int(design_residual_df)
            inference_df_note = (
                f"All tested edges share complete observations and pooled/design "
                f"residual df = {design_residual_df} (n - rank)."
            )
            scope_note = (
                f"Computed globally across the complete design matrix (all tested edges "
                f"share these observations and design/inference residual df = {design_residual_df})."
            )

    # Pure descriptive warnings (separate from analyze()'s strategy flags,
    # while surfacing actionable issues cleanly).
    diag_flags: List[str] = []
    if rank_deficient:
        diag_flags.append(
            f"Rank-deficient design matrix: rank {rank} < {p} columns "
            f"(nullity = {nullity}); some parameters are collinear/aliased."
        )
    if non_intercept_constant_cols:
        diag_flags.append(
            "Constant non-intercept column(s) detected: "
            + ", ".join(non_intercept_constant_cols)
            + "."
        )
    if has_non_estimable:
        diag_flags.append("Target contrast is not estimable from the design matrix.")
    if base["vif_max"] > 10.0:
        diag_flags.append(
            f"High collinearity: VIF for '{base['vif_max_col']}' = {base['vif_max']:.2f} (> 10)."
        )
    elif base["vif_max"] > 5.0:
        diag_flags.append(
            f"Moderate collinearity: VIF for '{base['vif_max_col']}' = {base['vif_max']:.2f} (> 5)."
        )
    if site_summary is not None:
        if site_summary.get("perfectly_confounded_with_site"):
            diag_flags.append(
                "Each site contains only a single group/level (site and interest are completely separated)."
            )
        elif site_summary.get("empty_cells"):
            diag_flags.append(
                f"Site × interest table has {len(site_summary['empty_cells'])} empty cell(s)."
            )
        if site_summary.get("single_subject_sites"):
            diag_flags.append(
                "Site(s) with fewer than 2 observations: "
                + ", ".join(site_summary["single_subject_sites"])
                + "."
            )
    if ac_summary is not None and ac_summary.get("n_edges_untested", 0) > 0:
        diag_flags.append(
            f"available_case skipped {ac_summary['n_edges_untested']} edge(s) below coverage/estimability requirements."
        )

    return {
        "design_kind": design_kind,
        "unit_of_observation": unit_of_observation,
        "scope": scope,
        "scope_note": scope_note,
        "design_note": design_note,
        "n_observations": int(n_obs),
        "n_columns": int(p),
        "matrix_rank": int(rank),
        "rank_deficient": bool(rank_deficient),
        "nullity": int(nullity),
        "residual_df": int(design_residual_df),
        "design_residual_df": int(design_residual_df),
        "inference_df": inference_df,
        "inference_df_kind": inference_df_kind,
        "inference_df_note": inference_df_note,
        "columns": col_names,
        "column_roles": {nm: roles.get(nm, "intercept" if nm == "intercept" else "predictor") for nm in col_names},
        "contrast_estimable": estimable_report,
        "focus_contrast": focus_contrast,
        "condition_number_XtX": _json_finite(base["condition_number"]),
        "condition_number_XtX_infinite": not bool(np.isfinite(base["condition_number"])),
        "condition_number_X": _json_finite(base["condition_number_X"]),
        "condition_number_standardized": _json_finite(base["condition_number_standardized"]),
        "condition_number_note": CONDITION_NUMBER_NOTE,
        "vif": vif_map,
        "vif_max": _json_finite(base["vif_max"]),
        "vif_max_col": base["vif_max_col"],
        "vif_convention": VIF_CONVENTION_NOTE,
        "constant_columns": base["constant_columns"],
        "non_intercept_constant_columns": non_intercept_constant_cols,
        "high_correlation_pairs": high_corr_pairs,
        "correlation": [[float(x) for x in row] for row in corr_arr],
        "sample_sizes": sample_sizes,
        "column_summaries": column_summaries,
        "categorical_factors": categorical_factors,
        "continuous_factors": continuous_factors,
        "interest_name": interest_name,
        "site_summary": site_summary,
        "strata_summary": strata_summary,
        "available_case": ac_summary,
        "flags": diag_flags,
    }


def block_mass(
    p_full: npt.NDArray[np.float64],
    net_labels: npt.NDArray[np.int_],
    alpha: float = 0.05,
    *,
    return_upper: bool = True,
) -> npt.NDArray[np.int_]:
    r"""Aggregate edge-level survival counts into a network-block matrix.

    For the full network-pair table (both tails, proportions, ``-log10 p``
    mass) from a result and an atlas, use
    :meth:`InferenceResult.network_summary <conninfpy.results.InferenceResult.network_summary>`.

    For an upper-triangular set of significant edges (``p <= alpha``),
    count how many fall within each unordered pair of network labels
    ``(i, j)``. The diagonal counts within-network edges.

    Parameters
    ----------
    p_full : ndarray of shape (N, N)
        Edge-wise p-value matrix (typically symmetric, with diagonal=1).
    net_labels : ndarray of shape (N,)
        Integer network assignment per node, ``0..K-1``. The Yeo-7
        partition stored in ``abide_prepared.npz['net_labels']`` is the
        canonical use case.
    alpha : float, default ``0.05``
        Significance threshold; edges with ``p <= alpha`` are counted.
    return_upper : bool, default ``True``
        If ``True``, the returned matrix is upper-triangular (lower
        triangle zero-filled). If ``False``, the matrix is symmetric.

    Returns
    -------
    M : ndarray of shape (K, K), int
        Block-mass matrix where ``M[i, j]`` is the number of significant
        edges with one endpoint in network ``i`` and the other in ``j``.

    Examples
    --------
    >>> import numpy as np
    >>> from conninfpy import block_mass
    >>> p = np.array([[1, 0.01, 0.5], [0.01, 1, 0.6], [0.5, 0.6, 1]])
    >>> labels = np.array([0, 0, 1])
    >>> block_mass(p, labels, alpha=0.05)
    array([[1, 0],
           [0, 0]])
    """
    if p_full.ndim != 2 or p_full.shape[0] != p_full.shape[1]:
        raise ValueError(f"p_full must be (N, N), got {p_full.shape}.")
    if net_labels.shape[0] != p_full.shape[0]:
        raise ValueError(
            f"net_labels has {net_labels.shape[0]} entries but p_full is "
            f"{p_full.shape[0]} × {p_full.shape[0]}."
        )

    K = int(net_labels.max() + 1)
    M = np.zeros((K, K), dtype=np.int64)

    sig = (p_full <= alpha) & np.isfinite(p_full)
    np.fill_diagonal(sig, False)

    iu, ju = np.triu_indices_from(p_full, k=1)
    sig_idx = sig[iu, ju]
    iu, ju = iu[sig_idx], ju[sig_idx]

    bi = net_labels[iu]
    bj = net_labels[ju]
    lo = np.minimum(bi, bj)
    hi = np.maximum(bi, bj)
    np.add.at(M, (lo, hi), 1)

    if not return_upper:
        # mirror to lower triangle (off-diagonal only)
        M = M + np.triu(M, 1).T

    return M
