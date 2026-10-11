"""FWER summaries and binomial calibration checks shared by simulations."""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd

def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute Clopper-Pearson exact binomial confidence interval.

    This is the "exact" confidence interval for binomial proportion,
    more accurate than the Wald interval for small samples or extreme proportions.
    """
    from scipy import stats

    if k == 0:
        ci_low = 0.0
    else:
        ci_low = stats.beta.ppf(alpha / 2, k, n - k + 1)

    if k == n:
        ci_high = 1.0
    else:
        ci_high = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)

    return ci_low, ci_high

def _ci_not_liberal(k: int, n: int, target_alpha: float) -> Tuple[float, float, bool]:
    """Return a 95% CP interval and whether it excludes liberal inflation.

    The validation question for an FWER-controlling method is one-sided:
    "is there evidence that the empirical false-positive probability is
    above alpha?" Conservative methods are valid, although less powerful.
    We therefore fail only when the lower bound of the 95% interval is
    above the target alpha.
    """
    ci_low, ci_high = clopper_pearson_ci(k, n, alpha=0.05)
    return ci_low, ci_high, bool(ci_low <= target_alpha)

def _binom_p_above(k: int, n: int, target_alpha: float) -> float:
    """Exact one-sided p-value for H0: FWER <= alpha, i.e. P(X >= k | n, alpha)."""
    from scipy import stats

    return float(stats.binom.sf(k - 1, n, target_alpha))

def simultaneous_calibration_test(summary_df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Bonferroni over all cells of the exact binomial tests of FWER <= alpha.

    The per-cell Clopper-Pearson intervals are descriptive: with many cells a
    few exclude alpha by chance. This is the simultaneous check, done
    separately for the directional tails (two per cell) and for the
    two-sided family (alpha/2 per tail).
    """
    directional = np.concatenate([summary_df["fwer_pos_binom_p"], summary_df["fwer_neg_binom_p"]])
    two_sided = summary_df["fwer_two_sided_binom_p"].to_numpy()
    out = {}
    for name, pvals in (("directional", directional), ("two_sided", two_sided)):
        k = len(pvals)
        p_adj = min(1.0, float(np.min(pvals)) * k)
        out[name] = {"n_tests": k, "min_p": float(np.min(pvals)), "p_bonferroni": p_adj,
                     "pass": p_adj > alpha}
    out["pass"] = out["directional"]["pass"] and out["two_sided"]["pass"]
    return out

def compute_fpr_summary(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Compute null-calibration statistics by method (and n_samples if present).

    Reports two key metrics:
    1. Directional FWER: P(at least 1 FP in a single tail | H0)
       - ConnInfPy returns positive and negative p-maps with separate
         max-statistic nulls, so this is the primary calibration target.
       - The summary reports the worst of the two tails.

    2. Two-sided Bonferroni-over-tails FWER:
       - P(any FP in either tail at alpha/2 per tail | H0).
       - This is the correct diagnostic for one undirected/two-sided
         family when using the existing per-tail p-maps.

    The unadjusted "either tail at alpha" rate is also reported as a
    diagnostic. It is expected to approach <= 2*alpha, not alpha.
    """
    from scipy import stats

    # Determine grouping columns
    has_n_samples = "n_samples" in df.columns
    multi_n = has_n_samples and df["n_samples"].nunique() > 1
    group_cols = ["method", "n_samples"] if multi_n else ["method"]

    summary_rows = []

    for group_key, method_df in df.groupby(group_cols):
        if multi_n:
            method, n_samples = group_key
        else:
            method = group_key if isinstance(group_key, str) else group_key[0]
            n_samples = int(method_df["n_samples"].iloc[0]) if has_n_samples else None

        n_runs = len(method_df)
        n_edges = method_df["n_edges"].iloc[0]

        # Directional FWER: proportion of runs with ANY significant edge
        # in each tail. This matches the separate positive/negative
        # max-statistic nulls returned by compute_p_val().
        n_runs_with_fp_pos = int(method_df["any_significant_pos"].sum())
        fwer_pos = n_runs_with_fp_pos / n_runs
        n_runs_with_fp_neg = int(method_df["any_significant_neg"].sum())
        fwer_neg = n_runs_with_fp_neg / n_runs

        fwer_pos_ci_low, fwer_pos_ci_high, fwer_pos_valid = _ci_not_liberal(
            n_runs_with_fp_pos, n_runs, alpha
        )
        fwer_neg_ci_low, fwer_neg_ci_high, fwer_neg_valid = _ci_not_liberal(
            n_runs_with_fp_neg, n_runs, alpha
        )

        if fwer_neg > fwer_pos:
            fwer_max_tail = fwer_neg
            fwer_max_tail_name = "negative"
            n_runs_with_fp_max_tail = n_runs_with_fp_neg
            fwer_ci_low = fwer_neg_ci_low
            fwer_ci_high = fwer_neg_ci_high
        else:
            fwer_max_tail = fwer_pos
            fwer_max_tail_name = "positive"
            n_runs_with_fp_max_tail = n_runs_with_fp_pos
            fwer_ci_low = fwer_pos_ci_low
            fwer_ci_high = fwer_pos_ci_high

        # Validity, not exact equality: conservative FWER is acceptable.
        directional_fwer_valid = bool(fwer_pos_valid and fwer_neg_valid)

        # Diagnostic: if users declare significance in either tail at alpha
        # without a tail correction, the family is the union of two directional
        # families. It should be judged against <= 2*alpha, not alpha.
        n_runs_with_fp_either_alpha = int(method_df["any_significant"].sum())
        fwer_either_at_alpha = n_runs_with_fp_either_alpha / n_runs

        # Correct two-sided diagnostic with existing per-tail p-maps:
        # threshold each directional map at alpha/2.
        two_sided_alpha_per_tail = alpha / 2.0
        two_sided_hits = (
            (method_df["min_p_pos"] <= two_sided_alpha_per_tail)
            | (method_df["min_p_neg"] <= two_sided_alpha_per_tail)
        )
        n_runs_with_fp_two_sided = int(two_sided_hits.sum())
        fwer_two_sided_bonf = n_runs_with_fp_two_sided / n_runs
        (
            fwer_two_sided_ci_low,
            fwer_two_sided_ci_high,
            fwer_two_sided_valid,
        ) = _ci_not_liberal(n_runs_with_fp_two_sided, n_runs, alpha)

        # Edge-wise FPR is a descriptive diagnostic. FWER-controlling methods
        # should generally be far below alpha here.
        mean_fpr_pos = method_df["fpr_pos"].mean()
        mean_fpr_neg = method_df["fpr_neg"].mean()
        mean_fpr_total = method_df["fpr_total"].mean()
        if mean_fpr_neg > mean_fpr_pos:
            edge_fpr_col = "fpr_neg"
            mean_fpr_max_tail = mean_fpr_neg
        else:
            edge_fpr_col = "fpr_pos"
            mean_fpr_max_tail = mean_fpr_pos
        se_fpr_max_tail = method_df[edge_fpr_col].std() / np.sqrt(n_runs)

        t_crit = stats.t.ppf(0.975, df=n_runs - 1)
        fpr_ci_low = max(0, mean_fpr_max_tail - t_crit * se_fpr_max_tail)
        fpr_ci_high = min(1, mean_fpr_max_tail + t_crit * se_fpr_max_tail)
        edge_fpr_not_liberal = bool(fpr_ci_low <= alpha)

        row = {
            "method": method,
            "n_null_runs": n_runs,
            "n_edges": n_edges,
            # FWER
            "fwer_pos": fwer_pos,
            "fwer_neg": fwer_neg,
            "fwer_both": fwer_either_at_alpha,
            "fwer_either_at_alpha": fwer_either_at_alpha,
            "fwer_either_expected_upper": min(1.0, 2.0 * alpha),
            "fwer_either_independent_expected": 1.0 - (1.0 - alpha) ** 2,
            "fwer_pos_ci_low": fwer_pos_ci_low,
            "fwer_pos_ci_high": fwer_pos_ci_high,
            "fwer_neg_ci_low": fwer_neg_ci_low,
            "fwer_neg_ci_high": fwer_neg_ci_high,
            "fwer_max_tail": fwer_max_tail,
            "fwer_max_tail_name": fwer_max_tail_name,
            "n_runs_with_fp_max_tail": n_runs_with_fp_max_tail,
            "fwer_ci_low": fwer_ci_low,
            "fwer_ci_high": fwer_ci_high,
            "fwer_calibrated": directional_fwer_valid,
            "directional_fwer_valid": directional_fwer_valid,
            "two_sided_alpha_per_tail": two_sided_alpha_per_tail,
            "fwer_two_sided_bonf": fwer_two_sided_bonf,
            "fwer_two_sided_bonf_ci_low": fwer_two_sided_ci_low,
            "fwer_two_sided_bonf_ci_high": fwer_two_sided_ci_high,
            "fwer_two_sided_bonf_valid": fwer_two_sided_valid,
            # Exact one-sided binomial p-values for H0: FWER <= alpha
            "fwer_pos_binom_p": _binom_p_above(n_runs_with_fp_pos, n_runs, alpha),
            "fwer_neg_binom_p": _binom_p_above(n_runs_with_fp_neg, n_runs, alpha),
            "fwer_two_sided_binom_p": _binom_p_above(n_runs_with_fp_two_sided, n_runs, alpha),
            # Edge-wise FPR
            "mean_fpr_pos": mean_fpr_pos,
            "mean_fpr_neg": mean_fpr_neg,
            "mean_fpr_max_tail": mean_fpr_max_tail,
            "mean_fpr_total": mean_fpr_total,
            "fpr_se": se_fpr_max_tail,
            "fpr_ci_low": fpr_ci_low,
            "fpr_ci_high": fpr_ci_high,
            "fpr_calibrated": edge_fpr_not_liberal,
            "expected": alpha,
            # Additional
            "mean_n_fp_pos": method_df["n_significant_pos"].mean(),
            "mean_n_fp_neg": method_df["n_significant_neg"].mean(),
            "mean_elapsed_time": method_df["elapsed_time"].mean(),
        }
        if has_n_samples:
            row["n_samples"] = n_samples
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)
