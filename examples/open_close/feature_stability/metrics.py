"""Metrics for a fixed scan panel with participant-level resampling."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score

METRICS = ("auc", "balanced_accuracy", "brier", "log_loss")


def scores(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, probability >= 0.5)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
    }


def agreement_icc(probabilities: np.ndarray) -> float:
    """ICC(A,1): rows are scan targets, columns are fitted bootstrap models.

    Two-way absolute agreement, single model. Descriptive only: the models
    share training participants, and EO/EC scan targets share participants.
    """
    n, k = probabilities.shape
    if n < 2 or k < 2:
        return float("nan")
    row = probabilities.mean(axis=1, keepdims=True)
    col = probabilities.mean(axis=0, keepdims=True)
    grand = probabilities.mean()
    ms_row = k * np.var(row, ddof=1)
    ms_col = n * np.var(col, ddof=1)
    ms_error = np.sum((probabilities - row - col + grand) ** 2) / ((n - 1) * (k - 1))
    denominator = ms_row + (k - 1) * ms_error + k * (ms_col - ms_error) / n
    return float((ms_row - ms_error) / denominator) if denominator > 0 else float("nan")


def prediction_stability(probabilities: np.ndarray, reference: np.ndarray) -> pd.DataFrame:
    """One row per scan; input is (bootstrap models, fixed test scans)."""
    n = probabilities.shape[0]
    positive = (probabilities >= 0.5).sum(axis=0)
    positive_fraction = positive / n
    disagreement = 2 * positive * (n - positive) / (n * (n - 1))
    return pd.DataFrame({
        "probability_mean": probabilities.mean(axis=0),
        "probability_sd": probabilities.std(axis=0, ddof=1),
        "probability_q025": np.quantile(probabilities, 0.025, axis=0),
        "probability_q975": np.quantile(probabilities, 0.975, axis=0),
        "closed_decision_fraction": positive_fraction,
        "consensus_agreement": np.maximum(positive_fraction, 1 - positive_fraction),
        "pairwise_decision_disagreement": disagreement,
        "decision_flip_vs_reference": ((probabilities >= 0.5) != (reference >= 0.5)).mean(axis=0),
    })


def coefficient_correlations(weights: np.ndarray) -> np.ndarray:
    """Pairwise Pearson pattern correlations on a common, unscaled feature axis."""
    centered = weights - weights.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    denominator = norms[:, None] * norms[None, :]
    numerator = centered @ centered.T
    correlation = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                            where=denominator > 0)
    return correlation[np.triu_indices(len(weights), k=1)]


def sign_stability(weights: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "coefficient_mean": weights.mean(axis=0),
        "coefficient_sd": weights.std(axis=0, ddof=1),
        "positive_fraction": (weights > 0).mean(axis=0),
        "negative_fraction": (weights < 0).mean(axis=0),
        "zero_fraction": (weights == 0).mean(axis=0),
        "sign_agreement": np.maximum((weights > 0).mean(axis=0), (weights < 0).mean(axis=0)),
    })


def test_participant_draws(scans: pd.DataFrame, n_bootstraps: int, seed: int) -> np.ndarray:
    """Site-stratified draws of test scan indices, copying EO/EC pairs together."""
    rng = np.random.default_rng(seed)
    sites = sorted(scans.site.unique())
    by_person = {p: np.flatnonzero(scans.participant_index.to_numpy() == p)
                 for p in scans.participant_index.unique()}
    site_people = [scans.loc[scans.site == site, "participant_index"].unique() for site in sites]
    return np.asarray([
        np.concatenate([by_person[p] for people in site_people
                        for p in rng.choice(people, len(people), replace=True)])
        for _ in range(n_bootstraps)
    ])


def paired_differences(table: pd.DataFrame, pairs: list[tuple[str, str]],
                       draw_column: str, uncertainty: str) -> pd.DataFrame:
    """Paired contrasts; intervals describe the specified resampling distribution."""
    records = []
    for first, second in pairs:
        a = table[table.pipeline == first].set_index([draw_column, "site"])
        b = table[table.pipeline == second].set_index([draw_column, "site"])
        delta = a[list(METRICS)] - b[list(METRICS)]
        for site in delta.index.get_level_values("site").unique():
            for metric in METRICS:
                values = delta.xs(site, level="site")[metric].to_numpy()
                records.append({
                    "first": first, "second": second, "contrast": "first_minus_second",
                    "site": site, "metric": metric, "uncertainty": uncertainty,
                    "mean_difference": values.mean(),
                    "q025": np.quantile(values, 0.025), "q975": np.quantile(values, 0.975),
                })
    return pd.DataFrame(records)
