"""Participant-level splits, folds and paired bootstrap draws."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def split_participants(participants: pd.DataFrame, test_counts: dict[str, int],
                       seed: int, split_from: Path | None = None) -> pd.DataFrame:
    result = participants.copy()
    if split_from is not None:
        previous = pd.read_csv(split_from)
        if previous.participant_id.duplicated().any() or set(previous.participant_id) != set(result.participant_id):
            raise ValueError("Saved split must contain exactly the current participants, once each.")
        previous = previous.set_index("participant_id").loc[result.participant_id]
        if not np.array_equal(previous.site.to_numpy(), result.site.to_numpy()):
            raise ValueError("Saved split has inconsistent sites.")
        result["split"] = previous["split"].to_numpy()
        if not result["split"].isin(["development", "test"]).all():
            raise ValueError("Saved split must use development/test labels.")
    else:
        rng = np.random.default_rng(seed)
        result["split"] = "development"
        for site in sorted(result.site.unique()):
            pool = result.index[result.site == site].to_numpy()
            n = test_counts[site]
            if n < 2 or n >= len(pool):
                raise ValueError(f"Invalid test count for {site}: {n}.")
            result.loc[rng.choice(pool, n, replace=False), "split"] = "test"
    for site in result.site.unique():
        for split in ("development", "test"):
            if ((result.site == site) & (result["split"] == split)).sum() < 2:
                raise ValueError("Each site needs at least two participants in each split.")
    return result


def development_folds(participants: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    assignments = np.full(len(participants), -1, dtype=int)
    for site in sorted(participants.site.unique()):
        pool = np.flatnonzero((participants.site == site) & (participants["split"] == "development"))
        if len(pool) < n_folds:
            raise ValueError(f"Site {site} has fewer development participants than CV folds.")
        pool = rng.permutation(pool)
        assignments[pool] = np.arange(len(pool)) % n_folds
    return assignments


def participant_bootstraps(participants: pd.DataFrame, n_bootstraps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pools = [participants.loc[(participants.site == site) & (participants["split"] == "development"),
                              "participant_index"].to_numpy() for site in sorted(participants.site.unique())]
    return np.asarray([np.concatenate([rng.choice(pool, len(pool), replace=True) for pool in pools])
                       for _ in range(n_bootstraps)])


def scan_indices(scans: pd.DataFrame, people: np.ndarray) -> np.ndarray:
    """Expand draws in order, preserving duplicate draws and the original EO/EC pair."""
    lookup = {p: np.flatnonzero(scans.participant_index.to_numpy() == p)
              for p in np.unique(people)}
    for indices in lookup.values():
        if len(indices) != 2 or set(scans.iloc[indices].state) != {0, 1}:
            raise ValueError("Every participant must have exactly one EO and one EC scan.")
    return np.concatenate([lookup[p] for p in people])
