"""Connectivity estimation from subject time series.

Only frames finite in *all* ROIs are used.  The optional diagnostics expose
valid-frame counts and per-observation shrinkage; inspect these against group
membership before inference, because they can differ systematically.
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from numbers import Real

import numpy as np
from scipy.linalg import cho_solve
from sklearn.covariance import ledoit_wolf, oas

_DEFAULT = object()


def compute_connectivity(ts, kind="correlation", shrinkage=_DEFAULT, standardize=True,
                         *, return_diagnostics=False, min_valid_tr=10):
    """Estimate correlation or partial correlation for one or more observations.

    ``ts`` is (T, ROI), (observations, T, ROI), or a list of (T, ROI)
    arrays with possibly different T. A single input returns a 2-D matrix;
    multiple observations return a 3-D stack. Each observation is centered,
    and, by default, standardized with population standard deviation before
    covariance estimation. Only complete finite frames are retained.

    Unspecified shrinkage means none for Pearson and Ledoit-Wolf for partial
    correlation. Explicit ``None`` disables shrinkage. ``'ledoit_wolf'`` and
    ``'oas'`` use scikit-learn's maximum-likelihood covariance estimators;
    a float in [0, 1] specifies the same shrinkage intensity for all subjects,
    towards ``trace(covariance) / n_rois * I``. Partial correlation requires
    positive definite covariance. Constant ROIs yield NaNs for Pearson and an
    informative error for partial correlation. Fewer than two valid frames is
    an error. A warning is issued below ``min_valid_tr`` (default 10).

    With ``return_diagnostics=True``, return ``(matrices, diagnostics)``;
    diagnostics contains arrays ``n_valid_tr`` and ``delta_s`` and the
    resolved estimator settings. ``delta_s`` is zero without shrinkage.
    Shrinkage intensities estimated separately may induce group differences;
    compare them and frame counts by group and consider them as covariates.
    """
    if kind not in ("correlation", "partial_correlation"):
        raise ValueError("kind must be 'correlation' or 'partial_correlation'.")
    if shrinkage is _DEFAULT:
        shrinkage = None if kind == "correlation" else "ledoit_wolf"
    if isinstance(shrinkage, bool) or not (shrinkage is None or
        (isinstance(shrinkage, str) and shrinkage in ("ledoit_wolf", "oas")) or
        (isinstance(shrinkage, Real) and np.isfinite(shrinkage) and 0 <= shrinkage <= 1)):
        raise ValueError("shrinkage must be None, 'ledoit_wolf', 'oas', or a float in [0, 1].")
    if not isinstance(min_valid_tr, int) or min_valid_tr < 2:
        raise ValueError("min_valid_tr must be an integer >= 2.")
    single = isinstance(ts, np.ndarray) and ts.ndim == 2
    if single:
        series = [ts]
    elif isinstance(ts, np.ndarray) and ts.ndim == 3:
        series = list(ts)
    elif isinstance(ts, (list, tuple)):
        series = list(ts)
    else:
        raise ValueError("ts must be (T, ROI), (subjects, T, ROI), or a list of (T, ROI) arrays.")
    if not series:
        raise ValueError("At least one observation is required.")
    cleaned = []
    groups = defaultdict(list)
    n_rois = None
    counts = []
    for i, raw in enumerate(series):
        x = np.asarray(raw, dtype=float)
        if x.ndim != 2 or x.shape[1] < 2:
            raise ValueError(f"Observation {i} must have shape (T, ROI) with at least two ROIs.")
        if n_rois is None:
            n_rois = x.shape[1]
        if x.shape[1] != n_rois:
            raise ValueError(f"Observation {i} has a different number of ROIs.")
        x = x[np.isfinite(x).all(axis=1)]
        n = len(x)
        if n < 2:
            raise ValueError(f"Observation {i} has only {n} complete finite frames; at least two are required.")
        if n < min_valid_tr:
            warnings.warn(f"Observation {i} has {n} valid frames (< {min_valid_tr}); connectivity may be unstable.", UserWarning, stacklevel=2)
        counts.append(n)
        cleaned.append(x)
        groups[n].append(i)
    result = np.empty((len(series), n_rois, n_rois), dtype=float)
    deltas = np.zeros(len(series))
    # Keep temporary covariance blocks near 32 MiB; output itself is unavoidable.
    for n, indices in groups.items():
        block_size = max(1, min(32, 32 * 1024**2 // (8 * (n * n_rois + 2 * n_rois**2))))
        for start in range(0, len(indices), block_size):
            ids = indices[start:start + block_size]
            x = np.stack([cleaned[i] for i in ids])
            x -= x.mean(axis=1, keepdims=True)
            std = np.sqrt(np.mean(x * x, axis=1))
            for j, i in enumerate(ids):
                bad = np.flatnonzero(std[j] == 0)
                if len(bad):
                    if kind == "partial_correlation":
                        raise ValueError(f"Observation {i} has constant ROIs {bad.tolist()}; partial correlation is undefined.")
                    warnings.warn(f"Observation {i} has constant ROIs {bad.tolist()}; their correlations are undefined (NaN).", UserWarning, stacklevel=2)
            if standardize:
                x /= np.where(std == 0, 1, std)[:, None, :]
            covs = x.transpose(0, 2, 1) @ x / n
            for j, i in enumerate(ids):
                if shrinkage in ("ledoit_wolf", "oas"):
                    covs[j], deltas[i] = (ledoit_wolf if shrinkage == "ledoit_wolf" else oas)(x[j], assume_centered=True)
                elif shrinkage is not None:
                    deltas[i] = float(shrinkage)
                    mu = np.trace(covs[j]) / n_rois
                    covs[j] *= 1 - deltas[i]
                    covs[j].flat[::n_rois + 1] += deltas[i] * mu
            if kind == "partial_correlation":
                for j, i in enumerate(ids):
                    if deltas[i] == 0 and n <= n_rois:
                        raise ValueError(f"Observation {i}: covariance is singular (valid frames <= ROIs); use shrinkage.")
                    if deltas[i] == 0 and np.linalg.cond(covs[j]) > 1 / (n_rois * np.finfo(float).eps):
                        raise ValueError(f"Observation {i}: covariance is singular or ill-conditioned; use shrinkage.")
                try:
                    factors = np.linalg.cholesky(covs)
                except np.linalg.LinAlgError as exc:
                    # Identify the observation while retaining batched factorization
                    # for the usual positive-definite case.
                    for j, i in enumerate(ids):
                        try:
                            np.linalg.cholesky(covs[j])
                        except np.linalg.LinAlgError:
                            raise ValueError(f"Observation {i}: covariance is singular or not positive definite; use shrinkage.") from exc
                    raise ValueError("Covariance is singular or not positive definite; use shrinkage.") from exc
            for j, i in enumerate(ids):
                cov = covs[j]
                if kind == "partial_correlation":
                    precision = cho_solve((factors[j], True), np.eye(n_rois), check_finite=False)
                    scale = np.sqrt(np.diag(precision))
                    matrix = -precision / np.outer(scale, scale)
                    np.fill_diagonal(matrix, 1.)
                else:
                    scale = np.sqrt(np.diag(cov))
                    with np.errstate(invalid="ignore", divide="ignore"):
                        matrix = cov / np.outer(scale, scale)
                    # Shrinkage must not turn an undefined ROI into measured connectivity.
                    bad = std[j] == 0
                    matrix[bad, :] = np.nan
                    matrix[:, bad] = np.nan
                    np.clip(matrix, -1., 1., out=matrix)
                result[i] = matrix
    diagnostics = {"n_valid_tr": np.asarray(counts), "delta_s": deltas,
                   "kind": kind, "shrinkage": shrinkage, "standardize": bool(standardize)}
    result = result[0] if single else result
    return (result, diagnostics) if return_diagnostics else result


def connectivity_from_dataset(dataset, *, kind="correlation", shrinkage=_DEFAULT,
                              standardize=True, group_column=None, min_valid_tr=10):
    """Convert a timeseries dataset in place and attach diagnostics to phenotype."""
    from conninfpy.preprocessing import (
        SOURCE_COMPUTED, connectivity_method_from_kind, fact, set_preprocessing_facts,
    )

    if dataset.data_kind != "timeseries":
        raise ValueError("Connectivity construction requires timeseries data.")
    matrices, info = compute_connectivity(dataset.data, kind, shrinkage, standardize,
                                          return_diagnostics=True, min_valid_tr=min_valid_tr)
    if matrices.ndim == 2:
        matrices = matrices[None, ...]
    if len(dataset.pheno) != len(matrices):
        raise ValueError("Phenotype rows must match time-series observations.")
    dataset.data = matrices
    dataset.data_kind = "correlation"
    dataset.pheno = dataset.pheno.copy()
    dataset.pheno["n_valid_tr"] = info["n_valid_tr"]
    dataset.pheno["delta_s"] = info["delta_s"]
    dataset.provenance["connectivity"] = {k: v for k, v in info.items() if k not in ("n_valid_tr", "delta_s")}
    # Three-state preprocessing facts with an explicit source. The estimator
    # facts state what THIS call actually resolved and therefore override any
    # older declaration about the same facts (a manifest may declare what some
    # upstream pipeline did; these matrices were just computed here). Facts
    # about the upstream data itself (gsr, tr_seconds) are untouched.
    set_preprocessing_facts(
        dataset,
        connectivity_method=fact(connectivity_method_from_kind(info["kind"]), SOURCE_COMPUTED),
        shrinkage=fact(info["shrinkage"], SOURCE_COMPUTED),
        standardize=fact(bool(info["standardize"]), SOURCE_COMPUTED),
    )
    warn_group_diagnostics(dataset, group_column)
    return dataset


def warn_group_diagnostics(dataset, group_column=None):
    """Warn on observed group mean differences, without a significance test."""
    if group_column is None:
        group_column = next((c for c in ("group", "group_interest", "Diagnosis") if c in dataset.pheno), None)
    if group_column is None or group_column not in dataset.pheno:
        return
    pheno = dataset.pheno
    for col in ("n_valid_tr", "delta_s"):
        if col not in pheno:
            continue
        grouped = pheno.groupby(group_column, observed=True)[col]
        samples = {str(k): np.sort(v.dropna().to_numpy(float)) for k, v in grouped}
        samples = {k: v for k, v in samples.items() if len(v)}
        if len(samples) < 2:
            continue
        # Compare empirical distributions descriptively; this is not a test.
        def distribution(v):
            values, counts = np.unique(v, return_counts=True)
            return values, counts / len(v)
        distributions = [distribution(v) for v in samples.values()]
        a, b = distributions[0]
        if any(len(x) != len(a) or not np.allclose(x, a, rtol=1e-10, atol=1e-12)
               or not np.allclose(y, b, rtol=1e-10, atol=1e-12) for x, y in distributions[1:]):
            details = ", ".join(f"{k}: mean={v.mean():.4g}, range={v.min():.4g}–{v.max():.4g}" for k, v in samples.items())
            msg = (f"Observed empirical distributions of {col} differ by {group_column} ({details}); "
                   "this is descriptive, not a significance test. Inspect the distributions and consider this diagnostic as a covariate when appropriate.")
            if msg not in dataset.warnings:
                warnings.warn(msg, UserWarning, stacklevel=2)
                dataset.warnings.append(msg)
