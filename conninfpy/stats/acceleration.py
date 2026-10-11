"""
Permutation acceleration methods for faster inference.

Implements tail approximation via Generalized Pareto Distribution (GPD)
and gamma approximation, following Winkler et al. (2016) "Faster
permutation inference in brain imaging", NeuroImage.

A finite set of J permutations resolves p-values down to about 1/J. Where
fewer than ``MIN_EXCEEDANCES`` null maxima reach the observed statistic, the
permutation p-value is too coarse and a fitted tail is used instead; every
other p-value stays the exact permutation p-value (Knijnenburg et al., 2009).
The fitted tail is an approximation, so the p-values it replaces are not
exact.

References
----------
Winkler et al. (2016). Faster permutation inference in brain imaging.
    NeuroImage, doi:10.1016/j.neuroimage.2016.05.068
Knijnenburg et al. (2009). Fewer permutations, more accurate P-values.
    Bioinformatics 25(12), i161-i168.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
from scipy import stats

logger = logging.getLogger(__name__)


__all__ = [
    "fit_gpd_tail",
    "fit_gamma_tail",
    "compute_p_values_accelerated",
]

# The tail fit replaces a permutation p-value only when fewer than this many
# null maxima reach the observed statistic; above it the permutation count is
# precise enough on its own (Knijnenburg et al., 2009).
MIN_EXCEEDANCES = 10


def _empirical_p(
    null_dist: npt.NDArray[np.float64],
    observed: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Tie-inclusive max-statistic p-values (Phipson-Smyth +1) and counts."""
    null_dist = np.asarray(null_dist, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if null_dist.ndim != 1 or not null_dist.size:
        raise ValueError(
            "null_dist must be a nonempty one-dimensional permutation null."
        )
    if not np.isfinite(null_dist).all() or not np.isfinite(observed).all():
        raise ValueError(
            "Tail approximation requires finite observed and null statistics."
        )
    count = np.sum(observed[..., np.newaxis] <= null_dist, axis=-1)
    return (count + 1.0) / (len(null_dist) + 1.0), count


def _replace_sparse_tail(
    p_empirical: npt.NDArray[np.float64],
    count: npt.NDArray[np.int64],
    tail_p: npt.NDArray[np.float64],
    J: int,
    min_exceedances: int,
    replacement_mask: Optional[npt.NDArray[np.bool_]] = None,
) -> npt.NDArray[np.float64]:
    """Use the fitted tail only where the permutation count is too coarse.

    ``tail_p`` holds fitted p-values for the statistics with
    ``count < min_exceedances`` intersected with ``replacement_mask`` when
    supplied (in that order). Model estimates can fall below 1/J. The upper
    cap keeps the switch from increasing p-values; float64's smallest normal
    positive value is only an underflow guard, not a statistical resolution
    or accuracy claim. Invalid fitted values retain their empirical p-values.
    """
    p = np.copy(p_empirical)
    sparse = count < min_exceedances
    if replacement_mask is not None:
        sparse &= replacement_mask
    upper = min(1.0, (min_exceedances + 1.0) / (J + 1.0))
    tail_p = np.asarray(tail_p, dtype=np.float64)
    valid = np.isfinite(tail_p) & (tail_p >= 0) & (tail_p <= 1)
    replacement = p[sparse].copy()
    replacement[valid] = np.clip(tail_p[valid], np.finfo(float).tiny, upper)
    p[sparse] = replacement
    return p


def _fit_gpd_mom(
    exceedances: npt.NDArray[np.float64],
) -> tuple[float, float, bool]:
    """
    Fit GPD parameters by method of moments.

    Parameters
    ----------
    exceedances : ndarray
        Positive values y = T - u for T > u (exceedances above threshold u).

    Returns
    -------
    sigma : float
        Scale parameter.
    xi : float
        Shape parameter.
    valid : bool
        Whether the fit is valid (enough exceedances, finite parameters).
    """
    n = len(exceedances)
    if n < 10:
        return 0.0, 0.0, False

    y_bar = np.mean(exceedances)
    s2 = np.var(exceedances, ddof=1)

    if y_bar <= 0 or s2 <= 0:
        return 0.0, 0.0, False

    # Method of moments (Hosking & Wallis, 1987). Their shape k is -xi in the
    # survival form used here, S(y) = (1 + xi y / sigma)^(-1/xi), for which
    # mean^2 / variance = 1 - 2 xi.
    ratio = y_bar**2 / s2
    xi = (1 - ratio) / 2
    sigma = y_bar * (ratio + 1) / 2

    if sigma <= 0 or not np.isfinite(xi) or not np.isfinite(sigma):
        return 0.0, 0.0, False

    return sigma, xi, True


def _gpd_sf(
    x: npt.NDArray[np.float64],
    sigma: float,
    xi: float,
) -> npt.NDArray[np.float64]:
    """
    GPD survival function P(Y > x).

    Parameters
    ----------
    x : ndarray
        Values at which to evaluate.
    sigma : float
        Scale parameter.
    xi : float
        Shape parameter.

    Returns
    -------
    ndarray
        Survival function values.
    """
    x = np.asarray(x, dtype=np.float64)

    return stats.genpareto.sf(x, c=xi, scale=sigma)


def _anderson_darling_gpd(
    exceedances: npt.NDArray[np.float64],
    sigma: float,
    xi: float,
) -> float:
    """
    Anderson-Darling test statistic for GPD fit.

    Lower values indicate better fit. A rough threshold of 2.5
    indicates acceptable fit (Choulakian & Stephens, 2001).

    Parameters
    ----------
    exceedances : sorted ndarray
        Exceedances in ascending order.
    sigma, xi : float
        GPD parameters.

    Returns
    -------
    float
        Anderson-Darling statistic.
    """
    n = len(exceedances)
    if n < 5:
        return np.inf

    # CDF values: F(y) = 1 - SF(y)
    F = 1.0 - _gpd_sf(exceedances, sigma, xi)
    F = np.clip(F, 1e-15, 1 - 1e-15)

    i = np.arange(1, n + 1, dtype=np.float64)
    A2 = -n - np.sum((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1]))) / n

    return A2


def fit_gpd_tail(
    null_dist: npt.NDArray[np.float64],
    observed: npt.NDArray[np.float64],
    n_thresholds: int = 5,
    ad_threshold: float = 2.5,
    min_exceedances: int = MIN_EXCEEDANCES,
) -> npt.NDArray[np.float64]:
    """
    Compute p-values with a GPD tail approximation where permutations are too few.

    Every statistic first gets its permutation p-value. Only statistics that
    fewer than ``min_exceedances`` null maxima reach — where that p-value
    rests on a handful of permutations — get the GPD tail value instead.
    This is a tail approximation, not an exact finite-permutation procedure.

    Algorithm (Winkler et al., 2016; Knijnenburg et al., 2009):

    1. Set the threshold ``u`` at the upper quartile of ``null_dist``.
    2. Fit a GPD to the exceedances ``y = T_star_j - u`` by method of moments.
    3. Test the fit with Anderson-Darling; if poor, raise ``u`` (up to the
       0.95 quantile) and retry.
    4. For the sparse-tail statistics, ``p = (k / J) * GPD_sf(T - u)`` where
       ``k`` is the number of exceedances.

    Parameters
    ----------
    null_dist : ndarray of shape (J,)
        Max-statistic null distribution from permutations.
    observed : ndarray of shape ``(*spatial_dims,)``
        Observed statistics (e.g., shape (N, N) for connectivity).
    n_thresholds : int, default=5
        Number of thresholds tried between the 0.75 and 0.95 quantiles.
    ad_threshold : float, default=2.5
        Anderson-Darling threshold for acceptable fit.
    min_exceedances : int, default=10
        The GPD replaces the permutation p-value only for statistics reached
        by fewer than this many null maxima.

    Returns
    -------
    p_values : ndarray of same shape as observed
        Permutation p-values (tie-inclusive, Phipson-Smyth +1), with GPD
        values in the sparse tail when a good fit is found.

    Notes
    -----
    Fitted values may be smaller than 1/J and are approximate. Observations
    below the fitted threshold retain empirical p-values. Fits whose finite
    upper endpoint excludes any candidate observation are rejected. Failed
    fits fall back to counting; the positive numerical floor guards underflow.
    """
    J = len(null_dist)
    if not isinstance(min_exceedances, (int, np.integer)) or min_exceedances < 1:
        raise ValueError("min_exceedances must be a positive integer.")
    if not isinstance(n_thresholds, (int, np.integer)) or n_thresholds < 1:
        raise ValueError("n_thresholds must be a positive integer.")
    null_sorted = np.sort(null_dist)
    p_empirical, count = _empirical_p(null_dist, observed)

    sparse = count < min_exceedances
    if not np.any(sparse):
        return p_empirical

    for q in np.linspace(0.75, 0.95, n_thresholds):
        u = np.quantile(null_sorted, q)
        exceedances = null_sorted[null_sorted > u] - u

        if len(exceedances) < 10:
            continue

        sigma, xi, valid = _fit_gpd_mom(exceedances)
        if not valid:
            continue

        # Check fit quality
        ad_stat = _anderson_darling_gpd(exceedances, sigma, xi)
        if not np.isfinite(ad_stat) or ad_stat > ad_threshold:
            continue

        # Good fit found: P(T > u) times the GPD survival above u
        k = len(exceedances)
        eligible = sparse & (observed >= u)
        if not np.any(eligible):
            continue
        if xi < 0 and np.any(observed[eligible] - u >= -sigma / xi):
            # A fitted finite endpoint is not evidence of an impossible
            # permutation statistic. Try another fit, otherwise use counting.
            logger.debug(
                "GPD bounded endpoint excludes observed statistics; rejecting fit"
            )
            continue
        tail_p = (k / J) * _gpd_sf(observed[eligible] - u, sigma, xi)

        logger.debug(
            f"GPD fit: u={u:.3f} (q={q:.2f}), k={k}, xi={xi:.4f}, "
            f"sigma={sigma:.4f}, AD={ad_stat:.3f}, sparse={int(sparse.sum())}"
        )
        return _replace_sparse_tail(
            p_empirical, count, tail_p, J, min_exceedances, replacement_mask=eligible
        )

    # No good fit found — fall back to empirical
    logger.debug("GPD fit failed at all thresholds, using empirical p-values")
    return p_empirical


def fit_gamma_tail(
    null_dist: npt.NDArray[np.float64],
    observed: npt.NDArray[np.float64],
    min_exceedances: int = MIN_EXCEEDANCES,
) -> npt.NDArray[np.float64]:
    """
    Compute p-values with a gamma (Pearson type III) tail where permutations are too few.

    Fits a gamma distribution to the null distribution by the method of
    moments (first three moments); a null that is not right-skewed cannot be
    fitted and keeps its permutation p-values. As in :func:`fit_gpd_tail`, the fitted
    value replaces the permutation p-value only for statistics reached by
    fewer than ``min_exceedances`` null maxima. Simpler than the GPD, but
    still an approximation to the finite-permutation tail.

    Parameters
    ----------
    null_dist : ndarray of shape (J,)
        Max-statistic null distribution.
    observed : ndarray of shape ``(*spatial_dims,)``
        Observed statistics.
    min_exceedances : int, default=10
        See :func:`fit_gpd_tail`.

    Returns
    -------
    p_values : ndarray of same shape as observed
        Permutation p-values (tie-inclusive, Phipson-Smyth +1), with gamma
        values in the sparse tail when the gamma can be fitted.
    """

    J = len(null_dist)
    if not isinstance(min_exceedances, (int, np.integer)) or min_exceedances < 1:
        raise ValueError("min_exceedances must be a positive integer.")
    p_empirical, count = _empirical_p(null_dist, observed)
    sparse = count < min_exceedances
    if not np.any(sparse):
        return p_empirical

    mean = np.mean(null_dist)
    var = np.var(null_dist, ddof=1)
    skew_val = float(stats.skew(null_dist))

    if (
        not np.isfinite(var)
        or not np.isfinite(skew_val)
        or var <= 0
        or skew_val < 1e-10
    ):
        # A gamma with positive scale is always right-skewed (skewness
        # 2 / sqrt(a) > 0): it cannot represent a symmetric or left-skewed
        # null, so keep the permutation p-values.
        return p_empirical

    # Gamma parameters from moments
    # shape a = 4/skew^2, scale b = sqrt(var / a), loc = mean - a*b
    a = 4.0 / (skew_val**2)
    b = np.sqrt(var / a)
    loc = mean - a * b

    tail_p = stats.gamma.sf(observed[sparse], a, loc=loc, scale=b)
    return _replace_sparse_tail(p_empirical, count, tail_p, J, min_exceedances)


def compute_p_values_accelerated(
    emp_stat_dict: Dict[str, npt.NDArray],
    max_null_dict: Dict[str, npt.NDArray],
    method: str = "gpd",
) -> Dict[str, npt.NDArray]:
    """
    Compute p-values using accelerated tail approximation.

    Drop-in replacement for ``_compute_p_values_from_null`` that uses
    GPD or gamma fitting instead of pure empirical counting. Tail
    estimates are approximate; fallback empirical p-values use the same
    tie-inclusive max-statistic counting as the standard path.

    Parameters
    ----------
    emp_stat_dict : dict
        Dictionary with empirical statistic arrays (any keys).
    max_null_dict : dict
        Dictionary with null distribution arrays.
    method : {'gpd', 'gamma'}, default='gpd'
        Acceleration method.

    Returns
    -------
    dict
        Dictionary with p-value arrays for each key.
    """
    fit_func = fit_gpd_tail if method == "gpd" else fit_gamma_tail
    p_values = {}

    from conninfpy.stats._engine import _check_finite_statistics

    for key in emp_stat_dict:
        emp = emp_stat_dict[key]
        null = max_null_dict[key]
        _check_finite_statistics(np.asarray(emp), np.asarray(null), key)

        if emp.ndim == 2:
            # Standard case: (N, N) vs (J,)
            p_values[key] = fit_func(null, emp)
        else:
            # Multi-param: (N, N, n_params) vs (J, n_params)
            n_params = emp.shape[-1]
            p_list = []
            for p_idx in range(n_params):
                p_slice = fit_func(null[:, p_idx], emp[..., p_idx])
                p_list.append(p_slice)
            p_values[key] = np.stack(p_list, axis=-1)

    return p_values
