"""Design diagnostics and network-level detection counts."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import numpy.typing as npt


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
        > 100 is strong concern)
      - ``vif``: VIF per column
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

    XtX = X.T @ X
    try:
        cond = float(np.linalg.cond(XtX))
    except np.linalg.LinAlgError:
        cond = np.inf

    vif = compute_vif(X)
    vif_argmax = int(np.argmax(vif))

    # Pairwise correlation (intercept → NaN row/col; report as 0)
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    flags = []
    if cond > 100:
        flags.append(
            f"Condition number {cond:.1f} > 100: strong multicollinearity — "
            f"coefficient estimates may be unstable."
        )
    elif cond > 30:
        flags.append(f"Condition number {cond:.1f} > 30: moderate multicollinearity.")
    if np.max(vif) > 10:
        flags.append(
            f"VIF for '{names[vif_argmax]}' = {vif[vif_argmax]:.1f} > 10: "
            f"strong collinearity."
        )
    elif np.max(vif) > 5:
        flags.append(
            f"VIF for '{names[vif_argmax]}' = {vif[vif_argmax]:.1f} > 5: "
            f"moderate collinearity."
        )

    return {
        "condition_number": cond,
        "vif": vif,
        "vif_max": float(vif[vif_argmax]),
        "vif_max_col": names[vif_argmax],
        "correlation": corr,
        "names": list(names),
        "flags": flags,
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
