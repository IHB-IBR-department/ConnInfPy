"""OLS design, contrast estimability and degrees-of-freedom primitives."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from conninfpy._prepare import check_finite_design

logger = logging.getLogger(__name__)


def _precompute_ols(
    X: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Precompute OLS quantities that are constant across permutations.

    Parameters
    ----------
    X : ndarray of shape (n_subjects, p)
        Design matrix (must include intercept if desired).

    Returns
    -------
    X_pinv : ndarray of shape (p, n_subjects)
        Moore-Penrose pseudoinverse of the design matrix.
    XtX_inv_diag : ndarray of shape (p,)
        Diagonal of the generalized inverse of X'X, needed for SE computation.
    XtX_inv : ndarray of shape (p, p)
        Generalized inverse of X'X, needed for contrast SE computation.
    """
    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        logger.warning(
            "Design matrix is rank deficient "
            f"(rank={rank}, columns={X.shape[1]}). "
            "Using Moore-Penrose pseudoinverse; non-estimable contrasts may "
            "still be unstable."
        )
    else:
        cond = np.linalg.cond(X)
        if cond > 1e12:
            logger.warning(
                f"Design matrix is ill-conditioned (cond={cond:.1e}). "
                f"Results may be numerically unstable."
            )

    X_pinv = np.linalg.pinv(X)
    XtX_inv = X_pinv @ X_pinv.T
    XtX_inv_diag = np.diag(XtX_inv)
    return X_pinv, XtX_inv_diag, XtX_inv


def contrast_is_estimable(
    contrast: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    X_pinv: Optional[npt.NDArray[np.float64]] = None,
) -> bool:
    """True when every contrast row is a combination of the rows of X (C = C X⁺X).

    Only then is Cβ the same for every least-squares solution β. With
    collinear columns (two copies of a variable, a group dummy nested in site
    dummies) a coefficient on one of them alone is not identified, and the
    pseudoinverse would report an arbitrary split of their joint effect.
    """
    C = np.atleast_2d(np.asarray(contrast, dtype=np.float64))
    if X_pinv is None:
        X_pinv = np.linalg.pinv(X)
    return bool(np.allclose(C @ X_pinv @ X, C, atol=1e-8 * max(1.0, np.abs(C).max())))


def _check_estimable(
    contrast: npt.NDArray[np.float64],
    X: npt.NDArray[np.float64],
    name: str = "Contrast",
) -> None:
    if not contrast_is_estimable(contrast, X):
        raise ValueError(
            f"{name} is not estimable: it is not a combination of the rows of the "
            "design matrix (C != C X⁺X), so its value depends on an arbitrary choice "
            "among equally good fits. This happens with collinear columns (e.g. two "
            "copies of one variable, or a group indicator that is a sum of site "
            "dummies). Drop the redundant column or test a combination that is "
            "identified (e.g. the sum of the collinear coefficients)."
        )


def _residual_degrees_of_freedom(X: npt.NDArray[np.float64]) -> int:
    """Return residual df based on design rank, not raw column count."""
    df = X.shape[0] - int(np.linalg.matrix_rank(X))
    if df <= 0:
        raise ValueError(
            "Design matrix has no residual degrees of freedom "
            f"(n={X.shape[0]}, rank={np.linalg.matrix_rank(X)})."
        )
    return df


def build_design_matrix(
    interest: npt.NDArray[np.float64],
    confounds: Optional[npt.NDArray[np.float64]] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Build design matrix and contrast vector from interest and confound variables.

    Constructs X = [intercept, confounds, interest] and contrast = [0, ..., 0, 1]
    targeting the last column (interest variable).

    Parameters
    ----------
    interest : ndarray of shape (n_subjects,)
        Variable of interest (continuous or binary).
    confounds : ndarray of shape (n_subjects,) or (n_subjects, q), optional
        Confound variables. If 1D, treated as single confound.

    Returns
    -------
    X : ndarray of shape (n_subjects, p)
        Design matrix with intercept, confounds, and interest.
    contrast : ndarray of shape (p,)
        Contrast vector [0, ..., 0, 1] targeting the interest column.
    """
    n = len(interest)
    interest = check_finite_design(interest, "interest")
    if interest.ndim == 1:
        interest = interest[:, np.newaxis]

    intercept = np.ones((n, 1), dtype=np.float64)

    if confounds is not None:
        confounds = check_finite_design(confounds, "confounds")
        if confounds.ndim == 1:
            confounds = confounds[:, np.newaxis]
        X = np.hstack([intercept, confounds, interest])
    else:
        X = np.hstack([intercept, interest])

    contrast = np.zeros(X.shape[1], dtype=np.float64)
    contrast[-1] = 1.0

    return X, contrast
