"""Inference methods, test types and parameter-grid metadata."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt


class TestType(str, Enum):
    """Statistical test types for permutation testing."""

    PAIRED = "paired"
    """Paired samples t-test (within-subjects design)."""

    ONE_SAMPLE = "one-sample"
    """One-sample t-test against zero."""

    TWO_SAMPLE = "two-sample"
    """Independent samples t-test (between-subjects design)."""


class StatMethod(str, Enum):
    """Statistical method for network analysis."""

    TFNBS = "tfnbs"
    """Threshold-Free Network-Based Statistics (TFCE-style)."""

    TSTAT = "tstat"
    """Raw t-statistics without enhancement."""

    NBS = "nbs"
    """Classical Network-Based Statistics with fixed threshold."""

    CNBS = "cnbs"
    """Constrained NBS with predefined network partitions."""

    NI_TFNBS = "ni_tfnbs"
    """Network-Informed TFNBS with functional block density weighting."""

    FBC_TFNBS = "fbc_tfnbs"
    """Functional Block Clustering TFNBS (block-defined clustering)."""

    BONFERRONI = "bonferroni"
    """Parametric Bonferroni correction (no permutation testing)."""

    BH_FDR = "bh_fdr"
    """Parametric Benjamini-Hochberg FDR correction (no permutation testing)."""

    BH_FDR_PERM = "bh_fdr_perm"
    """Permutation-based BH-FDR correction (per-edge permutation p-values)."""


def _grid_from_kwargs(
    method_enum: "StatMethod",
    enhance_kwargs: Dict[str, Any],
) -> Tuple[Optional[npt.NDArray[np.float64]], Optional[npt.NDArray[np.float64]]]:
    """Return ``(e_grid, h_grid)`` arrays iff ``e``/``h`` were arrays.

    Used to label the parameter axis of multi-(E, H) :class:`InferenceResult`
    p-maps. Returns ``(None, None)`` for scalar (E, H) calls or
    non-TFNBS-family methods.
    """
    if method_enum not in TFNBS_FAMILY:
        return None, None
    e = enhance_kwargs.get("e")
    h = enhance_kwargs.get("h")
    if e is None or h is None:
        return None, None
    if np.isscalar(e) and np.isscalar(h):
        return None, None
    return (
        np.asarray(e, dtype=np.float64),
        np.asarray(h, dtype=np.float64),
    )


class GLMStatType(str, Enum):
    """Statistic type for GLM inference."""

    TSTAT = "tstat"
    """t-statistic: beta / SE(beta). Scale-invariant, start_thres=1.65 valid."""

    BETA = "beta"
    """Raw regression coefficient. Interpretable, needs adapted threshold."""

    FSTAT = "fstat"
    """F-statistic for an omnibus (multi-row) contrast.

    Non-negative by construction — the enhancement pipeline sees a single
    ``'omnibus'`` key instead of a ``positive``/``negative`` split. For a
    single-row contrast, F == t² (same rejection regions, same rankings)."""


CONSTRAINED_METHODS = {StatMethod.CNBS, StatMethod.NI_TFNBS, StatMethod.FBC_TFNBS}
PARAMETRIC_METHODS = {StatMethod.BONFERRONI, StatMethod.BH_FDR}
TFNBS_FAMILY = {StatMethod.TFNBS, StatMethod.NI_TFNBS, StatMethod.FBC_TFNBS}
