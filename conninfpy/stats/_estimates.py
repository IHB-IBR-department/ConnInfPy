"""Coefficient (beta) maps saved with every result, and baseline coupling from them.

Every inference result keeps the coefficients of the model it tested, on the
response scale (``beta``, shape (p, N, N)), with a ``design`` description:
column names and means, the contrast, and named covariate profiles. Any
predicted connectivity is ``x' beta`` at a profile ``x``.

Baselines come from predictions, never from the intercept map alone: with
uncentred covariates and reference-coded sites, the intercept is the
connectivity at age 0, sex 0 and the reference site. At the column means the
prediction equals the sample mean connectivity (ordinary least squares), and
predictions inside the design's row space are estimable even when an
individual coefficient is not.

Pairwise tests use the equivalent designs:

* two-sample: columns [group-1 mean, group 2 - group 1];
* paired: [condition-A mean, B - A];
* one-sample: [mean].
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

COUPLING_CLASSES = (
    "stronger positive coupling",
    "weaker positive coupling",
    "stronger anticorrelation",
    "weaker anticorrelation",
    "sign change: positive to negative",
    "sign change: negative to positive",
    "emerges from ~0",
    "returns to ~0",
    "no change",
)


def glm_design(
    X: npt.NDArray[np.float64],
    contrast: npt.NDArray[np.float64],
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Design description for a GLM result (JSON-serialisable)."""
    X = np.asarray(X, dtype=np.float64)
    p = X.shape[1]
    names = (
        list(columns)
        if columns is not None
        else (
            ["intercept"] + [f"x{k}" for k in range(1, p)]
            if np.allclose(X[:, 0], 1.0)
            else [f"x{k}" for k in range(p)]
        )
    )
    if len(names) != p:
        raise ValueError(f"design columns must name all {p} columns; got {len(names)}.")
    means = X.mean(axis=0)
    C = np.atleast_2d(np.asarray(contrast, dtype=np.float64))
    from conninfpy.stats._design import contrast_is_estimable

    design: Dict[str, Any] = {
        "kind": "glm",
        "columns": names,
        "column_means": means.tolist(),
        "contrast": np.asarray(contrast, dtype=np.float64).tolist(),
        "n_subjects": int(X.shape[0]),
        "estimable_columns": [
            bool(contrast_is_estimable(np.eye(p)[k], X)) for k in range(p)
        ],
        "profiles": {"mean": means.tolist()},
        "comparison": None,
        "interest": None,
        "scale": "unknown",
    }
    nz = np.flatnonzero(np.any(C != 0, axis=0))
    if C.shape[0] == 1 and nz.size == 1:  # a selection contrast: one tested column
        k = int(nz[0])
        x = X[:, k]
        values = np.unique(x)
        if values.size == 2:
            lo, hi = values
            kind, labels = "binary", (f"{names[k]}={lo:g}", f"{names[k]}={hi:g}")
        else:
            m, s = float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else 0.0
            lo, hi = max(m - s, float(x.min())), min(m + s, float(x.max()))
            kind, labels = (
                "continuous",
                (f"{names[k]}={lo:.4g} (mean-1SD)", f"{names[k]}={hi:.4g} (mean+1SD)"),
            )
        for value, label in zip((lo, hi), labels):
            prof = means.copy()
            prof[k] = value
            design["profiles"][label] = prof.tolist()
        design["comparison"] = list(labels)  # low -> high value of the tested column
        design["interest"] = {"column": names[k], "kind": kind}
    return design


def pairwise_estimates(
    group1: npt.NDArray[np.float64],
    group2: Optional[npt.NDArray[np.float64]],
    test_type: str,
) -> Tuple[npt.NDArray[np.float64], Dict[str, Any]]:
    """Coefficient maps and design of the design equivalent to a pairwise test (NaN-aware)."""
    with np.errstate(invalid="ignore"), _quiet():
        if test_type == "one-sample":
            beta = np.nanmean(group1, axis=0)[None]
            design = {
                "kind": "one-sample",
                "columns": ["mean"],
                "contrast": [1.0],
                "profiles": {"mean": [1.0]},
                "comparison": None,
            }
        elif test_type == "two-sample":
            m1, m2 = np.nanmean(group1, axis=0), np.nanmean(group2, axis=0)
            beta = np.stack([m1, m2 - m1])
            design = {
                "kind": "two-sample",
                "columns": ["group1_mean", "group2_minus_group1"],
                "contrast": [0.0, 1.0],
                "profiles": {"group1": [1.0, 0.0], "group2": [1.0, 1.0]},
                "comparison": ["group1", "group2"],
            }
        elif test_type == "paired":
            beta = np.stack(
                [np.nanmean(group1, axis=0), np.nanmean(group2 - group1, axis=0)]
            )
            design = {
                "kind": "paired",
                "columns": ["condition_A_mean", "B_minus_A"],
                "contrast": [0.0, 1.0],
                "profiles": {"A": [1.0, 0.0], "B": [1.0, 1.0]},
                "comparison": ["A", "B"],
            }
        else:
            raise ValueError(f"unknown test_type {test_type!r}")
    design.update(
        {
            "n_subjects": int(
                group1.shape[0]
                + (0 if group2 is None or test_type == "paired" else group2.shape[0])
            ),
            "scale": "unknown",
            "interest": None,
        }
    )
    return beta, design


class _quiet:
    """Silence 'mean of empty slice' for edges without data (they stay NaN)."""

    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


def predict(
    beta: npt.NDArray[np.float64], x: Sequence[float]
) -> npt.NDArray[np.float64]:
    """Predicted connectivity ``x' beta`` (N, N) at a covariate profile ``x``."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (beta.shape[0],):
        raise ValueError(f"profile must have {beta.shape[0]} values; got {x.shape}.")
    return np.tensordot(x, beta, axes=(0, 0))


def effect(
    beta: npt.NDArray[np.float64], contrast: Sequence[float]
) -> Optional[npt.NDArray[np.float64]]:
    """Contrast effect ``c' beta`` (N, N) for a one-row contrast, else None."""
    c = np.asarray(contrast, dtype=np.float64)
    if c.ndim != 1:
        return None
    return np.tensordot(c, beta, axes=(0, 0))


def coupling_change(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    *,
    scale: str = "fisher_z",
    near_zero: float = 0.0,
) -> Tuple[npt.NDArray[np.object_], npt.NDArray[np.float64]]:
    """Classify the change of coupling from ``a`` to ``b`` and its size in r units.

    ``a`` and ``b`` are predicted connectivity on the analysis scale. With
    ``scale='fisher_z'`` they are converted to correlations (tanh) and the
    size is Δr = r_b - r_a; otherwise the classes use the values as they are
    and Δ is in their units. ``near_zero`` (prespecified, in r or native
    units) defines "~0" for the emerges / returns classes; 0 uses literal
    signs. Classes: see :data:`COUPLING_CLASSES`. NaN where either is NaN.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra, rb = (np.tanh(a), np.tanh(b)) if scale == "fisher_z" else (a, b)
    out = np.full(ra.shape, "", dtype=object)
    with np.errstate(invalid="ignore"):
        za, zb = np.abs(ra) <= near_zero, np.abs(rb) <= near_zero
        both = ~(za | zb)
        same = both & (np.sign(ra) == np.sign(rb))
        stronger = np.abs(rb) > np.abs(ra)
        out[same & (ra > 0) & stronger] = "stronger positive coupling"
        out[same & (ra > 0) & ~stronger] = "weaker positive coupling"
        out[same & (ra < 0) & stronger] = "stronger anticorrelation"
        out[same & (ra < 0) & ~stronger] = "weaker anticorrelation"
        out[both & (ra > 0) & (rb < 0)] = "sign change: positive to negative"
        out[both & (ra < 0) & (rb > 0)] = "sign change: negative to positive"
        out[za & ~zb] = "emerges from ~0"
        out[~za & zb] = "returns to ~0"
        out[za & zb] = "no change"
        out[(ra == rb) & both] = "no change"
    nan = np.isnan(ra) | np.isnan(rb)
    out[nan] = None
    return out, rb - ra


__all__ = [
    "glm_design",
    "pairwise_estimates",
    "predict",
    "effect",
    "coupling_change",
    "COUPLING_CLASSES",
]
