"""Monte-Carlo precision metadata: permutation space, empirical exceedance
counts, conditional binomial (Wilson) intervals and RNG provenance.

Why this module exists
----------------------
A permutation run reports ``p = (count + 1) / (B + 1)`` (Phipson & Smyth
2010), where ``count`` is the number of i.i.d. draws whose max statistic
reaches the observed one. Three different quantities are easy to confuse:

1. ``count / B`` — the empirical exceedance rate of the draws; this is the
   conditional exceedance probability a binomial interval can describe, and
   it is *not* the reported p-value;
2. ``(count + 1) / (B + 1)`` — the reported, slightly conservative p-value;
3. ``1 / (B + 1)`` — the numerical resolution, i.e. the smallest attainable
   p-value. It is a grid spacing, not a bound on the true exceedance
   probability and not an error bar.

With ``B`` draws the reported p-value can only take ``B + 1`` values, so two
p-values within about ``1 / (B + 1)`` of each other (or of a threshold such
as alpha) cannot be separated by the run. The Wilson interval implemented
here describes the conditional exceedance probability of a single draw; it
is not family-wise or FDR error control, and it is never an interval for a
BH-adjusted q value or for an accelerated (GPD/gamma) p-value.

Sampling scheme and RNG provenance
----------------------------------
The current pipelines draw i.i.d. permutations/seeds *with replacement*:
one ``np.random.RandomState(seed)`` per permutation, with
``seeds = randint(0, 2**32 - 1, size=B)``. Duplicate permutations are
therefore possible, and ``B`` larger than the theoretical permutation space
is *not* a reason to suppress an interval (it is reported as an
informational flag only). The recorded provenance describes exactly this
scheme; nothing here changes an RNG stream.

Count kinds
-----------
Every count stored by the pipelines is a sum of 0/1 indicators, one per
draw, which is what makes the binomial reading legitimate:

* ``max_stat`` — independent one-sided max-statistic null; one indicator
  per draw and tail.
* ``joint_max`` — two-sided run: the null is ``max(positive, negative)``, a
  single number per draw. Both tails count against that same joint null, so
  their counts (and intervals) are not independent; each tail still counts
  against its own observed statistic, so the count arrays differ.
* ``edge_wise`` — per-edge directional exceedance count (``bh_fdr_perm``);
  one indicator per draw, edge and tail.
* ``two_sided_sum_stat`` — ``bh_fdr_perm`` two-sided: the per-draw
  indicator is ``(t+ + t- >= observed_pos + observed_neg)``, again a single
  0/1 per edge, shared by both tails.

Intervals are emitted only for purely empirical counts from i.i.d. draws.
Any acceleration (GPD/gamma) replaces the sparse tail by a data-dependent
choice (``count < min_exceedances``), which invalidates the conditional
binomial reading for every cell of that result, even the surviving
empirical ones; such results report interval status ``unknown`` with that
reason. The same holds for exhaustive enumeration, non-i.i.d./unknown
schemes, block-level (cNBS) inference and available-case analyses with
per-edge sample sizes.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

__all__ = [
    "COUNT_KIND_EDGE_WISE",
    "COUNT_KIND_JOINT_MAX",
    "COUNT_KIND_MAX_STAT",
    "COUNT_KIND_TWO_SIDED_SUM_STAT",
    "COUNT_UNAVAILABLE",
    "INTERVAL_AVAILABLE",
    "INTERVAL_UNKNOWN",
    "build_report",
    "interval_status",
    "mark_untested",
    "permutation_space",
    "report_summary",
    "rng_provenance",
    "wilson_interval",
]

# ---------------------------------------------------------------------------
# RNG provenance (describes CURRENT behaviour; never changes it)
# ---------------------------------------------------------------------------

#: Scheme used by every pipeline today: a legacy ``RandomState`` (MT19937)
#: draws one uint32 seed per permutation, and each permutation restarts its
#: generator from that seed.
RNG_SCHEME = "legacy-randomstate-randint-uint32"
RNG_BIT_GENERATOR = "MT19937"
RNG_SCHEMA_VERSION = 1


def rng_provenance(seed: Optional[int]) -> Dict[str, Any]:
    """Describe the RNG scheme behind one permutation run.

    ``seed`` is the *resolved* integer seed used at the existing
    :func:`conninfpy._shared.rng.resolve_seed` call site, or ``None`` when the
    run is non-deterministic (``RandomState(None)``). Recording it draws no
    random numbers, so no stream shifts.
    """
    return {
        "seed": None if seed is None else int(seed),
        "scheme": RNG_SCHEME,
        "bit_generator": RNG_BIT_GENERATOR,
        "schema_version": RNG_SCHEMA_VERSION,
    }


# ---------------------------------------------------------------------------
# Theoretical permutation space
# ---------------------------------------------------------------------------


def permutation_space(
    test_type: Any,
    n1: int,
    n2: Optional[int] = None,
    strata: Optional[npt.ArrayLike] = None,
) -> Optional[int]:
    """Number of admissible relabellings/transformations, or ``None``.

    This is the size of the assignment space, never the number of distinct
    *statistic values* (many assignments share a statistic):

    * paired / one-sample sign flip: ``2 ** n`` — strata do not restrict sign
      flips, and an edge mask restricts edges, not assignments;
    * two-sample group shuffle: ``comb(n1 + n2, n1)``, or, with strata, the
      product over strata of ``comb(n_s, n1_s)`` (per-stratum group totals
      stay fixed).

    ``None`` means unknown/ambiguous (Freedman-Lane residual exchangeability,
    available-case per-edge sample sizes, block-level inference). The value is
    exact Python integer arithmetic.
    """
    test = getattr(test_type, "value", test_type)
    if test in ("paired", "one-sample"):
        n = int(n1)
        if n < 0:
            return None
        return int(1 << n)
    if test == "two-sample":
        if n2 is None:
            return None
        n1_i, n2_i = int(n1), int(n2)
        if n1_i < 0 or n2_i < 0:
            return None
        if strata is None:
            return int(math.comb(n1_i + n2_i, n1_i))
        labels = np.asarray(strata)
        if labels.shape[0] != n1_i + n2_i:
            return None
        _, codes = np.unique(labels, return_inverse=True)
        total = 1
        for code in np.unique(codes):
            in_stratum = codes == code
            n_s = int(in_stratum.sum())
            n1_s = int(in_stratum[:n1_i].sum())
            if not 0 <= n1_s <= n_s:
                return None
            total *= math.comb(n_s, n1_s)
        return int(total)
    return None


# ---------------------------------------------------------------------------
# Exceedance counts
# ---------------------------------------------------------------------------

#: Stored where no genuine exceedance count exists for a cell (diagonal,
#: outside a declared family mask, NaN p-value, a replaced accelerated tail,
#: or a cell a tail does not use). Never a fabricated 0.
COUNT_UNAVAILABLE = -1


def mark_untested(
    counts: npt.ArrayLike,
    p_map: npt.ArrayLike,
    *,
    unavailable: Optional[npt.ArrayLike] = None,
    n_draws: Optional[int] = None,
) -> npt.NDArray[np.int64]:
    """Copy ``counts`` and mark untested cells with :data:`COUNT_UNAVAILABLE`.

    Untested means: the p-value is not finite (edge outside a declared family
    mask, available-case edge without coverage), the cell is the diagonal, the
    caller says the count is not the basis of the p-value (``unavailable``,
    e.g. an accelerated tail replacement), or the count exceeds the number of
    draws (a 0/1 indicator sum cannot). Genuine counts are kept unchanged.
    """
    out = np.array(counts, dtype=np.int64, copy=True)
    p = np.asarray(p_map, dtype=np.float64)
    if out.shape != p.shape:
        raise ValueError(
            f"exceedance counts shape {out.shape} does not match p shape {p.shape}"
        )
    invalid = ~np.isfinite(p)
    if out.ndim >= 2 and out.shape[0] == out.shape[1]:
        diag = np.eye(out.shape[0], dtype=bool)
        invalid = invalid | (diag[:, :, None] if out.ndim >= 3 else diag)
    if unavailable is not None:
        invalid = invalid | np.asarray(unavailable, dtype=bool)
    if n_draws is not None:
        invalid = invalid | (out > int(n_draws))
    out[invalid] = COUNT_UNAVAILABLE
    return out


# ---------------------------------------------------------------------------
# Conditional binomial (Wilson) interval
# ---------------------------------------------------------------------------

INTERVAL_AVAILABLE = "available"
INTERVAL_UNKNOWN = "unknown"

INTERVAL_TARGET = (
    "conditional exceedance probability P(T* >= T_obs | H0) = count/B of a "
    "single draw; not the family-wise/FDR error rate"
)
REPORTED_ESTIMATE = "(count + 1) / (B + 1)"
EMPIRICAL_ESTIMATE = "count / B"
RESOLUTION_DESCRIPTION = "1 / (B + 1) (smallest attainable p-value; a grid step, not an error bar)"

COUNT_KIND_NOTES: Dict[str, str] = {
    "max_stat": (
        "one 0/1 exceedance indicator per i.i.d. draw and tail "
        "(one-sided max-statistic null)"
    ),
    "joint_max": (
        "two-sided run: the null is max(positive, negative), a single number "
        "per draw; both tails count against this same joint null (each against "
        "its own observed statistic), so their intervals are not independent"
    ),
    "edge_wise": (
        "per-edge directional exceedance count: one 0/1 indicator per draw, "
        "edge and tail"
    ),
    "two_sided_sum_stat": (
        "two-sided per-edge count with the single indicator "
        "(t+ + t- >= observed_pos + observed_neg) per draw; both tails share "
        "one count array"
    ),
}

COUNT_KIND_MAX_STAT = "max_stat"
COUNT_KIND_JOINT_MAX = "joint_max"
COUNT_KIND_EDGE_WISE = "edge_wise"
COUNT_KIND_TWO_SIDED_SUM_STAT = "two_sided_sum_stat"


def wilson_interval(
    count: npt.ArrayLike,
    n_draws: int,
    alpha: float = 0.05,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Score (Wilson) interval for a binomial proportion, vectorized.

    ``count`` are 0/1 indicator sums over ``n_draws`` independent draws, so
    ``count ~ Binomial(n_draws, p)``; the returned bounds are for ``p``, the
    *conditional* exceedance probability, and are computed at confidence
    ``1 - alpha``. No new dependency: the normal quantile comes from
    :class:`statistics.NormalDist`.
    """
    b = int(n_draws)
    if b <= 0:
        raise ValueError("n_draws must be a positive integer for a Wilson interval.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}.")
    counts = np.asarray(count, dtype=np.float64)
    if np.any(counts < 0) or np.any(counts > b):
        raise ValueError("counts must be 0/1 indicator sums in [0, n_draws].")
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    p_hat = counts / b
    denom = 1.0 + z * z / b
    center = (p_hat + z * z / (2.0 * b)) / denom
    half = (z / denom) * np.sqrt(p_hat * (1.0 - p_hat) / b + z * z / (4.0 * b * b))
    low = np.clip(center - half, 0.0, 1.0)
    high = np.clip(center + half, 0.0, 1.0)
    return low, high


def interval_status(
    *,
    n_permutations: int = 0,
    counts_available: bool = False,
    method: Optional[str] = None,
    acceleration: Optional[str] = None,
    available_case: bool = False,
    iid_with_replacement: bool = True,
    count_kind: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Whether a conditional binomial interval is justified, and why not.

    Returns ``('available', None)`` only for a purely empirical count from
    i.i.d. draws whose count kind is a sum of 0/1 indicators. Every other
    case returns ``('unknown', reason)`` with the actual reason.
    """
    if int(n_permutations) <= 0:
        return INTERVAL_UNKNOWN, "no permutations (analytic p-values)"
    if method == "cnbs":
        return INTERVAL_UNKNOWN, "block-level inference (cNBS) has no per-edge exceedance counts"
    if acceleration is not None:
        return (
            INTERVAL_UNKNOWN,
            f"data-dependent tail replacement (acceleration={acceleration})",
        )
    if available_case:
        return (
            INTERVAL_UNKNOWN,
            "available-case analysis: per-edge sample sizes, exchangeability ambiguous",
        )
    if not iid_with_replacement:
        return INTERVAL_UNKNOWN, "sampling scheme is not i.i.d. draws with replacement"
    if not counts_available:
        return INTERVAL_UNKNOWN, "exceedance counts were not recorded for this result"
    if count_kind not in COUNT_KIND_NOTES:
        return INTERVAL_UNKNOWN, "exceedance count kind is not a 0/1 indicator sum"
    return INTERVAL_AVAILABLE, None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _project(arr: Any, param_idx: Optional[int], what: str) -> Any:
    """Return a 2D map, a whole (N, N, K) grid, or one grid cell."""
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim == 3:
        if param_idx is None:
            return a
        if not 0 <= int(param_idx) < a.shape[-1]:
            raise ValueError(f"param_idx={param_idx} is out of range for {what}.")
        return a[:, :, int(param_idx)]
    if a.ndim == 2 and param_idx not in (None, 0):
        raise ValueError(f"param_idx is only valid for a parameter grid; {what} is 2D.")
    return a


def build_report(
    result: Any,
    *,
    alpha: float = 0.05,
    param_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble the Monte-Carlo precision report for one result.

    Pure description: no inference is re-run and nothing is reconstructed
    from p-values. Counts come from the producer; missing pieces are reported
    as unavailable/unknown instead of being invented.
    """
    n_perm = int(getattr(result, "n_permutations", 0) or 0)
    acceleration = getattr(result, "acceleration", None)
    method = getattr(result, "method", None)
    space = getattr(result, "permutation_space", None)
    rng = getattr(result, "rng_provenance", None)
    count_kind = getattr(result, "count_kind", None)
    stored_counts = getattr(result, "exceedance_counts", None) or {}
    missing_report = getattr(result, "missing_report", None)

    tails: Sequence[str]
    if "omnibus" in result:
        tails = ["omnibus"]
    else:
        tails = [tail for tail in ("positive", "negative") if tail in result]

    resolution = 1.0 / (n_perm + 1) if n_perm > 0 else None
    flags = []
    notes = []
    if space is not None and n_perm > space:
        flags.append("draws-exceed-permutation-space")
        notes.append(
            "B exceeds the theoretical permutation space; draws are i.i.d. with "
            "replacement, so duplicate permutations are expected. Informational: "
            "it does not invalidate the intervals."
        )
    if acceleration is not None:
        notes.append(
            "Accelerated (GPD/gamma) p-values are data-dependent tail estimates; "
            "no Wilson interval applies to any cell of this result, including "
            "cells whose p-value remained empirical."
        )
    if method in ("bh_fdr", "bh_fdr_perm"):
        notes.append(
            "Reported p-values are BH-adjusted q values; the interval applies only "
            "to the underlying per-edge empirical exceedance count, never to q."
        )
    if method == "cnbs":
        notes.append(
            "cNBS infers at the block level; edge cells carry block p-values and no "
            "per-edge exceedance count."
        )
    if missing_report:
        notes.append(
            "Available-case analysis: per-edge sample sizes make exchangeability "
            "and a single permutation space ambiguous."
        )
    if count_kind in (COUNT_KIND_JOINT_MAX, COUNT_KIND_TWO_SIDED_SUM_STAT):
        if count_kind == COUNT_KIND_JOINT_MAX:
            notes.append(
                "Two-sided run: both tails count against the same joint null "
                "max(positive, negative) (one 0/1 indicator per draw), so their "
                "counts and intervals are not independent; each tail counts "
                "against its own observed statistic."
            )
        else:
            notes.append(
                "Two-sided run: both tails use one shared per-edge count array "
                "(single 0/1 indicator (t+ + t- >= observed_pos + observed_neg) "
                "per draw), so their counts and intervals are not independent."
            )

    if space is None:
        if missing_report:
            space_reason = (
                "per-edge subject sets differ (available-case); "
                "admissible-assignment cardinality ambiguous"
            )
        else:
            space_reason = (
                "admissible-assignment cardinality is ambiguous for this "
                "permutation scheme (e.g. Freedman-Lane residual exchangeability)"
            )
    else:
        space_reason = None

    report: Dict[str, Any] = {
        "alpha": float(alpha),
        "n_permutations": n_perm,
        "resolution": resolution,
        "resolution_description": RESOLUTION_DESCRIPTION,
        "reported_estimate": REPORTED_ESTIMATE,
        "empirical_estimate": EMPIRICAL_ESTIMATE,
        "interval_target": INTERVAL_TARGET,
        "interval_confidence": 1.0 - float(alpha) if n_perm > 0 else None,
        "sampling": "iid_draws_with_replacement",
        "duplicate_draws_possible": None if space is None else bool(n_perm > space),
        "permutation_space": None if space is None else int(space),
        "permutation_space_known": space is not None,
        "permutation_space_reason": space_reason,
        "count_kind": count_kind,
        "count_kind_note": COUNT_KIND_NOTES.get(count_kind) if count_kind is not None else None,
        "counts_unavailable_sentinel": COUNT_UNAVAILABLE,
        "rng": dict(rng) if isinstance(rng, Mapping) else None,
        "flags": flags,
        "notes": notes,
        "tails": {},
    }

    for tail in tails:
        p_2d = _project(result[tail], param_idx, f"{tail} p map")
        raw_counts = stored_counts.get(tail) if isinstance(stored_counts, Mapping) else None
        counts_2d = (
            _project(raw_counts, param_idx, f"{tail} exceedance counts")
            if raw_counts is not None
            else None
        )
        counts_available = counts_2d is not None and np.any(
            np.asarray(counts_2d) >= 0
        )
        status, reason = interval_status(
            n_permutations=n_perm,
            counts_available=bool(counts_available),
            method=method,
            acceleration=acceleration,
            available_case=bool(missing_report),
            count_kind=count_kind,
        )
        entry: Dict[str, Any] = {
            "counts_available": bool(counts_available),
            "count": None,
            "tested_mask": None,
            "interval_status": status,
            "interval_reason": reason,
            "interval_target": INTERVAL_TARGET,
            "interval_low": None,
            "interval_high": None,
            "n_cells_counted": 0,
            "n_cells_interval": 0,
        }
        if counts_2d is not None:
            stored = np.asarray(counts_2d, dtype=np.int64)
            tested = stored >= 0
            entry["count"] = stored
            entry["tested_mask"] = tested
            entry["n_cells_counted"] = int(tested.sum())
        if status == INTERVAL_AVAILABLE and counts_2d is not None:
            tested = np.asarray(counts_2d) >= 0
            low = np.full(p_2d.shape, np.nan, dtype=np.float64)
            high = np.full(p_2d.shape, np.nan, dtype=np.float64)
            if np.any(tested):
                low[tested], high[tested] = wilson_interval(
                    np.asarray(counts_2d)[tested], n_perm, alpha
                )
            entry["interval_low"] = low
            entry["interval_high"] = high
            entry["n_cells_interval"] = int(tested.sum())
        report["tails"][tail] = entry
    return report


def report_summary(report: Mapping[str, Any]) -> Dict[str, Any]:
    """JSON-safe **run-level aggregate** summary of :func:`build_report`.

    No arrays and no per-edge values: this is the shape written into
    ``provenance.json``, where every number here summarizes the whole run.
    The per-edge quantities (B, the count of each edge, the +1 relation
    ``p = (count + 1) / (B + 1)``) live in ``edges.csv`` and in the bundle's
    ``metadata['mc']['exceedance_counts']``; nothing per-edge is recoverable
    from this block, and nothing here is a per-edge claim.
    """
    summary: Dict[str, Any] = {
        "level": "run",
        "aggregate": True,
        "per_edge_source": (
            "edges.csv (per-edge p) and the result bundle "
            "(result_bundle.zip; legacy CLI filename bundle*.pkl) "
            "metadata['mc']['exceedance_counts'] (per-edge B and counts)"
        ),
        "p_relation": "(count + 1) / (B + 1) on counted cells",
        "n_permutations": report.get("n_permutations"),
        "resolution": report.get("resolution"),
        "permutation_space": report.get("permutation_space"),
        "permutation_space_reason": report.get("permutation_space_reason"),
        "rng": report.get("rng"),
        "count_kind": report.get("count_kind"),
        "sampling": report.get("sampling"),
        "reported_estimate": report.get("reported_estimate"),
        "interval_target": report.get("interval_target"),
        "interval_confidence": report.get("interval_confidence"),
        "flags": list(report.get("flags") or []),
        "tails": {},
    }
    for tail, entry in (report.get("tails") or {}).items():
        count = entry.get("count")
        intervals = entry.get("interval_low")
        valid = None
        if count is not None:
            kept = np.asarray(count)[np.asarray(count) >= 0]
            valid = kept if kept.size else None
        summary["tails"][tail] = {
            "counts_available": bool(entry.get("counts_available")),
            "n_cells_counted": int(entry.get("n_cells_counted") or 0),
            # Aggregate over the counted cells of this tail; per-edge values
            # are in edges.csv / the bundle, not here.
            "count_min_aggregate": None if valid is None else int(valid.min()),
            "count_max_aggregate": None if valid is None else int(valid.max()),
            "interval_status": entry.get("interval_status"),
            "interval_reason": entry.get("interval_reason"),
            "n_cells_interval": int(entry.get("n_cells_interval") or 0),
            "interval_low_min_aggregate": None
            if intervals is None or not np.any(np.isfinite(intervals))
            else float(np.nanmin(intervals)),
            "interval_high_max_aggregate": None
            if entry.get("interval_high") is None
            or not np.any(np.isfinite(entry["interval_high"]))
            else float(np.nanmax(entry["interval_high"])),
        }
    return summary
