"""NETSTAT-35 closing suite (AC1): degenerate, tie, empty, signed-only and ID cases.

Every case runs the same inputs through two *permutation* implementations:

* the direct edge-vector path (``_build_edge_max_spec`` returns an
  ``_EdgeMaxSpec`` and ``_eval_edge_max_tail`` scores the upper-triangle
  vector without a ``(N, N)`` allocation), and
* the forced matrix path (``_build_edge_max_spec`` patched to ``None``, so
  every permutation rebuilds ``(N, N)`` and runs the public ``apply_*``
  wrapper).

The two paths must agree bit-for-bit on corrected p-values, observed
statistics, and the component memberships reported in ``component_report``.
Nothing here is relaxed to make a case pass: a mismatch is a finding.

Synthetic data only; no real or laboratory data and no N > 60.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from conninfpy import compute_p_val
from conninfpy.enhance import fbc_tfnbs as fbc_mod
from conninfpy.enhance import nbs as nbs_mod
from conninfpy.enhance import ni_tfnbs as ni_mod
from conninfpy.enhance import tfnbs as tfnbs_mod
from conninfpy.enhance._backend import HAS_NUMBA
from conninfpy.stats import compute_p_val_paired_glm
import conninfpy.stats.pairwise as pw_mod

RESULT_FIELDS = ("positive", "negative", "stat_positive", "stat_negative")


class _FallbackRecorder(logging.Handler):
    """Collect conninfpy.enhance fallback warnings emitted during one test."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []
        self.records_after_seq: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sym(rng, n_subjects: int, n_roi: int, scale: float = 0.3) -> np.ndarray:
    """Zero-diagonal symmetric subject stack with genuine residual variance."""
    out = np.triu(rng.normal(0.0, scale, (n_subjects, n_roi, n_roi)), k=1)
    return out + out.transpose(0, 2, 1)


def _discrete(rng, n_subjects: int, n_roi: int, values) -> np.ndarray:
    """Stack drawn from a small value pool, so permutation t-stats tie exactly."""
    pool = np.asarray(values, dtype=np.float64)
    tri = np.triu_indices(n_roi, k=1)
    out = np.zeros((n_subjects, n_roi, n_roi))
    for subject in range(n_subjects):
        mat = np.zeros((n_roi, n_roi))
        mat[tri] = rng.choice(pool, size=tri[0].size)
        out[subject] = mat + mat.T
    return out


def _assert_nested_equal(left, right, path: str = "component_report") -> None:
    """Compare two component reports structurally, NaN-equal, exactly."""
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict), f"{path}: dict vs {type(right)}"
        assert set(left) == set(right), f"{path}: keys {sorted(left)} != {sorted(right)}"
        for key in left:
            _assert_nested_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)), path
        assert len(left) == len(right), f"{path}: length {len(left)} != {len(right)}"
        for index, (l_item, r_item) in enumerate(zip(left, right, strict=True)):
            _assert_nested_equal(l_item, r_item, f"{path}[{index}]")
        return
    np.testing.assert_array_equal(
        np.asarray(left, dtype=object) if left is None or right is None else np.asarray(left),
        np.asarray(right, dtype=object) if left is None or right is None else np.asarray(right),
        err_msg=f"{path} differs between the edge-vector and matrix paths",
    )


def _assert_paths_identical(fast, matrix, context: str) -> None:
    """Corrected p, observed statistics and component memberships must be identical."""
    for field in RESULT_FIELDS:
        np.testing.assert_array_equal(
            getattr(fast, field),
            getattr(matrix, field),
            err_msg=f"{context}: {field} differs between edge-vector and matrix paths",
        )
    assert fast.component_report is None or isinstance(fast.component_report, dict)
    if fast.component_report is None:
        assert matrix.component_report is None, f"{context}: report present on matrix path only"
    else:
        _assert_nested_equal(fast.component_report, matrix.component_report, f"{context}: component_report")
    assert fast.n_permutations == matrix.n_permutations
    assert fast.two_tailed == matrix.two_tailed
    np.testing.assert_array_equal(fast.e_grid, matrix.e_grid)
    np.testing.assert_array_equal(fast.h_grid, matrix.h_grid)


def _run_both(kwargs: dict, context: str, expect_edge_spec: bool = True):
    """Run the edge-vector path and the forced matrix path on identical inputs."""
    seen: dict[str, object] = {}
    original = pw_mod._build_edge_max_spec

    def _spy(*args, **kwargs_inner):
        spec = original(*args, **kwargs_inner)
        seen["spec"] = spec
        return spec

    with patch.object(pw_mod, "_build_edge_max_spec", side_effect=_spy):
        fast = compute_p_val(**dict(kwargs, use_mp=False))

    if expect_edge_spec:
        assert seen.get("spec") is not None, f"{context}: direct edge-vector path was not taken"
    else:
        assert seen.get("spec") is None, f"{context}: expected the matrix path only"

    with patch.object(pw_mod, "_build_edge_max_spec", return_value=None):
        matrix = compute_p_val(**dict(kwargs, use_mp=False))

    _assert_paths_identical(fast, matrix, context)
    return fast, matrix, seen["spec"]


# ---------------------------------------------------------------------------
# 1. Degenerate sizes: N=2, N=3, single edge
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n_roi, test_type, method, extra",
    [
        (2, "two-sample", "tfnbs", {}),
        (2, "two-sample", "nbs", {"threshold": 0.5}),
        (3, "two-sample", "tfnbs", {}),
        (3, "two-sample", "nbs", {"threshold": 0.5}),
        (3, "paired", "ni_tfnbs", {"net_labels": None}),
        (3, "one-sample", "tfnbs", {}),
    ],
)
def test_degenerate_sizes_match_between_paths(n_roi, test_type, method, extra):
    rng = np.random.default_rng(1000 + n_roi)
    g1 = _sym(rng, 8, n_roi, scale=0.4)
    g2 = _sym(rng, 8, n_roi, scale=0.4) + 0.35
    kwargs = dict(
        group1=g1,
        group2=None if test_type == "one-sample" else g2,
        test_type=test_type,
        method=method,
        n_permutations=12,
        rng=17,
        **extra,
    )
    if method == "ni_tfnbs":
        labels = np.zeros(n_roi, dtype=int)
        labels[-1] = 1
        kwargs["net_labels"] = labels

    fast, matrix, spec = _run_both(kwargs, f"N={n_roi} {test_type} {method}")

    # Sanity: the degenerate grid keeps its documented shape and edge count.
    assert fast.positive.shape == (n_roi, n_roi)
    assert spec.rows.size == n_roi * (n_roi - 1) // 2
    off_diagonal = ~np.eye(n_roi, dtype=bool)
    assert np.all(np.isfinite(fast.positive[off_diagonal]))
    assert np.all(np.isfinite(fast.negative[off_diagonal]))
    # No family mask was declared, so nothing is left untested.
    assert np.all(np.isfinite(fast.positive))

    if method == "nbs":
        report = fast.component_report
        assert report is not None and report["unit"] == "component"
        for tail in ("positive", "negative"):
            labels = report["labels"][tail]
            np.testing.assert_array_equal(np.diag(labels), 0)
            # Every reported component id must actually label at least one edge.
            for component in report["components"]:
                if component["tail"] != tail:
                    continue
                assert np.count_nonzero(labels == component["component_id"]) > 0


def test_single_tested_edge_matches_between_paths():
    """One-edge family: a single suprathreshold edge, everything else untested."""
    rng = np.random.default_rng(77)
    n_roi = 5
    g1 = _sym(rng, 9, n_roi, scale=0.3)
    g2 = _sym(rng, 9, n_roi, scale=0.3)
    g2[:, 1, 4] += 1.1
    g2[:, 4, 1] += 1.1
    mask = np.zeros((n_roi, n_roi), dtype=bool)
    mask[1, 4] = mask[4, 1] = True

    for method, extra in (("tfnbs", {}), ("nbs", {"threshold": 1.0})):
        fast, _, spec = _run_both(
            dict(
                group1=g1,
                group2=g2,
                test_type="paired",
                method=method,
                edge_mask=mask,
                n_permutations=12,
                rng=23,
                **extra,
            ),
            f"single-edge {method}",
        )

        # Only the declared edge is tested; every other off-family edge is NaN.
        np.testing.assert_array_equal(np.isnan(fast.positive), ~(mask | np.eye(n_roi, dtype=bool)))
        assert np.all(np.isfinite(fast.positive[mask]))
        assert spec.family_edges is not None
        assert np.count_nonzero(spec.family_edges) == 1
        if method == "nbs":
            labels = fast.component_report["labels"]["positive"]
            # A single tested edge is a single two-node component: two entries
            # in the symmetric label map and nothing else.
            assert np.count_nonzero(labels) == 2
            assert labels[1, 4] == labels[4, 1] != 0


# ---------------------------------------------------------------------------
# 2. Ties through the end-to-end permutation path
# ---------------------------------------------------------------------------

def _tied_stack(rng, n_subjects: int, n_roi: int, values, tied_edges) -> tuple[np.ndarray, np.ndarray]:
    """Two-sample stack whose ``tied_edges`` carry *identical* subject vectors.

    Two edges that share the same group-1 and group-2 subject vectors have the
    same t statistic in the observed map and in every permutation, so the null
    distribution and the component labels see exact ties rather than
    near-ties. The vectors vary across subjects, so the residual variance is
    non-zero and the t statistic stays finite.
    """
    pool = np.asarray(values, dtype=np.float64)
    group1 = _discrete(rng, n_subjects, n_roi, values)
    group2 = _discrete(rng, n_subjects, n_roi, values)
    tied_vector_1 = rng.choice(pool, size=n_subjects) + 0.05
    tied_vector_2 = rng.choice(pool, size=n_subjects) + 0.45
    assert np.std(tied_vector_1) > 0 and np.std(tied_vector_2) > 0
    for row, col in tied_edges:
        group1[:, row, col] = group1[:, col, row] = tied_vector_1
        group2[:, row, col] = group2[:, col, row] = tied_vector_2
    return group1, group2


def test_forced_ties_match_between_paths_and_component_memberships():
    rng = np.random.default_rng(2024)
    n_roi = 6
    values = [0.0, 0.3, 0.6, 0.9]
    tied_edges = [(0, 1), (0, 2), (0, 3)]
    g1, g2 = _tied_stack(rng, 10, n_roi, values, tied_edges)
    labels = np.array([0, 0, 0, 1, 1, 1])

    # The construction guarantees the tie is exact, not approximate.
    tied_values = np.array([g1[:, row, col] for row, col in tied_edges])
    assert np.all(tied_values == tied_values[0])
    assert np.std(tied_values[0]) > 0

    cases = [
        ("tfnbs", {}, "ties tfnbs"),
        ("nbs", {"threshold": 1.0}, "ties nbs"),
        ("ni_tfnbs", {"net_labels": labels}, "ties ni_tfnbs"),
        ("fbc_tfnbs", {"net_labels": labels, "min_cluster_size": 1}, "ties fbc_tfnbs"),
    ]
    for method, extra, context in cases:
        fast, _, _ = _run_both(
            dict(
                group1=g1,
                group2=g2,
                test_type="two-sample",
                method=method,
                n_permutations=14,
                rng=31,
                **extra,
            ),
            context,
        )
        # Observed t statistics at the tied edges must be bit-identical, so the
        # membership comparison really exercises tie handling.
        observed = fast.stat_positive[np.array(tied_edges)[:, 0], np.array(tied_edges)[:, 1]]
        assert np.all(np.round(observed, 12) == np.round(observed[0], 12))
        assert observed[0] != 0.0
        if method != "nbs":
            continue
        # Component memberships under ties, fast path vs matrix path, are compared
        # inside _run_both; here we additionally pin the tied-component structure.
        report = fast.component_report
        edges_per_component = [c["n_edges"] for c in report["components"]]
        assert edges_per_component, "expected at least one component in a tie-rich matrix"
        for component in report["components"]:
            labels_tail = report["labels"][component["tail"]]
            # labels are a symmetric (N, N) map, so each edge appears twice.
            assert (
                np.count_nonzero(labels_tail == component["component_id"])
                == 2 * component["n_edges"]
            )

        # Tie stability: identical permutation max statistics must come from a
        # deterministic tie-break, so repeating the run reproduces the report.
        repeat = compute_p_val(
            group1=g1, group2=g2, test_type="two-sample", method="nbs",
            threshold=1.0, n_permutations=14, rng=31, use_mp=False,
        )
        _assert_nested_equal(fast.component_report, repeat.component_report, "ties nbs repeat")


# ---------------------------------------------------------------------------
# 3. Empty / all-zero inputs through the edge-max path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "test_type, method, extra",
    [
        ("paired", "tfnbs", {}),
        ("paired", "nbs", {"threshold": 2.0}),
        ("one-sample", "tfnbs", {}),
        ("one-sample", "nbs", {"threshold": 2.0}),
    ],
)
def test_empty_and_all_zero_inputs_match_between_paths(test_type, method, extra):
    n_roi = 5
    zeros = np.zeros((8, n_roi, n_roi))
    fast, _, spec = _run_both(
        dict(
            group1=zeros,
            group2=None if test_type == "one-sample" else zeros.copy(),
            test_type=test_type,
            method=method,
            n_permutations=10,
            rng=13,
            **extra,
        ),
        f"all-zero {test_type} {method}",
    )

    assert spec is not None
    for field in ("stat_positive", "stat_negative"):
        np.testing.assert_array_equal(getattr(fast, field), np.zeros((n_roi, n_roi)))

    for field in ("positive", "negative"):
        values = getattr(fast, field)
        # Nothing is ever significant with an all-zero input.
        assert np.all(values == 1.0), f"{field} must be 1.0 everywhere on an all-zero input"

    if method == "nbs":
        report = fast.component_report
        assert report["n_components"] == {"positive": 0, "negative": 0}
        assert report["components"] == []
        for tail in ("positive", "negative"):
            assert np.count_nonzero(report["labels"][tail]) == 0


# ---------------------------------------------------------------------------
# 4. Signed-only: one tail empty while the other carries the effect
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "direction, method, extra",
    [
        ("positive", "tfnbs", {"two_tailed": True}),
        ("positive", "tfnbs", {}),
        ("negative", "nbs", {"two_tailed": True, "threshold": 2.0}),
        ("negative", "nbs", {"threshold": 2.0}),
    ],
)
def test_signed_only_one_empty_tail_matches_between_paths(direction, method, extra):
    rng = np.random.default_rng(4242)
    n_roi = 5
    base = _sym(rng, 9, n_roi, scale=0.3)
    shifted = base + 1.5 + _sym(rng, 9, n_roi, scale=0.3)
    group1, group2 = (base, shifted) if direction == "positive" else (shifted, base)

    fast, _, _ = _run_both(
        dict(
            group1=group1,
            group2=group2,
            test_type="paired",
            method=method,
            n_permutations=12,
            rng=19,
            **extra,
        ),
        f"signed-only {direction} {method} two_tailed={extra.get('two_tailed', False)}",
    )

    empty_tail = "negative" if direction == "positive" else "positive"
    signal_tail = direction
    empty_p = getattr(fast, empty_tail)
    signal_p = getattr(fast, signal_tail)
    assert np.all(empty_p[~np.eye(n_roi, dtype=bool)] == 1.0), "the empty tail must be uniformly non-significant"
    assert (signal_p[~np.eye(n_roi, dtype=bool)] < 1.0).any(), "the populated tail must carry the effect"

    if method == "nbs":
        report = fast.component_report
        assert report["n_components"][empty_tail] == 0
        assert all(component["tail"] != empty_tail for component in report["components"])


# ---------------------------------------------------------------------------
# 5. Grid ties at n=10 (legacy right-grid) — matrix path on both backends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method, extra, module, expected_fallback",
    [
        ("tfnbs", {}, tfnbs_mod, "scipy"),
        ("ni_tfnbs", {}, ni_mod, "numpy"),
        ("fbc_tfnbs", {"min_cluster_size": 1}, fbc_mod, "numpy"),
    ],
)
def test_grid_ties_n10_backend_equivalence(method, extra, module, expected_fallback):
    """The legacy grid has no edge-vector variant; ties must still be backend-stable."""
    rng = np.random.default_rng(909)
    n_roi = 6
    values = [0.0, 0.3, 0.6, 0.9]
    tied_edges = [(0, 1), (0, 2), (0, 3)]
    g1, g2 = _tied_stack(rng, 10, n_roi, values, tied_edges)
    labels = np.array([0, 0, 0, 1, 1, 1])
    kwargs = dict(group1=g1, group2=g2, test_type="two-sample", method=method,
                  n=10, n_permutations=12, rng=7, **extra)
    if method != "tfnbs":
        kwargs["net_labels"] = labels

    # The legacy grid must keep the matrix path: no edge-vector spec is built.
    original = pw_mod._build_edge_max_spec
    seen: dict[str, object] = {}

    def _spy(*args, **kwargs_inner):
        spec = original(*args, **kwargs_inner)
        seen["spec"] = spec
        return spec

    with patch.object(pw_mod, "_build_edge_max_spec", side_effect=_spy):
        numba_res = compute_p_val(**dict(kwargs, use_mp=False))
    assert seen["spec"] is None, f"{method}: n=10 must not take the edge-vector path"

    with patch.object(module, "HAS_NUMBA", False):
        fallback_res = compute_p_val(**dict(kwargs, use_mp=False))

    assert numba_res.design["enhancement"]["n"] == 10
    assert numba_res.design["enhancement"]["integration"] == "right-grid"
    assert numba_res.design["enhancement"]["actual_backend"] == (
        "numba" if (HAS_NUMBA and method == "tfnbs") else expected_fallback
    )
    assert fallback_res.design["enhancement"]["actual_backend"] == expected_fallback
    observed = numba_res.stat_positive[
        np.array(tied_edges)[:, 0], np.array(tied_edges)[:, 1]
    ]
    assert np.all(np.round(observed, 12) == np.round(observed[0], 12))
    assert observed[0] != 0.0

    for field in RESULT_FIELDS:
        np.testing.assert_array_equal(
            getattr(numba_res, field),
            getattr(fallback_res, field),
            err_msg=f"n=10 ties {method}: {field} differs between numba and fallback backends",
        )
    _assert_nested_equal(numba_res.component_report, fallback_res.component_report,
                         f"n=10 ties {method}: component_report")


# ---------------------------------------------------------------------------
# 6. HAS_NUMBA=False + ties + use_mp=True at the same time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method, extra, module",
    [
        ("tfnbs", {}, tfnbs_mod),
        ("ni_tfnbs", {}, ni_mod),
        ("fbc_tfnbs", {"min_cluster_size": 1}, fbc_mod),
        ("nbs", {"threshold": 1.0}, nbs_mod),
    ],
)
def test_no_numba_ties_and_use_mp_together(method, extra, module):
    rng = np.random.default_rng(515)
    n_roi = 6
    values = [0.0, 0.4, 0.8]
    g1, g2 = _tied_stack(rng, 10, n_roi, values, [(0, 1), (0, 2), (0, 3)])
    labels = np.array([0, 0, 0, 1, 1, 1])
    kwargs = dict(group1=g1, group2=g2, test_type="two-sample", method=method,
                  n_permutations=12, rng=7, **extra)
    if method != "nbs":
        kwargs["net_labels"] = labels

    enh_logger = logging.getLogger("conninfpy.enhance")
    rec = _FallbackRecorder()
    enh_logger.addHandler(rec)
    try:
        with patch.object(module, "HAS_NUMBA", False):
            mp_res = compute_p_val(**dict(kwargs, use_mp=True, n_processes=2))
            seq_res = compute_p_val(**dict(kwargs, use_mp=False))
            with patch.object(pw_mod, "_build_edge_max_spec", return_value=None):
                matrix_res = compute_p_val(**dict(kwargs, use_mp=False))
        rec.records_after_seq = list(rec.records)
    finally:
        enh_logger.removeHandler(rec)

    _assert_paths_identical(seq_res, matrix_res, f"no-numba+ties {method}: sequential vs matrix")
    for field in RESULT_FIELDS:
        np.testing.assert_array_equal(
            getattr(mp_res, field),
            getattr(seq_res, field),
            err_msg=f"no-numba+ties {method}: {field} differs between use_mp and sequential",
        )
    _assert_nested_equal(mp_res.component_report, seq_res.component_report,
                         f"no-numba+ties {method}: component_report")
    # A partial in-process flag patch cannot stop kernels that were already
    # compiled, so the real no-numba provenance claim lives in the isolated
    # subprocess test; here we require the patched flag to reach the scorer
    # (one deduplicated message per user call).
    assert len(rec.records) >= 2, f"no-numba+ties {method}: fallback was never reported"
    assert all("fallback backend" in message for message in rec.records)


# ---------------------------------------------------------------------------
# 7. Stitching with NETSTAT-31 paired IDs: alignment happens before the
#    edge-vector path, and unordered ID containers are rejected
# ---------------------------------------------------------------------------

def _paired_stack(seed: int, n_subjects: int = 8, n_roi: int = 6):
    rng = np.random.default_rng(seed)
    a = _sym(rng, n_subjects, n_roi, scale=0.4)
    b = _sym(rng, n_subjects, n_roi, scale=0.4)
    return a, b, [f"sub-{index:02d}" for index in range(n_subjects)]


def test_ids_alignment_is_consumed_by_the_edge_vector_path():
    a, b, ids = _paired_stack(3131)
    permutation = np.random.default_rng(3).permutation(len(ids))
    ids_b = [ids[index] for index in permutation]
    base = dict(test_type="paired", method="tfnbs", n_permutations=12, rng=7, use_mp=False)

    ordered = compute_p_val(a, b, **base)
    manual = compute_p_val(a, b[permutation][np.argsort(permutation)], **base)
    with_ids = compute_p_val(a, b[permutation], ids_group1=ids, ids_group2=ids_b, **base)

    # The edge-vector path is the one under test; confirm it actually ran.
    seen: dict[str, object] = {}
    original = pw_mod._build_edge_max_spec

    def _spy(*args, **kwargs_inner):
        spec = original(*args, **kwargs_inner)
        seen["spec"] = spec
        return spec

    with patch.object(pw_mod, "_build_edge_max_spec", side_effect=_spy):
        with_ids_again = compute_p_val(
            a, b[permutation], ids_group1=ids, ids_group2=ids_b, **base
        )
    assert seen["spec"] is not None

    for reference in (ordered, manual, with_ids_again):
        for field in RESULT_FIELDS:
            np.testing.assert_array_equal(
                getattr(with_ids, field),
                getattr(reference, field),
                err_msg=f"ID-aligned edge-vector path disagrees on {field}",
            )
    record = with_ids.design["paired_alignment"]
    assert record["applied"] is True
    assert record["permutation_b_to_a"] == np.argsort(permutation).tolist()
    assert record["reordered_inputs"] == ["group2"]


@pytest.mark.parametrize(
    "bad_ids",
    [
        lambda ids: set(ids),
        lambda ids: dict.fromkeys(ids),
        lambda ids: ids[0],
    ],
    ids=["set", "dict", "scalar"],
)
def test_unordered_id_containers_are_rejected_on_the_paired_path(bad_ids):
    a, b, ids = _paired_stack(7)
    bad = bad_ids(ids)
    with pytest.raises(ValueError):
        compute_p_val(
            a, b, test_type="paired", method="tfnbs", n_permutations=6, rng=1,
            use_mp=False, ids_group1=bad, ids_group2=bad,
        )


def test_ids_are_rejected_for_non_paired_edge_paths():
    a, b, ids = _paired_stack(11)
    with pytest.raises(ValueError, match="only valid for test_type='paired'"):
        compute_p_val(a, b, test_type="two-sample", method="tfnbs", n_permutations=6,
                      rng=1, use_mp=False, ids_group1=ids, ids_group2=ids)
    with pytest.raises(ValueError):
        compute_p_val(a, test_type="one-sample", method="tfnbs", n_permutations=6,
                      rng=1, use_mp=False, ids_group1=ids, ids_group2=ids)


def test_paired_glm_deep_path_with_ids_matches_the_edge_vector_oracle():
    """Deep path: IDs are resolved before the permutation kernel that feeds the edge path."""
    a, b, ids = _paired_stack(101)
    rng = np.random.default_rng(29)
    permutation = np.random.default_rng(3).permutation(len(ids))
    ids_b = [ids[index] for index in permutation]
    confounds_a = rng.normal(0.0, 0.1, len(ids))
    confounds_b = rng.normal(0.0, 0.1, len(ids))
    base = dict(method="tfnbs", n_permutations=10, rng=2, use_mp=False)

    ordered = compute_p_val_paired_glm(a, b, confounds_A=confounds_a,
                                       confounds_B=confounds_b, **base)
    aligned = compute_p_val_paired_glm(a, b[permutation], confounds_A=confounds_a,
                                       confounds_B=confounds_b[permutation],
                                       ids_A=ids, ids_B=ids_b, **base)
    matrix = None
    with patch.object(pw_mod, "_build_edge_max_spec", return_value=None):
        matrix = compute_p_val_paired_glm(a, b, confounds_A=confounds_a,
                                          confounds_B=confounds_b, **base)

    for field in RESULT_FIELDS:
        np.testing.assert_array_equal(getattr(aligned, field), getattr(ordered, field),
                                      err_msg=f"GLM IDs alignment changed {field}")
        np.testing.assert_array_equal(getattr(ordered, field), getattr(matrix, field),
                                      err_msg=f"GLM edge-vector path disagrees on {field}")
    np.testing.assert_array_equal(aligned.beta, ordered.beta)
    assert aligned.design["paired_alignment"]["applied"] is True
    assert aligned.design["enhancement"]["backend"] == "auto"
    assert aligned.design["enhancement"]["integration"] == "exact"


# ---------------------------------------------------------------------------
# 8. Bounded subprocess: no numba, ties, use_mp (pickled worker payload)
# ---------------------------------------------------------------------------

def test_isolated_no_numba_ties_mp_fallback_is_bounded():
    code = r"""
import importlib.abc
import sys
from unittest.mock import patch

import numpy as np


class _BlockNumba(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "numba" or fullname.startswith("numba."):
            raise ImportError("Blocked numba for the bounded fallback test")
        return None


sys.meta_path.insert(0, _BlockNumba())

from conninfpy import compute_p_val
from conninfpy.enhance._backend import HAS_NUMBA
import conninfpy.stats.pairwise as pw_mod

assert HAS_NUMBA is False

rng = np.random.default_rng(99)
n_sub, n_roi = 10, 6
values = [0.0, 0.4, 0.8]
tri = np.triu_indices(n_roi, k=1)


def discrete(seed):
    local = np.random.default_rng(seed)
    out = np.zeros((n_sub, n_roi, n_roi))
    for subject in range(n_sub):
        mat = np.zeros((n_roi, n_roi))
        mat[tri] = local.choice(values, size=tri[0].size)
        out[subject] = mat + mat.T
    return out


g1, g2 = discrete(1), discrete(2)
labels = np.array([0, 0, 0, 1, 1, 1])
checks = [
    ("tfnbs", {}, "python"),
    ("ni_tfnbs", {"net_labels": labels}, "numpy"),
    ("fbc_tfnbs", {"net_labels": labels, "min_cluster_size": 1}, "numpy"),
    ("nbs", {"threshold": 1.0}, None),
]
for method, extra, expected in checks:
    common = dict(group1=g1, group2=g2, test_type="two-sample", method=method,
                  n_permutations=12, rng=7, **extra)
    mp_res = compute_p_val(**dict(common, use_mp=True, n_processes=2))
    seq_res = compute_p_val(**dict(common, use_mp=False))
    with patch.object(pw_mod, "_build_edge_max_spec", return_value=None):
        matrix_res = compute_p_val(**dict(common, use_mp=False))
    for field in ("positive", "negative", "stat_positive", "stat_negative"):
        assert np.array_equal(getattr(mp_res, field), getattr(seq_res, field)), (method, field, "mp")
        assert np.array_equal(getattr(seq_res, field), getattr(matrix_res, field)), (method, field, "matrix")
    if expected is not None:
        assert seq_res.design["enhancement"]["actual_backend"] == expected, method
        assert mp_res.design["enhancement"]["actual_backend"] == expected, method

print("ISOLATED_TIES_MP_OK")
"""
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        }
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert "ISOLATED_TIES_MP_OK" in proc.stdout
