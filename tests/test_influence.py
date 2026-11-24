"""Descriptive per-subject leave-one-out influence on explicitly chosen edges.

Synthetic data only (small matrices, few subjects). Every test checks a
property of the descriptive report against an independent manual computation
or against a documented unavailable reason; no test relies on a p-value.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

import conninfpy
from conninfpy import analyze, compute_p_val_glm
from conninfpy.influence import (
    CHECK_TOLERANCE,
    MAX_EDGES,
    MAX_RECORDS,
    subject_influence,
)

N = 6
FAST = {"method": "tstat", "n_permutations": 19, "use_mp": False, "rng": 0,
        "input": "z"}
EDGES = [(0, 3), (1, 2)]
ALLOWED_STATUSES = {
    "ok", "zero_mad", "constant", "rank_loss", "not_estimable",
    "no_residual_df", "not_in_coverage", "nan_edge",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def stack(n, *, mean=0.3, sd=0.2, seed=0, regions=N):
    """Symmetric, zero-diagonal (n, regions, regions) matrices on the z scale."""
    rng = np.random.default_rng(seed)
    out = mean + sd * rng.normal(size=(n, regions, regions))
    out = (out + out.transpose(0, 2, 1)) / 2
    out[:, np.arange(regions), np.arange(regions)] = 0.0
    return out


def records(report):
    return {(row["subject"], tuple(row["edge"])): row
            for row in report["per_subject_edge"]}


def edge_entry(report, edge):
    return next(row for row in report["edges"] if tuple(row["edge"]) == edge)


def _assert_provenance_equal(left, right):
    """Recursive provenance equality that tolerates numpy payloads."""
    if isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_provenance_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_provenance_equal(a, b)
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        np.testing.assert_array_equal(left, right)
        return
    assert left == right


def codes(report):
    return [item["code"] for item in report["unavailable"]]


def one_sample(values, seed=1):
    """One-sample result whose edge (1, 2) holds the given values."""
    matrices = stack(len(values), seed=seed)
    matrices[:, 1, 2] = matrices[:, 2, 1] = np.asarray(values, dtype=float)
    result = analyze(group1=matrices, test_type="one-sample", **FAST).inference
    return matrices, result


# ----------------------------------------------------------------------
# 1. Known influential participant
# ----------------------------------------------------------------------
def test_one_sample_points_at_the_influential_subject():
    matrices = stack(10, seed=1)
    matrices[3, 1, 2] = matrices[3, 2, 1] = 1.8
    result = analyze(group1=matrices, test_type="one-sample", **FAST).inference

    report = subject_influence(result, edges=[(1, 2), (0, 3)], group1=matrices)
    assert report["unavailable"] == []
    assert report["descriptive_only"] is True
    assert "pulled the effect down" in report["delta"]["definition"]
    # the AnalyzeResult wrapper is accepted as well as its .inference
    analyzed = analyze(group1=matrices, test_type="one-sample", **FAST)
    assert subject_influence(analyzed, edges=[(1, 2)], group1=matrices)[
        "summary"
    ] == subject_influence(analyzed.inference, edges=[(1, 2)],
                           group1=matrices)["summary"]
    assert report["summary"]["top_subject_by_max_abs_delta"] == 3
    assert report["summary"]["top_edge"] == [1, 2]
    assert report["ranking"][0] == 3

    rows = records(report)
    others = max(abs(rows[(subject, (1, 2))]["delta"]) for subject in (0, 1, 2))
    assert abs(rows[(3, (1, 2))]["delta"]) > 5 * others
    # the effect and the t are the native ones, not a rescaled statistic
    values = matrices[:, 1, 2]
    entry = edge_entry(report, (1, 2))
    assert entry["effect"] == pytest.approx(values.mean(), rel=1e-12)
    manual_t = values.mean() / (values.std(ddof=1) / np.sqrt(values.size))
    assert entry["stat"] == pytest.approx(manual_t, rel=1e-10)
    assert entry["df"] == values.size - 1
    assert rows[(3, (1, 2))]["leverage"] == pytest.approx(1 / values.size)


def test_two_sample_welch_and_student_both_point_at_the_outlier():
    first = stack(7, mean=0.2, sd=0.15, seed=2)
    second = stack(6, mean=0.6, sd=0.15, seed=3)
    first[2, 1, 2] = first[2, 2, 1] = 1.7

    welch = analyze(group1=first, group2=second, test_type="two-sample",
                    var_equal=False, **FAST).inference
    student = analyze(group1=first, group2=second, test_type="two-sample",
                      var_equal=True, **FAST).inference
    welch_report = subject_influence(welch, edges=[(1, 2)], group1=first,
                                     group2=second, var_equal=False)
    student_report = subject_influence(student, edges=[(1, 2)], group1=first,
                                       group2=second, var_equal=True)
    assert welch_report["unavailable"] == []
    assert student_report["unavailable"] == []
    assert welch_report["summary"]["top_subject_by_max_abs_delta"] == 2
    assert student_report["summary"]["top_subject_by_max_abs_delta"] == 2

    welch_row = records(welch_report)[(2, (1, 2))]
    student_row = records(student_report)[(2, (1, 2))]
    # Welch df are edge-specific and differ from the pooled n1 + n2 - 2
    assert welch_row["n_loo"] == 7 + 6 - 1
    assert welch_row["n_loo_group1"] == 6
    assert welch_row["n_loo_group2"] == 6
    assert welch_row["df_loo"] != 7 + 6 - 2 - 1
    assert student_row["df_loo"] == 7 + 6 - 2 - 1
    # Welch-Satterthwaite recomputed from the reduced sample
    keep = np.concatenate([np.arange(7) != 2, np.ones(6, dtype=bool)])
    a = first[keep[:7], 1, 2]
    b = second[keep[7:], 1, 2]
    se1, se2 = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    manual_df = (se1 + se2) ** 2 / (se1**2 / (a.size - 1) + se2**2 / (b.size - 1))
    assert welch_row["df_loo"] == pytest.approx(manual_df, rel=1e-10)
    # the two t statistics are genuinely different native statistics
    assert welch_report["edges"][0]["stat"] != pytest.approx(
        student_report["edges"][0]["stat"], rel=1e-6
    )


def test_paired_deletes_the_whole_pair():
    condition_a = stack(8, mean=0.25, seed=4)
    condition_b = stack(8, mean=0.55, seed=5)
    condition_b[5] = condition_a[5] + 0.9  # one subject with a large B - A
    result = analyze(group1=condition_a, group2=condition_b, test_type="paired",
                     **FAST).inference

    report = subject_influence(result, edges=[(1, 2)], group1=condition_a,
                               group2=condition_b)
    assert report["unavailable"] == []
    assert report["summary"]["top_subject_by_max_abs_delta"] == 5
    row = records(report)[(5, (1, 2))]
    difference = condition_b[:, 1, 2] - condition_a[:, 1, 2]
    assert row["n_loo"] == difference.size - 1
    # removing the pair removes the row from BOTH conditions
    reduced = np.delete(difference, 5)
    assert row["effect_loo"] == pytest.approx(reduced.mean(), rel=1e-12)
    manual_t = reduced.mean() / (reduced.std(ddof=1) / np.sqrt(reduced.size))
    assert row["stat_loo"] == pytest.approx(manual_t, rel=1e-10)
    assert row["df_loo"] == reduced.size - 1
    # a paired result reports the Fisher z difference, never a transformed mean
    assert report["units"] == "Fisher z"


def test_glm_with_confound_points_at_the_influential_subject():
    rng = np.random.default_rng(6)
    age = rng.uniform(10, 60, 9)
    age[4] = 95.0  # high leverage
    sex = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    response = stack(9, mean=0.3, seed=7)
    response[4, 1, 2] = response[4, 2, 1] = 2.2
    result = analyze(Y=response, interest=age, confounds=sex, **FAST).inference
    design = np.column_stack([np.ones(age.size), sex, age])

    report = subject_influence(result, edges=[(1, 2), (0, 3)], Y=response,
                               design_matrix=design)
    assert report["unavailable"] == []
    assert report["summary"]["top_subject_by_max_abs_delta"] == 4
    entry = edge_entry(report, (1, 2))
    assert entry["df"] == age.size - int(np.linalg.matrix_rank(design))
    assert entry["rank"] == 3
    rows = records(report)
    assert rows[(4, (1, 2))]["leverage"] > 0.5  # high leverage from the design
    manual = np.linalg.pinv(design) @ response[:, 1, 2].reshape(-1, 1)
    assert entry["effect"] == pytest.approx(float(manual[2, 0]), rel=1e-10)


# ----------------------------------------------------------------------
# 2. Majority ties, MAD and constancy
# ----------------------------------------------------------------------
def test_zero_mad_is_not_constant_and_the_outlier_is_found():
    values = np.full(10, 0.5)
    values[4] = -1.5
    matrices, result = one_sample(values, seed=8)

    report = subject_influence(result, edges=[(1, 2)], group1=matrices)
    assert report["unavailable"] == []
    entry = edge_entry(report, (1, 2))
    assert entry["mad"] == 0.0
    assert entry["status"] == "ok"  # the full edge is not constant
    rows = records(report)
    assert report["summary"]["top_subject_by_max_abs_delta"] == 4
    assert rows[(4, (1, 2))]["status"] == "zero_mad"
    assert rows[(4, (1, 2))]["loo_fit_status"] == "constant"
    assert rows[(4, (1, 2))]["stat_loo"] is None  # tied values: t is undefined
    assert rows[(0, (1, 2))]["status"] == "zero_mad"
    assert rows[(0, (1, 2))]["stat_loo"] is not None
    assert abs(rows[(4, (1, 2))]["delta"]) > 5 * abs(rows[(0, (1, 2))]["delta"])

    constant_matrices = stack(9, seed=9)
    constant_matrices[:, 1, 2] = constant_matrices[:, 2, 1] = 0.0
    constant_result = analyze(group1=constant_matrices, test_type="one-sample",
                              **FAST).inference
    constant_report = subject_influence(constant_result, edges=[(1, 2)],
                                        group1=constant_matrices)
    entry = edge_entry(constant_report, (1, 2))
    assert entry["status"] == "constant"
    assert entry["stat"] is None
    assert entry["variance"] == 0.0
    assert {row["status"] for row in constant_report["per_subject_edge"]} == {
        "constant"
    }


# ----------------------------------------------------------------------
# 3. Direction and sign in native units
# ----------------------------------------------------------------------
def test_effect_and_delta_have_the_native_direction():
    rng = np.random.default_rng(10)
    matrices = stack(9, mean=0.4, sd=0.15, seed=11)
    edge_values = 0.4 + 0.15 * rng.normal(size=9)
    matrices[:, 1, 2] = matrices[:, 2, 1] = edge_values
    result = analyze(group1=matrices, test_type="one-sample", **FAST).inference

    report = subject_influence(result, edges=[(1, 2)], group1=matrices)
    entry = edge_entry(report, (1, 2))
    assert entry["effect"] > 0  # non-zero native effect, correct sign
    assert report["scale"] == "fisher_z"
    assert report["units"] == "Fisher z"
    assert report["contrast"] == [1.0]

    rows = records(report)
    highest = int(np.argmax(edge_values))
    lowest = int(np.argmin(edge_values))
    # delta = c'beta_full - c'beta_(-s): the highest value pulled the mean up
    assert rows[(highest, (1, 2))]["delta"] > 0
    assert rows[(lowest, (1, 2))]["delta"] < 0
    manual_loo = np.delete(edge_values, highest).mean()
    assert rows[(highest, (1, 2))]["delta"] == pytest.approx(
        edge_values.mean() - manual_loo, rel=1e-10
    )


# ----------------------------------------------------------------------
# 4. Paired subject IDs and the double-reorder guard
# ----------------------------------------------------------------------
def _paired_with_shuffled_condition_b():
    condition_a = stack(7, mean=0.2, seed=12)
    condition_b = stack(7, mean=0.5, seed=13)
    condition_b[2] = condition_a[2] + 0.7
    ids = [f"s{index}" for index in range(7)]
    permutation = np.random.default_rng(14).permutation(7)
    shuffled_b = condition_b[permutation]
    ids_b = [ids[index] for index in permutation]
    result = analyze(group1=condition_a, group2=shuffled_b, test_type="paired",
                     ids_group1=ids, ids_group2=ids_b, **FAST).inference
    return condition_a, condition_b, shuffled_b, ids, ids_b, permutation, result


def test_paired_ids_apply_the_recorded_alignment_exactly_once():
    condition_a, condition_b, shuffled_b, ids, ids_b, _, result = (
        _paired_with_shuffled_condition_b()
    )
    assert result.design["paired_alignment"]["identity_permutation"] is False

    report = subject_influence(result, edges=[(1, 2)], group1=condition_a,
                               group2=shuffled_b, ids_group1=ids,
                               ids_group2=ids_b)
    assert report["unavailable"] == []
    assert report["paired_alignment"]["permutation_applied"] is True
    assert report["summary"]["top_subject_by_max_abs_delta"] == 2

    aligned = subject_influence(result, edges=[(1, 2)], group1=condition_a,
                                group2=condition_b, paired_inputs_aligned=True)
    assert aligned["unavailable"] == []
    assert aligned["paired_alignment"]["permutation_applied"] is False
    for row in report["per_subject_edge"]:
        match = records(aligned)[(row["subject"], (1, 2))]
        assert row["delta"] == pytest.approx(match["delta"], rel=1e-10)
        assert row["stat_loo"] == pytest.approx(match["stat_loo"], rel=1e-10)

    # The already aligned input cannot be reordered a second time. For a paired
    # design c'beta (the mean B - A) is invariant to the row pairing, so the
    # beta check passes; the observed statistic is what catches the mismatch.
    double = subject_influence(result, edges=[(1, 2)], group1=condition_a,
                               group2=shuffled_b, paired_inputs_aligned=True)
    assert codes(double) == ["statistic_mismatch"]

    # A non-identity record without IDs cannot be applied or guessed.
    missing_ids = subject_influence(result, edges=[(1, 2)], group1=condition_a,
                                    group2=shuffled_b)
    assert codes(missing_ids) == ["paired_alignment_ids_required"]


def test_ids_without_a_recorded_alignment_are_refused():
    condition_a = stack(6, seed=15)
    condition_b = stack(6, seed=16)
    result = analyze(group1=condition_a, group2=condition_b, test_type="paired",
                     **FAST).inference
    ids = [f"s{index}" for index in range(6)]
    report = subject_influence(result, edges=[(1, 2)], group1=condition_a,
                               group2=condition_b, ids_group1=ids,
                               ids_group2=ids)
    assert codes(report) == ["paired_alignment_unrecorded"]


# ----------------------------------------------------------------------
# 5. Available-case coverage, rank loss, degrees of freedom
# ----------------------------------------------------------------------
def test_available_case_coverage_and_not_in_coverage_rows():
    matrices = stack(10, seed=17)
    matrices[0, 2, 3] = matrices[0, 3, 2] = np.nan
    matrices[1, 2, 3] = matrices[1, 3, 2] = np.nan
    result = analyze(group1=matrices, test_type="one-sample", **FAST,
                     missing="available_case", min_coverage=0.8).inference

    report = subject_influence(result, edges=[(2, 3), (0, 1)], group1=matrices)
    assert report["unavailable"] == []
    checks = {item["check"]: item["status"] for item in report["checks"]}
    assert checks["coverage_matches_missing_report"] == "passed"
    entry = edge_entry(report, (2, 3))
    assert entry["n"] == 8
    assert entry["max_abs_delta_subject"] not in (0, 1)
    rows = records(report)
    for subject in (0, 1):
        row = rows[(subject, (2, 3))]
        assert row["status"] == "not_in_coverage"
        assert row["delta"] == 0.0
        assert row["n_loo"] == 8
        assert row["stat_loo"] == pytest.approx(entry["stat"], rel=1e-12)
        assert row["leverage"] is None  # no leverage outside the edge coverage
    assert rows[(2, (2, 3))]["status"] in ALLOWED_STATUSES
    assert rows[(2, (2, 3))]["n_loo"] == 7
    # per-edge leverage: 1/n_edge of THIS edge, not 1/n of the whole group
    assert rows[(2, (2, 3))]["leverage"] == pytest.approx(1 / 8)
    assert rows[(2, (0, 1))]["leverage"] == pytest.approx(1 / matrices.shape[0])

    # An edge below min_coverage was never tested; it cannot be diagnosed.
    untested = matrices.copy()
    untested[:4, 1, 2] = untested[:4, 2, 1] = np.nan
    untested_result = analyze(group1=untested, test_type="one-sample", **FAST,
                              missing="available_case", min_coverage=0.8).inference
    assert not untested_result.analysis_provenance["tested"]["positive"][1, 2]
    with pytest.raises(ValueError, match="not in the producer's tested mask"):
        subject_influence(untested_result, edges=[(1, 2)], group1=untested)


def test_available_case_glm_leverage_uses_edge_coverage_rows():
    rng = np.random.default_rng(36)
    age = rng.uniform(10, 60, 9)
    sex = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    response = stack(9, seed=37)
    response[0, 1, 2] = response[0, 2, 1] = np.nan
    result = analyze(Y=response, interest=age, confounds=sex,
                     missing="available_case", min_coverage=0.8, **FAST).inference
    design = np.column_stack([np.ones(9), sex, age])

    report = subject_influence(result, edges=[(1, 2)], Y=response,
                               design_matrix=design)
    assert report["unavailable"] == []
    rows = records(report)
    assert rows[(0, (1, 2))]["status"] == "not_in_coverage"
    assert rows[(0, (1, 2))]["leverage"] is None
    covered = design[1:]
    hat = np.einsum("sj,sj->s", np.linalg.pinv(covered).T, covered)
    for subject in range(1, 9):
        assert rows[(subject, (1, 2))]["leverage"] == pytest.approx(
            hat[subject - 1], rel=1e-10
        )
    # n, df and variance come from the rows actually used on this edge
    assert rows[(1, (1, 2))]["n_loo"] == 7
    assert rows[(1, (1, 2))]["df_loo"] == 7 - int(np.linalg.matrix_rank(covered))
    assert report["edges"][0]["n"] == 8


def test_rank_loss_after_deletion_is_reported():
    rng = np.random.default_rng(18)
    age = rng.uniform(10, 60, 8)
    indicator = np.zeros(8)
    indicator[3] = 1.0  # a column that only one subject identifies
    response = stack(8, seed=19)
    result = analyze(Y=response, interest=age, confounds=indicator,
                     **FAST).inference
    design = np.column_stack([np.ones(8), indicator, age])

    report = subject_influence(result, edges=[(1, 2)], Y=response,
                               design_matrix=design)
    assert report["unavailable"] == []
    rows = records(report)
    assert rows[(3, (1, 2))]["status"] == "rank_loss"
    assert rows[(3, (1, 2))]["delta"] is not None
    assert all(
        rows[(subject, (1, 2))]["status"] == "ok" for subject in (0, 1, 2)
    )
    # a lost design direction is flagged, counted separately and never ranked
    assert report["summary"]["n_rank_loss_records"] == 1
    assert report["subject_summary"][3]["n_rank_loss"] == 1
    assert report["subject_summary"][3]["n_ok"] == 0
    assert report["subject_summary"][3]["max_abs_delta"] is None
    assert report["ranking"][-1] == 3
    assert report["edges"][0]["max_abs_delta_subject"] != 3


def test_rank_loss_is_relative_to_the_full_edge_design():
    rng = np.random.default_rng(38)
    age = rng.uniform(10, 60, 8)
    indicator = np.zeros(8)
    indicator[3] = 1.0  # one subject carries the only non-zero level
    response = stack(8, seed=39)
    # duplicated nuisance columns: rank 3 < p, and the contrast stays estimable
    result = analyze(Y=response, interest=age,
                     confounds=np.column_stack([indicator, indicator]),
                     **FAST).inference
    design = np.column_stack([np.ones(8), indicator, indicator, age])
    assert np.linalg.matrix_rank(design) == 3 < design.shape[1]

    report = subject_influence(result, edges=[(1, 2)], Y=response,
                               design_matrix=design)
    assert report["unavailable"] == []
    rows = records(report)
    # rank 3 -> 3: nothing was lost, so this is NOT a rank loss
    assert rows[(0, (1, 2))]["status"] == "ok"
    assert rows[(0, (1, 2))]["loo_fit_status"] == "ok"
    # removing the carrier drops rank 3 -> 2 on the same edge
    assert rows[(3, (1, 2))]["status"] == "rank_loss"
    assert rows[(3, (1, 2))]["delta"] is not None
    assert report["edges"][0]["rank"] == 3
    assert report["summary"]["n_rank_loss_records"] == 1
    assert report["subject_summary"][3]["max_abs_delta"] is None

    # losing estimability is a separate, explicit unavailable entry
    single = analyze(Y=response, interest=indicator, **FAST).inference
    single_design = np.column_stack([np.ones(8), indicator])
    single_report = subject_influence(single, edges=[(1, 2)], Y=response,
                                      design_matrix=single_design)
    assert single_report["unavailable"] == []
    single_rows = records(single_report)
    assert single_rows[(3, (1, 2))]["status"] == "not_estimable"
    assert single_rows[(3, (1, 2))]["delta"] is None
    assert single_rows[(3, (1, 2))]["stat_loo"] is None
    assert single_rows[(3, (1, 2))]["effect_loo"] is None
    assert single_rows[(0, (1, 2))]["status"] == "ok"


def test_small_n_loses_residual_df():
    matrices = stack(2, seed=20)
    result = analyze(group1=matrices, test_type="one-sample", **FAST).inference
    report = subject_influence(result, edges=[(1, 2)], group1=matrices)
    assert report["unavailable"] == []
    rows = records(report)
    for subject in (0, 1):
        assert rows[(subject, (1, 2))]["status"] == "no_residual_df"
        assert rows[(subject, (1, 2))]["df_loo"] == 0
        assert rows[(subject, (1, 2))]["stat_loo"] is None
        assert rows[(subject, (1, 2))]["delta"] is not None
        assert rows[(subject, (1, 2))]["n_loo"] == 1


# ----------------------------------------------------------------------
# 6. Unavailable reasons instead of invented numbers
# ----------------------------------------------------------------------
def test_no_input_data_is_unavailable():
    matrices = stack(6, seed=21)
    result = analyze(group1=matrices, test_type="one-sample", **FAST).inference
    report = subject_influence(result, edges=[(1, 2)])
    assert codes(report) == ["no_input_data"]
    assert report["per_subject_edge"] == []
    assert report["descriptive_only"] is True


def test_legacy_result_without_beta_is_unavailable():
    matrices = stack(6, seed=22)
    result = analyze(group1=matrices, test_type="one-sample", **FAST).inference
    legacy = copy.deepcopy(result)
    legacy.beta = None
    report = subject_influence(legacy, edges=[(1, 2)], group1=matrices)
    assert codes(report) == ["no_beta"]


def test_omnibus_f_result_is_unavailable():
    rng = np.random.default_rng(23)
    age = rng.uniform(10, 60, 7)
    response = stack(7, seed=24)
    result = compute_p_val_glm(response, interest=age, stat_type="fstat",
                               method="tstat", n_permutations=19, use_mp=False,
                               rng=0)
    assert type(result).__name__ == "OmnibusInferenceResult"
    report = subject_influence(result, edges=[(1, 2)], Y=response,
                               design_matrix=np.column_stack([np.ones(7), age]))
    assert codes(report) == ["omnibus_not_supported"]


def test_two_sample_without_var_equal_is_unavailable():
    first = stack(6, mean=0.2, sd=0.1, seed=25)
    second = stack(6, mean=0.6, sd=0.4, seed=26)
    result = analyze(group1=first, group2=second, test_type="two-sample",
                     var_equal=False, **FAST).inference
    report = subject_influence(result, edges=[(1, 2)], group1=first,
                               group2=second)
    assert codes(report) == ["var_equal_not_declared"]
    with pytest.raises(ValueError, match="explicit bool"):
        subject_influence(result, edges=[(1, 2)], group1=first, group2=second,
                          var_equal="welch")


def test_declaring_the_wrong_variance_assumption_is_a_statistic_mismatch():
    first = stack(8, mean=0.2, sd=0.1, seed=27)
    second = stack(5, mean=0.6, sd=0.4, seed=28)
    result = analyze(group1=first, group2=second, test_type="two-sample",
                     var_equal=False, **FAST).inference
    report = subject_influence(result, edges=[(1, 2)], group1=first,
                               group2=second, var_equal=True)
    assert codes(report) == ["statistic_mismatch"]
    checks = {item["check"]: item["status"] for item in report["checks"]}
    assert checks["beta_reproduced"] == "passed"
    assert checks["stat_reproduced"] == "failed"


def test_missing_stat_maps_keep_the_beta_part_available():
    matrices = stack(7, seed=35)
    result = analyze(group1=matrices, test_type="one-sample", **FAST).inference
    stripped = copy.deepcopy(result)
    stripped.stat_positive = None
    stripped.stat_negative = None
    report = subject_influence(stripped, edges=[(1, 2)], group1=matrices)
    assert report["unavailable"] == []
    checks = {item["check"]: item["status"] for item in report["checks"]}
    assert checks["beta_reproduced"] == "passed"
    assert checks["stat_maps"] == "unavailable"
    assert report["edges"][0]["effect"] is not None
    assert report["edges"][0]["stat"] is not None  # computed, not copied
    assert report["edges"][0]["stat_matches_result"] is None


def test_inputs_that_do_not_reproduce_the_model_are_unavailable():
    matrices = stack(7, seed=29)
    result = analyze(group1=matrices, test_type="one-sample", **FAST).inference
    report = subject_influence(result, edges=[(1, 2)], group1=matrices * 1.5)
    assert codes(report) == ["model_mismatch"]
    assert report["summary"] == {}


# ----------------------------------------------------------------------
# 7. The source result is not modified and no random state is consumed
# ----------------------------------------------------------------------
def test_result_is_not_mutated_and_no_rng_state_is_consumed():
    first = stack(7, mean=0.2, sd=0.15, seed=30)
    second = stack(6, mean=0.6, sd=0.25, seed=31)
    first[1, 1, 2] = first[1, 2, 1] = 1.4
    result = analyze(group1=first, group2=second, test_type="two-sample",
                     var_equal=False, **FAST).inference

    before = copy.deepcopy(result)
    stats_before = {
        key: np.array(result[key], copy=True) for key in ("positive", "negative")
    }
    beta_before = np.array(result.beta, copy=True)
    provenance_before = copy.deepcopy(result.analysis_provenance)
    np.random.seed(1234)
    state_before = np.random.get_state()

    report = subject_influence(result, edges=EDGES, group1=first, group2=second,
                               var_equal=False)
    assert report["unavailable"] == []

    for key in ("positive", "negative"):
        np.testing.assert_array_equal(result[key], stats_before[key])
    np.testing.assert_array_equal(result.beta, beta_before)
    np.testing.assert_array_equal(result.stat_signed, before.stat_signed)
    _assert_provenance_equal(result.analysis_provenance, provenance_before)
    state_after = np.random.get_state()
    assert state_after[0] == state_before[0]
    np.testing.assert_array_equal(state_after[1], state_before[1])
    assert state_after[2] == state_before[2]


# ----------------------------------------------------------------------
# 8. Edge selection contract and JSON output
# ----------------------------------------------------------------------
def test_edge_cap_and_invalid_edges_raise_before_computing():
    regions = 12
    matrices = stack(8, seed=32, regions=regions)
    result = analyze(group1=matrices, test_type="one-sample", **FAST).inference
    pairs = [(i, j) for i in range(regions) for j in range(i + 1, regions)]
    assert len(pairs) > MAX_EDGES
    with pytest.raises(ValueError, match=f"at most {MAX_EDGES} edges"):
        subject_influence(result, edges=pairs[: MAX_EDGES + 1], group1=matrices)
    boundary = subject_influence(result, edges=pairs[:MAX_EDGES],
                                 group1=matrices)
    assert boundary["n_edges"] == MAX_EDGES

    with pytest.raises(ValueError, match="duplicates"):
        subject_influence(result, edges=[(1, 2), (2, 1)], group1=matrices)
    with pytest.raises(ValueError, match="diagonal"):
        subject_influence(result, edges=[(3, 3)], group1=matrices)
    with pytest.raises(ValueError, match="outside the ROI range"):
        subject_influence(result, edges=[(0, regions)], group1=matrices)
    with pytest.raises(ValueError, match="non-integer"):
        subject_influence(result, edges=[(0, 1.5)], group1=matrices)
    with pytest.raises(ValueError, match="must not be empty"):
        subject_influence(result, edges=[], group1=matrices)
    with pytest.raises(ValueError, match="explicit finite list"):
        subject_influence(result, edges=None, group1=matrices)


def test_record_cap_is_an_unavailable_report_not_a_quiet_refit():
    regions = 12
    matrices = stack(70, seed=40, regions=regions)
    result = analyze(group1=matrices, test_type="one-sample", **FAST).inference
    pairs = [(i, j) for i in range(regions) for j in range(i + 1, regions)]
    pairs = pairs[:MAX_EDGES]
    assert 70 * len(pairs) > MAX_RECORDS

    report = subject_influence(result, edges=pairs, group1=matrices)
    assert codes(report) == ["too_many_records"]
    assert report["per_subject_edge"] == []
    assert report["edges"] == []
    assert report["summary"] == {}

    smaller = subject_influence(result, edges=pairs[:8], group1=matrices)
    assert smaller["unavailable"] == []
    assert smaller["summary"]["n_subject_edge_records"] == 70 * 8


def test_report_is_json_serialisable_with_the_documented_schema():
    matrices = stack(8, seed=33)
    result = analyze(group1=matrices, test_type="one-sample", **FAST).inference
    report = subject_influence(result, edges=[(0, 1), (3, 4)], group1=matrices)

    decoded = json.loads(json.dumps(report))
    assert decoded["schema_version"] == 1
    for key in ("test_type", "kind", "units", "scale", "contrast", "stat_type",
                "tolerance", "inference_unit", "checks", "edges",
                "per_subject_edge", "subject_summary", "ranking", "summary",
                "unavailable", "limitations", "descriptive_only", "mad",
                "statuses"):
        assert key in decoded
    assert decoded["tolerance"] == CHECK_TOLERANCE
    assert decoded["inference_unit"]["inference_unit"] == "edge"
    assert {row["status"] for row in decoded["per_subject_edge"]} <= ALLOWED_STATUSES
    for row in decoded["per_subject_edge"]:
        for key in ("subject", "edge", "delta", "stat_loo", "df_loo", "n_loo",
                    "leverage", "mad", "status"):
            assert key in row
    for row in decoded["edges"]:
        for key in ("edge", "effect", "stat", "n", "variance", "df", "mad",
                    "status"):
            assert key in row
    assert len(decoded["per_subject_edge"]) == 8 * 2


def test_influence_module_is_reachable_without_touching_the_root_surface():
    from conninfpy import influence

    assert influence.subject_influence is subject_influence
    assert "subject_influence" not in conninfpy.__all__
    assert len(conninfpy.__all__) == 21


def test_component_level_result_is_labelled_as_not_per_edge():
    matrices = stack(9, seed=34)
    matrices[3, 1, 2] = matrices[3, 2, 1] = 1.6
    result = analyze(group1=matrices, test_type="one-sample", method="nbs",
                     n_permutations=19, use_mp=False, rng=0, input="z",
                     threshold=1.0).inference
    assert result.inference_unit == "component"
    report = subject_influence(result, edges=[(1, 2)], group1=matrices)
    assert report["unavailable"] == []
    context = report["inference_unit"]
    assert context["inference_unit"] == "component"
    assert "NOT independent edge discoveries" in context["note"]
