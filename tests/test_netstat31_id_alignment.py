"""NETSTAT-31: optional subject IDs in the paired Python API.

Synthetic data only (small n, few regions, few permutations). The oracle for
every alignment test is the *same analysis on already ordered rows*: aligning
before inference makes the inference input bit-identical, so statistics, effect
maps and (same-seed) p-values must match exactly.
"""
from __future__ import annotations

import contextlib
import io
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from conninfpy import analyze, compute_p_val, compute_p_val_paired_glm, ResultBundle
from conninfpy._shared.alignment import (
    IdAlignmentError,
    alignment_record,
    paired_id_permutation,
    validate_subject_ids,
)
from conninfpy.cli import ConfigError, build_analysis, run
from conninfpy.loaders.base import LoadedDataset

FAST = {"method": "tstat", "n_permutations": 39, "use_mp": False, "rng": 7,
        "acceleration": None}
FAST_TWO_SIDED = dict(FAST, two_tailed=True)


def matrices(n: int, seed: int = 0, N: int = 4) -> np.ndarray:
    """Symmetric zero-diagonal (n, N, N) Fisher-z-like matrices."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.3, (n, N, N))
    x = (x + x.transpose(0, 2, 1)) / 2
    x[:, np.arange(N), np.arange(N)] = 0.0
    return x


def ids_for(n: int, start: int = 0):
    return [f"s{i:02d}" for i in range(start, start + n)]


def shuffled(n: int, seed: int = 3):
    """A non-identity row permutation plus its ID vector."""
    perm = np.random.default_rng(seed).permutation(n)
    ids = ids_for(n)
    return perm, ids, [ids[i] for i in perm]


def assert_same_result(a, b):
    """Exact equality of everything the caller observes from an aligned call."""
    for field in ("positive", "negative", "stat_positive", "stat_negative",
                  "stat_signed", "beta"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    np.testing.assert_array_equal(a.effect_map(), b.effect_map())


# ---------------------------------------------------------------------------
# AC#1 — the ID contract
# ---------------------------------------------------------------------------

def test_bijection_aligns_b_and_reproduces_the_ordered_call_exactly():
    n = 8
    a, b = matrices(n, 1), matrices(n, 2)
    perm, ids, ids_b = shuffled(n)
    ordered = analyze(group1=a, group2=b, test_type="paired", input="z", **FAST).inference
    aligned = analyze(group1=a, group2=b[perm], test_type="paired",
                      ids_group1=ids, ids_group2=ids_b, input="z", **FAST).inference
    assert_same_result(ordered, aligned)
    record = aligned.design["paired_alignment"]
    assert record["applied"] is True
    assert record["reference"].startswith("group1")
    assert record["n_subjects"] == n
    assert record["identity_permutation"] is False
    assert record["permutation_b_to_a"] == np.argsort(perm).tolist()
    assert record["reordered_inputs"] == ["group2"]
    # Delta z is the subject-wise mean difference of the *aligned* rows.
    np.testing.assert_array_equal(
        aligned.effect_map(), (b - a).mean(axis=0)
    )


def test_identity_ids_record_the_check_without_permuting():
    n = 6
    a, b = matrices(n, 1), matrices(n, 2)
    ids = ids_for(n)
    ordered = analyze(group1=a, group2=b, test_type="paired", input="z", **FAST).inference
    checked = analyze(group1=a, group2=b, test_type="paired",
                      ids_group1=ids, ids_group2=ids, input="z", **FAST).inference
    assert_same_result(ordered, checked)
    assert checked.design["paired_alignment"]["identity_permutation"] is True
    assert checked.design["paired_alignment"]["permutation_b_to_a"] == list(range(n))


def test_no_ids_keeps_the_documented_positional_behaviour():
    n = 6
    a, b = matrices(n, 1), matrices(n, 2)
    ids = ids_for(n)
    positional = analyze(group1=a, group2=b, test_type="paired", input="z", **FAST).inference
    explicit_none = analyze(group1=a, group2=b, test_type="paired", input="z",
                            ids_group1=None, ids_group2=None, **FAST).inference
    assert_same_result(positional, explicit_none)
    assert "paired_alignment" not in positional.design
    # A shuffled B without IDs is still (wrongly) paired positionally — the
    # documented positional contract, unchanged by this feature.
    perm, _, _ = shuffled(n)
    positional_shuffled = analyze(group1=a, group2=b[perm], test_type="paired",
                                  input="z", **FAST).inference
    assert not np.array_equal(positional["positive"], positional_shuffled["positive"]) or \
        not np.array_equal(positional.effect_map(), positional_shuffled.effect_map())


def test_one_sided_ids_are_a_clear_error():
    n = 6
    a, b = matrices(n, 1), matrices(n, 2)
    ids = ids_for(n)
    with pytest.raises(ValueError, match="one-sided"):
        analyze(group1=a, group2=b, test_type="paired", ids_group1=ids, input="z", **FAST)
    with pytest.raises(ValueError, match="one-sided"):
        analyze(group1=a, group2=b, test_type="paired", ids_group2=ids, input="z", **FAST)
    with pytest.raises(IdAlignmentError, match="one-sided"):
        paired_id_permutation(ids, None)


def test_missing_pair_is_a_clear_error():
    n = 6
    a, b = matrices(n, 1), matrices(n, 2)
    ids, ids_b = ids_for(n), ids_for(n, 1)  # s06 present in A only, s01 in B only
    with pytest.raises(ValueError, match="only one condition"):
        analyze(group1=a, group2=b, test_type="paired",
                ids_group1=ids, ids_group2=ids_b, input="z", **FAST)


def test_duplicate_ids_within_a_condition_are_a_clear_error():
    n = 6
    a, b = matrices(n, 1), matrices(n, 2)
    ids, ids_b = ids_for(n), ids_for(n)
    ids_b = [ids_b[0]] + ids_b[:-1]
    with pytest.raises(ValueError, match="repeats subject ID"):
        analyze(group1=a, group2=b, test_type="paired",
                ids_group1=ids, ids_group2=ids_b, input="z", **FAST)
    with pytest.raises(IdAlignmentError, match="repeats subject ID"):
        validate_subject_ids(ids_b, "ids_group2")


def test_id_length_must_match_the_subject_axis():
    n = 6
    a, b = matrices(n, 1), matrices(n, 2)
    ids = ids_for(n)
    with pytest.raises(ValueError, match="subject axis"):
        analyze(group1=a, group2=b, test_type="paired",
                ids_group1=ids, ids_group2=ids[:-1], input="z", **FAST)
    with pytest.raises(ValueError, match="subject axis"):
        analyze(group1=a, group2=b, test_type="paired",
                ids_group1=ids[:-1], ids_group2=ids, input="z", **FAST)


def test_shape_mismatch_is_a_clear_error():
    a, b = matrices(6, 1), matrices(5, 2)
    with pytest.raises(ValueError, match="same subjects"):
        analyze(group1=a, group2=b, test_type="paired", input="z", **FAST)
    with pytest.raises(ValueError, match="shapes"):
        analyze(group1=a, group2=matrices(6, 2, N=3), test_type="paired",
                ids_group1=ids_for(6), ids_group2=ids_for(6), input="z", **FAST)


@pytest.mark.parametrize("bad", [
    [None] + ids_for(5),
    [np.nan] + ids_for(5),
    [""] + ids_for(5),
    ["   "] + ids_for(5),
    [["nested"]] + ids_for(5),
    "s00",
])
def test_invalid_ids_are_rejected(bad):
    with pytest.raises(IdAlignmentError):
        validate_subject_ids(bad, "ids_group1")


def test_matching_does_not_normalise_or_coerce_types():
    # 1 and "1" are different opaque IDs; no stringification happens.
    with pytest.raises(IdAlignmentError, match="only one condition"):
        paired_id_permutation([1, 2, 3], ["1", "2", "3"])
    # Exact values match, including non-ASCII IDs.
    perm = paired_id_permutation(["суб02", "суб01"], ["суб01", "суб02"])
    np.testing.assert_array_equal(perm, np.array([1, 0], dtype=np.intp))


@pytest.mark.parametrize("kwargs,test_type", [
    (dict(Y=None, group1=matrices(6, 1), group2=matrices(6, 2)), "two-sample"),
    (dict(Y=None, group1=matrices(6, 1)), "one-sample"),
    (dict(Y=matrices(6, 1), interest=np.arange(6.0)), None),
])
def test_ids_are_rejected_outside_paired_paths(kwargs, test_type):
    n = 6
    ids = ids_for(n)
    fed = {k: v for k, v in kwargs.items() if v is not None}
    if test_type is not None:
        fed["test_type"] = test_type
    with pytest.raises(ValueError, match="only valid for the paired design"):
        analyze(**fed, ids_group1=ids, ids_group2=ids, input="z", **FAST)
    if test_type is not None:
        with pytest.raises(ValueError, match="only valid for test_type='paired'"):
            compute_p_val(fed["group1"], fed.get("group2"), test_type=test_type,
                          ids_group1=ids, ids_group2=ids, **FAST)


# ---------------------------------------------------------------------------
# AC#2 — one permutation moves every condition-B input; A-order stays put
# ---------------------------------------------------------------------------

def test_all_condition_b_inputs_move_together_and_a_order_does_not():
    n = 10
    rng = np.random.default_rng(11)
    a, b = matrices(n, 1), matrices(n, 2)
    age = rng.normal(30, 5, n)          # subject-level interest, A order
    motion = rng.normal(0, 1, n)        # subject-level confound, A order
    fd_a, fd_b = rng.normal(0, .1, n), rng.normal(0, .1, n)  # condition-specific
    sites = np.array(["a", "b"] * (n // 2), dtype=object)     # A order
    perm_a = np.random.default_rng(21).permutation(n)         # consistent A shuffle
    perm_b, _, ids_b = shuffled(n, seed=5)
    ids = ids_for(n)

    def call(rows_a, rows_b, ids_a, ids_b_):
        return analyze(
            group1=a[rows_a], group2=b[rows_b], test_type="paired",
            interest=age[rows_a], confounds=motion[rows_a],
            confounds_group1=fd_a[rows_a], confounds_group2=fd_b[rows_b],
            sites=sites[rows_a], ids_group1=ids_a, ids_group2=ids_b_,
            input="z", **FAST).inference

    ordered = call(np.arange(n), np.arange(n), None, None)
    # A rows (with their A-order covariates) and B rows shuffled independently.
    shuffled_all = call(perm_a, perm_b, [ids[i] for i in perm_a], ids_b)
    # With A's (permuted) order held fixed — the reference order by contract —
    # the whole result, p-values included, is exactly the call where the two
    # conditions are already consistently ordered: alignment happens before
    # inference, so nothing but A's own row order survives.
    reference = call(perm_a, perm_a, [ids[i] for i in perm_a], [ids[i] for i in perm_a])
    assert reference.design["paired_alignment"]["identity_permutation"] is True
    assert shuffled_all.design["paired_alignment"]["identity_permutation"] is False
    assert_same_result(reference, shuffled_all)
    # The identity-A-order run answers the same data-set question; it can differ
    # only by floating-point round-off from A's different row order (BLAS), so
    # compare with a tight tolerance rather than bit-for-bit.
    np.testing.assert_allclose(ordered.stat_signed, shuffled_all.stat_signed, atol=1e-12)
    np.testing.assert_allclose(ordered.effect_map(), shuffled_all.effect_map(), atol=1e-12)
    record = shuffled_all.design["paired_alignment"]
    assert record["reordered_inputs"] == ["group2", "confounds_group2"]
    assert set(record["a_order_inputs_validated_only"]) == {
        "interest", "confounds", "sites", "confounds_group1"}
    # The paired-difference details carry the same alignment record.
    assert shuffled_all.design["paired_difference"]["alignment"] == record


def test_missing_value_masks_travel_with_their_condition():
    n = 8
    a, b = matrices(n, 1), matrices(n, 2)
    b = b.copy()
    b[:, 0, 1] = b[:, 1, 0] = np.nan
    b[[1, 4], 0, 1] = b[[1, 4], 1, 0] = 0.2      # observed for those subjects
    perm, ids, ids_b = shuffled(n, seed=9)
    ordered = analyze(group1=a, group2=b, test_type="paired", input="z",
                      missing="available_case", **FAST).inference
    aligned = analyze(group1=a, group2=b[perm], test_type="paired",
                      ids_group1=ids, ids_group2=ids_b, input="z",
                      missing="available_case", **FAST).inference
    assert_same_result(ordered, aligned)
    np.testing.assert_array_equal(
        ordered.missing_report["tested_mask"], aligned.missing_report["tested_mask"]
    )


def test_edge_mask_is_invariant_and_a_order_lengths_are_validated():
    n = 8
    a, b = matrices(n, 1), matrices(n, 2)
    perm, ids, ids_b = shuffled(n, seed=13)
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 1] = mask[1, 0] = True
    ordered = analyze(group1=a, group2=b, test_type="paired", input="z",
                      edge_mask=mask, **FAST).inference
    aligned = analyze(group1=a, group2=b[perm], test_type="paired",
                      ids_group1=ids, ids_group2=ids_b, input="z",
                      edge_mask=mask, **FAST).inference
    assert_same_result(ordered, aligned)
    # An A-order covariate of the wrong length is a clear error, not a silent
    # broadcast or a permuted guess.
    with pytest.raises(ValueError, match="condition A's row order"):
        analyze(group1=a, group2=b, test_type="paired", interest=np.arange(n - 1.0),
                ids_group1=ids, ids_group2=ids, input="z", **FAST)
    with pytest.raises(ValueError, match="condition A's row order"):
        analyze(group1=a, group2=b, test_type="paired", confounds_group1=np.ones(n - 1),
                confounds_group2=np.ones(n), ids_group1=ids, ids_group2=ids,
                input="z", **FAST)


# ---------------------------------------------------------------------------
# AC#3 — provenance/flags, exactness, bundle round trip
# ---------------------------------------------------------------------------

def test_flags_and_provenance_record_the_alignment():
    n = 6
    a, b = matrices(n, 1), matrices(n, 2)
    perm, ids, ids_b = shuffled(n, seed=2)
    out = analyze(group1=a, group2=b[perm], test_type="paired",
                  ids_group1=ids, ids_group2=ids_b, input="z", **FAST)
    assert any("paired subject IDs aligned" in flag for flag in out.flags)
    record = out.inference.design["paired_alignment"]
    assert record["n_subjects"] == n
    assert record["permutation_b_to_a"] == np.argsort(perm).tolist()
    assert out.inference.analysis_provenance["paired_alignment"] == record
    identity = analyze(group1=a, group2=b, test_type="paired",
                       ids_group1=ids, ids_group2=ids, input="z", **FAST)
    assert any("already followed" in flag for flag in identity.flags)


def test_alignment_record_survives_the_bundle_round_trip(tmp_path):
    n = 6
    a, b = matrices(n, 1), matrices(n, 2)
    perm, ids, ids_b = shuffled(n, seed=4)
    out = analyze(group1=a, group2=b[perm], test_type="paired",
                  ids_group1=ids, ids_group2=ids_b, input="z", **FAST).inference
    record = out.design["paired_alignment"]
    bundle = out.to_bundle()
    assert bundle.metadata["design"]["paired_alignment"] == record
    path = tmp_path / "bundle.pkl"
    bundle.save(path)
    loaded = ResultBundle.load(path)
    assert loaded.metadata["design"]["paired_alignment"] == record
    assert any("paired subject IDs aligned" in flag for flag in loaded.metadata["flags"])
    np.testing.assert_array_equal(loaded.effect, out.effect_map())


def test_observed_statistics_are_bit_identical_after_alignment():
    n = 9
    a, b = matrices(n, 1), matrices(n, 2)
    perm, ids, ids_b = shuffled(n, seed=6)
    ordered = analyze(group1=a, group2=b, test_type="paired", input="z", **FAST).inference
    aligned = analyze(group1=a, group2=b[perm], test_type="paired",
                      ids_group1=ids, ids_group2=ids_b, input="z", **FAST).inference
    # Pre-inference alignment: the t maps and every p-value agree exactly.
    np.testing.assert_array_equal(ordered.stat_signed, aligned.stat_signed)
    np.testing.assert_array_equal(ordered["positive"], aligned["positive"])
    np.testing.assert_array_equal(ordered["negative"], aligned["negative"])


# ---------------------------------------------------------------------------
# AC#4 — NETSTAT-16 designs, low-level entry points, CLI equivalence
# ---------------------------------------------------------------------------

def test_paired_interest_with_interactions_and_condition_confounds():
    n = 10
    rng = np.random.default_rng(17)
    a, b = matrices(n, 1), matrices(n, 2)
    listed, rating = rng.normal(size=n), rng.normal(size=n)
    fd_a, fd_b = rng.normal(0, .1, n), rng.normal(0, .1, n)
    perm_a = np.random.default_rng(31).permutation(n)
    perm_b, ids, ids_b = shuffled(n, seed=8)

    def call(rows_a, rows_b, ids_a, ids_b_):
        return analyze(
            group1=a[rows_a], group2=b[rows_b], test_type="paired",
            interest={"listed": listed[rows_a], "rating": rating[rows_a]},
            interactions=[("listed", "rating")],
            confounds_group1=fd_a[rows_a], confounds_group2=fd_b[rows_b],
            ids_group1=ids_a, ids_group2=ids_b_, input="z", **FAST)

    ordered = call(np.arange(n), np.arange(n), None, None)
    aligned = call(perm_a, perm_b, [ids[i] for i in perm_a], ids_b)
    reference = call(perm_a, perm_a, [ids[i] for i in perm_a], [ids[i] for i in perm_a])
    assert set(aligned) == {"listed", "rating", "listed:rating"}
    for name in ordered:
        # With A's row order fixed the full result is exact (alignment before
        # inference); against the identity-A-order run only BLAS round-off from
        # A's row order can differ.
        assert_same_result(reference[name].inference, aligned[name].inference)
        np.testing.assert_allclose(ordered[name].inference.stat_signed,
                                   aligned[name].inference.stat_signed, atol=1e-12)
        np.testing.assert_allclose(ordered[name].inference.effect_map(),
                                   aligned[name].inference.effect_map(), atol=1e-12)
    # NETSTAT-16: the condition-confound difference is still B - A per subject.
    assert aligned["listed"].inference.design[
        "paired_difference"]["condition_confound_transform"] == "B - A"
    np.testing.assert_array_equal(
        aligned["listed:rating"].inference.design["paired_difference"]["alignment"],
        aligned["listed"].inference.design["paired_alignment"],
    )


def test_paired_glm_path_accepts_ids():
    n = 9
    rng = np.random.default_rng(23)
    a, b = matrices(n, 1), matrices(n, 2)
    fd_a, fd_b = rng.normal(0, .1, n), rng.normal(0, .1, n)
    perm, ids, ids_b = shuffled(n, seed=12)
    ordered = analyze(group1=a, group2=b, test_type="paired",
                      confounds_group1=fd_a, confounds_group2=fd_b,
                      input="z", **FAST).inference
    aligned = analyze(group1=a, group2=b[perm], test_type="paired",
                      confounds_group1=fd_a, confounds_group2=fd_b[perm],
                      ids_group1=ids, ids_group2=ids_b, input="z", **FAST).inference
    assert_same_result(ordered, aligned)
    assert aligned.design["paired_alignment"]["reordered_inputs"] == [
        "group2", "confounds_group2"]


def test_low_level_entry_points_accept_ids():
    n = 8
    rng = np.random.default_rng(29)
    a, b = matrices(n, 1), matrices(n, 2)
    ids = ids_for(n)
    perm = np.random.default_rng(3).permutation(n)
    ids_b = [ids[i] for i in perm]
    t_ordered = compute_p_val(a, b, test_type="paired", **FAST)
    t_aligned = compute_p_val(a, b[perm], test_type="paired",
                              ids_group1=ids, ids_group2=ids_b, **FAST)
    assert_same_result(t_ordered, t_aligned)
    fd_a, fd_b = rng.normal(0, .1, n), rng.normal(0, .1, n)
    g_ordered = compute_p_val_paired_glm(a, b, confounds_A=fd_a, confounds_B=fd_b, **FAST)
    g_aligned = compute_p_val_paired_glm(a, b[perm], confounds_A=fd_a,
                                         confounds_B=fd_b[perm],
                                         ids_A=ids, ids_B=ids_b, **FAST)
    assert_same_result(g_ordered, g_aligned)
    assert g_aligned.design["paired_alignment"]["reference"] == "Y_A"
    np.testing.assert_array_equal(g_aligned.beta, g_ordered.beta)
    # No-confound route also records the alignment.
    plain = compute_p_val_paired_glm(a, b[perm], ids_A=ids, ids_B=ids_b, **FAST)
    assert plain.design["paired_alignment"]["n_subjects"] == n
    with pytest.raises(ValueError, match="only valid for test_type='paired'"):
        compute_p_val(a, b, test_type="two-sample", ids_group1=ids,
                      ids_group2=ids_b, **FAST)
    with pytest.raises(ValueError, match="one-sided"):
        compute_p_val_paired_glm(a, b, ids_A=ids, **FAST)


def _paired_dataset(n: int = 8, seed: int = 41):
    """Paired phenotype (rows shuffled) + synthetic data, as the CLI sees it."""
    rng = np.random.default_rng(seed)
    pheno = pd.DataFrame({
        "subject_id": ids_for(n) * 2,
        "condition": ["rest"] * n + ["task"] * n,
        "age": list(rng.normal(35, 6, n)) * 2,          # subject-level interest
        "site": list(np.repeat(["a", "b"], n // 2)) * 2,  # subject-level site
        "fd": rng.normal(0.2, 0.05, 2 * n),             # condition-specific
    })
    order = rng.permutation(2 * n)
    pheno = pheno.iloc[order].reset_index(drop=True)
    data = matrices(2 * n, seed)
    n_rest = (pheno["condition"] == "rest").to_numpy()
    return LoadedDataset(data=data, pheno=pheno, data_kind="fisher_z"), data, pheno, n_rest


def test_cli_matches_the_api_on_a_shuffled_phenotype(tmp_path):
    n = 8
    dataset, data, pheno, n_rest = _paired_dataset(n)
    design = {"type": "paired", "condition": "condition", "levels": ["rest", "task"],
              "interest": ["age"], "confounds": ["fd"], "sites": "site"}
    np.save(tmp_path / "conn.npy", data)
    pheno.to_csv(tmp_path / "pheno.csv", index=False)
    (tmp_path / "study.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "loader": "NumpyLoader",
        "paths": {"data_path": "conn.npy", "pheno_path": "pheno.csv"},
    }))
    config = {"data": "study.yaml", "input": "z", "design": design,
              "inference": dict(FAST), "output": "out"}
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(config))
    with contextlib.redirect_stdout(io.StringIO()):
        written = run(path)
    # The same study, loaded and driven exactly as run() does it, with the IDs
    # the CLI derived from the (shuffled) subject column.
    from conninfpy.loaders import load
    reloaded = load(tmp_path / "study.yaml")
    kwargs, summary = build_analysis(reloaded, {"design": design, "input": "z"})
    assert summary["n_subjects_used"] == n
    api = analyze(**kwargs, **FAST_TWO_SIDED).inference
    assert api.design["paired_alignment"]["n_subjects"] == n
    with np.load(written["pvals"]) as z:
        np.testing.assert_array_equal(z["positive"], api["positive"])
        np.testing.assert_array_equal(z["negative"], api["negative"])
        np.testing.assert_array_equal(z["effect"], api.effect_map())
    prov = json.loads(written["provenance"].read_text())
    assert prov["data"]["n_subjects_used"] == n
    bundle = ResultBundle.load(written["bundle"])
    record = bundle.metadata["design"]["paired_alignment"]
    assert record["n_subjects"] == n and record["identity_permutation"] is True
    assert record["reordered_inputs"] == ["group2", "confounds_group2"]


def test_cli_paired_id_errors_keep_their_messages():
    n = 6
    pheno = pd.DataFrame({
        "subject_id": ids_for(n) * 2,
        "condition": ["rest"] * n + ["task"] * n,
        "fd": np.zeros(2 * n),
    })
    dataset = LoadedDataset(data=matrices(2 * n, 0), pheno=pheno, data_kind="fisher_z")
    design = {"type": "paired", "condition": "condition", "levels": ["rest", "task"],
              "confounds": ["fd"]}
    missing = LoadedDataset(data=dataset.data[:-1],
                            pheno=pheno.iloc[:-1].reset_index(drop=True),
                            data_kind="fisher_z")
    with pytest.raises(ConfigError, match="missing pair"):
        build_analysis(missing, {"design": design})
    duplicate = LoadedDataset(data=dataset.data, pheno=pheno.copy(), data_kind="fisher_z")
    duplicate.pheno.loc[0, "subject_id"] = duplicate.pheno.loc[1, "subject_id"]
    with pytest.raises(ConfigError, match="repeats"):
        build_analysis(duplicate, {"design": design})
