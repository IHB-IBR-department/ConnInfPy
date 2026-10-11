"""Bounded serial/process equivalence through every permutation task family."""

import multiprocessing
from unittest.mock import patch

import numpy as np
import pytest

from conninfpy import analyze
from conninfpy.enhance.nbs import nbs_bct
from conninfpy.results import InferenceResult, OmnibusInferenceResult
from conninfpy.stats import _workers
from conninfpy.stats.glm import (
    compute_p_val_glm,
    compute_p_val_glm_multi,
    compute_p_val_paired_glm,
)
from conninfpy.stats.pairwise import (
    _compute_t_stat_ind,
    compute_null_dist,
    compute_p_val,
    compute_t_stat_diff,
)

METHODS = ("tstat", "tfnbs", "nbs", "cnbs", "ni_tfnbs", "fbc_tfnbs", "bh_fdr_perm")
START_METHODS = [
    m for m in ("spawn", "forkserver") if m in multiprocessing.get_all_start_methods()
]
CASES = (
    [f"pairwise/{m}" for m in METHODS]
    + [
        "one_sample/tstat",
        "one_sample/tfnbs",
        "two_sample/tstat",
        "two_sample/tfnbs",
        "two_sample/bh_fdr_perm",
        "grid/tfnbs",
        "grid/ni_tfnbs",
        "grid/fbc_tfnbs",
    ]
    + [f"missing_pairwise/{m}" for m in METHODS]
    + [
        "missing_two_sample/tstat",
        "missing_two_sample/bh_fdr_perm",
        "missing_one_sample/tfnbs",
    ]
    + [f"glm/{m}" for m in METHODS if m != "bh_fdr_perm"]
    + [f"missing_glm/{m}" for m in METHODS if m != "bh_fdr_perm"]
    + [
        "glm_f/tfnbs",
        "glm_beta/tstat",
        "paired_glm/ni_tfnbs",
        "missing_paired_glm/tfnbs",
        "multi_glm/tfnbs",
        "multi_glm/cnbs",
        "missing_multi_glm/ni_tfnbs",
        "analyze_combat/tfnbs",
        "analyze_sites/ni_tfnbs",
        "accelerated_pairwise_gpd/tstat",
        "accelerated_pairwise_gamma/tfnbs",
        "accelerated_glm_gpd/tstat",
        "accelerated_glm_gamma/tfnbs",
    ]
    + [f"legacy_nbs/{tt}" for tt in ("paired", "one-sample", "two-sample")]
    + [f"default_null/{tt}" for tt in ("paired", "one-sample", "two-sample")]
)


def _example():
    rg = np.random.default_rng(704)
    y = rg.normal(0, 0.2, (24, 6, 6))
    y = (y + y.transpose(0, 2, 1)) / 2
    for matrix in y:
        np.fill_diagonal(matrix, 0)
    interest = rg.normal(size=24)
    nuisance = rg.normal(size=24)
    y[:, 0, 1] += 0.12 * interest
    y[:, 1, 0] = y[:, 0, 1]
    a, b = y[:12].copy(), y[12:].copy()
    for i, j in ((0, 1), (0, 2), (1, 2)):
        b[:, i, j] += 0.2
        b[:, j, i] = b[:, i, j]
    sites = np.tile(np.repeat(["a", "b"], 6), 2)
    return y, a, b, interest, nuisance, sites


def _missing(y):
    y = y.copy()
    y[0, 2, :] = y[0, :, 2] = np.nan
    y[1, 3, :] = y[1, :, 3] = np.nan
    y[0, 2, 2] = y[1, 3, 3] = 0.0
    return y


def _run(case, use_mp):
    family, method = case.split("/")
    y, a, b, x, nuisance, sites = _example()
    kw = dict(n_permutations=9, rng=41, n_processes=2, use_mp=use_mp)
    labels = np.repeat([0, 1], 3)
    if family == "default_null":
        g1, g2 = (b - a, None) if method == "one-sample" else (a, b)
        default = compute_null_dist(g1, g2, test_type=method, **kw)
        func = _compute_t_stat_ind if method == "two-sample" else compute_t_stat_diff
        explicit = compute_null_dist(
            g1, g2, func=func, test_type=method, **{**kw, "use_mp": False}
        )
        _assert_equal(default, explicit)
        return default
    if family == "legacy_nbs":
        g1, g2 = (b - a, None) if method == "one-sample" else (a, b)
        return nbs_bct(g1, g2, test_type=method, **kw)
    kw.update(method=method, n=3)
    if family.startswith("accelerated"):
        kw.update(acceleration=family.rsplit("_", 1)[1], n_permutations=49)
        family = "pairwise" if "pairwise" in family else "glm"
    if method in ("cnbs", "ni_tfnbs", "fbc_tfnbs"):
        kw["net_labels"] = labels
    if "missing" in family:
        y, a = _missing(y), _missing(a)
        kw["missing"] = "available_case"
    if family in (
        "pairwise",
        "grid",
        "missing_pairwise",
        "two_sample",
        "one_sample",
        "missing_two_sample",
        "missing_one_sample",
    ):
        test_type = "paired"
        if "two_sample" in family:
            test_type = "two-sample"
            kw["strata"] = sites
        if "one_sample" in family:
            a, b, test_type = b - a, None, "one-sample"
        if family == "grid":
            kw.update(e=[0.3, 0.5], h=[2.0, 3.0])
        return compute_p_val(a, b, test_type=test_type, two_tailed=True, **kw)
    if family in ("analyze_combat", "analyze_sites"):
        # Exercise the orchestration routes around the same edge-wise GLM.
        return analyze(
            Y=y,
            interest=x,
            confounds=nuisance[:, None],
            sites=sites,
            harmonize=(
                "combat_site_dummies_glm"
                if family == "analyze_combat"
                else "site_dummies_glm"
            ),
            input="z",
            **kw,
        ).inference
    if "paired_glm" in family:
        return compute_p_val_paired_glm(
            a, b, confounds_A=np.zeros((12, 1)), confounds_B=nuisance[:12, None], **kw
        )
    design = np.column_stack([np.ones(24), x, nuisance])
    if "multi_glm" in family:
        if method == "tfnbs":
            kw.update(e=[0.3, 0.5], h=[2.0, 3.0])
        return compute_p_val_glm_multi(
            y,
            design,
            {"x": np.array([0.0, 1.0, 0.0]), "nuisance": np.array([0.0, 0.0, 1.0])},
            strata=sites,
            two_tailed=True,
            **kw,
        )
    contrast = (
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        if family == "glm_f"
        else np.array([0.0, 1.0, 0.0])
    )
    stat_type = (
        "fstat" if family == "glm_f" else "beta" if family == "glm_beta" else "tstat"
    )
    return compute_p_val_glm(
        y,
        design_matrix=design,
        contrast=contrast,
        stat_type=stat_type,
        strata=sites,
        two_tailed=True,
        **kw,
    )


def _assert_equal(actual, expected):
    if isinstance(expected, (InferenceResult, OmnibusInferenceResult)):
        assert type(actual) is type(expected)
        _assert_equal(dict(actual), dict(expected))
        # Ignore timing; compare inferential outputs, effects and provenance.
        for name in (
            "stat_positive",
            "stat_negative",
            "stat_omnibus",
            "beta",
            "design",
            "block_report",
            "missing_report",
            "method",
            "n_permutations",
            "two_tailed",
            "stat_type",
            "e_grid",
            "h_grid",
            "harmonized",
            "strata_provided",
            "combat_diagnostics",
        ):
            _assert_equal(getattr(actual, name, None), getattr(expected, name, None))
    elif isinstance(expected, (tuple, list)):
        for a, b in zip(actual, expected, strict=True):
            _assert_equal(a, b)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, np.ndarray):
        np.testing.assert_allclose(
            actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True
        )
    else:
        assert actual == expected


@pytest.mark.parametrize("method", ["bonferroni", "bh_fdr"])
def test_parametric_methods_do_not_create_processes(method):
    y, a, b, x, nuisance, _ = _example()
    with patch.object(_workers.multiprocessing, "get_context") as factory:
        compute_p_val(a, b, method=method)
        with pytest.raises(ValueError, match="not supported in the GLM pipeline"):
            compute_p_val_glm(y, interest=x, confounds=nuisance[:, None], method=method)
    factory.assert_not_called()


@pytest.mark.parametrize("start_method", START_METHODS)
@pytest.mark.parametrize("case", CASES)
def test_inference_serial_parallel_equivalence(case, start_method):
    serial = _run(case, False)
    ctx = multiprocessing.get_context(start_method)
    with patch.object(_workers.multiprocessing, "get_context", return_value=ctx):
        parallel = _run(case, True)
    _assert_equal(parallel, serial)
