"""Bounded scientific regression references: tiny graphs, at most 19 permutations."""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

from conninfpy import analyze
from conninfpy._cnbs import block_p_values
from conninfpy.acceleration import _replace_sparse_tail, fit_gamma_tail, fit_gpd_tail
from conninfpy.glm_stats import compute_p_val_glm, compute_p_val_glm_multi
from conninfpy.pairwise_stats import compute_p_val
from conninfpy.tfnbs_score import _precompute_block_metadata, get_network_informed_tfnbs_score


def _data():
    edges = np.random.default_rng(23).normal(size=(8, 3)) + [0.7, -0.9, 0.1]
    return _matrices(edges), edges


def _matrices(edges):
    out = np.zeros((len(edges), 3, 3))
    ij = np.triu_indices(3, 1)
    out[:, ij[0], ij[1]] = edges
    out[:, ij[1], ij[0]] = edges
    return out


def _reference(edges, method, *, permutations=19, seed=7):
    """SciPy t statistics and independently replayed shared subject signs."""
    valid = np.sum(np.isfinite(edges), axis=0) >= 4
    obs = stats.ttest_1samp(edges[:, valid], 0, nan_policy="omit").statistic
    if method in ("bonferroni", "bh_fdr"):
        raw = stats.ttest_1samp(edges[:, valid], 0, nan_policy="omit").pvalue
    else:
        seeds = np.random.RandomState(seed).randint(0, 2**32 - 1, permutations, dtype=np.int64)
        null = np.array([
            stats.ttest_1samp(
                edges[:, valid] * np.random.RandomState(s).choice([1, -1], len(edges))[:, None],
                0, nan_policy="omit",
            ).statistic for s in seeds
        ])
        if method == "tstat":
            raw = (np.sum(np.abs(null).max(axis=1)[:, None] >= np.abs(obs), axis=0) + 1) / (permutations + 1)
        else:
            raw = (np.sum(np.abs(null) >= np.abs(obs), axis=0) + 1) / (permutations + 1)
    corrected = (np.minimum(raw * valid.sum(), 1) if method == "bonferroni"
                 else stats.false_discovery_control(raw) if "bh_fdr" in method else raw)
    return valid, obs, corrected


@pytest.mark.parametrize("method", ["bonferroni", "bh_fdr", "bh_fdr_perm", "tstat"])
@pytest.mark.parametrize("missing", [False, True])
def test_two_sided_edges_against_scipy_and_replayed_permutations(method, missing):
    _, edges = _data()
    if missing:
        edges[0, 0] = np.nan
        edges[:6, 2] = np.nan  # outside the tested family
    r = compute_p_val(_matrices(edges), test_type="one-sample", method=method,
                      missing="available_case" if missing else "raise", min_coverage=0.5,
                      two_tailed=True, n_permutations=19, rng=7, use_mp=False, acceleration=None)
    valid, signed, corrected = _reference(edges, method)
    ij = np.triu_indices(3, 1)
    for key, sign in (("positive", 1), ("negative", -1)):
        values = r[key][ij]
        np.testing.assert_allclose(values[valid], np.where(sign * signed > 0, corrected, 1))
        assert np.isnan(values[~valid]).all()
    assert r.tail_policy == "two-sided"
    assert r.inference_unit == "edge"
    assert r.correction == ("bh_fdr" if "bh_fdr" in method else "bonferroni"
                            if method == "bonferroni" else "permutation_max")


def test_one_edge_parametric_tail_changes_significance():
    edge = (np.arange(8) - 3.5) / np.std(np.arange(8), ddof=1) + 0.75
    y = np.zeros((8, 2, 2))
    y[:, 0, 1] = y[:, 1, 0] = edge
    one = compute_p_val(y, test_type="one-sample", method="bonferroni")
    two = compute_p_val(y, test_type="one-sample", method="bonferroni", two_tailed=True)
    np.testing.assert_allclose(two.positive[0, 1], stats.ttest_1samp(edge, 0).pvalue)
    assert one.positive[0, 1] < 0.05 < two.positive[0, 1]


@pytest.mark.parametrize("method", ["tfnbs", "ni_tfnbs", "fbc_tfnbs", "nbs", "cnbs"])
@pytest.mark.parametrize("missing", [False, True])
def test_enhancement_routes_have_explicit_tail_policy(method, missing):
    y, edges = _data()
    if missing:
        edges[0, 0] = np.nan
        y = _matrices(edges)
    r = compute_p_val(y, test_type="one-sample", method=method,
                      net_labels=np.array([0, 0, 1]), n=4, start_thres=0,
                      min_cluster_size=1, threshold=0.5, missing="available_case" if missing else "raise",
                      two_tailed=True, n_permutations=7, rng=5, use_mp=False, acceleration=None)
    assert r.tail_policy == "two-sided"
    assert np.isfinite(r.positive).all()
    if method == "cnbs":
        assert r.block_report["tail_policy"] == "two-sided"
        assert r.inference_unit == "block"


def test_grid_joint_maxima_match_separately_corrected_settings():
    y, _ = _data()
    opts = dict(test_type="one-sample", method="ni_tfnbs", net_labels=[0, 0, 1],
                n=4, start_thres=0, two_tailed=True, n_permutations=7,
                rng=4, use_mp=False, acceleration=None)
    grid = compute_p_val(y, e=[0.5, 1.0], h=[1.0, 2.0], **opts)
    for k, (e, h) in enumerate(zip([0.5, 1.0], [1.0, 2.0])):
        single = compute_p_val(y, e=e, h=h, **opts)
        for key in single:
            np.testing.assert_array_equal(grid[key][..., k], single[key])
        assert grid.select(k).two_tailed


@pytest.mark.parametrize("correction", ["bonferroni", "bh"])
def test_cnbs_corrects_absolute_block_family_once(correction):
    observed = np.array([3.0, -4.0, 0.5])
    null = np.random.default_rng(12).normal(size=(19, 3))
    maps, report = block_p_values({"positive": observed, "negative": -observed},
                                 {"positive": null, "negative": -null},
                                 np.array([0, 0, 1, 1]), 4,
                                 correction=correction, two_tailed=True)
    raw = (np.sum(np.abs(null) >= np.abs(observed), axis=0) + 1) / 20
    expected = np.minimum(raw * 3, 1) if correction == "bonferroni" else stats.false_discovery_control(raw)
    for k, block in enumerate(report["blocks"]):
        assert block["p_two_sided"] == pytest.approx(expected[k])
        assert block["p_positive"] == pytest.approx(expected[k] if observed[k] > 0 else 1)
        assert block["p_negative"] == pytest.approx(expected[k] if observed[k] < 0 else 1)
    assert np.all(np.diag(maps["positive"]) == 1)


@pytest.mark.parametrize("missing", [False, True])
def test_glm_cnbs_two_sided_bh_matches_block_family(missing):
    y, edges = _data()
    if missing:
        edges[0, 0] = np.nan
        y = _matrices(edges)
    r = compute_p_val_glm(y, design_matrix=np.ones((8, 1)), contrast=[1], method="cnbs",
                          net_labels=[0, 0, 1], cnbs_correction="bh", two_tailed=True,
                          missing="available_case" if missing else "raise", min_coverage=0.5,
                          n_permutations=19, rng=2, use_mp=False, acceleration=None)
    rows = r.block_report["blocks"]
    expected = stats.false_discovery_control(np.array([b["p_two_sided_uncorrected"] for b in rows]))
    np.testing.assert_allclose([b["p_two_sided"] for b in rows], expected)
    assert r.two_tailed


def test_multi_contrast_cnbs_carries_joint_tail_metadata():
    y, _ = _data()
    x = np.column_stack([np.ones(8), np.linspace(-1, 1, 8)])
    results = compute_p_val_glm_multi(y, x, {"mean": [1, 0], "slope": [0, 1]},
                                     method="cnbs", net_labels=[0, 0, 1], cnbs_correction="bh",
                                     two_tailed=True, n_permutations=7, rng=0,
                                     use_mp=False, acceleration=None)
    assert all(r.two_tailed and r.block_report["tail_policy"] == "two-sided"
               for r in results.values())


def test_unknown_method_options_raise_through_public_apis():
    y, _ = _data()
    with pytest.raises(TypeError, match="m_min"):
        compute_p_val(y, test_type="one-sample", m_min=20)
    with pytest.raises(TypeError, match="m_min"):
        analyze(group1=y, test_type="one-sample", input="z", m_min=20)


def test_ni_capacity_is_independent_of_active_edges_and_respects_eligibility():
    labels = np.zeros(4, dtype=int)
    _, small = _precompute_block_metadata(np.array([0]), np.array([1]), labels)
    _, full = _precompute_block_metadata(*np.triu_indices(4, 1), labels)
    np.testing.assert_allclose(small, full)
    assert small[0] == pytest.approx(np.sqrt(6))
    t = np.zeros((4, 4))
    t[0, 1] = t[1, 0] = 2
    mask = t > 0
    unrestricted = get_network_informed_tfnbs_score(t, labels, 1, 1, 4, start_thres=0)
    restricted = get_network_informed_tfnbs_score(t, labels, 1, 1, 4, start_thres=0, edge_mask=mask)
    np.testing.assert_allclose(restricted, unrestricted * np.sqrt(6))


def test_ni_between_and_within_capacities():
    labels = np.array([0, 0, 1, 1, 1])
    _, capacities = _precompute_block_metadata(np.array([0]), np.array([2]), labels)
    np.testing.assert_allclose(capacities[[0, 1, 3]], np.sqrt([1, 6, 3]))


@pytest.mark.parametrize("glm", [False, True])
def test_available_case_ni_uses_the_same_tested_capacity_in_every_map(glm):
    _, edges = _data()
    edges[:6, 2] = np.nan
    y = _matrices(edges)
    with patch("conninfpy.tfnbs_score._precompute_block_metadata",
               wraps=_precompute_block_metadata) as metadata:
        opts = dict(method="ni_tfnbs", net_labels=[0, 0, 0], n=4, start_thres=0,
                    missing="available_case", min_coverage=0.5, two_tailed=True,
                    n_permutations=7, rng=1, use_mp=False, acceleration=None)
        if glm:
            compute_p_val_glm(y, design_matrix=np.ones((8, 1)), contrast=[1], **opts)
        else:
            compute_p_val(y, test_type="one-sample", **opts)
    assert metadata.call_count > 0
    for call in metadata.call_args_list:
        mask = call.args[3]
        assert mask[np.triu_indices(3, 1)].tolist() == [True, True, False]
        assert _precompute_block_metadata(*call.args)[1][0] == pytest.approx(np.sqrt(2))


@pytest.mark.parametrize("method", ["tstat", "tfnbs", "ni_tfnbs", "nbs"])
def test_zero_edge_with_other_missing_values_is_never_significant(method):
    _, edges = _data()
    edges[:, 2] = 0
    edges[0, 0] = np.nan
    r = compute_p_val(_matrices(edges), test_type="one-sample", method=method,
                      net_labels=[0, 0, 0], missing="available_case", n=4, start_thres=0,
                      threshold=0.5, two_tailed=True, n_permutations=7, rng=0,
                      use_mp=False, acceleration=None)
    assert r.positive[1, 2] == 1
    assert r.negative[1, 2] == 1


def test_cnbs_metadata_reports_empirical_inference_even_if_acceleration_requested():
    y, _ = _data()
    r = compute_p_val(y, test_type="one-sample", method="cnbs", net_labels=[0, 0, 1],
                      n_permutations=7, rng=0, use_mp=False, acceleration="gpd")
    assert r.acceleration is None


def test_legacy_open_close_loader_keeps_undefined_correlations(tmp_path):
    from apps.utils.lab_loaders import OpenCloseLoader
    signals = np.random.default_rng(11).normal(size=(2, 8, 3))
    signals[:, :, 1] = 4
    path = tmp_path / "signals.npy"
    np.save(path, signals)
    with np.errstate(invalid="ignore", divide="ignore"):
        loaded = OpenCloseLoader(str(path), str(path)).load()
    assert np.isnan(loaded.data[:, 0, 1]).all()


def test_gpd_custom_cutoff_retains_below_threshold_values():
    null = np.arange(200, dtype=float)
    observed = np.array([188.0, 205.0])
    with patch("conninfpy.acceleration._fit_gpd_mom", return_value=(1.0, 0.0, True)), \
            patch("conninfpy.acceleration._anderson_darling_gpd", side_effect=[3, 3, 3, 3, 0]):
        p = fit_gpd_tail(null, observed, min_exceedances=25)
    assert p[0] == pytest.approx(13 / 201)
    assert p[1] == pytest.approx(0.05 * np.exp(-(205 - np.quantile(null, 0.95))))
    assert 0 < p[1] < 1 / 200


def test_gpd_bounded_endpoint_falls_back_to_empirical():
    null = np.arange(200, dtype=float)
    with patch("conninfpy.acceleration._fit_gpd_mom", return_value=(1.0, -0.5, True)), \
            patch("conninfpy.acceleration._anderson_darling_gpd", return_value=0):
        p = fit_gpd_tail(null, np.array([205.0]))
    assert p[0] == pytest.approx(1 / 201)


def test_gamma_can_extrapolate_below_permutation_resolution():
    null = np.arange(1, 201, dtype=float) ** 2
    observed = np.array([null.max() * 5])
    p = fit_gamma_tail(null, observed)
    a = 4 / stats.skew(null) ** 2
    b = np.sqrt(np.var(null, ddof=1) / a)
    expected = stats.gamma.sf(observed, a, loc=null.mean() - a * b, scale=b)
    np.testing.assert_allclose(p, expected)
    assert 0 < p[0] < 1 / len(null)


def test_tail_invalid_values_fallback_and_numerical_zero_is_positive():
    empirical = np.full(4, 1 / 201)
    p = _replace_sparse_tail(empirical, np.zeros(4), np.array([1e-8, 0, np.nan, np.inf]), 200, 10)
    np.testing.assert_allclose(p[[0, 2, 3]], [1e-8, 1 / 201, 1 / 201])
    assert p[1] == np.finfo(float).tiny


@pytest.mark.parametrize("kind,expected", [("correlation", "r"), ("fisher_z", "z"), (None, None)])
def test_app_primary_and_companion_dispatch_scale_and_tail(kind, expected):
    # Load the actual worker without importing/rendering the Streamlit UI.
    source = Path(__file__).resolve().parents[1] / "apps/components/design_inference.py"
    tree = ast.parse(source.read_text())
    names = {"_run_inference_background", "_method_plan_parameters"}
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
                        type_ignores=[])
    namespace = {}
    exec(compile(module, str(source), "exec"), namespace)
    worker = namespace["_run_inference_background"]
    args = {name: None for name in inspect.signature(worker).parameters}
    task = SimpleNamespace()
    args.update(task=task, data_kind=kind, test_type="one-sample", operator_choice="fbc_tfnbs",
                op_kwargs={"min_cluster_size": 20}, run_sensitivity=True, comp_method="fbc_tfnbs",
                comp_kwargs={"min_cluster_size": 22}, atlas=SimpleNamespace(network_index=lambda: [0, 0]),
                acceleration_choice="none")
    fake_result = SimpleNamespace(significant_edges=lambda **kw: [])
    with patch("conninfpy.analyze", return_value=fake_result) as run:
        worker(**args)
    assert task.status == "success"
    assert run.call_count == 2
    assert [c.kwargs["input"] for c in run.call_args_list] == [expected, expected]
    assert [c.kwargs["min_cluster_size"] for c in run.call_args_list] == [20, 22]
    assert all(c.kwargs["two_tailed"] for c in run.call_args_list)
    assert task.run_plan["tail_policy"] == "two-sided"


def test_statistic_sort_is_not_named_effect_size():
    y, _ = _data()
    r = compute_p_val(y, test_type="one-sample", method="bonferroni")
    table = r.significant_edges(sort="statistic", include_nonsig=True)
    assert np.all(np.diff(table.t_signed.abs()) <= 0)
    with pytest.warns(DeprecationWarning, match="test statistic"):
        legacy = r.significant_edges(sort="effect_size", include_nonsig=True)
    np.testing.assert_array_equal(table.edge_id, legacy.edge_id)
