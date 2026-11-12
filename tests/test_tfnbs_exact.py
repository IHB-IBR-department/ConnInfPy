"""Small independent checks for exact TFNBS integration and provenance."""
import pickle
import warnings

import numpy as np
import pytest
from scipy.stats import f

from conninfpy.enhance.tfnbs import get_tfnbs_score
from conninfpy.enhance.ni_tfnbs import get_network_informed_tfnbs_score
from conninfpy.enhance.fbc_tfnbs import get_fbc_tfnbs_score
from conninfpy.stats.glm import compute_p_val_glm


def _oracle(matrix, e, h, start, mode="graph", labels=None, mask=None,
            normalization="sqrt", minimum=1):
    """Enumerate open intervals and find graph components by plain BFS."""
    mat = np.round(matrix, 10)
    n = len(mat)
    symmetric = np.allclose(mat, mat.T, rtol=0, atol=1e-8 * max(1, np.nanmax(np.abs(mat))))
    entries = [(i, j, mat[i, j]) for i in range(n) for j in range(n)
               if i != j and (not symmetric or i < j) and mat[i, j] > start
               and (mask is None or mask[i, j])]
    out = np.zeros((n, n))
    levels = sorted({start, *(w for _, _, w in entries)})
    def block(i, j):
        return tuple(sorted((labels[i], labels[j])))
    for a, b in zip(levels, levels[1:]):
        active = [(i, j) for i, j, w in entries if w >= b]
        for i, j in active:
            if mode == "graph":
                reached = {i}
                while True:
                    new = reached | {v for x, y in active for u, v in ((x, y), (y, x)) if u in reached}
                    if new == reached:
                        break
                    reached = new
                support = sum(x in reached and y in reached for x, y in active)
            else:
                key = block(i, j)
                count = sum(block(x, y) == key for x, y in active)
                if mode == "fbc":
                    support = count if count >= minimum else 0
                else:
                    capacity = sum(block(x, y) == key and (mask is None or mask[x, y])
                                   for x in range(n) for y in range(x + 1, n))
                    support = count / ({"sqrt": np.sqrt(capacity), "linear": capacity,
                                        "none": 1}[normalization])
            if support:
                increment = support**e * (b**(h + 1) - a**(h + 1)) / (h + 1)
                out[i, j] += increment
                if symmetric:
                    out[j, i] += increment
    return out


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("e,h", [(0, 0), (0.4, 3), (1.3, 0.5)])
def test_graph_exact_against_intervals(directed, e, h):
    mat = np.array([[0, 3, 2, 0], [3, 0, 2, 1.7], [2, 2, 0, 1.65], [0, 1.7, 1.65, 0.]])
    if directed:
        mat[1, 0] = 2.5
    want = _oracle(mat, e, h, 1.65)
    for backend in ("scipy", "numba"):
        np.testing.assert_allclose(get_tfnbs_score(mat, e, h, start_thres=1.65, backend=backend), want, atol=1e-10)


def test_blocks_fixed_capacity_ties_grid_and_minimum():
    mat = np.array([[0, 3, 2, 2], [3, 0, 2, 1.7], [2, 2, 0, 2.5], [2, 1.7, 2.5, 0.]])
    labels = np.array([0, 0, 1, 1])
    mask = np.ones((4, 4), bool); mask[0, 3] = mask[3, 0] = False
    for norm in ("sqrt", "linear", "none"):
        result = get_network_informed_tfnbs_score(mat, labels, [0, 0.4], [0, 3], start_thres=1.65, edge_mask=mask, normalization=norm)
        for k, (e, h) in enumerate(((0, 0), (0.4, 3))):
            np.testing.assert_allclose(result[:, :, k], _oracle(mat, e, h, 1.65, "ni", labels, mask, norm), atol=1e-10)
    for minimum in (1, 3, 5):
        result = get_fbc_tfnbs_score(mat, labels, 0, 0, start_thres=1.65, min_cluster_size=minimum)
        np.testing.assert_allclose(result, _oracle(mat, 0, 0, 1.65, "fbc", labels, minimum=minimum), atol=1e-10)


@pytest.mark.parametrize("n", [10, 30, 50, 100])
def test_legacy_grid_single_edge_and_start_band(n):
    mat = np.array([[0., 2.], [2., 0.]])
    levels = np.linspace(1.65 + 0.35 / n, 2, n)
    expected = (0.35 / n) * np.sum(levels**3)
    assert get_tfnbs_score(mat, .4, 3, n=n)[0, 1] == pytest.approx(expected)
    assert get_tfnbs_score(mat, .4, 3)[0, 1] == pytest.approx((2**4 - 1.65**4) / 4)


def _glm_data():
    rng = np.random.default_rng(12)
    a = rng.normal(size=(12, 4, 4))
    a = (a + a.transpose(0, 2, 1)) / 2
    for item in a:
        np.fill_diagonal(item, 0)
    x = np.column_stack([np.ones(12), np.linspace(-1, 1, 12), np.tile([-1., 1.], 6)])
    return a, x


def test_f_threshold_and_reload():
    y, x = _glm_data()
    result = compute_p_val_glm(y, design_matrix=x, contrast=[0, 1, 0], stat_type="fstat", n_permutations=3, use_mp=False, rng=4)
    meta = pickle.loads(pickle.dumps(result)).design["enhancement"]
    assert meta["integration"] == "exact" and meta["n"] is None
    assert meta["start_thres"] == pytest.approx(f.ppf(.9, 1, 9))
    assert meta["df2"] == [9]
    with pytest.warns(RuntimeWarning, match="below F0.90"):
        result = compute_p_val_glm(y, design_matrix=x, contrast=[[0, 1, 0], [0, 0, 1]], stat_type="fstat", start_thres=0, n_permutations=3, use_mp=False, rng=4)
    assert result.design["enhancement"]["df1"] == 2


def test_f_available_case_df_policy():
    y, x = _glm_data()
    y[0, 0, 1] = y[0, 1, 0] = np.nan
    opts = dict(design_matrix=x, contrast=[0, 1, 0], stat_type="fstat", missing="available_case", min_coverage=.8, n_permutations=3, use_mp=False, rng=3)
    with pytest.raises(ValueError, match="differing tested residual"):
        compute_p_val_glm(y, **opts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = compute_p_val_glm(y, start_thres=1., **opts)
    assert result.design["enhancement"]["df2"] == [8, 9]
    assert result.design["enhancement"]["start_thres"] == 1.


def test_f_square_identity_and_separate_tails():
    mat = np.array([[0., 2.0, 3.0], [2., 0, 2.5], [3., 2.5, 0.]])
    a = get_tfnbs_score(mat**2, .4, 3, start_thres=1.65**2)
    b = 2 * get_tfnbs_score(mat, .4, 7, start_thres=1.65)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_legacy_band_and_backend():
    mat = np.array([[0., 1.66, 3.], [1.66, 0., 0.], [3., 0., 0.]])
    assert get_tfnbs_score(mat, 0.4, 3, n=10)[0, 1] == 0
    assert get_tfnbs_score(mat, 0.4, 3)[0, 1] > 0
    for n in (10, 30, 50, 100):
        np.testing.assert_allclose(get_tfnbs_score(mat, .4, 3, n=n, backend="scipy"),
                                   get_tfnbs_score(mat, .4, 3, n=n, backend="numba"))


def test_equal_df_available_case_and_unrelated_f_method():
    y, x = _glm_data()
    y[0, 0, 1] = y[0, 1, 0] = np.nan
    y[1, 0, 2] = y[1, 2, 0] = np.nan
    # Every edge has one missing observation, but not necessarily the same one.
    for i in range(4):
        for j in range(i + 1, 4):
            if not np.isnan(y[:, i, j]).any():
                y[0, i, j] = y[0, j, i] = np.nan
    result = compute_p_val_glm(y, design_matrix=x, contrast=[0, 1, 0], stat_type="fstat",
                               missing="available_case", min_coverage=.8,
                               n_permutations=3, use_mp=False, rng=3)
    assert result.design["enhancement"]["df2"] == [8]
    assert result.design["enhancement"]["start_thres"] == pytest.approx(f.ppf(.9, 1, 8))
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        raw = compute_p_val_glm(y, design_matrix=x, contrast=[0, 1, 0], stat_type="fstat",
                                missing="available_case", min_coverage=.8, method="tstat",
                                n_permutations=3, use_mp=False, rng=3)
    assert "enhancement" not in raw.design
    assert not any("F0.90" in str(item.message) for item in captured)


def test_no_numba_installed_subprocess():
    import os
    import subprocess
    import sys
    script = '''
import builtins
original = builtins.__import__
def without_numba(name, *args, **kwargs):
    if name == "numba" or name.startswith("numba."):
        raise ImportError("test without numba")
    return original(name, *args, **kwargs)
builtins.__import__ = without_numba
import numpy as np
from conninfpy.enhance.tfnbs import get_tfnbs_score
from conninfpy.enhance.ni_tfnbs import get_network_informed_tfnbs_score
from conninfpy.enhance.fbc_tfnbs import get_fbc_tfnbs_score
m = np.array([[0., 2.], [2., 0.]])
labels = np.array([0, 0])
expected = (2**4 - 1.65**4) / 4
for result in (get_tfnbs_score(m, .4, 3),
               get_network_informed_tfnbs_score(m, labels, .4, 3),
               get_fbc_tfnbs_score(m, labels, .4, 3, min_cluster_size=1)):
    np.testing.assert_allclose(result[0, 1], expected)
'''
    child = subprocess.run([sys.executable, "-c", script], env=os.environ.copy(),
                           capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stderr


def test_ui_run_plan_exposes_effective_mode():
    from apps.components.design_inference import _method_plan_parameters
    exact = _method_plan_parameters("tfnbs", {"e": .4, "h": 3., "n": None, "start_thres": 1.65})
    legacy = _method_plan_parameters("tfnbs", {"e": .4, "h": 3., "n": 10, "start_thres": 1.65})
    assert exact["integration"] == "exact" and exact["start threshold"] == 1.65
    assert legacy["integration"] == "right-grid" and legacy["integration steps"] == 10


def test_paper_shared_selector_parameters_are_explicit():
    from examples.open_close._shared.inference import SELECTORS
    for method in ("tfnbs", "ni_tfnbs", "fbc_tfnbs"):
        config = SELECTORS[method]
        assert (config["e"], config["h"], config["n"], config["start_thres"]) == (.4, 3., 10, 1.65)
