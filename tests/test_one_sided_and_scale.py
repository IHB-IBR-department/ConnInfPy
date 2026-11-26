"""Prespecified one-sided tests and the analysis scale of correlations.

* ``InferenceResult.one_sided`` keeps the hypothesised tail of a
  ``two_tailed=False`` run and reports the other direction as not tested.
* ``analyze(..., r_to_z=False)`` analyses correlations as given and records
  ``design['scale'] = 'correlation'``; the default still applies Fisher z.
* The Streamlit worker runs on the loaded scale (no silent r -> z) and passes
  the chosen direction to both the primary and the companion run.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from conninfpy import analyze, compute_p_val_glm


def _glm_data(n=30, N=8, seed=0):
    """Correlation-like stacks (diagonal 1) with a positive effect of x."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    Y = 0.3 * rng.standard_normal((n, N, N))
    Y = (Y + Y.transpose(0, 2, 1)) / 2
    Y[:, 0, 1] = Y[:, 1, 0] = 0.2 + 0.15 * x + 0.05 * rng.standard_normal(n)
    Y = np.clip(Y, -0.9, 0.9)
    idx = np.arange(N)
    Y[:, idx, idx] = 1.0
    return Y, x


def test_one_sided_keeps_the_tested_tail_and_blanks_the_other():
    Y, x = _glm_data()
    Y0 = Y.copy()
    Y0[:, np.arange(8), np.arange(8)] = 0.0
    res = compute_p_val_glm(Y0, interest=x, method="tstat", n_permutations=200,
                            rng=1, use_mp=False, two_tailed=False)
    pos = res.one_sided("positive")
    np.testing.assert_array_equal(pos.positive, res.positive)
    assert np.all(pos.negative == 1.0)
    assert pos.tested_tail == "positive"
    assert pos.tail_policy == "one-sided-positive"
    np.testing.assert_array_equal(pos.stat_signed, res.stat_signed)
    assert pos.n_significant()["negative"] == 0
    neg = res.one_sided("negative")
    assert np.all(neg.positive == 1.0) and neg.tail_policy == "one-sided-negative"
    assert res.tail_policy == "separate-one-sided"     # the original is unchanged


def test_one_sided_rejects_two_sided_results_and_bad_directions():
    Y, x = _glm_data()
    Y[:, np.arange(8), np.arange(8)] = 0.0
    two = compute_p_val_glm(Y, interest=x, method="tstat", n_permutations=50,
                            rng=1, use_mp=False, two_tailed=True)
    with pytest.raises(ValueError, match="two_tailed=False"):
        two.one_sided("positive")
    one = compute_p_val_glm(Y, interest=x, method="tstat", n_permutations=50,
                            rng=1, use_mp=False, two_tailed=False)
    with pytest.raises(ValueError, match="direction"):
        one.one_sided("up")


def test_analyze_result_one_sided_wraps_the_inference():
    Y, x = _glm_data()
    res = analyze(Y, interest=x, input="r", method="tstat", n_permutations=50,
                  rng=1, use_mp=False, two_tailed=False).one_sided("positive")
    assert res.inference.tail_policy == "one-sided-positive"
    assert np.all(res.negative == 1.0)


def test_r_to_z_false_analyses_correlations_as_given():
    Y, x = _glm_data()
    kw = dict(interest=x, input="r", method="tstat", n_permutations=50, rng=1,
              use_mp=False, two_tailed=False)
    raw = analyze(Y, r_to_z=False, **kw).inference
    fz = analyze(Y, **kw).inference
    assert raw.design["scale"] == "correlation"
    assert fz.design["scale"] == "fisher_z"
    Yr = Y.copy()
    Yr[:, np.arange(8), np.arange(8)] = 0.0
    direct = compute_p_val_glm(Yr, interest=x, method="tstat", n_permutations=50,
                               rng=1, use_mp=False, two_tailed=False)
    np.testing.assert_allclose(raw.stat_signed, direct.stat_signed, atol=1e-12)
    assert not np.allclose(raw.stat_signed, fz.stat_signed)


def _app_worker():
    """The Streamlit worker, loaded without importing the UI."""
    source = Path(__file__).resolve().parents[1] / "conninfpy/app/components/design_inference.py"
    tree = ast.parse(source.read_text())
    names = {"_run_inference_background", "_method_plan_parameters"}
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
                        type_ignores=[])
    namespace = {}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["_run_inference_background"]


@pytest.mark.parametrize("tested_tail", [None, "positive", "negative"])
def test_app_worker_runs_on_loaded_scale_with_chosen_direction(tested_tail):
    worker = _app_worker()
    args = {name: None for name in inspect.signature(worker).parameters}
    task = SimpleNamespace()
    restricted = []

    def one_sided(direction):
        restricted.append(direction)
        return fake

    fake = SimpleNamespace(significant_edges=lambda **kw: [], one_sided=one_sided)
    args.update(task=task, data_kind="correlation", test_type="glm", operator_choice="nbs",
                op_kwargs={"threshold": 3.1}, run_sensitivity=True, comp_method="tstat",
                comp_kwargs={}, acceleration_choice="none", tested_tail=tested_tail)
    with patch("conninfpy.analyze", return_value=fake) as run:
        worker(**args)
    assert task.status == "success", getattr(task, "error", None)
    calls = run.call_args_list
    assert len(calls) == 2
    assert all(c.kwargs["input"] == "r" and c.kwargs["r_to_z"] is False for c in calls)
    # The direction is declared to analyze() before the run (prespecified);
    # the worker never restricts a finished result post hoc.
    assert all(c.kwargs["tested_tail"] is tested_tail for c in calls)
    assert all("two_tailed" not in c.kwargs for c in calls)
    assert restricted == []
    assert task.run_plan["analysis_scale"] == "correlation"
    assert task.run_plan["tested_tail"] == tested_tail
    assert task.run_plan["tail_policy"] == (
        "two-sided" if tested_tail is None else f"one-sided-{tested_tail}"
    )
    assert task.run_plan["two_tailed"] is (tested_tail is None)
    assert task.run_plan["tail_selection"] == ("prespecified" if tested_tail else None)
    # GLM run: no variance choice is passed (the GLM t is pooled).
    assert all("var_equal" not in c.kwargs for c in calls)
    assert task.run_plan["t_statistic"] == "glm"


@pytest.mark.parametrize("test_type,var_equal,expected", [
    ("two-sample", True, "student"),
    ("two-sample", False, "welch"),
    ("paired", None, None),
])
def test_app_worker_passes_the_variance_choice_to_the_two_sample_test(test_type, var_equal, expected):
    worker = _app_worker()
    args = {name: None for name in inspect.signature(worker).parameters}
    task = SimpleNamespace()
    fake = SimpleNamespace(significant_edges=lambda **kw: [])
    args.update(task=task, data_kind="fisher_z", test_type=test_type, operator_choice="tstat",
                op_kwargs={}, run_sensitivity=True, comp_method="tstat", comp_kwargs={},
                acceleration_choice="none", var_equal=var_equal)
    with patch("conninfpy.analyze", return_value=fake) as run:
        worker(**args)
    assert task.status == "success", getattr(task, "error", None)
    for call in run.call_args_list:
        if expected is None:
            assert "var_equal" not in call.kwargs
        else:
            assert call.kwargs["var_equal"] is var_equal
    assert task.run_plan["t_statistic"] == expected
