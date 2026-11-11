"""Prespecified one-sided tests and the analysis scale of correlations.

* ``InferenceResult.one_sided`` keeps the hypothesised tail of a
  ``two_tailed=False`` run and reports the other direction as not tested.
* ``analyze(..., r_to_z=False)`` analyses correlations as given and records
  ``design['scale'] = 'correlation'``; the default still applies Fisher z.
"""
from __future__ import annotations

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
