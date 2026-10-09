"""Tests for conninfpy._prepare (input preparation shared by all pipelines).

Covers:
- check_connectivity: shape validation, diagonal zeroing, caller's array
  never modified, no copy when the input is already clean, asymmetric
  (directed) input accepted.
- infer_input_kind: correlations vs Fisher-z vs ambiguous.
- prepare_connectivity: Fisher r→z applied exactly once for input='r'.
"""
import unittest
import warnings

import numpy as np

from conninfpy import compute_p_val
from conninfpy._prepare import (
    check_connectivity,
    infer_input_kind,
    prepare_connectivity,
)


def _stack(n=6, N=5, scale=0.3, diag=0.0, seed=0):
    rng = np.random.default_rng(seed)
    Y = np.tanh(rng.normal(0, scale, (n, N, N)))
    Y = (Y + Y.transpose(0, 2, 1)) / 2
    idx = np.arange(N)
    Y[:, idx, idx] = diag
    return Y


class TestCheckConnectivity(unittest.TestCase):

    def test_rejects_non_stack_shapes(self):
        """Check: anything but (n, N, N) raises a ValueError naming the argument."""
        for bad in (np.zeros((5, 10)), np.zeros((3, 4, 5)), np.zeros(7)):
            with self.assertRaisesRegex(ValueError, r"group1 must have shape"):
                check_connectivity(bad, "group1")

    def test_zeroes_diagonal_without_touching_input(self):
        """Check: a non-zero diagonal is zeroed on a copy, input unchanged."""
        Y = _stack(diag=1.0)
        out = check_connectivity(Y)
        idx = np.arange(Y.shape[-1])
        self.assertTrue(np.all(out[:, idx, idx] == 0))
        self.assertTrue(np.all(Y[:, idx, idx] == 1.0))
        np.testing.assert_array_equal(out[:, 0, 1], Y[:, 0, 1])

    def test_clean_float64_input_is_not_copied(self):
        """Check: an already-clean float64 stack is passed through, no copy."""
        Y = _stack(diag=0.0)
        self.assertIs(check_connectivity(Y), Y)

    def test_nan_diagonal_is_zeroed(self):
        """Check: NaN on the diagonal (e.g. self-correlation not computed) is zeroed."""
        Y = _stack(diag=np.nan)
        out = check_connectivity(Y)
        self.assertFalse(np.isnan(out).any())

    def test_asymmetric_input_is_rejected(self):
        """Check: directed (asymmetric) connectivity is rejected.

        Regression: it was accepted, but the fast pairwise path tested only the
        upper triangle while the observed maps kept both directions, so a
        lower-triangle effect got p = 1/(B+1) against an all-zero null.
        """
        Y = _stack()
        Y[:, 0, 1] += 0.2
        with self.assertRaisesRegex(ValueError, "not symmetric"):
            check_connectivity(Y)

    def test_one_sided_missing_value_is_asymmetric(self):
        Y = _stack()
        Y[2, 0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "not symmetric"):
            check_connectivity(Y, allow_missing=True)

    def test_rounding_level_asymmetry_is_accepted(self):
        Y = _stack()
        Y[:, 0, 1] += 1e-13
        check_connectivity(Y)


class TestInferInputKind(unittest.TestCase):

    def test_unit_diagonal_means_correlations(self):
        self.assertEqual(infer_input_kind(_stack(diag=1.0)), "r")

    def test_values_beyond_one_mean_fisher_z(self):
        Y = _stack(diag=0.0)
        Y[:, 0, 1] = Y[:, 1, 0] = 1.4
        self.assertEqual(infer_input_kind(Y), "z")

    def test_dead_region_does_not_hide_correlations(self):
        """Check: an all-zero row (region with no signal) keeps input='r'.

        Real case: ABIDE-mini subject 191 has a constant ROI whose whole
        row, diagonal included, is stored as 0.
        """
        Y = _stack(diag=1.0)
        Y[2, 3, :] = 0.0
        Y[2, :, 3] = 0.0
        self.assertEqual(infer_input_kind(Y), "r")

    def test_all_zero_stack_is_ambiguous(self):
        self.assertIsNone(infer_input_kind(np.zeros((3, 4, 4))))

    def test_zero_diagonal_within_unit_range_is_ambiguous(self):
        """Check: zeroed-diagonal data within [-1, 1] cannot be classified."""
        self.assertIsNone(infer_input_kind(_stack(diag=0.0)))

    def test_any_stack_beyond_one_decides_for_all(self):
        """Check: two-sample inputs are classified jointly."""
        a, b = _stack(diag=0.0, seed=1), _stack(diag=0.0, seed=2)
        b[:, 2, 3] = b[:, 3, 2] = -2.0
        self.assertEqual(infer_input_kind(a, b), "z")


class TestPrepareConnectivity(unittest.TestCase):

    def test_r_input_is_transformed_once_without_boundary_warning(self):
        """Check: unit diagonal is zeroed before arctanh, so no ±1 warning."""
        Y = _stack(diag=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = prepare_connectivity(Y, "r")
        np.testing.assert_allclose(out[:, 0, 1], np.arctanh(Y[:, 0, 1]))

    def test_z_input_is_left_on_its_scale(self):
        Y = _stack(diag=0.0)
        np.testing.assert_array_equal(prepare_connectivity(Y, "z"), Y)

    def test_invalid_kind_raises(self):
        with self.assertRaises(ValueError):
            prepare_connectivity(_stack(), "fisher")


class TestPipelinesUseTheCheck(unittest.TestCase):

    def test_compute_p_val_accepts_varying_diagonal(self):
        """Check: a subject-varying diagonal no longer crashes TFNBS.

        Before the shared check, TFNBS raised "Diagonal elements must be
        zero" from deep inside scoring when the diagonal varied across
        subjects (Fisher-z cap at r = 1, covariance input).
        """
        g1, g2 = _stack(n=8, seed=3), _stack(n=8, seed=4)
        rng = np.random.default_rng(5)
        idx = np.arange(g1.shape[-1])
        g1[:, idx, idx] = rng.normal(5, 1, (8, g1.shape[-1]))
        res = compute_p_val(g1, g2, test_type="two-sample", method="tfnbs",
                            n_permutations=10, use_mp=False, rng=0)
        self.assertEqual(res["positive"].shape, g1.shape[1:])

    def test_compute_p_val_rejects_mismatched_node_counts(self):
        with self.assertRaisesRegex(ValueError, "same \\(N, N\\) shape"):
            compute_p_val(_stack(N=5), _stack(N=6), test_type="two-sample",
                          method="tstat", n_permutations=5, use_mp=False)


if __name__ == "__main__":
    unittest.main()


class TestExactSymmetry(unittest.TestCase):
    """Undirected stacks are made exactly symmetric; rounding noise never makes a map 'directed'.

    np.corrcoef can leave the two triangles one ulp apart. With a purely
    relative symmetry test, a near-zero statistic then failed it and the
    scorers counted every undirected edge twice (~0.4% of simulated maps).
    """

    def test_rounding_asymmetry_is_averaged_and_exact_input_is_untouched(self):
        rng = np.random.default_rng(0)
        Y = rng.normal(size=(3, 5, 5))
        Y = (Y + Y.transpose(0, 2, 1)) / 2
        for s in Y:
            np.fill_diagonal(s, 0)
        self.assertIs(check_connectivity(Y), Y)            # exact input: same array
        noisy = Y.copy()
        noisy[:, 0, 1] += 1e-16
        out = check_connectivity(noisy)
        np.testing.assert_array_equal(out, out.transpose(0, 2, 1))
        self.assertEqual(noisy[0, 0, 1], Y[0, 0, 1] + 1e-16)   # caller's array unchanged

    def test_scorers_treat_near_zero_noise_as_symmetric(self):
        from conninfpy.nbs_score import _is_symmetric as nbs_sym, get_nbs_score
        from conninfpy.tfnbs_score import _is_symmetric as tf_sym, get_tfnbs_score
        t = np.zeros((6, 6))
        for i, j, v in [(0, 1, 3.0), (1, 2, 2.5), (3, 4, 7.4e-06)]:
            t[i, j] = t[j, i] = v
        noisy = t.copy()
        noisy[4, 3] = 7.4e-06 + 3e-15                       # one-ulp-scale difference at a near-zero value
        self.assertTrue(tf_sym(noisy) and nbs_sym(noisy))
        np.testing.assert_allclose(get_tfnbs_score(noisy, e=0.4, h=3.0, n=10, backend="scipy"),
                                   get_tfnbs_score(t, e=0.4, h=3.0, n=10, backend="scipy"))
        np.testing.assert_allclose(get_nbs_score(noisy, threshold=2.0), get_nbs_score(t, threshold=2.0))
        directed = t.copy()
        directed[1, 0] = 0.0
        self.assertFalse(tf_sym(directed))
