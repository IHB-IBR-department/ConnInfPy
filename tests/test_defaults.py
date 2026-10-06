"""Every inference entry point shares the defaults in conninfpy.defaults.

Guards against the drift this suite was written for: analyze() once
defaulted to 200 permutations with GPD acceleration while compute_p_val*
defaulted to 1000 exact permutations, so the same method gave different
inference depending on which function was called.
"""
import inspect
import unittest

from conninfpy import (
    analyze,
    compute_p_val,
    compute_p_val_glm,
    compute_p_val_glm_multi,
    defaults,
    nbs_bct,
)

ENTRY_POINTS = (analyze, compute_p_val, compute_p_val_glm, compute_p_val_glm_multi)

EXPECTED = {
    "n_permutations": defaults.DEFAULT_N_PERMUTATIONS,
    "acceleration": defaults.DEFAULT_ACCELERATION,
    "e": defaults.DEFAULT_EXTENT_EXPONENT,
    "h": defaults.DEFAULT_HEIGHT_EXPONENT,
    "n": defaults.DEFAULT_N_THRESHOLDS_PERMUTATION,
}


class TestSharedDefaults(unittest.TestCase):

    def test_entry_points_use_shared_constants(self):
        for fn in ENTRY_POINTS:
            params = inspect.signature(fn).parameters
            for name, expected in EXPECTED.items():
                with self.subTest(fn=fn.__name__, param=name):
                    self.assertIn(name, params)
                    self.assertEqual(params[name].default, expected)

    def test_nbs_bct_permutation_budget_matches(self):
        params = inspect.signature(nbs_bct).parameters
        self.assertEqual(params["n_permutations"].default,
                         defaults.DEFAULT_N_PERMUTATIONS)


if __name__ == "__main__":
    unittest.main()
