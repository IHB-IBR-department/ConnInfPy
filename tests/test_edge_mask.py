"""Declared tested family (``edge_mask``): inference restricted to the marked edges."""
import unittest

import numpy as np

from conninfpy import analyze, compute_p_val

N, n = 12, 20


def _paired(seed=0, effects=()):
    rng = np.random.default_rng(seed)
    a = 0.2 * rng.normal(size=(n, N, N))
    a = (a + a.transpose(0, 2, 1)) / 2
    b = a + 0.2 * rng.normal(size=(n, N, N))
    b = (b + b.transpose(0, 2, 1)) / 2
    for i, j, d in effects:
        b[:, i, j] += d
        b[:, j, i] += d
    for x in (a, b):
        for s in x:
            np.fill_diagonal(s, 0.0)
    return a, b


def _mask():
    m = np.zeros((N, N), dtype=bool)
    m[:6, :6] = True                      # the family: all pairs among nodes 0-5
    np.fill_diagonal(m, False)
    return m


RUN = dict(test_type="paired", n_permutations=99, use_mp=False, rng=0)


class TestEdgeMask(unittest.TestCase):

    def test_full_mask_equals_no_mask(self):
        a, b = _paired(effects=[(1, 2, 0.4)])
        full = np.ones((N, N), dtype=bool)
        r0 = compute_p_val(a, b, method="tfnbs", **RUN)
        r1 = compute_p_val(a, b, method="tfnbs", edge_mask=full, **RUN)
        iu = np.triu_indices(N, 1)
        for key in ("positive", "negative"):
            np.testing.assert_array_equal(np.asarray(r0[key])[iu], np.asarray(r1[key])[iu])

    def test_outside_edges_are_untested_and_cannot_change_the_family(self):
        mask = _mask()
        out = ~mask & ~np.eye(N, dtype=bool)
        a, b = _paired(effects=[(1, 2, 0.4)])
        _, b_big = _paired(effects=[(1, 2, 0.4), (8, 9, 3.0), (9, 10, 3.0)])   # huge effects outside
        for method in ("tfnbs", "tstat", "nbs"):
            r = compute_p_val(a, b, method=method, edge_mask=mask, **RUN)
            r_big = compute_p_val(a, b_big, method=method, edge_mask=mask, **RUN)
            for key in ("positive", "negative"):
                p, p_big = np.asarray(r[key]), np.asarray(r_big[key])
                self.assertTrue(np.isnan(p[out]).all(), method)
                self.assertTrue(np.isfinite(p[mask]).all(), method)
                np.testing.assert_array_equal(p[mask], p_big[mask])
            if method != "nbs":                           # NBS: a single-edge component is not detected
                self.assertLess(np.asarray(r["positive"])[1, 2], 0.05, method)

    def test_ni_tfnbs_counts_capacity_inside_the_family(self):
        mask = _mask()
        labels = np.repeat([0, 1], N // 2)
        a, b = _paired(effects=[(1, 2, 0.4), (2, 3, 0.4)])
        r = compute_p_val(a, b, method="ni_tfnbs", net_labels=labels, edge_mask=mask, **RUN)
        self.assertTrue(np.isnan(np.asarray(r["positive"])[~mask & ~np.eye(N, dtype=bool)]).all())
        self.assertLess(np.asarray(r["positive"])[1, 2], 0.05)

    def test_analyze_passes_the_mask_through(self):
        mask = _mask()
        a, b = _paired(effects=[(1, 2, 0.4)])
        out = analyze(group1=a, group2=b, input="z", method="tfnbs", edge_mask=mask, **RUN)
        self.assertTrue(np.isnan(out.inference.positive[0, 7]))
        self.assertTrue(np.isfinite(out.inference.positive[0, 1]))
        self.assertNotIn((0, 7), set(zip(*np.nonzero(out.inference.positive < 0.05))))

    def test_validation(self):
        a, b = _paired()
        bad = _mask()
        bad[0, 9] = True                                  # not symmetric
        with self.assertRaisesRegex(ValueError, "symmetric"):
            compute_p_val(a, b, method="tfnbs", edge_mask=bad, **RUN)
        with self.assertRaisesRegex(ValueError, "shape"):
            compute_p_val(a, b, method="tfnbs", edge_mask=np.ones((3, 3), bool), **RUN)
        with self.assertRaises(NotImplementedError):
            compute_p_val(a, b, method="cnbs", net_labels=np.repeat([0, 1], N // 2), edge_mask=_mask(), **RUN)


if __name__ == "__main__":
    unittest.main()
