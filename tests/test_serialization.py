"""Serialization tests for inference results (NETSTAT-8).

``pickle.dumps``/``pickle.loads`` (at ``pickle.DEFAULT_PROTOCOL`` and
``pickle.HIGHEST_PROTOCOL``), ``copy.copy`` and ``copy.deepcopy`` must
work for :class:`~conninfpy.InferenceResult`,
:class:`~conninfpy.OmnibusInferenceResult` and
:class:`~conninfpy.analysis.AnalyzeResult`, preserve every p-value /
statistic map and metadata attribute, and leave deep-copied arrays
independent of the original. Restored results must keep working
(``significant_edges``, ``n_significant``, ``select``, ``one_sided``,
``network_summary``, ``to_csv``) and compare equal to the original.

Comparisons are recursive and NaN-aware: numpy arrays (``equal_nan``),
pandas DataFrames (``assert_frame_equal``), nested dicts, lists and
scalars — never a plain ``==`` over ``vars()``.
"""
import copy
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from conninfpy import (
    AnalyzeResult,
    InferenceResult,
    OmnibusInferenceResult,
    analyze,
    compute_p_val,
    compute_p_val_glm,
    compute_p_val_glm_multi,
    compute_p_val_paired_glm,
    fisher_r_to_z,
    generate_fc_matrices,
)

N_ROI = 6
N_PERM = 20
LABELS = np.array([0, 0, 0, 1, 1, 2])  # cNBS partition; network 2 is a single node
NETWORK_LABELS = ["A", "A", "B", "B", "C", "C"]


# ---------------------------------------------------------------------------
# Comparison helpers (recursive, NaN-aware)
# ---------------------------------------------------------------------------

def assert_tree_equal(tc, a, b, path="value"):
    """Recursive NaN-aware equality over ndarrays, DataFrames, dicts, lists."""
    if a is None or b is None:
        tc.assertIsNone(a, f"{path}: {a!r} vs {b!r}")
        tc.assertIsNone(b, f"{path}: {a!r} vs {b!r}")
        return
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        tc.assertEqual(a.shape, b.shape, path)
        tc.assertEqual(a.dtype, b.dtype, path)
        if a.dtype.kind == "f":
            tc.assertTrue(np.array_equal(a, b, equal_nan=True),
                          f"{path}: arrays differ (NaN-aware)")
        else:
            tc.assertTrue(np.array_equal(a, b), f"{path}: arrays differ")
        return
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b)
        return
    if isinstance(a, dict) and isinstance(b, dict):
        tc.assertEqual(sorted(a, key=repr), sorted(b, key=repr), path)
        for k in a:
            assert_tree_equal(tc, a[k], b[k], f"{path}[{k!r}]")
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        tc.assertIs(type(a), type(b), path)
        tc.assertEqual(len(a), len(b), path)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_tree_equal(tc, x, y, f"{path}[{i}]")
        return
    if isinstance(a, float) or isinstance(b, float):
        tc.assertTrue((a != a and b != b) or a == b,
                      f"{path}: {a!r} vs {b!r}")
        return
    tc.assertEqual(a, b, path)


def assert_results_equal(tc, original, restored):
    """Full equality of dict items and instance attributes of a result."""
    tc.assertIs(type(restored), type(original))
    tc.assertEqual(sorted(restored.keys()), sorted(original.keys()))
    for key in original.keys():
        assert_tree_equal(tc, original[key], restored[key], f"[{key!r}]")
    tc.assertEqual(sorted(vars(original)), sorted(vars(restored)))
    for name in vars(original):
        assert_tree_equal(tc, getattr(original, name), getattr(restored, name),
                          f".{name}")


def assert_analyze_result_equal(tc, original, restored):
    """Compare AnalyzeResult dataclass fields (inference + diagnostics)."""
    tc.assertIs(type(restored), type(original))
    assert_results_equal(tc, original.inference, restored.inference)
    for field in ("combat_diagnostics", "design_diagnostics", "flags"):
        assert_tree_equal(tc, getattr(original, field), getattr(restored, field),
                          f".{field}")


def assert_all_roundtrips(tc, obj):
    """pickle (DEFAULT_PROTOCOL + HIGHEST_PROTOCOL), copy and deepcopy all
    restore an equal object; returns the deepcopy."""
    comparer = (assert_results_equal
                if isinstance(obj, (InferenceResult, OmnibusInferenceResult))
                else assert_analyze_result_equal)
    for protocol in (pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL):
        restored = pickle.loads(pickle.dumps(obj, protocol=protocol))
        comparer(tc, obj, restored)
    shallow = copy.copy(obj)
    comparer(tc, obj, shallow)
    deep = copy.deepcopy(obj)
    comparer(tc, obj, deep)
    return deep


def assert_deepcopy_independent(tc, original, deep, array_attrs=()):
    """Filling every float array reachable via items / named attributes of
    ``deep`` with NaN must leave ``original`` untouched."""
    tc.assertIsNot(deep, original)
    before_items = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in original.items()}
    before_attrs = {n: (v.copy() if isinstance(v, np.ndarray) else v)
                    for n, v in vars(original).items()}
    mutated = 0
    for k, v in list(deep.items()):
        if isinstance(v, np.ndarray) and v.dtype.kind == "f" and v.size:
            v.fill(np.nan)
            mutated += 1
    for n in array_attrs:
        v = getattr(deep, n, None)
        if isinstance(v, np.ndarray) and v.dtype.kind == "f" and v.size:
            v.fill(np.nan)
            mutated += 1
    tc.assertGreater(mutated, 0)
    for k, v in before_items.items():
        assert_tree_equal(tc, v, original[k], f"independence[{k!r}]")
    for n, v in before_attrs.items():
        assert_tree_equal(tc, v, getattr(original, n), f"independence.{n}")


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _pairwise_z(n=16, seed=11):
    g1, g2, _ = generate_fc_matrices(
        N=N_ROI, effect_size=0.3, n_samples_group1=n,
        n_samples_group2=n, seed=seed,
    )
    return fisher_r_to_z(g1), fisher_r_to_z(g2)


def _glm_data(n=24, seed=0):
    rng = np.random.RandomState(seed)
    Y = rng.randn(n, N_ROI, N_ROI)
    Y = (Y + Y.transpose(0, 2, 1)) / 2
    for i in range(n):
        np.fill_diagonal(Y[i], 0.0)
    age = rng.randn(n)
    return Y, age


def _kill_roi(g, subject, roi):
    """Knock out one subject's row/column like available-case tests do."""
    g = g.copy()
    g[subject, roi, :] = np.nan
    g[subject, :, roi] = np.nan
    return g


# ---------------------------------------------------------------------------
# AC#1 / AC#2: fully populated objects, all protocols, copy semantics
# ---------------------------------------------------------------------------

class TestFullyPopulatedRoundTrip(unittest.TestCase):
    """A hand-built result with *every* attribute set round-trips through
    pickle (default + highest protocol), copy.copy and copy.deepcopy."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(42)
        p = rng.uniform(size=(N_ROI, N_ROI))
        cls.p = np.round((p + p.T) / 2, 3)
        np.fill_diagonal(cls.p, 1.0)
        cls.result = InferenceResult(
            cls.p.copy(), cls.p.copy(),
            method="tfnbs",
            n_permutations=N_PERM,
            acceleration=None,
            wall_time_s=1.25,
            null_max_dist={"positive": rng.rand(N_PERM),
                           "negative": rng.rand(N_PERM)},
            stat_positive=np.abs(rng.randn(N_ROI, N_ROI)),
            stat_negative=np.zeros((N_ROI, N_ROI)),
            stat_type="beta",
            harmonized=True,
            preserve_provided=True,
            strata_provided=True,
            combat_diagnostics={"strategy": "combat_only",
                                "between_site_variance_ratio_after_over_before": 0.5},
            e_grid=np.array([0.4, 0.5]),
            h_grid=np.array([3.0, 2.0]),
            missing_report={"subjects_with_missing": [3],
                            "regions_with_missing": [1],
                            "edge_n": np.full((N_ROI, N_ROI), 15.0)},
            block_report={"correction": "bonferroni",
                          "n_blocks_tested": 3,
                          "blocks": [{"network_a": "A", "network_b": "B",
                                      "p_positive": 0.1, "p_negative": np.nan}]},
            two_tailed=True,
            beta=rng.randn(2, N_ROI, N_ROI),
            design={"columns": ["intercept", "age"],
                    "column_means": np.array([1.0, 0.1]),
                    "contrast": np.array([0.0, 1.0]),
                    "profiles": {"mean": np.array([1.0, 0.1]),
                                 "age+1SD": np.array([1.0, 1.1])},
                    "comparison": ["mean", "age+1SD"],
                    "scale": "fisher_z"},
            tested_tail=None,
        )

    def test_pickle_copy_deepcopy_roundtrip(self):
        deep = assert_all_roundtrips(self, self.result)
        # every float array reachable through items and named attributes
        assert_deepcopy_independent(
            self, self.result, deep,
            array_attrs=("stat_positive", "stat_negative", "beta", "e_grid", "h_grid"),
        )

    def test_deepcopy_nested_dicts_independent(self):
        deep = copy.deepcopy(self.result)
        for key in ("positive", "negative"):
            deep.null_max_dist[key].fill(np.nan)
        deep.design["contrast"].fill(np.nan)
        deep.missing_report["edge_n"].fill(np.nan)
        self.assertFalse(np.isnan(self.result.null_max_dist["positive"]).any())
        self.assertFalse(np.isnan(self.result.design["contrast"]).any())
        self.assertFalse(np.isnan(self.result.missing_report["edge_n"]).any())

    def test_copy_shares_arrays(self):
        shallow = copy.copy(self.result)
        for key in ("positive", "negative"):
            self.assertIs(shallow[key], self.result[key])
        self.assertIs(shallow.beta, self.result.beta)

    def test_omnibus_roundtrip(self):
        rng = np.random.RandomState(43)
        F = np.abs(rng.randn(N_ROI, N_ROI))
        omni = OmnibusInferenceResult(
            self.p.copy(), method="tstat", n_permutations=N_PERM,
            acceleration=None, wall_time_s=0.5,
            stat_omnibus=F, stat_type="fstat",
            harmonized=True, preserve_provided=False, strata_provided=True,
            combat_diagnostics={"strategy": "site_dummies_glm"},
            missing_report=None, block_report=None,
            beta=rng.randn(2, N_ROI, N_ROI),
            design={"columns": ["intercept", "x1", "x2"],
                    "contrast": np.eye(3)[1:], "profiles": {},
                    "scale": "fisher_z"},
        )
        deep = assert_all_roundtrips(self, omni)
        assert_deepcopy_independent(self, omni, deep,
                                    array_attrs=("stat_omnibus", "beta"))

    def test_analyze_result_roundtrip(self):
        out = AnalyzeResult(
            inference=self.result,
            combat_diagnostics={"strategy": "combat_only", "ratio": 0.5},
            design_diagnostics={"columns": ["intercept", "age"]},
            flags=["residual site differences"],
        )
        for protocol in (pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL):
            restored = pickle.loads(pickle.dumps(out, protocol=protocol))
            self.assertIsInstance(restored, AnalyzeResult)
            assert_results_equal(self, out.inference, restored.inference)
            assert_tree_equal(self, out.combat_diagnostics, restored.combat_diagnostics)
            assert_tree_equal(self, out.design_diagnostics, restored.design_diagnostics)
            assert_tree_equal(self, out.flags, restored.flags)
        shallow = copy.copy(out)
        self.assertIs(shallow.inference, out.inference)  # shallow semantics
        deep = copy.deepcopy(out)
        assert_results_equal(self, out.inference, deep.inference)
        deep.inference["positive"].fill(np.nan)
        self.assertFalse(np.isnan(out.inference["positive"]).any())


# ---------------------------------------------------------------------------
# AC#3: round-trip on results of every inference path
# ---------------------------------------------------------------------------

class TestPairwisePaths(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.g1z, cls.g2z = _pairwise_z()

    def test_one_sample(self):
        r = compute_p_val(self.g1z, test_type="one-sample", method="tstat",
                          n_permutations=N_PERM, use_mp=False,
                          acceleration=None, rng=1)
        self.assertIsInstance(r, InferenceResult)
        assert_all_roundtrips(self, r)

    def test_paired(self):
        r = compute_p_val(self.g1z, self.g2z, test_type="paired", method="tstat",
                          n_permutations=N_PERM, use_mp=False,
                          acceleration=None, rng=2)
        assert_all_roundtrips(self, r)

    def test_two_sample(self):
        r = compute_p_val(self.g1z, self.g2z, test_type="two-sample", method="tstat",
                          n_permutations=N_PERM, use_mp=False,
                          acceleration=None, rng=3)
        assert_all_roundtrips(self, r)


class TestGLMPaths(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.Y, cls.age = _glm_data()

    def test_glm_tstat(self):
        r = compute_p_val_glm(self.Y, interest=self.age, stat_type="tstat",
                              method="tstat", n_permutations=N_PERM,
                              use_mp=False, acceleration=None, rng=4)
        self.assertIsInstance(r, InferenceResult)
        self.assertIsNotNone(r.design)
        self.assertIsNotNone(r.beta)
        assert_all_roundtrips(self, r)

    def test_glm_fstat_omnibus(self):
        rng = np.random.RandomState(5)
        x2 = rng.randn(len(self.Y))
        X = np.column_stack([np.ones(len(self.Y)), self.age, x2])
        contrast = np.array([[0, 1, 0], [0, 0, 1]], dtype=float)
        r = compute_p_val_glm(self.Y, design_matrix=X, contrast=contrast,
                              stat_type="fstat", method="tstat",
                              n_permutations=N_PERM, use_mp=False,
                              acceleration=None, rng=5)
        self.assertIsInstance(r, OmnibusInferenceResult)
        assert_all_roundtrips(self, r)

    def test_glm_multi_contrast(self):
        rng = np.random.RandomState(6)
        x2 = rng.randn(len(self.Y))
        X = np.column_stack([np.ones(len(self.Y)), self.age, x2])
        contrasts = {"age": np.array([0.0, 1.0, 0.0]),
                     "x2": np.array([0.0, 0.0, 1.0])}
        out = compute_p_val_glm_multi(self.Y, X, contrasts, method="tstat",
                                      n_permutations=N_PERM, use_mp=False,
                                      acceleration=None, rng=6)
        # the whole mapping pickles, and each result round-trips
        restored = pickle.loads(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
        self.assertEqual(sorted(restored), sorted(out))
        for name in out:
            self.assertIsInstance(out[name], InferenceResult)
            assert_results_equal(self, out[name], restored[name])

    def test_paired_glm(self):
        rng = np.random.RandomState(7)
        Y_A = rng.randn(12, N_ROI, N_ROI)
        Y_A = (Y_A + Y_A.transpose(0, 2, 1)) / 2
        Y_B = Y_A + 0.2 * rng.randn(12, N_ROI, N_ROI)
        Y_B = (Y_B + Y_B.transpose(0, 2, 1)) / 2
        for i in range(12):
            np.fill_diagonal(Y_A[i], 0.0)
            np.fill_diagonal(Y_B[i], 0.0)
        r = compute_p_val_paired_glm(Y_A, Y_B, method="tstat",
                                     n_permutations=N_PERM, use_mp=False,
                                     acceleration=None, rng=7)
        self.assertIsInstance(r, InferenceResult)
        assert_all_roundtrips(self, r)


class TestSpecializedPaths(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.g1z, cls.g2z = _pairwise_z(seed=12)

    def test_cnbs_block_report(self):
        r = compute_p_val(self.g1z, self.g2z, test_type="two-sample",
                          method="cnbs", net_labels=LABELS,
                          n_permutations=N_PERM, use_mp=False,
                          acceleration=None, rng=8)
        self.assertIsInstance(r, InferenceResult)
        self.assertIsNotNone(r.block_report)
        deep = assert_all_roundtrips(self, r)
        assert_deepcopy_independent(self, r, deep,
                                    array_attrs=("stat_positive", "stat_negative"))

    def test_tfnbs_parameter_grid(self):
        r = compute_p_val(self.g1z, self.g2z, test_type="two-sample",
                          method="tfnbs", e=[0.4, 0.5], h=[3.0, 2.0], n=10,
                          n_permutations=N_PERM, use_mp=False,
                          acceleration=None, rng=9)
        self.assertTrue(r.is_grid)
        self.assertEqual(r["positive"].ndim, 3)
        deep = assert_all_roundtrips(self, r)
        assert_deepcopy_independent(self, r, deep,
                                    array_attrs=("stat_positive", "stat_negative",
                                                 "e_grid", "h_grid"))

    def test_available_case_missing_report(self):
        g1_nan = self.g1z.copy()
        for subject in range(4):            # 12/16 = 75% < 80% coverage
            g1_nan = _kill_roi(g1_nan, subject=subject, roi=2)
        r = compute_p_val(g1_nan, self.g2z, test_type="two-sample",
                          method="tstat", n_permutations=N_PERM, use_mp=False,
                          acceleration=None, missing="available_case", rng=10)
        self.assertIsNotNone(r.missing_report)
        self.assertTrue(np.isnan(r["positive"]).any())  # NaN p-maps round-trip
        assert_all_roundtrips(self, r)

    def test_one_sided_result(self):
        r = compute_p_val(self.g1z, self.g2z, test_type="two-sample",
                          method="tstat", n_permutations=N_PERM, use_mp=False,
                          acceleration=None, rng=11, two_tailed=False)
        one = r.one_sided("positive")
        self.assertEqual(one.tested_tail, "positive")
        assert_all_roundtrips(self, one)
        self.assertEqual(one.tail_policy, "one-sided-positive")

    def test_analyze_result_full(self):
        Y, age = _glm_data(n=20, seed=13)
        sites = np.array([0] * 10 + [1] * 10)
        out = analyze(Y, interest=age, sites=sites, fisher_z=False,
                      n_permutations=N_PERM, use_mp=False,
                      acceleration=None, rng=12)
        self.assertIsInstance(out, AnalyzeResult)
        self.assertIsNotNone(out.combat_diagnostics)
        assert_all_roundtrips(self, out)


# ---------------------------------------------------------------------------
# AC#4: restored results stay functional and match the original
# ---------------------------------------------------------------------------

class TestRestoredFunctionality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        g1z, g2z = _pairwise_z(seed=21)
        cls.two_sample = compute_p_val(
            g1z, g2z, test_type="two-sample", method="tstat",
            n_permutations=N_PERM, use_mp=False, acceleration=None, rng=21, two_tailed=False)
        grid = compute_p_val(
            g1z, g2z, test_type="two-sample", method="tfnbs",
            e=[0.4, 0.5], h=[3.0, 2.0], n=10,
            n_permutations=N_PERM, use_mp=False, acceleration=None, rng=22)
        cls.grid = grid

    def _restored(self, obj):
        return pickle.loads(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def test_significant_edges_and_n_significant(self):
        restored = self._restored(self.two_sample)
        assert_frame = restored.significant_edges(alpha=0.9, include_nonsig=True)
        orig_frame = self.two_sample.significant_edges(alpha=0.9, include_nonsig=True)
        pd.testing.assert_frame_equal(orig_frame, assert_frame)
        assert_tree_equal(self, self.two_sample.n_significant(),
                          restored.n_significant(), "n_significant")

    def test_network_summary(self):
        restored = self._restored(self.two_sample)
        pd.testing.assert_frame_equal(
            self.two_sample.network_summary(NETWORK_LABELS),
            restored.network_summary(NETWORK_LABELS),
        )

    def test_to_csv(self):
        restored = self._restored(self.two_sample)
        with tempfile.TemporaryDirectory() as tmp:
            f_orig = Path(tmp) / "orig.csv"
            f_rest = Path(tmp) / "restored.csv"
            self.two_sample.to_csv(f_orig, alpha=0.9, include_nonsig=True)
            restored.to_csv(f_rest, alpha=0.9, include_nonsig=True)
            self.assertEqual(f_orig.read_bytes(), f_rest.read_bytes())

    def test_select_on_grid(self):
        restored = self._restored(self.grid)
        self.assertTrue(restored.is_grid)
        assert_results_equal(self, self.grid.select(1), restored.select(1))
        self.assertFalse(restored.select(0).is_grid)

    def test_one_sided_after_restore(self):
        restored = self._restored(self.two_sample)
        assert_results_equal(self, self.two_sample.one_sided("negative"),
                             restored.one_sided("negative"))
        with self.assertRaises(ValueError):
            restored.one_sided("bogus")


if __name__ == "__main__":
    unittest.main()
