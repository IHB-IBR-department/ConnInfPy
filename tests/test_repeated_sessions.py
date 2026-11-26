"""NETSTAT-38 regression: the narrow balanced complete repeated-sessions
omnibus F recipe (design D1).

Verified design D1 (owner-verified; do not redesign)
---------------------------------------------------
Balanced, complete within-subject session factor, ``k >= 3`` sessions,
``n >= 2`` subjects, data stacked subject-major: row ``r = s*k + j`` holds
subject ``s``, session ``j``. ``Y`` is ``(n*k, N, N)`` on the Fisher-z
scale, symmetric with zero diagonal. ``analyze()`` has no >= 3-session
path, so the advanced API :func:`conninfpy.compute_p_val_glm` is used.

Full design ``X = [S | T]``: ``S`` one-hot subject dummies (``n`` columns,
they span the constant, so no extra intercept) and ``T`` ``k-1``
treatment-coded session columns (session ``j`` vs the reference session 0),
``rank(X) = n + k - 1``. Subject fixed effects are in BOTH the full and the
reduced model. The omnibus contrast is ``C = [0 | I_{k-1}]`` with
``stat_type='fstat'``, giving a single ``'omnibus'`` p-map: joint equality
of session means, numerator df ``k-1``, residual df ``(n-1)(k-1)``. The
reduced design is ``X null(C) = span(S)``, so its projector is the
within-subject averaging operator and the reduced residuals sum to zero
within each subject. ``strata`` carries the subject code per row; the
permutation is WITHIN each subject block only -- there is no whole-block
exchangeability anywhere in the engine. ``scheme='auto'`` resolves to
``'permutation'`` because the reduced model spans the constant; sign flips
are not the scheme of this design.

Two numerical layers, deliberately kept separate
------------------------------------------------
* **Algebra layer** (independent nested-RSS operation order): ranks,
  projectors, residual sums, df, and the observed F against a manual
  RSS-difference OLS F with a tolerance of ``1e-8``. Never asserted as
  exact count equality.
* **Mechanism layer** (the engine's own operation order: quadratic form
  over ``sigma2``): seeded replay with independent within-block index
  draws reproduces ``exceedance_counts`` and ``p`` EXACTLY against a
  separate :func:`conninfpy.compute_p_val_glm` run.

Exhaustive enumeration uses the exact randomization reference c/G
(``G = (k!)^n``, identity included); the +1/(B+1) rule belongs to the
engine's Monte Carlo sampling. The number of assignments ``(k!)^n`` is not
the number of distinct null values: ``k!`` common session relabellings tie
structurally for any data, and data-dependent near-ties add more; there is
no universal bound on the resulting p difference.

The boundary between the layers is real: the ``#{null >= obs}`` count is
discontinuous at exact ties (the identity draw always ties in real
arithmetic), and two algebraically equivalent F formulas differ by ~1e-14
in floating point, enough to land on opposite sides of a tie. Hence exact
count/p equality is only asserted for the engine-order replica; the
independent RSS-order null is compared to the engine p with the
pre-declared Monte-Carlo tolerance below, and its OBSERVED F with the 1e-8
algebra tolerance.

Scope and non-claims
--------------------
Synthetic data only, deterministic seeds, no fixtures from ``datasets/``.
Everything here is a *mechanism* check: test
``test_engine_p_consistent_with_exact_enumeration`` is a consistency check
of the sampling machinery against exhaustive enumeration, NOT a
calibration study -- no FWER or power claim is made or implied. The recipe
rests on invariance of the JOINT distribution of the ENTIRE error array
(all subjects, all edges) under the product of independent within-subject
session permutations; a sufficient case is independent subjects with
exchangeable session-indexed edge vectors within each subject (laws may
differ across subjects). Per-edge within-subject exchangeability alone is
not enough for the max-F null across edges (common session shock
counterexample: U_j + V_sj with U_j shared). Compound symmetry of the
covariance alone is a second-moment property and is not sufficient. In D1
the Freedman-Lane reconstruction equals the within-block row permutation of
the data in exact arithmetic (Y* = P Y); the floating-point implementation
reconstructs Yhat0 + R0[perm] and can differ by rounding. In general,
Freedman-Lane is approximate. The designs that are NOT covered are
documented, not tested, in
``docs/source/workflows.rst`` (:ref:`wf-repeated-sessions`): pairwise /
single-session contrasts on the stack (stay with the NETSTAT-16/31
``test_type='paired'`` contract), between-subject effects, group x session
interaction, temporally autocorrelated within-subject errors, mixed /
random-effects models, and unbalanced or missing-session data
(``missing='available_case'`` is not verified for D1).
"""
from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

from conninfpy.stats import _degenerate
from conninfpy.stats._design import _residual_degrees_of_freedom
from conninfpy.stats._engine import _encode_strata
from conninfpy.stats.glm import (
    _compute_reduced_model,
    _precompute_ols,
    _resolve_scheme,
    compute_glm_stat,
    compute_p_val_glm,
)

# Tolerance of the independent (nested-RSS) algebra layer against the engine
# F statistic. Floating-point operation-order differences, never a modelling
# tolerance.
F_ALG_TOL = 1e-8

# Monte-Carlo consistency tolerance of the p-value mechanism against exhaustive
# enumeration: 4 * SE(p) + 1/(B+1), pre-declared in the packet.
MC_TOL_FACTOR = 4.0

N_PERMUTATIONS = 2000
MASTER_SEED = 42


# =============================================================================
# Design D1 helpers
# =============================================================================

def _design_d1(n, k):
    """Subject-major D1 design: subject dummies, treatment-coded sessions."""
    subj = np.repeat(np.arange(n), k)
    sess = np.tile(np.arange(k), n)
    S = np.eye(n)[subj]                                     # spans constant
    T = np.column_stack([(sess == j).astype(np.float64) for j in range(1, k)])
    X = np.column_stack([S, T])
    C = np.column_stack([np.zeros((k - 1, n)), np.eye(k - 1)])
    return subj, sess, S, T, X, C


def _data_d1(n, k, N, seed):
    """Synthetic symmetric zero-diagonal stack with subject and session effects."""
    subj, sess, S, T, X, C = _design_d1(n, k)
    rows = n * k
    rng = np.random.RandomState(seed)
    base = rng.normal(0, 0.3, size=(rows, N, N))
    Y = base + base.transpose(0, 2, 1)
    Y += rng.normal(0, 0.5, size=n)[subj][:, None, None]    # subject effects
    eff = np.zeros((k, N, N))
    eff[1, 0, 1] = eff[1, 1, 0] = 0.8                       # session 1 vs 0
    eff[2, 2, 3] = eff[2, 3, 2] = -0.6                      # session 2 vs 0
    Y += eff[sess]
    for r in range(rows):
        np.fill_diagonal(Y[r], 0.0)
    return Y, X, C, subj, S, T


def _edges(A, iu):
    return np.ascontiguousarray(A[:, iu[0], iu[1]])


def _f_rss_order(X, X0, k, df_res):
    """Independent algebra layer: F from nested RSS differences."""
    Xp = np.linalg.pinv(X)
    X0p = np.linalg.pinv(X0)

    def f(Ye_perm):
        b = Xp @ Ye_perm
        rss1 = ((Ye_perm - X @ b) ** 2).sum(axis=0)
        b0 = X0p @ Ye_perm
        rss0 = ((Ye_perm - X0 @ b0) ** 2).sum(axis=0)
        return ((rss0 - rss1) / (k - 1)) / (rss1 / df_res)

    return f


def _f_engine_order(X, C, df_res, k):
    """Mechanism layer: replica of the engine's F operation order.

    Quadratic form ``Cb' (C XtX^-1 C')^-1 Cb`` divided by ``(k-1) * sigma2``
    with the same degenerate-edge floor and negative clamp as the engine.
    """
    X_pinv, _, XtX_inv = _precompute_ols(X)
    M_inv = np.linalg.inv(C @ XtX_inv @ C.T)

    def f(Ye_perm):
        beta = X_pinv @ Ye_perm
        Cb = C @ beta
        resid = Ye_perm - X @ beta
        sigma2 = np.sum(resid * resid, axis=0) / df_res
        meansq = np.einsum("ij,ij->j", Ye_perm, Ye_perm) / Ye_perm.shape[0]
        floor = _degenerate.RTOL * meansq
        quad = np.einsum("ie,ij,je->e", Cb, M_inv, Cb)
        with np.errstate(divide="ignore", invalid="ignore"):
            F = quad / ((k - 1) * sigma2)
        F = np.where(sigma2 <= floor, 0.0, F)
        return np.where(F < 0, 0.0, F)

    return f


def _independent_block_perm(codes, draw):
    """Short independent loop over subject blocks (ascending code order).

    Deliberately does NOT call the production ``_stratified_perm``: the point
    is to reproduce the within-block index order from the documented
    contract (``rng.permutation`` per block of size > 1, ascending codes).
    """
    perm = np.arange(codes.size)
    for s in sorted(set(int(c) for c in codes)):
        idx = np.flatnonzero(codes == s)
        if idx.size > 1:
            perm[idx] = draw.permutation(idx)
    return perm


# =============================================================================
# 1. Design algebra (independent layer, tolerances)
# =============================================================================

class TestRepeatedSessionsDesignAlgebra(unittest.TestCase):
    """Design D1 algebra at n=3, k=3, N=4."""

    n, k, N = 3, 3, 4

    @classmethod
    def setUpClass(cls):
        cls.Y, cls.X, cls.C, cls.subj, cls.S, cls.T = _data_d1(
            cls.n, cls.k, cls.N, seed=20261124
        )
        cls.rows = cls.n * cls.k
        cls.iu = np.triu_indices(cls.N, 1)
        cls.X0 = _compute_reduced_model(cls.X, cls.C)
        cls.df_res = _residual_degrees_of_freedom(cls.X)
        cls.Ye = _edges(cls.Y, cls.iu)
        cls.stat = compute_glm_stat(
            cls.Y, cls.X, cls.C, stat_type="fstat"
        )

    def test_01_ranks_full_and_reduced(self):
        self.assertEqual(np.linalg.matrix_rank(self.X), self.n + self.k - 1)
        self.assertEqual(np.linalg.matrix_rank(self.X0), self.n)

    def test_02_reduced_space_is_subject_dummies(self):
        P_S = self.S @ np.linalg.pinv(self.S)
        P_X0 = self.X0 @ np.linalg.pinv(self.X0)
        np.testing.assert_allclose(P_X0, P_S, rtol=0.0, atol=1e-10)

    def test_03_reduced_fit_is_subject_mean_operator(self):
        """P_X0 Y equals the within-subject mean replicated per row."""
        Y_flat = self.Y.reshape(self.rows, -1)
        PY = (self.X0 @ (np.linalg.pinv(self.X0) @ Y_flat)).reshape(
            self.rows, self.N, self.N
        )
        H0Y = np.stack(
            [self.Y[self.subj == s].mean(axis=0) for s in range(self.n)]
        )[self.subj]
        np.testing.assert_allclose(PY, H0Y, rtol=0.0, atol=1e-10)

    def test_04_reduced_residuals_sum_to_zero_within_subject(self):
        Y_flat = self.Y.reshape(self.rows, -1)
        PY = (self.X0 @ (np.linalg.pinv(self.X0) @ Y_flat)).reshape(
            self.rows, self.N, self.N
        )
        R0 = self.Y - PY
        sums = np.stack(
            [R0[self.subj == s].sum(axis=0) for s in range(self.n)]
        )
        self.assertLess(np.max(np.abs(sums)), 1e-10)

    def test_05_auto_scheme_is_permutation(self):
        self.assertEqual(_resolve_scheme(self.X, self.X0, "auto"), "permutation")

    def test_06_observed_f_and_df(self):
        """Observed F == manual RSS-difference OLS F (algebra tolerance)."""
        self.assertEqual(self.df_res, (self.n - 1) * (self.k - 1))
        F_obs = self.stat["omnibus"]
        F_rss = _f_rss_order(self.X, self.X0, self.k, self.df_res)(self.Ye)
        nested = np.zeros((self.N, self.N))
        nested[self.iu] = F_rss
        nested = nested + nested.T
        self.assertLess(np.max(np.abs(F_obs - nested)), F_ALG_TOL)


# =============================================================================
# 2. Null space, seeded replay, guards (n=4, k=3, N=5)
# =============================================================================

class TestRepeatedSessionsNullMechanism(unittest.TestCase):
    """Exhaustive enumeration and seeded replay for design D1."""

    n, k, N = 4, 3, 5

    @classmethod
    def setUpClass(cls):
        cls.Y, cls.X, cls.C, cls.subj, cls.S, cls.T = _data_d1(
            cls.n, cls.k, cls.N, seed=20261124
        )
        cls.rows = cls.n * cls.k
        cls.iu = np.triu_indices(cls.N, 1)
        cls.X0 = _compute_reduced_model(cls.X, cls.C)
        cls.df_res = _residual_degrees_of_freedom(cls.X)
        cls.Ye = _edges(cls.Y, cls.iu)
        cls.F_obs = compute_glm_stat(
            cls.Y, cls.X, cls.C, stat_type="fstat"
        )["omnibus"]

        # Reduced-model fit and residuals (the objects that get permuted).
        cls.PY = (
            cls.X0 @ (np.linalg.pinv(cls.X0) @ cls.Y.reshape(cls.rows, -1))
        ).reshape(cls.rows, cls.N, cls.N)
        cls.R0 = cls.Y - cls.PY
        cls.yhat = _edges(cls.PY, cls.iu)
        cls.resid = cls.Ye - cls.yhat

        # Engine-order replica applied to the observed data.
        cls.f_fast = staticmethod(
            _f_engine_order(cls.X, cls.C, cls.df_res, cls.k)
        )
        cls.f_rss = staticmethod(_f_rss_order(cls.X, cls.X0, cls.k, cls.df_res))
        cls.F_obs_engine_order = cls.f_fast(cls.Ye)
        cls.F_obs_rss_order = cls.f_rss(cls.Ye)

        # One reference API run shared by the enumeration test.
        cls.api = compute_p_val_glm(
            cls.Y,
            design_matrix=cls.X,
            contrast=cls.C,
            stat_type="fstat",
            method="tstat",
            n_permutations=N_PERMUTATIONS,
            acceleration=None,
            rng=MASTER_SEED,
            use_mp=False,
            strata=cls.subj,
            missing="raise",
        )
        cls.codes = _encode_strata(cls.subj)

    def test_01_engine_order_replica_matches_observed_f(self):
        """Mechanism layer: replica of the engine F is bit-comparable."""
        self.assertLess(
            np.max(np.abs(self.F_obs_engine_order - self.F_obs[self.iu])),
            1e-12,
        )

    def test_02_exact_enumeration_and_engine_consistency(self):
        """Exhaustive (k!)^n null; engine p agrees within the MC tolerance.

        Mechanism consistency only: this compares the sampling machinery with
        exhaustive enumeration. It is not a FWER or power calibration study.
        """
        blocks = [np.flatnonzero(self.codes == s) for s in np.unique(self.codes)]
        block_perms = [list(itertools.permutations(b)) for b in blocks]
        draws = []
        for combo in itertools.product(*block_perms):
            perm = np.concatenate([np.array(p) for p in combo])
            Yperm = self.PY + self.R0[perm]
            # Subject means cannot move under a within-block draw.
            got = np.stack(
                [Yperm[self.subj == s].mean(axis=0) for s in range(self.n)]
            )
            want = np.stack(
                [self.Y[self.subj == s].mean(axis=0) for s in range(self.n)]
            )
            self.assertLess(np.max(np.abs(got - want)), 1e-9)
            draws.append(self.f_rss(_edges(Yperm, self.iu)).max())
        draws = np.asarray(draws)
        self.assertEqual(draws.size, math.factorial(self.k) ** self.n)
        self.assertEqual(draws.size, 1296)

        # Exhaustive enumeration of all G=(k!)^n assignments INCLUDING the
        # identity gives the exact randomization reference c/G; the
        # (b+1)/(B+1) rule belongs to the engine's Monte Carlo sampling, not
        # to the enumeration reference. Counts/p use the independent RSS-order
        # null and the RSS-order observed F: never compared for exact equality
        # with the engine.
        exact_counts = np.array(
            [(draws >= f).sum() for f in self.F_obs_rss_order]
        )
        exact_p = exact_counts / draws.size

        engine_p = self.api["omnibus"][self.iu]
        tol = (
            MC_TOL_FACTOR
            * np.sqrt(exact_p * (1.0 - exact_p) / (N_PERMUTATIONS + 1))
            + 1.0 / (N_PERMUTATIONS + 1)
        )
        self.assertTrue(
            np.all(np.abs(engine_p - exact_p) <= tol),
            msg=(
                "engine p vs exhaustive-enumeration p outside the pre-declared "
                f"MC tolerance: max dev {np.max(np.abs(engine_p - exact_p)):.4f}, "
                f"max tol {tol.max():.4f}"
            ),
        )

    def test_03_seeded_replay_reproduces_counts_and_p_exactly(self):
        """Mechanism layer: independent indices + engine F order -> exact."""
        seeds = np.random.RandomState(MASTER_SEED).randint(
            0, 2 ** 32 - 1, size=N_PERMUTATIONS, dtype=np.int64
        )
        perms = [
            _independent_block_perm(self.codes, np.random.RandomState(int(sd)))
            for sd in seeds
        ]
        # Every draw stays inside subject blocks and preserves each block.
        for perm in perms:
            np.testing.assert_array_equal(self.codes[perm], self.codes)

        null = np.array(
            [self.f_fast(self.resid[p] + self.yhat).max() for p in perms]
        )
        counts = np.array(
            [(null >= f).sum() for f in self.F_obs_engine_order]
        )
        p = (counts + 1) / (N_PERMUTATIONS + 1)

        # A separate API run, not reused from setUpClass.
        api = compute_p_val_glm(
            self.Y,
            design_matrix=self.X,
            contrast=self.C,
            stat_type="fstat",
            method="tstat",
            n_permutations=N_PERMUTATIONS,
            acceleration=None,
            rng=MASTER_SEED,
            use_mp=False,
            strata=self.subj,
            missing="raise",
        )
        self.assertTrue(api.strata_provided)
        np.testing.assert_array_equal(
            api.exceedance_counts["omnibus"][self.iu], counts
        )
        np.testing.assert_array_equal(api["omnibus"][self.iu], p)
        np.testing.assert_array_equal(api["omnibus"][self.iu], self.api["omnibus"][self.iu])

    def test_04_guards(self):
        with self.assertRaises(ValueError):
            compute_p_val_glm(
                self.Y,
                design_matrix=self.X,
                contrast=self.C,
                stat_type="fstat",
                method="tstat",
                n_permutations=10,
                acceleration=None,
                rng=1,
                use_mp=False,
                strata=self.subj[:-1],
                missing="raise",
            )

        Ynan = self.Y.copy()
        Ynan[0, 0, 1] = np.nan
        Ynan[0, 1, 0] = np.nan
        with self.assertRaises(ValueError):
            compute_p_val_glm(
                Ynan,
                design_matrix=self.X,
                contrast=self.C,
                stat_type="fstat",
                method="tstat",
                n_permutations=10,
                acceleration=None,
                rng=1,
                use_mp=False,
                strata=self.subj,
                missing="raise",
            )


if __name__ == "__main__":
    unittest.main()
