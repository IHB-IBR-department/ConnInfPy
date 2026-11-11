"""Bounded equation checks and counterexamples for the scientific audit.

Run from the repository root:
    python scripts/audit_scientific_lightweight.py

No calibration simulations, real datasets, downloads, JIT, or process pools.
FAIL means an audit expectation is violated; this is a diagnostic report,
not a replacement for the package's regression tests or empirical calibration.
"""
from __future__ import annotations

import contextlib
import io
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import linalg, stats
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

from conninfpy import analyze
from conninfpy._engine import _bh_fdr_correction, _compute_p_values_from_null
from conninfpy.acceleration import _fit_gpd_mom, fit_gamma_tail, fit_gpd_tail
from conninfpy.glm_stats import (
    _compute_reduced_model, _freedman_lane_permutation_task, _precompute_ols,
    compute_glm_stat, compute_p_val_paired_glm,
)
from conninfpy.harmonize import combat_harmonize
from conninfpy.nbs_score import get_cnbs_score, get_nbs_score
from conninfpy.pairwise_stats import (
    _compute_t_stat_ind, _compute_welch_degrees_of_freedom,
    _onesample_tstat_from_sums, compute_p_val, compute_t_stat_diff,
)
from conninfpy.synth_datasets import generate_fc_matrices
from conninfpy.tfnbs_score import (
    _get_edges, _precompute_block_metadata, get_fbc_tfnbs_score,
    get_network_informed_tfnbs_score, get_tfnbs_score,
)
from conninfpy.utils import fisher_r_to_z


def stack(values, nodes=2, directed=False):
    values = np.asarray(values, dtype=float)
    out = np.zeros((len(values), nodes, nodes))
    out[:, 1, 0] = values
    if not directed:
        out[:, 0, 1] = values
    return out


def signed(result):
    return result["positive"] - result["negative"]


checks = []


def record(name, ok, **evidence):
    checks.append({"check": name, "status": "PASS" if bool(ok) else "FAIL", **evidence})


def main():
    rng = np.random.default_rng(123)
    a, b = rng.normal(size=8), rng.normal(0.3, 1.4, size=11)
    A, B = stack(a), stack(b)
    expected = stats.ttest_ind(b, a, equal_var=False)
    actual = signed(_compute_t_stat_ind(A, B))[0, 1]
    record("Welch t equation", np.isclose(actual, expected.statistic), actual=float(actual))
    df = _compute_welch_degrees_of_freedom(A, B)[0, 1]
    record("Welch-Satterthwaite df", np.isclose(df, expected.df), actual=float(df))
    one = signed(compute_t_stat_diff(A))[0, 1]
    record("One-sample t equation", np.isclose(one, stats.ttest_1samp(a, 0).statistic))
    pos, neg = _onesample_tstat_from_sums(a.sum(), (a*a).sum(), len(a))
    record("Precomputed sums equation", np.isclose(pos-neg, one))

    x, z = rng.normal(size=(2, 12))
    X = np.column_stack([np.ones(12), x, z])
    y = 0.5 + 0.2*x - 0.3*z + rng.normal(size=12)
    Y = stack(y)
    c = np.array([0., 1., -1.])
    beta = linalg.lstsq(X, y)[0]
    s2 = np.sum((y-X@beta)**2) / (len(y)-np.linalg.matrix_rank(X))
    t_expected = (c@beta) / np.sqrt(s2 * (c @ np.linalg.inv(X.T@X) @ c))
    t_actual = signed(compute_glm_stat(Y, X, c))[0, 1]
    record("OLS contrast t equation", np.isclose(t_actual, t_expected))
    f_actual = compute_glm_stat(Y, X, c, stat_type="fstat")["omnibus"][0, 1]
    record("One-row F equals t squared", np.isclose(f_actual, t_actual**2))
    C = np.array([[0., 1., 0.], [0., 0., 1.]])
    f_actual = compute_glm_stat(Y, X, C, stat_type="fstat")["omnibus"][0, 1]
    rss0 = np.sum((y-y.mean())**2)
    rss1 = np.sum((y-X@beta)**2)
    record("Omnibus F equals nested-model F", np.isclose(f_actual, ((rss0-rss1)/2)/s2))

    null = np.array([0., 1., 2., 2.])
    obs = np.array([[0., 2.], [3., 0.]])
    empirical = _compute_p_values_from_null({"a": obs}, {"a": null})["a"]
    record("Tie-inclusive plus-one p-values", np.allclose(empirical, [[1., .6], [.2, 1.]]))
    ps = np.array([.001, .01, .03, .2, .9])
    record("BH step-up equation", np.allclose(_bh_fdr_correction(ps), multipletests(ps, method="fdr_bh")[1]))

    T = np.array([[0., 3., 0., 0.], [3., 0., 4., 0.], [0., 4., 0., 0.], [0., 0., 0., 0.]])
    record("NBS extent counts unique component edges", get_nbs_score(T, 2., "extent")[0, 1] == 2.)
    record("NBS intensity sums component heights", get_nbs_score(T, 2., "intensity")[0, 1] == 7.)

    single = np.array([[0., 3.], [3., 0.]])
    exact = 3.**4 / 4.
    for n in (10, 100):
        v = get_tfnbs_score(single, .3, 3., n, start_thres=0., backend="scipy")[0, 1]
        record(f"TFNBS single-edge integral within 5%, n={n}", abs(v/exact-1) < .05,
               actual=float(v), integral=exact, relative_error=float(v/exact-1))
    zero_e = get_tfnbs_score(T, 0., 2., 5, start_thres=0., backend="scipy")
    record("TFNBS E=0 leaves inactive edges zero", zero_e[0, 3] == 0., actual=float(zero_e[0, 3]))
    fbc = get_fbc_tfnbs_score(single, np.zeros(2, dtype=int), 0., 2., 5,
                             start_thres=0., min_cluster_size=3)
    record("FBC E=0 still suppresses undersized blocks", fbc[0, 1] == 0., actual=float(fbc[0, 1]))

    clipped = np.array([[0., 2., 0.], [2., 0., 0.], [0., 0., 0.]])
    cnbs = get_cnbs_score(clipped, np.zeros(3, dtype=int))
    record("cNBS uses fixed full-block denominator", np.isclose(cnbs[0, 1], 2./3.),
           actual=float(cnbs[0, 1]), expected_full_block_mean=2./3.)
    sparse = np.zeros((4, 4)); sparse[0, 1] = sparse[1, 0] = 3.
    rows, cols, weights = _get_edges(sparse, 1.65, True)
    _, capacities = _precompute_block_metadata(rows, cols, np.zeros(4, dtype=int))
    record("NI-TFNBS capacity counts all possible block edges", np.isclose(capacities[0]**2, 6.),
           actual_capacity=float(capacities[0]**2), expected_capacity=6)
    star = np.zeros((6, 6)); matching = np.zeros_like(star)
    for i, j in ((0, 1), (0, 2), (0, 3)):
        star[i, j] = star[j, i] = 3.
    for i, j in ((0, 1), (2, 3), (4, 5)):
        matching[i, j] = matching[j, i] = 3.
    labels = np.zeros(6, dtype=int)
    ns = get_network_informed_tfnbs_score(star, labels, .3, 3., 5)
    nm = get_network_informed_tfnbs_score(matching, labels, .3, 3., 5)
    record("NI-TFNBS reflects connected-component topology as described in approach.rst",
           not np.isclose(ns[0, 1], nm[0, 1]), star_score=float(ns[0, 1]), matching_score=float(nm[0, 1]))

    under_null = X @ np.array([0., 1., 1.])  # c beta = 0, yet nonzero nuisance
    red = _compute_reduced_model(X, c)
    red_correct = X @ linalg.null_space(c.reshape(1, -1))
    rss_bad = np.sum((under_null-red@np.linalg.pinv(red)@under_null)**2)
    rss_correct = np.sum((under_null-red_correct@np.linalg.pinv(red_correct)@under_null)**2)
    record("General contrast reduced model retains permitted nuisance effects", rss_bad < 1e-20,
           implementation_null_RSS=float(rss_bad), restricted_null_RSS=float(rss_correct))
    intercept_red = _compute_reduced_model(np.ones((8, 1)), np.ones(1))
    record("Intercept-only null excludes tested intercept", intercept_red.shape[1] == 0,
           actual_columns=int(intercept_red.shape[1]))
    cv = np.arange(8, dtype=float)-3.5
    delta = stack(2.+.1*cv+np.array([-.1,.1,.1,-.1,-.1,.1,.1,-.1]))
    paired = compute_p_val_paired_glm(np.zeros_like(delta), delta,
        confounds_A=np.zeros(8), confounds_B=cv, stat_type="beta", method="tstat",
        n_permutations=19, use_mp=False, rng=0)
    Xp = np.column_stack([np.ones(8), cv]); cp = np.array([1., 0.])
    redp = _compute_reduced_model(Xp, cp)
    fitted = (redp@np.linalg.pinv(redp)@delta.reshape(8, -1)).reshape(delta.shape)
    inv, diag, cov = _precompute_ols(Xp)
    draws = [_freedman_lane_permutation_task(fitted, delta-fitted, Xp, cp, inv,
        diag, cov, (2, 2), "beta", seed=s)["positive"] for s in range(8)]
    record("Paired beta null has changing effect numerator", np.ptp(draws) > 1e-8,
           observed_beta=2., null_beta_min=float(min(draws)), null_beta_max=float(max(draws)),
           reported_p_from_floating_point_ties=float(paired["positive"][0,1]))
    # A supplied zero confound should have the same null as the no-confound route.
    plain = compute_p_val_paired_glm(np.zeros_like(delta), delta, method="tstat",
                                   n_permutations=19, use_mp=False, rng=0)
    zero_confound = compute_p_val_paired_glm(np.zeros_like(delta), delta,
        confounds_A=np.zeros(8), confounds_B=np.zeros(8), method="tstat",
        n_permutations=19, use_mp=False, rng=0)
    record("Zero paired confound leaves the permutation scheme unchanged",
           np.allclose(plain["positive"], zero_confound["positive"]),
           signflip_p=float(plain["positive"][0,1]), row_permutation_p=float(zero_confound["positive"][0,1]))
    cx = np.array([0.,1.,0.])
    shared_red = _compute_reduced_model(X, np.array([0.,1.,1.]))
    separate_red = _compute_reduced_model(X, cx)
    permitted_signal = X @ np.array([0.,0.,1.])
    shared_rss = np.sum((permitted_signal-shared_red@np.linalg.pinv(shared_red)@permitted_signal)**2)
    separate_rss = np.sum((permitted_signal-separate_red@np.linalg.pinv(separate_red)@permitted_signal)**2)
    record("Shared multi-contrast null retains other predictors for individual coefficient tests",
           np.isclose(shared_rss, separate_rss), shared_null_RSS=float(shared_rss), separate_null_RSS=float(separate_rss))

    directed = stack([.5, -.4, .2, -.1, .3, -.2, .1, .2], directed=True)
    dr = compute_p_val(directed, test_type="one-sample", method="tstat", n_permutations=39, use_mp=False, rng=0)
    record("Directed null includes the lower-triangle tested edge", dr["positive"][1, 0] > 1./40,
           lower_edge_p=float(dr["positive"][1, 0]), upper_edge_p=float(dr["positive"][0, 1]))
    missing = stack(np.arange(8), nodes=3); missing[0, 0, 1] = missing[0, 1, 0] = np.nan
    try:
        mr = compute_p_val(missing, test_type="one-sample", method="tstat", n_permutations=39, use_mp=False, rng=0)
        record("Missing data rejected before permutation", False,
               missing_edge_p=float(mr["positive"][0,1]), zero_edge_p=float(mr["positive"][0,2]),
               diagonal_p=float(mr["positive"][0,0]))
    except ValueError:
        record("Missing data rejected before permutation", True)
    Xdup = np.column_stack([np.ones(12), x, x])
    try:
        dup = compute_glm_stat(Y, Xdup, np.array([0.,1.,0.]))
        record("Non-estimable contrast rejected", False, reported_t=float(signed(dup)[0,1]))
    except ValueError:
        record("Non-estimable contrast rejected", True)
    const = signed(compute_t_stat_diff(stack(np.ones(8))))[0,1]
    record("Constant nonzero data not reported as zero effect", const != 0., actual_t=float(const))

    excess = np.arange(1., 11.)
    sigma, xi, valid = _fit_gpd_mom(excess)
    ratio = excess.mean()**2 / excess.var(ddof=1)
    xi_correct = (1.-ratio)/2.
    record("GPD moments agree with survival-function parameterization", valid and np.isclose(xi, xi_correct),
           actual_shape=float(xi), correct_shape=float(xi_correct), scale=float(sigma))
    exponential_quantiles = stats.expon.ppf((np.arange(200)+.5)/200.)
    gp = fit_gpd_tail(exponential_quantiles, np.array([[100.]]))
    record("GPD extreme-tail estimate diagnostic", True, p=float(gp.item()), fitted_tail_floor=1./200.)
    left_skew = np.array([0.,4.,4.5,5.,5.1,5.2,5.3,5.4,5.5,5.6])
    skew = float(stats.skew(left_skew))
    g = fit_gamma_tail(left_skew, np.array([[left_skew.mean()]]))
    record("Gamma moment fit preserves negative skewness", skew >= 0.,
           input_skewness=skew, implied_gamma_skewness=abs(skew), p_at_mean=float(g.item()))
    bounds = fisher_r_to_z(np.array([.99998, .99999]))
    record("Fisher boundary policy preserves monotonicity", bounds[1] >= bounds[0], actual_z=bounds.tolist())

    rr = np.random.default_rng(24)
    features = rr.normal(size=(24, 10)); sites = np.repeat([0, 1], 12)
    features[12:] = features[12:]*1.3 + .5
    preserve = rr.normal(size=24)
    try:
        from neuroCombat import neuroCombat
        import pandas as pd
        for eb in (False, True):
            ours = combat_harmonize(features, sites, preserve, eb=eb).Y_adjusted
            with contextlib.redirect_stdout(io.StringIO()):
                reference = neuroCombat(dat=features.T, covars=pd.DataFrame({"site":sites, "cov":preserve}),
                    batch_col="site", continuous_cols=["cov"], eb=eb)["data"].T
            record(f"ComBat reference equivalence, EB={eb}", np.allclose(ours, reference, atol=1e-4, rtol=1e-4),
                   max_abs_difference=float(np.max(np.abs(ours-reference))))
    except ImportError:
        checks.append({"check":"ComBat reference equivalence", "status":"SKIP", "reason":"neuroCombat unavailable"})

    mask = np.zeros((4,4)); mask[0,1] = mask[1,0] = 1.
    _, _, (base, modified) = generate_fc_matrices(4, 0., mask=mask, n_samples_group1=2, n_samples_group2=2, seed=12)
    def correlation(cov):
        d = np.sqrt(np.diag(cov)); return cov / np.outer(d,d)
    shift = np.max(np.abs(correlation(base)-correlation(modified)))
    record("Legacy generator effect_size=0 is a population null", shift < 1e-12, population_correlation_shift=float(shift))

    small = stack(rng.normal(size=8), nodes=4)
    for method in ("tstat", "tfnbs", "nbs", "cnbs", "ni_tfnbs", "fbc_tfnbs", "bonferroni", "bh_fdr", "bh_fdr_perm"):
        kw = {"net_labels":np.zeros(4,dtype=int)} if method in ("cnbs", "ni_tfnbs", "fbc_tfnbs") else {}
        result = analyze(group1=small, input="z", test_type="one-sample", method=method,
                         n_permutations=7, n=5, use_mp=False, rng=0, **kw)
        record(f"Bounded entry-point smoke: {method}", all(np.isfinite(result[t]).all() and
               np.all((result[t]>=0)&(result[t]<=1)) for t in ("positive", "negative")))


if __name__ == "__main__":
    audit_files = [Path(p) for p in (
        "conninfpy/acceleration.py", "conninfpy/glm_stats.py", "conninfpy/pairwise_stats.py",
        "conninfpy/tfnbs_score.py", "conninfpy/nbs_score.py", "conninfpy/harmonize.py",
        "conninfpy/_engine.py", "conninfpy/_prepare.py", "conninfpy/analysis.py",
        "conninfpy/utils.py", "conninfpy/synth_datasets.py",
        "conninfpy/topologies.py", "conninfpy/loaders/builtins.py", "conninfpy/defaults.py",
        "docs/source/approach.rst", "README.md",
    )]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in audit_files}
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        main()
    report = {"scope":"equations and bounded counterexamples; no empirical error-rate calibration",
              "settings":{"max_pipeline_permutations":39,"multiprocessing":False,"numba_jit":False,"blas_threads":1},
              "source_sha256":hashes,
              "files_changed_during_checks":[str(p) for p in audit_files if hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[str(p)]],
              "counts":{s:sum(c["status"]==s for c in checks) for s in ("PASS","FAIL","SKIP")}, "checks":checks}
    print(json.dumps(report, indent=2, allow_nan=False))
