"""Fast, deterministic FWER regression check for CI (nightly / manual).

Run from the repository root::

    python scripts/ci_fwer_check.py                  # ~1 min, fixed seed
    python scripts/ci_fwer_check.py --n-null 100     # quicker smoke run
    python scripts/ci_fwer_check.py --csv out.csv --json out.json

What is checked
---------------
Under H0 (no planted effect) every core inference method must keep the
empirical FWER of the **joint two-sided family** at or below ``alpha``::

    FWER_joint = P( any edge with min(p_pos, p_neg) <= alpha | H0 )  <=  alpha

which is the quantity reported as ``fwer_two_sided_joint`` in
``examples/simulations/results/tables/v2_glm_fwer_summary.csv`` and the family
the ConnInfPy 2.0 contract holds at ``alpha`` (``two_tailed=True``: both p-maps
are corrected against the joint max-statistic null).

Arms (each applied to the same simulated null datasets within a run):

* ``tstat``, ``tfnbs`` (exact piecewise integration, ``n=None``, E=0.4, H=3.0,
  start_thres=1.65), ``nbs`` (extent, ``threshold=3.0``) and ``cnbs``
  (Bonferroni, 4-module labels) through ``compute_p_val`` (two-sample).
* ``glm_tfnbs`` through ``compute_p_val_glm`` with ``interest`` plus site
  dummies and one noise covariate as ``confounds`` (``two_tailed=True``), on a
  multi-site dataset with additive site shifts.

Permutation p-values are exact (no tail acceleration), ``use_mp=False`` and
every random stream is seeded from ``--seed``, so a run is reproducible on a
given platform.

Decision rule
-------------
For each arm the number of null runs with any false positive is
``k ~ Binomial(n_null, FWER_joint)``. With ``n = n_null``:

* ``p_hi = binom.sf(k - 1, n, alpha)`` is the exact one-sided binomial
  p-value of H1 ``FWER_joint > alpha``.
* FAIL (exit 1) if any arm has ``p_hi < 0.005``, i.e. ``k`` above the 99.5 %
  quantile of ``Binomial(n, alpha)``. This wide per-method sanity bound keeps
  the family-wise false-alarm of the check at most ``0.5 % x n_arms``
  (Bonferroni; valid under the dependence induced by the shared datasets).
  Holm-adjusted p-values are reported for awareness.
* The central 99 % exact binomial interval around ``alpha`` and the 99 %
  Clopper-Pearson interval of each arm are printed. An arm below the interval
  is *conservative* (valid, possibly underpowered) and only noted, never
  failed: the check is about validity, not exact equality. ``cnbs`` with
  Bonferroni correction has a smallest attainable p-value of
  ``K / (n_permutations + 1)`` (= 0.05 for 10 blocks x 199 permutations), so it
  can only ever reject at that boundary; this coarse resolution is expected
  at CI scale and is not a failure.

Exit codes: ``0`` PASS, ``1`` FAIL, ``2`` usage or environment error.
On GitHub Actions the verdict is also emitted as workflow annotations and a
step summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Tiny problems run fastest (and most predictably on 2-core runners) with
# single-threaded BLAS; an explicit setting from the caller wins.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import stats

ALPHA = 0.05
FAIL_P = 0.005  # per-method one-sided level (99.5 % upper sanity bound)
TWO_SIDED = True
TFNBS_KWARGS = {"e": 0.4, "h": 3.0, "n": None, "start_thres": 1.65}
N_MODULES = 4
PAIRWISE_ARMS = ("tstat", "tfnbs", "nbs", "cnbs")
GLM_ARMS = ("glm_tfnbs",)
ALL_ARMS = PAIRWISE_ARMS + GLM_ARMS


# --------------------------------------------------------------------------- #
# Statistics of the decision rule (pure functions; unit-tested separately)
# --------------------------------------------------------------------------- #
def exact_binomial_interval(n: int, p: float = ALPHA, level: float = 0.99) -> tuple[int, int]:
    """Central ``level`` interval of ``Binomial(n, p)`` as integer counts."""
    lo = int(stats.binom.ppf((1.0 - level) / 2.0, n, p))
    hi = int(stats.binom.ppf(1.0 - (1.0 - level) / 2.0, n, p))
    return lo, hi


def clopper_pearson(k: int, n: int, level: float = 0.99) -> tuple[float, float]:
    """Two-sided Clopper-Pearson interval for ``k / n``."""
    a = (1.0 - level) / 2.0
    low = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    high = 1.0 if k == n else float(stats.beta.ppf(1.0 - a, k + 1, n - k))
    return low, high


def one_sided_p_above(k: int, n: int, p0: float = ALPHA) -> float:
    """Exact binomial p-value of H1: ``p > p0``."""
    return float(stats.binom.sf(k - 1, n, p0))


def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def summarize_arm(arm: str, n_null: int, k_joint: int, seconds: float,
                  k_pos: int = 0, k_neg: int = 0) -> dict:
    """Decision statistics for one arm."""
    n = int(n_null)
    k = int(k_joint)
    p_hi = one_sided_p_above(k, n)
    cp_low, cp_high = clopper_pearson(k, n)
    lo, hi = exact_binomial_interval(n)
    return {
        "arm": arm,
        "n_null": n,
        "k_joint": k,
        "fwer_joint": k / n,
        "p_hi": p_hi,
        "fail": bool(p_hi < FAIL_P),
        "cp99_low": cp_low,
        "cp99_high": cp_high,
        "binom99_k_low": lo,
        "binom99_k_high": hi,
        "inside_binom99": bool(lo <= k <= hi),
        "conservative": bool(cp_high < ALPHA),
        "k_pos": int(k_pos),
        "k_neg": int(k_neg),
        "seconds": seconds,
    }


def finalize(rows: list[dict]) -> tuple[bool, list[dict]]:
    """Add Holm-adjusted p-values and return the overall verdict."""
    adj = holm_adjust([r["p_hi"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = a
    ok = not any(r["fail"] for r in rows)
    return ok, rows


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def _fisher_z(mats: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(mats, -0.999, 0.999))


def _joint_hit(p_pos: np.ndarray, p_neg: np.ndarray, n_nodes: int) -> tuple[bool, int, int]:
    iu = np.triu_indices(n_nodes, k=1)
    pos = np.asarray(p_pos)[iu] <= ALPHA
    neg = np.asarray(p_neg)[iu] <= ALPHA
    hit = bool(np.any(pos | neg))  # == any(min(p_pos, p_neg) <= alpha)
    return hit, int(pos.sum()), int(neg.sum())


def _site_dummies(sites: np.ndarray) -> np.ndarray:
    unique = np.unique(sites)
    out = np.zeros((len(sites), max(len(unique) - 1, 0)))
    for j, s in enumerate(unique[1:]):
        out[sites == s, j] = 1.0
    return out


def run_check(n_null: int, n_permutations: int, n_nodes: int, n_samples: int,
              n_sites: int, seed: int, use_mp: bool, verbose: bool) -> list[dict]:
    from conninfpy import compute_p_val, compute_p_val_glm, generate_fc_matrices, generate_multisite_glm_dataset

    net_labels = np.arange(n_nodes) % N_MODULES
    k = {a: 0 for a in ALL_ARMS}
    k_pos = {a: 0 for a in ALL_ARMS}
    k_neg = {a: 0 for a in ALL_ARMS}
    elapsed = {a: 0.0 for a in ALL_ARMS}

    def joint(p_pos, p_neg, arm, t0):
        hit, npos, nneg = _joint_hit(p_pos, p_neg, n_nodes)
        k[arm] += int(hit)
        k_pos[arm] += int(npos > 0)
        k_neg[arm] += int(nneg > 0)
        elapsed[arm] += time.time() - t0
        return hit

    for i in range(n_null):
        data_seed = seed + 7919 * i
        perm_seed = seed + 31 * i
        g1, g2, _ = generate_fc_matrices(
            n_nodes, 0.0, n_samples_group1=n_samples, n_samples_group2=n_samples, seed=data_seed,
        )
        g1z, g2z = _fisher_z(g1), _fisher_z(g2)
        for arm in PAIRWISE_ARMS:
            kwargs = dict(TFNBS_KWARGS) if arm == "tfnbs" else {}
            if arm == "nbs":
                kwargs = {"threshold": 3.0, "nbs_stat": "extent"}
            if arm == "cnbs":
                kwargs = {"net_labels": net_labels}
            t0 = time.time()
            res = compute_p_val(
                g1z, g2z, test_type="two-sample", method=arm, n_permutations=n_permutations,
                two_tailed=TWO_SIDED, use_mp=use_mp, random_state=perm_seed, **kwargs,
            )
            joint(np.asarray(res["g2>g1"]), np.asarray(res["g1>g2"]), arm, t0)
        # Multi-site GLM arm: additive site shifts, site dummies + noise covariate as confounds.
        # With corr_site_interest=0 the generator's ``interest`` is the first
        # standard-normal draw of default_rng(seed); the noise covariate therefore
        # comes from a distinct SeedSequence stream so it is not a duplicate column.
        d = generate_multisite_glm_dataset(
            n_subjects=2 * n_samples, N=n_nodes, n_sites=n_sites, effect_size=0.0,
            corr_site_interest=0.0, seed=data_seed,
        )
        sites = np.asarray(d["sites"])
        noise = np.random.default_rng([data_seed, 17]).standard_normal(len(sites))
        confounds = np.column_stack([_site_dummies(sites), noise])
        for arm in GLM_ARMS:
            t0 = time.time()
            res = compute_p_val_glm(
                d["Y"], interest=d["interest"], confounds=confounds, method="tfnbs",
                n_permutations=n_permutations, two_tailed=TWO_SIDED, use_mp=use_mp,
                random_state=perm_seed, **TFNBS_KWARGS,
            )
            joint(np.asarray(res["positive"]), np.asarray(res["negative"]), arm, t0)
        if verbose and (i + 1) % 25 == 0:
            print(f"  run {i + 1}/{n_null}: k_joint so far {k}", file=sys.stderr, flush=True)

    return [summarize_arm(a, n_null, k[a], elapsed[a], k_pos[a], k_neg[a]) for a in ALL_ARMS]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def format_table(rows: list[dict]) -> str:
    head = (f"{'arm':<11}{'n_null':>7}{'k_joint':>9}{'FWER':>8}{'p_hi':>10}{'p_holm':>10}"
            f"{'CP99 low':>10}{'CP99 high':>11}{'99% k-band':>13}{'note':>14}{'sec':>7}")
    lines = [head, "-" * len(head)]
    for r in rows:
        note = ""
        if r["fail"]:
            note = "FAIL"
        elif r["conservative"]:
            note = "conservative"
        elif not r["inside_binom99"]:
            note = "below band"
        band = f"[{r['binom99_k_low']}, {r['binom99_k_high']}]"
        lines.append(
            f"{r['arm']:<11}{r['n_null']:>7}{r['k_joint']:>9}{r['fwer_joint']:>8.3f}{r['p_hi']:>10.4f}"
            f"{r['p_holm']:>10.4f}{r['cp99_low']:>10.3f}{r['cp99_high']:>11.3f}{band:>13}{note:>14}{r['seconds']:>7.1f}"
        )
    return "\n".join(lines)


def github_steps_summary(rows: list[dict], ok: bool, cfg: dict) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"## FWER calibration check: {'PASS' if ok else 'FAIL'}\n\n")
        fh.write("Joint two-sided family, `min(p_pos, p_neg) <= alpha`; "
                 f"alpha={ALPHA}, fail if exact one-sided binomial p < {FAIL_P}.\n\n")
        fh.write("| arm | n_null | k_joint | FWER | p_hi | Holm p | CP99 | k in Binomial 99% band | verdict |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            verdict = "FAIL" if r["fail"] else ("conservative" if r["conservative"] else "ok")
            fh.write(f"| {r['arm']} | {r['n_null']} | {r['k_joint']} | {r['fwer_joint']:.3f} | {r['p_hi']:.4f} | "
                     f"{r['p_holm']:.4f} | [{r['cp99_low']:.3f}, {r['cp99_high']:.3f}] | "
                     f"{r['inside_binom99']} ([{r['binom99_k_low']}, {r['binom99_k_high']}]) | {verdict} |\n")
        fh.write(f"\nConfig: `{json.dumps(cfg)}`\n")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-null", type=int, default=200, help="null datasets per arm (default 200)")
    ap.add_argument("--n-permutations", type=int, default=199, help="permutations per test (default 199)")
    ap.add_argument("--n-nodes", type=int, default=20, help="nodes per graph (default 20)")
    ap.add_argument("--n-samples", type=int, default=20, help="subjects per group for the two-sample arms (default 20)")
    ap.add_argument("--n-sites", type=int, default=3, help="sites for the GLM arm (default 3)")
    ap.add_argument("--seed", type=int, default=2026, help="base seed (default 2026)")
    ap.add_argument("--use-mp", action="store_true", help="process pool for permutations (default off)")
    ap.add_argument("--csv", type=Path, default=None, help="write per-arm rows to this CSV")
    ap.add_argument("--json", type=Path, default=None, help="write per-arm rows + config to this JSON")
    ap.add_argument("--quiet", action="store_true", help="no progress lines")
    args = ap.parse_args(argv)
    if min(args.n_null, args.n_permutations, args.n_nodes, args.n_samples, args.n_sites) < 2:
        ap.error("--n-null/--n-permutations/--n-nodes/--n-samples/--n-sites must be >= 2")

    cfg = {"alpha": ALPHA, "two_tailed": TWO_SIDED, "n_null": args.n_null, "n_permutations": args.n_permutations,
           "n_nodes": args.n_nodes, "n_samples": args.n_samples, "n_sites": args.n_sites, "seed": args.seed,
           "use_mp": args.use_mp, "tfnbs": TFNBS_KWARGS, "nbs": {"threshold": 3.0, "nbs_stat": "extent"},
           "cnbs": {"net_labels": f"arange(N) % {N_MODULES}", "correction": "bonferroni"},
           "arms": list(ALL_ARMS), "fail_p": FAIL_P}
    print(f"FWER calibration check: {cfg['arms']} | n_null={args.n_null} n_perm={args.n_permutations} "
          f"N={args.n_nodes} n/group={args.n_samples} sites={args.n_sites} seed={args.seed} "
          f"two_tailed={TWO_SIDED} alpha={ALPHA}", flush=True)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = run_check(args.n_null, args.n_permutations, args.n_nodes, args.n_samples, args.n_sites,
                         args.seed, args.use_mp, not args.quiet)
    ok, rows = finalize(rows)
    total = time.time() - t0

    lo, hi = exact_binomial_interval(args.n_null)
    print(f"\nJoint two-sided family: any edge with min(p_pos, p_neg) <= {ALPHA}. "
          f"Exact binomial 99% count band under FWER={ALPHA}: [{lo}, {hi}] "
          f"(FWER [{lo / args.n_null:.3f}, {hi / args.n_null:.3f}]).")
    print(f"FAIL if exact one-sided binomial p(FWER > {ALPHA}) < {FAIL_P} in any arm "
          f"(Bonferroni over {len(rows)} arms: family false-alarm <= {FAIL_P * len(rows):.3f}).\n")
    print(format_table(rows))
    print(f"\nVerdict: {'PASS' if ok else 'FAIL'}  (total {total:.1f} s)")
    if os.environ.get("GITHUB_ACTIONS") == "true":
        for r in rows:
            if r["fail"]:
                print(f"::error::{r['arm']} joint two-sided FWER {r['fwer_joint']:.3f} "
                      f"({r['k_joint']}/{r['n_null']}) exceeds the 99.5% binomial bound; p={r['p_hi']:.4g}")
        print(f"::{'notice' if ok else 'error'}::FWER calibration check {'PASS' if ok else 'FAIL'} ({total:.0f} s)")
    github_steps_summary(rows, ok, cfg)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    if args.json:
        payload = {"config": cfg, "verdict": "PASS" if ok else "FAIL", "total_seconds": total,
                   "python": sys.version.split()[0], "numpy": np.__version__, "arms": rows}
        args.json.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(2)
