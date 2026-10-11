"""Open-Close EC−EO contrast stability: Schaefer-200 vs native BNA-246 vs BNA-gyrus.

Implements the vault plan ``openclose_atlas_contrast_stability_plan_2026-10-10``:

1. matched branches on a fixed common cortical support, and a provenance check
   (``adapter.py``);
2. paired TFNBS per branch with the paper settings (per-tail families,
   E = 0.4, H = 3, n = 10, 5,000 permutations, seed 42); the Schaefer full set
   must reproduce the stored paper maps;
3. BNA annotation of the Schaefer result: invariance of every result array,
   coverage, purity and attribution levels of the detected edges;
4. participant-level signed delta-z summaries at fixed common units (BNA lobe x
   hemisphere [primary], gyrus x hemisphere, Yeo-7 reference, own labels), a
   baseline split and between-edge heterogeneity;
5. a participant bootstrap (1,000 draws per cohort, seed 42, the same draws for
   every atlas branch): primary endpoint = cross-cohort agreement of unit
   effects per atlas and paired differences between atlases; secondary =
   within-atlas edge stability (with parcel volume), cross-atlas agreement,
   detection density, China EC0 sensitivity.

Descriptive comparisons; no new hypothesis test. Usage::

    python examples/open_close/atlas_stability/run_atlas_stability.py [--quick] [--reuse]

``--quick``: 500 permutations, 200 bootstrap draws. ``--reuse``: load saved
TFNBS maps instead of recomputing them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapter as A  # noqa: E402

from conninfpy import analyze  # noqa: E402
from conninfpy.annotate import aggregate_connections, summarize_annotation  # noqa: E402
from conninfpy.results import InferenceResult  # noqa: E402

OUT = HERE / "results"
TFNBS = dict(method="tfnbs", e=0.4, h=3.0, n=10, acceleration=None, rng=42, use_mp=True)
# Only IHB has a stored paper map with the same participants; the stored China run-0 map used all
# 48 participants, while every China branch here uses the matched 47.
STORED = {"ihb": A.ROOT / "examples/open_close/results/ihb_paired_tfnbs.npz"}
ALPHA, SEED, MIN_SUPPORT = 0.05, 42, 1.0
PRIMARY = "lobe"


# ----------------------------------------------------------------------
# Edge-level quantities
# ----------------------------------------------------------------------
def fisher_z(ts: np.ndarray) -> np.ndarray:
    out = np.empty((ts.shape[0], ts.shape[2], ts.shape[2]))
    for s, x in enumerate(ts):
        c = np.clip(np.corrcoef(x.T), -0.9999, 0.9999)
        np.fill_diagonal(c, 0.0)
        out[s] = np.arctanh(c)
    return out


def tfnbs(branch, eo_z, ec_z, n_perm, reuse):
    path = OUT / "tfnbs" / f"{branch.cohort}__{branch.name}.npz"
    if reuse and path.exists():
        d = np.load(path)
        return InferenceResult(d["positive"], d["negative"], method="tfnbs",
                               stat_positive=d["stat_positive"], stat_negative=d["stat_negative"])
    inf = analyze(group1=eo_z, group2=ec_z, test_type="paired", fisher_z=False, n_permutations=n_perm, **TFNBS).inference
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, positive=inf.positive, negative=inf.negative,
                        stat_positive=inf.stat_positive, stat_negative=inf.stat_negative)
    return inf


def _digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        if a is not None:
            h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------
# Common-unit summaries
# ----------------------------------------------------------------------
def unit_pairs(units):
    U = len(units)
    a, b = np.triu_indices(U)
    return a, b, [f"{units[i]} | {units[j]}" for i, j in zip(a, b)]


def unit_summaries(D: np.ndarray, M: np.ndarray, base: np.ndarray):
    """Participant-level weighted mean delta-z per unit pair, plus baseline split and heterogeneity.

    D: (S, N, N) EC−EO delta-z (zero diagonal); M: (N, U) memberships; base:
    (N, N) full-sample EO mean z. An edge (i, j) has weight m_i[a] m_j[b] + m_i[b] m_j[a]
    for a != b and m_i[a] m_j[a] for a = b.
    """
    S, N, _ = D.shape
    U = M.shape[1]
    off = 1.0 - np.eye(N)
    W = M.T @ off @ M                                       # summed weights (both orientations)
    num = np.matmul(M.T[None], np.matmul(D, M))              # (S, U, U)
    a, b, _ = unit_pairs([str(k) for k in range(U)])
    support = W[a, b] / np.where(a == b, 2.0, 1.0)          # edge-equivalents
    with np.errstate(invalid="ignore", divide="ignore"):
        E = num[:, a, b] / W[a, b]
    iu = np.triu_indices(N, 1)
    De = D[:, iu[0], iu[1]]                                 # (S, E)
    be = base[iu]
    dbar, se2 = De.mean(0), De.var(0, ddof=1) / S
    strong, weak, tau = np.full((S, len(a)), np.nan), np.full((S, len(a)), np.nan), np.full(len(a), np.nan)
    for k, (x, y) in enumerate(zip(a, b)):
        w = M[iu[0], x] * M[iu[1], y] + (M[iu[0], y] * M[iu[1], x] if x != y else 0.0)
        on = w > 0
        if w[on].sum() < MIN_SUPPORT:
            continue
        wk, bk = w[on], be[on]
        order = np.argsort(bk)
        cw = np.cumsum(wk[order]) / wk.sum()
        med = bk[order][np.searchsorted(cw, 0.5)]
        hi = bk >= med
        for out, sel in ((strong, hi), (weak, ~hi)):
            if wk[sel].sum() > 0:
                out[:, k] = De[:, on][:, sel] @ wk[sel] / wk[sel].sum()
        m = np.average(dbar[on], weights=wk)
        tau[k] = np.sqrt(max(0.0, np.average((dbar[on] - m) ** 2, weights=wk) - np.average(se2[on], weights=wk)))
    return {"E": E, "support": support, "strong": strong, "weak": weak, "tau": tau}


def draws(n_subjects: int, n_boot: int) -> np.ndarray:
    """(B, S) bootstrap count matrix (seed 42); identical for every branch of a cohort."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_subjects, size=(n_boot, n_subjects))
    C = np.zeros((n_boot, n_subjects))
    np.add.at(C, (np.repeat(np.arange(n_boot), n_subjects), idx.ravel()), 1.0)
    return C / n_subjects


def agreement(x: np.ndarray, y: np.ndarray) -> dict:
    """Agreement of two unit-effect vectors (rows = draws allowed)."""
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    mx, my = x.mean(1, keepdims=True), y.mean(1, keepdims=True)
    vx, vy = x.var(1), y.var(1)
    cov = ((x - mx) * (y - my)).mean(1)
    rx, ry = stats.rankdata(x, axis=1), stats.rankdata(y, axis=1)
    return {"ccc": 2 * cov / (vx + vy + (mx - my)[:, 0] ** 2), "pearson": cov / np.sqrt(vx * vy),
            "spearman": np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(rx, ry)]),
            "sign_agreement": (np.sign(x) == np.sign(y)).mean(1)}


def pct(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v.size else (np.nan, np.nan)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    n_perm, n_boot = (500, 200) if args.quick else (5000, 1000)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ref = A.reference()
    regions = ref.regions

    # 1. support and provenance --------------------------------------
    support, support_rec = A.common_support(regions)
    support_rec.to_csv(OUT / "common_support.csv", index=False)
    prov = pd.concat([A.provenance(e, support) for e in ("ihb", "china")], ignore_index=True)
    prov.to_csv(OUT / "provenance.csv", index=False)
    if (prov["median_r"] < 0.5).any():
        bad = prov[prov["median_r"] < 0.5]
        raise SystemExit(f"Provenance check failed for {len(bad)} scans; see provenance.csv")

    # 2. branches, contrasts, TFNBS ----------------------------------
    sch_rois = {}
    branches, data, results = {}, {}, {}
    for cohort in A.COHORTS:
        exp = A.COHORTS[cohort][0]
        if exp not in sch_rois:
            sch_rois[exp] = A.schaefer_annotation(ref, A.OpenCloseDataset.load(exp))
        bs = A.build_branches(cohort, support, ref, schaefer_rois=sch_rois[exp])
        for name, br in bs.items():
            eo_z, ec_z = fisher_z(br.eo), fisher_z(br.ec)
            branches[(cohort, name)] = br
            data[(cohort, name)] = {"D": ec_z - eo_z, "base": eo_z.mean(0)}
            results[(cohort, name)] = tfnbs(br, eo_z, ec_z, n_perm, args.reuse)
            print(f"{cohort:9s} {name:13s} N={br.eo.shape[2]:3d} S={br.eo.shape[0]:2d} "
                  f"detections +{int((results[(cohort, name)].positive < ALPHA).sum() // 2)} "
                  f"-{int((results[(cohort, name)].negative < ALPHA).sum() // 2)}  ({time.time() - t0:.0f}s)")

    # reproduction of the paper's Schaefer maps
    repro = {}
    for cohort, path in STORED.items():
        if path.exists():
            s, r = np.load(path), results[(cohort, "schaefer_full")]
            repro[cohort] = float(max(np.nanmax(np.abs(r.positive - s["positive"])), np.nanmax(np.abs(r.negative - s["negative"]))))

    # 3. annotation-only evaluation (Schaefer full set) --------------
    ann_rows = []
    for cohort in A.COHORTS:
        exp = A.COHORTS[cohort][0]
        inf, rois = results[(cohort, "schaefer_full")], sch_rois[exp]
        before = _digest(inf.positive, inf.negative, inf.stat_positive, inf.stat_negative)
        edges = inf.annotated_edges(rois, alpha=ALPHA)
        summarize_annotation(inf, rois, alpha=ALPHA)
        aggregate_connections(inf, rois, level="gyrus", tail="positive")
        after = _digest(inf.positive, inf.negative, inf.stat_positive, inf.stat_negative)
        level = rois.table["level"].to_numpy()
        rank = {"subregion": 0, "gyrus": 1, "lobe": 2}
        lv = np.array([rank.get(x, 3) for x in level])
        worst = np.maximum(lv[edges["roi_i"].to_numpy(int)], lv[edges["roi_j"].to_numpy(int)])
        ann_rows.append({"cohort": cohort, "arrays_unchanged": before == after, "n_detected": len(edges),
                         "both_endpoints_subregion": float((worst == 0).mean()), "at_gyrus_or_finer": float((worst <= 1).mean()),
                         "at_lobe_or_finer": float((worst <= 2).mean()),
                         "parcels_by_level": json.dumps(rois.table["level"].value_counts().to_dict()),
                         "median_subregion_purity": float(rois.table["region_purity"].median())})
    ann = pd.DataFrame(ann_rows)
    ann.to_csv(OUT / "annotation_evaluation.csv", index=False)

    # 4. unit summaries -----------------------------------------------
    summ = {}
    for (cohort, name), br in branches.items():
        if name not in A.MATCHED:
            continue
        for level, (M, units) in br.members.items():
            summ[(cohort, name, level)] = (unit_summaries(data[(cohort, name)]["D"], M, data[(cohort, name)]["base"]), units)

    # fixed common unit pairs per level: supported (>= 1 edge-equivalent) in every matched branch and cohort
    common = {}
    for level in A.LEVELS:
        units = summ[("ihb", "schaefer", level)][1]
        ok = None
        for cohort in A.COHORTS:
            for name in A.MATCHED:
                s, u = summ[(cohort, name, level)]
                if u != units:
                    raise RuntimeError(f"unit lists differ for {level}: {name}")
                m = s["support"] >= MIN_SUPPORT
                ok = m if ok is None else ok & m
        common[level] = (np.flatnonzero(ok), unit_pairs(units)[2])

    # 5. bootstrap ------------------------------------------------------
    C = {c: draws(len(branches[(c, "schaefer")].subjects), n_boot) for c in A.COHORTS}
    boot = {}                                                       # (cohort, name, level) -> (B, P_common)
    full = {}
    unit_rows = []
    for (cohort, name, level), (s, units) in summ.items():
        idx = common[level][0]
        E = s["E"][:, idx]
        full[(cohort, name, level)] = E.mean(0)
        boot[(cohort, name, level)] = C[cohort] @ E
        lo, hi = np.percentile(boot[(cohort, name, level)], [2.5, 97.5], axis=0)
        for k, p in enumerate(idx):
            unit_rows.append({"cohort": cohort, "atlas": name, "level": level, "unit_pair": common[level][1][p],
                              "support_edges": s["support"][p], "mean_dz": E[:, k].mean(), "ci_lo": lo[k], "ci_hi": hi[k],
                              "sign_freq": float((np.sign(boot[(cohort, name, level)][:, k]) == np.sign(E[:, k].mean())).mean()),
                              "mean_dz_strong_baseline": np.nanmean(s["strong"][:, p]), "mean_dz_weak_baseline": np.nanmean(s["weak"][:, p]),
                              "tau": s["tau"][p]})
    units_df = pd.DataFrame(unit_rows)
    units_df.to_csv(OUT / "unit_effects.csv", index=False)

    # primary endpoint: cross-cohort agreement per atlas, paired atlas differences
    prim_rows, diff_rows = [], []
    for china in ("china_ec1", "china_ec0"):
        for level in A.LEVELS:
            ag = {}
            for name in A.MATCHED:
                x, y = boot[("ihb", name, level)], boot[(china, name, level)]
                ag[name] = agreement(x, y)
                f = agreement(full[("ihb", name, level)], full[(china, name, level)])
                for metric in ("ccc", "pearson", "spearman", "sign_agreement"):
                    lo, hi = pct(ag[name][metric])
                    prim_rows.append({"china": china, "level": level, "atlas": name, "metric": metric,
                                      "full_sample": float(f[metric][0]), "ci_lo": lo, "ci_hi": hi,
                                      "n_unit_pairs": len(common[level][0])})
            for a_, b_ in (("bna", "schaefer"), ("bna_gyrus", "schaefer"), ("bna", "bna_gyrus")):
                for metric in ("ccc", "pearson", "spearman", "sign_agreement"):
                    d = ag[a_][metric] - ag[b_][metric]
                    lo, hi = pct(d)
                    diff_rows.append({"china": china, "level": level, "contrast": f"{a_} - {b_}", "metric": metric,
                                      "median_diff": float(np.nanmedian(d)), "ci_lo": lo, "ci_hi": hi,
                                      "share_draws_positive": float(np.nanmean(d > 0))})
    prim, diffs = pd.DataFrame(prim_rows), pd.DataFrame(diff_rows)

    # post hoc diagnostic (not prespecified): agreement without the within-occipital pairs,
    # whose effects have opposite signs in IHB and China for every atlas
    diag_rows = []
    for level in ("lobe", "gyrus"):
        labels = [common[level][1][p] for p in common[level][0]]
        keep = np.array([not (lab.count("Occipital") == 2 or lab.count("MVOcC") + lab.count("LOcC") == 2) for lab in labels])
        for name in A.MATCHED:
            f = agreement(full[("ihb", name, level)][keep], full[("china_ec1", name, level)][keep])["ccc"][0]
            lo, hi = pct(agreement(boot[("ihb", name, level)][:, keep], boot[("china_ec1", name, level)][:, keep])["ccc"])
            diag_rows.append({"level": level, "atlas": name, "ccc_without_within_occipital": float(f),
                              "ci_lo": lo, "ci_hi": hi, "n_unit_pairs": int(keep.sum())})
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(OUT / "posthoc_without_within_occipital.csv", index=False)
    prim.to_csv(OUT / "primary_cross_cohort_agreement.csv", index=False)
    diffs.to_csv(OUT / "primary_paired_atlas_differences.csv", index=False)

    # cross-atlas agreement within cohort
    xa_rows = []
    for cohort in A.COHORTS:
        for level in A.LEVELS:
            for a_, b_ in (("bna", "schaefer"), ("bna_gyrus", "schaefer"), ("bna", "bna_gyrus")):
                g = agreement(boot[(cohort, a_, level)], boot[(cohort, b_, level)])
                f = agreement(full[(cohort, a_, level)], full[(cohort, b_, level)])
                for metric in ("ccc", "pearson", "sign_agreement"):
                    lo, hi = pct(g[metric])
                    xa_rows.append({"cohort": cohort, "level": level, "pair": f"{a_} vs {b_}", "metric": metric,
                                    "full_sample": float(f[metric][0]), "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(xa_rows).to_csv(OUT / "cross_atlas_agreement.csv", index=False)

    # within-atlas edge stability with parcel volume
    st_rows = []
    for (cohort, name), br in branches.items():
        if name not in A.MATCHED:
            continue
        D = data[(cohort, name)]["D"]
        iu = np.triu_indices(D.shape[1], 1)
        De = D[:, iu[0], iu[1]]
        m = De.mean(0)
        t = m / (De.std(0, ddof=1) / np.sqrt(De.shape[0]))
        B = C[cohort] @ De
        sign_freq = (np.sign(B) == np.sign(m)).mean(0)
        r_map = np.array([np.corrcoef(b, m)[0, 1] for b in B])
        vol = br.parcels["volume_mm3"].to_numpy(float)
        vmin = np.minimum(vol[iu[0]], vol[iu[1]])
        st_rows.append({"cohort": cohort, "atlas": name, "n_parcels": D.shape[1], "n_edges": De.shape[1],
                        "median_parcel_volume_mm3": float(np.median(vol)),
                        "median_abs_t": float(np.median(np.abs(t))), "share_sign_freq_ge_0.95": float((sign_freq >= 0.95).mean()),
                        "median_draw_map_r": float(np.median(r_map)), "draw_map_r_lo": pct(r_map)[0],
                        "spearman_abs_t_vs_min_volume": float(stats.spearmanr(np.abs(t), vmin).statistic)})
    stab = pd.DataFrame(st_rows)
    stab.to_csv(OUT / "edge_stability.csv", index=False)

    # detection density at common units (descriptive)
    dd_rows, prof = [], {}
    for (cohort, name), br in branches.items():
        if name not in A.MATCHED:
            continue
        inf = results[(cohort, name)]
        for level in ("lobe", "gyrus"):
            M, units = br.members[level]
            N = M.shape[0]
            off = 1.0 - np.eye(N)
            W = M.T @ off @ M
            a, b, labels = unit_pairs(units)
            idx = common[level][0]
            for tail in ("positive", "negative"):
                S = (np.asarray(inf.positive if tail == "positive" else inf.negative) < ALPHA).astype(float) * off
                rate = (M.T @ S @ M)[a, b][idx] / W[a, b][idx]
                prof[(cohort, name, level, tail)] = rate
                for p, r in zip(idx, rate):
                    dd_rows.append({"cohort": cohort, "atlas": name, "level": level, "tail": tail,
                                    "unit_pair": labels[p], "detection_rate": r})
    pd.DataFrame(dd_rows).to_csv(OUT / "detection_density.csv", index=False)
    dprof_rows = []
    for level in ("lobe", "gyrus"):
        for tail in ("positive", "negative"):
            for cohort in A.COHORTS:
                for a_, b_ in (("bna", "schaefer"), ("bna_gyrus", "schaefer")):
                    dprof_rows.append({"comparison": f"{cohort}: {a_} vs {b_}", "level": level, "tail": tail,
                                       "spearman": float(stats.spearmanr(prof[(cohort, a_, level, tail)], prof[(cohort, b_, level, tail)]).statistic)})
            for name in A.MATCHED:
                dprof_rows.append({"comparison": f"ihb vs china_ec1: {name}", "level": level, "tail": tail,
                                   "spearman": float(stats.spearmanr(prof[("ihb", name, level, tail)], prof[("china_ec1", name, level, tail)]).statistic)})
    dprof = pd.DataFrame(dprof_rows)
    dprof.to_csv(OUT / "detection_density_agreement.csv", index=False)

    figures(full, boot, common)
    write_report(args, n_perm, n_boot, support_rec, support, prov, branches, results, repro, ann, prim, diffs,
                 pd.DataFrame(xa_rows), stab, dprof, units_df, common, diag)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


# ----------------------------------------------------------------------
# Figures and report
# ----------------------------------------------------------------------
def figures(full, boot, common):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {"schaefer": "Schaefer-200", "bna": "BNA-246 (native)", "bna_gyrus": "BNA-gyrus (resolution control)"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
    for row, level in enumerate(("lobe", "gyrus")):
        for ax, name in zip(axes[row], A.MATCHED):
            x, y = full[("ihb", name, level)], full[("china_ec1", name, level)]
            lim = np.nanmax(np.abs(np.r_[x, y])) * 1.08
            ax.axhline(0, color="#bbb", lw=0.6)
            ax.axvline(0, color="#bbb", lw=0.6)
            ax.plot([-lim, lim], [-lim, lim], color="#999", lw=0.6, ls="--")
            ax.scatter(x, y, s=12 if level == "lobe" else 5, alpha=0.7, color="#2b6cb0")
            ccc = agreement(x, y)["ccc"][0]
            ax.set_title(f"{titles[name]}\n{level} × hemisphere pairs: CCC {ccc:.2f}", fontsize=9)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel("IHB mean Δz (EC − EO)", fontsize=8)
            ax.set_ylabel("China (EC1) mean Δz", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("Cross-cohort agreement of common-unit EC−EO effects, per atlas", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cross_cohort_units.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, level in zip(axes, ("lobe", "gyrus")):
        vals = [agreement(boot[("ihb", n, level)], boot[("china_ec1", n, level)])["ccc"] for n in A.MATCHED]
        ax.violinplot(vals, showmedians=True)
        ax.set_xticks([1, 2, 3], [titles[n].split(" (")[0] for n in A.MATCHED], fontsize=8)
        ax.set_ylabel("IHB–China concordance (CCC)", fontsize=8)
        ax.set_title(f"{level} × hemisphere: bootstrap distribution (n={len(common[level][0])} unit pairs)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_primary_bootstrap.png", dpi=150)
    plt.close(fig)


def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_report(args, n_perm, n_boot, support_rec, support, prov, branches, results, repro, ann, prim, diffs,
                 xa, stab, dprof, units_df, common, diag):
    L = ["# Open-Close EC−EO contrast stability: Schaefer-200 vs native BNA-246 vs BNA-gyrus\n",
         f"Plan: `openclose_atlas_contrast_stability_plan_2026-10-10` (vault). {n_perm} permutations, "
         f"{n_boot} bootstrap draws, seed {SEED}; per-tail TFNBS (E = 0.4, H = 3, n = 10). Descriptive comparisons.\n"]

    excl = support_rec[support_rec.excluded != ""].groupby(["atlas", "excluded"]).size().reset_index(name="n")
    L.append(f"\n## 1. Common cortical support and provenance\n\nKept: {len(support['schaefer'])} Schaefer parcels, "
             f"{len(support['bna'])} BNA cortical regions; BNA-gyrus = "
             f"{len(branches[('ihb', 'bna_gyrus')].parcels)} gyrus × hemisphere signals.\n\n" + _md(excl) + "\n")
    q = prov.groupby(["experiment", "scan"])["median_r"].agg(["min", "median"]).reset_index()
    L.append("\nProvenance (per scan, median correlation of each BNA region with its most-overlapping "
             "Schaefer parcel; blocking threshold 0.5):\n\n" + _md(q) + "\n")

    rows = []
    for (cohort, name), br in branches.items():
        r = results[(cohort, name)]
        rows.append({"cohort": cohort, "branch": name, "parcels": br.eo.shape[2], "subjects": br.eo.shape[0],
                     "edges": br.eo.shape[2] * (br.eo.shape[2] - 1) // 2,
                     "EC>EO": int((r.positive < ALPHA).sum() // 2), "EO>EC": int((r.negative < ALPHA).sum() // 2)})
    L.append("\n## 2. Paired TFNBS per branch (α = 0.05 per tail)\n\nDetection counts are not comparable across "
             "branches (different edge families); see §6 for common-unit densities. Reproduction of the paper's "
             f"Schaefer map, IHB (max |Δp|): {json.dumps(repro)}. China branches use the matched 47 participants; "
             "their Schaefer counts equal the 47-participant runs of the Brainnetome interpretation example "
             "(EC1 +1795/−1247, EC0 +2216/−857), not the paper's 48-participant run-0 map.\n\n" + _md(pd.DataFrame(rows)) + "\n")

    L.append("\n## 3. Annotation only (Schaefer full set)\n\nArrays must be unchanged by annotation; attribution "
             "levels of the detected edges (worst endpoint):\n\n" + _md(ann) + "\n")

    p = prim[(prim.china == "china_ec1") & (prim.metric.isin(["ccc", "pearson", "sign_agreement"]))]
    L.append("\n## 4. Primary endpoint: IHB vs China (EC1) agreement of common-unit effects\n\n"
             "Full-sample value and 95 % bootstrap percentile interval (independent IHB and China draws).\n\n"
             + _md(p.drop(columns="china")) + "\n")
    d = diffs[(diffs.china == "china_ec1") & (diffs.metric == "ccc")]
    L.append("\n**Paired atlas differences in CCC** (same draws; an interval excluding 0 is a clear difference). "
             "The Yeo reference is the Brainnetome subregion crosswalk: native BNA matches it exactly while Schaefer "
             "parcels are mapped through it, so `yeo_ref` differences favour BNA by construction; `yeo_own` (each "
             "atlas's own labels) is the fair functional comparison.\n\n"
             + _md(d.drop(columns=["china", "metric"])) + "\n")
    L.append("\n**Post hoc diagnostic (not prespecified):** the within-occipital pairs carry the largest effects "
             "and have opposite signs in IHB and China for every atlas (the known within-visual disagreement). "
             "Agreement without them:\n\n" + _md(diag) + "\n")
    d0 = diffs[(diffs.china == "china_ec0") & (diffs.metric == "ccc") & (diffs.level == PRIMARY)]
    p0 = prim[(prim.china == "china_ec0") & (prim.metric == "ccc")]
    L.append("\n**China EC0 sensitivity** (run choice):\n\n" + _md(p0.drop(columns=["china", "metric"])) + "\n\n"
             + _md(d0.drop(columns=["china", "metric"])) + "\n")

    L.append("\n## 5. Cross-atlas agreement within cohort (same draws)\n\n"
             + _md(xa[xa.metric == "ccc"].drop(columns="metric")) + "\n")
    L.append("\n## 6. Secondary: edge stability, parcel volume, detection density\n\n"
             "Edge sign frequency is roughly Φ(t); it reflects per-edge SNR (parcel volume), not anatomical "
             "correctness.\n\n" + _md(stab) + "\n\nDetection-density profile agreement (Spearman over common "
             "unit pairs):\n\n" + _md(dprof) + "\n")

    key = units_df[(units_df.cohort.isin(["ihb", "china_ec1"])) & (units_df.level == "lobe")
                   & (units_df.unit_pair.str.count("Occipital") == 2)]
    key = key[["cohort", "atlas", "unit_pair", "mean_dz", "ci_lo", "ci_hi", "mean_dz_strong_baseline",
               "mean_dz_weak_baseline", "tau"]].sort_values(["unit_pair", "cohort", "atlas"])
    L.append("\n## 7. Within-occipital lobe pairs: signed effect, baseline split and heterogeneity\n\n"
             "Strong/weak = edges above/below the unit pair's weighted median EO coupling; τ = between-edge SD "
             "of true effects (debiased).\n\n" + _md(key) + "\n")
    L.append("\nFull tables: `unit_effects.csv`, `primary_*.csv`, `cross_atlas_agreement.csv`, `edge_stability.csv`, "
             "`detection_density*.csv`, `common_support.csv`, `provenance.csv`, `annotation_evaluation.csv`; "
             "figures `fig_cross_cohort_units.png`, `fig_primary_bootstrap.png`.\n")
    (OUT / "report.md").write_text("".join(L))


if __name__ == "__main__":
    main()
