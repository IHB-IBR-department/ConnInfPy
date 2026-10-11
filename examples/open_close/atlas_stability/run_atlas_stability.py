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


from pathlib import Path as _Path
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
from examples.open_close.atlas_stability import adapter as A  # noqa: E402

from examples.open_close.atlas_stability.inference import tfnbs
from examples.open_close.atlas_stability.metrics import fisher_z, _digest, unit_pairs, unit_summaries, draws, agreement, pct
from examples.open_close.atlas_stability.reporting import figures, write_report
from conninfpy.interpret.annotate import aggregate_connections, summarize_annotation  # noqa: E402

OUT = HERE / "results"
# Only IHB has a stored paper map with the same participants; the stored China run-0 map used all
# 48 participants, while every China branch here uses the matched 47.
STORED = {"ihb": A.ROOT / "examples/open_close/results/ihb_paired_tfnbs.npz"}
ALPHA, SEED, MIN_SUPPORT = 0.05, 42, 1.0
PRIMARY = "lobe"



# ----------------------------------------------------------------------
# Common-unit summaries
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="Reuse only maps with matching data/order/settings/source provenance")
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--report", type=Path, help="Optional Markdown report path in the project vault")
    ap.add_argument("--data-dir", type=Path, default=A.DEFAULT_INPUTS.data_dir)
    ap.add_argument("--reference-dir", type=Path, default=A.DEFAULT_INPUTS.reference_dir)
    ap.add_argument("--schaefer-image", type=Path, default=A.SCHAEFER_IMG)
    args = ap.parse_args()
    inputs = A.StudyInputs(args.data_dir.resolve(), args.reference_dir.resolve(), args.schaefer_image.resolve())
    OUT = args.output.resolve()
    n_perm, n_boot = (500, 200) if args.quick else (5000, 1000)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    (OUT / "run_config.json").write_text(json.dumps({
        "inputs": {k: str(v) for k, v in vars(inputs).items()},
        "n_permutations": n_perm, "n_bootstraps": n_boot,
        "seed": SEED, "contrast": "EC-EO", "reuse": args.reuse,
    }, indent=2) + "\n")
    ref = A.reference(inputs)
    regions = ref.regions

    # 1. support and provenance --------------------------------------
    support, support_rec = A.common_support(regions, inputs)
    support_rec.to_csv(OUT / "common_support.csv", index=False)
    prov = pd.concat([A.provenance(e, support, inputs) for e in ("ihb", "china")], ignore_index=True)
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
            sch_rois[exp] = A.schaefer_annotation(ref, A.OpenCloseDataset.load(exp, data_dir=inputs.data_dir), inputs)
        bs = A.build_branches(cohort, support, ref, schaefer_rois=sch_rois[exp], inputs=inputs)
        for name, br in bs.items():
            eo_z, ec_z = fisher_z(br.eo), fisher_z(br.ec)
            branches[(cohort, name)] = br
            data[(cohort, name)] = {"D": ec_z - eo_z, "base": eo_z.mean(0)}
            results[(cohort, name)] = tfnbs(br, eo_z, ec_z, n_perm, args.reuse, OUT)
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

    figures(full, boot, common, OUT)
    if args.report is not None:
        write_report(args, n_perm, n_boot, support_rec, support, prov, branches, results, repro, ann, prim, diffs,
                 pd.DataFrame(xa_rows), stab, dprof, units_df, common, diag, args.report)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")



if __name__ == "__main__":
    main()
