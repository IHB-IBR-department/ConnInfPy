"""Open-Close inference restricted to structurally supported edges (binarised Brainnetome SC).

Implements the structural part of the vault plan
``openclose_gsr_sc_followup_plan_2026-10-10``. The idea: binarise the Brainnetome
structural connectivity (SC) and compute TFNBS cluster support and NI-TFNBS block
density only over structurally supported edges (``compute_p_val(edge_mask=...)``),
so FWER is controlled over that family.

Native BNA-246 series (242 regions, GSR), IHB (84) and China EO vs counterbalanced
EC1 (47); paired tests, positive = EC > EO; paper settings (per-tail, E = 0.4,
H = 3, n = 10, seed 42). NI-TFNBS blocks: Brainnetome Yeo-7 crosswalk + subcortex.

E0  go/no-go: are |t| larger on SC-supported than unsupported pairs at matched
    distance (10 mm bins x within/between hemisphere x cortical/subcortical)?
    Participant bootstrap; overall and by system (sensory vs association).
E2  SC-restricted family (SC >= 0.05 primary; 0.01, 0.1, 0.2) vs the full-family
    result restricted afterwards to the same edges, and what the mask loses.
E3  controls: random masks matched to the SC mask stratum by stratum; the SC mask
    must beat them on IHB-China replication (Dice of detected edges), not only on
    detection count.
E4  validity: null datasets from random per-participant EO/EC swaps; FWER per tail
    within the SC family.

Usage::

    python examples/open_close/structural_prior/run_sc_inference.py [--quick] [--stage e0,e2,e3,e4] [--masks 40] [--nulls 200]
"""
from __future__ import annotations


from pathlib import Path as _Path
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


OUT = HERE / "results"
SEED, ALPHA = 42, 0.05
THRESHOLDS = (0.05, 0.01, 0.1, 0.2)                # first = primary
SETTINGS = dict(test_type="paired", e=0.4, h=3.0, n=10, acceleration=None, rng=SEED, use_mp=True)
METHODS = ("tfnbs", "ni_tfnbs")
SENSORY = {"Visual", "Somatomotor"}
COHORTS = {"ihb": ("ihb", 0, False), "china_ec1": ("china", 1, True)}


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
from examples.open_close.structural_prior.analysis import (
    load, edge_table, to_mask, dice, detected, e0, run, matched_random_masks,
)

def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="500 permutations, 10 masks, 20 nulls, 200 draws")
    ap.add_argument("--stage", default="e0,e2,e3,e4")
    ap.add_argument("--masks", type=int, default=None, help="random masks for E3 (default 40; quick 10)")
    ap.add_argument("--nulls", type=int, default=None, help="null datasets for E4 (default 200; quick 20)")
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--report", type=Path, help="Optional Markdown report path in the project vault")
    args = ap.parse_args()
    OUT = args.output.resolve()
    stages = set(args.stage.split(","))
    n_perm, n_masks, n_null, n_boot = (500, 10, 20, 200) if args.quick else (5000, 40, 200, 1000)
    n_masks = args.masks if args.masks is not None else n_masks
    n_null = args.nulls if args.nulls is not None else n_null
    null_perm = 500 if args.quick else 1000
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data, sc, dist, hemi, cortex, nets, labels, ids = load()
    N = len(ids)
    edges = edge_table(sc, dist, hemi, cortex, nets)
    edges.to_csv(OUT / "edges.csv", index=False)
    log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)   # noqa: E731

    # description of the masks
    rows = []
    for tau in THRESHOLDS:
        sup = edges["sc"] >= tau
        rows.append({"sc_threshold": tau, "density": float(sup.mean()), "n_edges": int(sup.sum()),
                     "median_dist_kept": float(edges.dist[sup].median()), "median_dist_all": float(edges.dist.median()),
                     "share_inter_kept": float(sup[edges.inter].mean()), "share_intra_kept": float(sup[~edges.inter].mean())})
    pd.DataFrame(rows).to_csv(OUT / "masks.csv", index=False)

    if "e0" in stages:
        e0(data, edges, n_boot).to_csv(OUT / "e0_enrichment.csv", index=False)
        log("E0 done")

    if "e2" in stages:
        rows, lost = [], []
        full = {(c, m): run(*data[c], m, labels, None, n_perm) for c in COHORTS for m in METHODS}
        for (c, m), r in full.items():
            np.savez_compressed(OUT / f"p_{c}_{m}_full.npz", positive=r["positive"], negative=r["negative"])
        log("E2 full-family runs done")
        for tau in THRESHOLDS:
            sup = (edges["sc"] >= tau).to_numpy()
            fam = set(np.flatnonzero(sup))
            mask = to_mask(sup, N, edges)
            for m in METHODS:
                res = {c: run(*data[c], m, labels, mask, n_perm) for c in COHORTS}
                if tau == THRESHOLDS[0]:
                    for c, r in res.items():
                        np.savez_compressed(OUT / f"p_{c}_{m}_sc{tau}.npz", positive=r["positive"], negative=r["negative"])
                for tail in ("positive", "negative"):
                    det = {c: detected(res[c], tail, edges) for c in COHORTS}
                    post = {c: detected(full[(c, m)], tail, edges) & fam for c in COHORTS}
                    allf = {c: detected(full[(c, m)], tail, edges) for c in COHORTS}
                    for c in COHORTS:
                        rows.append({"sc_threshold": tau, "method": m, "tail": tail, "cohort": c,
                                     "family_size": len(fam), "detected_sc_family": len(det[c]),
                                     "detected_full_restricted": len(post[c]), "detected_full_all": len(allf[c]),
                                     "share_full_outside_mask": 1 - len(allf[c] & fam) / max(len(allf[c]), 1)})
                    rows.append({"sc_threshold": tau, "method": m, "tail": tail, "cohort": "dice_ihb_china",
                                 "family_size": len(fam), "detected_sc_family": dice(det["ihb"], det["china_ec1"]),
                                 "detected_full_restricted": dice(post["ihb"], post["china_ec1"]),
                                 "detected_full_all": dice(allf["ihb"], allf["china_ec1"]), "share_full_outside_mask": np.nan})
                    if tau == THRESHOLDS[0]:
                        for c in COHORTS:
                            for e in allf[c]:
                                lost.append({"method": m, "tail": tail, "cohort": c, "in_sc_family": e in fam,
                                             "inter": bool(edges.inter[e]), "system": edges.system[e], "dist": edges.dist[e]})
                log(f"E2 SC >= {tau} {m} done")
        pd.DataFrame(rows).to_csv(OUT / "e2_inference.csv", index=False)
        pd.DataFrame(lost).to_csv(OUT / "e2_full_detections_by_mask.csv", index=False)

    if "e3" in stages:
        sup = (edges["sc"] >= THRESHOLDS[0]).to_numpy()
        rows = []
        for r, sel in enumerate(matched_random_masks(edges, sup, n_masks)):
            mask = to_mask(sel, N, edges)
            for m in METHODS:
                res = {c: run(*data[c], m, labels, mask, n_perm) for c in COHORTS}
                for tail in ("positive", "negative"):
                    det = {c: detected(res[c], tail, edges) for c in COHORTS}
                    rows.append({"mask": r, "method": m, "tail": tail, "detected_ihb": len(det["ihb"]),
                                 "detected_china": len(det["china_ec1"]), "dice_ihb_china": dice(det["ihb"], det["china_ec1"]),
                                 "overlap_with_sc_mask": float((sel & sup).sum() / sup.sum())})
            if (r + 1) % 10 == 0:
                log(f"E3 {r + 1}/{n_masks} masks")
                pd.DataFrame(rows).to_csv(OUT / "e3_random_masks.csv", index=False)
        pd.DataFrame(rows).to_csv(OUT / "e3_random_masks.csv", index=False)

    if "e4" in stages:
        mask = to_mask((edges["sc"] >= THRESHOLDS[0]).to_numpy(), N, edges)
        rows = []
        for c, (eo, ec) in data.items():
            for k in range(n_null):
                flip = np.random.default_rng(10_000 + k).random(eo.shape[0]) < 0.5
                a, b = np.where(flip[:, None, None], ec, eo), np.where(flip[:, None, None], eo, ec)
                for m in METHODS:
                    r = run(a, b, m, labels, mask, null_perm)
                    rows.append({"cohort": c, "null": k, "method": m,
                                 "any_positive": bool(np.nanmin(r["positive"]) < ALPHA),
                                 "any_negative": bool(np.nanmin(r["negative"]) < ALPHA)})
            log(f"E4 {c} done")
        pd.DataFrame(rows).to_csv(OUT / "e4_null_fwer.csv", index=False)

    if args.report is not None:
        report(n_perm, n_masks, n_null, null_perm, n_boot, args.report)
    log(f"done -> {OUT}")

def _md(df, fmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)

def _binom_ci(k, n):
    from scipy.stats import beta
    return (beta.ppf(0.025, k, n - k + 1) if k else 0.0, beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)

def report(n_perm, n_masks, n_null, null_perm, n_boot, report_path: Path):
    L = ["# Open-Close inference restricted to structurally supported edges\n",
         f"Native BNA-246 (242 regions, GSR); IHB 84, China EC1 47; paired, positive = EC > EO; "
         f"{n_perm} permutations; seed {SEED}. Plan: `openclose_gsr_sc_followup_plan_2026-10-10` (vault).\n"]
    if (OUT / "masks.csv").exists():
        L.append("\n## Masks\n\n" + _md(pd.read_csv(OUT / "masks.csv")) + "\n")
    if (OUT / "e0_enrichment.csv").exists():
        e = pd.read_csv(OUT / "e0_enrichment.csv")
        L.append(f"\n## E0: |t| on SC-supported minus unsupported pairs at matched distance ({n_boot} draws)\n\n"
                 "Stratified by 10 mm distance bin x within/between hemisphere x cortical/subcortical; "
                 "go if the primary threshold's interval excludes 0 in both cohorts.\n\n" + _md(e) + "\n")
    if (OUT / "e2_inference.csv").exists():
        e = pd.read_csv(OUT / "e2_inference.csv")
        L.append("\n## E2: SC-restricted family vs full family\n\n`detected_sc_family`: detections when the "
                 "family is the SC mask; `detected_full_restricted`: full-family detections that fall in the "
                 "mask; rows `dice_ihb_china` give IHB-China Dice of those sets.\n\n" + _md(e) + "\n")
    if (OUT / "e2_full_detections_by_mask.csv").exists():
        f = pd.read_csv(OUT / "e2_full_detections_by_mask.csv")
        g = f.groupby(["method", "cohort", "inter"])["in_sc_family"].agg(["size", "mean"]).reset_index()
        g.columns = ["method", "cohort", "interhemispheric", "full_detections", "share_in_sc_mask"]
        L.append("\n**What the primary mask removes** (full-family detections, both tails):\n\n" + _md(g) + "\n")
    if (OUT / "e3_random_masks.csv").exists() and (OUT / "e2_inference.csv").exists():
        r3 = pd.read_csv(OUT / "e3_random_masks.csv")
        e = pd.read_csv(OUT / "e2_inference.csv")
        prim = e[(e.sc_threshold == THRESHOLDS[0]) & (e.cohort == "dice_ihb_china")]
        rows = []
        for _, p in prim.iterrows():
            rr = r3[(r3["method"] == p["method"]) & (r3["tail"] == p["tail"])]
            det_sc = e[(e.sc_threshold == THRESHOLDS[0]) & (e["method"] == p["method"]) & (e["tail"] == p["tail"])
                       & (e.cohort != "dice_ihb_china")]["detected_sc_family"].sum()
            rows.append({"method": p["method"], "tail": p["tail"], "sc_dice": p["detected_sc_family"],
                         "random_dice_median": rr.dice_ihb_china.median(),
                         "random_dice_95pct": rr.dice_ihb_china.quantile(0.95),
                         "sc_dice_percentile": float((rr.dice_ihb_china < p["detected_sc_family"]).mean()),
                         "sc_detections": float(det_sc),
                         "random_detections_median": float((rr.detected_ihb + rr.detected_china).median()),
                         "n_masks": len(rr), "mean_overlap_with_sc": rr.overlap_with_sc_mask.mean()})
        L.append(f"\n## E3: SC mask vs {n_masks} distance-matched random masks (SC >= {THRESHOLDS[0]})\n\n"
                 + _md(pd.DataFrame(rows)) + "\n")
    if (OUT / "e4_null_fwer.csv").exists():
        f = pd.read_csv(OUT / "e4_null_fwer.csv")
        rows = []
        for (c, m), g in f.groupby(["cohort", "method"]):
            for tail in ("positive", "negative"):
                k = int(g[f"any_{tail}"].sum())
                lo, hi = _binom_ci(k, len(g))
                rows.append({"cohort": c, "method": m, "tail": tail, "fwer": k / len(g), "ci_lo": lo, "ci_hi": hi, "n_null": len(g)})
        L.append(f"\n## E4: FWER within the SC family under the null ({n_null} EO/EC-swap datasets, "
                 f"{null_perm} permutations each)\n\n" + _md(pd.DataFrame(rows)) + "\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(L))

if __name__ == "__main__":
    main()
