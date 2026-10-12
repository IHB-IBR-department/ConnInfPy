"""NI-TFNBS with blocks formed by structural connectivity, across module resolutions.

NI-TFNBS weights each edge by the activation density of its block (a pair of
modules); the partition is any support structure (the paper's analyses used
Yeo-7 as an example). Here the modules come from the Brainnetome structural
connectivity: Louvain modularity on the cortical SC, medoid of 50 seeds, at every
resolution gamma = 0.8, 0.9, ..., 2.0 (about 5-12 cortical modules; identical
partitions are run once). Subcortex is one module. Every edge belongs to a block,
so every edge is tested.

Native BNA-246 (242 regions, GSR); IHB (84), China EO vs EC1 (47; EC0 as a
sensitivity); paired, positive = EC > EO; paper settings (per-tail, E = 0.4,
H = 3, n = 10, 5,000 permutations, seed 42). References: standard TFNBS (no
prior) and NI-TFNBS with the Brainnetome Yeo-7 crosswalk + subcortex.

Reported per resolution: number of blocks, detections per tail and cohort,
IHB-China Dice of detected edges (same edge axis), share of between-hemisphere
detections, and agreement of the detected sets between neighbouring resolutions.

Usage::

    python examples/open_close/structural_prior/run_sc_blocks.py [--quick]
"""
from __future__ import annotations


from pathlib import Path as _Path
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from pathlib import Path
from examples._shared.tables import markdown_table
import argparse
import time

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from examples.open_close.structural_prior import analysis as R

OUT = R.HERE / "results" / "blocks"
COHORTS = {"ihb": ("ihb", 0, False), "china_ec1": ("china", 1, True), "china_ec0": ("china", 0, True)}
GAMMAS = tuple(float(g) for g in np.round(np.arange(0.8, 2.01, 0.1), 2))
N_SEEDS = 50


def louvain_medoid(sc, gamma):
    """Medoid of N_SEEDS Louvain partitions (highest mean ARI with the others) and that stability."""
    G = nx.from_numpy_array(sc)
    runs = []
    for s in range(N_SEEDS):
        lab = np.empty(len(sc), int)
        for k, part in enumerate(nx.community.louvain_communities(G, weight="weight", resolution=gamma, seed=s)):
            lab[list(part)] = k
        runs.append(lab)
    A = np.array([[adjusted_rand_score(a, b) for b in runs] for a in runs])
    return runs[int(A.mean(1).argmax())], float(A[np.triu_indices(N_SEEDS, 1)].mean())


def sc_partitions(sc, regions):
    """One partition per resolution (cortical SC modules + subcortex as one module); duplicates dropped."""
    ctx = (regions["structure"] == "cortex").to_numpy()
    sc_ctx = sc[np.ix_(ctx, ctx)].copy()
    np.fill_diagonal(sc_ctx, 0.0)
    parts, rows = {}, []
    for g in GAMMAS:
        lab, stab = louvain_medoid(sc_ctx, g)
        full = np.full(len(ctx), -1)
        full[ctx] = lab
        full[~ctx] = lab.max() + 1
        dup = next((k for k, v in parts.items() if adjusted_rand_score(v, full) == 1.0), None)
        rows.append({"gamma": g, "k_cortical": len(set(lab)), "stability_ari": stab, "same_as": dup or ""})
        if dup is None:
            parts[f"sc_g{g:.1f}"] = full
    return parts, pd.DataFrame(rows)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="500 permutations")
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--report", type=Path, help="Optional Markdown report path in the project vault")
    args = ap.parse_args()
    OUT = args.output.resolve()
    n_perm = 500 if args.quick else 5000
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data, sc, dist, hemi, cortex, nets, yeo_labels, ids = R.load(COHORTS)
    from conninfpy.atlas.brainnetome import BrainnetomeReference
    regions = BrainnetomeReference.load(profiles=False).regions.loc[ids]
    edges = R.edge_table(sc, dist, hemi, cortex, nets)
    parts, scan = sc_partitions(sc, regions)
    scan.to_csv(OUT / "sc_module_scan.csv", index=False)
    pd.DataFrame({"bna_id": ids, "name": regions["name"].to_numpy(), "lobe": regions["lobe"].to_numpy(),
                  "hemisphere": regions["hemisphere"].to_numpy(), "yeo": nets, **parts}).to_csv(OUT / "partitions.csv", index=False)
    print(f"[{time.time() - t0:5.0f}s] {len(parts)} distinct SC partitions", flush=True)

    priors = {"tfnbs": None, "yeo": yeo_labels, **parts}
    rows, detsets = [], {}
    for prior, labels in priors.items():
        method = "tfnbs" if labels is None else "ni_tfnbs"
        res = {c: R.run(*data[c], method, labels, None, n_perm) for c in COHORTS}
        (OUT / "maps").mkdir(exist_ok=True)
        for c, r in res.items():
            np.savez_compressed(OUT / "maps" / f"{c}__{prior}.npz", positive=r.positive, negative=r.negative,
                                stat_positive=r.stat_positive, stat_negative=r.stat_negative,
                                labels=np.full(len(ids), -1) if labels is None else labels)
        for tail in ("positive", "negative"):
            det = {c: R.detected(res[c], tail, edges) for c in COHORTS}
            detsets[(prior, tail)] = det
            rows.append({"prior": prior, "n_blocks_modules": 0 if labels is None else len(set(labels)),
                         "tail": tail, **{f"detected_{c}": len(det[c]) for c in COHORTS},
                         "dice_ihb_china_ec1": R.dice(det["ihb"], det["china_ec1"]),
                         "dice_ihb_china_ec0": R.dice(det["ihb"], det["china_ec0"]),
                         "between_hemisphere_share_ihb": float(np.mean(edges.inter.to_numpy()[list(det["ihb"])])) if det["ihb"] else np.nan,
                         "between_hemisphere_share_china_ec1": float(np.mean(edges.inter.to_numpy()[list(det["china_ec1"])])) if det["china_ec1"] else np.nan})
        print(f"[{time.time() - t0:5.0f}s] {prior}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "sc_block_priors.csv", index=False)

    # agreement of detected sets between neighbouring resolutions and with the references
    names = list(parts)
    agree = []
    for tail in ("positive", "negative"):
        for c in ("ihb", "china_ec1"):
            for a, b in zip(names[:-1], names[1:]):
                agree.append({"tail": tail, "cohort": c, "pair": f"{a} vs {b}",
                              "dice": R.dice(detsets[(a, tail)][c], detsets[(b, tail)][c])})
            for n in names:
                for ref in ("tfnbs", "yeo"):
                    agree.append({"tail": tail, "cohort": c, "pair": f"{n} vs {ref}",
                                  "dice": R.dice(detsets[(n, tail)][c], detsets[(ref, tail)][c])})
    pd.DataFrame(agree).to_csv(OUT / "resolution_agreement.csv", index=False)
    if args.report is not None:
        report(pd.DataFrame(rows), scan, pd.DataFrame(agree), n_perm, args.report)
    print(f"[{time.time() - t0:5.0f}s] done -> {OUT}", flush=True)


def report(df, scan, agree, n_perm, report_path: Path):
    L = ["# NI-TFNBS with structural-connectivity blocks across resolutions\n",
         f"Native BNA-246 (242 regions, GSR); IHB 84, China 47 (EC1; EC0 sensitivity); {n_perm} permutations, "
         "per-tail, seed 42. Blocks = pairs of SC modules (Louvain on cortical SC, medoid of 50 seeds) "
         "with subcortex as one module. References: standard TFNBS and NI-TFNBS with the Yeo-7 crosswalk.\n",
         "\n## Resolution scan\n\n" + markdown_table(scan) + "\n",
         "\n## Results per prior\n\n" + markdown_table(df) + "\n",
         "\n## Agreement of detected edges (Dice) between neighbouring resolutions and with the references\n\n"
         + markdown_table(agree) + "\n"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(L))


if __name__ == "__main__":
    main()
