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


from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

from conninfpy import compute_p_val  # noqa: E402
from conninfpy.atlas.brainnetome import BrainnetomeReference  # noqa: E402
from examples.open_close._shared.data import OpenCloseDataset  # noqa: E402

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
def fisher_z(ts):
    out = np.empty((ts.shape[0], ts.shape[2], ts.shape[2]))
    for s, x in enumerate(ts):
        c = np.clip(np.corrcoef(x.T), -0.9999, 0.9999)
        np.fill_diagonal(c, 0.0)
        out[s] = np.arctanh(c)
    return out

def load(cohorts=None):
    cohorts = COHORTS if cohorts is None else cohorts
    ref = BrainnetomeReference.load(profiles=False)
    data, ids, nets = {}, None, None
    for cohort, (exp, run, matched) in cohorts.items():
        ds = OpenCloseDataset.load(exp, atlas="brainnetome")
        keep = ds.retest_subject_mask() if matched else np.ones(ds.n_subjects, bool)
        eo_z, ec_z = ds.connectivity_z(run=run)
        data[cohort] = (eo_z[keep], ec_z[keep])
        ids, nets = ds.atlas_df["id"].to_numpy(int), list(ds.atlas_df["network"])
    sc = ref.structural_connectivity()
    if sc is None:
        raise SystemExit(f"Brainnetome SC (BNA_SC_3D_246.zip) not found in {ref.data_dir}; "
                         "set CONNINFPY_DATA or CONNINFPY_BNA_DIR (see atlas-data documentation).")
    sc = np.nan_to_num(sc[np.ix_(ids - 1, ids - 1)], nan=0.0)
    r = ref.regions.loc[ids]
    xyz = r[["x", "y", "z"]].to_numpy(float)
    dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    hemi = r["hemisphere"].to_numpy()
    cortex = (r["structure"] == "cortex").to_numpy()
    labels = np.unique(nets, return_inverse=True)[1]
    return data, sc, dist, hemi, cortex, np.asarray(nets), labels, ids

def edge_table(sc, dist, hemi, cortex, nets):
    N = len(hemi)
    i, j = np.triu_indices(N, 1)
    inter = hemi[i] != hemi[j]
    struct = np.where(cortex[i] & cortex[j], "cortex", np.where(cortex[i] | cortex[j], "cortex-subcortex", "subcortex"))
    cls = np.where(~(cortex[i] & cortex[j]), "subcortical",
                   np.where(np.isin(nets[i], list(SENSORY)) & np.isin(nets[j], list(SENSORY)), "sensory",
                            np.where(~np.isin(nets[i], list(SENSORY)) & ~np.isin(nets[j], list(SENSORY)),
                                     "association", "sensory-association")))
    bins = np.minimum(dist[i, j] // 10, 14).astype(int)
    stratum = [f"{b}|{'inter' if x else 'intra'}|{s}" for b, x, s in zip(bins, inter, struct)]
    return pd.DataFrame({"i": i, "j": j, "sc": sc[i, j], "dist": dist[i, j], "inter": inter,
                         "structure": struct, "system": cls, "stratum": stratum})

def to_mask(sel, N, edges):
    m = np.zeros((N, N), dtype=bool)
    m[edges.i[sel], edges.j[sel]] = True
    return m | m.T

def paired_t(eo, ec):
    d = ec - eo
    return d.mean(0) / (d.std(0, ddof=1) / np.sqrt(d.shape[0]))

def dice(a, b):
    return 2 * len(a & b) / (len(a) + len(b)) if (a or b) else np.nan

def detected(res, tail, edges):
    p = np.asarray(res[tail])[edges.i, edges.j]
    return set(np.flatnonzero(p < ALPHA))

def stratified_difference(abs_t, sup, strata, min_n=10):
    num = den = 0.0
    for s in np.unique(strata):
        k = strata == s
        a, b = abs_t[k & sup], abs_t[k & ~sup]
        if len(a) >= min_n and len(b) >= min_n:
            w = len(a) * len(b) / (len(a) + len(b))
            num += w * (a.mean() - b.mean())
            den += w
    return num / den if den else np.nan

def e0(data, edges, n_boot):
    rows = []
    strata = edges["stratum"].to_numpy()
    for tau in THRESHOLDS:
        sup = edges["sc"].to_numpy() >= tau
        for cohort, (eo, ec) in data.items():
            d = (ec - eo)[:, edges.i, edges.j]                       # (S, E)
            S = d.shape[0]
            rng = np.random.default_rng(SEED)
            idx = rng.integers(0, S, size=(n_boot, S))
            for system in ("all", "sensory", "association", "sensory-association", "subcortical"):
                k = np.ones(len(edges), bool) if system == "all" else (edges["system"] == system).to_numpy()

                def stat(dd):
                    t = dd.mean(0) / (dd.std(0, ddof=1) / np.sqrt(dd.shape[0]))
                    return stratified_difference(np.abs(t[k]), sup[k], strata[k])

                full = stat(d)
                boot = np.array([stat(d[b]) for b in idx])
                lo, hi = np.nanpercentile(boot, [2.5, 97.5])
                rows.append({"sc_threshold": tau, "cohort": cohort, "system": system, "n_edges": int(k.sum()),
                             "n_supported": int((k & sup).sum()), "diff_abs_t": full, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)

def run(eo, ec, method, labels, mask, n_perm):
    kw = dict(SETTINGS, method=method, n_permutations=n_perm)
    if method == "ni_tfnbs":
        kw["net_labels"] = labels
    if mask is not None:
        kw["edge_mask"] = mask
    return compute_p_val(eo, ec, **kw)

def matched_random_masks(edges, sup, n_masks):
    strata = edges["stratum"].to_numpy()
    out = []
    for r in range(n_masks):
        rng = np.random.default_rng(SEED + 1 + r)
        sel = np.zeros(len(edges), bool)
        for s in np.unique(strata):
            pool = np.flatnonzero(strata == s)
            k = int(sup[pool].sum())
            if k:
                sel[rng.choice(pool, size=k, replace=False)] = True
        out.append(sel)
    return out
