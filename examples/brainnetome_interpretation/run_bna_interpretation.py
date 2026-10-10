"""Brainnetome interpretation of the ABIDE and Open-Close results.

Re-runs the paper's primary analyses with their own settings (5,000
permutations, seed 42, TFNBS E = 0.4, H = 3, n = 10), checks the p-value maps
against the stored ones, and describes the ROIs and the detected connections
with the Brainnetome atlas (BNA-246) via ``conninfpy.annotate``:

* ROIs by the tissue they sampled: the Schaefer parcel's overlap with BNA
  regions (Schaefer-100 for ABIDE, Schaefer-200 / 182 retained parcels for
  Open-Close), assigned at the finest level holding >= 50 % of the parcel;
* connections by hemispheric relation, anatomical level, circuit class,
  distance, population structural connectivity and Yeo relation;
* direction against baseline coupling, from the coefficient maps:
  stronger / weaker positive coupling, stronger / weaker anticorrelation,
  sign change (both datasets used global-signal regression, which induces
  part of the anticorrelations);
* BrainMap behavioural domains of the endpoint regions (forward-inference
  literature context, not functions of the connections).

Everything is descriptive: no test is added and the correction is unchanged.

Analyses
--------
abide_age         ABIDE, age, ComBat + site dummies (Strategy D), sex and motion covariates
abide_dx          ABIDE, ASD (1) vs control (0), Strategy D, age / sex / motion covariates
openclose_ihb     IHB, eyes closed vs open (paired; positive = closed > open)
openclose_china   Beijing, eyes closed (run 1) vs open (paired; positive = closed > open)

Sensitivity (not in the default set): Beijing scan 1 is always eyes closed and
precedes eyes open; scans 2 and 3 are eyes open / eyes closed in random order.
The close run 1 used here is the counterbalanced one.

openclose_china_run1       eyes closed (counterbalanced run) vs open, 47 subjects
                           (sub-3258811 has an incomplete second close run)
openclose_china_run0_n47   openclose_china on the same 47 subjects

Usage
-----
    python examples/brainnetome_interpretation/run_bna_interpretation.py [--quick] [--analyses ...] [--report NAME]

Outputs go to ``examples/brainnetome_interpretation/results/<analysis>/`` plus
``results/rois_<dataset>.csv`` and ``results/report.md``. Each analysis also
gets Brainnetome-style connectograms (``connectogram_{gyrus,subregion}.{png,pdf}``:
each parcel in the gyrus or subregion holding most of it), one panel per tail,
chords coloured by the endpoints' Yeo networks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from conninfpy import analyze  # noqa: E402
from conninfpy.annotate import annotate_rois, summarize_annotation  # noqa: E402
from conninfpy.brainnetome import BrainnetomeReference  # noqa: E402
from conninfpy.interpret.evidence import build_annotation_evidence, describe_coverage  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SCHAEFER = Path.home() / "nilearn_data" / "schaefer_2018"
ABIDE_FILE = ROOT / "examples" / "abide_validation" / "results" / "abide_prepared.npz"
STORED = {
    "abide_age": ROOT / "examples/abide_validation/results/age_development/age_strategy_d.npz",
    "abide_dx": ROOT / "examples/abide_validation/results/diagnosis/dx_strategy_d.npz",
    "openclose_ihb": ROOT / "examples/openclose_validation/results/ihb_paired_tfnbs.npz",
    "openclose_china": ROOT / "examples/openclose_validation/results/china_paired_tfnbs_run0.npz",
}
TFNBS = dict(method="tfnbs", e=0.4, h=3.0, n=10, acceleration=None, rng=42, use_mp=True)
# name -> (experiment, close run, restrict to subjects with both close runs)
OPENCLOSE = {
    "openclose_ihb": ("ihb", 0, False),
    "openclose_china": ("china", 0, False),
    "openclose_china_run1": ("china", 1, True),
    "openclose_china_run0_n47": ("china", 0, True),
}


# ----------------------------------------------------------------------
# Data and ROI annotation
# ----------------------------------------------------------------------
def _abide():
    d = np.load(ABIDE_FILE, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _with_source_networks(rois, networks):
    """Network relations use the source atlas's own Yeo labels (authoritative for
    Schaefer parcels); the Brainnetome crosswalk is kept as ``yeo7_bna``."""
    rois.table["yeo7_bna"] = rois.table["yeo7"]
    rois.table["yeo7"] = [str(x) for x in networks]
    return rois


def _rois_abide(ref, data):
    """Schaefer-100 parcels (FSL MNI152 2 mm, as extracted from the CPAC data)."""
    names = [str(x) for x in data["roi_names"]]
    order = list(data["network_order"])
    nets = [order[i] for i in np.asarray(data["net_labels"], int)]
    return _with_source_networks(annotate_rois(
        reference=ref, names=names, support="parcel", space="MNI152NLin6Asym",
        parcel_image=SCHAEFER / "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
        parcel_labels=list(range(1, 101)),
    ), nets)


def _openclose(experiment):
    sys.path.insert(0, str(ROOT / "examples" / "openclose_validation"))
    from openclose_loader import OpenCloseDataset

    return OpenCloseDataset.load(experiment)


def _rois_openclose(ref, ds):
    """The 182 retained Schaefer-200 parcels, by their original label IDs.

    The fMRIPrep output space is not recorded with these time series, so the
    space is declared as generic MNI and every ROI is flagged approximate.
    """
    return _with_source_networks(annotate_rois(
        reference=ref, names=list(ds.atlas_df["name"]), support="parcel", space="MNI152",
        parcel_image=SCHAEFER / "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
        parcel_labels=[int(i) for i in ds.atlas_df["id"]],
    ), ds.atlas_df["network"])


# ----------------------------------------------------------------------
# Analyses (paper settings)
# ----------------------------------------------------------------------
def run_abide_age(data, n_perm):
    conf = np.column_stack([data["sex"].astype(float), data["mean_fd"].astype(float)])
    return analyze(data["connectivity_z"], interest=data["age"].astype(float), confounds=conf,
                   sites=data["site"], harmonize="combat_site_dummies_glm", fisher_z=False,
                   n_permutations=n_perm, **TFNBS)


def run_abide_dx(data, n_perm):
    conf = np.column_stack([data["age"].astype(float), data["sex"].astype(float),
                            data["mean_fd"].astype(float)])
    return analyze(data["connectivity_z"], interest=data["group"].astype(float), confounds=conf,
                   sites=data["site"], harmonize="combat_site_dummies_glm", fisher_z=False,
                   n_permutations=n_perm, **TFNBS)


def run_paired(ds, n_perm, run=0, both_runs_only=False):
    open_z, close_z = ds.connectivity_z(run=run)
    if both_runs_only:
        keep = ds.retest_subject_mask()
        open_z, close_z = open_z[keep], close_z[keep]
    return analyze(group1=open_z, group2=close_z, test_type="paired", fisher_z=False,
                   n_permutations=n_perm, **TFNBS)


# ----------------------------------------------------------------------
# Description
# ----------------------------------------------------------------------
def _reproduction(name, inf):
    path = STORED.get(name)
    if path is None or not path.exists():
        return "no stored map"
    s = np.load(path)
    d = max(float(np.nanmax(np.abs(np.asarray(inf[t]) - s[t]))) for t in ("positive", "negative"))
    iu = np.triu_indices(s["positive"].shape[0], 1)
    same = all(np.array_equal(np.asarray(inf[t])[iu] < 0.05, s[t][iu] < 0.05) for t in ("positive", "negative"))
    return f"max |Δp| = {d:.3g}; same detections at α = 0.05: {same}"


def connectograms(name, inf, rois):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from conninfpy.plot import plot_connectogram

    for level in ("gyrus", "subregion"):
        fig, axes = plt.subplots(1, 2, figsize=(18, 9.2))
        for ax, tail in zip(axes, ("positive", "negative")):
            plot_connectogram(inf, rois, tail=tail, level=level, labels=level, ax=ax)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.04)
        fig.suptitle(f"{name}: Brainnetome {level} connectogram (α = 0.05 per tail)", fontsize=13)
        for ext in ("png", "pdf"):
            fig.savefig(OUT / name / f"connectogram_{level}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)


def describe(name, out, rois, labels, dataset):
    inf = out.inference
    if labels and inf.design is not None:
        prof = inf.design["profiles"]
        for old, new in labels.items():
            if old in prof:
                prof[new] = prof.pop(old)
        inf.design["comparison"] = [labels.get(x, x) for x in inf.design["comparison"]]
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    edges = inf.annotated_edges(rois, alpha=0.05)
    edges.to_csv(d / "connections_annotated.csv", index=False)
    summaries = {}
    for level in ("lobe", "network"):
        s = summarize_annotation(inf, rois, alpha=0.05, level=level)
        s["regions"].to_csv(d / f"groups_{level}.csv", index=False)
        summaries[level] = s
    s = summaries["lobe"]
    s["relations"].to_csv(d / "relations.csv", index=False)
    s["domains"].to_csv(d / "brainmap_domains.csv", index=False)
    s["distance_qc"].to_csv(d / "distance_qc.csv", index=False)
    by_tail = []
    for tail in ("positive", "negative"):
        sel = edges[edges[f"p_{tail}"] <= 0.05]
        counts = sel["coupling_change"].value_counts() if "coupling_change" in sel else pd.Series(dtype=int)
        for cls, k in counts.items():
            by_tail.append({"tail": tail, "coupling_change": cls, "n": int(k), "share": k / len(sel)})
    pd.DataFrame(by_tail).to_csv(d / "coupling_by_tail.csv", index=False)
    tested = s["edges"]
    tail_rows = []
    for col in ("circuit", "hemispheric_relation", "distance_band", "network_relation", "anatomical_relation"):
        for cat, g in tested.groupby(col):
            tail_rows.append({"relation": col, "category": cat, "n_tested": len(g),
                              "share_of_tested": len(g) / len(tested),
                              "n_positive": int((g["p_positive"] <= 0.05).sum()),
                              "n_negative": int((g["p_negative"] <= 0.05).sum())})
    by_rel = pd.DataFrame(tail_rows)
    for tail in ("positive", "negative"):
        tot = by_rel.groupby("relation")[f"n_{tail}"].transform("sum")
        by_rel[f"share_{tail}"] = np.where(tot > 0, by_rel[f"n_{tail}"] / tot, np.nan)
    by_rel.to_csv(d / "relations_by_tail.csv", index=False)
    s["by_tail"] = by_rel
    evidence = build_annotation_evidence(edges, rois, contrast_name=name, source_atlas=dataset,
                                         inference_unit=inf.inference_unit, summary=s)
    (d / "evidence.json").write_text(json.dumps(evidence, indent=1, default=str))
    return edges, s, pd.DataFrame(by_tail), describe_coverage(evidence)["report_sentence"] if evidence["rois"] else "no detections"


def _md_table(df, cols=None, n=None, fmt="{:.3f}"):
    df = df if cols is None else df[cols]
    df = df if n is None else df.head(n)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return head + "\n".join(rows) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="500 permutations (smoke test)")
    ap.add_argument("--analyses", nargs="+", default=["abide_age", "abide_dx", "openclose_ihb", "openclose_china"])
    ap.add_argument("--report", default="report.md", help="report file name inside results/")
    args = ap.parse_args()
    n_perm = 500 if args.quick else 5000
    OUT.mkdir(parents=True, exist_ok=True)
    ref = BrainnetomeReference.load(data_dir=ROOT / "datasets" / "atlases")
    print("reference:", json.dumps({k: v for k, v in ref.provenance.items() if k != "profiles"}, default=str))

    report = ["# Brainnetome interpretation of the ABIDE and Open-Close results\n",
              f"Generated by `examples/brainnetome_interpretation/run_bna_interpretation.py` "
              f"({n_perm} permutations, TFNBS E = 0.4, H = 3; α = 0.05 per tail). Descriptive: "
              "no test is added and the correction is unchanged. Both datasets used global-signal "
              "regression, which induces part of the anticorrelations. BrainMap domains are "
              "forward-inference literature context of the endpoint regions, not functions of the "
              "connections.\n"]
    roi_cache = {}
    abide = None
    for name in args.analyses:
        t0 = time.time()
        if name.startswith("abide"):
            abide = abide if abide is not None else _abide()
            if "abide" not in roi_cache:
                roi_cache["abide"] = _rois_abide(ref, abide)
                roi_cache["abide"].table.to_csv(OUT / "rois_abide_schaefer100.csv", index=False)
            rois, dataset = roi_cache["abide"], "ABIDE I, Schaefer-100 / Yeo-7 (CPAC, FSL MNI152)"
            out = run_abide_age(abide, n_perm) if name == "abide_age" else run_abide_dx(abide, n_perm)
            if name == "abide_age":
                prof = out.inference.design["profiles"]
                labels = {k: f"age {float(k.split('=')[1].split()[0]):.1f} y" for k in prof if k.startswith("interest=")}
            else:
                labels = {"interest=0": "control", "interest=1": "ASD"}
        else:
            exp, run, both_runs_only = OPENCLOSE[name]
            ds = _openclose(exp)
            if "openclose" not in roi_cache:
                roi_cache["openclose"] = _rois_openclose(ref, ds)
                roi_cache["openclose"].table.to_csv(OUT / "rois_openclose_schaefer200.csv", index=False)
            n_sub = int(ds.retest_subject_mask().sum()) if both_runs_only else ds.n_subjects
            rois, dataset = (roi_cache["openclose"],
                             f"Open-Close {exp}, close run {run}, {n_sub} subjects, Schaefer-200 (182 parcels)")
            out = run_paired(ds, n_perm, run, both_runs_only)
            labels = {"A": "eyes open", "B": "eyes closed"}
        repro = _reproduction(name, out.inference)
        edges, s, by_tail, sentence = describe(name, out, rois, labels, dataset)
        connectograms(name, out.inference, rois)
        print(f"{name}: {time.time() - t0:.0f}s; {len(edges)} detections; reproduction: {repro}")

        rel = s["by_tail"]
        report.append(f"\n## {name}\n\n{dataset}. Reproduction against the stored map: {repro}.\n\n{sentence}\n"
                      f"\nConnectograms: `{name}/connectogram_gyrus.pdf`, `{name}/connectogram_subregion.pdf`.\n")
        for col, title in (("circuit", "Circuit class"), ("hemispheric_relation", "Hemispheric relation"),
                           ("anatomical_relation", "Anatomical level"), ("distance_band", "Distance"),
                           ("network_relation", "Yeo relation (source labels)")):
            t = rel[rel.relation == col][["category", "n_positive", "share_positive", "n_negative",
                                          "share_negative", "share_of_tested"]].sort_values("share_of_tested", ascending=False)
            report.append(f"\n**{title}**: detections per tail vs tested opportunities\n\n" + _md_table(t))
        if not by_tail.empty:
            report.append("\n**Coupling change of the detected connections** (from the coefficient maps; "
                          f"comparison {out.inference.design['comparison'][0]} → "
                          f"{out.inference.design['comparison'][1]})\n\n"
                          + _md_table(by_tail.sort_values(["tail", "n"], ascending=[True, False])))
        g = s["regions"].sort_values("n_detected", ascending=False)
        report.append("\n**Lobe groups** (ROIs not ≥ 50 % in one lobe are 'mixed')\n\n" + _md_table(g, n=12))
        report.append("\n**BrainMap domains** (unique ROIs; overlapping memberships)\n\n" + _md_table(s["domains"]))
        report.append("\n**Distance QC**\n\n" + _md_table(s["distance_qc"]))
    for key, rois in roi_cache.items():
        lv = rois.table["level"].value_counts().to_dict()
        report.append(f"\n## ROI annotation: {key}\n\nAssignment level of the {len(rois)} parcels: {lv}. "
                      f"Median subregion purity {rois.table['region_purity'].median():.2f}.\n")
    (OUT / args.report).write_text("".join(report))
    print(f"report: {OUT / args.report}")


if __name__ == "__main__":
    main()
