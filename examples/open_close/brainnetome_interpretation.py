"""Brainnetome interpretation of the Open-Close results (eyes closed vs eyes open).

Re-runs the paper's paired analyses (5,000 permutations, seed 42, TFNBS E = 0.4,
H = 3, n = 10), checks them against the stored maps and describes them with the
Brainnetome atlas; the shared steps are in ``examples/_bna_interpretation.py``.

Analyses (positive = eyes closed > eyes open)
--------
openclose_ihb      IHB
openclose_china    Beijing, eyes-closed run 0 (paper contrast)

Sensitivity (not in the default set): Beijing scan 1 is always eyes closed and
precedes eyes open; scans 2 and 3 are eyes open / eyes closed in random order.
Close run 1 is the counterbalanced scan.

openclose_china_run1       eyes closed (counterbalanced run) vs open, 47 subjects
                           (sub-3258811 has an incomplete second close run)
openclose_china_run0_n47   openclose_china on the same 47 subjects

Schaefer-200 (182 retained parcels); the fMRIPrep output space is not recorded
with these time series, so the space is declared as generic MNI and every ROI is
flagged approximate.

Usage::

    python examples/open_close/brainnetome_interpretation.py [--quick] [--analyses ...] [--report NAME]

Outputs: ``examples/open_close/results/brainnetome/<analysis>/``,
``rois_openclose_schaefer200.csv`` and ``report.md``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import _bna_interpretation as B  # noqa: E402
from openclose_loader import OpenCloseDataset  # noqa: E402

from conninfpy import analyze  # noqa: E402
from conninfpy.annotate import annotate_rois  # noqa: E402

OUT = HERE / "results" / "brainnetome"
STORED = {"openclose_ihb": HERE / "results/ihb_paired_tfnbs.npz",
          "openclose_china": HERE / "results/china_paired_tfnbs_run0.npz"}
# name -> (experiment, close run, restrict to subjects with both close runs)
ANALYSES = {
    "openclose_ihb": ("ihb", 0, False),
    "openclose_china": ("china", 0, False),
    "openclose_china_run1": ("china", 1, True),
    "openclose_china_run0_n47": ("china", 0, True),
}


def rois_openclose(ref, ds):
    return B.with_source_networks(annotate_rois(
        reference=ref, names=list(ds.atlas_df["name"]), support="parcel", space="MNI152",
        parcel_image=B.SCHAEFER / "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
        parcel_labels=[int(i) for i in ds.atlas_df["id"]],
    ), ds.atlas_df["network"])


def run_paired(ds, n_perm, run=0, both_runs_only=False):
    open_z, close_z = ds.connectivity_z(run=run)
    if both_runs_only:
        keep = ds.retest_subject_mask()
        open_z, close_z = open_z[keep], close_z[keep]
    return analyze(group1=open_z, group2=close_z, test_type="paired", fisher_z=False,
                   n_permutations=n_perm, **B.TFNBS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="500 permutations (smoke test)")
    ap.add_argument("--analyses", nargs="+", default=["openclose_ihb", "openclose_china"], choices=list(ANALYSES))
    ap.add_argument("--report", default="report.md", help="report file name inside results/brainnetome/")
    args = ap.parse_args()
    n_perm = 500 if args.quick else 5000
    OUT.mkdir(parents=True, exist_ok=True)
    ref = B.reference()
    report = [B.report_header("Brainnetome interpretation of the Open-Close results",
                              "examples/open_close/brainnetome_interpretation.py", n_perm)]
    rois = None
    for name in args.analyses:
        t0 = time.time()
        exp, run, both_runs_only = ANALYSES[name]
        ds = OpenCloseDataset.load(exp)
        if rois is None:
            rois = rois_openclose(ref, ds)
            rois.table.to_csv(OUT / "rois_openclose_schaefer200.csv", index=False)
        n_sub = int(ds.retest_subject_mask().sum()) if both_runs_only else ds.n_subjects
        dataset = f"Open-Close {exp}, close run {run}, {n_sub} subjects, Schaefer-200 (182 parcels)"
        out = run_paired(ds, n_perm, run, both_runs_only)
        repro = B.reproduction(STORED.get(name), out.inference)
        edges, s, by_tail, sentence = B.describe(name, out, rois, {"A": "eyes open", "B": "eyes closed"}, dataset, OUT)
        B.connectograms(name, out.inference, rois, OUT)
        print(f"{name}: {time.time() - t0:.0f}s; {len(edges)} detections; reproduction: {repro}")
        report.append(B.report_section(name, dataset, repro, sentence, s, by_tail, out.inference))
    if rois is not None:
        report.append(B.roi_section("openclose", rois))
    (OUT / args.report).write_text("".join(report))
    print(f"report: {OUT / args.report}")


if __name__ == "__main__":
    main()
