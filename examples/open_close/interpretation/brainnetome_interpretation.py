"""Brainnetome interpretation of the Open-Close results (eyes closed vs eyes open).

Re-runs the paper's paired analyses (5,000 permutations, seed 42, TFNBS E = 0.4,
H = 3, n = 10), checks them against the stored maps and describes them with
:func:`conninfpy.interpret.interpret_result`.

``--atlas schaefer200`` (default): the 182 retained Schaefer-200 parcels,
annotated by the Brainnetome tissue they cover; network labels are the parcels'
own Yeo-7 labels. The fMRIPrep output space is not recorded with these time
series, so the space is declared as generic MNI and every ROI is flagged
approximate. ``--atlas brainnetome``: the native BNA-246 series (242 regions);
each ROI is its Brainnetome region and network labels come from the Brainnetome
division (``bna246_hierarchy.csv``).

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

Usage::

    python examples/open_close/interpretation/brainnetome_interpretation.py [--atlas schaefer200|brainnetome] [--quick]
        [--analyses ...] [--report NAME]

Outputs: ``examples/open_close/results/brainnetome[_native]/<analysis>/`` (tables,
evidence packet, connectograms), the ROI table and the report.
"""
from __future__ import annotations


from pathlib import Path as _Path
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

from conninfpy import analyze  # noqa: E402
from conninfpy.atlas.brainnetome import BrainnetomeReference  # noqa: E402
from conninfpy.interpret import annotate_bna_regions, annotate_rois, interpret_result  # noqa: E402
from conninfpy.interpret.report import reproduction_note, roi_markdown  # noqa: E402
from examples.open_close._shared.data import OpenCloseDataset  # noqa: E402

SCHAEFER_200 = Path.home() / "nilearn_data/schaefer_2018/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
STORED = {"openclose_ihb": HERE / "results/ihb_paired_tfnbs.npz",
          "openclose_china": HERE / "results/china_paired_tfnbs_run0.npz"}
TFNBS = dict(method="tfnbs", e=0.4, h=3.0, n=10, acceleration=None, rng=42, use_mp=True)
# name -> (experiment, close run, restrict to subjects with both close runs)
ANALYSES = {
    "openclose_ihb": ("ihb", 0, False),
    "openclose_china": ("china", 0, False),
    "openclose_china_run1": ("china", 1, True),
    "openclose_china_run0_n47": ("china", 0, True),
}


def annotate(ref, ds):
    if ds.atlas == "brainnetome":
        return annotate_bna_regions(ds.atlas_df["id"], reference=ref)
    return annotate_rois(reference=ref, names=list(ds.atlas_df["name"]), support="parcel", space="MNI152",
                         parcel_image=SCHAEFER_200, parcel_labels=[int(i) for i in ds.atlas_df["id"]],
                         networks=list(ds.atlas_df["network"]))


def run_paired(ds, n_perm, run=0, both_runs_only=False):
    open_z, close_z = ds.connectivity_z(run=run)
    if both_runs_only:
        keep = ds.retest_subject_mask()
        open_z, close_z = open_z[keep], close_z[keep]
    return analyze(group1=open_z, group2=close_z, test_type="paired", fisher_z=False,
                   n_permutations=n_perm, **TFNBS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="schaefer200", choices=["schaefer200", "brainnetome"])
    ap.add_argument("--quick", action="store_true", help="500 permutations (smoke test)")
    ap.add_argument("--analyses", nargs="+", default=["openclose_ihb", "openclose_china"], choices=list(ANALYSES))
    ap.add_argument("--report", type=Path, help="Optional Markdown report path in the project vault")
    args = ap.parse_args()
    n_perm = 500 if args.quick else 5000
    native = args.atlas == "brainnetome"
    out_dir = HERE / "results" / ("brainnetome_native" if native else "brainnetome")
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = BrainnetomeReference.load(data_dir=HERE.parents[1] / "datasets" / "atlases")
    atlas_name = "BNA-246 (242 regions, native)" if native else "Schaefer-200 (182 parcels)"
    report = [f"# Brainnetome interpretation of the Open-Close results ({atlas_name})\n\n"
              f"`examples/open_close/interpretation/brainnetome_interpretation.py --atlas {args.atlas}`, {n_perm} permutations, "
              "TFNBS E = 0.4, H = 3, α = 0.05 per tail. Descriptive: no test is added and the correction is "
              "unchanged. The data used global-signal regression, which induces part of the anticorrelations. "
              "BrainMap domains are literature context of the endpoint regions.\n"]
    rois = None
    for name in args.analyses:
        t0 = time.time()
        exp, run, both_runs_only = ANALYSES[name]
        ds = OpenCloseDataset.load(exp, atlas=args.atlas)
        if rois is None:
            rois = annotate(ref, ds)
            rois.table.to_csv(out_dir / f"rois_openclose_{'bna242' if native else 'schaefer200'}.csv", index=False)
        n_sub = int(ds.retest_subject_mask().sum()) if both_runs_only else ds.n_subjects
        dataset = f"Open-Close {exp}, close run {run}, {n_sub} subjects, {atlas_name}"
        out = run_paired(ds, n_perm, run, both_runs_only)
        repro = "no stored map" if native else reproduction_note(out.inference, STORED.get(name))
        rep = interpret_result(out.inference, rois, name=name, source_atlas=dataset,
                               labels={"A": "eyes open", "B": "eyes closed"})
        rep.save(out_dir / name)
        report.append(rep.markdown(reproduction=repro))
        print(f"{name}: {time.time() - t0:.0f}s; {len(rep.edges)} detections; reproduction: {repro}")
    if rois is not None:
        report.append(roi_markdown(rois, f"Open-Close, {atlas_name}"))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("".join(report))
        print(f"report: {args.report}")


if __name__ == "__main__":
    main()
