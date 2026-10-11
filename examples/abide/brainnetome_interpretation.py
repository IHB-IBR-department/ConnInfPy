"""Brainnetome interpretation of the ABIDE results (age and diagnosis).

Re-runs the paper's ABIDE analyses (Strategy D: ComBat + site dummies; 5,000
permutations, seed 42, TFNBS E = 0.4, H = 3, n = 10), checks them against the
stored maps and describes them with the Brainnetome atlas; the shared steps are
in ``examples/_bna_interpretation.py``.

Analyses
--------
abide_age   age, sex and motion covariates
abide_dx    ASD (1) vs control (0), age / sex / motion covariates

Schaefer-100 parcels (FSL MNI152 2 mm, as extracted from the CPAC data).

Usage::

    python examples/abide/brainnetome_interpretation.py [--quick] [--analyses abide_age abide_dx]

Outputs: ``examples/abide/results/brainnetome/<analysis>/``, ``rois_abide_schaefer100.csv``
and ``report.md``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _bna_interpretation as B  # noqa: E402

from conninfpy import analyze  # noqa: E402
from conninfpy.annotate import annotate_rois  # noqa: E402

OUT = HERE / "results" / "brainnetome"
ABIDE_FILE = HERE / "results" / "abide_prepared.npz"
STORED = {"abide_age": HERE / "results/age_development/age_strategy_d.npz",
          "abide_dx": HERE / "results/diagnosis/dx_strategy_d.npz"}


def load():
    d = np.load(ABIDE_FILE, allow_pickle=True)
    return {k: d[k] for k in d.files}


def rois_abide(ref, data):
    names = [str(x) for x in data["roi_names"]]
    order = list(data["network_order"])
    nets = [order[i] for i in np.asarray(data["net_labels"], int)]
    return B.with_source_networks(annotate_rois(
        reference=ref, names=names, support="parcel", space="MNI152NLin6Asym",
        parcel_image=B.SCHAEFER / "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
        parcel_labels=list(range(1, 101)),
    ), nets)


def run(name, data, n_perm):
    conf_cols = ("sex", "mean_fd") if name == "abide_age" else ("age", "sex", "mean_fd")
    interest = data["age"] if name == "abide_age" else data["group"]
    conf = np.column_stack([data[c].astype(float) for c in conf_cols])
    return analyze(data["connectivity_z"], interest=interest.astype(float), confounds=conf,
                   sites=data["site"], harmonize="combat_site_dummies_glm", fisher_z=False,
                   n_permutations=n_perm, **B.TFNBS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="500 permutations (smoke test)")
    ap.add_argument("--analyses", nargs="+", default=["abide_age", "abide_dx"], choices=["abide_age", "abide_dx"])
    ap.add_argument("--report", default="report.md", help="report file name inside results/brainnetome/")
    args = ap.parse_args()
    n_perm = 500 if args.quick else 5000
    OUT.mkdir(parents=True, exist_ok=True)
    ref = B.reference()
    data = load()
    rois = rois_abide(ref, data)
    rois.table.to_csv(OUT / "rois_abide_schaefer100.csv", index=False)
    dataset = "ABIDE I, Schaefer-100 / Yeo-7 (CPAC, FSL MNI152)"
    report = [B.report_header("Brainnetome interpretation of the ABIDE results",
                              "examples/abide/brainnetome_interpretation.py", n_perm)]
    for name in args.analyses:
        t0 = time.time()
        out = run(name, data, n_perm)
        if name == "abide_age":
            prof = out.inference.design["profiles"]
            labels = {k: f"age {float(k.split('=')[1].split()[0]):.1f} y" for k in prof if k.startswith("interest=")}
        else:
            labels = {"interest=0": "control", "interest=1": "ASD"}
        repro = B.reproduction(STORED.get(name), out.inference)
        edges, s, by_tail, sentence = B.describe(name, out, rois, labels, dataset, OUT)
        B.connectograms(name, out.inference, rois, OUT)
        print(f"{name}: {time.time() - t0:.0f}s; {len(edges)} detections; reproduction: {repro}")
        report.append(B.report_section(name, dataset, repro, sentence, s, by_tail, out.inference))
    report.append(B.roi_section("abide", rois))
    (OUT / args.report).write_text("".join(report))
    print(f"report: {OUT / args.report}")


if __name__ == "__main__":
    main()
