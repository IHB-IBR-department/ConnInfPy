"""Brainnetome interpretation of the ABIDE results (age and diagnosis).

Re-runs the paper's ABIDE analyses (Strategy D: ComBat + site dummies; 5,000
permutations, seed 42, TFNBS E = 0.4, H = 3, n = 10), checks them against the
stored maps and describes them with :func:`conninfpy.interpret.interpret_result`.
ROIs are the Schaefer-100 parcels annotated by the Brainnetome tissue they cover;
network labels are the parcels' own Yeo-7 labels.

Analyses
--------
abide_age   age, sex and motion covariates
abide_dx    ASD (1) vs control (0), age / sex / motion covariates

Usage::

    python examples/abide/brainnetome_interpretation.py [--quick] [--analyses abide_age abide_dx]

Outputs go next to the result they describe: ``results/age/brainnetome/`` and
``results/diagnosis/brainnetome/`` (tables, evidence packet, connectograms,
ROI table and ``report.md``).
"""
from __future__ import annotations


from pathlib import Path as _Path
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from conninfpy import analyze  # noqa: E402
from conninfpy.atlas.brainnetome import BrainnetomeReference  # noqa: E402
from conninfpy.interpret import annotate_rois, interpret_result  # noqa: E402
from conninfpy.interpret.report import reproduction_note, roi_markdown  # noqa: E402

RESULTS = HERE / "results"
ABIDE_FILE = RESULTS / "abide_prepared.npz"
SCHAEFER_100 = Path.home() / "nilearn_data/schaefer_2018/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
# analysis -> (results folder of its runner, stored paper map)
ANALYSES = {"abide_age": (RESULTS / "age", RESULTS / "age/age_strategy_d.npz"),
            "abide_dx": (RESULTS / "diagnosis", RESULTS / "diagnosis/dx_strategy_d.npz")}
TFNBS = dict(method="tfnbs", e=0.4, h=3.0, n=10, acceleration=None, rng=42, use_mp=True)


def run(name, data, n_perm):
    conf_cols = ("sex", "mean_fd") if name == "abide_age" else ("age", "sex", "mean_fd")
    interest = data["age"] if name == "abide_age" else data["group"]
    conf = np.column_stack([data[c].astype(float) for c in conf_cols])
    return analyze(data["connectivity_z"], interest=interest.astype(float), confounds=conf,
                   sites=data["site"], harmonize="combat_site_dummies_glm", fisher_z=False,
                   n_permutations=n_perm, **TFNBS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="500 permutations (smoke test)")
    ap.add_argument("--analyses", nargs="+", default=list(ANALYSES), choices=list(ANALYSES))
    ap.add_argument("--report", type=Path, help="Optional combined Markdown report path in the project vault")
    args = ap.parse_args()
    n_perm = 500 if args.quick else 5000
    data = dict(np.load(ABIDE_FILE, allow_pickle=True))
    ref = BrainnetomeReference.load()
    order = list(data["network_order"])
    rois = annotate_rois(reference=ref, names=[str(x) for x in data["roi_names"]], support="parcel",
                         space="MNI152NLin6Asym", parcel_image=SCHAEFER_100, parcel_labels=list(range(1, 101)),
                         networks=[order[i] for i in np.asarray(data["net_labels"], int)])
    roi_section = roi_markdown(rois, "ABIDE, Schaefer-100")
    dataset = "ABIDE I, Schaefer-100 / Yeo-7 (CPAC, FSL MNI152)"
    header = (f"# Brainnetome interpretation of the ABIDE results\n\n`examples/abide/brainnetome_interpretation.py`, "
              f"{n_perm} permutations, TFNBS E = 0.4, H = 3, α = 0.05 per tail. Descriptive: no test is added and the "
              "correction is unchanged. The data used global-signal regression, which induces part of the "
              "anticorrelations. BrainMap domains are literature context of the endpoint regions.\n")
    report = [header]
    for name in args.analyses:
        t0 = time.time()
        out = run(name, data, n_perm)
        if name == "abide_age":
            labels = {k: f"age {float(k.split('=')[1].split()[0]):.1f} y"
                      for k in out.inference.design["profiles"] if k.startswith("interest=")}
        else:
            labels = {"interest=0": "control", "interest=1": "ASD"}
        results_dir, stored = ANALYSES[name]
        repro = reproduction_note(out.inference, stored)
        rep = interpret_result(out.inference, rois, name=name, source_atlas=dataset, labels=labels)
        out_dir = rep.save(results_dir / "brainnetome")
        rois.table.to_csv(out_dir / "rois_abide_schaefer100.csv", index=False)
        section = rep.markdown(reproduction=repro)
        (out_dir / "report.md").write_text(header + section + roi_section)
        report.append(section)
        print(f"{name}: {time.time() - t0:.0f}s; {len(rep.edges)} detections; reproduction: {repro} -> {out_dir}")
    report.append(roi_section)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("".join(report))
        print(f"report: {args.report}")


if __name__ == "__main__":
    main()
