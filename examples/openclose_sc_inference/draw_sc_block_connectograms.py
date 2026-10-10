"""Connectograms of the NI-TFNBS results with structural-connectivity blocks (``run_sc_blocks.py``).

Native BNA-246 regions are annotated exactly (each region is its own parcel, on the
atlas's own label image). Chords are coloured by the endpoints' Yeo-7 networks
(Brainnetome crosswalk; subcortex grey).

Outputs in ``results_blocks/connectograms/``:

* ``<cohort>__<prior>.{png,pdf}``: gyrus-level connectogram, both tails;
* ``overview_<cohort>_<tail>.png``: one small connectogram per prior, from
  standard TFNBS and the Yeo example across the SC resolutions.

Usage::

    python examples/openclose_sc_inference/draw_sc_block_connectograms.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "openclose_validation"))

from conninfpy.annotate import annotate_rois  # noqa: E402
from conninfpy.atlas import get_bna_246_nifti_path  # noqa: E402
from conninfpy.brainnetome import BrainnetomeReference  # noqa: E402
from conninfpy.plot import plot_connectogram  # noqa: E402
from conninfpy.results import InferenceResult  # noqa: E402
from openclose_loader import OpenCloseDataset  # noqa: E402

RES = HERE / "results_blocks"
OUT = RES / "connectograms"
TAILS = {"positive": "EC > EO", "negative": "EO > EC"}
COHORT_NAMES = {"ihb": "IHB", "china_ec1": "China (EC1)", "china_ec0": "China (EC0)"}


def prior_label(prior: str, n_modules: int) -> str:
    if prior == "tfnbs":
        return "TFNBS (no prior)"
    if prior == "yeo":
        return "NI-TFNBS, Yeo-7 + subcortex"
    return f"NI-TFNBS, SC modules γ {prior[4:]} ({n_modules - 1} cortical + subcortex)"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = BrainnetomeReference.load(data_dir=ROOT / "datasets" / "atlases", profiles=False)
    ds = OpenCloseDataset.load("ihb", atlas="brainnetome")
    ids = [int(i) for i in ds.atlas_df["id"]]
    nets = list(ds.atlas_df["network"])
    rois = annotate_rois(reference=ref, names=list(ds.atlas_df["name"]), support="parcel", space="MNI152NLin6Asym",
                         parcel_image=get_bna_246_nifti_path(1), parcel_labels=ids)
    table = pd.read_csv(RES / "sc_block_priors.csv")
    order = ["tfnbs", "yeo"] + [p for p in dict.fromkeys(table["prior"]) if p.startswith("sc_")]

    results = {}
    for path in sorted((RES / "maps").glob("*.npz")):
        cohort, prior = path.stem.split("__")
        d = np.load(path)
        results[(cohort, prior)] = (InferenceResult(d["positive"], d["negative"], method="ni_tfnbs",
                                                    stat_positive=d["stat_positive"], stat_negative=d["stat_negative"]),
                                    len(set(d["labels"])) if d["labels"].min() >= 0 else 0)

    def draw(ax, inf, tail, title, labels="gyrus"):
        n = int(np.nansum(np.asarray(inf.positive if tail == "positive" else inf.negative) < 0.05) // 2)
        plot_connectogram(inf, rois, tail=tail, level="gyrus", networks=nets, labels=labels, ax=ax,
                          title=f"{title}\n{TAILS[tail]}: {n} connections")

    for (cohort, prior), (inf, k) in results.items():
        fig, axes = plt.subplots(1, 2, figsize=(18, 9.2))
        for ax, tail in zip(axes, TAILS):
            draw(ax, inf, tail, f"{COHORT_NAMES[cohort]}: {prior_label(prior, k)}")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.04)
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{cohort}__{prior}.{ext}", dpi=140, bbox_inches="tight")
        plt.close(fig)

    for cohort in ("ihb", "china_ec1"):
        priors = [p for p in order if (cohort, p) in results]
        for tail in TAILS:
            cols = 4
            rows = int(np.ceil(len(priors) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 5.6))
            for ax in np.ravel(axes)[len(priors):]:
                ax.axis("off")
            for ax, prior in zip(np.ravel(axes), priors):
                inf, k = results[(cohort, prior)]
                draw(ax, inf, tail, prior_label(prior, k), labels="none")
                for leg in ax.get_legend(), *[a for a in ax.artists]:
                    if leg is not None and hasattr(leg, "set_visible"):
                        leg.set_visible(False)
                for t in list(ax.texts):                    # footers would shrink the panels
                    if t.get_text().startswith("Each of"):
                        t.remove()
            fig.suptitle(f"{COHORT_NAMES[cohort]}, {TAILS[tail]}: NI-TFNBS with SC blocks across resolutions "
                         "(chords coloured by Yeo network)", fontsize=14)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.05, hspace=0.15)
            fig.savefig(OUT / f"overview_{cohort}_{tail}.png", dpi=110)
            plt.close(fig)
    print(f"{len(results)} connectograms + overviews -> {OUT}")


if __name__ == "__main__":
    main()
