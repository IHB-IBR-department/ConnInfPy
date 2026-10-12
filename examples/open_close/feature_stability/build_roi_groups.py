"""Regenerate the fixed anatomical crosswalk without inference or state labels."""
from __future__ import annotations


from pathlib import Path as _Path
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

from conninfpy.interpret.annotate import annotate_rois  # noqa: E402
from conninfpy.atlas.resources import brainnetome_data_dir, bundled_path
from conninfpy.atlas.brainnetome import BrainnetomeReference  # noqa: E402
from examples._shared.paths import sha256  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parcel-image", type=Path, default=Path.home() / "nilearn_data/schaefer_2018/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")
    parser.add_argument("--reference-dir", type=Path, default=brainnetome_data_dir())
    parser.add_argument("--labels", type=Path, default=ROOT / "datasets/open_close/schaefer_labels.csv")
    parser.add_argument("--output", type=Path, default=HERE / "roi_groups.csv")
    args = parser.parse_args()
    labels = pd.read_csv(args.labels)
    reference = BrainnetomeReference.load(data_dir=args.reference_dir, profiles=False)
    annotation = annotate_rois(reference=reference, names=labels.name.tolist(), support="parcel",
                               space="MNI152", parcel_image=args.parcel_image,
                               parcel_labels=labels.id.astype(int).tolist())
    columns = ["roi", "level", "gyrus", "lobe", "hemisphere", "coverage", "flags"]
    annotation.table[columns].to_csv(args.output, index=False)
    inputs = [args.parcel_image, args.labels, bundled_path("bna246_hierarchy.csv"),
              ROOT / "conninfpy/interpret/annotate.py", ROOT / "conninfpy/atlas/brainnetome.py"]
    provenance = {
        "method": "Schaefer parcel overlap with Brainnetome; finest assignment holding >=50%",
        "space": "generic MNI152, approximate FSL MNI152 correspondence",
        "target_labels_used": False,
        "input_sha256": {p.name: sha256(p) for p in inputs},
        "label_image": reference.provenance["label_image"],
        "label_image_sha256": reference.provenance["label_image_sha256"],
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
