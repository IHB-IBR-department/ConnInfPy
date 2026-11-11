"""Build ``datasets/demo/NBS_schizophrenia/`` from the NBS v1.2 example data.

Input: the ``SchizophreniaExample/`` folder distributed with the MATLAB NBS
toolbox v1.2 (Zalesky, Fornito & Bullmore 2010, NeuroImage 53:1197-1207;
www.nitrc.org/projects/nbs):
  matrices/subject01.txt ... subject27.txt   74 x 74 wavelet correlation matrices
  designMatrix.txt                           27 x 2 group indicators
  COG.txt                                    74 x 3 node centres of gravity (MNI, mm)
  nodeLabels.txt                             74 AAL region names

NBS's help text (help/examples.txt) gives the design: 12 people with
schizophrenia (design column 1, subjects 01-12) and 15 healthy controls
(column 2, subjects 13-27). The values are copied unchanged: raw
correlation coefficients with ones on the diagonal, no Fisher z.

Two changes are made to the node table, both checked before writing:

* ``COG.txt`` stores x in radiological orientation: every ``*_L`` node has
  x > 0 and every ``*_R`` node x < 0. With x negated the centres agree with
  the AAL centroids of the same hemisphere in x, y and z (e.g. Precentral_R
  (42, -10, 50) against AAL (41.4, -8.2, 52.1)), so the labels are right and
  x is flipped to MNI (RAS, left = negative).
* The eight subcortical nodes are all labelled ``Subcortical``. Their order
  and centres are those of AAL 71-78 (caudate, putamen, pallidum, thalamus;
  left then right), so they are named accordingly. The NBS label is kept in
  the ``nbs_label`` column.

Outputs (written to ``--out-dir``):
  nbs_schizophrenia.npy   (27, 74, 74) float64
  pheno.csv               subject_id, group (SZ | HC)
  aal74_atlas.csv         name, network (empty), x, y, z (MNI), hemisphere, nbs_label

The example is distributed under the GNU GPL v3 or later, like the rest of
NBS; the output folder carries that licence (LICENSE-GPL-3.0.txt).

Usage:
    python scripts/build_nbs_schizophrenia_demo.py \
        --nbs-dir ../NBS1.2 [--out-dir datasets/demo/NBS_schizophrenia]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
N_NODES = 74
N_SZ, N_HC = 12, 15
# AAL 71-78, in AAL order; NBS labels all eight "Subcortical".
SUBCORTICAL_AAL = (
    "Caudate_L", "Caudate_R", "Putamen_L", "Putamen_R",
    "Pallidum_L", "Pallidum_R", "Thalamus_L", "Thalamus_R",
)


def _sha256(path: Path) -> str:
    """SHA-256 of a file, as hex."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(nbs_dir: Path, out_dir: Path) -> None:
    """Convert the NBS example into the ConnInfPy demo files."""
    src = nbs_dir / "SchizophreniaExample"
    files = sorted((src / "matrices").glob("subject*.txt"))
    if len(files) != N_SZ + N_HC:
        raise ValueError(f"{src}/matrices: expected {N_SZ + N_HC} subject files, found {len(files)}.")
    data = np.stack([np.loadtxt(f) for f in files]).astype(np.float64)
    if data.shape[1:] != (N_NODES, N_NODES):
        raise ValueError(f"Matrices have shape {data.shape[1:]}, expected ({N_NODES}, {N_NODES}).")
    if not np.allclose(data, data.transpose(0, 2, 1)):
        raise ValueError("Matrices are not symmetric.")

    design = np.loadtxt(src / "designMatrix.txt")
    expected = np.vstack([np.tile([1.0, 0.0], (N_SZ, 1)), np.tile([0.0, 1.0], (N_HC, 1))])
    if not np.array_equal(design, expected):
        raise ValueError("designMatrix.txt is not 12 rows [1 0] followed by 15 rows [0 1].")

    coords = np.loadtxt(src / "COG.txt")
    nbs_labels = (src / "nodeLabels.txt").read_text().split()
    if coords.shape != (N_NODES, 3) or len(nbs_labels) != N_NODES:
        raise ValueError(f"COG.txt {coords.shape} / nodeLabels.txt {len(nbs_labels)} do not match {N_NODES} nodes.")

    sub_idx = [i for i, lab in enumerate(nbs_labels) if lab == "Subcortical"]
    if len(sub_idx) != len(SUBCORTICAL_AAL):
        raise ValueError(f"Expected {len(SUBCORTICAL_AAL)} 'Subcortical' nodes, found {len(sub_idx)}.")
    labels = list(nbs_labels)
    for i, name in zip(sub_idx, SUBCORTICAL_AAL):
        labels[i] = name
    if len(set(labels)) != N_NODES:
        raise ValueError("Node names are not unique after naming the subcortical nodes.")
    hemisphere = [lab.rsplit("_", 1)[-1] for lab in labels]

    # Radiological x -> MNI x. Check the flip first: every left node must
    # have x > 0 in COG.txt and every right node x < 0.
    x_cog = coords[:, 0]
    is_left = np.array([h == "L" for h in hemisphere])
    if not (np.all(x_cog[is_left] > 0) and np.all(x_cog[~is_left] < 0)):
        raise ValueError("COG.txt x signs do not follow the radiological pattern; refusing to flip x.")
    coords = coords * np.array([-1.0, 1.0, 1.0])

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "nbs_schizophrenia.npy", data)
    pd.DataFrame({
        "subject_id": [f.stem for f in files],
        "group": ["SZ"] * N_SZ + ["HC"] * N_HC,
    }).to_csv(out_dir / "pheno.csv", index=False)
    pd.DataFrame({
        "name": labels,
        "network": [""] * N_NODES,
        "x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2],
        "hemisphere": hemisphere,
        "nbs_label": nbs_labels,
    }).to_csv(out_dir / "aal74_atlas.csv", index=False)

    for name in ("nbs_schizophrenia.npy", "pheno.csv", "aal74_atlas.csv"):
        path = out_dir / name
        print(f"{name}: {path.stat().st_size} bytes, sha256 {_sha256(path)}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--nbs-dir", type=Path, required=True, help="NBS v1.2 folder that contains SchizophreniaExample/.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "datasets" / "demo" / "NBS_schizophrenia")
    args = parser.parse_args()
    build(args.nbs_dir.expanduser().resolve(), args.out_dir)


if __name__ == "__main__":
    main()
