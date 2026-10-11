"""Matched atlas branches for the Open-Close contrast-stability comparison.

Plan: vault note ``openclose_atlas_contrast_stability_plan_2026-10-10``. Everything
here is fixed before any contrast is examined:

* the common cortical support (both label images share the FSL MNI152 2 mm grid):
  BNA dropout regions and BNA regions mostly inside the 18 Schaefer parcels removed
  for IHB coverage are excluded; Schaefer parcels mostly inside BNA dropout regions
  are excluded;
* the branches: ``schaefer`` (Schaefer-200 parcels), ``bna`` (native BNA-246 regions),
  ``bna_gyrus`` (BNA region signals averaged within gyrus x hemisphere, weighted by
  region volume: the same data at a coarser resolution), all on the common support,
  plus the descriptive full sets ``schaefer_full`` (182) and ``bna_full`` (242);
* memberships of every parcel in the common reporting units (BNA lobe x hemisphere,
  gyrus x hemisphere, a Yeo-7 reference, each atlas's own network labels);
* parcel volumes, and a provenance check that the two extraction products come
  from the same scans.
"""
from __future__ import annotations


from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

from conninfpy.interpret.annotate import _unit_weights, annotate_rois  # noqa: E402
from conninfpy.atlas import get_bna_246_nifti_path  # noqa: E402
from conninfpy.atlas.brainnetome import BrainnetomeReference  # noqa: E402
from examples.open_close._shared.data import BNA_DROPOUT_IDS, MISSED_ROI_IDS, OpenCloseDataset  # noqa: E402

SCHAEFER_IMG = Path.home() / "nilearn_data" / "schaefer_2018" / "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
@dataclass(frozen=True)
class StudyInputs:
    data_dir: Path = ROOT / "datasets" / "open_close"
    reference_dir: Path = ROOT / "datasets" / "atlases"
    schaefer_image: Path = SCHAEFER_IMG


DEFAULT_INPUTS = StudyInputs()

# cohort -> (experiment, close run, restrict to the 47 China participants with both EC runs)
COHORTS = {"ihb": ("ihb", 0, False), "china_ec1": ("china", 1, True), "china_ec0": ("china", 0, True)}
MATCHED = ("schaefer", "bna", "bna_gyrus")
FULL = ("schaefer_full", "bna_full")
LEVELS = ("lobe", "gyrus", "yeo_ref", "yeo_own")
VOXEL_MM3 = 8.0                                  # 2 mm grid
_NET = {"Default Mode": "Default", "Frontoparietal Control": "Frontoparietal"}


def _net(name: str) -> str:
    return _NET.get(str(name), str(name))


@dataclass
class Branch:
    name: str
    cohort: str
    atlas: str
    eo: np.ndarray                               # (S, T, N) time series
    ec: np.ndarray
    subjects: List[str]
    parcels: pd.DataFrame                        # id, name, volume_mm3 (+ members for bna_gyrus)
    members: Dict[str, Tuple[np.ndarray, List[str]]] = field(default_factory=dict)  # level -> (M (N, U), units)


# ----------------------------------------------------------------------
# Reference, label images and common support
# ----------------------------------------------------------------------
def reference(inputs: StudyInputs = DEFAULT_INPUTS) -> BrainnetomeReference:
    return BrainnetomeReference.load(data_dir=inputs.reference_dir, profiles=False)


def label_images(inputs: StudyInputs = DEFAULT_INPUTS) -> Tuple[np.ndarray, np.ndarray]:
    s, b = nib.load(str(inputs.schaefer_image)), nib.load(str(get_bna_246_nifti_path(2)))
    if s.shape != b.shape or not np.allclose(s.affine, b.affine):
        raise RuntimeError("Schaefer and BNA 2 mm label images are not on the same grid.")
    return (np.rint(np.asarray(s.dataobj)).astype(int), np.rint(np.asarray(b.dataobj)).astype(int))


def common_support(regions: pd.DataFrame, inputs: StudyInputs = DEFAULT_INPUTS) -> Tuple[Dict[str, List[int]], pd.DataFrame]:
    """Fixed common cortical support and the record of every decision."""
    s_lab, b_lab = label_images(inputs)
    sch_kept = sorted(set(range(1, 201)) - set(MISSED_ROI_IDS))
    cortex = [int(r) for r in regions.index[regions["structure"] == "cortex"]]
    rows, bna_keep, sch_keep = [], [], []
    for r in cortex:
        v = s_lab[b_lab == r]
        labelled = v[v > 0]
        share_removed = float(np.isin(labelled, MISSED_ROI_IDS).mean()) if labelled.size else np.nan
        if r in BNA_DROPOUT_IDS:
            reason = "BNA dropout (no signal without GSR)"
        elif not labelled.size:
            reason = "no Schaefer-labelled voxels"
        elif share_removed > 0.5:
            reason = "mostly inside Schaefer parcels removed for IHB coverage"
        else:
            reason = ""
            bna_keep.append(r)
        rows.append({"atlas": "bna", "id": r, "name": regions.at[r, "name"], "n_voxels_2mm": int((b_lab == r).sum()),
                     "share_in_other_excluded": share_removed, "excluded": reason})
    for p in sch_kept:
        v = b_lab[s_lab == p]
        labelled = v[np.isin(v, cortex)]
        share_drop = float(np.isin(labelled, BNA_DROPOUT_IDS).mean()) if labelled.size else np.nan
        reason = "mostly inside BNA dropout regions" if (labelled.size and share_drop > 0.5) else ""
        if not reason:
            sch_keep.append(p)
        rows.append({"atlas": "schaefer", "id": p, "name": "", "n_voxels_2mm": int((s_lab == p).sum()),
                     "share_in_other_excluded": share_drop, "excluded": reason})
    return {"bna": bna_keep, "schaefer": sch_keep}, pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Memberships in the common reporting units
# ----------------------------------------------------------------------
def canonical_units(regions: pd.DataFrame, level: str) -> List[str]:
    """The fixed, sorted list of cortical reporting units at ``level`` (identical for every atlas)."""
    c = regions[regions["structure"] == "cortex"]
    if level in ("lobe", "gyrus"):
        return sorted({f"{a}_{h}" for a, h in zip(c[level], c["hemisphere"])})
    return sorted({str(x) for x in c["yeo7"]})


def _onehot(labels: List[str], units: List[str]) -> Tuple[np.ndarray, List[str]]:
    M = np.zeros((len(labels), len(units)))
    M[np.arange(len(labels)), [units.index(x) for x in labels]] = 1.0
    return M, units


def _cortical_units(M: np.ndarray, units: List[str], regions: pd.DataFrame, level: str):
    """Columns in the canonical cortical unit order; each parcel's membership renormalised to 1."""
    canon = canonical_units(regions, level)
    out = np.zeros((M.shape[0], len(canon)))
    for k, u in enumerate(units):
        if u in canon:
            out[:, canon.index(u)] = M[:, k]
    tot = out.sum(axis=1, keepdims=True)
    return np.divide(out, tot, out=np.zeros_like(out), where=tot > 0), canon


def _yeo_from_regions(W: np.ndarray, regions: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """(N, 246) region weights -> (N, 7) Yeo-7 crosswalk shares over cortical regions."""
    net = [str(x) if s == "cortex" else "" for x, s in zip(regions["yeo7"], regions["structure"])]
    units = canonical_units(regions, "yeo_ref")
    P = np.zeros((len(net), len(units)))
    for k, x in enumerate(net):
        if x:
            P[k, units.index(x)] = 1.0
    M = W @ P
    tot = M.sum(axis=1, keepdims=True)
    return np.divide(M, tot, out=np.zeros_like(M), where=tot > 0), units


def schaefer_annotation(ref: BrainnetomeReference, ds: OpenCloseDataset, inputs: StudyInputs = DEFAULT_INPUTS):
    """BNA annotation of the 182 Schaefer parcels (parcel support, as in the interpretation example)."""
    return annotate_rois(reference=ref, names=list(ds.atlas_df["name"]), support="parcel", space="MNI152",
                         parcel_image=inputs.schaefer_image, parcel_labels=[int(i) for i in ds.atlas_df["id"]])


def schaefer_members(rois, ids: List[int], all_ids: List[int], networks: List[str], regions: pd.DataFrame):
    rows = [all_ids.index(i) for i in ids]
    out = {}
    for level in ("lobe", "gyrus"):
        W, units = _unit_weights(rois, level, "weighted")
        out[level] = _cortical_units(W[rows], units, regions, level)
    out["yeo_ref"] = _yeo_from_regions(np.nan_to_num(rois.labelled_weights(), nan=0.0)[rows], regions)
    out["yeo_own"] = _onehot([_net(networks[r]) for r in rows], canonical_units(regions, "yeo_own"))
    return out


def bna_members(ids: List[int], regions: pd.DataFrame):
    r = regions.loc[ids]
    lab = lambda col: [f"{a}_{h}" for a, h in zip(r[col], r["hemisphere"])]   # noqa: E731
    yeo = _onehot([str(x) for x in r["yeo7"]], canonical_units(regions, "yeo_ref"))
    return {"lobe": _onehot(lab("lobe"), canonical_units(regions, "lobe")),
            "gyrus": _onehot(lab("gyrus"), canonical_units(regions, "gyrus")), "yeo_ref": yeo, "yeo_own": yeo}


def bna_gyrus_members(groups: List[List[int]], names: List[str], regions: pd.DataFrame, volumes: Dict[int, float]):
    W = np.zeros((len(groups), len(regions)))
    for g, members in enumerate(groups):
        for r in members:
            W[g, regions.index.get_loc(r)] = volumes[r]
        W[g] /= W[g].sum()
    lobes = [regions.at[m[0], "lobe"] + "_" + regions.at[m[0], "hemisphere"] for m in groups]
    yeo = _yeo_from_regions(W, regions)
    return {"lobe": _onehot(lobes, canonical_units(regions, "lobe")),
            "gyrus": _onehot(names, canonical_units(regions, "gyrus")), "yeo_ref": yeo, "yeo_own": yeo}


# ----------------------------------------------------------------------
# Branches
# ----------------------------------------------------------------------
def _scans(ds: OpenCloseDataset, run: int, matched: bool):
    keep = ds.retest_subject_mask() if matched else np.ones(ds.n_subjects, bool)
    ec = ds.close_ts if run == 0 else ds.close_ts_run1
    return ds.open_ts[keep], ec[keep], [s for s, k in zip(ds.subject_ids, keep) if k]


def build_branches(cohort: str, support: Dict[str, List[int]], ref: BrainnetomeReference,
                   schaefer_rois=None, inputs: StudyInputs = DEFAULT_INPUTS) -> Dict[str, Branch]:
    experiment, run, matched = COHORTS[cohort]
    regions = ref.regions
    s_lab, b_lab = label_images(inputs)
    s_vox = np.bincount(s_lab.ravel(), minlength=201)
    b_vox = np.bincount(b_lab.ravel(), minlength=247)

    sch = OpenCloseDataset.load(experiment, data_dir=inputs.data_dir)
    bna = OpenCloseDataset.load(experiment, data_dir=inputs.data_dir, atlas="brainnetome")
    if sch.subject_ids != bna.subject_ids:
        raise RuntimeError(f"{experiment}: Schaefer and BNA subject orders differ.")
    rois = schaefer_rois if schaefer_rois is not None else schaefer_annotation(ref, sch, inputs)
    s_ids, b_ids = [int(i) for i in sch.atlas_df["id"]], [int(i) for i in bna.atlas_df["id"]]
    s_net = list(sch.atlas_df["network"])

    def make(name, atlas, ds, ids, all_ids, members, parcels, combine=None):
        eo, ec, subjects = _scans(ds, run, matched)
        cols = [all_ids.index(i) for i in ids]
        eo, ec = eo[..., cols], ec[..., cols]
        if combine is not None:
            eo, ec = combine(eo), combine(ec)
        return Branch(name, cohort, atlas, eo, ec, subjects, parcels, members)

    out = {}
    s_par = lambda ids: pd.DataFrame({"id": ids, "name": [sch.atlas_df["name"][s_ids.index(i)] for i in ids],   # noqa: E731
                                      "volume_mm3": [s_vox[i] * VOXEL_MM3 for i in ids]})
    b_par = lambda ids: pd.DataFrame({"id": ids, "name": list(regions.loc[ids, "name"]),   # noqa: E731
                                      "volume_mm3": [b_vox[i] * VOXEL_MM3 for i in ids]})
    out["schaefer"] = make("schaefer", "schaefer", sch, support["schaefer"], s_ids,
                           schaefer_members(rois, support["schaefer"], s_ids, s_net, regions), s_par(support["schaefer"]))
    out["schaefer_full"] = make("schaefer_full", "schaefer", sch, s_ids, s_ids,
                                schaefer_members(rois, s_ids, s_ids, s_net, regions), s_par(s_ids))
    out["bna"] = make("bna", "bna", bna, support["bna"], b_ids, bna_members(support["bna"], regions), b_par(support["bna"]))
    out["bna_full"] = make("bna_full", "bna", bna, b_ids, b_ids, {}, b_par(b_ids))

    # resolution control: volume-weighted mean of the kept BNA region signals per gyrus x hemisphere
    r = regions.loc[support["bna"]]
    keys = [f"{g}_{h}" for g, h in zip(r["gyrus"], r["hemisphere"])]
    names = list(dict.fromkeys(keys))
    groups = [[i for i, k in zip(support["bna"], keys) if k == n] for n in names]
    volumes = {i: float(b_vox[i]) for i in support["bna"]}
    A = np.zeros((len(support["bna"]), len(groups)))
    for g, members in enumerate(groups):
        for i in members:
            A[support["bna"].index(i), g] = volumes[i]
    A /= A.sum(axis=0, keepdims=True)
    par = pd.DataFrame({"id": names, "name": names, "volume_mm3": [sum(volumes[i] for i in m) * VOXEL_MM3 for m in groups],
                        "members": [" ".join(regions.loc[m, "name"]) for m in groups]})
    out["bna_gyrus"] = make("bna_gyrus", "bna_gyrus", bna, support["bna"], b_ids,
                            bna_gyrus_members(groups, names, regions, volumes), par, combine=lambda x: x @ A)
    return out


# ----------------------------------------------------------------------
# Provenance: do both extraction products come from the same scans?
# ----------------------------------------------------------------------
def provenance(experiment: str, support: Dict[str, List[int]], inputs: StudyInputs = DEFAULT_INPUTS) -> pd.DataFrame:
    """Per scan: median correlation of each kept BNA region with its most-overlapping Schaefer parcel."""
    s_lab, b_lab = label_images(inputs)
    sch = OpenCloseDataset.load(experiment, data_dir=inputs.data_dir)
    bna = OpenCloseDataset.load(experiment, data_dir=inputs.data_dir, atlas="brainnetome")
    s_ids, b_ids = [int(i) for i in sch.atlas_df["id"]], [int(i) for i in bna.atlas_df["id"]]
    pairs = []
    for r in support["bna"]:
        v = s_lab[b_lab == r]
        v = v[np.isin(v, support["schaefer"])]
        if v.size:
            pairs.append((b_ids.index(r), s_ids.index(int(np.bincount(v).argmax()))))
    bi, si = np.array(pairs).T
    scans = [("eo", sch.open_ts, bna.open_ts), ("ec_run0", sch.close_ts, bna.close_ts)]
    if sch.close_ts_run1 is not None:
        scans.append(("ec_run1", sch.close_ts_run1, bna.close_ts_run1))
    rows = []
    for scan, S, B in scans:
        for k, sub in enumerate(sch.subject_ids):
            x, y = B[k][:, bi], S[k][:, si]
            x = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), np.nan)
            y = (y - y.mean(0)) / np.where(y.std(0) > 0, y.std(0), np.nan)
            r = np.nanmean(x * y, axis=0)
            rows.append({"experiment": experiment, "scan": scan, "subject": sub, "median_r": float(np.nanmedian(r)),
                         "p10_r": float(np.nanpercentile(r, 10)), "n_pairs": int(np.isfinite(r).sum()),
                         "zero_trs_schaefer": int((np.abs(S[k]).sum(1) == 0).sum())})
    return pd.DataFrame(rows)
