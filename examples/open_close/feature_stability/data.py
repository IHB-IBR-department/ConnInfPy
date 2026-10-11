"""Individual edge inputs and post-fit reporting memberships."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from examples.open_close._shared.data import OpenCloseDataset


@dataclass
class FeatureSpace:
    name: str
    labels: list[str]
    edge_groups: np.ndarray | None
    counts: np.ndarray
    projection: csr_matrix | None

    def transform(self, edges: np.ndarray) -> np.ndarray:
        return edges if self.projection is None else np.asarray(edges @ self.projection)


@dataclass
class PreparedData:
    edges: np.ndarray
    participants: pd.DataFrame
    scans: pd.DataFrame
    rois: pd.DataFrame
    edge_definitions: pd.DataFrame
    spaces: dict[str, FeatureSpace]


def block_space(name: str, roi_groups: np.ndarray, rows: np.ndarray,
                cols: np.ndarray) -> FeatureSpace:
    pairs = [tuple(sorted((str(roi_groups[i]), str(roi_groups[j])))) for i, j in zip(rows, cols)]
    unique = sorted(set(pairs))  # Only nonempty blocks; diagonal self-edges are excluded.
    lookup = {pair: i for i, pair in enumerate(unique)}
    groups = np.asarray([lookup[pair] for pair in pairs])
    counts = np.bincount(groups, minlength=len(unique))
    projection = csr_matrix((1.0 / counts[groups], (np.arange(len(groups)), groups)),
                            shape=(len(groups), len(unique)))
    return FeatureSpace(name, [f"{a} | {b}" for a, b in unique], groups, counts, projection)


def feature_spaces(atlas: pd.DataFrame, annotations: pd.DataFrame,
                   anatomical_level: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, FeatureSpace]]:
    """Align by ROI name, then use one anatomical eligibility mask for every space."""
    if annotations.roi.duplicated().any() or atlas.name.duplicated().any():
        raise ValueError("ROI names must be unique.")
    if set(annotations.roi) != set(atlas.name):
        raise ValueError("Annotations must cover exactly the source atlas ROI names.")
    aligned = annotations.set_index("roi").loc[atlas.name].reset_index()
    aligned["source_index"] = np.arange(len(atlas))
    aligned["atlas_id"] = atlas.id.to_numpy()
    aligned["functional"] = atlas.network.to_numpy()  # Source Yeo labels are authoritative.
    eligible_levels = ("subregion", "gyrus", "lobe") if anatomical_level == "lobe" else ("subregion", "gyrus")
    eligible = aligned.level.isin(eligible_levels)
    aligned["eligible"] = eligible
    if aligned.loc[eligible, [anatomical_level, "hemisphere"]].isna().any().any():
        raise ValueError("Anatomically eligible ROIs need a group and hemisphere.")
    if not aligned.loc[eligible, "hemisphere"].isin(["L", "R"]).all():
        raise ValueError("Anatomically eligible ROIs must have L/R hemisphere labels.")
    rois = aligned.loc[eligible].copy().reset_index(drop=True)
    if len(rois) < 2:
        raise ValueError("At least two anatomically assigned ROIs are required.")
    rois["anatomical"] = rois[anatomical_level] + "_" + rois.hemisphere
    rows, cols = np.triu_indices(len(rois), k=1)
    labels = [f"{rois.roi[i]} | {rois.roi[j]}" for i, j in zip(rows, cols)]
    spaces = {
        "edges": FeatureSpace("edges", labels, None, np.ones(len(labels), dtype=int), None),
        "functional": block_space("functional", rois.functional.to_numpy(), rows, cols),
        "anatomical": block_space("anatomical", rois.anatomical.to_numpy(), rows, cols),
    }
    edges = pd.DataFrame({
        "edge_index": np.arange(len(rows)), "roi_i": rois.roi.to_numpy()[rows],
        "roi_j": rois.roi.to_numpy()[cols], "atlas_id_i": rois.atlas_id.to_numpy()[rows],
        "atlas_id_j": rois.atlas_id.to_numpy()[cols],
    })
    for name in ("functional", "anatomical"):
        edges[name + "_feature_index"] = spaces[name].edge_groups
    return aligned, edges, spaces


def load_data(data_dir: Path, annotations_path: Path, anatomical_level: str,
              china_close_run: int) -> PreparedData:
    datasets = {site: OpenCloseDataset.load(site, data_dir=data_dir) for site in ("ihb", "china")}
    atlas = datasets["ihb"].atlas_df
    if not atlas.equals(datasets["china"].atlas_df):
        raise ValueError("Site atlases are not aligned.")
    annotated, edge_definitions, spaces = feature_spaces(atlas, pd.read_csv(annotations_path), anatomical_level)
    keep = annotated.loc[annotated.eligible, "source_index"].to_numpy()
    rows, cols = np.triu_indices(len(keep), k=1)
    participants, scans, all_edges = [], [], []
    for site, ds in datasets.items():
        # Same complete China participants for either run, including run0 sensitivity.
        subject_indices = np.flatnonzero(ds.retest_subject_mask())
        if len(set(ds.subject_ids)) != len(ds.subject_ids):
            raise ValueError(f"Duplicate subject IDs at {site}.")
        close_ts = ds.close_ts_run1 if site == "china" and china_close_run == 1 else ds.close_ts
        open_z = ds._ts_to_fisher_z(ds.open_ts[subject_indices][:, :, keep])
        close_z = ds._ts_to_fisher_z(close_ts[subject_indices][:, :, keep])
        for local, source_index in enumerate(subject_indices):
            p = len(participants)
            subject_id = ds.subject_ids[source_index]
            participant_id = f"{site}:{subject_id}"
            participants.append({"participant_index": p, "participant_id": participant_id,
                                 "subject_id": subject_id, "site": site, "source_index": int(source_index)})
            for state, z in ((0, open_z[local]), (1, close_z[local])):
                scans.append({"scan_index": len(scans), "participant_index": p,
                              "participant_id": participant_id, "subject_id": subject_id,
                              "site": site, "state": state,
                              "scan_id": participant_id + (":EO" if state == 0 else ":EC"),
                              "close_run": (china_close_run if site == "china" else 0) if state else -1})
                all_edges.append(z[rows, cols])
    edges = np.asarray(all_edges)
    if not np.isfinite(edges).all():
        raise ValueError("Nonfinite FC features; inspect time series before fitting.")
    return PreparedData(edges, pd.DataFrame(participants), pd.DataFrame(scans), annotated,
                        edge_definitions, spaces)
