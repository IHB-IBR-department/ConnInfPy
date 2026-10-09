"""Loaders for the lab's own datasets: Open-Close (IHB and Beijing), Stress,
Zerssen and the China close-close retest.

They encode one lab's file layouts, so they live with the app rather than in
the ``conninfpy`` package. Importing this module registers them with
:func:`conninfpy.loaders.register_loader`, so manifests (e.g.
``datasets/open_close.yaml``) can name them in ``loader:``.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from conninfpy.atlas import AtlasInfo
from conninfpy.loaders import register_loader
from conninfpy.loaders.base import BaseDataLoader, DatasetPreview, LoadedDataset
from conninfpy.loaders.builtins import (
    CSVDirectoryLoader,
    NiftiDirectoryLoader,
    _load_atlas_if_present,
)


class OpenCloseLoader(BaseDataLoader):
    """Loader for the open_close dataset containing paired timeseries arrays for different conditions."""
    name = "OpenCloseLoader"

    def __init__(self, open_path: str, close_path: str, subject_list_path: str | None = None, atlas: str | None = None, drop_missing_rois: bool = False):
        self.open_path = open_path
        self.close_path = close_path
        self.subject_list_path = subject_list_path
        self.atlas = atlas
        self.drop_missing_rois = drop_missing_rois

    def _get_kept_indices(self) -> np.ndarray | None:
        """ROI indices surviving the missing-ROI drop."""
        if self.drop_missing_rois and self.atlas and os.path.exists(self.atlas):
            try:
                df = pd.read_csv(self.atlas)
                if "id" in df.columns:
                    return df["id"].values - 1
            except Exception:
                pass
        return None

    def preview(self) -> DatasetPreview:
        """Cheap preview of the dataset without a full load."""
        if not os.path.exists(self.open_path) or not os.path.exists(self.close_path):
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["Open or close file path not found."])
            
        try:
            d_open = np.load(self.open_path)
            shape = d_open.shape # (subjects, timepoints, rois)
            n_sub, n_tp, n_rois = shape
            
            kept_idx = self._get_kept_indices()
            if kept_idx is not None:
                n_rois = len(kept_idx)
            
            subject_ids = []
            if self.subject_list_path and os.path.exists(self.subject_list_path):
                with open(self.subject_list_path, 'r') as f:
                    subject_ids = [line.strip() for line in f if line.strip()]
            else:
                subject_ids = [f"sub_{i:02d}" for i in range(n_sub)]
                
            return DatasetPreview(
                n_subjects=n_sub,
                n_observations=n_sub * 2,
                n_rois=n_rois,
                n_timepoints=n_tp,
                data_kind_guess="timeseries",
                atlas_guess=os.path.basename(self.atlas) if self.atlas else f"Schaefer-{n_rois}",
                conditions=["open", "close"],
                subject_ids_sample=subject_ids[:5],
                file_count=2,
                file_sizes_mb=(os.path.getsize(self.open_path) + os.path.getsize(self.close_path)) / (1024 * 1024)
            )
        except Exception as e:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 2, None, [f"Error previewing open_close: {e}"])

    def load(self) -> LoadedDataset:
        """Load the dataset into a :class:`LoadedDataset`."""
        open_data = np.load(self.open_path)
        close_data = np.load(self.close_path)
        
        if close_data.ndim == 4:
            close_data = close_data[..., 0]
            
        n_sub, n_tp, n_rois = open_data.shape
        
        kept_idx = self._get_kept_indices()
        if kept_idx is not None:
            open_data = open_data[:, :, kept_idx]
            close_data = close_data[:, :, kept_idx]
            n_rois = len(kept_idx)
            
        subject_ids = []
        if self.subject_list_path and os.path.exists(self.subject_list_path):
            with open(self.subject_list_path, 'r') as f:
                subject_ids = [line.strip() for line in f if line.strip()]
        else:
            subject_ids = [f"sub_{i:02d}" for i in range(n_sub)]
            
        if len(subject_ids) != n_sub:
            # Fallback if subject list length mismatch
            subject_ids = [f"sub_{i:02d}" for i in range(n_sub)]

        # Convert both to connectivity first
        open_corr = np.array([np.corrcoef(ts.T) for ts in open_data])
        close_corr = np.array([np.corrcoef(ts.T) for ts in close_data])
        
        # Combine into long format
        data = np.vstack([open_corr, close_corr]) # shape: (n_sub * 2, n_rois, n_rois)
        
        pheno = pd.DataFrame({
            "subject_id": subject_ids * 2,
            "condition": ["open"] * n_sub + ["close"] * n_sub
        })
        
        # Load atlas
        atlas = _load_atlas_if_present(self.atlas)
        
        return LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind="correlation",
            subject_ids=subject_ids * 2,
            conditions=["open", "close"],
            condition_column="condition",
            atlas=atlas,
            provenance={"open_path": self.open_path, "close_path": self.close_path}
        )


class MultiSiteOpenCloseLoader(BaseDataLoader):
    """Combine paired Open/Close cohorts while preserving their site labels.

    Every subject contributes one open and one close observation at the same
    acquisition site. Subject IDs are prefixed with that site to guarantee
    uniqueness when paired inference is configured from the combined table.
    """
    name = "MultiSiteOpenCloseLoader"

    def __init__(
        self,
        site_configs: dict[str, dict[str, str]],
        atlas: str | None = None,
        drop_missing_rois: bool = True,
    ):
        if not site_configs:
            raise ValueError("site_configs must contain at least one site.")
        self.site_configs = site_configs
        self.atlas = atlas
        self.drop_missing_rois = drop_missing_rois

    def _site_loaders(self) -> list[tuple[str, OpenCloseLoader]]:
        """One OpenCloseLoader per site, with site labels."""
        loaders = []
        for site, config in self.site_configs.items():
            loaders.append((
                site,
                OpenCloseLoader(
                    config["open_path"],
                    config["close_path"],
                    subject_list_path=config.get("subject_list_path"),
                    atlas=self.atlas,
                    drop_missing_rois=self.drop_missing_rois,
                ),
            ))
        return loaders

    def preview(self) -> DatasetPreview:
        """Cheap preview of the dataset without a full load."""
        previews = [(site, loader.preview()) for site, loader in self._site_loaders()]
        warnings = [
            f"{site}: {warning}"
            for site, preview in previews
            for warning in preview.warnings
        ]
        valid = [preview for _, preview in previews if preview.n_rois is not None]
        if not valid:
            return DatasetPreview(
                None, None, None, None, "unknown", None, None, [], 0, 0.0, warnings
            )

        roi_counts = {preview.n_rois for preview in valid}
        if len(roi_counts) != 1:
            warnings.append(f"Site ROI counts differ: {sorted(roi_counts)}.")

        timepoints = [preview.n_timepoints for preview in valid if preview.n_timepoints is not None]
        n_timepoints = (
            timepoints[0] if len(set(timepoints)) == 1
            else (min(timepoints), max(timepoints))
        )
        return DatasetPreview(
            n_subjects=sum(preview.n_subjects or 0 for preview in valid),
            n_observations=sum(preview.n_observations or 0 for preview in valid),
            n_rois=valid[0].n_rois,
            n_timepoints=n_timepoints,
            data_kind_guess="correlation",
            atlas_guess=valid[0].atlas_guess,
            conditions=["open", "close"],
            subject_ids_sample=[
                f"{site}:{subject_id}"
                for site, preview in previews
                for subject_id in preview.subject_ids_sample[:2]
            ][:5],
            file_count=sum(preview.file_count or 0 for preview in valid),
            file_sizes_mb=sum(preview.file_sizes_mb or 0.0 for preview in valid),
            warnings=warnings,
        )

    def load(self) -> LoadedDataset:
        """Load the dataset into a :class:`LoadedDataset`."""
        datasets = [(site, loader.load()) for site, loader in self._site_loaders()]
        roi_counts = {dataset.data.shape[1] for _, dataset in datasets}
        if len(roi_counts) != 1:
            raise ValueError(f"All sites must have the same ROI count; got {sorted(roi_counts)}.")

        data_parts = []
        pheno_parts = []
        subject_ids = []
        for site, dataset in datasets:
            site_pheno = dataset.pheno.copy()
            original_ids = site_pheno["subject_id"].astype(str)
            site_pheno["site"] = site
            site_pheno["subject_id"] = site + ":" + original_ids
            data_parts.append(dataset.data)
            pheno_parts.append(site_pheno)
            subject_ids.extend(site_pheno["subject_id"].tolist())

        return LoadedDataset(
            data=np.concatenate(data_parts, axis=0),
            pheno=pd.concat(pheno_parts, ignore_index=True),
            data_kind="correlation",
            subject_ids=subject_ids,
            conditions=["open", "close"],
            condition_column="condition",
            atlas=datasets[0][1].atlas,
            provenance={
                "sites": list(self.site_configs),
                "paired_within_site": True,
                "site_configs": self.site_configs,
            },
        )


class StressTimeseriesLoader(BaseDataLoader):
    """Loader for the stress dataset containing individual headerless CSV timeseries files."""
    name = "StressTimeseriesLoader"

    def __init__(self, dir_path: str):
        self.dir_path = dir_path

    def preview(self) -> DatasetPreview:
        # Thin wrapper over CSVDirectoryLoader, customized for Brainnetome-246
        """Cheap preview of the dataset without a full load."""
        loader = CSVDirectoryLoader(self.dir_path, header=None)
        prev = loader.preview()
        prev.atlas_guess = "Brainnetome-246"
        return prev

    def load(self) -> LoadedDataset:
        # Load timeseries using standard CSV loader
        """Load the dataset into a :class:`LoadedDataset`."""
        loader = CSVDirectoryLoader(self.dir_path, header=None)
        ds = loader.load()
        
        # Infer Brainnetome atlas
        try:
            ds.atlas = AtlasInfo.bna_246()
            ds.roi_labels = ds.atlas.labels
        except Exception:
            pass
            
        return ds


class ZerssenNiftiLoader(BaseDataLoader):
    """Loader for the Zerssen dataset containing preprocessed fMRI files separated by HC and Patients folders."""
    name = "ZerssenNiftiLoader"

    def __init__(self, subjects_dir: str, atlas_image_path: str, cache_dir: str | None = None):
        self.subjects_dir = subjects_dir
        self.atlas_image_path = atlas_image_path
        self.cache_dir = cache_dir

    def _get_folders_and_files(self) -> tuple[list[str], list[str]]:
        """Subject folders and their file paths."""
        hc_dir = os.path.join(self.subjects_dir, "HC")
        pat_dir = os.path.join(self.subjects_dir, "Patients")
        
        hc_files = sorted(glob.glob(os.path.join(hc_dir, "*.nii*")))
        pat_files = sorted(glob.glob(os.path.join(pat_dir, "*.nii*")))
        return hc_files, pat_files

    def preview(self) -> DatasetPreview:
        """Cheap preview of the dataset without a full load."""
        hc_files, pat_files = self._get_folders_and_files()
        all_files = hc_files + pat_files
        
        if not all_files:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["No HC or Patients functional NIfTI files found."])
            
        total_size = sum(os.path.getsize(f) for f in all_files) / (1024 * 1024)
        
        n_rois = None
        if os.path.isdir(self.atlas_image_path):
            n_rois = len(glob.glob(os.path.join(self.atlas_image_path, "*.nii*")))
            
        subject_ids_sample = [os.path.basename(f).split("_")[0] for f in all_files[:5]]
        
        return DatasetPreview(
            n_subjects=len(all_files),
            n_observations=len(all_files),
            n_rois=n_rois,
            n_timepoints=None,
            data_kind_guess="timeseries",
            atlas_guess="Brainnetome-246",
            conditions=["HC", "PAT"],
            subject_ids_sample=subject_ids_sample,
            file_count=len(all_files),
            file_sizes_mb=total_size
        )

    def load(self) -> LoadedDataset:
        """Load the dataset into a :class:`LoadedDataset`."""
        hc_files, pat_files = self._get_folders_and_files()
        
        # Load HC group using NiftiDirectoryLoader
        hc_loader = NiftiDirectoryLoader(os.path.join(self.subjects_dir, "HC"), self.atlas_image_path, self.cache_dir)
        hc_ds = hc_loader.load()
        
        # Load Patients group
        pat_loader = NiftiDirectoryLoader(os.path.join(self.subjects_dir, "Patients"), self.atlas_image_path, self.cache_dir)
        pat_ds = pat_loader.load()
        
        # Combine data
        data = np.vstack([hc_ds.data, pat_ds.data])
        
        # Generate Pheno DataFrame
        subject_ids = hc_ds.subject_ids + pat_ds.subject_ids
        groups = ["HC"] * len(hc_ds.subject_ids) + ["PAT"] * len(pat_ds.subject_ids)
        
        pheno = pd.DataFrame({
            "subject_id": subject_ids,
            "group": groups
        })
        atlas = None
        try:
            atlas = AtlasInfo.bna_246()
        except Exception:
            pass

        return LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind="timeseries",
            subject_ids=subject_ids,
            atlas=atlas,
            provenance={"subjects_dir": self.subjects_dir, "atlas_image_path": self.atlas_image_path}
        )


class CustomPreparedZerssenLoader(BaseDataLoader):
    """Loader for precomputed Zerssen Prepared set84 NPZ datasets."""
    name = "CustomPreparedZerssenLoader"
    
    def __init__(
        self,
        prepared_npz: str,
        set84_bnt_ids: str,
        atlas_metadata: str,
        matrix_key: str = "set84m_pearson_z",
        data_kind: str = "fisher_z",
    ):
        self.prepared_npz = prepared_npz
        self.set84_bnt_ids = set84_bnt_ids
        self.atlas_metadata = atlas_metadata
        self.matrix_key = matrix_key
        self.data_kind = data_kind

    @staticmethod
    def _available_connectivity_keys(archive) -> list[str]:
        """Return square subject-by-ROI-by-ROI arrays stored in the archive."""
        return [
            key for key in archive.files
            if archive[key].ndim == 3 and archive[key].shape[1] == archive[key].shape[2]
        ]

    def _selected_matrix(self, archive) -> tuple[str, np.ndarray]:
        """The subject's connectivity matrix for the chosen condition."""
        available = self._available_connectivity_keys(archive)
        if self.matrix_key not in available:
            raise ValueError(
                f"Zerssen matrix_key {self.matrix_key!r} is unavailable or not a square "
                f"connectivity tensor. Available matrix keys: {available}"
            )
        return self.matrix_key, np.asarray(archive[self.matrix_key])

    def matrix_options(self) -> list[dict[str, object]]:
        """Describe selectable matrices for an optional manifest UI control."""
        if not os.path.exists(self.prepared_npz):
            return []
        with np.load(self.prepared_npz) as archive:
            options = []
            for key in self._available_connectivity_keys(archive):
                n_rois = int(archive[key].shape[1])
                if n_rois not in {84, 246}:
                    continue
                data_kind = "fisher_z" if key.endswith("pearson_z") else "correlation"
                atlas_label = "BNA-246 set84 subset" if n_rois == 84 else "BNA-246 full atlas"
                options.append({
                    "key": key,
                    "n_rois": n_rois,
                    "data_kind": data_kind,
                    "label": f"{key} ({n_rois} ROIs; {atlas_label})",
                })
            return options

    def set_matrix_key(self, matrix_key: str) -> dict[str, object]:
        """Apply a UI-selected matrix and its corresponding input scale."""
        options = {option["key"]: option for option in self.matrix_options()}
        if matrix_key not in options:
            raise ValueError(f"Unsupported Zerssen matrix selection: {matrix_key!r}")
        selected = options[matrix_key]
        self.matrix_key = matrix_key
        self.data_kind = str(selected["data_kind"])
        return selected

    def _set84_roi_ids(self, archive) -> list[int]:
        """Read the 1-based BNA IDs in the exact prepared-matrix order."""
        if "set84_bnt_ids" in archive:
            return [int(value) for value in np.asarray(archive["set84_bnt_ids"]).ravel()]
        if not os.path.exists(self.set84_bnt_ids):
            return []
        try:
            with open(self.set84_bnt_ids, "r") as file_handle:
                metadata = json.load(file_handle)
            values = metadata.get("bnt_ids", []) if isinstance(metadata, dict) else metadata
            return [int(value) for value in values]
        except (OSError, TypeError, ValueError):
            return []

    def preview(self) -> DatasetPreview:
        """Cheap preview of the dataset without a full load."""
        if not os.path.exists(self.prepared_npz):
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["Prepared NPZ not found."])
        try:
            d = np.load(self.prepared_npz)
            matrix_key, data = self._selected_matrix(d)
            keys = self._available_connectivity_keys(d)
            subject_ids = [f"sub_{i:02d}" for i in range(len(data))]
            
            return DatasetPreview(
                n_subjects=len(subject_ids),
                n_observations=len(subject_ids),
                n_rois=data.shape[1],
                n_timepoints=None,
                data_kind_guess=self.data_kind,
                atlas_guess=(
                    "BNA-246" if data.shape[1] == 246
                    else "BNA-246 set84 subset" if data.shape[1] == 84
                    else os.path.basename(self.atlas_metadata)
                ),
                conditions=None,
                subject_ids_sample=subject_ids[:5],
                file_count=1,
                file_sizes_mb=os.path.getsize(self.prepared_npz) / (1024 * 1024),
                warnings=[
                    f"Selected matrix: {matrix_key} ({data.shape[1]} ROIs).",
                    f"Available connectivity matrices: {keys}",
                ]
            )
        except Exception as e:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, [f"Error: {e}"])

    def load(self) -> LoadedDataset:
        """Load the dataset into a :class:`LoadedDataset`."""
        if not os.path.exists(self.prepared_npz):
            raise FileNotFoundError(f"Prepared NPZ not found: {self.prepared_npz}")
            
        d = np.load(self.prepared_npz)
        matrix_key, data = self._selected_matrix(d)
        
        subject_ids = [f"sub_{i:02d}" for i in range(len(data))]
        groups = d["group"].tolist() if "group" in d and len(d["group"]) == len(data) else []
        if not groups:
            groups = ["PAT"] * 20 + ["HC"] * 41
            
        pheno = pd.DataFrame({
            "subject_id": subject_ids,
            "group": groups
        })
        # The prepared archive includes subject-aligned clinical, demographic,
        # and behavioral scores that can serve as GLM predictors or covariates.
        for column in ("age", "sex", "bdi", "hads_d", "hads_a", "zerssen", "ds_rt", "dn_rt"):
            if column not in d.files:
                continue
            values = np.asarray(d[column])
            if values.ndim == 1 and len(values) == len(data):
                pheno[column] = pd.to_numeric(values, errors="coerce")
        
        atlas = None
        if os.path.exists(self.atlas_metadata):
            try:
                roi_ids = self._set84_roi_ids(d)
                atlas_df = pd.read_csv(self.atlas_metadata)
                if data.shape[1] == 84:
                    if len(roi_ids) != data.shape[1]:
                        raise ValueError(
                            "Prepared set84 matrix requires 84 ROI IDs; "
                            f"found {len(roi_ids)}."
                        )
                    id_column = "roi_id" if "roi_id" in atlas_df.columns else "id"
                    if id_column not in atlas_df.columns:
                        raise ValueError(
                            "BNA atlas metadata must include an 'roi_id' or 'id' column."
                        )
                    atlas_df = atlas_df.set_index(id_column).reindex(roi_ids)
                    if atlas_df.isna().all(axis=1).any():
                        missing = [
                            roi_id for roi_id, row in atlas_df.iterrows()
                            if row.isna().all()
                        ]
                        raise ValueError(f"BNA atlas is missing set84 ROI IDs: {missing}")
                    atlas = AtlasInfo(
                        labels=atlas_df["name"].astype(str).tolist(),
                        networks=atlas_df["network"].astype(str).tolist(),
                        coords=atlas_df[["x", "y", "z"]].to_numpy(dtype=float),
                        hemisphere=atlas_df["hemisphere"].astype(str).tolist(),
                        source="Zerssen prepared set84 BNA subset",
                    )
                elif data.shape[1] == 246:
                    atlas = AtlasInfo.from_csv(self.atlas_metadata)
                else:
                    raise ValueError(
                        f"Matrix {matrix_key!r} has {data.shape[1]} ROIs. "
                        "This manifest supplies BNA metadata only for the 84-ROI subset "
                        "or the full 246-ROI atlas."
                    )
            except Exception as exc:
                raise ValueError(
                    "Failed to construct atlas metadata for the prepared Zerssen matrix: "
                    f"{exc}"
                ) from exc
                
        return LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind=self.data_kind,
            subject_ids=subject_ids,
            atlas=atlas,
            provenance={
                "prepared_npz": self.prepared_npz,
                "matrix_key": matrix_key,
                "phenotype_columns": list(pheno.columns),
            }
        )


class ChinaCloseCloseLoader(BaseDataLoader):
    """Loader for China Eyes-Closed Run 1 vs Run 2 (test-retest) comparison."""
    name = "ChinaCloseCloseLoader"
    
    def __init__(self, close_path: str, subject_list_path: str | None = None, atlas: str | None = None, drop_missing_rois: bool = True):
        self.close_path = close_path
        self.subject_list_path = subject_list_path
        self.atlas = atlas
        self.drop_missing_rois = drop_missing_rois
        
    def _get_kept_indices(self) -> np.ndarray | None:
        """ROI indices surviving the missing-ROI drop."""
        if self.drop_missing_rois and self.atlas and os.path.exists(self.atlas):
            try:
                df = pd.read_csv(self.atlas)
                if "id" in df.columns:
                    return df["id"].values - 1
            except Exception:
                pass
        return None

    def preview(self) -> DatasetPreview:
        """Cheap preview of the dataset without a full load."""
        if not os.path.exists(self.close_path):
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["Close file path not found."])
        try:
            close_raw = np.load(self.close_path)
            if close_raw.ndim != 4 or close_raw.shape[-1] != 2:
                return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["Expected 4D array with 2 runs."])
            n_sub, n_tp, n_rois, n_runs = close_raw.shape
            
            kept_idx = self._get_kept_indices()
            if kept_idx is not None:
                n_rois = len(kept_idx)
                
            subject_ids = []
            if self.subject_list_path and os.path.exists(self.subject_list_path):
                with open(self.subject_list_path, 'r') as f:
                    subject_ids = [line.strip() for line in f if line.strip()]
            else:
                subject_ids = [f"sub_{i:02d}" for i in range(n_sub)]
                
            return DatasetPreview(
                n_subjects=n_sub,
                n_observations=n_sub * 2,
                n_rois=n_rois,
                n_timepoints=n_tp,
                data_kind_guess="timeseries",
                atlas_guess=os.path.basename(self.atlas) if self.atlas else f"Schaefer-{n_rois}",
                conditions=["close_run1", "close_run2"],
                subject_ids_sample=subject_ids[:5],
                file_count=1,
                file_sizes_mb=os.path.getsize(self.close_path) / (1024 * 1024)
            )
        except Exception as e:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 1, None, [f"Error previewing close_close: {e}"])

    def load(self) -> LoadedDataset:
        """Load the dataset into a :class:`LoadedDataset`."""
        close_raw = np.load(self.close_path)
        if close_raw.ndim != 4 or close_raw.shape[-1] != 2:
            raise ValueError(f"Expected 4D array with 2 runs; got shape {close_raw.shape}")
            
        run0_data = close_raw[..., 0]
        run1_data = close_raw[..., 1]
        
        n_sub, n_tp, n_rois = run0_data.shape
        
        kept_idx = self._get_kept_indices()
        if kept_idx is not None:
            run0_data = run0_data[:, :, kept_idx]
            run1_data = run1_data[:, :, kept_idx]
            n_rois = len(kept_idx)
            
        subject_ids = []
        if self.subject_list_path and os.path.exists(self.subject_list_path):
            with open(self.subject_list_path, 'r') as f:
                subject_ids = [line.strip() for line in f if line.strip()]
        else:
            subject_ids = [f"sub_{i:02d}" for i in range(n_sub)]
            
        if len(subject_ids) != n_sub:
            subject_ids = [f"sub_{i:02d}" for i in range(n_sub)]
            
        keep_mask = np.ones(n_sub, dtype=bool)
        for idx, sid in enumerate(subject_ids):
            if sid == "sub-3258811":
                keep_mask[idx] = False
                
        run0_data = run0_data[keep_mask]
        run1_data = run1_data[keep_mask]
        subject_ids = [sid for idx, sid in enumerate(subject_ids) if keep_mask[idx]]
        n_sub = len(subject_ids)
        
        run0_corr = np.array([np.corrcoef(ts.T) for ts in run0_data])
        run1_corr = np.array([np.corrcoef(ts.T) for ts in run1_data])
        
        data = np.vstack([run0_corr, run1_corr])
        pheno = pd.DataFrame({
            "subject_id": subject_ids * 2,
            "condition": ["close_run1"] * n_sub + ["close_run2"] * n_sub
        })
        
        atlas = _load_atlas_if_present(self.atlas)
                
        return LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind="correlation",
            subject_ids=subject_ids * 2,
            conditions=["close_run1", "close_run2"],
            condition_column="condition",
            atlas=atlas,
            provenance={"close_path": self.close_path}
        )


for _cls in (
    OpenCloseLoader,
    MultiSiteOpenCloseLoader,
    StressTimeseriesLoader,
    ZerssenNiftiLoader,
    CustomPreparedZerssenLoader,
    ChinaCloseCloseLoader,
):
    register_loader(_cls)
