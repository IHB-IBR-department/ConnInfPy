from __future__ import annotations

import os
import glob
import logging
import pickle
import hashlib
import re
import numpy as np
import pandas as pd
from typing import Any, Literal
from pathlib import Path

from conninfpy.atlas.metadata import AtlasInfo
from conninfpy.loaders.base import BaseDataLoader
from conninfpy.loaders.base import LoadedDataset
from conninfpy.loaders.base import DatasetPreview
from conninfpy.loaders.validation import validate_loaded_dataset

logger = logging.getLogger(__name__)

def compute_dir_hash(files: list[str], extra_params: dict) -> str:
    """Compute a deterministic hash for a list of files and parameters to check cache validity."""
    h = hashlib.sha256()
    for f in sorted(files):
        if os.path.exists(f):
            stat = os.stat(f)
            h.update(f.encode('utf-8'))
            h.update(str(stat.st_mtime).encode('utf-8'))
            h.update(str(stat.st_size).encode('utf-8'))
    for k, v in sorted(extra_params.items()):
        h.update(f"{k}:{v}".encode('utf-8'))
    return h.hexdigest()




def _load_atlas_if_present(path: "str | None") -> "AtlasInfo | None":
    """Load an :class:`AtlasInfo` from ``path``; None if absent or unreadable."""
    if path and os.path.exists(path):
        try:
            return AtlasInfo.from_csv(path)
        except Exception:
            return None
    return None


def _masker_standardize(value: "str | bool | None") -> "str | bool | None":
    """Translate ``standardize`` for nilearn maskers.

    nilearn 0.14 accepts only ``'zscore_sample'`` (sample SD); the older
    ``'zscore'`` (population SD) raises. Map it forward: the two differ by a
    constant factor per time series, so correlations are unchanged.
    """
    return "zscore_sample" if value == "zscore" else value


def _pack_timeseries(series):
    """Stack equal-length observations; retain different frame counts otherwise."""
    if len({x.shape for x in series}) == 1:
        return np.stack(series)
    return list(series)


_BIDS_NA_VALUES = ["n/a", "N/A", "na", "NA", "NaN", "null", "NULL", ""]
_SUBJECT_ID_CANDIDATES = ("subject_id", "participant_id", "subject", "sub_id", "id")


def _read_pheno_table(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a phenotype table (.csv, .tsv, .xlsx/.xls) with BIDS n/a handling."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(p, na_values=_BIDS_NA_VALUES, keep_default_na=True)
    if suffix == ".tsv":
        return pd.read_csv(p, sep="\t", na_values=_BIDS_NA_VALUES, keep_default_na=True)
    return pd.read_csv(p, na_values=_BIDS_NA_VALUES, keep_default_na=True)


def _match_subject_id_column(df: pd.DataFrame) -> str | None:
    """Return the first recognized subject ID column (case-insensitive), or None."""
    return next(
        (str(c) for c in df.columns if str(c).lower() in _SUBJECT_ID_CANDIDATES),
        None,
    )


def _strip_sub_prefix(val: str) -> str:
    return val[4:] if val.startswith("sub-") else val


def _align_pheno_ids(pheno_ids: pd.Series, ext_ids: pd.Series) -> pd.Series:
    """Map external phenotype IDs to ``pheno_ids``, falling back to ``sub-``-stripped matching."""
    ext_str = ext_ids.astype(str)
    pheno_str = pheno_ids.astype(str)
    exact_set = set(pheno_str)
    if exact_set.issubset(set(ext_str)):
        return ext_str

    stripped_to_pheno: dict[str, list[str]] = {}
    for pid in pheno_str:
        stripped_to_pheno.setdefault(_strip_sub_prefix(pid), []).append(pid)
    unique_map = {k: v[0] for k, v in stripped_to_pheno.items() if len(set(v)) == 1}

    def _map_id(eid: str) -> str:
        if eid in exact_set:
            return eid
        return unique_map.get(_strip_sub_prefix(eid), eid)

    return ext_str.map(_map_id)


def _merge_external_pheno(pheno: pd.DataFrame, pheno_path: str | None) -> pd.DataFrame:
    """Read and left-merge an external phenotype table onto ``pheno`` by subject_id."""
    if not (pheno_path and os.path.exists(pheno_path)):
        return pheno
    ext_pheno = _read_pheno_table(pheno_path)
    id_col = _match_subject_id_column(ext_pheno)
    if id_col is None:
        return pheno
    if id_col != "subject_id":
        ext_pheno = ext_pheno.rename(columns={id_col: "subject_id"})
    pheno = pheno.copy()
    pheno["subject_id"] = pheno["subject_id"].astype(str)
    ext_pheno["subject_id"] = _align_pheno_ids(pheno["subject_id"], ext_pheno["subject_id"])
    return pd.merge(pheno, ext_pheno, on="subject_id", how="left")


class NumpyLoader(BaseDataLoader):
    """Loader for standard numpy files (.npy or .npz) containing 3D arrays."""
    name = "NumpyLoader"

    def __init__(
        self,
        data_path: str,
        pheno_path: str | None = None,
        data_kind: Literal["timeseries", "correlation", "fisher_z"] = "correlation",
        atlas_metadata: str | None = None,
    ):
        self.data_path = data_path
        self.pheno_path = pheno_path
        self.data_kind = data_kind
        # Optional atlas CSV (see AtlasInfo.from_csv), one row per ROI in data order.
        self.atlas_metadata = atlas_metadata

    def preview(self) -> DatasetPreview:
        """Cheap preview of the dataset without a full load."""
        warnings = []
        if not os.path.exists(self.data_path):
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], None, None, [f"File not found: {self.data_path}"])

        try:
            # Check if it is a pickled dict (like ABIDE)
            d = np.load(self.data_path, allow_pickle=True)
            if d.ndim == 0:  # object array containing dict
                return DatasetPreview(
                    n_subjects=None, n_observations=None, n_rois=None, n_timepoints=None,
                    data_kind_guess="unknown", atlas_guess=None, conditions=None,
                    subject_ids_sample=[], file_count=1,
                    file_sizes_mb=os.path.getsize(self.data_path) / (1024 * 1024),
                    warnings=["This file appears to contain a pickled dictionary. Please use AbideSchaeferLoader instead."]
                )

            shape = d.shape
            n_obs = shape[0]
            n_rois = shape[2] if len(shape) == 3 else None
            n_tp = shape[1] if len(shape) == 3 and self.data_kind == "timeseries" else None

            return DatasetPreview(
                n_subjects=n_obs,
                n_observations=n_obs,
                n_rois=n_rois,
                n_timepoints=n_tp,
                data_kind_guess=self.data_kind,
                atlas_guess=None,
                conditions=None,
                subject_ids_sample=[f"sub_{i:02d}" for i in range(min(5, n_obs))],
                file_count=1,
                file_sizes_mb=os.path.getsize(self.data_path) / (1024 * 1024)
            )
        except Exception as e:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 1, None, [f"Error reading file: {e}"])

    def load(self) -> LoadedDataset:
        """Load the dataset into a :class:`LoadedDataset`."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"File not found: {self.data_path}")

        data = np.load(self.data_path, allow_pickle=True)
        if data.ndim == 0:
            raise ValueError("This numpy file contains a pickled dictionary. Use AbideSchaeferLoader.")

        n_obs = data.shape[0]

        # Load pheno or generate a default one
        if self.pheno_path and os.path.exists(self.pheno_path):
            pheno = _read_pheno_table(self.pheno_path)
            id_col = _match_subject_id_column(pheno)
            if id_col and id_col != "subject_id" and "subject_id" not in pheno.columns:
                pheno = pheno.rename(columns={id_col: "subject_id"})
        else:
            pheno = pd.DataFrame({
                "subject_id": [f"sub_{i:02d}" for i in range(n_obs)]
            })

        subject_ids = list(pheno["subject_id"].values) if "subject_id" in pheno.columns else [f"sub_{i:02d}" for i in range(n_obs)]

        atlas = None
        if self.atlas_metadata:
            if not os.path.exists(self.atlas_metadata):
                raise FileNotFoundError(f"Atlas metadata not found: {self.atlas_metadata}")
            atlas = AtlasInfo.from_csv(self.atlas_metadata)
            if len(atlas) != data.shape[-1]:
                raise ValueError(
                    f"Atlas metadata has {len(atlas)} ROIs, but the data have {data.shape[-1]}."
                )

        return LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind=self.data_kind,
            subject_ids=subject_ids,
            atlas=atlas,
            roi_labels=list(atlas.labels) if atlas is not None else None,
            provenance={
                "data_path": self.data_path,
                "pheno_path": self.pheno_path,
                "atlas_metadata": self.atlas_metadata,
            },
        )


class CSVDirectoryLoader(BaseDataLoader):
    """Loader for a directory containing individual subject CSV/TSV timeseries files."""
    name = "CSVDirectoryLoader"

    def __init__(self, dir_path: str, header: bool | None = None, sep: str = ","):
        self.dir_path = dir_path
        self.header = header
        self.sep = sep

    def _get_files(self) -> list[str]:
        """Sorted list of data files in the loader's directory."""
        if not os.path.exists(self.dir_path):
            return []
        files = glob.glob(os.path.join(self.dir_path, "*.csv")) + glob.glob(os.path.join(self.dir_path, "*.tsv"))
        return sorted(files)

    def preview(self) -> DatasetPreview:
        """Cheap preview of the dataset without a full load."""
        files = self._get_files()
        if not files:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["No CSV or TSV files found in directory."])

        total_size = sum(os.path.getsize(f) for f in files) / (1024 * 1024)

        # Read a sample file to guess shapes
        try:
            sample_file = files[0]
            # Detect header: if first line is numeric, it is headerless
            header_choice = self.header
            if header_choice is None:
                with open(sample_file, 'r') as f:
                    first_line = f.readline().strip()
                try:
                    [float(x) for x in first_line.split(self.sep)]
                    header_choice = None  # No header
                except ValueError:
                    header_choice = 0  # First row is header

            df = pd.read_csv(sample_file, sep=self.sep, header=header_choice)
            n_timepoints, n_rois = df.shape

            # Extract sample subject IDs
            subject_ids_sample = [os.path.basename(f).split(".")[0] for f in files[:5]]

            return DatasetPreview(
                n_subjects=len(files),
                n_observations=len(files),
                n_rois=n_rois,
                n_timepoints=n_timepoints,
                data_kind_guess="timeseries",
                atlas_guess=None,
                conditions=None,
                subject_ids_sample=subject_ids_sample,
                file_count=len(files),
                file_sizes_mb=total_size
            )
        except Exception as e:
            return DatasetPreview(len(files), len(files), None, None, "timeseries", None, None, [], len(files), total_size, [f"Error previewing sample file: {e}"])

    def load(self) -> LoadedDataset:
        """Load the dataset into a :class:`LoadedDataset`."""
        files = self._get_files()
        if not files:
            raise FileNotFoundError(f"No CSV/TSV files found in {self.dir_path}")

        # Determine header
        header_choice = self.header
        if header_choice is None:
            with open(files[0], 'r') as f:
                first_line = f.readline().strip()
            try:
                [float(x) for x in first_line.split(self.sep)]
                header_choice = None
            except ValueError:
                header_choice = 0

        data_list = []
        subject_ids = []

        for f in files:
            df = pd.read_csv(f, sep=self.sep, header=header_choice)
            data_list.append(df.values)
            # Infer subject ID from filename
            sub_id = os.path.basename(f).split(".")[0]
            # Strip common timeseries suffixes if present
            sub_id = re.sub(r'(_ts|_timeseries|_Brainnetome|_Schaefer\d+)', '', sub_id)
            subject_ids.append(sub_id)

        data = _pack_timeseries(data_list)
        pheno = pd.DataFrame({"subject_id": subject_ids})

        return LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind="timeseries",
            subject_ids=subject_ids,
            provenance={"dir_path": self.dir_path, "file_count": len(files)}
        )


class NiftiDirectoryLoader(BaseDataLoader):
    """Loader for directories containing preprocessed functional NIfTIs (.nii/.nii.gz)."""
    name = "NiftiDirectoryLoader"

    def __init__(
        self,
        dir_path: str | None = None,
        atlas_image_path: str | None = None,
        cache_dir: str | None = None,
        confounds_pattern: str | None = None,
        file_pattern: str | None = None,
        subject_regex: str | None = None,
        group_from_folder: bool = False,
        extraction_strategy: str = "mean_signal",
        standardize: str | bool = "zscore_sample",
        detrend: bool = False,
        subjects: dict[str, str] | None = None,
        pheno_path: str | None = None,
        atlas_metadata: str | None = None,
        atlas_lut: str | None = None
    ):
        self.dir_path = dir_path
        self.atlas_image_path = atlas_image_path
        self.cache_dir = cache_dir or os.path.expanduser("~/.conninfpy/loaders")
        self.confounds_pattern = confounds_pattern
        self.file_pattern = file_pattern
        self.subject_regex = subject_regex
        self.group_from_folder = group_from_folder
        self.extraction_strategy = extraction_strategy
        self.standardize = standardize
        self.detrend = detrend
        self.subjects = subjects
        self.pheno_path = pheno_path
        self.atlas_metadata = atlas_metadata
        self.atlas_lut = atlas_lut

    def _get_files(self) -> list[str]:
        """Sorted list of data files in the loader's directory."""
        if not self.dir_path or not os.path.exists(self.dir_path):
            return []
        if self.file_pattern:
            if "*" in self.file_pattern or "?" in self.file_pattern:
                files = glob.glob(os.path.join(self.dir_path, self.file_pattern))
                if not files:
                    files = glob.glob(os.path.join(self.dir_path, "**", self.file_pattern), recursive=True)
            else:
                files = [os.path.join(self.dir_path, self.file_pattern)]
        else:
            files = glob.glob(os.path.join(self.dir_path, "*.nii")) + glob.glob(os.path.join(self.dir_path, "*.nii.gz"))
        return sorted([f for f in files if os.path.exists(f)])

    def preview(self) -> DatasetPreview:
        """Cheap preview of the dataset without a full load."""
        if self.subjects:
            total_files = 0
            total_size = 0.0
            warnings = []
            subject_ids = []
            for grp, sdir in self.subjects.items():
                temp_loader = NiftiDirectoryLoader(
                    dir_path=sdir,
                    atlas_image_path=self.atlas_image_path,
                    file_pattern=self.file_pattern,
                    subject_regex=self.subject_regex
                )
                files = temp_loader._get_files()
                total_files += len(files)
                total_size += sum(os.path.getsize(f) for f in files) / (1024 * 1024)
                for f in files[:3]:
                    sub_id = os.path.basename(f).split(".")[0]
                    if self.subject_regex:
                        m = re.search(self.subject_regex, os.path.basename(f))
                        if m:
                            sub_id = m.group(1)
                    subject_ids.append(f"{grp}/{sub_id}")

            n_rois = None
            if self.atlas_image_path and os.path.isdir(self.atlas_image_path):
                n_rois = len(glob.glob(os.path.join(self.atlas_image_path, "*.nii*")))

            return DatasetPreview(
                n_subjects=total_files,
                n_observations=total_files,
                n_rois=n_rois,
                n_timepoints=None,
                data_kind_guess="timeseries",
                atlas_guess=os.path.basename(self.atlas_image_path) if self.atlas_image_path else None,
                conditions=list(self.subjects.keys()),
                subject_ids_sample=subject_ids[:5],
                file_count=total_files,
                file_sizes_mb=total_size,
                warnings=warnings
            )

        files = self._get_files()
        if not files:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["No NIfTI files found in directory."])

        total_size = sum(os.path.getsize(f) for f in files) / (1024 * 1024)

        # Check cache status
        cache_key = compute_dir_hash(files, {
            "atlas": self.atlas_image_path,
            "confounds": self.confounds_pattern,
            "file_pattern": self.file_pattern,
            "subject_regex": self.subject_regex,
            "standardize": self.standardize,
            "detrend": self.detrend,
            "extraction_strategy": self.extraction_strategy
        })
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        cached_msg = " [Cached]" if os.path.exists(cache_file) else " [Extraction Needed]"

        n_rois = None
        if self.atlas_image_path and os.path.isdir(self.atlas_image_path):
            n_rois = len(glob.glob(os.path.join(self.atlas_image_path, "*.nii*")))

        subject_ids_sample = []
        for f in files[:5]:
            sub_id = os.path.basename(f).split(".")[0]
            if self.subject_regex:
                m = re.search(self.subject_regex, os.path.basename(f))
                if m:
                    sub_id = m.group(1)
            subject_ids_sample.append(sub_id)

        return DatasetPreview(
            n_subjects=len(files),
            n_observations=len(files),
            n_rois=n_rois,
            n_timepoints=None,
            data_kind_guess="timeseries",
            atlas_guess=os.path.basename(self.atlas_image_path) if self.atlas_image_path else None,
            conditions=None,
            subject_ids_sample=subject_ids_sample,
            file_count=len(files),
            file_sizes_mb=total_size,
            warnings=[f"Cache Status: {cached_msg}"]
        )

    def load(self) -> LoadedDataset:
        """Load the dataset into a :class:`LoadedDataset`."""
        import nibabel as nib
        from nilearn.maskers import NiftiLabelsMasker, NiftiMasker

        if self.subjects:
            all_data = []
            all_subject_ids = []
            all_groups = []
            for grp, sdir in self.subjects.items():
                sub_loader = NiftiDirectoryLoader(
                    dir_path=sdir,
                    atlas_image_path=self.atlas_image_path,
                    cache_dir=self.cache_dir,
                    confounds_pattern=self.confounds_pattern,
                    file_pattern=self.file_pattern,
                    subject_regex=self.subject_regex,
                    group_from_folder=self.group_from_folder,
                    extraction_strategy=self.extraction_strategy,
                    standardize=self.standardize,
                    detrend=self.detrend,
                    atlas_metadata=self.atlas_metadata,
                    atlas_lut=self.atlas_lut
                )
                sub_ds = sub_loader.load()
                all_data.append(sub_ds.data)
                all_subject_ids.extend(sub_ds.subject_ids)
                all_groups.extend([grp] * len(sub_ds.subject_ids))

            # Keep each observation's available frames across cohorts.
            data = _pack_timeseries([ts for group in all_data for ts in group])

            pheno = pd.DataFrame({
                "subject_id": all_subject_ids,
                "group": all_groups
            })

            pheno = _merge_external_pheno(pheno, self.pheno_path)

            atlas = _load_atlas_if_present(self.atlas_metadata)

            return LoadedDataset(
                data=data,
                pheno=pheno,
                data_kind="timeseries",
                subject_ids=all_subject_ids,
                atlas=atlas,
                provenance={
                    "subjects": self.subjects,
                    "atlas_image_path": self.atlas_image_path,
                    "pheno_path": self.pheno_path
                }
            )

        files = self._get_files()
        if not files:
            raise FileNotFoundError(f"No NIfTI files found in {self.dir_path}")

        os.makedirs(self.cache_dir, exist_ok=True)
        cache_key = compute_dir_hash(files, {
            "atlas": self.atlas_image_path,
            "confounds": self.confounds_pattern,
            "file_pattern": self.file_pattern,
            "subject_regex": self.subject_regex,
            "standardize": self.standardize,
            "detrend": self.detrend,
            "extraction_strategy": self.extraction_strategy
        })
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        data_list = []
        subject_ids = []
        groups = []

        is_mask_dir = self.atlas_image_path and os.path.isdir(self.atlas_image_path)
        if is_mask_dir:
            mask_files = sorted(glob.glob(os.path.join(self.atlas_image_path, "*.nii*")))
            if not mask_files:
                raise ValueError(f"No mask files found in atlas directory: {self.atlas_image_path}")

        std_val = _masker_standardize(self.standardize)

        for idx, f in enumerate(files):
            sub_id = os.path.basename(f).split(".")[0]
            group_val = None
            if self.subject_regex:
                m = re.search(self.subject_regex, os.path.basename(f))
                if m:
                    sub_id = m.group(1)
                    if len(m.groups()) >= 2:
                        group_val = m.group(2)
            subject_ids.append(sub_id)

            if self.group_from_folder:
                group_val = os.path.basename(os.path.dirname(f))
            if group_val:
                groups.append(group_val)

            confounds = None
            if self.confounds_pattern:
                parent_dir = os.path.dirname(f)
                confound_files = glob.glob(os.path.join(parent_dir, f"*{sub_id}*{self.confounds_pattern}*"))
                if not confound_files:
                    confound_files = glob.glob(os.path.join(self.dir_path, f"*{sub_id}*{self.confounds_pattern}*"))
                if confound_files:
                    confounds = confound_files[0]

            if is_mask_dir:
                ts_cols = []
                for mask_file in mask_files:
                    masker = NiftiMasker(mask_img=mask_file, standardize=std_val, detrend=self.detrend)
                    ts = masker.fit_transform(f, confounds=confounds)
                    ts_cols.append(ts.mean(axis=1))
                timeseries = np.column_stack(ts_cols)
            else:
                masker = NiftiLabelsMasker(labels_img=self.atlas_image_path, standardize=std_val, detrend=self.detrend)
                timeseries = masker.fit_transform(f, confounds=confounds)

            data_list.append(timeseries)

        data = _pack_timeseries(data_list)

        pheno = pd.DataFrame({"subject_id": subject_ids})
        if groups:
            pheno["group"] = groups

        pheno = _merge_external_pheno(pheno, self.pheno_path)

        atlas = _load_atlas_if_present(self.atlas_metadata)

        result = LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind="timeseries",
            subject_ids=subject_ids,
            atlas=atlas,
            provenance={
                "dir_path": self.dir_path,
                "atlas_image_path": self.atlas_image_path,
                "confounds_pattern": self.confounds_pattern,
                "hash": cache_key
            }
        )

        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

        return result


class AbideSchaeferLoader(BaseDataLoader):
    """Loader for the pickled dictionary format of the ABIDE Schaefer-100 dataset."""
    name = "AbideSchaeferLoader"

    def __init__(self, data_path: str, pheno_csv_path: str | None = None,
                 connectivity_kind="correlation", shrinkage="auto"):
        self.data_path = data_path
        self.pheno_csv_path = pheno_csv_path
        self.connectivity_kind = connectivity_kind
        self.shrinkage = shrinkage

    def preview(self) -> DatasetPreview:
        """Cheap preview of the dataset without a full load."""
        if not os.path.exists(self.data_path):
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["File not found."])

        try:
            d = np.load(self.data_path, allow_pickle=True).item()
            keys = list(d.keys())
            sample_key = keys[0]
            ts = d[sample_key]["time_series"]
            # ts is shape (timepoints, rois)
            n_tp, n_rois = ts.shape

            return DatasetPreview(
                n_subjects=len(keys),
                n_observations=len(keys),
                n_rois=n_rois,
                n_timepoints=n_tp,
                data_kind_guess="timeseries",
                atlas_guess="Schaefer-100",
                conditions=None,
                subject_ids_sample=keys[:5],
                file_count=1,
                file_sizes_mb=os.path.getsize(self.data_path) / (1024 * 1024)
            )
        except Exception as e:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 1, None, [f"Error reading ABIDE dict: {e}"])

    def load(self) -> LoadedDataset:
        """Load the dataset into a :class:`LoadedDataset`."""
        d = np.load(self.data_path, allow_pickle=True).item()

        subject_ids = list(d.keys())
        labels = []
        sites = []
        series = []
        for sub_id in subject_ids:
            series.append(d[sub_id]["time_series"])
            labels.append(d[sub_id].get("label", "unknown"))
            sites.append(d[sub_id].get("site", "unknown"))

        pheno = pd.DataFrame({
            "subject_id": subject_ids,
            "group_interest": labels,
            "site": sites
        })

        # Merge with external phenotypic CSV if provided
        if self.pheno_csv_path and os.path.exists(self.pheno_csv_path):
            try:
                ext_pheno = pd.read_csv(self.pheno_csv_path)

                # The .npy keys correspond to FILE_ID in the ABIDE dataset.
                if "FILE_ID" in ext_pheno.columns:
                    ext_pheno["FILE_ID"] = ext_pheno["FILE_ID"].astype(str)
                    pheno = pheno.merge(ext_pheno, left_on="subject_id", right_on="FILE_ID", how="left")
                else:
                    sub_col = None
                    for col in ext_pheno.columns:
                        if col.lower() in {"subject_id", "sub_id", "sub_id_str"}:
                            sub_col = col
                            break
                    if sub_col:
                        ext_pheno[sub_col] = ext_pheno[sub_col].astype(str)
                        pheno = pheno.merge(ext_pheno, left_on="subject_id", right_on=sub_col, how="left")

                # Filter down to core ABIDE columns to prevent UI clutter
                core_columns = [
                    "subject_id", "site", "group_interest", "DX_GROUP",
                    "AGE_AT_SCAN", "SEX", "FIQ", "func_mean_fd",
                    "ADOS_TOTAL", "SRS_RAW_TOTAL"
                ]
                keep_cols = [c for c in core_columns if c in pheno.columns]
                pheno = pheno[keep_cols]

                # Coerce target columns to numeric (except DX_GROUP which we map to categorical strings)
                numeric_cols = ["AGE_AT_SCAN", "SEX", "FIQ", "func_mean_fd", "ADOS_TOTAL", "SRS_RAW_TOTAL"]
                for nc in numeric_cols:
                    if nc in pheno.columns:
                        pheno[nc] = pd.to_numeric(pheno[nc], errors='coerce')

                # Rename columns for clarity in the UI
                rename_map = {
                    "DX_GROUP": "Diagnosis",
                    "AGE_AT_SCAN": "Age",
                    "SEX": "Sex",
                    "func_mean_fd": "Motion_FD"
                }
                pheno = pheno.rename(columns=rename_map)

                # Map Diagnosis to match group_interest string values (1 -> autism, 2 -> healthy_control)
                if "Diagnosis" in pheno.columns:
                    pheno["Diagnosis"] = pheno["Diagnosis"].map({1: "autism", 2: "healthy_control"})

            except Exception as e:
                # Add warning but continue
                pass

        from conninfpy.connectivity import connectivity_from_dataset
        dataset = LoadedDataset(
            data=series, pheno=pheno, data_kind="timeseries", subject_ids=subject_ids,
            provenance={"data_path": self.data_path, "pheno_csv_path": self.pheno_csv_path})
        options = {} if self.shrinkage == "auto" else {"shrinkage": self.shrinkage}
        return connectivity_from_dataset(dataset, kind=self.connectivity_kind, **options)



def build_fmriprep_confounds(img_path: str, strategy: int, n_compcor: int | None = 10, use_GSR: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    """Build an fMRIPrep confound regressor matrix for one image."""
    from nilearn.interfaces.fmriprep import load_confounds

    assert strategy in [1, 2, 3, 4, 5, 6], "Strategy must be 1-6"

    strategy_1 = {'strategy': ['motion'], 'motion': 'full'}
    strategy_2 = {'strategy': ['motion', 'compcor', 'high_pass'], 'motion': 'derivatives', 'compcor': 'anat_combined', 'n_compcor': n_compcor}
    strategy_3 = {'strategy': ['motion', 'compcor', 'high_pass'], 'motion': 'derivatives', 'compcor': 'anat_combined', 'n_compcor': 'all'}
    strategy_4 = {'strategy': ['motion', 'compcor', 'high_pass'], 'motion': 'full', 'compcor': 'anat_combined', 'n_compcor': n_compcor}
    strategy_5 = {'strategy': ['motion', 'compcor', 'high_pass'], 'motion': 'full', 'compcor': 'anat_combined', 'n_compcor': 'all'}
    strategy_6 = {'strategy': ['motion', 'compcor', 'high_pass'], 'motion': 'full', 'compcor': 'temporal_anat_combined', 'n_compcor': 'all'}

    strategy_dict = [strategy_1, strategy_2, strategy_3, strategy_4, strategy_5, strategy_6][strategy - 1].copy()

    if use_GSR:
        strategy_dict['strategy'] = list(strategy_dict['strategy']) + ['global_signal']
        strategy_dict['global_signal'] = 'full'

    return load_confounds(img_path, **strategy_dict)


class FmriprepDerivativesLoader(BaseDataLoader):
    """Loader for fMRIPrep derivatives, performing atlas extraction and confound regression."""
    name = "FmriprepDerivativesLoader"

    def __init__(
        self,
        derivatives: dict[str, str] | None = None,
        derivatives_dir: str | None = None,
        pheno_path: str | None = None,
        atlas_image_path: str | None = None,
        atlas_metadata: str | None = None,
        bold_pattern: str = "sub-*/func/*desc-preproc_bold.nii.gz",
        confounds_pattern: str = "sub-*/func/*desc-confounds_timeseries.tsv",
        subject_regex: str = "sub-([^_/]+)",
        group_from_folder: bool = False,
        denoising: dict | None = None,
        cache_dir: str | None = None
    ):
        self.derivatives = derivatives
        self.derivatives_dir = derivatives_dir
        self.pheno_path = pheno_path
        self.atlas_image_path = atlas_image_path
        self.atlas_metadata = atlas_metadata
        self.bold_pattern = bold_pattern
        self.confounds_pattern = confounds_pattern
        self.subject_regex = subject_regex
        self.group_from_folder = group_from_folder
        self.denoising = denoising or {}
        self.cache_dir = cache_dir or os.path.expanduser("~/.conninfpy/loaders")

    def _get_files(self, base_dir: str) -> list[tuple[str, str]]:
        """Sorted list of data files in the loader's directory."""
        if not os.path.exists(base_dir):
            return []

        bold_files = glob.glob(os.path.join(base_dir, self.bold_pattern))
        if not bold_files:
            bold_files = glob.glob(os.path.join(base_dir, "**", self.bold_pattern), recursive=True)

        results = []
        for bf in sorted(bold_files):
            sub_id = "sub-.*"
            m = re.search(self.subject_regex, os.path.basename(bf))
            if m:
                sub_id = m.group(1)

            parent_dir = os.path.dirname(bf)
            confound_files = glob.glob(os.path.join(parent_dir, f"*confounds*.tsv"))
            if not confound_files:
                confound_files = glob.glob(os.path.join(base_dir, "**", f"sub-{sub_id}*confounds*.tsv"), recursive=True)

            cf = confound_files[0] if confound_files else None
            results.append((bf, cf))
        return results

    def preview(self) -> DatasetPreview:
        """Cheap preview of the dataset without a full load."""
        dirs_to_scan = {}
        if self.derivatives:
            dirs_to_scan = self.derivatives
        elif self.derivatives_dir:
            dirs_to_scan = {"all": self.derivatives_dir}

        total_files = 0
        total_size = 0.0
        sample_subs = []
        for grp, sdir in dirs_to_scan.items():
            pairs = self._get_files(sdir)
            total_files += len(pairs)
            for bf, _ in pairs:
                total_size += os.path.getsize(bf) / (1024 * 1024)
            for bf, _ in pairs[:3]:
                sub_id = os.path.basename(bf).split("_")[0]
                sample_subs.append(f"{grp}/{sub_id}")

        n_rois = None
        if self.atlas_image_path and os.path.isdir(self.atlas_image_path):
            n_rois = len(glob.glob(os.path.join(self.atlas_image_path, "*.nii*")))

        return DatasetPreview(
            n_subjects=total_files,
            n_observations=total_files,
            n_rois=n_rois,
            n_timepoints=None,
            data_kind_guess="timeseries",
            atlas_guess=os.path.basename(self.atlas_image_path) if self.atlas_image_path else None,
            conditions=list(dirs_to_scan.keys()) if self.derivatives else None,
            subject_ids_sample=sample_subs[:5],
            file_count=total_files,
            file_sizes_mb=total_size
        )

    def load(self) -> LoadedDataset:
        """Load the dataset into a :class:`LoadedDataset`."""
        import nibabel as nib
        from nilearn.maskers import NiftiLabelsMasker, NiftiMasker

        dirs_to_scan = {}
        if self.derivatives:
            dirs_to_scan = self.derivatives
        elif self.derivatives_dir:
            dirs_to_scan = {"all": self.derivatives_dir}

        all_data = []
        all_subject_ids = []
        all_groups = []

        strategy = self.denoising.get("strategy", 4)
        n_compcor = self.denoising.get("n_compcor", 10)
        use_GSR = self.denoising.get("use_GSR", False)
        smoothing = self.denoising.get("smoothing", None)

        is_mask_dir = self.atlas_image_path and os.path.isdir(self.atlas_image_path)
        if is_mask_dir:
            mask_files = sorted(glob.glob(os.path.join(self.atlas_image_path, "*.nii*")))
            if not mask_files:
                raise ValueError(f"No mask files found in atlas directory: {self.atlas_image_path}")

        for grp, sdir in dirs_to_scan.items():
            pairs = self._get_files(sdir)
            for bf, cf in pairs:
                sub_id = os.path.basename(bf).split(".")[0]
                m = re.search(self.subject_regex, os.path.basename(bf))
                if m:
                    sub_id = m.group(1)

                cache_key = compute_dir_hash([bf, cf] if cf else [bf], {
                    "atlas": self.atlas_image_path,
                    "strategy": strategy,
                    "n_compcor": n_compcor,
                    "use_GSR": use_GSR,
                    "smoothing": smoothing,
                    # series cached before the censoring fix kept flagged volumes
                    "sample_mask": "applied",
                })
                os.makedirs(self.cache_dir, exist_ok=True)
                cache_file = os.path.join(self.cache_dir, f"fmriprep_{cache_key}.npy")

                if os.path.exists(cache_file):
                    timeseries = np.load(cache_file)
                else:
                    # load_confounds returns the volumes to keep (non-steady-state
                    # volumes, scrubbed outliers removed); the masker must apply it
                    confounds, sample_mask = None, None
                    if cf:
                        confounds, sample_mask = build_fmriprep_confounds(bf, strategy, n_compcor, use_GSR)

                    if is_mask_dir:
                        ts_cols = []
                        for mask_file in mask_files:
                            masker = NiftiMasker(mask_img=mask_file, standardize="zscore_sample")
                            if smoothing is not None:
                                masker.set_params(smoothing_fwhm=smoothing)
                            ts = masker.fit_transform(bf, confounds=confounds, sample_mask=sample_mask)
                            ts_cols.append(ts.mean(axis=1))
                        timeseries = np.column_stack(ts_cols)
                    else:
                        masker = NiftiLabelsMasker(labels_img=self.atlas_image_path, standardize="zscore_sample")
                        if smoothing is not None:
                            masker.set_params(smoothing_fwhm=smoothing)
                        timeseries = masker.fit_transform(bf, confounds=confounds, sample_mask=sample_mask)

                    np.save(cache_file, timeseries)

                all_data.append(timeseries)
                all_subject_ids.append(sub_id)
                if self.derivatives:
                    all_groups.append(grp)
                elif self.group_from_folder:
                    all_groups.append(os.path.basename(os.path.dirname(os.path.dirname(bf))))
                else:
                    all_groups.append("control")

        n_volumes = [int(ts.shape[0]) for ts in all_data]
        data = _pack_timeseries(all_data)

        pheno = pd.DataFrame({
            "subject_id": all_subject_ids,
            "group": all_groups
        })

        pheno = _merge_external_pheno(pheno, self.pheno_path)

        atlas = _load_atlas_if_present(self.atlas_metadata)

        from conninfpy.preprocessing import SOURCE_LOADER_DENOISING, attach_preprocessing, fact

        result = LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind="timeseries",
            subject_ids=all_subject_ids,
            atlas=atlas,
            provenance={
                "bold_pattern": self.bold_pattern,
                "strategy": strategy,
                "use_GSR": use_GSR,
                "censoring": "load_confounds sample_mask applied",
                "n_volumes": dict(zip(map(str, all_subject_ids), n_volumes)),
                "truncated_to": n_volumes[0] if len(set(n_volumes)) == 1 else None,
            }
        )
        # The denoising (incl. its global-signal option) is performed by this
        # loader's pipeline; the fact records what was actually configured.
        attach_preprocessing(result, gsr=fact(bool(use_GSR), SOURCE_LOADER_DENOISING))
        return result


class TimeseriesDirectoryLoader(BaseDataLoader):
    """Loader for directories containing already extracted time-series CSV/NPY files."""
    name = "TimeseriesDirectoryLoader"

    def __init__(
        self,
        dir_path: str | None = None,
        dir_paths: dict[str, str] | None = None,
        pheno_path: str | None = None,
        atlas: str | None = None,
        pattern: str = "*.csv",
        sep: str = ",",
        header: str | bool | None = "infer",
        subject_regex: str = "sub-([^_/]+)",
        run_regex: str | None = None
    ):
        self.dir_path = dir_path
        self.dir_paths = dir_paths
        self.pheno_path = pheno_path
        self.atlas = atlas
        self.pattern = pattern
        self.sep = sep
        self.header = header
        self.subject_regex = subject_regex
        self.run_regex = run_regex

    def _get_files_from_dir(self, directory: str) -> list[str]:
        """Sorted data files inside one directory."""
        if not os.path.exists(directory):
            return []
        files = glob.glob(os.path.join(directory, self.pattern))
        if not files:
            files = glob.glob(os.path.join(directory, "**", self.pattern), recursive=True)
        return sorted(files)

    def _get_all_files(self) -> dict[str, list[str]]:
        """Condition → file list mapping for all conditions."""
        if self.dir_paths:
            return {grp: self._get_files_from_dir(d) for grp, d in self.dir_paths.items()}
        elif self.dir_path:
            return {"control": self._get_files_from_dir(self.dir_path)}
        return {}

    def preview(self) -> DatasetPreview:
        """Cheap preview of the dataset without a full load."""
        all_groups_files = self._get_all_files()
        total_files = sum(len(f) for f in all_groups_files.values())
        if total_files == 0:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["No files matched pattern."])

        first_grp = list(all_groups_files.keys())[0]
        first_file = all_groups_files[first_grp][0]
        n_tp, n_rois = None, None
        try:
            if first_file.endswith(".npy"):
                arr = np.load(first_file)
                n_tp, n_rois = arr.shape
            else:
                hdr = 0 if self.header == "infer" or self.header is True else None
                df = pd.read_csv(first_file, sep=self.sep, header=hdr)
                n_tp, n_rois = df.shape
        except Exception:
            pass

        subject_ids = []
        for grp, files in all_groups_files.items():
            for f in files[:3]:
                sub_id = os.path.basename(f).split(".")[0]
                m = re.search(self.subject_regex, os.path.basename(f))
                if m:
                    sub_id = m.group(1)
                subject_ids.append(f"{grp}/{sub_id}")

        total_sizes = 0.0
        for grp, files in all_groups_files.items():
            total_sizes += sum(os.path.getsize(f) for f in files) / (1024 * 1024)

        return DatasetPreview(
            n_subjects=total_files,
            n_observations=total_files,
            n_rois=n_rois,
            n_timepoints=n_tp,
            data_kind_guess="timeseries",
            atlas_guess=os.path.basename(self.atlas) if self.atlas else None,
            conditions=list(all_groups_files.keys()) if self.dir_paths else None,
            subject_ids_sample=subject_ids[:5],
            file_count=total_files,
            file_sizes_mb=total_sizes
        )

    def load(self) -> LoadedDataset:
        """Load the dataset into a :class:`LoadedDataset`."""
        all_groups_files = self._get_all_files()
        total_files = sum(len(f) for f in all_groups_files.values())
        if total_files == 0:
            raise FileNotFoundError(f"No files matched pattern: {self.pattern}")

        data_list = []
        subject_ids = []
        groups = []

        hdr = 0 if self.header == "infer" or self.header is True else None

        for grp, files in all_groups_files.items():
            for f in files:
                sub_id = os.path.basename(f).split(".")[0]
                m = re.search(self.subject_regex, os.path.basename(f))
                if m:
                    sub_id = m.group(1)
                subject_ids.append(sub_id)
                groups.append(grp)

                if f.endswith(".npy"):
                    arr = np.asarray(np.load(f), dtype=np.float64)
                else:
                    df = pd.read_csv(f, sep=self.sep, header=hdr)
                    try:
                        arr = df.apply(pd.to_numeric, errors="raise").to_numpy(
                            dtype=np.float64,
                        )
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Time-series file contains non-numeric values: {f}. "
                            "Check the manifest 'header' and 'sep' settings."
                        ) from exc
                data_list.append(arr)

        data = _pack_timeseries(data_list)

        pheno = pd.DataFrame({
            "subject_id": subject_ids,
            "group": groups
        })

        pheno = _merge_external_pheno(pheno, self.pheno_path)

        atlas = _load_atlas_if_present(self.atlas)

        return LoadedDataset(
            data=data,
            pheno=pheno,
            data_kind="timeseries",
            subject_ids=subject_ids,
            atlas=atlas,
            provenance={
                "dir_path": self.dir_path,
                "dir_paths": self.dir_paths,
                "pattern": self.pattern
            }
        )


class ConditionTimeseriesArrayLoader(BaseDataLoader):
    """Loader for paired-condition arrays (e.g. Open-Close) in .npy format."""
    name = "ConditionTimeseriesArrayLoader"

    def __init__(
        self,
        data_dir: str,
        subject_order_path: str,
        atlas: str | None = None,
        site: str = "ihb",
        conditions: list[str] | None = None,
        atlas_token: str = "Schaefer200",
        strategy: str = "4",
        gsr: str = "GSR",
        file_template: str = "{site}_{condition}_{atlas_token}_strategy-{strategy}_{gsr}.npy",
        array_shape: str = "subjects_timepoints_rois",
        close_session_policy: str = "first",
        connectivity_kind: str = "correlation",
        shrinkage="auto"
    ):
        self.data_dir = data_dir
        self.subject_order_path = subject_order_path
        self.atlas = atlas
        self.site = site
        self.conditions = conditions or ["open", "close"]
        self.atlas_token = atlas_token
        self.strategy = strategy
        self.gsr = gsr
        self.file_template = file_template
        self.array_shape = array_shape
        self.close_session_policy = close_session_policy
        self.connectivity_kind = connectivity_kind
        self.shrinkage = shrinkage

    def _get_subject_order(self) -> list[str]:
        """Subject order reconstructed from the listing file."""
        if os.path.exists(self.subject_order_path):
            with open(self.subject_order_path, "r") as f:
                return [line.strip() for line in f if line.strip()]
        return []

    def preview(self) -> DatasetPreview:
        """Cheap preview of the dataset without a full load."""
        sub_list = self._get_subject_order()
        if not sub_list:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["Subject order list empty or missing."])

        cond = self.conditions[0]
        fname = self.file_template.format(
            site=self.site, condition=cond, atlas_token=self.atlas_token,
            strategy=self.strategy, gsr=self.gsr
        )
        fpath = os.path.join(self.data_dir, fname)
        if not os.path.exists(fpath):
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, [f"File not found: {fname}"])

        try:
            arr = np.load(fpath)
            n_sub = arr.shape[0]
            n_tp = arr.shape[1]
            n_rois = arr.shape[2]

            return DatasetPreview(
                n_subjects=len(sub_list),
                n_observations=len(sub_list) * len(self.conditions),
                n_rois=n_rois,
                n_timepoints=n_tp,
                data_kind_guess="timeseries",
                atlas_guess=os.path.basename(self.atlas) if self.atlas else self.atlas_token,
                conditions=self.conditions,
                subject_ids_sample=sub_list[:5],
                file_count=len(self.conditions),
                file_sizes_mb=os.path.getsize(fpath) * len(self.conditions) / (1024 * 1024)
            )
        except Exception as e:
            return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, [f"Error: {e}"])

    def load(self) -> LoadedDataset:
        """Load the dataset into a :class:`LoadedDataset`."""
        sub_list = self._get_subject_order()
        if not sub_list:
            raise FileNotFoundError(f"Subject list not found: {self.subject_order_path}")

        all_series = []
        all_conditions = []
        all_pheno_subs = []

        for cond in self.conditions:
            fname = self.file_template.format(
                site=self.site, condition=cond, atlas_token=self.atlas_token,
                strategy=self.strategy, gsr=self.gsr
            )
            fpath = os.path.join(self.data_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing file for condition '{cond}': {fpath}")

            arr = np.load(fpath)

            if arr.ndim == 4:
                if self.close_session_policy == "first":
                    arr = arr[:, :, :, 0]
                elif self.close_session_policy == "mean":
                    arr = arr.mean(axis=-1)
                else:
                    raise ValueError(f"Unsupported session policy: {self.close_session_policy}")

            all_series.extend(arr)
            all_conditions.extend([cond] * len(sub_list))
            all_pheno_subs.extend(sub_list)

        data = _pack_timeseries(all_series)
        pheno = pd.DataFrame({
            "subject_id": all_pheno_subs,
            "condition": all_conditions
        })

        atlas = _load_atlas_if_present(self.atlas)

        from conninfpy.connectivity import connectivity_from_dataset
        dataset = LoadedDataset(
            data=data, pheno=pheno, data_kind="timeseries", subject_ids=all_pheno_subs,
            conditions=self.conditions, condition_column="condition", atlas=atlas,
            provenance={"data_dir": self.data_dir, "strategy": self.strategy})
        # The file-name token declares the upstream denoising variant; it was
        # not performed by this library (no inference from the matrix values).
        from conninfpy.preprocessing import SOURCE_FILE_TOKEN, attach_preprocessing, fact, gsr_from_file_token
        gsr_status = gsr_from_file_token(self.gsr)
        attach_preprocessing(dataset, gsr=fact(gsr_status, SOURCE_FILE_TOKEN) if gsr_status is not None else None)
        options = {} if self.shrinkage == "auto" else {"shrinkage": self.shrinkage}
        return connectivity_from_dataset(dataset, kind=self.connectivity_kind, **options)



class ConnectivityMatrixLoader(BaseDataLoader):
    """Loader for inference-ready connectivity matrices (.npy/.npz or per-subject matrix files)."""
    name = "ConnectivityMatrixLoader"

    def __init__(
        self,
        data_path: str | None = None,
        data_paths: dict[str, str] | None = None,
        pheno_path: str | None = None,
        atlas: str | None = None,
        data_kind: Literal["correlation", "fisher_z"] = "fisher_z",
        dir_path: str | None = None,
        pattern: str = "*.npy",
        sep: str = ",",
        header: str | bool | None = None,
        subject_regex: str = r"sub-([^_/.]+)",
    ):
        self.data_path = data_path
        self.data_paths = data_paths
        self.pheno_path = pheno_path
        self.atlas = atlas
        self.data_kind = data_kind
        self.dir_path = dir_path
        self.pattern = pattern
        self.sep = sep
        self.header = header
        self.subject_regex = subject_regex

    def _get_dir_files(self) -> list[str]:
        if not self.dir_path or not os.path.exists(self.dir_path):
            return []
        files = glob.glob(os.path.join(self.dir_path, self.pattern))
        if not files:
            files = glob.glob(os.path.join(self.dir_path, "**", self.pattern), recursive=True)
        pheno_abs = os.path.abspath(self.pheno_path) if self.pheno_path else None
        out: list[str] = []
        for f in sorted(files):
            if pheno_abs and os.path.abspath(f) == pheno_abs:
                continue
            if os.path.basename(f).lower() in ("participants.tsv", "participants.csv"):
                continue
            out.append(f)
        return out

    def _extract_subject_id(self, filepath: str) -> str:
        base = os.path.basename(filepath)
        sub_id = base.split(".")[0]
        if self.subject_regex:
            m = re.search(self.subject_regex, base)
            if m:
                sub_id = m.group(1)
        return sub_id

    def _read_matrix_file(self, filepath: str) -> np.ndarray:
        if filepath.endswith(".npy"):
            arr = np.asarray(np.load(filepath), dtype=np.float64)
        else:
            sep = "\t" if (self.sep == "," and filepath.lower().endswith(".tsv")) else self.sep
            hdr = 0 if self.header == "infer" or self.header is True else None
            df = pd.read_csv(filepath, sep=sep, header=hdr)
            try:
                arr = df.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Connectivity matrix file contains non-numeric values: {filepath}. "
                    "Check the manifest 'header' and 'sep' settings."
                ) from exc
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"Expected square 2D connectivity matrix in {filepath}; got shape {arr.shape}.")
        return arr

    def preview(self) -> DatasetPreview:
        """Cheap preview of the dataset without a full load."""
        if self.dir_path:
            files = self._get_dir_files()
            if not files:
                return DatasetPreview(None, None, None, None, "unknown", None, None, [], 0, 0.0, ["No files matched pattern."])
            n_rois = None
            try:
                mat = self._read_matrix_file(files[0])
                n_rois = int(mat.shape[0])
            except Exception:
                pass
            sample_ids = [self._extract_subject_id(f) for f in files[:5]]
            total_sizes = sum(os.path.getsize(f) for f in files) / (1024 * 1024)
            return DatasetPreview(
                n_subjects=len(files),
                n_observations=len(files),
                n_rois=n_rois,
                n_timepoints=None,
                data_kind_guess=self.data_kind,
                atlas_guess=os.path.basename(self.atlas) if self.atlas else None,
                conditions=None,
                subject_ids_sample=sample_ids,
                file_count=len(files),
                file_sizes_mb=total_sizes,
            )
        if self.data_paths:
            first_grp = list(self.data_paths.keys())[0]
            nl = NumpyLoader(data_path=self.data_paths[first_grp], pheno_path=self.pheno_path, data_kind=self.data_kind)
            prev = nl.preview()
            prev.atlas_guess = os.path.basename(self.atlas) if self.atlas else None

            total_observations = 0
            total_sizes = 0.0
            for grp, pth in self.data_paths.items():
                if os.path.exists(pth):
                    try:
                        arr = np.load(pth)
                        total_observations += arr.shape[0]
                        total_sizes += os.path.getsize(pth) / (1024 * 1024)
                    except Exception:
                        pass
            prev.n_observations = total_observations
            prev.n_subjects = total_observations
            prev.conditions = list(self.data_paths.keys())
            prev.file_count = len(self.data_paths)
            prev.file_sizes_mb = total_sizes
            return prev
        else:
            nl = NumpyLoader(data_path=self.data_path, pheno_path=self.pheno_path, data_kind=self.data_kind)
            prev = nl.preview()
            prev.atlas_guess = os.path.basename(self.atlas) if self.atlas else None
            return prev

    def load(self) -> LoadedDataset:
        """Load the dataset into a :class:`LoadedDataset`."""
        if self.dir_path:
            files = self._get_dir_files()
            if not files:
                raise FileNotFoundError(f"No files matched pattern: {self.pattern} in {self.dir_path}")
            matrices = [self._read_matrix_file(f) for f in files]
            data = np.stack(matrices, axis=0)
            subject_ids = [self._extract_subject_id(f) for f in files]
            pheno = pd.DataFrame({"subject_id": subject_ids})
            pheno = _merge_external_pheno(pheno, self.pheno_path)
            atlas = _load_atlas_if_present(self.atlas)
            return LoadedDataset(
                data=data,
                pheno=pheno,
                data_kind=self.data_kind,
                subject_ids=subject_ids,
                atlas=atlas,
                roi_labels=list(atlas.labels) if atlas is not None else None,
                provenance={
                    "dir_path": self.dir_path,
                    "pheno_path": self.pheno_path,
                    "pattern": self.pattern,
                },
            )
        if self.data_paths:
            all_data = []
            all_groups = []
            all_subject_ids = []
            for grp, pth in self.data_paths.items():
                nl = NumpyLoader(data_path=pth, pheno_path=self.pheno_path, data_kind=self.data_kind)
                ds = nl.load()
                all_data.append(ds.data)
                all_groups.extend([grp] * len(ds.data))
                all_subject_ids.extend([f"{grp}_{sid}" for sid in ds.subject_ids])

            data = np.vstack(all_data)
            pheno = pd.DataFrame({
                "subject_id": all_subject_ids,
                "group": all_groups
            })

            atlas = _load_atlas_if_present(self.atlas)

            return LoadedDataset(
                data=data,
                pheno=pheno,
                data_kind=self.data_kind,
                subject_ids=all_subject_ids,
                atlas=atlas,
                provenance={"data_paths": self.data_paths}
            )
        else:
            nl = NumpyLoader(data_path=self.data_path, pheno_path=self.pheno_path, data_kind=self.data_kind)
            dataset = nl.load()

            if self.atlas and os.path.exists(self.atlas):
                try:
                    dataset.atlas = AtlasInfo.from_csv(self.atlas)
                    dataset.roi_labels = dataset.atlas.labels
                except Exception:
                    pass

            return dataset


