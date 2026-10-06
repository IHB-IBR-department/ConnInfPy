from conninfpy.loaders.base import (
    LoadedDataset,
    DatasetPreview,
    DataValidationReport,
    BaseDataLoader
)
from conninfpy.loaders.validation import validate_loaded_dataset
from conninfpy.loaders.builtins import (
    NumpyLoader,
    CSVDirectoryLoader,
    NiftiDirectoryLoader,
    AbideSchaeferLoader,
    OpenCloseLoader,
    MultiSiteOpenCloseLoader,
    StressTimeseriesLoader,
    ZerssenNiftiLoader,
    FmriprepDerivativesLoader,
    TimeseriesDirectoryLoader,
    ConnectivityMatrixLoader,
    ConditionTimeseriesArrayLoader,
    CustomPreparedZerssenLoader,
    ChinaCloseCloseLoader
)
from conninfpy.loaders.manifest import (
    DatasetManifest,
    ManifestLoader,
    validate_manifest_dataset,
    load_manifest
)


def load(manifest_path) -> LoadedDataset:
    """Load a dataset described by a YAML manifest.

    Shortcut for ``ManifestLoader(manifest_path).load()``. The result
    carries ``data`` (n_subjects, N, N) or timeseries, ``pheno`` (a
    per-subject DataFrame), ``data_kind`` and, when known, ``atlas``.
    """
    return ManifestLoader(str(manifest_path)).load()


__all__ = [
    "load",
    "LoadedDataset",
    "DatasetPreview",
    "DataValidationReport",
    "BaseDataLoader",
    "validate_loaded_dataset",
    "NumpyLoader",
    "CSVDirectoryLoader",
    "NiftiDirectoryLoader",
    "AbideSchaeferLoader",
    "OpenCloseLoader",
    "MultiSiteOpenCloseLoader",
    "StressTimeseriesLoader",
    "ZerssenNiftiLoader",
    "FmriprepDerivativesLoader",
    "TimeseriesDirectoryLoader",
    "ConnectivityMatrixLoader",
    "ConditionTimeseriesArrayLoader",
    "CustomPreparedZerssenLoader",
    "ChinaCloseCloseLoader",
    "DatasetManifest",
    "ManifestLoader",
    "validate_manifest_dataset",
    "load_manifest"
]
