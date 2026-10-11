from conninfpy.loaders.base import LoadedDataset
from conninfpy.loaders.base import DatasetPreview
from conninfpy.loaders.base import DataValidationReport
from conninfpy.loaders.base import BaseDataLoader
from conninfpy.loaders.validation import validate_loaded_dataset
from conninfpy.loaders.builtins import NumpyLoader
from conninfpy.loaders.builtins import CSVDirectoryLoader
from conninfpy.loaders.builtins import NiftiDirectoryLoader
from conninfpy.loaders.builtins import AbideSchaeferLoader
from conninfpy.loaders.builtins import FmriprepDerivativesLoader
from conninfpy.loaders.builtins import TimeseriesDirectoryLoader
from conninfpy.loaders.builtins import ConnectivityMatrixLoader
from conninfpy.loaders.builtins import ConditionTimeseriesArrayLoader
from conninfpy.loaders.manifest import DatasetManifest
from conninfpy.loaders.manifest import ManifestLoader
from conninfpy.loaders.manifest import get_loader
from conninfpy.loaders.manifest import register_loader
from conninfpy.loaders.manifest import validate_manifest_dataset
from conninfpy.loaders.manifest import load_manifest


def load(manifest_path) -> LoadedDataset:
    """Load a dataset described by a YAML manifest.

    Shortcut for ``ManifestLoader(manifest_path).load()``. The result
    carries ``data`` (n_subjects, N, N) or timeseries, ``pheno`` (a
    per-subject DataFrame), ``data_kind`` and, when known, ``atlas``.
    """
    return ManifestLoader(str(manifest_path)).load()


__all__ = [
    "load",
    "register_loader",
    "get_loader",
    "LoadedDataset",
    "DatasetPreview",
    "DataValidationReport",
    "BaseDataLoader",
    "validate_loaded_dataset",
    "NumpyLoader",
    "CSVDirectoryLoader",
    "NiftiDirectoryLoader",
    "AbideSchaeferLoader",
    "FmriprepDerivativesLoader",
    "TimeseriesDirectoryLoader",
    "ConnectivityMatrixLoader",
    "ConditionTimeseriesArrayLoader",
    "DatasetManifest",
    "ManifestLoader",
    "validate_manifest_dataset",
    "load_manifest"
]
