from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from conninfpy.loaders.base import LoadedDataset
from conninfpy.loaders.base import DataValidationReport

def validate_loaded_dataset(dataset: LoadedDataset, check_inference_ready: bool = False) -> DataValidationReport:
    """Validate a LoadedDataset object for structural coherence and mathematical validity.
    
    Parameters
    ----------
    dataset : LoadedDataset
        The loaded dataset to validate.
    check_inference_ready : bool
        If True, validates that the dataset is fully preprocessed and ready for statistical inference
        (e.g., matrices are square, symmetric, data_kind is correlation/fisher_z).
        If False, validates the raw loaded format (which might be raw timeseries).
        
    Returns
    -------
    DataValidationReport
        Contains the pass status (ok), list of error strings, list of warning strings, and summary info.
    """
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {}

    data = dataset.data
    pheno = dataset.pheno
    data_kind = dataset.data_kind

    if data_kind == "timeseries":
        if check_inference_ready:
            errors.append("Inference-ready dataset must be connectivity, not timeseries.")
        if not isinstance(pheno, pd.DataFrame):
            errors.append("dataset.pheno must be a pandas DataFrame.")
            return DataValidationReport(False, errors, warnings, summary)
        if isinstance(data, np.ndarray) and data.ndim == 3:
            series = list(data)
        elif isinstance(data, list):
            series = data
        else:
            errors.append("Timeseries data must be a 3D array or a list of 2D arrays.")
            return DataValidationReport(False, errors, warnings, summary)
        if not series:
            errors.append("Timeseries data has no observations.")
            return DataValidationReport(False, errors, warnings, summary)
        if len(series) != len(pheno):
            errors.append(f"Subject count mismatch: data has {len(series)} observations, but pheno has {len(pheno)} rows.")
        shapes = []
        for i, raw in enumerate(series):
            arr = np.asarray(raw)
            if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.number):
                errors.append(f"Observation {i} must be a numeric 2D timeseries.")
                continue
            shapes.append(arr.shape)
            valid = arr[np.isfinite(arr).all(axis=1)]
            if len(valid) < 2:
                errors.append(f"Observation {i} has fewer than two complete finite frames.")
            else:
                if len(valid) < 10:
                    warnings.append(f"Observation {i} has only {len(valid)} valid frames.")
                bad = np.flatnonzero(np.var(valid, axis=0) == 0)
                if len(bad):
                    warnings.append(f"Observation {i} has constant timeseries in ROIs: {bad.tolist()}.")
                if len(valid) < len(arr):
                    warnings.append(f"Observation {i} has {len(arr) - len(valid)} incomplete frames to scrub.")
        if shapes:
            roi_counts = {shape[1] for shape in shapes}
            if len(roi_counts) != 1:
                errors.append("Timeseries observations have inconsistent ROI counts.")
            n_rois = shapes[0][1]
            lengths = [shape[0] for shape in shapes]
            summary.update(n_timepoints=lengths[0] if len(set(lengths)) == 1 else (min(lengths), max(lengths)), n_rois=n_rois)
            if dataset.atlas is not None and len(dataset.atlas) != n_rois:
                errors.append(f"Atlas/Node count mismatch: atlas has {len(dataset.atlas)} labels, but data has {n_rois} nodes.")
        ids = dataset.subject_ids
        if ids is not None:
            if len(ids) != len(series):
                errors.append(f"Length of subject_ids ({len(ids)}) does not match observations ({len(series)}).")
            elif len(set(ids)) < len(ids):
                if not dataset.condition_column or dataset.condition_column not in pheno:
                    warnings.append("Duplicate subject IDs detected, but no valid condition_column is specified.")
                elif len(set(zip(ids, pheno[dataset.condition_column]))) < len(ids):
                    errors.append("Duplicate subject IDs found within the same condition.")
        summary.update(n_observations=len(series), data_kind=data_kind, ok=not errors)
        return DataValidationReport(not errors, errors, warnings, summary)

    # Basic type checks
    if not isinstance(data, np.ndarray):
        errors.append("dataset.data must be a numpy ndarray.")
        return DataValidationReport(ok=False, errors=errors, warnings=warnings, summary={})

    if not isinstance(pheno, pd.DataFrame):
        errors.append("dataset.pheno must be a pandas DataFrame.")
        return DataValidationReport(ok=False, errors=errors, warnings=warnings, summary={})

    if data.ndim != 3:
        errors.append(f"dataset.data must be 3D (got ndim={data.ndim}, shape={data.shape}).")
        return DataValidationReport(ok=False, errors=errors, warnings=warnings, summary={})

    n_observations = data.shape[0]
    n_pheno_rows = len(pheno)

    if n_observations != n_pheno_rows:
        errors.append(f"Subject count mismatch: data has {n_observations} observations, but pheno has {n_pheno_rows} rows.")

    # Check numeric type before calling np.isfinite(), whose error for object
    # arrays hides the real ingestion problem (usually a CSV header mismatch).
    if not np.issubdtype(data.dtype, np.number):
        errors.append(
            f"Data must be numeric, got dtype {data.dtype}. Check CSV header and separator settings."
        )
    elif not np.isfinite(data).all():
        if data.ndim == 3 and data.shape[1] == data.shape[2]:
            from conninfpy._prepare import _format_nonfinite
            from conninfpy._prepare import nonfinite_report

            report = nonfinite_report(data)
            errors.append(_format_nonfinite(report, "Connectivity data") if report["n_values"]
                          else "Connectivity data has NaN/infinite values on the diagonal only "
                               "(a region with a constant signal?).")
        else:
            errors.append("Data contains missing, infinite, or NaN values.")

    # Retrieve subject IDs if present
    subj_ids = dataset.subject_ids
    if subj_ids is not None:
        if len(subj_ids) != n_observations:
            errors.append(f"Length of subject_ids ({len(subj_ids)}) does not match data shape ({n_observations}).")
        
        # Check uniqueness of subject IDs
        # Allow repeats ONLY if there is a condition_column declared and valid
        if len(set(subj_ids)) < len(subj_ids):
            if not dataset.condition_column or dataset.condition_column not in pheno.columns:
                warnings.append("Duplicate subject IDs detected, but no valid condition_column is specified.")
            else:
                # Validate that combinations of subject_id + condition are unique
                combos = list(zip(subj_ids, pheno[dataset.condition_column]))
                if len(set(combos)) < len(combos):
                    errors.append("Duplicate subject IDs found within the same condition.")

    # Validation based on kind
    if data_kind == "timeseries":
        n_timepoints = data.shape[1]
        n_nodes = data.shape[2]
        summary["n_timepoints"] = n_timepoints
        summary["n_rois"] = n_nodes
        
        if n_timepoints < 10:
            warnings.append(f"Very short timeseries detected ({n_timepoints} timepoints). Correlation estimation may be unstable.")
            
        # Check for constant timeseries (zero variance)
        for i in range(n_observations):
            var = np.var(data[i], axis=0)
            if np.any(var == 0):
                zero_nodes = np.where(var == 0)[0]
                warnings.append(f"Observation {i} has constant timeseries (zero variance) in ROIs: {zero_nodes.tolist()}.")
                
    elif data_kind in {"correlation", "fisher_z"}:
        n_nodes_i = data.shape[1]
        n_nodes_j = data.shape[2]
        summary["n_rois"] = n_nodes_i

        if n_nodes_i != n_nodes_j:
            errors.append(f"Connectivity matrices are not square: got shape ({n_nodes_i}, {n_nodes_j}).")
        else:
            # Check symmetry (within numerical precision)
            for i in range(n_observations):
                matrix = data[i]
                if not np.allclose(matrix, matrix.T, atol=1e-5):
                    errors.append(f"Connectivity matrix for observation {i} is not symmetric.")
                    break
    else:
        errors.append(f"Unknown data_kind: {data_kind}. Must be 'timeseries', 'correlation', or 'fisher_z'.")

    # If checking inference readiness, enforce that it is connectivity and contains no NaNs
    if check_inference_ready:
        if data_kind not in {"correlation", "fisher_z"}:
            errors.append(f"Inference-ready dataset must be 'correlation' or 'fisher_z', not {data_kind}.")
        if data.shape[1] != data.shape[2]:
            errors.append("Inference-ready connectivity matrices must be square.")

    # Check atlas matching if atlas is attached
    if dataset.atlas is not None:
        expected_nodes = len(dataset.atlas)
        actual_nodes = summary.get("n_rois")
        if actual_nodes is not None and actual_nodes != expected_nodes:
            errors.append(f"Atlas/Node count mismatch: atlas has {expected_nodes} labels, but data has {actual_nodes} nodes.")

    summary["n_observations"] = n_observations
    summary["data_kind"] = data_kind
    summary["ok"] = len(errors) == 0

    return DataValidationReport(
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        summary=summary
    )
