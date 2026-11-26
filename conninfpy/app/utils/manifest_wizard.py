"""Folder inspection and DatasetManifest v1 builder helpers for the Streamlit app."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import yaml

from conninfpy._cli_tools import (
    _infer_matrix_kind,
    _inspect_tabular_sample,
    _rel_manifest_path,
    init_manifest,
)

ChecklistStatus = Literal["ok", "warning", "error"]


@dataclass
class ChecklistItem:
    """Single readiness check item for folder inspection."""

    label: str
    status: ChecklistStatus
    detail: str
    guidance: Optional[str] = None


@dataclass
class FolderInspectionResult:
    """Result of inspecting a study data folder and preparing a DatasetManifest."""

    folder_exists: bool
    detected_loader: Optional[str] = None
    data_kind: Optional[str] = None
    n_subjects: int = 0
    n_rois: Optional[int] = None
    bold_files: list[str] = field(default_factory=list)
    confound_files: list[str] = field(default_factory=list)
    pheno_path: Optional[str] = None
    atlas_image_path: Optional[str] = None
    atlas_metadata: Optional[str] = None
    nilearn_available: bool = False
    ready_for_manifest: bool = False
    checklist: list[ChecklistItem] = field(default_factory=list)
    manifest_yaml: Optional[str] = None
    manifest_dict: Optional[dict[str, Any]] = None


def _auto_detect_pheno(data_dir: Path, explicit_pheno: Optional[str]) -> tuple[Optional[Path], bool, Optional[str]]:
    """Resolve explicit or auto-detected phenotype table.

    Returns ``(resolved_path, was_explicit, error_message)``.
    """
    if explicit_pheno and explicit_pheno.strip():
        p = Path(explicit_pheno.strip()).expanduser()
        if p.is_file():
            return p.resolve(), True, None
        return None, True, f"Phenotype file not found: {p}"

    candidates = [
        data_dir / "participants.tsv",
        data_dir / "participants.csv",
        data_dir / "pheno.tsv",
        data_dir / "pheno.csv",
        data_dir.parent / "participants.tsv",
        data_dir.parent / "participants.csv",
        data_dir.parent.parent / "participants.tsv",
        data_dir.parent.parent / "participants.csv",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve(), False, None
    return None, False, None


def inspect_and_prepare_manifest(
    data_dir: str | Path,
    *,
    pheno_path: Optional[str] = None,
    atlas_image_path: Optional[str] = None,
    use_builtin_bna: bool = False,
    dataset_name: Optional[str] = None,
    out_manifest_path: Optional[str | Path] = None,
) -> FolderInspectionResult:
    """Inspect a study folder and generate a v1 ``DatasetManifest`` YAML when possible."""
    checklist: list[ChecklistItem] = []
    nilearn_ok = importlib.util.find_spec("nilearn") is not None

    raw_dir = str(data_dir or "").strip()
    if not raw_dir:
        checklist.append(
            ChecklistItem(
                label="Data folder",
                status="error",
                detail="No data folder path provided.",
                guidance="Enter a path to an fMRIPrep derivatives directory or a folder of `.npy`/`.csv`/`.tsv` files.",
            )
        )
        return FolderInspectionResult(folder_exists=False, nilearn_available=nilearn_ok, checklist=checklist)

    dd_p = Path(raw_dir).expanduser()
    if not dd_p.is_dir():
        checklist.append(
            ChecklistItem(
                label="Data folder",
                status="error",
                detail=f"Directory not found: {dd_p}",
                guidance=(
                    "Verify the folder path exists on disk, or run "
                    "`python examples/fmriprep_nilearn/run_fmriprep_workflow.py --synthetic` "
                    "to generate a small synthetic fMRIPrep layout for testing."
                ),
            )
        )
        return FolderInspectionResult(folder_exists=False, nilearn_available=nilearn_ok, checklist=checklist)

    dd_resolved = dd_p.resolve()
    checklist.append(ChecklistItem(label="Data folder", status="ok", detail=str(dd_resolved)))

    base_dir = Path(out_manifest_path).resolve().parent if out_manifest_path is not None else None
    resolved_pheno, pheno_explicit, pheno_err = _auto_detect_pheno(dd_resolved, pheno_path)
    if pheno_err:
        checklist.append(
            ChecklistItem(
                label="Phenotype table",
                status="warning",
                detail=pheno_err,
                guidance=(
                    "Provide a valid `participants.tsv` / `pheno.csv` with `participant_id` or `subject_id` "
                    "(plus group/age/covariates; BIDS `n/a` values are supported), or leave blank if using group subfolders."
                ),
            )
        )
    elif resolved_pheno is not None:
        src = "provided" if pheno_explicit else "auto-detected"
        checklist.append(
            ChecklistItem(
                label="Phenotype table",
                status="ok",
                detail=f"{resolved_pheno.name} ({src})",
            )
        )
    else:
        checklist.append(
            ChecklistItem(
                label="Phenotype table",
                status="warning",
                detail="No `participants.tsv` or `pheno.csv` found.",
                guidance=(
                    "Expected columns: `participant_id` or `subject_id` plus group/age/covariates "
                    "(BIDS `n/a` supported). Optional if data are organized into group subfolders."
                ),
            )
        )

    # 1. Check for fMRIPrep derivatives
    bold_files = sorted(dd_resolved.rglob("*desc-preproc_bold.nii.gz"))
    if bold_files:
        confound_files = sorted(dd_resolved.rglob("*confounds*.tsv"))
        checklist.append(
            ChecklistItem(
                label="fMRIPrep BOLD files",
                status="ok",
                detail=f"Found {len(bold_files)} `*_desc-preproc_bold.nii.gz` file(s).",
            )
        )
        if confound_files and len(confound_files) >= len(bold_files):
            checklist.append(
                ChecklistItem(
                    label="fMRIPrep confounds",
                    status="ok",
                    detail=f"Found {len(confound_files)} confound TSV file(s).",
                )
            )
        elif confound_files:
            checklist.append(
                ChecklistItem(
                    label="fMRIPrep confounds",
                    status="warning",
                    detail=f"Found {len(confound_files)} confound TSV(s) for {len(bold_files)} BOLD file(s).",
                    guidance=(
                        "Some subjects may be missing `*_desc-confounds_timeseries.tsv` next to their BOLD file "
                        "under `sub-*/func/`."
                    ),
                )
            )
        else:
            checklist.append(
                ChecklistItem(
                    label="fMRIPrep confounds",
                    status="warning",
                    detail="No `*_desc-confounds_timeseries.tsv` files found.",
                    guidance=(
                        "Expected BIDS derivatives layout: `sub-*/func/*_desc-preproc_bold.nii.gz` and "
                        "`*_desc-confounds_timeseries.tsv`. For a quick test layout, run "
                        "`python examples/fmriprep_nilearn/run_fmriprep_workflow.py --synthetic`."
                    ),
                )
            )

        # Atlas NIfTI resolution for fMRIPrep
        atlas_img_ref: Optional[str] = None
        atlas_meta_ref: Optional[str] = None
        n_rois: Optional[int] = None
        raw_atlas = str(atlas_image_path or "").strip()

        if use_builtin_bna:
            atlas_img_ref = "BUILTIN:bna_246_2mm"
            atlas_meta_ref = "BUILTIN:bna246.csv"
            n_rois = 246
            checklist.append(
                ChecklistItem(
                    label="Volumetric atlas NIfTI",
                    status="ok",
                    detail="Using built-in Brainnetome-246 (`BUILTIN:bna_246_2mm` + `BUILTIN:bna246.csv`).",
                )
            )
        elif raw_atlas:
            if raw_atlas.startswith("BUILTIN:"):
                atlas_img_ref = raw_atlas
                if raw_atlas == "BUILTIN:bna_246_2mm":
                    atlas_meta_ref = "BUILTIN:bna246.csv"
                    n_rois = 246
                checklist.append(ChecklistItem(label="Volumetric atlas NIfTI", status="ok", detail=raw_atlas))
            else:
                ap = Path(raw_atlas).expanduser()
                if ap.is_file():
                    atlas_img_ref = _rel_manifest_path(ap.resolve(), base_dir)
                    checklist.append(
                        ChecklistItem(label="Volumetric atlas NIfTI", status="ok", detail=str(ap.resolve()))
                    )
                else:
                    checklist.append(
                        ChecklistItem(
                            label="Volumetric atlas NIfTI",
                            status="error",
                            detail=f"Atlas NIfTI not found: {ap}",
                            guidance=(
                                "Check the NIfTI path, select built-in `BUILTIN:bna_246_2mm`, or fetch "
                                "Schaefer-100 / Craddock-200 via `nilearn.datasets.fetch_atlas_schaefer_2018()` / "
                                "`fetch_atlas_craddock_2012()` (see `examples/fmriprep_nilearn/`)."
                            ),
                        )
                    )
        else:
            atlas_img_ref = "BUILTIN:bna_246_2mm"
            atlas_meta_ref = "BUILTIN:bna246.csv"
            n_rois = 246
            checklist.append(
                ChecklistItem(
                    label="Volumetric atlas NIfTI",
                    status="warning",
                    detail="No custom atlas NIfTI provided; defaulting to `BUILTIN:bna_246_2mm` (`BUILTIN:bna246.csv`).",
                    guidance=(
                        "You can use the built-in Brainnetome-246 atlas right away, or provide a custom 3D parcel "
                        "NIfTI fetched via `nilearn.datasets.fetch_atlas_schaefer_2018()` / "
                        "`fetch_atlas_craddock_2012()` (see `examples/fmriprep_nilearn/`)."
                    ),
                )
            )

        if not nilearn_ok:
            checklist.append(
                ChecklistItem(
                    label="nilearn dependency",
                    status="warning",
                    detail="`nilearn` is not installed in this environment (required for fMRIPrep ROI extraction).",
                    guidance='Install with `pip install "conninfpy[app]"` or `pip install nilearn`.',
                )
            )
        else:
            checklist.append(
                ChecklistItem(label="nilearn dependency", status="ok", detail="`nilearn` is available.")
            )

        has_atlas_error = any(c.label == "Volumetric atlas NIfTI" and c.status == "error" for c in checklist)
        if has_atlas_error or atlas_img_ref is None:
            return FolderInspectionResult(
                folder_exists=True,
                detected_loader="FmriprepDerivativesLoader",
                data_kind="timeseries",
                n_subjects=len(bold_files),
                bold_files=[str(p) for p in bold_files],
                confound_files=[str(p) for p in confound_files],
                pheno_path=str(resolved_pheno) if resolved_pheno else None,
                nilearn_available=nilearn_ok,
                ready_for_manifest=False,
                checklist=checklist,
            )

        paths: dict[str, Any] = {
            "derivatives_dir": _rel_manifest_path(dd_resolved, base_dir),
        }
        if resolved_pheno is not None:
            paths["pheno_path"] = _rel_manifest_path(resolved_pheno, base_dir)
        paths["atlas_image_path"] = atlas_img_ref
        if atlas_meta_ref is not None:
            paths["atlas_metadata"] = atlas_meta_ref

        checks: dict[str, Any] = {"expected_data_kind": "timeseries"}
        if n_rois is not None:
            checks["expected_rois"] = n_rois

        manifest_dict: dict[str, Any] = {
            "schema_version": 1,
            "name": (dataset_name or "").strip() or dd_resolved.name,
            "loader": "FmriprepDerivativesLoader",
            "paths": paths,
            "params": {
                "bold_pattern": "sub-*/func/*desc-preproc_bold.nii.gz",
                "confounds_pattern": "sub-*/func/*desc-confounds_timeseries.tsv",
                "subject_regex": "sub-([^_/]+)",
                "denoising": {
                    "strategy": 4,
                    "n_compcor": 10,
                    "use_GSR": False,
                },
            },
            "checks": checks,
        }
        header = (
            "# ConnInfPy dataset manifest (loader=FmriprepDerivativesLoader)\n"
            f"# Detected: bold_files={len(bold_files)}, confounds={len(confound_files)}\n"
        )
        manifest_yaml = header + yaml.safe_dump(manifest_dict, sort_keys=False)

        return FolderInspectionResult(
            folder_exists=True,
            detected_loader="FmriprepDerivativesLoader",
            data_kind="timeseries",
            n_subjects=len(bold_files),
            n_rois=n_rois,
            bold_files=[str(p) for p in bold_files],
            confound_files=[str(p) for p in confound_files],
            pheno_path=str(resolved_pheno) if resolved_pheno else None,
            atlas_image_path=atlas_img_ref,
            atlas_metadata=atlas_meta_ref,
            nilearn_available=nilearn_ok,
            ready_for_manifest=True,
            checklist=checklist,
            manifest_yaml=manifest_yaml,
            manifest_dict=manifest_dict,
        )

    # 2. Check for connectivity matrices or timeseries files (.npy, .tsv, .csv)
    pheno_abs = resolved_pheno.resolve() if resolved_pheno is not None else None
    chosen_ext: Optional[str] = None
    matched_files: list[Path] = []
    for ext in (".npy", ".tsv", ".csv"):
        cands = sorted(dd_resolved.glob(f"*{ext}"))
        if not cands:
            cands = sorted(dd_resolved.rglob(f"*{ext}"))
        cands = [
            p
            for p in cands
            if (pheno_abs is None or p.resolve() != pheno_abs)
            and p.name.lower() not in ("participants.tsv", "participants.csv", "pheno.tsv", "pheno.csv")
        ]
        if cands:
            chosen_ext = ext
            matched_files = cands
            break

    if not matched_files or chosen_ext is None:
        checklist.append(
            ChecklistItem(
                label="Data files",
                status="error",
                detail=f"No supported neuroimaging files found in {dd_resolved}.",
                guidance=(
                    "Expected either fMRIPrep derivatives (`sub-*/func/*_desc-preproc_bold.nii.gz` & "
                    "`*_desc-confounds_timeseries.tsv`) or per-subject `.npy` / `.csv` / `.tsv` connectivity "
                    "matrices or ROI time series. To generate a synthetic fMRIPrep test folder, run "
                    "`python examples/fmriprep_nilearn/run_fmriprep_workflow.py --synthetic`."
                ),
            )
        )
        return FolderInspectionResult(
            folder_exists=True,
            pheno_path=str(resolved_pheno) if resolved_pheno else None,
            nilearn_available=nilearn_ok,
            ready_for_manifest=False,
            checklist=checklist,
        )

    sample = matched_files[0]
    try:
        if chosen_ext == ".npy":
            arr = np.asarray(np.load(sample), dtype=np.float64)
        else:
            sep = "\t" if chosen_ext == ".tsv" else ","
            arr, _ = _inspect_tabular_sample(sample, sep=sep)
    except Exception as exc:
        checklist.append(
            ChecklistItem(
                label="Data files",
                status="error",
                detail=f"Could not parse sample file `{sample.name}`: {exc}",
                guidance="Ensure each `.npy`, `.csv`, or `.tsv` file contains a numeric 2D array (ROIs×ROIs or T×ROIs).",
            )
        )
        return FolderInspectionResult(
            folder_exists=True,
            pheno_path=str(resolved_pheno) if resolved_pheno else None,
            nilearn_available=nilearn_ok,
            ready_for_manifest=False,
            checklist=checklist,
        )

    if arr.ndim != 2:
        checklist.append(
            ChecklistItem(
                label="Data files",
                status="error",
                detail=f"Expected a 2D array in `{sample.name}`, got shape {arr.shape}.",
                guidance="Per-subject files should be 2D: `(ROIs, ROIs)` for matrices or `(timepoints, ROIs)` for time series.",
            )
        )
        return FolderInspectionResult(
            folder_exists=True,
            pheno_path=str(resolved_pheno) if resolved_pheno else None,
            nilearn_available=nilearn_ok,
            ready_for_manifest=False,
            checklist=checklist,
        )

    n_rois = int(arr.shape[1])
    is_square = arr.shape[0] == arr.shape[1]
    is_sym = is_square and bool(np.allclose(arr, arr.T, atol=1e-4, equal_nan=True))
    if is_square and is_sym:
        detected_loader = "ConnectivityMatrixLoader"
        data_kind = _infer_matrix_kind(arr)
    else:
        detected_loader = "TimeseriesDirectoryLoader"
        data_kind = "timeseries"

    checklist.append(
        ChecklistItem(
            label="Data files",
            status="ok",
            detail=f"Found {len(matched_files)} `{chosen_ext}` file(s) ({detected_loader}, {n_rois} ROIs, `{data_kind}`).",
        )
    )

    atlas_arg: Optional[str] = "bna246.csv" if use_builtin_bna else None
    manifest_yaml = init_manifest(
        data_dir=dd_resolved,
        pheno_path=resolved_pheno,
        out_path=None,
        kind="auto",
        atlas=atlas_arg,
        name=(dataset_name or "").strip() or dd_resolved.name,
    )
    # Strip the init_manifest stdout print side-effect? init_manifest prints only when out_path is None,
    # which is harmless, and returns the YAML string.
    manifest_dict = yaml.safe_load(manifest_yaml)
    if out_manifest_path is not None and isinstance(manifest_dict, dict):
        # Re-relativize paths if out_manifest_path was specified without writing to disk yet
        paths_block = manifest_dict.get("paths", {})
        if "dir_path" in paths_block:
            paths_block["dir_path"] = _rel_manifest_path(dd_resolved, base_dir)
        if "pheno_path" in paths_block and resolved_pheno is not None:
            paths_block["pheno_path"] = _rel_manifest_path(resolved_pheno, base_dir)
        manifest_yaml = yaml.safe_dump(manifest_dict, sort_keys=False)

    return FolderInspectionResult(
        folder_exists=True,
        detected_loader=detected_loader,
        data_kind=data_kind,
        n_subjects=len(matched_files),
        n_rois=n_rois,
        pheno_path=str(resolved_pheno) if resolved_pheno else None,
        atlas_metadata="BUILTIN:bna246.csv" if use_builtin_bna else None,
        nilearn_available=nilearn_ok,
        ready_for_manifest=True,
        checklist=checklist,
        manifest_yaml=manifest_yaml,
        manifest_dict=manifest_dict,
    )
