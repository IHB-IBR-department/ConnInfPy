"""Custom atlas readiness validation, CSV template, and NIfTI+LUT loading helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

from conninfpy.app.utils.manifest_wizard import ChecklistItem
from conninfpy.atlas import AtlasInfo

_CUSTOM_ATLAS_CSV_TEMPLATE = """id,name,network,x,y,z,hemisphere
1,LH_Vis_1,Visual,-18,-86,-4,L
2,RH_Vis_1,Visual,20,-84,-2,R
3,LH_SomMot_1,Somatomotor,-38,-22,54,L
4,RH_SomMot_1,Somatomotor,40,-20,52,R
5,LH_Default_PFC_1,Default,-8,52,18,L
6,RH_Default_PCC_1,Default,6,-54,26,R
"""


def custom_atlas_csv_template() -> str:
    """Return a valid, copy-pasteable CSV template for custom ROI atlases."""
    return _CUSTOM_ATLAS_CSV_TEMPLATE


def _has_usable_networks(atlas: AtlasInfo) -> bool:
    raw_nets = getattr(atlas, "networks", None)
    if raw_nets is None:
        return False
    nets = [str(x).strip() for x in raw_nets]
    labels = [str(x).strip() for x in getattr(atlas, "labels", [])]
    if not nets or not any(nets):
        return False
    if set(nets) <= {"", "Unknown", "unknown", "unassigned"}:
        return False
    if len(nets) == len(labels) and nets == labels:
        return False
    return True


def _has_finite_coords(atlas: AtlasInfo) -> bool:
    coords = getattr(atlas, "coords", None)
    if coords is None:
        return False
    arr = np.asarray(coords)
    if arr.ndim != 2 or arr.shape != (len(atlas), 3):
        return False
    return bool(np.isfinite(arr.astype(float)).all())


def inspect_atlas_readiness(
    atlas: Optional[AtlasInfo],
    *,
    n_nodes: Optional[int] = None,
) -> list[ChecklistItem]:
    """Inspect an ``AtlasInfo`` and report readiness for downstream analyses."""
    if atlas is None:
        return [
            ChecklistItem(
                label="Atlas metadata",
                status="warning",
                detail="No atlas metadata loaded.",
                guidance="Load a bundled atlas, custom ROI CSV, or 3D integer NIfTI to enable labeled edge tables, network summaries, and anatomical annotation.",
            )
        ]

    checklist: list[ChecklistItem] = []

    if n_nodes is not None:
        n_atlas = len(atlas)
        if n_atlas == int(n_nodes):
            checklist.append(
                ChecklistItem(
                    label="ROI count vs connectivity matrix",
                    status="ok",
                    detail=f"{n_atlas} ROIs match connectivity matrix ({n_nodes} nodes).",
                )
            )
        else:
            checklist.append(
                ChecklistItem(
                    label="ROI count vs connectivity matrix",
                    status="error",
                    detail=f"Atlas has {n_atlas} ROIs, but connectivity matrix has {n_nodes} nodes.",
                    guidance="Ensure custom atlas rows/labels match the connectivity matrix node order 0..N-1.",
                )
            )

    if _has_usable_networks(atlas):
        n_nets = len({str(x).strip() for x in atlas.networks if str(x).strip()})
        checklist.append(
            ChecklistItem(
                label="Network labels",
                status="ok",
                detail=f"{n_nets} network(s) defined.",
            )
        )
    else:
        checklist.append(
            ChecklistItem(
                label="Network labels",
                status="warning",
                detail="No usable network labels (missing or per-ROI-unique).",
                guidance="network_summary, cNBS, NI-TFNBS, and FBC-TFNBS need a 'network' column (in CSV or LUT) or a networks= mapping.",
            )
        )

    if _has_finite_coords(atlas):
        checklist.append(
            ChecklistItem(
                label="MNI coordinates",
                status="ok",
                detail=f"Finite (N={len(atlas)}, 3) MNI coordinates available.",
            )
        )
    else:
        checklist.append(
            ChecklistItem(
                label="MNI coordinates",
                status="warning",
                detail="MNI coordinates (x, y, z) are missing or incomplete.",
                guidance="Brainnetome sphere annotation (annotate_rois(..., support='sphere')) and distance/glass-brain plots need x, y, z in MNI mm (or a 3D NIfTI whose centroids are computed automatically).",
            )
        )

    if getattr(atlas, "image", None) is not None:
        checklist.append(
            ChecklistItem(
                label="3D label image",
                status="ok",
                detail="3D integer label volume attached (enables support='parcel' Brainnetome + SUIT annotation and fMRIPrep ROI extraction).",
            )
        )
    else:
        checklist.append(
            ChecklistItem(
                label="3D label image",
                status="warning",
                detail="No 3D label volume attached.",
                guidance="CSV and coordinate atlases work with support='sphere'; exact support='parcel' Brainnetome + SUIT annotation and fMRIPrep ROI extraction need a 3D integer NIfTI loaded via AtlasInfo.from_nifti().",
            )
        )

    return checklist


def _suffix_for_nifti(name: Optional[str], raw: bytes) -> str:
    lower = (name or "").lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    if lower.endswith(".nii"):
        return ".nii"
    if raw[:2] == b"\x1f\x8b":
        return ".nii.gz"
    return ".nii"


def _suffix_for_lut(name: Optional[str]) -> str:
    lower = (name or "").lower()
    for ext in (".csv", ".tsv", ".txt", ".lut"):
        if lower.endswith(ext):
            return ext
    return ".txt"


def _materialize_temp_file(data: bytes, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        return Path(tmp.name)


def load_custom_atlas_nifti(
    nifti_path_or_bytes: Any,
    *,
    lut_path_or_buffer: Any = None,
    source: str = "Custom NIfTI atlas",
) -> AtlasInfo:
    """Load a 3D integer label NIfTI (with optional LUT) into an ``AtlasInfo``.

    Accepts a filesystem path, raw bytes, or a file-like upload buffer.
    Rejects 4D probabilistic images with a clear ``ValueError``.
    """
    import nibabel as nib

    temp_paths: list[Path] = []
    try:
        if isinstance(nifti_path_or_bytes, (str, Path)):
            nifti_path = Path(str(nifti_path_or_bytes).strip()).expanduser()
            if not nifti_path.is_file():
                raise FileNotFoundError(f"NIfTI atlas file not found: {nifti_path}")
            nib_img = nib.load(str(nifti_path))
        else:
            if isinstance(nifti_path_or_bytes, (bytes, bytearray, memoryview)):
                raw = bytes(nifti_path_or_bytes)
                name = None
            elif hasattr(nifti_path_or_bytes, "read"):
                if hasattr(nifti_path_or_bytes, "seek"):
                    nifti_path_or_bytes.seek(0)
                raw = nifti_path_or_bytes.read()
                name = getattr(nifti_path_or_bytes, "name", None)
            else:
                raise TypeError("nifti_path_or_bytes must be a file path, bytes, or file-like object.")
            if not raw:
                raise ValueError("Uploaded NIfTI file is empty.")
            tmp_nii = _materialize_temp_file(raw, _suffix_for_nifti(name, raw))
            temp_paths.append(tmp_nii)
            nib_img = nib.load(str(tmp_nii))

        data = np.asarray(nib_img.dataobj)
        if data.ndim == 4 and data.shape[3] == 1:
            data = data[..., 0]
        if data.ndim != 3:
            raise ValueError(
                "AtlasInfo.from_nifti() requires a 3D integer label volume "
                "(0 = background, non-zero integer ROI IDs in ascending order); "
                f"got shape {data.shape}. Probabilistic 4D maps are not label masks — "
                "convert to 3D integer labels or supply an MNI coordinate CSV."
            )
        # Keep an in-memory image so temporary upload files can be safely removed.
        mem_img = nib.Nifti1Image(np.array(data, copy=True), nib_img.affine, nib_img.header)

        lut_arg: Any = None
        if lut_path_or_buffer is not None and lut_path_or_buffer != "":
            if isinstance(lut_path_or_buffer, (str, Path)):
                lut_str = str(lut_path_or_buffer).strip()
                if lut_str:
                    lut_path = Path(lut_str).expanduser()
                    if not lut_path.is_file():
                        raise FileNotFoundError(f"LUT file not found: {lut_path}")
                    lut_arg = lut_path
            elif isinstance(lut_path_or_buffer, (bytes, bytearray, memoryview)):
                tmp_lut = _materialize_temp_file(bytes(lut_path_or_buffer), ".csv")
                temp_paths.append(tmp_lut)
                lut_arg = tmp_lut
            elif hasattr(lut_path_or_buffer, "read"):
                if hasattr(lut_path_or_buffer, "seek"):
                    lut_path_or_buffer.seek(0)
                lut_raw = lut_path_or_buffer.read()
                if isinstance(lut_raw, str):
                    lut_raw = lut_raw.encode("utf-8")
                if lut_raw:
                    lut_name = getattr(lut_path_or_buffer, "name", None)
                    tmp_lut = _materialize_temp_file(lut_raw, _suffix_for_lut(lut_name))
                    temp_paths.append(tmp_lut)
                    lut_arg = tmp_lut
            else:
                lut_arg = lut_path_or_buffer

        return AtlasInfo.from_nifti(mem_img, lut=lut_arg, source=source)
    finally:
        for p in temp_paths:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
