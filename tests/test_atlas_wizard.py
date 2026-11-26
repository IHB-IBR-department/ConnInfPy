"""Unit tests for custom atlas readiness validation, CSV template, and NIfTI+LUT loading."""

from __future__ import annotations

import io
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from apps.utils.atlas_wizard import (
    custom_atlas_csv_template,
    inspect_atlas_readiness,
    load_custom_atlas_nifti,
)
from apps.utils.helpers import load_custom_atlas_csv
from conninfpy.atlas import AtlasInfo


def _make_3d_label_nifti() -> nib.Nifti1Image:
    data = np.zeros((6, 6, 6), dtype=np.int32)
    data[1:3, 1:3, 1:3] = 1
    data[3:5, 3:5, 3:5] = 2
    affine = np.eye(4)
    return nib.Nifti1Image(data, affine)


def test_custom_atlas_csv_template_parses_cleanly(tmp_path: Path) -> None:
    tmpl = custom_atlas_csv_template()
    assert tmpl.splitlines()[0] == "id,name,network,x,y,z,hemisphere"

    csv_path = tmp_path / "template.csv"
    csv_path.write_text(tmpl, encoding="utf-8")

    atlas_core = AtlasInfo.from_csv(csv_path)
    atlas_app = load_custom_atlas_csv(io.StringIO(tmpl))

    assert len(atlas_core) == 6
    assert len(atlas_app) == 6
    assert atlas_core.labels == atlas_app.labels
    assert atlas_core.networks == atlas_app.networks
    assert atlas_core.coords is not None and atlas_core.coords.shape == (6, 3)


def test_load_custom_atlas_nifti_with_and_without_lut(tmp_path: Path) -> None:
    nii_path = tmp_path / "labels_3d.nii.gz"
    nib.save(_make_3d_label_nifti(), nii_path)

    atlas_no_lut = load_custom_atlas_nifti(nii_path)
    assert len(atlas_no_lut) == 2
    assert atlas_no_lut.labels == ["region_1", "region_2"]
    assert atlas_no_lut.coords is not None and atlas_no_lut.coords.shape == (2, 3)
    assert atlas_no_lut.image is not None

    lut_path = tmp_path / "lut.csv"
    lut_path.write_text("id,name,network\n1,ROI_A,Visual\n2,ROI_B,Default\n", encoding="utf-8")
    atlas_with_lut = load_custom_atlas_nifti(nii_path, lut_path_or_buffer=lut_path)
    assert atlas_with_lut.labels == ["ROI_A", "ROI_B"]
    assert atlas_with_lut.networks == ["Visual", "Default"]
    assert atlas_with_lut.image is not None

    # Also test uploaded file-like buffer inputs
    nii_buf = io.BytesIO(nii_path.read_bytes())
    nii_buf.name = "labels_3d.nii.gz"
    lut_buf = io.BytesIO(lut_path.read_bytes())
    lut_buf.name = "lut.csv"
    atlas_buf = load_custom_atlas_nifti(nii_buf, lut_path_or_buffer=lut_buf)
    assert atlas_buf.labels == ["ROI_A", "ROI_B"]
    assert atlas_buf.networks == ["Visual", "Default"]
    assert np.asarray(atlas_buf.image.dataobj).shape == (6, 6, 6)


def test_load_custom_atlas_nifti_rejects_4d_image(tmp_path: Path) -> None:
    data_4d = np.zeros((4, 4, 4, 3), dtype=np.float32)
    img_4d = nib.Nifti1Image(data_4d, np.eye(4))
    nii_4d_path = tmp_path / "prob_4d.nii.gz"
    nib.save(img_4d, nii_4d_path)

    with pytest.raises(ValueError, match="3D integer label volume"):
        load_custom_atlas_nifti(nii_4d_path)


def test_inspect_atlas_readiness_flags_all_conditions(tmp_path: Path) -> None:
    # 1. Complete NIfTI + LUT atlas with matching n_nodes
    nii_path = tmp_path / "labels.nii.gz"
    nib.save(_make_3d_label_nifti(), nii_path)
    lut_path = tmp_path / "lut.csv"
    lut_path.write_text("id,name,network\n1,ROI_1,Visual\n2,ROI_2,Default\n", encoding="utf-8")
    full_atlas = load_custom_atlas_nifti(nii_path, lut_path_or_buffer=lut_path)

    items_ok = {item.label: item for item in inspect_atlas_readiness(full_atlas, n_nodes=2)}
    assert items_ok["ROI count vs connectivity matrix"].status == "ok"
    assert items_ok["Network labels"].status == "ok"
    assert items_ok["MNI coordinates"].status == "ok"
    assert items_ok["3D label image"].status == "ok"

    # 2. Mismatched n_nodes
    items_mismatch = {item.label: item for item in inspect_atlas_readiness(full_atlas, n_nodes=10)}
    assert items_mismatch["ROI count vs connectivity matrix"].status == "error"

    # 3. NIfTI without LUT (per-ROI-unique networks)
    no_lut_atlas = load_custom_atlas_nifti(nii_path)
    items_no_lut = {item.label: item for item in inspect_atlas_readiness(no_lut_atlas, n_nodes=2)}
    assert items_no_lut["Network labels"].status == "warning"

    # 4. CSV-like atlas with Unknown networks, no coords, no 3D image
    minimal_atlas = AtlasInfo(labels=["A", "B"], networks=["Unknown", "Unknown"], coords=None)
    items_min = {item.label: item for item in inspect_atlas_readiness(minimal_atlas, n_nodes=2)}
    assert items_min["Network labels"].status == "warning"
    assert items_min["MNI coordinates"].status == "warning"
    assert items_min["3D label image"].status == "warning"
