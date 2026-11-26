"""Unit tests for the Streamlit folder inspection & manifest wizard helper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from apps.utils.manifest_wizard import inspect_and_prepare_manifest
from conninfpy.loaders.manifest import DatasetManifest, ManifestLoader


def test_inspect_nonexistent_and_empty_folder(tmp_path: Path) -> None:
    missing = inspect_and_prepare_manifest(tmp_path / "does_not_exist")
    assert missing.folder_exists is False
    assert missing.ready_for_manifest is False
    assert any(item.status == "error" for item in missing.checklist)

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    empty = inspect_and_prepare_manifest(empty_dir)
    assert empty.folder_exists is True
    assert empty.ready_for_manifest is False
    assert any(item.status == "error" and "run_fmriprep_workflow.py --synthetic" in (item.guidance or "") for item in empty.checklist)


def test_inspect_complete_fmriprep_folder(tmp_path: Path) -> None:
    deriv = tmp_path / "bids" / "derivatives" / "fmriprep"
    func1 = deriv / "sub-01" / "func"
    func2 = deriv / "sub-02" / "func"
    func1.mkdir(parents=True)
    func2.mkdir(parents=True)

    (func1 / "sub-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz").write_bytes(b"bold1")
    (func1 / "sub-01_task-rest_desc-confounds_timeseries.tsv").write_text("trans_x\n0.1\n", encoding="utf-8")
    (func2 / "sub-02_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz").write_bytes(b"bold2")
    (func2 / "sub-02_task-rest_desc-confounds_timeseries.tsv").write_text("trans_x\n0.2\n", encoding="utf-8")

    pheno = tmp_path / "bids" / "participants.tsv"
    pheno.write_text("participant_id\tgroup\nsub-01\tHC\nsub-02\tPT\n", encoding="utf-8")

    res = inspect_and_prepare_manifest(deriv, use_builtin_bna=True)
    assert res.folder_exists is True
    assert res.detected_loader == "FmriprepDerivativesLoader"
    assert res.n_subjects == 2
    assert res.ready_for_manifest is True
    assert res.manifest_yaml is not None

    parsed = yaml.safe_load(res.manifest_yaml)
    manifest = DatasetManifest(parsed, "manifest.yaml")
    assert manifest.loader == "FmriprepDerivativesLoader"
    assert "derivatives_dir" in manifest.paths
    assert manifest.paths["atlas_image_path"] == "BUILTIN:bna_246_2mm"
    assert manifest.paths["atlas_metadata"] == "BUILTIN:bna246.csv"

    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text(res.manifest_yaml, encoding="utf-8")
    ml = ManifestLoader(str(manifest_file))
    assert ml.manifest.loader == "FmriprepDerivativesLoader"


def test_inspect_incomplete_fmriprep_folder_reports_missing_and_guidance(tmp_path: Path) -> None:
    deriv = tmp_path / "fmriprep_incomplete"
    func1 = deriv / "sub-01" / "func"
    func1.mkdir(parents=True)
    (func1 / "sub-01_task-rest_desc-preproc_bold.nii.gz").write_bytes(b"bold1")

    res = inspect_and_prepare_manifest(deriv, use_builtin_bna=False)
    assert res.folder_exists is True
    assert res.detected_loader == "FmriprepDerivativesLoader"
    assert res.ready_for_manifest is True  # falls back to BUILTIN:bna_246_2mm

    by_label = {item.label: item for item in res.checklist}
    assert by_label["fMRIPrep confounds"].status == "warning"
    assert "run_fmriprep_workflow.py --synthetic" in (by_label["fMRIPrep confounds"].guidance or "")
    assert by_label["Phenotype table"].status == "warning"
    assert "participant_id" in (by_label["Phenotype table"].guidance or "")
    assert by_label["Volumetric atlas NIfTI"].status == "warning"
    assert "fetch_atlas_schaefer_2018" in (by_label["Volumetric atlas NIfTI"].guidance or "")


def test_inspect_connectivity_matrix_folder(tmp_path: Path) -> None:
    mat_dir = tmp_path / "matrices"
    mat_dir.mkdir()
    rng = np.random.default_rng(0)
    for idx in range(3):
        a = rng.standard_normal((10, 10))
        sym = (a + a.T) / 2.0
        np.fill_diagonal(sym, 1.0)
        sym = np.clip(sym, -0.99, 0.99)
        np.fill_diagonal(sym, 1.0)
        np.save(mat_dir / f"sub-{idx:02d}.npy", sym)

    pheno = mat_dir / "pheno.csv"
    pheno.write_text("subject_id,group\nsub-00,HC\nsub-01,HC\nsub-02,PT\n", encoding="utf-8")

    manifest_dest = tmp_path / "conn_manifest.yaml"
    res = inspect_and_prepare_manifest(mat_dir, out_manifest_path=manifest_dest)
    assert res.folder_exists is True
    assert res.detected_loader == "ConnectivityMatrixLoader"
    assert res.n_subjects == 3
    assert res.n_rois == 10
    assert res.ready_for_manifest is True
    assert res.manifest_yaml is not None

    manifest_dest.write_text(res.manifest_yaml, encoding="utf-8")
    ml = ManifestLoader(str(manifest_dest))
    assert ml.manifest.loader == "ConnectivityMatrixLoader"
    assert ml.manifest.checks.get("expected_rois") == 10
