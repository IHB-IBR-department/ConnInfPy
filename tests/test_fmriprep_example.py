"""Offline checks for the fMRIPrep + nilearn workflow example (NETSTAT-63)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from conninfpy.atlas.resources import atlas_resource_available
from conninfpy.loaders.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples" / "fmriprep_nilearn"
HAS_NILEARN = importlib.util.find_spec("nilearn") is not None
HAS_BNA_CACHE = (
    atlas_resource_available("BN_Atlas_246_1mm.nii.gz")
    and atlas_resource_available("bna246_hierarchy.csv")
)


def test_manifest_template_parses():
    manifest = load_manifest(str(EXAMPLE_DIR / "manifest_template.yaml"))
    assert manifest.schema_version == 1
    assert manifest.loader == "FmriprepDerivativesLoader"
    assert "derivatives_dir" in manifest.paths
    assert "atlas_image_path" in manifest.paths
    assert manifest.params["denoising"]["strategy"] == 4


def test_no_bundled_nifti_or_array_datasets_in_example():
    forbidden = (".nii", ".nii.gz", ".npy", ".npz")
    found = [
        p for p in EXAMPLE_DIR.rglob("*")
        if p.is_file() and any(p.name.endswith(ext) for ext in forbidden)
    ]
    assert found == [], f"Unexpected bundled data files in {EXAMPLE_DIR}: {found}"


@pytest.mark.skipif(not HAS_NILEARN, reason="nilearn not installed ([app] extra)")
@pytest.mark.skipif(not HAS_BNA_CACHE, reason="Brainnetome atlas cache not downloaded; run scripts/download_atlases.py")
def test_synthetic_fmriprep_workflow_runs_offline(tmp_path):
    script = EXAMPLE_DIR / "run_fmriprep_workflow.py"
    out_dir = tmp_path / "out"
    res = subprocess.run(
        [
            sys.executable,
            str(script),
            "--synthetic",
            "--n-subjects",
            "4",
            "--n-permutations",
            "10",
            "--out-dir",
            str(out_dir),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, res.stderr
    assert (out_dir / "manifest.yaml").is_file()
    assert (out_dir / "pheno.csv").is_file()
    assert (out_dir / "connectogram_positive.png").is_file()
    assert (out_dir / "methods_boilerplate.md").is_file()
