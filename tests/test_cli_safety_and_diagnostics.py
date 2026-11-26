"""Tests for CLI overwrite safety, run overrides, reference selftest, JSON info, and init template."""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import yaml

from conninfpy import __version__
from conninfpy.cli import ConfigError, main, run


def _write_minimal_study(
    tmp_path: Path,
    *,
    n_subjects: int = 12,
    n_rois: int = 6,
    design: dict | None = None,
    alpha: float = 0.05,
    atlas: str | None = None,
) -> Path:
    rng = np.random.default_rng(123)
    A = rng.normal(size=(n_subjects, n_rois, n_rois))
    mats = 0.5 * (A + np.swapaxes(A, 1, 2))
    idx = np.arange(n_rois)
    mats[:, idx, idx] = 1.0
    mats = np.clip(mats * 0.25, -0.95, 0.95)
    mats[:, idx, idx] = 1.0

    pheno = pd.DataFrame({
        "subject_id": [f"sub-{i:02d}" for i in range(n_subjects)],
        "age": np.linspace(20.0, 50.0, n_subjects),
        "group": ["HC" if i < n_subjects // 2 else "SZ" for i in range(n_subjects)],
    })
    data_path = tmp_path / "mats.npy"
    pheno_path = tmp_path / "pheno.csv"
    manifest_path = tmp_path / "manifest.yaml"
    np.save(data_path, mats)
    pheno.to_csv(pheno_path, index=False)
    manifest_path.write_text(
        yaml.safe_dump({
            "schema_version": 1,
            "name": "cli_safety_test",
            "loader": "NumpyLoader",
            "paths": {
                "data_path": data_path.name,
                "pheno_path": pheno_path.name,
            },
            "metadata": {"source_type": "correlation", "subject_id_column": "subject_id"},
            "params": {"data_kind": "correlation"},
        }),
        encoding="utf-8",
    )

    cfg: dict = {
        "data": manifest_path.name,
        "input": "r",
        "design": design or {"type": "glm", "interest": "age"},
        "inference": {"method": "tstat", "n_permutations": 10, "use_mp": False, "rng": 0},
        "alpha": alpha,
        "output": "out",
    }
    if atlas is not None:
        cfg["atlas"] = atlas
    cfg_path = tmp_path / "analysis.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _write_atlas_csv(path: Path, n_rois: int = 6, prefix: str = "ROI") -> Path:
    df = pd.DataFrame({
        "name": [f"{prefix}_{i}" for i in range(n_rois)],
        "network": ["NetA" if i < n_rois // 2 else "NetB" for i in range(n_rois)],
    })
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# 1. Overwrite protection and --alpha / --atlas overrides on `run`
# ---------------------------------------------------------------------------

def test_run_refuses_nonempty_output_dir_without_force(tmp_path: Path):
    cfg = _write_minimal_study(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "existing.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(ConfigError, match="--force"):
        run(cfg)

    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        rc = main(["run", str(cfg)])
    assert rc == 1
    assert "--force" in err.getvalue()
    assert (out_dir / "existing.txt").read_text(encoding="utf-8") == "keep me"


def test_run_allows_empty_existing_output_dir_and_dry_run_on_nonempty(tmp_path: Path):
    cfg = _write_minimal_study(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Empty existing directory must succeed without --force
    with contextlib.redirect_stdout(io.StringIO()):
        written = run(cfg)
    assert written["provenance"].is_file()

    # Non-empty directory must still allow --dry-run without --force
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["run", str(cfg), "--dry-run"])
    assert rc == 0
    assert "dry-run" in buf.getvalue()


def test_run_force_overwrites_nonempty_output_dir(tmp_path: Path):
    cfg = _write_minimal_study(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "old_marker.txt").write_text("old", encoding="utf-8")

    with contextlib.redirect_stdout(io.StringIO()):
        rc = main(["run", str(cfg), "--force"])
    assert rc == 0
    assert (out_dir / "provenance.json").is_file()


def test_run_alpha_and_atlas_cli_overrides(tmp_path: Path):
    atlas_a = _write_atlas_csv(tmp_path / "atlas_a.csv", prefix="AlphaROI")
    atlas_b = _write_atlas_csv(tmp_path / "atlas_b.csv", prefix="BetaROI")
    cfg = _write_minimal_study(tmp_path, alpha=0.05, atlas=atlas_a.name)

    with contextlib.redirect_stdout(io.StringIO()):
        rc = main([
            "run", str(cfg),
            "--alpha", "0.01",
            "--atlas", str(atlas_b),
            "--out", str(tmp_path / "override_out"),
        ])
    assert rc == 0

    prov = json.loads((tmp_path / "override_out" / "provenance.json").read_text(encoding="utf-8"))
    assert prov["alpha"] == pytest.approx(0.01)
    assert prov["config"]["alpha"] == pytest.approx(0.01)
    assert prov["config"]["atlas"] == str(atlas_b)

    edges_df = pd.read_csv(tmp_path / "override_out" / "edges.csv")
    if not edges_df.empty and "roi_i_name" in edges_df.columns:
        assert edges_df["roi_i_name"].str.startswith("BetaROI_").all()

    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        rc_bad = main(["run", str(cfg), "--alpha", "1.5", "--out", str(tmp_path / "bad_out")])
    assert rc_bad == 1
    assert "--alpha" in err.getvalue()


# ---------------------------------------------------------------------------
# 2. Reference selftest (`conninfpy selftest --reference`)
# ---------------------------------------------------------------------------

def test_selftest_reference_text_and_json():
    buf_txt = io.StringIO()
    with contextlib.redirect_stdout(buf_txt):
        rc_txt = main(["selftest", "--reference"])
    assert rc_txt == 0
    txt = buf_txt.getvalue()
    assert "reference_nbs_schizophrenia" in txt
    assert "reference_tfnbs_reproducibility" in txt

    buf_json = io.StringIO()
    with contextlib.redirect_stdout(buf_json):
        rc_json = main(["selftest", "--reference", "--json"])
    assert rc_json == 0
    payload = json.loads(buf_json.getvalue())
    assert payload["ok"] is True
    nbs_check = payload["checks"]["reference_nbs_schizophrenia"]
    assert nbs_check["status"] == "PASS"
    assert "40" in nbs_check["detail"]
    tfnbs_check = payload["checks"]["reference_tfnbs_reproducibility"]
    assert tfnbs_check["status"] == "PASS"
    assert "exact_match=True" in tfnbs_check["detail"]


def test_selftest_reference_skips_gracefully_when_demo_absent():
    buf = io.StringIO()
    with mock.patch("conninfpy._cli_tools._find_nbs_schizophrenia_demo", return_value=None):
        with contextlib.redirect_stdout(buf):
            rc = main(["selftest", "--reference", "--json"])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["ok"] is True
    assert payload["checks"]["reference_nbs_schizophrenia"]["status"] == "SKIP"
    assert "synthetic NBS fallback" in payload["checks"]["reference_nbs_schizophrenia"]["detail"]
    assert payload["checks"]["reference_tfnbs_reproducibility"]["status"] == "PASS"


# ---------------------------------------------------------------------------
# 3. Machine-readable `conninfpy info --json`
# ---------------------------------------------------------------------------

def test_info_json_fields_and_no_numba_cache_creation(tmp_path: Path):
    cache_dir = tmp_path / "numba_cache_should_not_exist"
    env = {**os.environ, "NUMBA_CACHE_DIR": str(cache_dir)}
    proc = subprocess.run(
        [sys.executable, "-m", "conninfpy.cli", "info", "--json"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["conninfpy_version"] == __version__
    assert isinstance(payload["python"], str) and payload["python"]
    assert isinstance(payload["platform"], str) and payload["platform"]
    assert isinstance(payload["os"], str) and payload["os"]

    deps = payload["dependencies"]
    for pkg in ("numpy", "scipy", "pandas", "numba", "nibabel", "nilearn", "streamlit"):
        assert pkg in deps

    assert payload["numba"]["imported"] is False
    assert isinstance(payload["numba"]["installed"], bool)
    assert payload["resources"]["bna_246"] is True
    assert payload["resources"]["suit_cerebellum"] is True
    assert not cache_dir.exists()


# ---------------------------------------------------------------------------
# 4. Updated `conninfpy init` YAML template
# ---------------------------------------------------------------------------

def test_init_template_contains_updated_guidance_and_is_valid_yaml(tmp_path: Path):
    _write_minimal_study(tmp_path, design={"type": "two-sample", "group": "group", "levels": ["HC", "SZ"]})
    manifest = tmp_path / "manifest.yaml"
    out_yaml = tmp_path / "generated_analysis.yaml"

    with contextlib.redirect_stdout(io.StringIO()):
        rc = main(["init", "--manifest", str(manifest), "--out", str(out_yaml)])
    assert rc == 0

    text = out_yaml.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict)
    assert parsed["inference"]["n_permutations"] == 1000

    assert "# atlas:" in text
    assert "# atlas_lut:" in text
    assert "# missing: available_case" in text
    assert "# net_labels: atlas" in text
    assert "# interpret:" in text
    assert "#   cerebellum: anatom" in text
    assert "#   methods: true" in text
    assert "bonferroni" in text
    assert "bh_fdr" in text
    assert "interest_reference" in text
    assert "5000" in text and "recommended for publication" in text
