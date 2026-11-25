"""Tests for ConnInfPy CLI extensions: qc/check, init, interpret, and app."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import yaml

from conninfpy.cli import load_config, main, run


def _make_study(tmp_path: Path, n: int = 16, n_rois: int = 6, *, collinear: bool = False) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(12)
    r = np.tanh(rng.normal(0, 0.25, (n, n_rois, n_rois)))
    r = (r + r.transpose(0, 2, 1)) / 2
    idx = np.arange(n_rois)
    r[:, idx, idx] = 1.0
    # Plant a strong positive shift with age on edge (0, 1) so interpret finds detections
    age = np.linspace(20.0, 50.0, n)
    r[:, 0, 1] = np.clip(0.02 * (age - 35.0) + rng.normal(0, 0.05, n), -0.95, 0.95)
    r[:, 1, 0] = r[:, 0, 1]

    np.save(tmp_path / "conn.npy", r)
    pheno = pd.DataFrame({
        "subject_id": [f"s{i:02d}" for i in range(n)],
        "age": age,
        "age_copy": age * 2.0 if collinear else rng.normal(0, 1, n),
        "sex": ["F", "M"] * (n // 2),
        "site": ["siteA"] * (n // 2) + ["siteB"] * (n - n // 2),
        "Motion_FD": rng.uniform(0.05, 0.35, n),
    })
    pheno.to_csv(tmp_path / "pheno.csv", index=False)

    # Small synthetic atlas CSV with MNI coordinates and Yeo-7 network labels
    coords = np.array([
        [-40.0, 20.0, 30.0],
        [40.0, 20.0, 30.0],
        [-20.0, -70.0, 10.0],
        [20.0, -70.0, 10.0],
        [-10.0, -60.0, -30.0],  # cerebellar MNI coordinate
        [10.0, -60.0, -30.0],
    ][:n_rois])
    atlas_df = pd.DataFrame({
        "name": [f"ROI_{i}" for i in range(n_rois)],
        "network": ["Default", "Default", "Visual", "Visual", "Somatomotor", "Somatomotor"][:n_rois],
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
    })
    atlas_path = tmp_path / "atlas.csv"
    atlas_df.to_csv(atlas_path, index=False)

    manifest = tmp_path / "study.yaml"
    manifest.write_text(yaml.safe_dump({
        "schema_version": 1,
        "loader": "NumpyLoader",
        "paths": {"data_path": "conn.npy", "pheno_path": "pheno.csv"},
    }))

    cfg = tmp_path / "analysis.yaml"
    cfg.write_text(yaml.safe_dump({
        "data": "study.yaml",
        "atlas": "atlas.csv",
        "design": {
            "type": "glm",
            "interest": "age",
            "confounds": ["age_copy"] if collinear else ["sex"],
            "sites": "site",
        },
        "inference": {"method": "tstat", "n_permutations": 30, "use_mp": False, "rng": 0},
        "output": "out",
    }))
    return manifest, cfg, atlas_path


def test_qc_and_check_alias_valid_and_json(tmp_path):
    _, cfg, _ = _make_study(tmp_path, collinear=False)
    out_json = tmp_path / "qc.json"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(["qc", str(cfg), "--motion-qc", "--json", "--out", str(out_json)])
    assert code == 0
    payload = json.loads(buf.getvalue())
    assert payload["ok"] is True
    assert payload["design_diagnostics"]["rank_deficient"] is False
    assert payload["motion_qc"]["status"] == "available"
    assert out_json.is_file()

    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        code2 = main(["check", str(cfg)])
    assert code2 == 0
    assert "conninfpy qc [PASS]" in buf2.getvalue()


def test_qc_detects_rank_deficiency(tmp_path):
    _, cfg, _ = _make_study(tmp_path, collinear=True)
    # Force collinearity: interest == age, confound == age * 2 -> rank deficient
    cfg_data = yaml.safe_load(cfg.read_text())
    cfg_data["design"]["confounds"] = ["age_copy"]
    # Replace age_copy exactly with 2 * age
    pheno = pd.read_csv(tmp_path / "pheno.csv")
    pheno["age_copy"] = 2.0 * pheno["age"]
    pheno.to_csv(tmp_path / "pheno.csv", index=False)
    cfg.write_text(yaml.safe_dump(cfg_data))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(["qc", str(cfg), "--json"])
    assert code == 1
    payload = json.loads(buf.getvalue())
    assert payload["ok"] is False
    assert payload["design_diagnostics"]["rank_deficient"] is True


def test_init_generates_valid_yaml_and_respects_force(tmp_path):
    manifest, _, _ = _make_study(tmp_path)
    out_yaml = tmp_path / "generated.yaml"

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(["init", "--manifest", str(manifest), "--design", "glm", "--out", str(out_yaml)])
    assert code == 0
    assert out_yaml.is_file()
    loaded = load_config(out_yaml)
    assert loaded["design"]["type"] == "glm"
    assert loaded["inference"]["two_tailed"] is True

    # Without --force, refuses to overwrite
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        code_dup = main(["init", "--manifest", str(manifest), "--out", str(out_yaml)])
    assert code_dup == 1
    assert "already exists" in err.getvalue()

    # With --force, overwrites cleanly
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["init", "--manifest", str(manifest), "--out", str(out_yaml), "--force"]) == 0


def test_interpret_exports_tables_and_plots(tmp_path):
    _, cfg, _ = _make_study(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        written = run(cfg)
    run_dir = written["provenance"].parent

    interp_dir = tmp_path / "interp_out"
    with contextlib.redirect_stdout(io.StringIO()):
        code = main([
            "interpret", str(run_dir),
            "--cerebellum", "anatom",
            "--out", str(interp_dir),
        ])
    assert code == 0
    assert (interp_dir / "node_summary.csv").is_file()
    assert (interp_dir / "network_summary.csv").is_file()
    assert (interp_dir / "annotated_edges.csv").is_file()
    assert (interp_dir / "summary_figure.png").is_file()

    # Also test --no-plots targeting bundle.pkl directly
    interp_no_plots = tmp_path / "interp_no_plots"
    with contextlib.redirect_stdout(io.StringIO()):
        code2 = main([
            "interpret", str(written["bundle"]),
            "--no-plots",
            "--out", str(interp_no_plots),
        ])
    assert code2 == 0
    assert (interp_no_plots / "node_summary.csv").is_file()
    assert not (interp_no_plots / "summary_figure.png").exists()


def test_app_launch_and_missing_streamlit_guard(monkeypatch):
    # Guard when streamlit is unavailable
    real_find_spec = importlib.util.find_spec

    def _no_streamlit(name, package=None):
        if name == "streamlit":
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", _no_streamlit)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        code = main(["app"])
    assert code == 1
    assert "streamlit is not installed" in err.getvalue()

    # Launch when streamlit is mocked available
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, package=None: mock.MagicMock(origin="apps/streamlit_app.py") if name == "streamlit" else real_find_spec(name, package),
    )
    with mock.patch("subprocess.run", return_value=mock.MagicMock(returncode=0)) as m_run:
        code_ok = main(["app", "--server.port", "8505"])
    assert code_ok == 0
    cmd = m_run.call_args[0][0]
    assert cmd[1:4] == ["-m", "streamlit", "run"]
    assert cmd[4].endswith("apps/streamlit_app.py")
    assert cmd[5:] == ["--server.port", "8505"]
