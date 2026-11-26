"""Regression tests for NETSTAT-67 CLI & atlas fixes."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import yaml

from conninfpy._cli_tools import _resolve_atlas_for_interpret, run_qc
from conninfpy.analysis import analyze
from conninfpy.cli import ConfigError, _atlas, build_analysis, main, run


def _make_study(tmp: Path, pheno: pd.DataFrame, data: np.ndarray) -> Path:
    np.save(tmp / "conn.npy", data)
    pheno.to_csv(tmp / "pheno.csv", index=False)
    manifest = tmp / "study.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "loader": "NumpyLoader",
                "paths": {"data_path": "conn.npy", "pheno_path": "pheno.csv"},
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _make_nifti_atlas(path: Path, n_rois: int = 6, *, is_4d: bool = False) -> Path:
    if is_4d:
        vol = np.zeros((4, 4, 4, 2), dtype=np.float32)
        vol[0, 0, 0, 0] = 1.0
        img = nib.Nifti1Image(vol, np.eye(4))
    else:
        vol = np.zeros((4, 4, max(4, n_rois)), dtype=np.int32)
        for i in range(1, n_rois + 1):
            vol[0, 0, i - 1] = i
        img = nib.Nifti1Image(vol, np.eye(4))
    nib.save(img, str(path))
    return path


def _make_lut(path: Path, n_rois: int = 6) -> Path:
    rows = [
        {"id": i, "name": f"ROI_{i}", "network": f"Net_{(i - 1) % 2}"}
        for i in range(1, n_rois + 1)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _sym_mats(n: int, N: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.tanh(rng.normal(0, 0.25, (n, N, N)))
    r = (r + r.transpose(0, 2, 1)) / 2
    idx = np.arange(N)
    r[:, idx, idx] = 1.0
    return r


# ---------------------------------------------------------------------------
# 1. Forbid silent dropping of confounds/sites in two-sample and one-sample
# ---------------------------------------------------------------------------


def test_two_sample_with_confounds_raises_config_error_with_glm_example(tmp_path: Path):
    pheno = pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(8)],
            "group": ["ctrl", "case"] * 4,
            "age": [20, 25, 30, 35, 22, 27, 32, 37],
            "sex": ["F", "M"] * 4,
        }
    )
    _make_study(tmp_path, pheno, _sym_mats(8))
    from conninfpy.loaders import load

    dataset = load(tmp_path / "study.yaml")
    cfg = {
        "data": "study.yaml",
        "design": {
            "type": "two-sample",
            "group": "group",
            "levels": ["ctrl", "case"],
            "confounds": ["age", "sex"],
        },
    }
    with pytest.raises(ConfigError) as exc_info:
        build_analysis(dataset, cfg)
    msg = str(exc_info.value)
    assert "type: glm" in msg
    assert "interest: group" in msg
    assert "interest_reference: ctrl" in msg
    assert "confounds: [age, sex]" in msg


@pytest.mark.parametrize(
    "extra_design",
    [
        {"confounds": ["age"]},
        {"sites": "site"},
        {"confounds": ["age"], "sites": "site"},
    ],
)
def test_one_sample_with_confounds_or_sites_raises_config_error(
    tmp_path: Path, extra_design: dict
):
    pheno = pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(6)],
            "age": [20, 25, 30, 35, 22, 27],
            "site": ["a", "a", "a", "b", "b", "b"],
        }
    )
    _make_study(tmp_path, pheno, _sym_mats(6))
    from conninfpy.loaders import load

    dataset = load(tmp_path / "study.yaml")
    cfg = {"data": "study.yaml", "design": {"type": "one-sample", **extra_design}}
    with pytest.raises(ConfigError, match="one-sample"):
        build_analysis(dataset, cfg)


def test_two_sample_with_sites_still_succeeds(tmp_path: Path):
    pheno = pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(8)],
            "group": ["ctrl", "case"] * 4,
            "site": ["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"],
        }
    )
    _make_study(tmp_path, pheno, _sym_mats(8))
    from conninfpy.loaders import load

    dataset = load(tmp_path / "study.yaml")
    cfg = {
        "data": "study.yaml",
        "design": {
            "type": "two-sample",
            "group": "group",
            "levels": ["ctrl", "case"],
            "sites": "site",
        },
    }
    kwargs, summary = build_analysis(dataset, cfg)
    assert kwargs["test_type"] == "two-sample"
    assert "sites" in kwargs
    assert summary["n_subjects_used"] == 8


# ---------------------------------------------------------------------------
# 2. Paired scan/pair counts and contrast direction
# ---------------------------------------------------------------------------


def test_paired_contrast_direction_and_counts_in_build_qc_and_dry_run(tmp_path: Path):
    n_pairs = 6
    subj_ids = [f"s{i}" for i in range(n_pairs)]
    pheno = pd.DataFrame(
        {
            "subject_id": subj_ids + subj_ids,
            "cond": ["pre"] * n_pairs + ["post"] * n_pairs,
        }
    )
    Y = _sym_mats(2 * n_pairs, N=6, seed=1)
    # Make 'post' (levels[1]) clearly higher than 'pre' (levels[0]) at edge (0, 1)
    # with nonzero variance across pairs so the t-statistic is finite.
    pre_vals = 0.05 + 0.01 * np.arange(n_pairs)
    post_vals = 0.75 + 0.02 * np.arange(n_pairs)
    Y[:n_pairs, 0, 1] = Y[:n_pairs, 1, 0] = pre_vals
    Y[n_pairs:, 0, 1] = Y[n_pairs:, 1, 0] = post_vals

    _make_study(tmp_path, pheno, Y)
    cfg = {
        "data": "study.yaml",
        "design": {
            "type": "paired",
            "condition": "cond",
            "levels": ["pre", "post"],
            "subject": "subject_id",
        },
        "inference": {"method": "tstat", "n_permutations": 20, "use_mp": False, "rng": 0},
        "output": "out",
    }
    cfg_path = tmp_path / "analysis.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    from conninfpy.loaders import load

    dataset = load(tmp_path / "study.yaml")
    kwargs, summary = build_analysis(dataset, cfg)
    assert summary["n_scans_in_data"] == 12
    assert summary["n_pairs_used"] == 6
    assert summary["n_scans_used"] == 12
    assert summary["n_dropped"] == 0
    assert summary["contrast_direction"] == "positive: post > pre"

    # Verify engine sign matches contrast_direction
    res = analyze(**kwargs, method="tstat", n_permutations=20, use_mp=False, rng=0)
    assert res.inference.stat_signed[0, 1] > 0

    # Verify qc produces no false dropped-observation warning and keeps JSON keys
    qc_buf = io.StringIO()
    with contextlib.redirect_stdout(qc_buf):
        code, report = run_qc(cfg_path)
    assert code == 0
    assert report["n_subjects_in_data"] == 12
    assert report["n_subjects_used"] == 6
    assert report["n_dropped"] == 0
    assert report["n_scans_in_data"] == 12
    assert report["n_pairs_used"] == 6
    assert report["n_scans_used"] == 12
    assert report["contrast_direction"] == "positive: post > pre"
    assert not any("dropped" in w for w in report["warnings"])
    qc_out = qc_buf.getvalue()
    assert "12 scans -> 6 complete pairs (0 dropped)" in qc_out
    assert "positive: post > pre" in qc_out

    # Verify dry-run output
    dry_buf = io.StringIO()
    with contextlib.redirect_stdout(dry_buf):
        run(cfg_path, dry_run=True)
    dry_out = dry_buf.getvalue()
    assert "12 scans -> 6 complete pairs (0 dropped)" in dry_out
    assert "positive: post > pre" in dry_out


# ---------------------------------------------------------------------------
# 3. Narrow interpret --cerebellum choices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_choice", ["buckner7", "buckner17", "nettekoven"])
def test_interpret_cerebellum_rejects_unsupported_choices(bad_choice: str):
    with pytest.raises(SystemExit) as exc_info:
        main(["interpret", "dummy_target", "--cerebellum", bad_choice])
    assert exc_info.value.code != 0


def test_interpret_cerebellum_help_allows_only_none_and_anatom(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["interpret", "--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "{none,anatom}" in out
    assert "buckner7" not in out


# ---------------------------------------------------------------------------
# 4. NIfTI atlas (.nii / .nii.gz) + atlas_lut support in CLI
# ---------------------------------------------------------------------------


def test_nifti_atlas_and_lut_in_run_qc_and_interpret(tmp_path: Path):
    n_rois = 6
    pheno = pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(10)],
            "age": np.linspace(20, 40, 10),
        }
    )
    _make_study(tmp_path, pheno, _sym_mats(10, N=n_rois, seed=2))
    nii_path = _make_nifti_atlas(tmp_path / "atlas_6.NII.GZ", n_rois=n_rois)
    lut_path = _make_lut(tmp_path / "atlas_lut.csv", n_rois=n_rois)

    cfg = {
        "data": "study.yaml",
        "atlas": nii_path.name,
        "atlas_lut": lut_path.name,
        "design": {"type": "glm", "interest": "age"},
        "inference": {"method": "tstat", "n_permutations": 10, "use_mp": False, "rng": 0},
        "output": "out",
    }
    cfg_path = tmp_path / "analysis.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with contextlib.redirect_stdout(io.StringIO()):
        code, qc_report = run_qc(cfg_path)
    assert code == 0
    assert qc_report["n_regions"] == n_rois

    with contextlib.redirect_stdout(io.StringIO()):
        written = run(cfg_path)
    assert written["edges"].exists()

    # Provenance-based resolution in interpret without --atlas
    resolved = _resolve_atlas_for_interpret(None, written["provenance"], n_rois)
    assert resolved is not None
    assert len(resolved) == n_rois
    assert resolved.labels[0] == "ROI_1"
    assert resolved.networks[0] == "Net_0"

    # Explicit --atlas + --atlas-lut via CLI interpret
    interp_dir = tmp_path / "interp_out"
    rc = main(
        [
            "interpret",
            str(tmp_path / "out"),
            "--atlas",
            str(nii_path),
            "--atlas-lut",
            str(lut_path),
            "--no-plots",
            "--out",
            str(interp_dir),
        ]
    )
    assert rc == 0
    assert (interp_dir / "detected_pairs_positive.csv").exists()


def test_nifti_atlas_region_mismatch_and_4d_errors(tmp_path: Path):
    from conninfpy.loaders import load

    pheno = pd.DataFrame({"subject_id": ["s0", "s1"], "age": [20, 30]})
    _make_study(tmp_path, pheno, _sym_mats(2, N=6))
    dataset = load(tmp_path / "study.yaml")

    bad_count_nii = _make_nifti_atlas(tmp_path / "atlas_4.nii.gz", n_rois=4)
    with pytest.raises(ConfigError, match="atlas has 4 regions but the data has 6"):
        _atlas(bad_count_nii.name, dataset, tmp_path, 6)

    bad_4d_nii = _make_nifti_atlas(tmp_path / "prob_4d.nii.gz", is_4d=True)
    with pytest.raises(ValueError, match="4-D"):
        _atlas(bad_4d_nii.name, dataset, tmp_path, 6)

    with pytest.raises(ValueError, match="4-D"):
        _resolve_atlas_for_interpret(str(bad_4d_nii), None, 6)
