"""Tests for interpret atlas resolution/validation, ABIDE -9999 missingness, and qc/dry-run model sync."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from conninfpy._cli_tools import _resolve_atlas_for_interpret, interpret_outputs, run_qc
from conninfpy.cli import build_analysis, run
from conninfpy.loaders import load
from conninfpy.loaders.builtins import AbideSchaeferLoader
from conninfpy.post_analysis import ResultBundle
from conninfpy.results import InferenceResult


def _make_study(
    tmp_path: Path,
    pheno: pd.DataFrame,
    mats: np.ndarray,
    *,
    study_name: str = "study.yaml",
    data_kind: str = "correlation",
) -> Path:
    np.save(tmp_path / "mats.npy", mats)
    pheno.to_csv(tmp_path / "pheno.csv", index=False)
    manifest = {
        "loader": "ConnectivityMatrixLoader",
        "paths": {"data_path": "mats.npy", "pheno_path": "pheno.csv"},
        "params": {"data_kind": data_kind},
    }
    p = tmp_path / study_name
    p.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return p


def _sym_mats(n_sub: int, n_rois: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mats = rng.uniform(-0.3, 0.3, size=(n_sub, n_rois, n_rois))
    mats = 0.5 * (mats + np.swapaxes(mats, 1, 2))
    idx = np.arange(n_rois)
    mats[:, idx, idx] = 0.0
    return mats


# ---------------------------------------------------------------------------
# 1. interpret atlas resolution & ROI ID consistency
# ---------------------------------------------------------------------------


def _make_cc200_bundle_dir(tmp_path: Path, n_rois: int = 200) -> tuple[Path, ResultBundle]:
    out_dir = tmp_path / "cc200_results"
    out_dir.mkdir()
    pos = np.ones((n_rois, n_rois), dtype=np.float64)
    neg = np.ones((n_rois, n_rois), dtype=np.float64)
    pos[0, 1] = pos[1, 0] = 0.005
    stat_pos = np.zeros((n_rois, n_rois), dtype=np.float64)
    stat_pos[0, 1] = stat_pos[1, 0] = 4.2
    beta = np.zeros((2, n_rois, n_rois), dtype=np.float64)
    beta[1, 0, 1] = beta[1, 1, 0] = 0.25

    roi_ids = tuple(f"CC200_{i + 1}" for i in range(n_rois))
    groupings = {
        "cc200_net": tuple("Visual" if i < n_rois // 2 else "Default" for i in range(n_rois))
    }
    eligible = np.ones((n_rois, n_rois), dtype=bool)
    np.fill_diagonal(eligible, False)
    design = {
        "kind": "glm",
        "columns": ["intercept", "Age"],
        "contrast": [0.0, 1.0],
        "interest": "Age",
    }
    inf = InferenceResult(
        pos,
        neg,
        method="tstat",
        n_permutations=20,
        stat_positive=stat_pos,
        stat_negative=np.zeros_like(stat_pos),
        two_tailed=True,
        beta=beta,
        design=design,
        analysis_provenance={
            "roi_ids": roi_ids,
            "groupings": groupings,
            "eligible": eligible,
            "estimable": eligible,
            "tested": {"positive": eligible, "negative": eligible},
            "model_id": "glm_age",
            "contrast_id": "Age",
            "units": "z",
            "scale": "fisher_z",
            "family": {"two_tailed": True},
        },
    )
    bundle = ResultBundle.from_result(inf)
    bundle.save(out_dir / "bundle.pkl")
    np.savez_compressed(
        out_dir / "pvals.npz",
        positive=pos,
        negative=neg,
        stat_signed=stat_pos,
        beta=beta,
    )
    prov = {
        "alpha": 0.05,
        "analysis_file": "/nonexistent/other_machine/abide_cc200.yaml",
        "config": {"atlas": "/nonexistent/other_machine/cc200_roi_atlas.nii.gz"},
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    return out_dir, bundle


def test_interpret_does_not_guess_schaefer_for_200_roi_cc200_bundle(tmp_path: Path):
    out_dir, bundle = _make_cc200_bundle_dir(tmp_path, n_rois=200)
    prov_path = out_dir / "provenance.json"

    with pytest.warns(UserWarning, match="returning None without guessing"):
        resolved = _resolve_atlas_for_interpret(
            None, prov_path, 200, bundle_roi_ids=bundle.roi_ids
        )
    assert resolved is None

    interp_dir = tmp_path / "interp_no_atlas"
    with pytest.warns(UserWarning, match="returning None without guessing"):
        with contextlib.redirect_stdout(io.StringIO()):
            written = interpret_outputs(out_dir, out_dir=interp_dir, no_plots=True)

    assert "detected_pairs_positive" in written
    pairs_df = pd.read_csv(written["detected_pairs_positive"])
    assert list(pairs_df["roi_i"]) == ["CC200_1"]
    assert list(pairs_df["roi_j"]) == ["CC200_2"]

    assert "bundle_summary_cc200_net_positive" in written
    summary_df = pd.read_csv(written["bundle_summary_cc200_net_positive"])
    assert set(summary_df["group_a"]).issubset({"Visual", "Default"})

    assert "node_summary" in written
    node_df = pd.read_csv(written["node_summary"])
    assert "roi_id" in node_df.columns
    assert set(node_df["roi_id"].head(2)) == {"CC200_1", "CC200_2"}

    assert "annotated_edges" not in written
    assert "network_summary" not in written


def test_interpret_rejects_mismatched_explicit_atlas_roi_ids(tmp_path: Path):
    out_dir, bundle = _make_cc200_bundle_dir(tmp_path, n_rois=200)

    with pytest.raises(ValueError, match="do not match bundle roi_ids"):
        _resolve_atlas_for_interpret(
            "schaefer_200_yeo7",
            None,
            200,
            bundle_roi_ids=bundle.roi_ids,
        )

    with pytest.raises(ValueError, match="do not match bundle roi_ids"):
        interpret_outputs(
            out_dir,
            atlas_spec="schaefer_200_yeo7",
            out_dir=tmp_path / "interp_bad",
            no_plots=True,
        )


def test_interpret_allows_default_integer_roi_ids_with_explicit_atlas():
    resolved = _resolve_atlas_for_interpret(
        "schaefer_100_yeo7",
        None,
        100,
        bundle_roi_ids=tuple(range(100)),
    )
    assert resolved is not None
    assert len(resolved) == 100


# ---------------------------------------------------------------------------
# 2. ABIDE -9999 missing value code + qc missing_by_column summary
# ---------------------------------------------------------------------------


def test_abide_loader_converts_minus_9999_to_nan():
    repo_root = Path(__file__).resolve().parents[1]
    abide_dir = repo_root / "datasets" / "demo" / "ABIDE_mini"
    loader = AbideSchaeferLoader(
        data_path=str(abide_dir / "abide_mini.npz"),
        pheno_csv_path=str(abide_dir / "ABIDE1_Phenotypic_mini.csv"),
    )
    dataset = loader.load()
    pheno = dataset.pheno
    assert "FIQ" in pheno.columns
    # OHSU_0050156 has FIQ = -9999.0 in ABIDE1_Phenotypic_mini.csv (row 13)
    assert int(pheno["FIQ"].isna().sum()) >= 1
    assert not (pheno["FIQ"] == -9999).any()
    assert not (pheno["FIQ"] == -9999.0).any()


def test_qc_reports_missing_by_column_in_json_and_text(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    abide_dir = repo_root / "datasets" / "demo" / "ABIDE_mini"
    manifest = {
        "loader": "AbideSchaeferLoader",
        "paths": {
            "data_path": str(abide_dir / "abide_mini.npz"),
            "pheno_csv_path": str(abide_dir / "ABIDE1_Phenotypic_mini.csv"),
        },
    }
    manifest_path = tmp_path / "abide_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    cfg = {
        "data": "abide_manifest.yaml",
        "design": {
            "type": "glm",
            "interest": "Age",
            "confounds": ["Sex", "FIQ", "Motion_FD"],
            "sites": "site",
        },
        "inference": {"method": "tstat", "n_permutations": 10},
    }
    cfg_path = tmp_path / "analysis.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _code, report = run_qc(cfg_path)
    out_txt = buf.getvalue()

    assert report["missing_by_column"] == {
        "Age": 0,
        "Sex": 0,
        "FIQ": 2,
        "Motion_FD": 0,
        "site": 0,
    }
    assert report["n_dropped"] == 2
    assert "missing_by_column:" in out_txt
    assert "FIQ=2" in out_txt


# ---------------------------------------------------------------------------
# 3. qc and dry-run synchronization with actual run model
# ---------------------------------------------------------------------------


def test_qc_combat_only_excludes_site_dummies_from_design_matrix(tmp_path: Path):
    n_sub = 12
    pheno = pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(n_sub)],
            "age": np.linspace(10.0, 30.0, n_sub),
            "sex": [0, 1] * (n_sub // 2),
            "site": ["S1"] * 4 + ["S2"] * 4 + ["S3"] * 4,
        }
    )
    _make_study(tmp_path, pheno, _sym_mats(n_sub, n_rois=6, seed=1))

    cfg_combat_only = {
        "data": "study.yaml",
        "design": {
            "type": "glm",
            "interest": "age",
            "confounds": ["sex"],
            "sites": "site",
            "harmonize": "combat_only",
        },
        "inference": {"method": "tstat", "n_permutations": 10},
    }
    p1 = tmp_path / "combat_only.yaml"
    p1.write_text(yaml.safe_dump(cfg_combat_only), encoding="utf-8")

    with contextlib.redirect_stdout(io.StringIO()):
        code1, rep1 = run_qc(p1)
    assert code1 == 0
    cols1 = rep1["design_diagnostics"]["columns"]
    assert cols1 == ["intercept", "age", "sex"]
    assert not any(c.startswith("site[") for c in cols1)
    assert "combat_only" in rep1["model_summary"]

    cfg_site_glm = {
        "data": "study.yaml",
        "design": {
            "type": "glm",
            "interest": "age",
            "confounds": ["sex"],
            "sites": "site",
            "harmonize": "combat_site_dummies_glm",
        },
        "inference": {"method": "tstat", "n_permutations": 10},
    }
    p2 = tmp_path / "combat_site_glm.yaml"
    p2.write_text(yaml.safe_dump(cfg_site_glm), encoding="utf-8")

    with contextlib.redirect_stdout(io.StringIO()):
        code2, rep2 = run_qc(p2)
    assert code2 == 0
    cols2 = rep2["design_diagnostics"]["columns"]
    assert "site[S2]" in cols2 and "site[S3]" in cols2


def test_paired_glm_with_interest_contrast_direction_and_model_summary(tmp_path: Path):
    n_subj = 6
    pheno = pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(n_subj)] * 2,
            "condition": ["pre"] * n_subj + ["post"] * n_subj,
            "age": list(np.linspace(20.0, 40.0, n_subj)) * 2,
            "motion": [0.12, 0.25, 0.14, 0.31, 0.18, 0.22] * 2,
            "site": (["A", "B"] * 3) * 2,
        }
    )
    _make_study(tmp_path, pheno, _sym_mats(2 * n_subj, n_rois=6, seed=3))

    cfg = {
        "data": "study.yaml",
        "design": {
            "type": "paired",
            "condition": "condition",
            "levels": ["pre", "post"],
            "interest": "age",
            "subject_confounds": ["motion"],
            "sites": "site",
        },
        "inference": {"method": "tstat", "n_permutations": 10},
    }
    cfg_path = tmp_path / "paired_glm.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    dataset = load(tmp_path / "study.yaml")
    _, summary = build_analysis(dataset, cfg)
    expected_dir = "positive: higher age -> larger (post - pre)"
    assert summary["contrast_direction"] == expected_dir

    qc_buf = io.StringIO()
    with contextlib.redirect_stdout(qc_buf):
        code, rep = run_qc(cfg_path)
    assert code == 0
    assert rep["contrast_direction"] == expected_dir
    assert "predictor=age" in rep["model_summary"]
    assert "model:" in qc_buf.getvalue()

    dry_buf = io.StringIO()
    with contextlib.redirect_stdout(dry_buf):
        run(cfg_path, dry_run=True)
    dry_txt = dry_buf.getvalue()
    assert expected_dir in dry_txt
    assert "model:" in dry_txt
    assert "predictor=age" in dry_txt


def test_glm_dry_run_and_qc_print_clear_model_summary(tmp_path: Path):
    n_sub = 10
    pheno = pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(n_sub)],
            "age": np.linspace(10.0, 25.0, n_sub),
            "sex": [0, 1] * (n_sub // 2),
            "fiq": [95.0, 112.0, 101.0, 118.0, 99.0, 108.0, 121.0, 92.0, 115.0, 104.0],
            "site": ["S1"] * 5 + ["S2"] * 5,
        }
    )
    _make_study(tmp_path, pheno, _sym_mats(n_sub, n_rois=6, seed=5))

    cfg = {
        "data": "study.yaml",
        "design": {
            "type": "glm",
            "interest": "age",
            "confounds": ["sex", "fiq"],
            "sites": "site",
        },
        "inference": {"method": "tstat", "n_permutations": 10},
    }
    cfg_path = tmp_path / "glm.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    qc_buf = io.StringIO()
    with contextlib.redirect_stdout(qc_buf):
        code, rep = run_qc(cfg_path)
    assert code == 0
    expected = "predictor=age | confounds=[sex, fiq] | sites=site (combat_site_dummies_glm) | scale=fisher_z"
    assert rep["model_summary"] == expected
    assert expected in qc_buf.getvalue()

    dry_buf = io.StringIO()
    with contextlib.redirect_stdout(dry_buf):
        run(cfg_path, dry_run=True)
    assert expected in dry_buf.getvalue()
