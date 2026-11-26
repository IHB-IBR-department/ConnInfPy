"""Tests for CLI (E, H) grid support, block net_labels auto-derivation, and effect export."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from conninfpy import AtlasInfo, InferenceResult, analyze
from conninfpy._cli_tools import _reconstruct_inference_result, init_manifest, interpret_outputs, run_qc
from conninfpy.cli import main, run


def _write_synthetic_cli_project(
    tmp_path: Path,
    *,
    n_subjects: int = 12,
    n_regions: int = 8,
    inference_cfg: dict | None = None,
    atlas_extra_col: str | None = None,
) -> Path:
    rng = np.random.default_rng(42)
    half = n_subjects // 2
    g1 = rng.normal(0.0, 0.25, size=(half, n_regions, n_regions))
    g2 = rng.normal(0.0, 0.25, size=(n_subjects - half, n_regions, n_regions))
    # Symmetrize and zero diagonal; plant a strong signal on (0, 1)
    for stack in (g1, g2):
        for s in range(stack.shape[0]):
            stack[s] = 0.5 * (stack[s] + stack[s].T)
            np.fill_diagonal(stack[s], 0.0)
    g2[:, 0, 1] += 0.9
    g2[:, 1, 0] += 0.9
    Y = np.concatenate([g1, g2], axis=0)

    stack_path = tmp_path / "connectivity.npy"
    np.save(stack_path, Y)

    pheno_path = tmp_path / "pheno.csv"
    pd.DataFrame(
        {
            "subject_id": [f"sub-{i:02d}" for i in range(n_subjects)],
            "group": ["ctrl"] * half + ["case"] * (n_subjects - half),
        }
    ).to_csv(pheno_path, index=False)

    # Networks deliberately out of alphabetical order to test row-order determinism
    networks = ["Vis", "Vis", "DMN", "DMN", "SomMot", "SomMot", "FPN", "FPN"][:n_regions]
    atlas_df = pd.DataFrame(
        {
            "name": [f"ROI_{i}" for i in range(n_regions)],
            "network": networks,
            "hemisphere": ["L" if i % 2 == 0 else "R" for i in range(n_regions)],
        }
    )
    if atlas_extra_col:
        atlas_df[atlas_extra_col] = [f"Mod_{i // 4}" for i in range(n_regions)]
    atlas_path = tmp_path / "atlas.csv"
    atlas_df.to_csv(atlas_path, index=False)

    manifest_path = tmp_path / "manifest.yaml"
    init_manifest(
        data_file=stack_path,
        pheno_path=pheno_path,
        out_path=manifest_path,
        kind="fisher_z",
        id_col="subject_id",
        name="synth_test",
    )

    cfg = {
        "data": "manifest.yaml",
        "atlas": "atlas.csv",
        "design": {"type": "two-sample", "group": "group", "levels": ["ctrl", "case"]},
        "inference": inference_cfg or {"method": "tfnbs", "n_permutations": 19, "use_mp": False, "rng": 0},
        "alpha": 0.05,
        "output": "results",
    }
    cfg_path = tmp_path / "analysis.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def test_cli_run_eh_grid_exports_cells_summary_npz_and_provenance(tmp_path: Path) -> None:
    cfg_path = _write_synthetic_cli_project(
        tmp_path,
        inference_cfg={
            "method": "tfnbs",
            "e": [0.4, 0.6],
            "h": [1.5, 2.0],
            "n_permutations": 19,
            "use_mp": False,
            "rng": 0,
        },
    )
    out_dir = tmp_path / "out_grid"
    written = run(cfg_path, out_dir)

    assert not (out_dir / "edges.csv").exists()
    assert (out_dir / "edges_cell0.csv").is_file()
    assert (out_dir / "edges_cell1.csv").is_file()
    assert (out_dir / "grid_summary.csv").is_file()
    assert "edges_cell0" in written and "edges_cell1" in written and "grid_summary" in written

    with np.load(out_dir / "pvals.npz", allow_pickle=False) as z:
        assert z["positive"].shape == (8, 8, 2)
        assert z["negative"].shape == (8, 8, 2)
        np.testing.assert_allclose(z["e_grid"], [0.4, 0.6])
        np.testing.assert_allclose(z["h_grid"], [1.5, 2.0])

    summary_df = pd.read_csv(out_dir / "grid_summary.csv")
    assert list(summary_df.columns) == [
        "param_idx",
        "E",
        "H",
        "n_sig_pos",
        "n_sig_neg",
        "jaccard_pos_vs_cell0",
        "jaccard_neg_vs_cell0",
    ]
    assert len(summary_df) == 2
    assert list(summary_df["param_idx"]) == [0, 1]
    assert np.allclose(summary_df["E"], [0.4, 0.6])
    assert np.allclose(summary_df["H"], [1.5, 2.0])
    # Empty union produces NaN; non-empty cell 0 vs itself produces 1.0
    for tail in ("pos", "neg"):
        col = f"jaccard_{tail}_vs_cell0"
        if summary_df.loc[0, f"n_sig_{tail}"] == 0:
            assert np.isnan(summary_df.loc[0, col])
        else:
            assert summary_df.loc[0, col] == pytest.approx(1.0)

    prov = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    res_prov = prov["results"]["result"]
    assert res_prov["is_grid"] is True
    assert res_prov["param_grid"] == [
        {"param_idx": 0, "E": 0.4, "H": 1.5},
        {"param_idx": 1, "E": 0.6, "H": 2.0},
    ]
    assert isinstance(res_prov["n_significant"]["positive"], list)
    assert len(res_prov["n_significant"]["positive"]) == 2
    assert len(res_prov["n_significant"]["negative"]) == 2


def test_cli_interpret_grid_defaults_to_cell0_with_warning_and_respects_param_idx(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_path = _write_synthetic_cli_project(
        tmp_path,
        inference_cfg={
            "method": "tfnbs",
            "e": [0.4, 0.6],
            "h": [1.5, 2.0],
            "n_permutations": 19,
            "use_mp": False,
            "rng": 0,
        },
    )
    out_dir = tmp_path / "out_grid"
    run(cfg_path, out_dir)

    from conninfpy.post_analysis import ResultBundle

    bundle = ResultBundle.load(out_dir / "bundle.pkl")
    reconstructed = _reconstruct_inference_result(bundle, out_dir / "pvals.npz")
    assert reconstructed.is_grid is True
    np.testing.assert_allclose(reconstructed.e_grid, [0.4, 0.6])
    np.testing.assert_allclose(reconstructed.h_grid, [1.5, 2.0])

    # Defaulting to idx=0 warns and prints message
    with pytest.warns(UserWarning, match="idx=0"):
        written0 = interpret_outputs(out_dir, out_dir=out_dir / "interp0", no_plots=True)
    err = capsys.readouterr().err
    assert "idx=0" in err and "--param-idx" in err
    assert (out_dir / "interp0" / "node_summary.csv").is_file()
    assert "node_summary" in written0

    # Explicit valid param_idx=1
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        interpret_outputs(out_dir, out_dir=out_dir / "interp1", no_plots=True, param_idx=1)
    assert not any("defaulting to cell idx=0" in str(w.message) for w in rec)
    assert (out_dir / "interp1" / "node_summary.csv").is_file()

    # Out-of-range param_idx raises ValueError
    with pytest.raises(ValueError, match="out of range"):
        interpret_outputs(out_dir, out_dir=out_dir / "interp_bad", no_plots=True, param_idx=5)

    # CLI entry point with --param-idx
    rc = main(["interpret", str(out_dir), "--param-idx", "1", "--no-plots", "--out", str(out_dir / "interp_cli")])
    assert rc == 0
    assert (out_dir / "interp_cli" / "node_summary.csv").is_file()


@pytest.mark.parametrize("method", ["cnbs", "ni_tfnbs", "fbc_tfnbs"])
def test_analyze_derives_net_labels_from_atlas_for_block_methods(method: str) -> None:
    rng = np.random.default_rng(7)
    g1 = rng.normal(0.0, 0.2, size=(6, 6, 6))
    g2 = rng.normal(0.0, 0.2, size=(6, 6, 6))
    for stack in (g1, g2):
        for s in range(6):
            stack[s] = 0.5 * (stack[s] + stack[s].T)
            np.fill_diagonal(stack[s], 0.0)
    atlas = AtlasInfo(
        labels=[f"R{i}" for i in range(6)],
        networks=["Vis", "Vis", "DMN", "DMN", "FPN", "FPN"],
    )
    res = analyze(
        group1=g1,
        group2=g2,
        input="z",
        method=method,
        atlas=atlas,
        n_permutations=19,
        use_mp=False,
        rng=0,
        cnbs_correction="bh" if method == "cnbs" else "bonferroni",
    )
    mapping = res.inference.analysis_provenance.get("net_labels_mapping")
    assert mapping == {"Vis": 0, "DMN": 1, "FPN": 2}


def test_cli_net_labels_atlas_and_csv_column_and_provenance(tmp_path: Path) -> None:
    # 1. inference.net_labels: 'atlas'
    (tmp_path / "case_atlas").mkdir(exist_ok=True)
    cfg_atlas = _write_synthetic_cli_project(
        tmp_path / "case_atlas",
        inference_cfg={
            "method": "cnbs",
            "net_labels": "atlas",
            "cnbs_correction": "bh",
            "n_permutations": 19,
            "use_mp": False,
            "rng": 0,
        },
    )
    out_atlas = tmp_path / "case_atlas" / "out"
    run(cfg_atlas, out_atlas)
    prov_atlas = json.loads((out_atlas / "provenance.json").read_text(encoding="utf-8"))
    assert prov_atlas["results"]["result"]["net_labels_mapping"] == {
        "Vis": 0,
        "DMN": 1,
        "SomMot": 2,
        "FPN": 3,
    }

    # 2. inference.net_labels: custom CSV column
    (tmp_path / "case_col").mkdir(exist_ok=True)
    cfg_col = _write_synthetic_cli_project(
        tmp_path / "case_col",
        atlas_extra_col="yeo_coarse",
        inference_cfg={
            "method": "cnbs",
            "net_labels": "yeo_coarse",
            "cnbs_correction": "bh",
            "n_permutations": 19,
            "use_mp": False,
            "rng": 0,
        },
    )
    out_col = tmp_path / "case_col" / "out"
    run(cfg_col, out_col)
    prov_col = json.loads((out_col / "provenance.json").read_text(encoding="utf-8"))
    assert prov_col["results"]["result"]["net_labels_mapping"] == {"Mod_0": 0, "Mod_1": 1}


def test_cnbs_bonferroni_min_p_warning_in_analyze_qc_and_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # M=4 networks -> K=10 blocks; n_permutations=19 -> min_p = 10/20 = 0.5 > 0.05
    cfg_path = _write_synthetic_cli_project(
        tmp_path,
        inference_cfg={
            "method": "cnbs",
            "cnbs_correction": "bonferroni",
            "n_permutations": 19,
            "use_mp": False,
            "rng": 0,
        },
    )

    # 1. qc warning
    code, report = run_qc(cfg_path)
    assert code == 0
    assert any("Bonferroni" in w and "K=10" in w for w in report["warnings"])
    capsys.readouterr()

    # 2. run --dry-run warning
    run(cfg_path, dry_run=True)
    dry_out = capsys.readouterr().out
    assert "WARNING:" in dry_out and "Bonferroni" in dry_out

    # 3. analyze() RuntimeWarning
    rng = np.random.default_rng(1)
    g1 = rng.normal(0.0, 0.2, size=(6, 8, 8))
    g2 = rng.normal(0.0, 0.2, size=(6, 8, 8))
    for stack in (g1, g2):
        for s in range(6):
            stack[s] = 0.5 * (stack[s] + stack[s].T)
            np.fill_diagonal(stack[s], 0.0)
    atlas = AtlasInfo(
        labels=[f"R{i}" for i in range(8)],
        networks=["Vis", "Vis", "DMN", "DMN", "SomMot", "SomMot", "FPN", "FPN"],
    )
    with pytest.warns(RuntimeWarning, match="Bonferroni"):
        analyze(
            group1=g1,
            group2=g2,
            input="z",
            method="cnbs",
            cnbs_correction="bonferroni",
            atlas=atlas,
            n_permutations=19,
            use_mp=False,
            rng=0,
        )

    # No warning when cnbs_correction='bh'
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        analyze(
            group1=g1,
            group2=g2,
            input="z",
            method="cnbs",
            cnbs_correction="bh",
            atlas=atlas,
            n_permutations=19,
            use_mp=False,
            rng=0,
        )
    assert not any("Bonferroni" in str(w.message) for w in rec)


def test_significant_edges_and_csv_include_effect_scale_and_predicted_after_t_signed(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(11)
    g1 = rng.normal(0.0, 0.15, size=(8, 6, 6))
    g2 = rng.normal(0.0, 0.15, size=(8, 6, 6))
    for stack in (g1, g2):
        for s in range(8):
            stack[s] = 0.5 * (stack[s] + stack[s].T)
            np.fill_diagonal(stack[s], 0.0)
    g2[:, 0, 1] += 1.0
    g2[:, 1, 0] += 1.0

    res = analyze(
        group1=g1,
        group2=g2,
        test_type="two-sample",
        method="tstat",
        n_permutations=39,
        use_mp=False,
        rng=0,
    )
    inf = res.inference
    df = res.significant_edges(include_nonsig=True)
    cols = list(df.columns)
    t_idx = cols.index("t_signed")
    comp_labels = list(inf.design["comparison"])
    expected_after_t = ["effect", "scale"] + [f"fc_{lbl}" for lbl in comp_labels]
    assert cols[t_idx + 1 : t_idx + 1 + len(expected_after_t)] == expected_after_t

    eff = inf.effect_map()
    row0 = df.iloc[0]
    i0, j0 = int(row0["roi_i"]), int(row0["roi_j"])
    assert row0["effect"] == pytest.approx(eff[i0, j0])
    assert row0["scale"] == inf.design["scale"]
    for lbl in comp_labels:
        assert row0[f"fc_{lbl}"] == pytest.approx(inf.predicted(lbl)[i0, j0])

    csv_path = tmp_path / "edges.csv"
    res.to_csv(csv_path, include_nonsig=True)
    csv_df = pd.read_csv(csv_path)
    assert list(csv_df.columns) == cols


def test_significant_edges_preserves_columns_when_beta_absent() -> None:
    p_pos = np.full((4, 4), 0.5)
    p_neg = np.full((4, 4), 0.5)
    p_pos[0, 1] = p_pos[1, 0] = 0.01
    stat_pos = np.zeros((4, 4))
    stat_pos[0, 1] = stat_pos[1, 0] = 3.5
    inf = InferenceResult(
        p_pos,
        p_neg,
        method="tstat",
        n_permutations=19,
        stat_positive=stat_pos,
        stat_negative=np.zeros((4, 4)),
    )
    df = inf.significant_edges(alpha=0.05)
    assert list(df.columns) == [
        "edge_id",
        "roi_i",
        "roi_j",
        "t_signed",
        "p_positive",
        "p_negative",
        "p_min",
        "tail",
    ]
