"""Tests for CLI alpha propagation to methods boilerplate, NBS/cNBS summary & blocks.csv export,
reference selftest exact edge-set & skip reporting, and provenance-bound bundle loading.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import yaml

from conninfpy.cli import main
from conninfpy.reporting import methods_boilerplate


def _write_two_group_dataset(
    root: Path,
    *,
    n_per_group: int = 8,
    n_nodes: int = 6,
    planted_edges: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (0, 2)),
    effect: float = 2.0,
    with_networks: bool = False,
) -> tuple[Path, Path]:
    """Create a synthetic two-group connectivity dataset + manifest (and optional atlas CSV)."""
    rng = np.random.default_rng(123)
    n_subj = 2 * n_per_group
    mats = rng.normal(0.0, 0.15, size=(n_subj, n_nodes, n_nodes))
    mats = 0.5 * (mats + np.swapaxes(mats, 1, 2))
    for i in range(n_nodes):
        mats[:, i, i] = 0.0
    for u, v in planted_edges:
        mats[n_per_group:, u, v] += effect
        mats[n_per_group:, v, u] += effect

    data_path = root / "conn.npy"
    np.save(data_path, mats)

    pheno = pd.DataFrame(
        {
            "subject_id": [f"sub-{k:02d}" for k in range(n_subj)],
            "group": ["A"] * n_per_group + ["B"] * n_per_group,
        }
    )
    pheno_path = root / "pheno.csv"
    pheno.to_csv(pheno_path, index=False)

    manifest: dict = {
        "schema_version": 1,
        "name": "synth",
        "loader": "NumpyLoader",
        "paths": {
            "data_path": str(data_path.name),
            "pheno_path": str(pheno_path.name),
        },
        "metadata": {
            "subject_id_column": "subject_id",
            "group_column": "group",
        },
        "params": {
            "data_kind": "fisher_z",
        },
    }
    if with_networks:
        atlas_df = pd.DataFrame(
            {
                "roi_id": list(range(1, n_nodes + 1)),
                "name": [f"ROI_{k}" for k in range(n_nodes)],
                "network": ["Net1" if k < n_nodes // 2 else "Net2" for k in range(n_nodes)],
                "x": [float(k * 5) for k in range(n_nodes)],
                "y": [0.0] * n_nodes,
                "z": [0.0] * n_nodes,
            }
        )
        atlas_path = root / "atlas.csv"
        atlas_df.to_csv(atlas_path, index=False)
        manifest["paths"]["atlas_metadata"] = str(atlas_path.name)

    manifest_path = root / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return manifest_path, (root / "atlas.csv")


def test_interpret_alpha_override_propagates_to_methods_boilerplate(tmp_path: Path) -> None:
    manifest_path, _ = _write_two_group_dataset(tmp_path)
    out_dir = tmp_path / "run_out"
    cfg_path = tmp_path / "analysis.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": str(manifest_path.name),
                "design": {"type": "two-sample", "group": "group", "levels": ["A", "B"]},
                "inference": {"method": "tstat", "n_permutations": 19, "use_mp": False, "rng": 0},
                "alpha": 0.05,
            }
        ),
        encoding="utf-8",
    )
    assert main(["run", str(cfg_path), "--out", str(out_dir)]) == 0

    interp_dir = tmp_path / "interp_out"
    assert main(["interpret", str(out_dir), "--alpha", "0.01", "--methods", "--no-plots", "--out", str(interp_dir)]) == 0
    md_text = (interp_dir / "methods_boilerplate.md").read_text(encoding="utf-8")
    assert "alpha = 0.01" in md_text
    assert "alpha = 0.05" not in md_text

    # Direct API: provenance alpha acts as fallback when alpha=None, explicit alpha wins
    prov = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    assert "alpha = 0.05" in methods_boilerplate(prov)
    assert "alpha = 0.01" in methods_boilerplate(prov, alpha=0.01)


def test_run_nbs_reports_component_counts_in_stdout_and_provenance(tmp_path: Path) -> None:
    manifest_path, _ = _write_two_group_dataset(tmp_path, effect=3.0)
    out_dir = tmp_path / "nbs_out"
    cfg_path = tmp_path / "analysis_nbs.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": str(manifest_path.name),
                "design": {"type": "two-sample", "group": "group", "levels": ["A", "B"]},
                "inference": {
                    "method": "nbs",
                    "threshold": 4.0,
                    "n_permutations": 99,
                    "use_mp": False,
                    "rng": 1,
                },
                "alpha": 0.05,
            }
        ),
        encoding="utf-8",
    )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["run", str(cfg_path), "--out", str(out_dir)])
    assert rc == 0
    stdout = buf.getvalue()
    assert "component(s)" in stdout
    assert "edges)" in stdout

    prov = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    entry = prov["results"]["result"]
    assert "n_components" in entry
    assert entry["n_components"]["positive"] >= 1
    assert entry["n_significant"]["positive"] >= 3


def test_run_cnbs_exports_blocks_csv_and_reports_block_counts(tmp_path: Path) -> None:
    manifest_path, atlas_path = _write_two_group_dataset(
        tmp_path,
        n_nodes=6,
        planted_edges=((0, 1), (1, 2), (0, 2)),
        effect=3.0,
        with_networks=True,
    )
    out_dir = tmp_path / "cnbs_out"
    cfg_path = tmp_path / "analysis_cnbs.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": str(manifest_path.name),
                "atlas": str(atlas_path.name),
                "design": {"type": "two-sample", "group": "group", "levels": ["A", "B"]},
                "inference": {
                    "method": "cnbs",
                    "cnbs_correction": "bh",
                    "n_permutations": 99,
                    "use_mp": False,
                    "rng": 1,
                },
                "alpha": 0.05,
            }
        ),
        encoding="utf-8",
    )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["run", str(cfg_path), "--out", str(out_dir)])
    assert rc == 0
    stdout = buf.getvalue()
    assert "block(s)" in stdout
    assert "edges)" in stdout

    blocks_csv = out_dir / "blocks.csv"
    assert blocks_csv.is_file()
    df = pd.read_csv(blocks_csv)
    for col in ("network_a", "network_b", "n_edges", "mean_stat", "p_positive", "p_negative", "significant"):
        assert col in df.columns
    assert len(df) == 3  # 2 networks -> 3 blocks (Net1-Net1, Net1-Net2, Net2-Net2)

    prov = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    entry = prov["results"]["result"]
    assert "n_blocks" in entry
    assert entry["n_blocks"]["positive"] >= 1
    assert entry["n_significant"]["positive"] >= 3


def test_selftest_reference_checks_exact_edge_set_and_distinguishes_skipped() -> None:
    buf_json = io.StringIO()
    with contextlib.redirect_stdout(buf_json):
        rc = main(["selftest", "--reference", "--json"])
    assert rc == 0
    payload = json.loads(buf_json.getvalue())
    assert payload["ok"] is True
    assert payload["status"] == "PASS"
    assert payload["skipped"] == []
    assert "exact_edge_set_match=True" in payload["checks"]["reference_nbs_schizophrenia"]["detail"]

    buf_skip_json = io.StringIO()
    buf_skip_txt = io.StringIO()
    with mock.patch("conninfpy._cli_tools._find_nbs_schizophrenia_demo", return_value=None):
        with contextlib.redirect_stdout(buf_skip_json):
            rc_j = main(["selftest", "--reference", "--json"])
        with contextlib.redirect_stdout(buf_skip_txt):
            rc_t = main(["selftest", "--reference"])
    assert rc_j == 0
    assert rc_t == 0
    skip_payload = json.loads(buf_skip_json.getvalue())
    assert skip_payload["ok"] is True
    assert skip_payload["status"] == "SKIPPED"
    assert skip_payload["skipped"] == ["reference_nbs_schizophrenia"]
    assert "PASS (reference skipped)" in buf_skip_txt.getvalue()


def test_interpret_uses_bundles_list_from_provenance_and_ignores_stale_bundles(tmp_path: Path) -> None:
    manifest_path, _ = _write_two_group_dataset(tmp_path)
    out_dir = tmp_path / "run_out"
    cfg_path = tmp_path / "analysis.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": str(manifest_path.name),
                "design": {"type": "two-sample", "group": "group", "levels": ["A", "B"]},
                "inference": {"method": "tstat", "n_permutations": 19, "use_mp": False, "rng": 0},
            }
        ),
        encoding="utf-8",
    )
    assert main(["run", str(cfg_path), "--out", str(out_dir)]) == 0

    prov = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    assert prov["bundles"] == ["bundle.pkl"]

    # Simulate a leftover bundle from an earlier run with a different contrast name
    stale_bundle = out_dir / "bundle_stale_contrast.pkl"
    stale_bundle.write_bytes((out_dir / "bundle.pkl").read_bytes())

    interp_dir = tmp_path / "interp_out"
    assert main(["interpret", str(out_dir), "--no-plots", "--out", str(interp_dir)]) == 0
    assert (interp_dir / "node_summary.csv").is_file()
    assert not (interp_dir / "node_summary_stale_contrast.csv").exists()
    assert not list(interp_dir.glob("*stale_contrast*"))

    # Fallback when provenance.json has no "bundles" key
    prov.pop("bundles")
    (out_dir / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    interp_fallback = tmp_path / "interp_fallback"
    assert main(["interpret", str(out_dir), "--no-plots", "--out", str(interp_fallback)]) == 0
    assert (interp_fallback / "node_summary.csv").is_file()
    assert (interp_fallback / "node_summary_stale_contrast.csv").is_file()
