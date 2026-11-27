"""NETSTAT-84: ``conninfpy download-atlases`` CLI and reproducible BNA derivations.

Covers the package CLI (help, --verify-only on an empty cache, consent gate),
the thin ``scripts/download_atlases.py`` wrapper, and the deterministic
``bna246.csv`` / ``bna246_hierarchy.csv`` builders on synthetic sources
(byte-identical rebuilds, SHA-256 checked). When the official Brainnetome
originals are available locally, the builders must also reproduce the exact
published SHA-256 recorded in ``brainnetome_external.json``.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conninfpy.atlas import download as dl
from conninfpy.atlas.resources import atlas_cache_dir, brainnetome_data_dir, bundled_path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_download_atlases_help(capsys):
    from conninfpy.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["download-atlases", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--verify-only" in out and "--accept-licenses" in out and "--from-dir" in out


def test_cli_top_level_help_lists_download_atlases(capsys):
    from conninfpy.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "download-atlases" in capsys.readouterr().out


def test_cli_verify_only_empty_cache(tmp_path, capsys, monkeypatch):
    from conninfpy.cli import main

    empty = tmp_path / "empty_atlases"
    monkeypatch.setenv("CONNINFPY_ATLAS_DIR", str(empty))
    assert main(["download-atlases", "--verify-only"]) == 1
    out = capsys.readouterr().out
    assert "[MISSING] bna246.csv" in out
    assert "[MISSING] bna246_hierarchy.csv" in out
    assert "0/" in out  # nothing valid


def test_cli_refuses_without_consent(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("CONNINFPY_ATLAS_DIR", str(tmp_path / "empty_atlases"))
    assert dl.main(["--dest", str(tmp_path / "empty_atlases")]) == 2
    assert "--accept-licenses" in capsys.readouterr().err


def test_script_wrapper_verify_only(tmp_path):
    empty = tmp_path / "empty_atlases"
    empty.mkdir()
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "download_atlases.py"),
         "--verify-only", "--dest", str(empty)],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 1
    assert "[MISSING]" in r.stdout


def test_missing_inputs_reported_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("CONNINFPY_ATLAS_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CONNINFPY_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("CONNINFPY_BNA_DIR", raising=False)
    raw, missing = dl._build_derived("bna246.csv", tmp_path / "cache", None)
    assert raw is None
    assert "BN_Atlas_246_1mm.nii.gz" in missing
    assert "sources/BNA_subregions.xlsx" in missing
    raw, missing = dl._build_derived("bna246_hierarchy.csv", tmp_path / "cache", None)
    assert raw is None
    assert "bna246.csv (cache)" in missing


# ---------------------------------------------------------------------------
# Synthetic sources for the reproducible builders
# ---------------------------------------------------------------------------

def _synthetic_sources(tmp_path: Path) -> dict:
    """A 6-region (3 bilateral pairs) mini Brainnetome: 4 cortical + 2 thalamic."""
    import nibabel as nib
    import pandas as pd

    data = np.zeros((12, 12, 12), dtype=np.int16)
    blocks = {1: (2, 3, 4), 2: (8, 3, 4), 3: (2, 7, 4), 4: (8, 7, 4), 5: (4, 5, 8), 6: (7, 5, 8)}
    for rid, (x, y, z) in blocks.items():
        data[x:x + 2, y, z] = rid  # two voxels each -> half-integer centroids
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-10.0, -12.0, -8.0]
    nifti = tmp_path / "mini_1mm.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(nifti))

    lut = tmp_path / "BN_Atlas_246_LUT.txt"
    names = {1: "A8m_L", 2: "A8m_R", 3: "A8dl_L", 4: "A8dl_R", 5: "mPFtha_L", 6: "mPFtha_R"}
    lut.write_text("0 Unknown 0 0 0 0\n" + "".join(
        f"{i} {n} 0 0 0 0\n" for i, n in names.items()), encoding="utf-8")

    xlsx = tmp_path / "BNA_subregions.xlsx"
    table = pd.DataFrame([
        ["Lobe", "Gyrus", "Code", "L", "R", "Description"],
        ["Frontal Lobe", "SFG, Superior Frontal Gyrus", "SFG _L(R)_7_1", 1, 2, "A8m, medial area 8"],
        [None, None, "SFG _L(R)_7_2", 3, 4, "A8dl, dorsolateral area 8"],
        ["Subcortical Structures", "Tha, Thalamus", "Tha _L(R)_8_1", 5, 6, "mPFtha, medial pre-frontal thalamus"],
    ])
    table.to_excel(xlsx, header=False, index=False)

    yeo = tmp_path / "subregion_func_network_Yeo_updated.csv"
    yeo.write_text(
        "subregion_func_network_Yeo_updated\n"
        "Label,subregion_name,region,Yeo_7network,Yeo_17network\n"
        "1,A8m,SFG_L_7_1,6,17\n"
        "2,A8m,SFG_R_7_1,4,8\n"
        "3,A8dl,SFG_L_7_2,6,16\n"
        "4,A8dl,SFG_R_7_2,4,7\n"
        "5,mPFtha,Tha_L_8_1,0,0\n"
        "6,mPFtha,Tha_R_8_1,0,0\n",
        encoding="utf-8",
    )
    return {"nifti": nifti, "lut": lut, "xlsx": xlsx, "yeo": yeo, "blocks": blocks, "affine": affine}


def test_build_bna246_csv_reproducible_and_correct(tmp_path):
    src = _synthetic_sources(tmp_path)
    b1 = dl.build_bna246_csv(src["nifti"], src["lut"], src["xlsx"])
    b2 = dl.build_bna246_csv(src["nifti"], src["lut"], src["xlsx"])
    assert b1 == b2
    assert _sha(b1) == _sha(b2)
    assert b"\r\n" in b1  # published byte layout

    import io

    import pandas as pd
    df = pd.read_csv(io.BytesIO(b1))
    assert list(df.columns) == ["roi_id", "name", "network", "hemisphere", "x", "y", "z"]
    assert df.roi_id.tolist() == [1, 2, 3, 4, 5, 6]
    assert df.network.tolist() == ["Frontal"] * 4 + ["Subcortical"] * 2
    assert df.hemisphere.tolist() == ["L", "R"] * 3
    # Centroids: mean voxel index transformed by the affine, 2 decimals.
    for rid, (x, y, z) in src["blocks"].items():
        vox_mean = np.array([x + 0.5, y, z])
        expected = src["affine"] @ np.append(vox_mean, 1.0)
        row = df[df.roi_id == rid].iloc[0]
        assert [row.x, row.y, row.z] == [round(float(v), 2) for v in expected[:3]]
    assert b"-5.00," in b1  # fixed %.2f formatting, not shortest float repr


def test_build_bna_hierarchy_csv_reproducible_and_cross_checked(tmp_path):
    src = _synthetic_sources(tmp_path)
    bna = dl.build_bna246_csv(src["nifti"], src["lut"], src["xlsx"])
    bna_csv = tmp_path / "bna246.csv"
    bna_csv.write_bytes(bna)
    h1 = dl.build_bna_hierarchy_csv(bna_csv, src["xlsx"], src["yeo"])
    h2 = dl.build_bna_hierarchy_csv(bna_csv, src["xlsx"], src["yeo"])
    assert h1 == h2
    assert _sha(h1) == _sha(h2)
    assert b"\r\n" not in h1  # hierarchy uses LF

    import io

    import pandas as pd
    df = pd.read_csv(io.BytesIO(h1))
    assert len(df) == 6
    assert df.homolog_id.tolist() == [2, 1, 4, 3, 6, 5]
    assert df.structure.tolist() == ["cortex"] * 4 + ["thalamus"] * 2
    assert df.yeo7.tolist() == ["Frontoparietal", "Ventral Attention", "Frontoparietal",
                                "Ventral Attention", "not applicable", "not applicable"]
    assert df.code.tolist() == ["SFG_L_7_1", "SFG_R_7_1", "SFG_L_7_2", "SFG_R_7_2",
                                "Tha_L_8_1", "Tha_R_8_1"]

    # A name that disagrees with the subregion table must fail the cross-check.
    bad = pd.read_csv(io.BytesIO(bna))
    bad.loc[0, "name"] = "WRONG_L"
    bad_csv = tmp_path / "bad.csv"
    bad.to_csv(bad_csv, index=False)
    with pytest.raises(AssertionError):
        dl.build_bna_hierarchy_csv(bad_csv, src["xlsx"], src["yeo"])


# ---------------------------------------------------------------------------
# Real originals: reproduce the published SHA-256 exactly (when available)
# ---------------------------------------------------------------------------

def _real_sources_available() -> bool:
    cache = atlas_cache_dir()
    sources = brainnetome_data_dir() / "sources"
    return all((cache / f).is_file() for f in ("BN_Atlas_246_1mm.nii.gz",)) and all(
        (sources / f).is_file() for f in (
            "BN_Atlas_246_LUT.txt", "BNA_subregions.xlsx", "subregion_func_network_Yeo_updated.csv"))


@pytest.mark.skipif(not _real_sources_available(),
                    reason="official Brainnetome originals not available locally")
def test_real_sources_reproduce_published_sha256(tmp_path):
    manifest = json.loads(bundled_path("brainnetome_external.json").read_text(encoding="utf-8"))
    cache = atlas_cache_dir()
    sources = brainnetome_data_dir() / "sources"
    bna = dl.build_bna246_csv(cache / "BN_Atlas_246_1mm.nii.gz",
                              sources / "BN_Atlas_246_LUT.txt",
                              sources / "BNA_subregions.xlsx")
    assert _sha(bna) == manifest["atlas_files"]["bna246.csv"]["sha256"]
    bna_csv = tmp_path / "bna246.csv"
    bna_csv.write_bytes(bna)
    hier = dl.build_bna_hierarchy_csv(bna_csv, sources / "BNA_subregions.xlsx",
                                      sources / "subregion_func_network_Yeo_updated.csv")
    assert _sha(hier) == manifest["atlas_files"]["bna246_hierarchy.csv"]["sha256"]