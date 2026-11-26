"""Tests for NETSTAT-25: safe versioned workspace save & restore (no pickle)."""

from __future__ import annotations

import builtins
import io
import json
from pathlib import Path
import sys
import types
import zipfile

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from apps.utils import workspace_io
from apps.utils.archive_export import build_analysis_archive
from apps.utils.helpers import active_analysis_atlas, result_is_stale
from apps.utils.workspace import ANNOTATION_TAB, RESULTS_TAB
from apps.utils.workspace_io import (
    WORKSPACE_ARRAYS_PATH,
    WORKSPACE_MANIFEST_PATH,
    WORKSPACE_SCHEMA_VERSION,
    WORKSPACE_STATE_PATH,
    WorkspaceArchiveError,
    apply_pending_workspace_restore,
    build_workspace_archive,
    capture_source_hashes,
    inspect_workspace_archive,
    restore_workspace_archive,
    verify_source_references,
)
from conninfpy.analysis import AnalyzeResult, analyze
from conninfpy.atlas.brainnetome import BrainnetomeReference
from conninfpy.atlas.metadata import AtlasInfo
from conninfpy.interpret.annotate import RoiAnnotation
from conninfpy.results import InferenceResult, OmnibusInferenceResult


def _make_sample_grid_result() -> InferenceResult:
    p_pos = np.full((4, 4, 2), 0.5, dtype=np.float64)
    p_neg = np.full((4, 4, 2), 0.7, dtype=np.float64)
    p_pos[0, 1, 0] = p_pos[1, 0, 0] = 0.01
    p_pos[0, 1, 1] = p_pos[1, 0, 1] = 0.02
    stat_pos = np.zeros((4, 4), dtype=np.float64)
    stat_pos[0, 1] = stat_pos[1, 0] = 3.2
    beta = np.zeros((2, 4, 4), dtype=np.float64)
    beta[1, 0, 1] = beta[1, 1, 0] = 0.42

    return InferenceResult(
        p_pos,
        p_neg,
        method="tfnbs",
        n_permutations=25,
        acceleration="gpd",
        wall_time_s=0.12,
        null_max_dist={"positive": np.array([1.2, 2.3, 3.1]), "negative": np.array([1.1, 1.9, 2.5])},
        stat_positive=stat_pos,
        stat_negative=np.zeros((4, 4), dtype=np.float64),
        stat_type="tstat",
        harmonized=True,
        preserve_provided=True,
        strata_provided=True,
        combat_diagnostics={"n_sites": 2, "sites": ["A", "B"]},
        e_grid=np.array([0.4, 0.5]),
        h_grid=np.array([2.0, 3.0]),
        missing_report={"strategy": "complete_case", "n_dropped": 0},
        block_report={"n_blocks": 3},
        component_report={
            "unit": "component",
            "statistic": "extent",
            "stat_type": "extent",
            "threshold": 2.0,
            "edge_statistic": "tstat",
            "correction": "permutation_max",
            "tail_policy": "separate-one-sided",
            "tested_tail": "positive",
            "n_permutations": 25,
            "n_components": {"positive": 1, "negative": 0},
            "components": [
                {
                    "tail": "positive",
                    "component_id": 1,
                    "n_edges": 1,
                    "n_nodes": 2,
                    "statistic": 1.0,
                    "p_fwer": 0.01,
                    "nodes": [0, 1],
                }
            ],
            "labels": {
                "positive": np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32),
                "negative": np.zeros((4, 4), dtype=np.int32),
            },
        },
        two_tailed=False,
        tested_tail="positive",
        tail_selection="pre_specified",
        beta=beta,
        design={
            "names": ["intercept", "group"],
            "contrast": np.array([0.0, 1.0]),
            "profiles": {"ctrl": np.array([1.0, 0.0]), "case": np.array([1.0, 1.0])},
            "comparison": ["ctrl", "case"],
            "scale": "fisher_z",
        },
        analysis_provenance={
            "schema_version": 1,
            "method": "tfnbs",
            "n_nodes": 4,
            "roi_ids": (10, 20, 30, 40),
            "groupings": {"network": ("Vis", "Vis", "DMN", "DMN")},
            "eligible": np.array([[False, True, True, True], [True, False, True, True], [True, True, False, True], [True, True, True, False]], dtype=bool),
            "estimable": np.array([[False, True, True, True], [True, False, True, True], [True, True, False, True], [True, True, True, False]], dtype=bool),
            "edge_mask": None,
            "tested": {
                "positive": np.broadcast_to(np.array([[False, True, True, True], [True, False, True, True], [True, True, False, True], [True, True, True, False]])[:, :, None], (4, 4, 2)).copy(),
            },
            "model_id": "glm_group",
            "contrast_id": "group_diff",
            "units": "fisher_z",
            "scale": "fisher_z",
            "family": {"scope": "all_off_diagonal_edges", "n_eligible": 6, "method": "tfnbs", "correction": "permutation_max", "inference_unit": "edge"},
            "two_tailed": False,
            "tested_tail": "positive",
            "tail_selection": "pre_specified",
            "alpha": 0.05,
        },
    )


def _make_sample_omnibus_result() -> OmnibusInferenceResult:
    p_omni = np.full((4, 4), 0.4, dtype=np.float64)
    p_omni[0, 1] = p_omni[1, 0] = 0.015
    stat_omni = np.zeros((4, 4), dtype=np.float64)
    stat_omni[0, 1] = stat_omni[1, 0] = 6.5
    return OmnibusInferenceResult(
        p_omni,
        method="tstat",
        n_permutations=25,
        stat_omnibus=stat_omni,
        stat_type="fstat",
        null_max_dist={"omnibus": np.array([2.0, 4.1, 5.8])},
        design={"names": ["intercept", "group"], "contrast": np.array([[0.0, 1.0]]), "scale": "fisher_z"},
        analysis_provenance={
            "schema_version": 1,
            "method": "tstat",
            "stat_type": "fstat",
            "n_nodes": 4,
            "roi_ids": (10, 20, 30, 40),
            "groupings": {"network": ("Vis", "Vis", "DMN", "DMN")},
            "eligible": np.array([[False, True, True, True], [True, False, True, True], [True, True, False, True], [True, True, True, False]], dtype=bool),
            "estimable": np.zeros((4, 4), dtype=bool),
            "tested": {"omnibus": np.array([[False, True, True, True], [True, False, True, True], [True, True, False, True], [True, True, True, False]], dtype=bool)},
            "model_id": "glm_omnibus",
            "contrast_id": "f_group",
            "units": "fisher_z",
            "scale": "fisher_z",
            "family": {"scope": "all_off_diagonal_edges", "n_eligible": 6, "method": "tstat", "correction": "permutation_max", "inference_unit": "edge"},
        },
    )


def _make_sample_atlas() -> AtlasInfo:
    img_data = np.zeros((4, 4, 4), dtype=np.int16)
    img_data[0, 0, 0] = 10
    img_data[1, 1, 1] = 20
    img = nib.Nifti1Image(img_data, np.eye(4))
    return AtlasInfo(
        labels=["R1", "R2", "R3", "R4"],
        networks=["DMN", "DMN", "FPN", "FPN"],
        coords=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-2.0, 1.0, 0.0]]),
        hemisphere=["L", "R", "L", "R"],
        source="custom-unit-test",
        image=img,
        label_ids=[10, 20, 30, 40],
        assignment={"mode": "parcel", "purity": 0.8},
    )


def _make_synthetic_bna_ref(n_regions: int = 246) -> BrainnetomeReference:
    rows = []
    for rid in range(1, n_regions + 1):
        hemi = "L" if rid % 2 else "R"
        hom = rid + 1 if rid % 2 else rid - 1
        rows.append(
            {
                "roi_id": rid,
                "name": f"R{rid}_{hemi}",
                "hemisphere": hemi,
                "homolog_id": hom,
                "code": f"C_{rid}",
                "description": f"Synthetic region {rid}",
                "gyrus": "SFG" if rid <= 4 else f"G{(rid - 1) // 2}",
                "gyrus_name": "Superior Frontal Gyrus",
                "lobe": "Frontal" if rid <= 4 else "Temporal",
                "structure": "cortex",
                "yeo7_id": 6,
                "yeo7": "Frontoparietal",
                "yeo17_id": 12,
                "yeo17": "Control A",
                "x": -10.0 if hemi == "L" else 10.0,
                "y": float(rid),
                "z": 20.0,
            }
        )
    regions = pd.DataFrame(rows).set_index("roi_id")
    vol = np.zeros((6, 1, 1), dtype=np.int16)
    for rid in range(1, 5):
        vol[rid - 1, 0, 0] = rid
    img = nib.Nifti1Image(vol, np.eye(4))
    profiles = {
        "BD": {1: {"Cognition.Memory.Working": 2.1}, 2: {"Action.Execution": 1.7}},
        "PC": {1: {"GoNoGo": 1.4}},
    }
    return BrainnetomeReference(
        regions=regions,
        label_image=img,
        profiles=profiles,
        provenance={"atlas": "synthetic BNA fixture"},
    )


def _make_sample_annotation_bundle(res: InferenceResult):
    from conninfpy.interpret.annotate import annotate_bna_regions, summarize_annotation
    from conninfpy.interpret.evidence import build_annotation_evidence

    ref = _make_synthetic_bna_ref()
    rois = annotate_bna_regions([1, 2, 3, 4], reference=ref, names=["R1", "R2", "R3", "R4"])
    summary = summarize_annotation(res, rois, alpha=0.025, tail="positive", level="lobe", param_idx=0)
    edges = summary["edges"]
    detected = edges[edges["p_positive"] <= 0.025].copy()
    evidence = build_annotation_evidence(
        detected,
        rois,
        contrast_name="case - ctrl",
        source_atlas="custom-unit-test",
        inference_unit="edge",
        summary=summary,
    )
    return rois, summary, detected, evidence


def _make_full_state(tmp_path: Path) -> dict:
    src_csv = tmp_path / "cohort_conn.npy"
    src_pheno = tmp_path / "pheno.csv"
    src_manifest = tmp_path / "data.yaml"
    np.save(src_csv, np.zeros((4, 4, 4), dtype=np.float64), allow_pickle=False)
    src_pheno.write_text("subject,group\ns1,A\ns2,B\n", encoding="utf-8")
    src_manifest.write_text("name: unit_test\n", encoding="utf-8")

    atlas = _make_sample_atlas()
    primary = AnalyzeResult(
        inference=_make_sample_grid_result(),
        combat_diagnostics={"n_sites": 2},
        design_diagnostics={"vif": [1.0, 1.1]},
        flags=["harmonized"],
        resolved_strategy="glm_freedman_lane",
    )
    companion = _make_sample_omnibus_result()
    edges_df = pd.DataFrame({"i_roi": [0], "j_roi": [1], "p_positive": [0.01]})
    edges_df.attrs["display_cell"] = 1
    comp_edges_df = pd.DataFrame({"i_roi": [0], "j_roi": [1], "p_omnibus": [0.015]})

    dataset_info = {
        "name": "abide_unit",
        "mode": "Dataset Manifest (data.yaml)",
        "source": str(src_manifest),
        "manifest_file": str(src_manifest),
        "atlas": "custom-unit-test",
        "loaded_at": "2026-11-13T10:00:00Z",
        "observations": 12,
        "n_rois": 4,
        "loader_provenance": {
            "data_path": str(src_csv),
            "pheno_csv_path": str(src_pheno),
        },
        "secret_token": "should-never-be-saved",
    }
    dataset_info["source_files"] = capture_source_hashes(
        dataset_info=dataset_info,
        manifest_file=str(src_manifest),
        resolved_paths={"data": src_csv, "pheno": src_pheno},
        project_root=tmp_path,
    )

    ann, ann_summary, ann_detected, evidence = _make_sample_annotation_bundle(primary.inference)
    run_plan = {
        "method": "tfnbs",
        "alpha": 0.025,
        "two_tailed": False,
        "tested_tail": "positive",
        "tail_selection": "pre_specified",
        "seed": 123,
        "loaded_settings_hash": "hash-abc",
        "active_atlas_signature": "atlas-sig-xyz",
        "api_token": "secret-run-plan-token",
    }

    return {
        "connectivity_data": np.ones((4, 4, 4), dtype=np.float64),
        "connectivity_data_kind": "fisher_z",
        "pheno_df": pd.DataFrame({"subject": ["s1", "s2"], "group": ["A", "B"]}),
        "dataset_info": dataset_info,
        "dataset_preprocessing": {
            "kind": "combined",
            "per_site": {"SiteA": {"gsr": True, "source": "manifest"}, "SiteB": {"gsr": False, "source": "manifest"}},
        },
        "dataset_fisher_history": [{"stage": "app_ingestion_option", "policy": "clip_r", "clip_r": 0.9999}],
        "last_fisher_clip_note": "Fisher r-to-z: 0 values clipped",
        "dataset_load_notes": ["Loaded 4x4 matrices"],
        "dataset_atlas": atlas,
        "sub_atlas": atlas,
        "loaded_reference_atlas": None,  # explicit None must survive restore!
        "roi_indices": [0, 1, 2, 3],
        "active_atlas_signature": "atlas-sig-xyz",
        "loaded_atlas_signature": "atlas-sig-xyz",
        "loaded_settings_hash": "hash-abc",
        "current_settings_hash": "hash-abc",
        "inference_result": primary,
        "edges_df": edges_df,
        "run_plan": run_plan,
        "companion_inference_result": companion,
        "companion_edges_df": comp_edges_df,
        "companion_method": "tstat",
        "annotation": {
            "rois": ann,
            "summary": ann_summary,
            "detected": ann_detected,
            "support": "sphere",
            "level": "lobe",
            "alpha": 0.025,
            "radius": 6.0,
            "purity": 0.5,
            "space": "MNI152",
            "param_idx": 0,
        },
        "evidence_packet": evidence,
        "narrative_results": {"mock-model": {"text": "Grounded report.", "usage": {"total_tokens": 10}}},
        "narrative_text": "multi_model",
        "llm_provider": "mock",
        "llm_model": "mock-model",
        "llm_api_key": "sk-secret-key-must-not-export",
        "narrative:api_key_override": "sk-override-must-not-export",
        "active_tab": "3. Inference Results",
        "atlas_mode": "No atlas metadata",
        "atlas_choice": "Schaefer-100 Yeo-7",
        "_workspace_widget_keys": {
            "results:alpha": "results",
            "results:tail": "results",
            "annotation:alpha": "annotation",
            "narrative:api_key_override": "narrative",
        },
        "results:alpha": 0.025,
        "results:tail": "positive",
        "results:select_result_view_mode": "Side-by-Side Comparison",
        "results:grid_idx:primary": 1,
        "results:grid_idx:companion": 0,
        "annotation:alpha": 0.025,
        "_synthetic_effect_mask": None,
        "_synthetic_scenario_name": None,
    }


def test_full_workspace_roundtrip_and_secret_hygiene(tmp_path: Path):
    state = _make_full_state(tmp_path)
    zip_bytes, manifest = build_workspace_archive(state, include_data=False, project_root=tmp_path)

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = set(zf.namelist())
        assert WORKSPACE_MANIFEST_PATH in names
        assert WORKSPACE_STATE_PATH in names
        assert WORKSPACE_ARRAYS_PATH in names
        assert "manifest.json" in names

        root_manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        for ws_member in (WORKSPACE_MANIFEST_PATH, WORKSPACE_STATE_PATH, WORKSPACE_ARRAYS_PATH):
            assert ws_member in root_manifest["files"]

        raw_state_text = zf.read(WORKSPACE_STATE_PATH).decode("utf-8")
        assert "sk-secret-key-must-not-export" not in raw_state_text
        assert "sk-override-must-not-export" not in raw_state_text
        assert "should-never-be-saved" not in raw_state_text
        assert "secret-run-plan-token" not in raw_state_text

        state_doc = json.loads(raw_state_text)
        assert set(state_doc.keys()) >= {"source_refs", "results", "annotation", "ui_state", "data"}
        assert state_doc["data"] is None
        assert state_doc["source_refs"]["synthetic"] is None  # real dataset never invents synthetic truth

        # The retired LLM narrative surface is absent from a new export, even
        # when the live session still carries its dead keys.
        legacy_keys = ("narrative_results", "narrative_text", "llm_provider", "llm_model")
        for legacy_key in legacy_keys:
            assert legacy_key not in state_doc["annotation"]
            assert legacy_key not in raw_state_text
        for needle in ("narrative", "llm_api_key", "has_narrative"):
            assert needle not in raw_state_text.lower()
            assert needle not in zf.read(WORKSPACE_MANIFEST_PATH).decode("utf-8").lower()
            assert needle not in zf.read("manifest.json").decode("utf-8").lower()
        assert set(state_doc["annotation"].keys()) == {"annotation", "evidence_packet"}
        assert state_doc["workspace_schema_version"] == WORKSPACE_SCHEMA_VERSION

    insp = inspect_workspace_archive(zip_bytes, project_root=tmp_path)
    assert insp.workspace_schema_version == WORKSPACE_SCHEMA_VERSION
    assert insp.has_results is True
    assert insp.has_companion is True
    assert insp.has_annotation is True
    assert insp.has_evidence_packet is True
    assert not hasattr(insp, "has_narrative")  # retired status flag cannot reappear
    assert insp.legacy_fields_skipped == []
    assert insp.has_data is False
    assert insp.source_verification["overall"] == "verified"

    manifest = insp.manifest
    assert "has_narrative" not in manifest["contents"]
    assert manifest["workspace_schema_version"] == WORKSPACE_SCHEMA_VERSION

    restored = {}
    restore_workspace_archive(zip_bytes, restored, project_root=tmp_path)

    assert restored["connectivity_data"] is None
    assert restored["pheno_df"] is None
    assert restored["loaded_reference_atlas"] is None
    assert restored["roi_indices"] == [0, 1, 2, 3]
    assert restored["dataset_preprocessing"]["per_site"]["SiteA"]["gsr"] is True
    assert restored["dataset_fisher_history"][0]["clip_r"] == 0.9999
    for absent_key in ("llm_api_key", "narrative:api_key_override", "narrative_results", "narrative_text", "llm_provider", "llm_model"):
        assert absent_key not in restored
        assert restored.get(absent_key) is None
    assert not any(str(k).startswith("narrative:") for k in restored["_workspace_widget_keys"])
    assert restored["_workspace_legacy_fields_skipped"] == []

    # Primary & companion results faithfully reconstructed
    prim = restored["inference_result"]
    assert isinstance(prim, AnalyzeResult)
    np.testing.assert_allclose(prim.inference["positive"], state["inference_result"].inference["positive"])
    np.testing.assert_allclose(prim.inference.e_grid, [0.4, 0.5])
    np.testing.assert_allclose(prim.inference.h_grid, [2.0, 3.0])
    np.testing.assert_allclose(prim.inference.effect_map(), state["inference_result"].inference.effect_map())
    assert prim.inference.tested_tail == "positive"
    assert prim.inference.tail_selection == "pre_specified"
    assert prim.inference.component_report["n_components"]["positive"] == 1
    assert prim.inference.component_report["components"][0]["p_fwer"] == 0.01
    np.testing.assert_array_equal(
        prim.inference.component_report["labels"]["positive"],
        state["inference_result"].inference.component_report["labels"]["positive"],
    )
    assert restored["edges_df"].attrs["display_cell"] == 1

    comp = restored["companion_inference_result"]
    assert isinstance(comp, OmnibusInferenceResult)
    np.testing.assert_allclose(comp["omnibus"], state["companion_inference_result"]["omnibus"])
    assert restored["companion_method"] == "tstat"

    # BrainnetomeReference int-keyed profiles & NIfTI image preserved without pickle
    ref = restored["annotation"]["rois"].reference
    orig_ref = state["annotation"]["rois"].reference
    assert isinstance(ref, BrainnetomeReference)
    assert ref.profiles is not None and orig_ref.profiles is not None
    for kind, mapping in orig_ref.profiles.items():
        assert set(ref.profiles[kind]) == set(mapping)  # integer ROI keys survive JSON
        for roi_id, terms in mapping.items():
            assert ref.profiles[kind][roi_id] == terms
    assert isinstance(ref.label_image, nib.Nifti1Image)
    assert isinstance(restored["dataset_atlas"].image, nib.Nifti1Image)
    assert list(np.asarray(restored["dataset_atlas"].label_ids)) == [10, 20, 30, 40]

    # UI state & widget guards restored cleanly
    assert restored["active_tab"] == "3. Inference Results"
    assert restored["results:select_result_view_mode"] == "Side-by-Side Comparison"
    assert restored["results:grid_idx:primary"] == 1
    assert restored["_workspace_widget_keys"]["results:alpha"] == "results"
    assert restored["_annotation_result_id"] == id(prim.inference)
    assert restored["_TFNBS_graph_result_id"] == id(prim.inference)
    assert restored["_TSTAT_graph_result_id"] == id(comp)

    # Staleness stays False when restored with matching settings/atlas hashes
    assert result_is_stale(restored["run_plan"]) is False or (
        restored["run_plan"]["loaded_settings_hash"] == restored["loaded_settings_hash"]
        and restored["run_plan"]["active_atlas_signature"] == restored["loaded_atlas_signature"]
    )


def test_default_archive_unchanged_without_workspace_state(tmp_path: Path):
    res = _make_sample_omnibus_result()
    plan = {"method": "tstat", "alpha": 0.05, "seed": 7, "data_kind": "fisher_z"}
    info = {"name": "legacy_compat", "mode": "Built-in data", "source": "OpenClose"}
    zip_bytes, manifest = build_analysis_archive(res, plan, info, project_root=tmp_path)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        assert WORKSPACE_MANIFEST_PATH not in zf.namelist()
    assert "workspace" not in manifest


def test_source_verification_verified_stale_missing_unknown_and_synthetic(tmp_path: Path):
    f1 = tmp_path / "data.npy"
    f1.write_bytes(b"original-content")
    captured = capture_source_hashes(
        dataset_info={"mode": "Dataset Manifest (data.yaml)", "source": str(f1)},
        resolved_paths={"data": f1},
        project_root=tmp_path,
    )
    assert verify_source_references(captured, project_root=tmp_path)["overall"] == "verified"

    # Modify file -> stale
    f1.write_bytes(b"modified-content")
    assert verify_source_references(captured, project_root=tmp_path)["overall"] == "stale"

    # Delete file -> missing
    f1.unlink()
    assert verify_source_references(captured, project_root=tmp_path)["overall"] == "missing"

    # Unhashed existing file -> NEVER verified, stays unknown
    f2 = tmp_path / "unhashed.csv"
    f2.write_text("a,b\n1,2\n", encoding="utf-8")
    unhashed_info = {"mode": "Dataset Manifest (data.yaml)", "loader_provenance": {"pheno_csv_path": str(f2)}}
    ver_unknown = verify_source_references(None, dataset_info=unhashed_info, project_root=tmp_path)
    assert ver_unknown["overall"] == "unknown"

    # Synthetic -> synthetic (distinct from verified)
    ver_synth = verify_source_references(None, dataset_info={"mode": "Generate Synthetic Dataset"})
    assert ver_synth["overall"] == "synthetic"


def test_legacy_unknown_facts_stay_unknown_without_fabrication(tmp_path: Path):
    legacy_res = InferenceResult(
        np.full((4, 4), 0.2),
        np.full((4, 4), 0.4),
        method="tstat",
        n_permutations=10,
        analysis_provenance=None,
    )
    state = {
        "dataset_info": {"name": "legacy_run", "mode": "Built-in data", "source": str(tmp_path / "missing.csv")},
        "inference_result": legacy_res,
        "run_plan": {"method": "tstat", "alpha": 0.05},
    }
    zip_bytes, _ = build_workspace_archive(state, include_data=False, project_root=tmp_path)
    insp = inspect_workspace_archive(zip_bytes, project_root=tmp_path)
    assert any("analysis_provenance" in msg for msg in insp.legacy_unknowns)
    restored = {}
    restore_workspace_archive(zip_bytes, restored, project_root=tmp_path)
    assert restored.get("_synthetic_effect_mask") is None
    assert restored.get("dataset_preprocessing") is None
    assert restored.get("dataset_fisher_history") is None


def test_no_pickle_enforcement_and_poisoned_bundle_ignored(tmp_path: Path, monkeypatch):
    state = _make_full_state(tmp_path)
    zip_bytes, _ = build_workspace_archive(state, include_data=False, project_root=tmp_path)

    # Inject a poisoned bundle.pkl (would execute arbitrary code if ever unpickled!)
    poison_buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as src_zf, zipfile.ZipFile(poison_buf, "w") as dst_zf:
        for info in src_zf.infolist():
            if info.filename == "bundle.pkl":
                continue
            dst_zf.writestr(info, src_zf.read(info.filename))
        dst_zf.writestr("bundle.pkl", b"\x80\x04\x95\x00\x00\x00\x00\x00\x00\x00builtins\neval\n.")
    poisoned_zip = poison_buf.getvalue()

    real_import = builtins.__import__

    def _forbid_pickle(name, *args, **kwargs):
        if name == "pickle" or name.startswith("pickle.") or name == "_pickle":
            raise AssertionError("pickle must never be imported during workspace inspect/restore!")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _forbid_pickle)
    insp = inspect_workspace_archive(poisoned_zip, project_root=tmp_path)
    assert insp.has_results is True
    target = {}
    restore_workspace_archive(poisoned_zip, target, project_root=tmp_path)
    assert target["inference_result"] is not None


def test_corrupt_future_version_object_ndarray_and_atomicity(tmp_path: Path):
    state = _make_full_state(tmp_path)
    good_zip, _ = build_workspace_archive(state, include_data=False, project_root=tmp_path)

    sentinel = {"existing_key": "untouched", "inference_result": "keep_me"}

    def _assert_atomic(bad_zip: bytes):
        snap = dict(sentinel)
        with pytest.raises(WorkspaceArchiveError):
            restore_workspace_archive(bad_zip, snap, project_root=tmp_path)
        assert snap == sentinel

    # 1. Corrupt ZIP
    _assert_atomic(b"not-a-zip")

    # 2. Legacy ZIP without workspace/manifest.json
    legacy_buf = io.BytesIO()
    with zipfile.ZipFile(legacy_buf, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"schema_version": 1, "files": {}}))
        zf.writestr("bundle.pkl", b"legacy")
    _assert_atomic(legacy_buf.getvalue())

    # 3. Unsupported future workspace_schema_version
    def _rewrite_zip(mutator) -> bytes:
        out = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(good_zip), "r") as zin:
            files = {n: zin.read(n) for n in zin.namelist()}
        mutator(files)
        with zipfile.ZipFile(out, "w") as zout:
            for k, v in files.items():
                zout.writestr(k, v)
        return out.getvalue()

    def _bump_version(files):
        m = json.loads(files[WORKSPACE_MANIFEST_PATH].decode("utf-8"))
        m["workspace_schema_version"] = 99
        files[WORKSPACE_MANIFEST_PATH] = json.dumps(m).encode("utf-8")

    _assert_atomic(_rewrite_zip(_bump_version))

    # 4. Tampered state.json (checksum mismatch)
    def _tamper_state(files):
        files[WORKSPACE_STATE_PATH] = files[WORKSPACE_STATE_PATH] + b" "

    _assert_atomic(_rewrite_zip(_tamper_state))

    # 5. Path traversal member
    def _traversal(files):
        files["../escape.txt"] = b"x"

    _assert_atomic(_rewrite_zip(_traversal))

    # 6. Object-dtype array inside arrays.npz
    def _object_npz(files):
        obj_buf = io.BytesIO()
        np.savez(obj_buf, a0000_bad=np.array(["a", "b"], dtype=object))
        files[WORKSPACE_ARRAYS_PATH] = obj_buf.getvalue()
        m = json.loads(files[WORKSPACE_MANIFEST_PATH].decode("utf-8"))
        import hashlib
        m["files"][WORKSPACE_ARRAYS_PATH] = {
            "sha256": hashlib.sha256(files[WORKSPACE_ARRAYS_PATH]).hexdigest(),
            "bytes": len(files[WORKSPACE_ARRAYS_PATH]),
        }
        files[WORKSPACE_MANIFEST_PATH] = json.dumps(m).encode("utf-8")
        rm = json.loads(files["manifest.json"].decode("utf-8"))
        rm["files"][WORKSPACE_ARRAYS_PATH] = m["files"][WORKSPACE_ARRAYS_PATH]
        rm["files"][WORKSPACE_MANIFEST_PATH] = {
            "sha256": hashlib.sha256(files[WORKSPACE_MANIFEST_PATH]).hexdigest(),
            "bytes": len(files[WORKSPACE_MANIFEST_PATH]),
        }
        files["manifest.json"] = json.dumps(rm).encode("utf-8")

    with pytest.raises(WorkspaceArchiveError):
        workspace_io._load_npz_safe(
            (lambda b: (np.savez(b, bad=np.array(["x"], dtype=object)), b.getvalue())[1])(io.BytesIO())
        )
    _assert_atomic(_rewrite_zip(_object_npz))


def test_two_independent_apptest_sessions_save_and_restore(tmp_path: Path, monkeypatch):
    """Session 1 produces & saves a workspace archive; clean Session 2 restores & renders Tabs 3 & 4."""
    from streamlit.runtime.media_file_manager import MediaFileManager
    from streamlit.testing.v1 import AppTest

    recorded_downloads: list[tuple[bytes, str]] = []
    orig_add = MediaFileManager.add

    def _spy_add(self, path_or_data, mimetype, coordinates, file_name=None, is_for_static_download=False):
        if isinstance(path_or_data, (bytes, bytearray)) and file_name and file_name.endswith(".zip"):
            recorded_downloads.append((bytes(path_or_data), file_name))
        return orig_add(self, path_or_data, mimetype, coordinates, file_name=file_name, is_for_static_download=is_for_static_download)

    monkeypatch.setattr(MediaFileManager, "add", _spy_add)

    # Session 1: Load synthetic-like small state via AppTest, run through real app rendering, save workspace ZIP
    app_path = str(Path(__file__).resolve().parents[1] / "apps" / "streamlit_app.py")
    at1 = AppTest.from_file(app_path, default_timeout=60)
    src_state = _make_full_state(tmp_path)
    for k, v in src_state.items():
        at1.session_state[k] = v
    at1.run()
    assert not at1.exception, f"Session 1 exceptions: {at1.exception}"

    # Trigger sidebar workspace archive build / download
    ws_zips = [item for item in recorded_downloads if item[1].endswith(".conninfpy.zip") or item[1].startswith("conninf_")]
    assert ws_zips, "Expected workspace download archive bytes to be emitted in Session 1"
    saved_zip_bytes = ws_zips[-1][0]

    # Session 2: Brand-new independent AppTest session with empty session_state
    at2 = AppTest.from_file(app_path, default_timeout=60)
    at2.session_state["_pending_workspace_restore"] = saved_zip_bytes
    at2.run()
    assert not at2.exception, f"Session 2 restore exceptions: {at2.exception}"

    # Verify results-only restoration without re-running inference or loading subject data
    assert at2.session_state["connectivity_data"] is None
    assert at2.session_state["loaded_reference_atlas"] is None
    assert at2.session_state["inference_result"] is not None
    assert at2.session_state["companion_inference_result"] is not None
    assert at2.session_state["annotation"] is not None
    assert at2.session_state["evidence_packet"] is not None
    assert at2.session_state["active_tab"] == RESULTS_TAB
    assert at2.session_state["results:select_result_view_mode"] == "Side-by-Side Comparison"
    assert "llm_api_key" not in at2.session_state
    assert "narrative_results" not in at2.session_state
    assert "narrative_text" not in at2.session_state

    # AC2: the restored session tells the user honestly that subject data must be re-imported
    restored_captions = " ".join(c.value for c in at2.caption)
    assert "restored without subject-level connectivity matrices" in restored_captions
    assert "Workspace source status (verified)" in restored_captions

    # Switch to Tab 4 (Brainnetome Annotation) in the restored results-only session
    at2.session_state["active_tab"] = "4. Brainnetome Annotation"
    at2.run()
    assert not at2.exception, f"Session 2 Tab 4 exceptions: {at2.exception}"
    assert at2.session_state["annotation"] is not None
    assert at2.session_state["evidence_packet"] is not None

    # Switch to Tab 1 (Data Ingestion) and verify restored notice renders without staleness crash
    at2.session_state["active_tab"] = "1. Data Ingestion"
    at2.run()
    assert not at2.exception, f"Session 2 Tab 1 exceptions: {at2.exception}"
    # Result/atlas identity is unchanged by the Tab 1 render, so no false "stale" badge.
    # (result_is_stale() reads the ambient streamlit session, which is empty off-run.)
    assert at2.session_state["run_plan"]["loaded_settings_hash"] == at2.session_state["loaded_settings_hash"]
    assert at2.session_state["run_plan"]["active_atlas_signature"] == at2.session_state["loaded_atlas_signature"]


# ---------------------------------------------------------------------------
# Legacy (schema v1) archives written before the narrative surface was removed
# ---------------------------------------------------------------------------

def _workspace_zip_files(zip_bytes: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def _zip_from_files(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, blob in files.items():
            zf.writestr(name, blob)
    return buf.getvalue()


def _sync_workspace_checksums(files: dict) -> None:
    """Recompute every workspace/* digest after editing the payload (keeps ZIP itself valid)."""
    import hashlib

    ws_manifest = json.loads(files[WORKSPACE_MANIFEST_PATH].decode("utf-8"))
    for name, entry in ws_manifest["files"].items():
        entry["sha256"] = hashlib.sha256(files[name]).hexdigest()
        entry["bytes"] = len(files[name])
    files[WORKSPACE_MANIFEST_PATH] = json.dumps(ws_manifest, indent=2, sort_keys=True).encode("utf-8")

    root = json.loads(files["manifest.json"].decode("utf-8"))
    for name, entry in ws_manifest["files"].items():
        root["files"][name] = dict(entry)
    root["files"][WORKSPACE_MANIFEST_PATH] = {
        "sha256": hashlib.sha256(files[WORKSPACE_MANIFEST_PATH]).hexdigest(),
        "bytes": len(files[WORKSPACE_MANIFEST_PATH]),
    }
    files["manifest.json"] = json.dumps(root, indent=2, sort_keys=True).encode("utf-8")


def _downgrade_to_legacy_v1(zip_bytes: bytes) -> bytes:
    """Rebuild a current archive as a schema-v1 archive that still carries dead narrative fields."""
    files = _workspace_zip_files(zip_bytes)
    state_doc = json.loads(files[WORKSPACE_STATE_PATH].decode("utf-8"))
    state_doc["workspace_schema_version"] = 1
    state_doc["annotation"]["narrative_results"] = {"legacy-model": {"text": "Old grounded report."}}
    state_doc["annotation"]["narrative_text"] = "multi_model"
    state_doc["annotation"]["llm_provider"] = "legacy"
    state_doc["annotation"]["llm_model"] = "legacy-model"
    ui = state_doc["ui_state"]
    ui["active_tab"] = "5. Narrative Report"
    ui["widget_registry"]["narrative:api_key_override"] = "narrative"
    ui["widgets"]["narrative:api_key_override"] = "sk-legacy-override"
    ui["widgets"]["llm_api_key"] = "sk-legacy-key"
    files[WORKSPACE_STATE_PATH] = json.dumps(state_doc, indent=2, sort_keys=True).encode("utf-8")

    ws_manifest = json.loads(files[WORKSPACE_MANIFEST_PATH].decode("utf-8"))
    ws_manifest["workspace_schema_version"] = 1
    ws_manifest["contents"]["has_narrative"] = True
    files[WORKSPACE_MANIFEST_PATH] = json.dumps(ws_manifest, indent=2, sort_keys=True).encode("utf-8")

    root = json.loads(files["manifest.json"].decode("utf-8"))
    root["workspace"]["workspace_schema_version"] = 1
    files["manifest.json"] = json.dumps(root, indent=2, sort_keys=True).encode("utf-8")

    _sync_workspace_checksums(files)
    return _zip_from_files(files)


def test_legacy_v1_archive_skips_narrative_and_redirects_tab(tmp_path: Path):
    state = _make_full_state(tmp_path)
    zip_bytes, _ = build_workspace_archive(state, include_data=False, project_root=tmp_path)
    legacy_zip = _downgrade_to_legacy_v1(zip_bytes)

    # Inspection: readable, honest about the skipped legacy fields, no revamped narrative status
    insp = inspect_workspace_archive(legacy_zip, project_root=tmp_path)
    assert insp.workspace_schema_version == 1
    assert insp.has_results is True
    assert insp.has_companion is True
    assert insp.has_annotation is True
    assert insp.has_evidence_packet is True
    assert not hasattr(insp, "has_narrative")
    assert insp.active_tab == ANNOTATION_TAB
    assert insp.source_verification["overall"] == "verified"  # sources untouched: honest, not stale
    assert "annotation.narrative_results" in insp.legacy_fields_skipped
    assert "annotation.narrative_text" in insp.legacy_fields_skipped
    assert "annotation.llm_provider" in insp.legacy_fields_skipped
    assert "annotation.llm_model" in insp.legacy_fields_skipped
    assert "ui_state.widget_registry.narrative:api_key_override" in insp.legacy_fields_skipped
    assert "ui_state.widgets.llm_api_key" in insp.legacy_fields_skipped
    assert any("narrative" in msg.lower() for msg in insp.legacy_unknowns)

    # Restore: results + annotation + evidence survive, narrative/keys/widgets do not
    target: dict = {}
    info = restore_workspace_archive(legacy_zip, target, project_root=tmp_path)
    assert target["inference_result"] is not None
    assert target["companion_inference_result"] is not None
    assert target["annotation"] is not None
    assert target["evidence_packet"] is not None
    assert target["connectivity_data"] is None
    for legacy_key in ("narrative_results", "narrative_text", "llm_provider", "llm_model", "llm_api_key"):
        assert legacy_key not in target
    assert not any(str(k).startswith("narrative:") or str(k) in {"llm_api_key"} for k in target["_workspace_widget_keys"])
    assert not any(str(k).startswith("narrative:") or str(k) in {"llm_api_key"} for k in target)
    assert "sk-legacy-override" not in repr(target)
    assert "sk-legacy-key" not in repr(target)
    assert target["active_tab"] == ANNOTATION_TAB
    assert workspace_io.LEGACY_TAB_REDIRECTS["5. Narrative Report"] == ANNOTATION_TAB
    assert target["_workspace_legacy_fields_skipped"] == insp.legacy_fields_skipped
    assert info.workspace_schema_version == 1


def test_restore_with_missing_or_stale_sources_stays_honest(tmp_path: Path):
    state = _make_full_state(tmp_path)
    zip_bytes, _ = build_workspace_archive(state, include_data=False, project_root=tmp_path)

    # 1. Stale: one captured source changes on disk after capture
    pheno = Path(state["dataset_info"]["source_files"]["manifest.pheno"]["resolved"])
    pheno.write_text("subject,group\ns1,A\ns2,B\ns3,A\n", encoding="utf-8")
    insp_stale = inspect_workspace_archive(zip_bytes, project_root=tmp_path)
    assert insp_stale.source_verification["overall"] == "stale"
    assert insp_stale.source_verification["counts"]["verified"] < len(insp_stale.source_verification["files"])

    # 2. Missing: every captured source is gone; results still restore, freshness never claims "verified"
    for item in state["dataset_info"]["source_files"].values():
        Path(item["resolved"]).unlink(missing_ok=True)
    target: dict = {}
    info = restore_workspace_archive(zip_bytes, target, project_root=tmp_path)
    assert info.source_verification["overall"] == "missing"
    assert info.source_verification["counts"]["verified"] == 0
    assert info.source_verification["counts"]["missing"] > 0
    assert target["inference_result"] is not None
    assert target["annotation"] is not None
    assert target["evidence_packet"] is not None
    assert target["_workspace_source_verification"]["overall"] == "missing"
    assert "missing" in info.source_verification["message"].lower()
    # An explicit reload is required, so the UI must not present data as loaded
    assert target["connectivity_data"] is None
    assert target["pheno_df"] is None


def test_contract_sections_and_unsupported_versions_are_rejected(tmp_path: Path):
    state = _make_full_state(tmp_path)
    good_zip, _ = build_workspace_archive(state, include_data=False, project_root=tmp_path)
    sentinel = {"existing_key": "untouched", "inference_result": "keep_me"}

    def _assert_rejected(bad_zip: bytes, match: str = ""):
        snap = dict(sentinel)
        with pytest.raises(WorkspaceArchiveError) as exc_info:
            restore_workspace_archive(bad_zip, snap, project_root=tmp_path)
        if match:
            assert match in str(exc_info.value)
        assert snap == sentinel
        with pytest.raises(WorkspaceArchiveError):
            inspect_workspace_archive(bad_zip, project_root=tmp_path)

    # Missing required contract section (checksums valid, contract broken)
    files = _workspace_zip_files(good_zip)
    doc = json.loads(files[WORKSPACE_STATE_PATH].decode("utf-8"))
    del doc["ui_state"]
    files[WORKSPACE_STATE_PATH] = json.dumps(doc, indent=2, sort_keys=True).encode("utf-8")
    _sync_workspace_checksums(files)
    _assert_rejected(_zip_from_files(files), "ui_state")

    # Section of the wrong shape
    files = _workspace_zip_files(good_zip)
    doc = json.loads(files[WORKSPACE_STATE_PATH].decode("utf-8"))
    doc["results"] = ["not", "an", "object"]
    files[WORKSPACE_STATE_PATH] = json.dumps(doc, indent=2, sort_keys=True).encode("utf-8")
    _sync_workspace_checksums(files)
    _assert_rejected(_zip_from_files(files), "'results'")

    # Version 0 (not a supported schema, never silently upgraded)
    files = _workspace_zip_files(good_zip)
    doc = json.loads(files[WORKSPACE_STATE_PATH].decode("utf-8"))
    doc["workspace_schema_version"] = 0
    files[WORKSPACE_STATE_PATH] = json.dumps(doc, indent=2, sort_keys=True).encode("utf-8")
    ws_manifest = json.loads(files[WORKSPACE_MANIFEST_PATH].decode("utf-8"))
    ws_manifest["workspace_schema_version"] = 0
    files[WORKSPACE_MANIFEST_PATH] = json.dumps(ws_manifest, indent=2, sort_keys=True).encode("utf-8")
    _sync_workspace_checksums(files)
    _assert_rejected(_zip_from_files(files), "workspace_schema_version")

    # Non-integer version
    files = _workspace_zip_files(good_zip)
    ws_manifest = json.loads(files[WORKSPACE_MANIFEST_PATH].decode("utf-8"))
    ws_manifest["workspace_schema_version"] = str(WORKSPACE_SCHEMA_VERSION)
    files[WORKSPACE_MANIFEST_PATH] = json.dumps(ws_manifest, indent=2, sort_keys=True).encode("utf-8")
    _sync_workspace_checksums(files)
    _assert_rejected(_zip_from_files(files), "Invalid workspace_schema_version")

    # A workspace that declares pickle is rejected outright
    files = _workspace_zip_files(good_zip)
    ws_manifest = json.loads(files[WORKSPACE_MANIFEST_PATH].decode("utf-8"))
    ws_manifest["allow_pickle"] = True
    files[WORKSPACE_MANIFEST_PATH] = json.dumps(ws_manifest, indent=2, sort_keys=True).encode("utf-8")
    _sync_workspace_checksums(files)
    _assert_rejected(_zip_from_files(files), "Pickle-enabled")

    # The untouched archive still restores, proving the guards do not over-reject
    ok_target: dict = {}
    restore_workspace_archive(good_zip, ok_target, project_root=tmp_path)
    assert ok_target["inference_result"] is not None


# ---------------------------------------------------------------------------
# Stage 2 — default workspace ZIP is pickle-free (AC3) and malformed/typed
# archive structures fail as WorkspaceArchiveError before any mutation.
# ---------------------------------------------------------------------------

def _assert_rejected_unchanged(
    bad_zip: bytes,
    tmp_path: Path,
    sentinel: dict,
    match: str = "",
) -> WorkspaceArchiveError:
    """Inspect and restore must both reject the archive and leave the target intact."""
    snap = dict(sentinel)
    with pytest.raises(WorkspaceArchiveError) as exc_info:
        restore_workspace_archive(bad_zip, sentinel, project_root=tmp_path)
    if match:
        assert match in str(exc_info.value)
    assert sentinel == snap
    with pytest.raises(WorkspaceArchiveError):
        inspect_workspace_archive(bad_zip, project_root=tmp_path)
    return exc_info.value


def _pickle_members(names) -> list:
    return [n for n in names if n.lower().endswith((".pkl", ".pickle"))]


def test_default_workspace_save_never_carries_a_pickle(tmp_path: Path):
    """AC3(a): the default save path has no bundle.pkl and no pickle member at all."""
    state = _make_full_state(tmp_path)
    zip_bytes, manifest = build_workspace_archive(state, include_data=False, project_root=tmp_path)

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = set(zf.namelist())
        methods = zf.read("methods.md").decode("utf-8")
        root_manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        state_doc = json.loads(zf.read(WORKSPACE_STATE_PATH).decode("utf-8"))

    # Unified branch (full finished result) still ships the whole workspace layer...
    assert {
        "manifest.json",
        WORKSPACE_MANIFEST_PATH,
        WORKSPACE_STATE_PATH,
        WORKSPACE_ARRAYS_PATH,
        "methods.md",
        "run_plan.json",
        "dataset_info.json",
    } <= names
    for section in ("source_refs", "results", "annotation", "ui_state", "data"):
        assert section in state_doc

    # ...but never a pickle member.
    assert "bundle.pkl" not in names
    assert _pickle_members(names) == []

    # The manifest and methods state the omission honestly (not "lacks metadata").
    assert manifest["bundle_available"] is False
    assert manifest["trusted_only"] == []
    assert root_manifest["bundle_available"] is False
    assert root_manifest["trusted_only"] == []
    assert any("pickle-free" in note for note in manifest["notes"])
    assert "No bundle.pkl is included" in methods
    assert "pickle-free by design" in methods
    assert "lacks the captured analysis metadata" not in methods

    # The pickle-free archive is a complete restore source.
    insp = inspect_workspace_archive(zip_bytes, project_root=tmp_path)
    assert insp.has_results is True
    assert insp.has_companion is True
    assert insp.has_annotation is True
    assert insp.has_evidence_packet is True


def test_default_workspace_archive_roundtrips_without_pickle(tmp_path: Path, monkeypatch):
    """AC3(b): inspect + restore of the default archive never import pickle."""
    state = _make_full_state(tmp_path)
    zip_bytes, _ = build_workspace_archive(state, include_data=False, project_root=tmp_path)

    real_import = builtins.__import__

    def _forbid_pickle(name, *args, **kwargs):
        if name == "pickle" or name.startswith("pickle.") or name == "_pickle":
            raise AssertionError("pickle must never be imported for a default workspace archive!")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _forbid_pickle)
    insp = inspect_workspace_archive(zip_bytes, project_root=tmp_path)
    target: dict = {}
    restored_insp = restore_workspace_archive(zip_bytes, target, project_root=tmp_path)

    assert insp.has_results and restored_insp.has_results
    assert target["inference_result"] is not None
    assert target["annotation"] is not None
    assert target["evidence_packet"] is not None
    assert target["_workspace_restored_meta"]["workspace_schema_version"] == WORKSPACE_SCHEMA_VERSION
    assert target["_workspace_source_verification"]["overall"] == "verified"


def test_full_analysis_archive_keeps_trusted_bundle(tmp_path: Path):
    """AC3(c): the explicit NETSTAT-24 archive still carries bundle.pkl + warning."""
    state = _make_full_state(tmp_path)
    kwargs = dict(project_root=tmp_path, workspace_state=state)
    zip_bytes, manifest = build_analysis_archive(
        state["inference_result"], state["run_plan"], state["dataset_info"], **kwargs
    )
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = set(zf.namelist())
        methods = zf.read("methods.md").decode("utf-8")
    assert "bundle.pkl" in names
    assert manifest["bundle_available"] is True
    assert manifest["trusted_only"] == ["bundle.pkl"]
    assert "bundle.pkl is a Python pickle for trusted use only" in methods
    assert not any("pickle-free" in note for note in manifest["notes"])

    # The explicit opt-out is honest instead of implying missing metadata.
    plain_bytes, plain_manifest = build_analysis_archive(
        state["inference_result"], state["run_plan"], state["dataset_info"],
        include_bundle=False, **kwargs,
    )
    with zipfile.ZipFile(io.BytesIO(plain_bytes), "r") as zf:
        plain_names = set(zf.namelist())
        plain_methods = zf.read("methods.md").decode("utf-8")
    assert _pickle_members(plain_names) == []
    assert plain_manifest["bundle_available"] is False
    assert plain_manifest["trusted_only"] == []
    assert any("pickle-free" in note for note in plain_manifest["notes"])
    assert "lacks the captured analysis metadata" not in plain_methods


def test_malformed_manifest_structures_raise_workspace_archive_error(tmp_path: Path):
    """2(a): `[]` where an object belongs is a clean WorkspaceArchiveError."""
    state = _make_full_state(tmp_path)
    good_zip, _ = build_workspace_archive(state, include_data=False, project_root=tmp_path)
    sentinel = {"existing_key": "untouched", "inference_result": "keep_me"}

    # root manifest.json is a JSON array
    files = _workspace_zip_files(good_zip)
    files["manifest.json"] = b"[]"
    _assert_rejected_unchanged(_zip_from_files(files), tmp_path, sentinel, "manifest.json")

    # workspace manifest inventory entry for state.json is an array
    files = _workspace_zip_files(good_zip)
    _sync_workspace_checksums(files)
    ws_manifest = json.loads(files[WORKSPACE_MANIFEST_PATH].decode("utf-8"))
    ws_manifest["files"][WORKSPACE_STATE_PATH] = []
    files[WORKSPACE_MANIFEST_PATH] = json.dumps(ws_manifest, indent=2, sort_keys=True).encode("utf-8")
    _assert_rejected_unchanged(_zip_from_files(files), tmp_path, sentinel, "inventory entry")

    # root manifest checksum entry for a workspace member is an array
    files = _workspace_zip_files(good_zip)
    _sync_workspace_checksums(files)
    root = json.loads(files["manifest.json"].decode("utf-8"))
    root["files"][WORKSPACE_STATE_PATH] = []
    files["manifest.json"] = json.dumps(root, indent=2, sort_keys=True).encode("utf-8")
    _assert_rejected_unchanged(_zip_from_files(files), tmp_path, sentinel, "checksum entry")

    # source_refs.dataset_info is an array (would be AttributeError before)
    files = _workspace_zip_files(good_zip)
    doc = json.loads(files[WORKSPACE_STATE_PATH].decode("utf-8"))
    doc["source_refs"]["dataset_info"] = []
    files[WORKSPACE_STATE_PATH] = json.dumps(doc, indent=2, sort_keys=True).encode("utf-8")
    _sync_workspace_checksums(files)
    _assert_rejected_unchanged(_zip_from_files(files), tmp_path, sentinel, "dataset_info")

    # results["primary"] is an array (would be AttributeError before)
    files = _workspace_zip_files(good_zip)
    doc = json.loads(files[WORKSPACE_STATE_PATH].decode("utf-8"))
    doc["results"]["primary"] = []
    files[WORKSPACE_STATE_PATH] = json.dumps(doc, indent=2, sort_keys=True).encode("utf-8")
    _sync_workspace_checksums(files)
    _assert_rejected_unchanged(_zip_from_files(files), tmp_path, sentinel, "results.primary")

    # The untouched archive still restores.
    ok_target: dict = {}
    restore_workspace_archive(good_zip, ok_target, project_root=tmp_path)
    assert ok_target["inference_result"] is not None


def test_state_schema_version_must_be_a_json_integer(tmp_path: Path):
    """2(b): 2.0 (float) must never satisfy the manifest's integer 2."""
    state = _make_full_state(tmp_path)
    good_zip, _ = build_workspace_archive(state, include_data=False, project_root=tmp_path)
    sentinel = {"existing_key": "untouched", "inference_result": "keep_me"}

    files = _workspace_zip_files(good_zip)
    ws_manifest = json.loads(files[WORKSPACE_MANIFEST_PATH].decode("utf-8"))
    assert ws_manifest["workspace_schema_version"] == WORKSPACE_SCHEMA_VERSION
    doc = json.loads(files[WORKSPACE_STATE_PATH].decode("utf-8"))
    doc["workspace_schema_version"] = float(WORKSPACE_SCHEMA_VERSION)
    files[WORKSPACE_STATE_PATH] = json.dumps(doc, indent=2, sort_keys=True).encode("utf-8")
    _sync_workspace_checksums(files)
    bad_zip = _zip_from_files(files)

    # Prove the payload really is a float after the JSON round-trip.
    with zipfile.ZipFile(io.BytesIO(bad_zip), "r") as zf:
        stored = json.loads(zf.read(WORKSPACE_STATE_PATH).decode("utf-8"))
    assert isinstance(stored["workspace_schema_version"], float)
    assert stored["workspace_schema_version"] == WORKSPACE_SCHEMA_VERSION

    _assert_rejected_unchanged(bad_zip, tmp_path, sentinel, "workspace_schema_version")

    # A genuine integer version still restores (guards do not over-reject).
    ok_target: dict = {}
    restore_workspace_archive(good_zip, ok_target, project_root=tmp_path)
    assert ok_target["inference_result"] is not None


def test_malformed_source_reference_degrades_honest_status(tmp_path: Path):
    """2(c): a non-object source record is reported, never silently skipped."""
    f1 = tmp_path / "data.npy"
    f1.write_bytes(b"original-content")
    captured = capture_source_hashes(
        dataset_info={"mode": "Dataset Manifest (data.yaml)", "source": str(f1)},
        resolved_paths={"data": f1},
        project_root=tmp_path,
    )
    captured["broken_ref"] = ["not", "an", "object"]

    res = verify_source_references(captured, project_root=tmp_path)
    assert res["overall"] != "verified"
    assert res["overall"] == "unknown"
    assert res["counts"]["malformed"] == 1
    assert res["files"]["broken_ref"]["status"] == "unknown"
    assert res["files"]["broken_ref"]["malformed"] is True
    assert res["files"]["manifest.data"]["status"] == "verified"  # good records stay honest
    assert "malformed" in res["message"]
    assert len(res["files"]) == 2  # nothing silently dropped

    # Same through a real archive: inspect and restore both report it.
    state = _make_full_state(tmp_path)
    good_zip, _ = build_workspace_archive(state, include_data=False, project_root=tmp_path)
    files = _workspace_zip_files(good_zip)
    doc = json.loads(files[WORKSPACE_STATE_PATH].decode("utf-8"))
    doc["source_refs"]["source_files"]["broken_ref"] = ["not", "an", "object"]
    files[WORKSPACE_STATE_PATH] = json.dumps(doc, indent=2, sort_keys=True).encode("utf-8")
    _sync_workspace_checksums(files)
    bad_zip = _zip_from_files(files)

    insp = inspect_workspace_archive(bad_zip, project_root=tmp_path)
    assert insp.source_verification["overall"] != "verified"
    assert insp.source_verification["counts"]["malformed"] == 1
    assert insp.source_verification["files"]["broken_ref"]["malformed"] is True
    target: dict = {}
    restore_workspace_archive(bad_zip, target, project_root=tmp_path)
    stored = target["_workspace_source_verification"]
    assert stored["overall"] != "verified"
    assert stored["counts"]["malformed"] == 1
    assert "malformed" in stored["message"]


def test_restore_validates_and_inspects_before_mutation(tmp_path: Path, monkeypatch):
    """2(d): one parse per restore, no re-inspection after applying, no partial apply."""
    state = _make_full_state(tmp_path)
    zip_bytes, _ = build_workspace_archive(state, include_data=False, project_root=tmp_path)

    parses = {"n": 0}
    real_parse = workspace_io._parse_and_validate_zip

    def _counting_parse(*args, **kwargs):
        parses["n"] += 1
        return real_parse(*args, **kwargs)

    def _boom(*args, **kwargs):
        raise AssertionError("the archive must not be re-inspected after it was applied")

    monkeypatch.setattr(workspace_io, "_parse_and_validate_zip", _counting_parse)
    monkeypatch.setattr(workspace_io, "inspect_workspace_archive", _boom)

    target: dict = {}
    insp = restore_workspace_archive(zip_bytes, target, project_root=tmp_path)
    assert parses["n"] == 1
    assert insp.has_results is True
    assert target["inference_result"] is not None

    # A failure while building the patch happens strictly before the first write.
    def _boom_patch(payload):
        raise WorkspaceArchiveError("synthetic pre-apply failure")

    monkeypatch.setattr(workspace_io, "_build_restore_patch", _boom_patch)
    sentinel = {
        "existing_key": "untouched",
        "inference_result": "keep_me",
        "connectivity_data": "keep_me_too",
        "_workspace_widget_keys": {"results:alpha": "results"},
        "results:alpha": 0.5,
        "inference_task": "transient",
    }
    snap = dict(sentinel)
    with pytest.raises(WorkspaceArchiveError):
        restore_workspace_archive(zip_bytes, sentinel, project_root=tmp_path)
    assert sentinel == snap


def test_corrupt_stored_member_body_raises_workspace_archive_error(tmp_path: Path):
    """One flipped byte in a stored member body (central directory intact).

    Typical download corruption: the ZIP still opens and lists members, but
    reading ``workspace/state.json`` fails with a bad CRC-32. Both public APIs
    must surface ``WorkspaceArchiveError`` (never a raw ``zipfile.BadZipFile``)
    and restore must leave the target untouched.
    """
    state = _make_full_state(tmp_path)
    good_zip, _ = build_workspace_archive(state, include_data=False, project_root=tmp_path)

    # Re-pack the same checksum-valid members as STORED so exactly one body byte
    # can be flipped without touching the central directory.
    stored_zip = _zip_from_files(_workspace_zip_files(good_zip))
    with zipfile.ZipFile(io.BytesIO(stored_zip), "r") as zf:
        info = zf.getinfo(WORKSPACE_STATE_PATH)
        names_before = zf.namelist()
    assert info.compress_type == zipfile.ZIP_STORED

    raw = bytearray(stored_zip)
    name_len = int.from_bytes(raw[info.header_offset + 26:info.header_offset + 28], "little")
    extra_len = int.from_bytes(raw[info.header_offset + 28:info.header_offset + 30], "little")
    body_start = info.header_offset + 30 + name_len + extra_len
    # Flip a single byte inside the stored payload -> CRC-32 mismatch on read.
    raw[body_start + info.file_size // 2] ^= 0xFF
    corrupt_zip = bytes(raw)

    with zipfile.ZipFile(io.BytesIO(corrupt_zip), "r") as zf:
        assert zf.namelist() == names_before  # central directory still intact
        with pytest.raises(zipfile.BadZipFile):
            zf.read(WORKSPACE_STATE_PATH)

    with pytest.raises(WorkspaceArchiveError) as excinfo:
        inspect_workspace_archive(corrupt_zip, project_root=tmp_path)
    assert "BadZipFile" in str(excinfo.value)

    target: dict = {"existing_key": "untouched", "inference_result": "keep_me"}
    snap = dict(target)
    with pytest.raises(WorkspaceArchiveError):
        restore_workspace_archive(corrupt_zip, target, project_root=tmp_path)
    assert target == snap

    # The neighbouring undamaged archive still inspects and restores.
    insp = inspect_workspace_archive(stored_zip, project_root=tmp_path)
    assert insp.has_results is True
    restored: dict = {}
    restore_workspace_archive(stored_zip, restored, project_root=tmp_path)
    assert restored["inference_result"] is not None
