"""NETSTAT-24: the application analysis archive and its CLI replay contract.

Covers the ZIP allowlist and schema, determinism, post-analysis from the
extracted bundle without observations, the honest replay statuses, secret
redaction, the Fisher-policy replay adapter (an in-app clip-r run must not
replay as the library's clip-z default), grid exports, unknown legacy facts
and a real Streamlit AppTest of the export button.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
import yaml

from conninfpy import ResultBundle, analyze, fisher_r_to_z
from conninfpy._prepare import make_fisher_entry
from conninfpy.atlas import AtlasInfo

from apps.utils.archive_export import (
    ARCHIVE_SCHEMA_VERSION,
    build_analysis_archive,
    build_replay_config,
)

REPO = Path(__file__).resolve().parents[1]
N = 6
ATLAS = AtlasInfo(labels=[f"roi{i}" for i in range(N)],
                  networks=["Vis", "Vis", "DMN", "DMN", "SMN", "SMN"],
                  hemisphere=["L", "R", "L", "R", "L", "R"])


def _batch(rng, n, shift=0.0):
    a = rng.normal(scale=0.2, size=(n, N, N))
    a = 0.5 * (a + a.transpose(0, 2, 1)) + shift
    idx = np.arange(N)
    a[:, idx, idx] = 1.0
    return a


def _two_groups(seed=3, extreme=False):
    rng = np.random.default_rng(seed)
    g1 = _batch(rng, 10)
    g2 = _batch(rng, 10, shift=0.05)
    if extreme:
        # An edge straddling both clip thresholds (tanh(5) ≈ 0.999909 and
        # the in-app clip-r bound 0.9999) with subject jitter: the clip-r
        # and clip-z policies give different Fisher-z values, so a replay
        # with the wrong policy cannot reproduce the statistics.
        jitter = rng.uniform(-3e-5, 3e-5, 10)
        g1[:, 0, 2] = g1[:, 2, 0] = np.clip(0.99980 + jitter, -1, 1)
        g2[:, 0, 2] = g2[:, 2, 0] = np.clip(0.99995 + jitter, -1, 1)
    g2[:, 1, 4] += 0.30; g2[:, 4, 1] += 0.30
    g2[:, 1, 5] += 0.30; g2[:, 5, 1] += 0.30
    g2 = np.clip(g2, -0.9999995, 0.9999995)   # keep valid correlations
    return g1, g2


def _run_plan(**overrides):
    plan = {
        "question_type": "Group Difference", "alpha": 0.05,
        "design_family": "two-sample", "method": "tstat", "n_permutations": 20,
        "site_recipe": "No site handling", "effective_site_recipe": "No site handling",
        "harmonize": None, "group_col": "group", "reference_group": 0,
        "target_group": 1, "input": "r", "data_kind": "correlation",
        "analysis_scale": "correlation", "two_tailed": True, "t_statistic": "student",
        "seed": 7, "started_at": "2026-11-13T10:00:00+00:00", "wall_time_s": 0.5,
        "use_mp": False, "acceleration": None, "var_equal": True,
        "net_labels": None, "operator_kwargs": {}, "roi_indices": None,
    }
    plan.update(overrides)
    return plan


def _dataset_info(**overrides):
    info = {
        "name": "tiny-study", "source": "Synthetic simulation",
        "mode": "Generate Synthetic Dataset", "observations": 20, "n_rois": N,
        "data_kind": "correlation", "atlas": "fixture",
        "generation_parameters": {
            "simulation_paradigm": "Two-Group Contrast", "scenario": "hub",
            "n_rois": N, "generator_seed": 11, "signal_effect_size": 0.25,
            "note": "generator inputs captured at generation time; the inference RNG seed is separate",
        },
    }
    info.update(overrides)
    return info


def _archive(result, plan=None, info=None, atlas=ATLAS, effect_mask=None, **kwargs):
    return build_analysis_archive(
        result, plan or _run_plan(), info or _dataset_info(),
        atlas=atlas, effect_mask=effect_mask, **kwargs)


def _open(data):
    zf = zipfile.ZipFile(io.BytesIO(data))
    assert zf.testzip() is None
    return zf


def _manifest_of(zf):
    return json.loads(zf.read("manifest.json"))


# ---------------------------------------------------------------------------
# ZIP content, schema, allowlist, determinism
# ---------------------------------------------------------------------------

class TestArchiveContent:
    def test_allowlist_schema_and_bundle_roundtrip(self):
        g1, g2 = _two_groups()
        res = analyze(group1=g1, group2=g2, test_type="two-sample", input="r",
                      method="tstat", n_permutations=20, use_mp=False, rng=7,
                      atlas=ATLAS)
        data, manifest = _archive(res)
        zf = _open(data)
        names = set(zf.namelist())
        expected = {
            "manifest.json", "run_plan.json", "dataset_info.json", "methods.md",
            "bundle.pkl", "arrays/index.json", "arrays/p_positive.npy",
            "arrays/p_negative.npy", "arrays/stat_signed.npy", "arrays/beta.npy",
            "arrays/effect.npy", "tables/edges.csv", "tables/network_summary.csv",
            "tables/node_summary.csv", "figures/connectivity_maps.png",
            "figures/network_summary.png",
        }
        assert expected <= names
        # Allowlist: nothing outside the result/run/source artifact set.
        assert names <= expected | {"analysis.yaml", "tables/nbs_components.csv",
                                    "tables/nbs_components.attrs.json",
                                    "tables/cnbs_blocks.csv", "figures/null_max.png",
                                    "figures/cnbs_blocks.png",
                                    "arrays/synthetic_effect_mask.npy",
                                    "arrays/e_grid.npy", "arrays/h_grid.npy",
                                    "arrays/p_omnibus.npy", "arrays/stat_omnibus.npy",
                                    "arrays/null_max_positive.npy",
                                    "arrays/null_max_negative.npy"}
        m = _manifest_of(zf)
        assert m["schema_version"] == ARCHIVE_SCHEMA_VERSION
        assert m["kind"] == "conninfpy-analysis-archive"
        assert m["trusted_only"] == ["bundle.pkl"]
        assert m["run"]["alpha"] == 0.05 and m["run"]["seed"] == 7
        assert m["dataset"]["synthetic"] is True
        assert set(m["files"]) == names - {"manifest.json"}
        for entry in m["files"].values():
            assert entry["bytes"] > 0 and len(entry["sha256"]) == 64

        # Post-analysis from the extracted bundle, without observations.
        extract = zf.read("bundle.pkl")
        Path_ = Path(__file__).parent  # noqa: F841  (context only)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bundle.pkl"
            p.write_bytes(extract)
            bundle = ResultBundle.load(p)
            assert bundle.roi_ids == tuple(ATLAS.labels)
            summary = bundle.summarize(grouping="network", measure="contrast_beta",
                                       contrast_id=bundle.metadata["contrast_id"],
                                       tail="positive")
            assert not summary.empty
            pairs = bundle.detected_pairs(tail="positive")
            assert set(pairs.columns) >= {"roi_i", "roi_j", "p_corrected", "effect"}

    def test_deterministic_members_for_same_facts(self):
        g1, g2 = _two_groups()
        res = analyze(group1=g1, group2=g2, test_type="two-sample", input="r",
                      method="tstat", n_permutations=20, use_mp=False, rng=7,
                      atlas=ATLAS)
        first, _ = _archive(res)
        second, _ = _archive(res)
        # The export timestamp is a fact of the export moment; every captured
        # artifact (methods, YAML, tables, arrays, bundle) must be identical.
        za, zb = _open(first), _open(second)
        assert sorted(za.namelist()) == sorted(zb.namelist())
        for name in za.namelist():
            if name == "manifest.json":
                ma = json.loads(za.read(name)); mb = json.loads(zb.read(name))
                ma["export"].pop("exported_utc"); mb["export"].pop("exported_utc")
                assert ma == mb
            else:
                assert za.read(name) == zb.read(name), name

    def test_methods_md_is_deterministic_and_reports_unknowns(self):
        g1, g2 = _two_groups()
        res = analyze(group1=g1, group2=g2, test_type="two-sample", input="r",
                      method="tstat", n_permutations=20, use_mp=False, rng=7,
                      atlas=ATLAS)
        # Legacy plan: missing capture stays unknown, never re-chosen.
        legacy = _run_plan(alpha=None, started_at=None, seed=None, use_mp=None,
                           acceleration=None, var_equal=None)
        data, _ = _archive(res, plan=legacy)
        zf = _open(data)
        methods = zf.read("methods.md").decode("utf-8")
        assert "unknown / not recorded" in methods
        assert "α-dependent tables and figures are omitted" in json.loads(
            zf.read("manifest.json"))["notes"][0] or any(
            "α-dependent" in n for n in _manifest_of(zf)["notes"])
        assert "tables" not in {n.split("/")[0] for n in zf.namelist() if "/" in n} or all(
            not n.startswith("tables/") and not n.startswith("figures/")
            for n in zf.namelist())
        first = zf.read("methods.md")
        again = _open(_archive(res, plan=legacy)[0]).read("methods.md")
        assert first == again

    def test_secret_redaction(self):
        g1, g2 = _two_groups()
        res = analyze(group1=g1, group2=g2, test_type="two-sample", input="r",
                      method="tstat", n_permutations=20, use_mp=False, rng=7,
                      atlas=ATLAS)
        plan = _run_plan(api_token="sk-abcdefghijklmnop12345", password="hunter2",
                         operator_kwargs={})
        data, _ = _archive(res, plan=plan)
        raw = data  # the whole archive must not carry the raw secrets
        assert b"sk-abcdefghijklmnop12345" not in raw
        assert b"hunter2" not in raw
        zf = _open(data)
        saved = json.loads(zf.read("run_plan.json"))
        assert "api_token" not in saved and "password" not in saved
        assert any("redacted" in n for n in _manifest_of(zf)["notes"])

    def test_empty_detections_still_export_table(self):
        g1, g2 = _two_groups(seed=5)
        res = analyze(group1=g1, group2=g2, test_type="two-sample", input="r",
                      method="tstat", n_permutations=20, use_mp=False, rng=7,
                      atlas=ATLAS)
        data, _ = _archive(res, plan=_run_plan(alpha=1e-9))
        zf = _open(data)
        assert "tables/edges.csv" in zf.namelist()
        rows = zf.read("tables/edges.csv").decode().strip().splitlines()
        assert len(rows) == 1  # header only


# ---------------------------------------------------------------------------
# Replay statuses
# ---------------------------------------------------------------------------

    def test_legacy_result_without_provenance_omits_bundle_honestly(self):
        from conninfpy.results import InferenceResult

        stat = np.array([[0, 4, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=float)
        p = np.ones((N, N)); p[0, 1] = p[1, 0] = 0.01
        legacy = InferenceResult(p, np.ones((N, N)), method="tstat",
                                 stat_positive=stat, stat_negative=np.zeros((N, N)))
        data, manifest = _archive(legacy)
        zf = _open(data)
        assert "bundle.pkl" not in zf.namelist()
        assert manifest["trusted_only"] == []
        assert any("bundle.pkl omitted" in n for n in manifest["notes"])
        methods = zf.read("methods.md").decode()
        assert "No bundle.pkl is included" in methods
        # Evidence arrays remain a faithful record of the finished run.
        saved = np.load(io.BytesIO(zf.read("arrays/stat_signed.npy")))
        assert saved[0, 1] == 4.0


class TestReplayStatus:
    def test_synthetic_is_not_replayable_and_keeps_generator_seed(self):
        g1, g2 = _two_groups()
        res = analyze(group1=g1, group2=g2, test_type="two-sample", input="r",
                      method="tstat", n_permutations=20, use_mp=False, rng=7,
                      atlas=ATLAS)
        data, _ = _archive(res)
        m = _manifest_of(_open(data))
        assert m["replay"]["status"] == "not-replayable"
        assert "analysis.yaml" not in _open(data).namelist()
        saved = json.loads(_open(data).read("dataset_info.json"))
        gen = saved["generation_parameters"]
        assert gen["generator_seed"] == 11            # generator seed ...
        assert m["run"]["seed"] == 7                  # ... separate from inference RNG
        methods = _open(data).read("methods.md").decode()
        assert "generator_seed: 11" in methods

    def test_missing_manifest_is_requires_input(self, tmp_path):
        g1, g2 = _two_groups()
        res = analyze(group1=g1, group2=g2, test_type="two-sample", input="r",
                      method="tstat", n_permutations=20, use_mp=False, rng=7,
                      atlas=ATLAS)
        info = _dataset_info(
            mode="Import from manifest", source=str(tmp_path / "gone.yaml"),
            manifest_file=str(tmp_path / "gone.yaml"))
        data, _ = _archive(res, info=info, project_root=tmp_path)
        m = _manifest_of(_open(data))
        assert m["replay"]["status"] == "requires-input"
        assert any(p["kind"] == "missing-file" for p in m["replay"]["prerequisites"])
        source_state = m["dataset"]["source_files"]["manifest"]
        assert source_state["exists"] is False

    def test_legacy_cnbs_without_label_capture_is_not_replayable(self):
        rng = np.random.default_rng(2)
        Y = _batch(rng, 14)
        age = rng.normal(30, 6, 14)
        res = analyze(Y=Y, interest=age, input="r", method="cnbs", n_permutations=10,
                      use_mp=False, rng=7, atlas=ATLAS, net_labels=ATLAS.network_index())
        assert res.inference.block_report is not None
        plan = _run_plan(design_family="glm", question_type="Continuous Predictor",
                         interest_var="age", method="cnbs", net_labels=None,
                         t_statistic="glm")
        replay = build_replay_config(plan, _dataset_info(mode="Import from manifest",
                                                          source="datasets/abide_mini.yaml"),
                                     project_root=REPO)
        assert replay["status"] == "not-replayable"
        assert any("label" in r for r in replay["reasons"])

    def test_sc_prior_labels_replay_from_captured_vector(self):
        labels = [0, 0, 1, 1, 2, 2]
        plan = _run_plan(method="cnbs", design_family="glm", question_type="Continuous Predictor",
                         interest_var="age", t_statistic="glm",
                         net_labels={"source": "sc_prior", "labels": labels})
        replay = build_replay_config(plan, _dataset_info(mode="Import from manifest",
                                                          source="datasets/abide_mini.yaml"),
                                     project_root=REPO)
        assert replay["status"] == "replayable"
        assert replay["config"]["inference"]["net_labels"] == labels

    def test_roi_subset_is_requires_input(self):
        plan = _run_plan(roi_indices=[0, 1, 2, 3])
        replay = build_replay_config(plan, _dataset_info(mode="Import from manifest",
                                                          source="datasets/abide_mini.yaml"),
                                     project_root=REPO)
        assert replay["status"] == "requires-input"
        assert any(p["kind"] == "roi-subset" for p in replay["prerequisites"])

    def test_paired_design_maps_baseline_target_order(self):
        plan = _run_plan(design_family="paired", question_type="Paired Condition",
                         subject_col="subject_id", condition_col="condition",
                         baseline_condition="close", target_condition="open")
        replay = build_replay_config(plan, _dataset_info(mode="Import from manifest",
                                                          source="datasets/open_close.yaml"),
                                     project_root=REPO)
        assert replay["status"] == "replayable"
        design = replay["config"]["design"]
        assert design["type"] == "paired"
        # The CLI pairs levels[0] as group1 (baseline) and levels[1] as
        # group2 (target): the difference stays target − baseline.
        assert design["levels"] == ["close", "open"]


# ---------------------------------------------------------------------------
# Real CLI replay (exactness of the Fisher policy adapter)
# ---------------------------------------------------------------------------

def _write_study(tmp_path, extreme=False):
    g1, g2 = _two_groups(seed=3, extreme=extreme)
    data = np.concatenate([g1, g2], axis=0)
    pheno = pd.DataFrame({
        "subject_id": [f"s{i:02d}" for i in range(20)],
        "group": ["hc"] * 10 + ["pt"] * 10,
        "age": np.random.default_rng(1).normal(30, 6, 20),
    })
    np.save(tmp_path / "conn.npy", data)
    pheno.to_csv(tmp_path / "pheno.csv", index=False)
    (tmp_path / "study.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "loader": "NumpyLoader",
        "paths": {"data_path": "conn.npy", "pheno_path": "pheno.csv"},
    }))
    return g1, g2, pheno


def _run_cli(config_path, out_name="out"):
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
                "MPLBACKEND": "Agg"})
    proc = subprocess.run(
        [sys.executable, "-m", "conninfpy", "run", str(config_path), "--out",
         str(config_path.parent / out_name)],
        capture_output=True, text=True, timeout=600, env=env, cwd=str(REPO))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc


def _extract_archive(tmp_path, data):
    target = tmp_path / "archive"
    target.mkdir(exist_ok=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(target)
    return target


class TestCliReplay:
    def test_clip_r_policy_replays_exactly_not_as_clip_z(self, tmp_path):
        g1, g2, _ = _write_study(tmp_path, extreme=True)
        # Simulate the application flow: raw correlations → the explicit
        # Fisher button (clip-r ±0.9999) → analyze(input='z').
        z1, rep1 = fisher_r_to_z(g1, clip_r=0.9999, return_report=True)
        z2, rep2 = fisher_r_to_z(g2, clip_r=0.9999, return_report=True)
        entry = make_fisher_entry(stage="app_design_button",
                                  scope="dataset_in_memory", report=rep1)
        res = analyze(group1=z1, group2=z2, test_type="two-sample", input="z",
                      method="tstat", n_permutations=20, use_mp=False, rng=7,
                      atlas=ATLAS, var_equal=True, fisher_history=[entry])
        plan = _run_plan(input="z", data_kind="fisher_z",
                         analysis_scale="fisher_z", fisher=res.fisher,
                         reference_group="hc", target_group="pt")
        info = _dataset_info(mode="Import from manifest",
                             source=str(tmp_path / "study.yaml"),
                             manifest_file=str(tmp_path / "study.yaml"),
                             data_kind="correlation")
        data, _ = _archive(res, plan=plan, info=info, project_root=tmp_path)
        zf = _open(data)
        assert _manifest_of(zf)["replay"]["status"] == "replayable"
        cfg = yaml.safe_load(zf.read("analysis.yaml"))
        assert cfg["connectivity"]["fisher"] == {"policy": "clip_r", "clip_r": 0.9999}
        assert cfg["inference"]["r_to_z"] is False if "r_to_z" in cfg["inference"] else True

        target = _extract_archive(tmp_path, data)
        _run_cli(target / "analysis.yaml", out_name="replay")
        with np.load(tmp_path / "archive" / "replay" / "pvals.npz") as zr:
            assert np.array_equal(zr["positive"], np.load(io.BytesIO(zf.read("arrays/p_positive.npy"))))
            assert np.array_equal(zr["negative"], np.load(io.BytesIO(zf.read("arrays/p_negative.npy"))))
            assert np.array_equal(zr["stat_signed"], np.load(io.BytesIO(zf.read("arrays/stat_signed.npy"))))

        # The clip-z default is a different policy and must NOT reproduce it.
        wrong = yaml.safe_load(zf.read("analysis.yaml"))
        wrong["connectivity"] = {}   # no adapter: raw r, r_to_z default applies clip-z
        wrong["inference"].pop("r_to_z", None)
        wrong_path = tmp_path / "wrong.yaml"
        wrong_path.write_text(yaml.safe_dump(wrong))
        _run_cli(wrong_path, out_name="wrong")
        with np.load(tmp_path / "wrong" / "pvals.npz") as zr:
            assert not np.array_equal(zr["stat_signed"], np.load(io.BytesIO(zf.read("arrays/stat_signed.npy"))))

    def test_raw_correlation_run_replays_exactly(self, tmp_path):
        g1, g2, _ = _write_study(tmp_path)
        res = analyze(group1=g1, group2=g2, test_type="two-sample", input="r",
                      method="tstat", n_permutations=20, use_mp=False, rng=7,
                      atlas=ATLAS, r_to_z=False, var_equal=True)
        assert res.inference.design["scale"] == "correlation"
        info = _dataset_info(mode="Import from manifest",
                             source=str(tmp_path / "study.yaml"),
                             manifest_file=str(tmp_path / "study.yaml"))
        data, _ = _archive(res, info=info, project_root=tmp_path,
                           plan=_run_plan(reference_group="hc", target_group="pt"))
        zf = _open(data)
        cfg = yaml.safe_load(zf.read("analysis.yaml"))
        assert cfg["inference"]["r_to_z"] is False
        target = _extract_archive(tmp_path, data)
        _run_cli(target / "analysis.yaml", out_name="replay_r")
        with np.load(tmp_path / "archive" / "replay_r" / "pvals.npz") as zr:
            assert np.array_equal(zr["positive"], np.load(io.BytesIO(zf.read("arrays/p_positive.npy"))))

    def test_glm_group_reference_coding_replays_exactly(self, tmp_path):
        g1, g2, _ = _write_study(tmp_path)
        data = np.concatenate([g1, g2], axis=0)
        pheno = pd.DataFrame({
            "subject_id": [f"s{i:02d}" for i in range(20)],
            "group": ["hc"] * 10 + ["pt"] * 10,
            "age": np.random.default_rng(1).normal(30, 6, 20),
        })
        # The application codes the group indicator itself (target vs reference)
        # and adjusts for age; the CLI must reproduce that coding exactly.
        interest = (pheno["group"] == "pt").to_numpy(float)
        res = analyze(Y=data, interest=interest, confounds=pheno[["age"]].to_numpy(float),
                      input="r", method="tstat", n_permutations=20, use_mp=False,
                      rng=7, atlas=ATLAS, contrast_id="group", r_to_z=False)
        plan = _run_plan(design_family="glm", question_type="Group Difference",
                         group_col="group", reference_group="hc", target_group="pt",
                         confound_vars=["age"], t_statistic="glm")
        info = _dataset_info(mode="Import from manifest",
                             source=str(tmp_path / "study.yaml"),
                             manifest_file=str(tmp_path / "study.yaml"))
        archive, _ = _archive(res, plan=plan, info=info, project_root=tmp_path)
        zf = _open(archive)
        cfg = yaml.safe_load(zf.read("analysis.yaml"))
        assert cfg["design"]["interest"] == "group"
        assert cfg["design"]["interest_reference"] == "hc"
        target = _extract_archive(tmp_path, archive)
        _run_cli(target / "analysis.yaml", out_name="replay_glm")
        with np.load(tmp_path / "archive" / "replay_glm" / "pvals.npz") as zr:
            assert np.array_equal(zr["positive"], np.load(io.BytesIO(zf.read("arrays/p_positive.npy"))))
            # The phenotype round-trips through CSV on replay: numeric
            # confounds carry float noise ≈1e-15, never a policy difference.
            assert np.allclose(zr["stat_signed"],
                               np.load(io.BytesIO(zf.read("arrays/stat_signed.npy"))),
                               rtol=0, atol=1e-9)

    def test_cli_rejects_contradicted_fisher_config(self, tmp_path):
        g1, g2, _ = _write_study(tmp_path)
        cfg = {
            "data": str(tmp_path / "study.yaml"),
            "design": {"type": "glm", "interest": "age"},
            "inference": {"method": "tstat", "n_permutations": 5, "use_mp": False, "rng": 0},
            "connectivity": {"fisher": {"policy": "clip_r", "clip_r": 0.9999}},
            "input": "r",
            "output": "bad",
        }
        path = tmp_path / "bad_fisher.yaml"
        path.write_text(yaml.safe_dump(cfg))
        env = dict(os.environ, OMP_NUM_THREADS="1")
        proc = subprocess.run([sys.executable, "-m", "conninfpy", "run", str(path)],
                              capture_output=True, text=True, timeout=600, env=env,
                              cwd=str(REPO))
        assert proc.returncode != 0
        assert "remove the explicit" in proc.stderr


# ---------------------------------------------------------------------------
# Native units, grids, masks
# ---------------------------------------------------------------------------

class TestNativeUnitsAndGrids:
    def test_nbs_archive_keeps_component_table_and_native_p(self):
        rng = np.random.default_rng(9)
        g1 = _batch(rng, 10)
        g2 = _batch(rng, 10)
        for u, v in ((1, 4), (1, 5), (4, 5)):
            g1[:, u, v] -= 0.05; g1[:, v, u] -= 0.05
            g2[:, u, v] += 0.30; g2[:, v, u] += 0.30
        g2 = np.clip(g2, -0.9999995, 0.9999995)
        res = analyze(group1=g1, group2=g2, test_type="two-sample", input="r",
                      method="nbs", n_permutations=40, use_mp=False, rng=7,
                      atlas=ATLAS, threshold=2.5, var_equal=True)
        assert res.inference.component_report is not None
        data, _ = _archive(res, plan=_run_plan(method="nbs", operator_kwargs={
            "threshold": 1.5, "nbs_stat": "extent"}))
        zf = _open(data)
        assert "tables/nbs_components.csv" in zf.namelist()
        comps = pd.read_csv(io.BytesIO(zf.read("tables/nbs_components.csv")))
        assert {"tail", "component_id", "p_fwer", "n_edges"} <= set(comps.columns)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "b.pkl"
            p.write_bytes(zf.read("bundle.pkl"))
            bundle = ResultBundle.load(p)
            assert bundle.metadata["inference_unit"] == "component"
            native = bundle.native_units(tail="positive")
            assert not native.empty
            assert set(native.columns) >= {"native_id", "p_corrected", "n_locations"}

    def test_cnbs_grid_and_block_export(self):
        rng = np.random.default_rng(4)
        Y = _batch(rng, 14)
        age = rng.normal(30, 6, 14)
        res = analyze(Y=Y, interest=age, input="r", method="cnbs", n_permutations=10,
                      use_mp=False, rng=7, atlas=ATLAS, net_labels=ATLAS.network_index())
        plan = _run_plan(design_family="glm", question_type="Continuous Predictor",
                         interest_var="age", method="cnbs", t_statistic="glm",
                         net_labels={"source": "atlas_networks",
                                     "labels": list(ATLAS.network_index())})
        data, _ = _archive(res, plan=plan)
        zf = _open(data)
        assert "tables/cnbs_blocks.csv" in zf.namelist()
        assert "figures/cnbs_blocks.png" in zf.namelist()
        assert _manifest_of(zf)["replay"]["status"] == "not-replayable"  # synthetic data

    def test_tfnbs_grid_exports_selected_cell_figures_and_full_grid_arrays(self):
        rng = np.random.default_rng(6)
        g1 = _batch(rng, 10)
        g2 = _batch(rng, 10, shift=0.05)
        res = analyze(group1=g1, group2=g2, test_type="two-sample", input="r",
                      method="tfnbs", e=[0.4, 0.5], h=[3.0, 3.0],
                      n_permutations=10, use_mp=False, rng=7, atlas=ATLAS)
        assert res.inference.is_grid
        plan = _run_plan(method="tfnbs", operator_kwargs={"e": [0.4, 0.5],
                                                          "h": [3.0, 3.0]})
        data, _ = _archive(res, plan=plan, grid_cell=1)
        zf = _open(data)
        p = np.load(io.BytesIO(zf.read("arrays/p_positive.npy")))
        assert p.ndim == 3, "arrays keep the finished run's full grid evidence"
        arrays_index = json.loads(zf.read("arrays/index.json"))
        assert arrays_index["p_positive"]["display_cell"] == 1
        assert "full parameter grid" in arrays_index["p_positive"]["content"]
        m = _manifest_of(zf)
        assert m["export"]["display"]["grid_cell"] == 1
        methods = zf.read("methods.md").decode()
        assert "keep the full grid" in methods and "cell 1" in methods
        # Full grid evidence stays available through the bundle.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pth = Path(tmp) / "b.pkl"
            pth.write_bytes(zf.read("bundle.pkl"))
            bundle = ResultBundle.load(pth)
            assert bundle.corrected_p["positive"].ndim == 3

    def test_effect_mask_saved_without_simulation(self):
        g1, g2 = _two_groups()
        res = analyze(group1=g1, group2=g2, test_type="two-sample", input="r",
                      method="tstat", n_permutations=20, use_mp=False, rng=7,
                      atlas=ATLAS)
        mask = np.zeros((N, N)); mask[1, 4] = mask[4, 1] = 1.0
        data, _ = _archive(res, effect_mask=mask)
        zf = _open(data)
        saved = np.load(io.BytesIO(zf.read("arrays/synthetic_effect_mask.npy")))
        assert np.array_equal(saved, mask)


# ---------------------------------------------------------------------------
# Owner-review replay-contract regressions (round 2)
# ---------------------------------------------------------------------------

def _fisher_record_from_call(data, **policy):
    z, report = fisher_r_to_z(data, return_report=True, **policy)
    return z, make_fisher_entry(stage="app_design_button", scope="dataset_in_memory",
                                report=report)


class TestOwnerReplayContract:
    def test_partial_fisher_replays_exactly_via_real_cli(self, tmp_path):
        """partial_correlation + oas + standardize + upstream clip-r replay.

        The construction block must survive next to the Fisher adapter
        (round-1 defect: kind was dropped and the CLI silently computed
        Pearson correlations instead of partial ones).
        """
        from conninfpy.connectivity import compute_connectivity

        rng = np.random.default_rng(44)
        ts = rng.normal(size=(12, 40, 4))
        ts[:, :, 1] += 0.7 * ts[:, :, 0]
        np.save(tmp_path / "ts.npy", ts)
        pheno = pd.DataFrame({"subject_id": [f"s{i}" for i in range(12)],
                              "group": ["a"] * 6 + ["b"] * 6})
        pheno.to_csv(tmp_path / "pheno.csv", index=False)
        (tmp_path / "study.yaml").write_text(yaml.safe_dump({
            "schema_version": 1, "loader": "NumpyLoader",
            "paths": {"data_path": "ts.npy", "pheno_path": "pheno.csv"},
            "params": {"data_kind": "timeseries"},
        }))
        # Application flow: construct partial correlations with shrinkage,
        # then the design-step Fisher button with a NON-default clip_r.
        partial = compute_connectivity(ts, kind="partial_correlation", shrinkage="oas",
                                       standardize=True)
        z, entry = _fisher_record_from_call(partial, clip_r=0.99)
        res = analyze(group1=z[:6], group2=z[6:], test_type="two-sample", input="z",
                      method="tstat", n_permutations=20, use_mp=False, rng=7,
                      var_equal=True, fisher_history=[entry])
        plan = _run_plan(input="z", data_kind="fisher_z", analysis_scale="fisher_z",
                         fisher=res.fisher, reference_group="a", target_group="b")
        info = _dataset_info(mode="Import from manifest",
                             source=str(tmp_path / "study.yaml"),
                             manifest_file=str(tmp_path / "study.yaml"),
                             loader_provenance={"connectivity": {
                                 "kind": "partial_correlation", "shrinkage": "oas",
                                 "standardize": True}})
        data, _ = _archive(res, plan=plan, info=info, project_root=tmp_path, atlas=None)
        zf = _open(data)
        cfg = yaml.safe_load(zf.read("analysis.yaml"))
        conn = cfg["connectivity"]
        assert conn["kind"] == "partial_correlation"      # not dropped again
        assert conn["shrinkage"] == "oas" and conn["standardize"] is True
        assert conn["fisher"] == {"policy": "clip_r", "clip_r": 0.99}
        target = _extract_archive(tmp_path, data)
        _run_cli(target / "analysis.yaml", out_name="replay_partial")
        with np.load(tmp_path / "archive" / "replay_partial" / "pvals.npz") as zr:
            assert np.array_equal(zr["positive"], np.load(io.BytesIO(zf.read("arrays/p_positive.npy"))))
            assert np.array_equal(zr["stat_signed"], np.load(io.BytesIO(zf.read("arrays/stat_signed.npy"))))

    def test_nondefault_fisher_parameters_are_never_invented(self, tmp_path):
        probe = np.array([[0.0, 0.9], [0.9, 0.0]])[None, ...]
        (tmp_path / "m.yaml").write_text("loader: placeholder\n")
        # clip_r=0.99 must stay 0.99 (round-1 defect substituted 0.9999).
        _, entry99 = _fisher_record_from_call(probe, clip_r=0.99)
        replay = build_replay_config(
            _run_plan(input="z", fisher={"schema_version": 1, "applied": [],
                                         "upstream": [entry99]}),
            _dataset_info(mode="Import from manifest", source=str(tmp_path / "m.yaml"),
                          manifest_file=str(tmp_path / "m.yaml")),
            project_root=tmp_path)
        (tmp_path / "m.yaml").write_text("loader: placeholder\n")
        assert replay["status"] == "replayable"
        assert replay["config"]["connectivity"]["fisher"] == {"policy": "clip_r", "clip_r": 0.99}
        # clip_z with a non-default max_z=4 must stay 4 (not the default 5).
        _, entry4 = _fisher_record_from_call(probe, handle_bounds=True, max_z=4)
        replay4 = build_replay_config(
            _run_plan(input="z", fisher={"schema_version": 1, "applied": [],
                                         "upstream": [entry4]}),
            _dataset_info(mode="Import from manifest", source=str(tmp_path / "m.yaml"),
                          manifest_file=str(tmp_path / "m.yaml")),
            project_root=tmp_path)
        assert replay4["config"]["connectivity"]["fisher"] == {"policy": "clip_z", "max_z": 4}
        # A policy without its recorded parameter is a blocker, not a default.
        stripped = {"stage": "app_design_button", "scope": "dataset_in_memory",
                    "source": "conninfpy.fisher_r_to_z", "policy": "clip_r",
                    "parameters": {}, "counts": {"policy": "clip_r"}}
        replay_none = build_replay_config(
            _run_plan(input="z", fisher={"schema_version": 1, "applied": [],
                                         "upstream": [stripped]}),
            _dataset_info(mode="Import from manifest", source=str(tmp_path / "m.yaml"),
                          manifest_file=str(tmp_path / "m.yaml")),
            project_root=tmp_path)
        assert replay_none["status"] == "not-replayable"
        assert any("not invented" in r for r in replay_none["reasons"])

    def test_cli_fisher_block_requires_explicit_parameters(self, tmp_path):
        g1, g2, _ = _write_study(tmp_path)
        cfg = {"data": str(tmp_path / "study.yaml"),
               "design": {"type": "glm", "interest": "age"},
               "inference": {"method": "tstat", "n_permutations": 5, "use_mp": False, "rng": 0},
               "connectivity": {"fisher": {"policy": "clip_r"}},
               "output": "bad"}
        path = tmp_path / "no_param.yaml"
        path.write_text(yaml.safe_dump(cfg))
        env = dict(os.environ, OMP_NUM_THREADS="1")
        proc = subprocess.run([sys.executable, "-m", "conninfpy", "run", str(path)],
                              capture_output=True, text=True, timeout=600, env=env,
                              cwd=str(REPO))
        assert proc.returncode != 0
        assert "explicit number" in proc.stderr

    def test_atlas_labels_replay_end_to_end_without_atlas(self, tmp_path):
        """Captured atlas-network labels replay as the actual vector.

        Round-1 defect: only sc_prior labels were emitted; an atlas-sourced
        partition was silently replaced by whatever the CLI-side atlas (or
        its absence) would produce today.
        """
        rng = np.random.default_rng(5)
        Y = _batch(rng, 14)
        Y[:, 1, 4] += 0.15; Y[:, 4, 1] += 0.15
        age = rng.integers(20, 45, 14).astype(float)   # CSV-exact roundtrip
        captured_labels = [9, 9, 8, 8, 7, 7]     # first-appearance-free coding
        res = analyze(Y=Y, interest=age, input="r", method="cnbs", n_permutations=10,
                      use_mp=False, rng=7, net_labels=np.asarray(captured_labels),
                      r_to_z=False)
        assert res.inference.block_report is not None
        np.save(tmp_path / "conn.npy", Y)
        pd.DataFrame({"subject_id": [f"s{i}" for i in range(14)], "age": age}
                     ).to_csv(tmp_path / "pheno.csv", index=False)
        (tmp_path / "study.yaml").write_text(yaml.safe_dump({
            "schema_version": 1, "loader": "NumpyLoader",
            "paths": {"data_path": "conn.npy", "pheno_path": "pheno.csv"},
        }))
        plan = _run_plan(design_family="glm", question_type="Continuous Predictor",
                         interest_var="age", method="cnbs", t_statistic="glm",
                         input="r", n_permutations=10,
                         net_labels={"source": "atlas_networks",
                                     "labels": captured_labels})
        info = _dataset_info(mode="Import from manifest",
                             source=str(tmp_path / "study.yaml"),
                             manifest_file=str(tmp_path / "study.yaml"))
        data, _ = _archive(res, plan=plan, info=info, project_root=tmp_path)
        zf = _open(data)
        cfg = yaml.safe_load(zf.read("analysis.yaml"))
        assert cfg["inference"]["net_labels"] == captured_labels   # the actual vector
        assert "atlas" not in cfg                                  # nothing substituted
        target = _extract_archive(tmp_path, data)
        _run_cli(target / "analysis.yaml", out_name="replay_labels")
        with np.load(tmp_path / "archive" / "replay_labels" / "pvals.npz") as zr:
            assert np.array_equal(zr["positive"], np.load(io.BytesIO(zf.read("arrays/p_positive.npy"))))
            assert np.array_equal(zr["negative"], np.load(io.BytesIO(zf.read("arrays/p_negative.npy"))))

    def test_relative_manifest_replays_from_independent_directory(self, tmp_path):
        """A relatively-captured manifest must not break after extraction.

        The YAML carries the resolved absolute path, so the replay works from
        any directory; raw data are never copied into the archive.
        """
        repo_root = tmp_path / "repo"
        (repo_root / "datasets").mkdir(parents=True)
        g1, g2, _ = _write_study(repo_root / "datasets")
        res = analyze(group1=g1, group2=g2, test_type="two-sample", input="r",
                      method="tstat", n_permutations=20, use_mp=False, rng=7,
                      atlas=ATLAS, r_to_z=False, var_equal=True)
        info = _dataset_info(mode="Import from manifest", source="datasets/study.yaml",
                             manifest_file="datasets/study.yaml")
        data, _ = _archive(res, plan=_run_plan(reference_group="hc", target_group="pt"),
                           info=info, project_root=repo_root)
        zf = _open(data)
        cfg = yaml.safe_load(zf.read("analysis.yaml"))
        assert Path(cfg["data"]).is_absolute() and Path(cfg["data"]).exists()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        extract = _extract_archive(tmp_path, data)
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1", MPLBACKEND="Agg")
        proc = subprocess.run(
            [sys.executable, "-m", "conninfpy", "run", str(extract / "analysis.yaml"),
             "--out", str(elsewhere / "out")],
            capture_output=True, text=True, timeout=600, env=env, cwd=str(tmp_path))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        with np.load(elsewhere / "out" / "pvals.npz") as zr:
            assert np.array_equal(zr["positive"], np.load(io.BytesIO(zf.read("arrays/p_positive.npy"))))

    def test_missing_launch_facts_block_replay_honestly(self, tmp_path):
        (tmp_path / "m.yaml").write_text("loader: placeholder\n")
        info = _dataset_info(mode="Import from manifest", source=str(tmp_path / "m.yaml"),
                             manifest_file=str(tmp_path / "m.yaml"))
        no_seed = _run_plan(); del no_seed["seed"]
        r = build_replay_config(no_seed, info, project_root=tmp_path)
        assert r["status"] == "not-replayable" and r["config"] is None
        assert any("seed" in reason for reason in r["reasons"])
        no_tail = _run_plan(); del no_tail["two_tailed"]
        r = build_replay_config(no_tail, info, project_root=tmp_path)
        assert r["status"] == "not-replayable"
        assert any("directional family" in reason for reason in r["reasons"])
        no_var = _run_plan(); del no_var["t_statistic"]
        r = build_replay_config(no_var, info, project_root=tmp_path)
        assert r["status"] == "not-replayable"
        assert any("Student/Welch" in reason for reason in r["reasons"])
        # A value the result genuinely records may back the YAML, once each,
        # under the explicit contract — here the NBS threshold.
        rng = np.random.default_rng(9)
        g1, g2 = _batch(rng, 10), _batch(rng, 10)
        res = analyze(group1=g1, group2=g2, test_type="two-sample", input="r",
                      method="nbs", n_permutations=10, use_mp=False, rng=7,
                      threshold=2.2, var_equal=True)
        plan = _run_plan(method="nbs", reference_group="hc", target_group="pt")
        plan.pop("operator_kwargs")
        plan.pop("method_parameters", None)
        r = build_replay_config(plan, info, result=res, project_root=tmp_path)
        assert r["status"] == "replayable"
        assert r["config"]["inference"]["threshold"] == 2.2
        assert r["config"]["inference"]["nbs_stat"] == "extent"
        assert "inference.threshold" in r["sourced_from_result"]

    def test_manifest_dependencies_are_explicit_prerequisites(self, tmp_path):
        dep_dir = tmp_path / "study"
        dep_dir.mkdir()
        (dep_dir / "pheno.csv").write_text("subject_id,age\ns0,30\n")
        (dep_dir / "study.yaml").write_text(yaml.safe_dump({
            "schema_version": 1, "loader": "NumpyLoader",
            "paths": {"data_path": "missing.npy", "pheno_path": "pheno.csv"},
        }))
        info = _dataset_info(mode="Import from manifest",
                             source=str(dep_dir / "study.yaml"),
                             manifest_file=str(dep_dir / "study.yaml"))
        r = build_replay_config(_run_plan(reference_group="hc", target_group="pt"),
                                info, project_root=tmp_path)
        assert r["status"] == "requires-input"
        missing = [p for p in r["prerequisites"] if p["kind"] == "missing-file"]
        assert any(p["path"].endswith("missing.npy") for p in missing)
        # An unreadable manifest is not replayable.
        broken = tmp_path / "broken.yaml"
        broken.write_text("::not yaml {\n")
        info2 = _dataset_info(mode="Import from manifest", source=str(broken),
                              manifest_file=str(broken))
        r2 = build_replay_config(_run_plan(), info2, project_root=tmp_path)
        assert r2["status"] == "not-replayable"
        assert any("cannot be read" in reason for reason in r2["reasons"])


    def test_enhancement_backend_capture_distinguishes_installation_and_execution(self):
        """The worker records installation facts, never a guessed backend.

        Methods without enhancement kernels are marked not-applicable;
        kernel methods stay unknown (nothing records which backend executed);
        numba presence is a separate installation fact.
        """
        from apps.components.design_inference import InferenceTask, _run_inference_background

        rng = np.random.default_rng(11)
        g1 = _batch(rng, 8)
        g2 = _batch(rng, 8, shift=0.05)

        def _plan_for(method, **op_kwargs):
            task = InferenceTask()
            _run_inference_background(
                task,
                Y=None, interest=None, confounds=None,
                group1=g1, group2=g2, test_type="two-sample", sites=None,
                harmonization_choice=None, operator_choice=method, n_perms=5,
                seed_val=1, use_mp=False, acceleration_choice="none",
                op_kwargs=dict(op_kwargs), run_sensitivity=False, comp_method=None,
                comp_kwargs={}, sc_net_labels=None, use_sc_prior=False,
                atlas=ATLAS, annotation_atlas=ATLAS, alpha=0.05,
                question_choice="Group Difference", loaded_settings_hash="h",
                recipe_choice="auto", effective_recipe_choice="auto",
                group_col="group", ref_group="0", target_group="1",
                interest_var=None, subject_col="subject_id", condition_col=None,
                baseline_val=None, target_val=None, confound_vars=[], site_col=None,
                active_atlas_signature=None, data_kind="correlation",
            )
            assert task.status == "success", task.error
            return task.run_plan

        tstat_plan = _plan_for("tstat")
        assert tstat_plan["enhancement_backend"] == "not-applicable"
        assert isinstance(tstat_plan["numba_module_found"], bool)
        cnbs_plan = _plan_for("cnbs", net_labels=list(ATLAS.network_index()))
        assert cnbs_plan["enhancement_backend"] is None
        assert isinstance(cnbs_plan["numba_module_found"], bool)

    def test_methods_md_qualifies_backend_facts_and_legacy_labels(self):
        g1, g2 = _two_groups()
        res = analyze(group1=g1, group2=g2, test_type="two-sample", input="r",
                      method="tstat", n_permutations=20, use_mp=False, rng=7,
                      atlas=ATLAS)
        # New capture: installation fact + not-applicable for tstat.
        data, _ = _archive(res, plan=_run_plan(enhancement_backend="not-applicable",
                                               numba_module_found=True))
        methods = _open(data).read("methods.md").decode()
        assert "not applicable — this method runs no enhancement-kernel computation" in methods
        assert "numba module present in the run environment (installation fact" in methods
        # Legacy ambiguous label: qualified, never relabelled as executed.
        legacy = _open(_archive(res, plan=_run_plan(enhancement_backend="scipy"))[0]) \
            .read("methods.md").decode()
        assert "that label measured module availability, not execution" in legacy
        # Kernel method without any runtime record: executed stays unknown.
        rng = np.random.default_rng(4)
        Y = _batch(rng, 14)
        age = rng.integers(20, 45, 14).astype(float)
        cnbs = analyze(Y=Y, interest=age, input="r", method="cnbs", n_permutations=5,
                       use_mp=False, rng=7, net_labels=np.asarray(ATLAS.network_index()))
        cnbs_methods = _open(_archive(
            cnbs, plan=_run_plan(design_family="glm", question_type="Continuous Predictor",
                                 interest_var="age", method="cnbs", t_statistic="glm",
                                 enhancement_backend=None, numba_module_found=False))[0]) \
            .read("methods.md").decode()
        assert "Enhancement backend (executed): unknown / not recorded" in cnbs_methods


# ---------------------------------------------------------------------------
# Real Streamlit AppTest of the export flow
# ---------------------------------------------------------------------------

streamlit = pytest.importorskip("streamlit")


def test_app_export_button_downloads_real_archive(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    from streamlit.runtime.media_file_manager import MediaFileManager

    # Capture the payload the real app registers for the real download
    # button (this Streamlit version stores it in the media manager instead
    # of the element proto; nothing in the app is mocked).
    captured = {}
    original_add = MediaFileManager.add

    def spy_add(self, path_or_data, mimetype, coordinates, file_name=None,
                is_for_static_download=False):
        if is_for_static_download and str(file_name or "").endswith(".zip"):
            captured["data"] = path_or_data
            captured["name"] = file_name
            captured["mime"] = mimetype
        return original_add(self, path_or_data, mimetype, coordinates,
                            file_name=file_name,
                            is_for_static_download=is_for_static_download)

    monkeypatch.setattr(MediaFileManager, "add", spy_add)

    g1, g2 = _two_groups()
    result = analyze(group1=g1, group2=g2, test_type="two-sample", input="r",
                     method="tstat", n_permutations=20, use_mp=False, rng=7,
                     atlas=ATLAS)
    app = AppTest.from_file(str(REPO / "apps" / "streamlit_app.py"), default_timeout=180)
    app.session_state.connectivity_data = np.concatenate([g1, g2], axis=0)
    app.session_state.pheno_df = pd.DataFrame({
        "subject_id": [f"s{i}" for i in range(20)],
        "group": [0] * 10 + [1] * 10,
    })
    app.session_state.loaded_reference_atlas = ATLAS
    app.session_state.loaded_atlas_signature = "fixture atlas"
    app.session_state.loaded_settings_hash = "fixture data"
    app.session_state.dataset_info = _dataset_info()
    app.session_state.inference_result = result
    app.session_state.edges_df = result.significant_edges(atlas=ATLAS, alpha=0.05)
    app.session_state.run_plan = _run_plan()
    app.session_state.active_tab = "3. Inference Results"
    app.run()
    assert not app.exception
    downloads = [d for d in app.get("download_button")
                 if "archive" in (d.proto.label or "").lower()]
    assert downloads, "archive download button not rendered"
    assert captured["mime"] == "application/zip"
    assert captured["name"].startswith("conninf_archive_")
    payload = captured["data"]
    assert payload and payload[:2] == b"PK"
    zf = _open(payload)
    m = _manifest_of(zf)
    assert m["schema_version"] == ARCHIVE_SCHEMA_VERSION
    assert "methods.md" in zf.namelist() and "bundle.pkl" in zf.namelist()
