"""NETSTAT-29: preprocessing provenance, shared Fisher policies, floor docs.

Covers the four acceptance criteria with small oracles:

* AC1 — three-state preprocessing facts (gsr / estimator / standardize /
  shrinkage / tr_seconds with sources) thread loader → analyze → bundle →
  reload; explicit ``False`` never becomes missing; unknown stays unknown.
* AC2 — one shared Fisher implementation with both historical policies kept
  numerically intact; the clip report separates the structural diagonal from
  observed off-diagonal values with explicit count units.
* AC3 — partial-Fisher scale labels describe atanh(partial r) as a scale
  transform, and coupling context disclaims causal/physiological reading.
* AC4 — the scale-relative degeneracy floor is exercised by behaviour
  oracles (its numerical value is not re-tuned here).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conninfpy import ResultBundle, analyze, fisher_r_to_z
from conninfpy._prepare import make_fisher_entry, normalize_fisher_history, prepare_connectivity
from conninfpy.loaders import load
from conninfpy.connectivity import compute_connectivity, connectivity_from_dataset
from conninfpy.loaders.base import LoadedDataset
from conninfpy.loaders.builtins import ConditionTimeseriesArrayLoader
from conninfpy.loaders.manifest import ManifestLoader
from conninfpy.preprocessing import (
    SOURCE_COMPUTED,
    SOURCE_FILE_TOKEN,
    SOURCE_LOADER_DENOISING,
    SOURCE_MANIFEST,
    aggregate_preprocessing_records,
    attach_preprocessing,
    connectivity_method_from_kind,
    empty_record,
    fact,
    gsr_from_file_name,
    gsr_from_file_token,
    merge_preprocessing_records,
    normalize_preprocessing_record,
    record_from_manifest_applied,
    set_preprocessing_facts,
)

pytest.importorskip("streamlit")  # the UI helpers below import streamlit
from apps.utils.helpers import describe_preprocessing_facts, fisher_clip_note
from apps.utils.result_views import describe_result_scale_and_stat

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# AC2 — one shared Fisher implementation, both historical policies intact
# ---------------------------------------------------------------------------

class TestFisherPolicies:
    """fisher_r_to_z: clip-z default and clip-r policy preserve history."""

    def test_default_policy_matches_direct_oracle(self):
        rng = np.random.default_rng(3)
        r = np.tanh(rng.normal(0, 0.4, (6, 5, 5)))
        r[0, 1, 2] = r[0, 2, 1] = 1.0
        r[1, 0, 0] = np.nan
        r[3, 4, 4] = 0.999995
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = fisher_r_to_z(r)
        oracle = np.clip(np.arctanh(r), -5.0, 5.0)
        np.testing.assert_allclose(z, oracle, equal_nan=True)

    def test_default_policy_warning_count_unchanged(self):
        r = np.array([1.0, 0.99999, 0.5, np.nan])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fisher_r_to_z(r)
        assert len(caught) == 1
        assert "(2 value(s))" in str(caught[0].message)

    def test_clip_r_policy_equals_historical_ui_formula(self):
        rng = np.random.default_rng(4)
        r = np.tanh(rng.normal(0, 0.4, (4, 6, 6)))
        r[:, 0, 1] = r[:, 1, 0] = 1.0          # exact boundary
        r[:, 2, 3] = r[:, 3, 2] = 0.99999       # between the two bounds
        r[:, 4, 4] = np.nan                     # NaN passes through
        with warnings.catch_warnings():
            warnings.simplefilter("error")      # historical policy was silent
            z = fisher_r_to_z(r, clip_r=0.9999)
        oracle = np.arctanh(np.clip(r, -0.9999, 0.9999))
        np.testing.assert_allclose(z, oracle, equal_nan=True)
        assert z[0, 0, 1] == np.arctanh(0.9999)

    def test_policies_differ_near_bounds(self):
        r = np.array([0.99995])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z_default = fisher_r_to_z(r)
        z_clip_r = fisher_r_to_z(r, clip_r=0.9999)
        assert z_default[0] == 5.0
        assert z_clip_r[0] == pytest.approx(4.9517, abs=1e-4)
        assert z_default[0] != z_clip_r[0]

    def test_clip_r_validation(self):
        with pytest.raises(ValueError):
            fisher_r_to_z(np.array([0.5]), clip_r=1.0)
        with pytest.raises(ValueError):
            fisher_r_to_z(np.array([0.5]), clip_r=0.0)
        with pytest.raises(ValueError):
            fisher_r_to_z(np.array([0.5]), clip_r=True)
        with pytest.raises(ValueError):
            fisher_r_to_z(np.array([0.5]), clip_r=0.9999, handle_bounds=False)
        with pytest.raises(ValueError):
            fisher_r_to_z(np.array([1.5]), clip_r=0.9999)

    def test_unbounded_policy_report_counts_zero_and_reports_infinity(self):
        r = np.array([[1.0, 0.5], [-0.5, -1.0]])
        with pytest.warns(UserWarning, match="infinite"):
            z, report = fisher_r_to_z(r, handle_bounds=False, return_report=True)
        assert z[0, 0] == np.inf and z[1, 1] == -np.inf
        assert z[0, 1] == pytest.approx(0.5493, abs=1e-4)
        # Nothing is clipped under the unbounded policy.
        assert report["policy"] == "unbounded"
        assert report["n_values_clipped"] == 0
        assert report["diagonal"]["n_cells_clipped"] == 0
        assert report["off_diagonal"]["n_cells_clipped"] == 0
        assert report["off_diagonal"]["n_unique_undirected_edges_clipped"] == 0
        # Values left at ±infinity are separate diagnostics (same units).
        assert report["left_at_infinity"]["n_values"] == 2
        assert report["left_at_infinity"]["diagonal"]["n_cells"] == 2
        assert report["left_at_infinity"]["off_diagonal"]["n_cells"] == 0

    def test_unbounded_note_never_says_clipped(self):
        with pytest.warns(UserWarning):
            _, report = fisher_r_to_z(np.array([[1.0, 0.5], [0.5, 1.0]]),
                                      handle_bounds=False, return_report=True)
        note = fisher_clip_note(report)
        assert note is not None and "clipped" not in note
        assert "left as ±infinity" in note

    def test_prepare_connectivity_default_numbers_unchanged(self):
        rng = np.random.default_rng(5)
        r = np.tanh(rng.normal(0, 0.3, (7, 5, 5)))
        r = (r + r.transpose(0, 2, 1)) / 2
        r[:, 0, 1] = r[:, 1, 0] = 1.0
        idx = np.arange(5)
        r[:, idx, idx] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prepared = prepare_connectivity(r, "r")
        oracle = np.clip(np.arctanh(r), -5, 5)
        np.testing.assert_allclose(prepared, oracle, equal_nan=True)


class TestFisherClipReport:
    """Clip report: diagonal vs off-diagonal, explicit count units."""

    def _stack(self):
        a = np.eye(4)
        a[0, 1] = a[1, 0] = 0.99995        # observed saturated edge (both cells)
        a[3, 2] = a[2, 3] = -1.0           # observed saturated edge (exact -1)
        b = np.eye(4)                       # only the structural diagonal
        return np.stack([a, b])

    def test_report_separates_diagonal_and_edges(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, report = fisher_r_to_z(self._stack(), return_report=True)
        assert report["policy"] == "clip_z"
        assert report["parameters"] == {"max_z": 5.0}
        assert report["n_observations"] == 2
        assert report["matrix_shape"] == (4, 4)
        # 8 diagonal cells (4 per matrix) are the expected r = 1 diagonal
        assert report["diagonal"]["n_cells_clipped"] == 8
        # 4 off-diagonal cells = 2 undirected edges x 2 symmetric cells, in one subject
        assert report["off_diagonal"]["n_cells_clipped"] == 4
        assert report["off_diagonal"]["n_unique_undirected_edges_clipped"] == 2
        assert "matrix cells" in report["count_units"]
        assert "undirected" in report["count_units"]
        assert report["n_values_clipped"] == 12

    def test_report_counts_cells_per_observation_edges_deduplicated(self):
        # The same edge saturated in two subjects: 4 cells, 1 unique edge.
        a = np.eye(4); a[0, 1] = a[1, 0] = 0.99999
        b = np.eye(4); b[0, 1] = b[1, 0] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, report = fisher_r_to_z(np.stack([a, b]), return_report=True)
        assert report["n_observations"] == 2
        assert report["diagonal"]["n_cells_clipped"] == 8
        assert report["off_diagonal"]["n_cells_clipped"] == 4
        assert report["off_diagonal"]["n_unique_undirected_edges_clipped"] == 1

    def test_report_clip_r_policy_counts_before_transform(self):
        a = np.eye(3); a[0, 2] = a[2, 0] = 0.99991
        z, report = fisher_r_to_z(np.stack([a]), clip_r=0.9999, return_report=True)
        assert report["policy"] == "clip_r"
        assert report["parameters"] == {"clip_r": 0.9999}
        assert report["diagonal"]["n_cells_clipped"] == 3
        assert report["off_diagonal"]["n_cells_clipped"] == 2
        assert report["off_diagonal"]["n_unique_undirected_edges_clipped"] == 1
        np.testing.assert_allclose(z, np.arctanh(np.clip(np.stack([a]), -0.9999, 0.9999)))

    def test_non_matrix_input_has_no_matrix_sections(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, report = fisher_r_to_z(np.array([0.5, 0.99995]), return_report=True)
        assert report["matrix_shape"] is None
        assert report["n_observations"] is None
        assert report["diagonal"] is None
        assert report["off_diagonal"] is None
        assert report["n_values_clipped"] == 1

    def test_ui_note_lists_units_and_diagonal_caveat(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, report = fisher_r_to_z(self._stack(), return_report=True)
        note = fisher_clip_note(report)
        assert "off-diagonal matrix cell(s)" in note
        assert "unique undirected edge(s)" in note
        assert "diagonal cell(s) (r = 1 by construction, not observed edge saturation)" in note
        assert fisher_clip_note({"policy": "clip_z", "parameters": {}, "diagonal": {},
                                 "off_diagonal": {}}) is None


# ---------------------------------------------------------------------------
# AC1 — three-state preprocessing record
# ---------------------------------------------------------------------------

class TestPreprocessingRecord:
    def test_missing_fact_is_unknown_and_explicit_false_survives(self):
        record = normalize_preprocessing_record({"gsr": fact(False, SOURCE_MANIFEST)})
        assert record["gsr"] == {"value": False, "source": SOURCE_MANIFEST}
        assert "tr_seconds" not in record          # unknown, not False/0
        again = normalize_preprocessing_record(record)
        assert again == record                      # roundtrip keeps explicit False

    def test_validation_rejects_malformed_facts(self):
        with pytest.raises(ValueError):
            normalize_preprocessing_record({"gsr": {"value": False}})          # no source
        with pytest.raises(ValueError):
            normalize_preprocessing_record({"gsr": "no"})                      # not a fact
        with pytest.raises(ValueError):
            normalize_preprocessing_record({"connectivity_method": fact("mutual", "x")})
        with pytest.raises(ValueError):
            normalize_preprocessing_record({"shrinkage": fact(1.5, "x")})
        with pytest.raises(ValueError):
            normalize_preprocessing_record({"tr_seconds": fact(-2.0, "x")})
        with pytest.raises(ValueError):
            normalize_preprocessing_record({"gsr?": fact(True, "x")})          # typo key

    def test_future_schema_version_is_rejected_not_rewritten(self):
        with pytest.raises(ValueError, match="schema_version"):
            normalize_preprocessing_record({"schema_version": 2, "gsr": fact(True, "x")})
        from conninfpy._prepare import normalize_fisher_record
        with pytest.raises(ValueError, match="schema_version"):
            normalize_fisher_record({"schema_version": 2, "applied": [], "upstream": None})

    def test_shrinkage_none_is_a_known_value(self):
        record = normalize_preprocessing_record({"shrinkage": fact(None, SOURCE_COMPUTED)})
        assert record["shrinkage"] == {"value": None, "source": SOURCE_COMPUTED}

    def test_merge_library_beats_declaration(self):
        performed = {"gsr": fact(False, SOURCE_LOADER_DENOISING)}
        declared = {"gsr": fact(True, SOURCE_MANIFEST), "tr_seconds": fact(2.0, SOURCE_MANIFEST)}
        merged = merge_preprocessing_records(performed, declared)
        assert merged["gsr"]["value"] is False                # performed wins
        assert merged["gsr"]["source"] == SOURCE_LOADER_DENOISING
        assert merged["tr_seconds"] == {"value": 2.0, "source": SOURCE_MANIFEST}
        assert merge_preprocessing_records(None, None) == empty_record()

    def test_gsr_file_tokens(self):
        assert gsr_from_file_token("GSR") is True
        assert gsr_from_file_token("gsr") is True
        assert gsr_from_file_token("noGSR") is False
        assert gsr_from_file_token("no_GSR") is False
        assert gsr_from_file_token("no-GSR") is False
        assert gsr_from_file_token("whatever") is None
        assert gsr_from_file_token(None) is None
        assert gsr_from_file_name("ihb_open_Schaefer200_strategy-4_GSR.npy") is True
        assert gsr_from_file_name("sub-01_task-rest_noGSR.npy") is False
        assert gsr_from_file_name("sub-01_task-rest_bold.npy") is None
        # Negation must survive separators and case, and never read as GSR.
        assert gsr_from_file_name("s_no_GSR.npy") is False
        assert gsr_from_file_name("s_no-GSR.npy") is False
        assert gsr_from_file_name("s_no.gsr.npy") is False
        assert gsr_from_file_name("s_NO_gsr.npy") is False
        assert gsr_from_file_name("xgsr_bold.npy") is None   # embedded, not a token

    def test_manifest_applied_parsing(self):
        declared = record_from_manifest_applied({
            "use_GSR": False, "correlation": "pearson", "tr_seconds": 2.0,
            "denoising_strategy": 4, "spatial_preprocessing": "SPM",
        })
        assert declared["gsr"] == {"value": False, "source": SOURCE_MANIFEST}
        assert declared["connectivity_method"] == {"value": "pearson", "source": SOURCE_MANIFEST}
        assert declared["tr_seconds"] == {"value": 2.0, "source": SOURCE_MANIFEST}
        assert "standardize" not in declared      # not declared → unknown
        assert "denoising_strategy" not in declared
        with pytest.raises(ValueError):
            record_from_manifest_applied({"shrinkage": 7})
        assert record_from_manifest_applied(None) == empty_record()

    def test_connectivity_backend_records_library_facts(self):
        rng = np.random.default_rng(6)
        ts = [rng.normal(size=(30, 6)) for _ in range(4)]
        dataset = LoadedDataset(
            data=ts, pheno=pd.DataFrame({"subject_id": [f"s{i}" for i in range(4)]}),
            data_kind="timeseries", provenance={},
        )
        connectivity_from_dataset(dataset, kind="correlation")       # Pearson → no shrinkage
        record = dataset.provenance["preprocessing"]
        assert record["connectivity_method"] == {"value": "pearson", "source": SOURCE_COMPUTED}
        assert record["shrinkage"] == {"value": None, "source": SOURCE_COMPUTED}   # known none
        assert record["standardize"] == {"value": True, "source": SOURCE_COMPUTED}
        assert "gsr" not in record                                   # unknown, not guessed
        assert connectivity_method_from_kind("partial_correlation") == "partial"

        dataset2 = LoadedDataset(
            data=ts, pheno=pd.DataFrame({"subject_id": [f"s{i}" for i in range(4)]}),
            data_kind="timeseries", provenance={},
        )
        connectivity_from_dataset(dataset2, kind="partial_correlation", shrinkage="oas")
        assert dataset2.provenance["preprocessing"]["shrinkage"] == {
            "value": "oas", "source": SOURCE_COMPUTED}

    def test_set_facts_replaces_estimator_for_spearman(self):
        dataset = LoadedDataset(
            data=[np.eye(3) for _ in range(2)],
            pheno=pd.DataFrame({"subject_id": ["a", "b"]}),
            data_kind="correlation", provenance={},
        )
        dataset.provenance["connectivity"] = {"kind": "correlation"}
        dataset.provenance["connectivity"]["kind"] = "spearman"
        set_preprocessing_facts(dataset, connectivity_method=fact("spearman", "app_rank_correlation_pipeline"))
        assert dataset.provenance["preprocessing"]["connectivity_method"] == {
            "value": "spearman", "source": "app_rank_correlation_pipeline"}


def _condition_loader_dataset(tmp_path: Path, *, gsr: str = "GSR"):
    """Small paired-condition .npy fixture for ConditionTimeseriesArrayLoader."""
    rng = np.random.default_rng(7)
    data_dir = tmp_path / f"oc_{gsr}".replace("-", "_")
    data_dir.mkdir(exist_ok=True)
    for cond in ("open", "close"):
        np.save(data_dir / f"ihb_{cond}_Schaefer200_strategy-4_{gsr}.npy",
                rng.normal(size=(3, 20, 5)))
    order = tmp_path / "order.txt"
    order.write_text("s1\ns2\ns3\n")
    loader = ConditionTimeseriesArrayLoader(
        data_dir=str(data_dir), subject_order_path=str(order),
        conditions=["open", "close"], gsr=gsr,
    )
    return loader.load()


class TestLoaderRecords:
    def test_condition_loader_gsr_from_file_token(self, tmp_path):
        ds = _condition_loader_dataset(tmp_path, gsr="GSR")
        record = ds.provenance["preprocessing"]
        assert record["gsr"] == {"value": True, "source": SOURCE_FILE_TOKEN}
        assert record["connectivity_method"] == {"value": "pearson", "source": SOURCE_COMPUTED}
        assert "n_valid_tr" in ds.pheno and "delta_s" in ds.pheno

    def test_condition_loader_no_gsr_token_and_unknown_token(self, tmp_path):
        ds = _condition_loader_dataset(tmp_path, gsr="noGSR")
        assert ds.provenance["preprocessing"]["gsr"] == {"value": False, "source": SOURCE_FILE_TOKEN}
        ds2 = _condition_loader_dataset(tmp_path, gsr="custom")
        assert "gsr" not in ds2.provenance["preprocessing"]

    def test_manifest_declared_facts_fill_unknowns(self, tmp_path):
        rng = np.random.default_rng(8)
        data_dir = tmp_path / "ts"
        data_dir.mkdir()
        for sub in ("sub-01", "sub-02"):
            np.savetxt(data_dir / f"{sub}.csv", rng.normal(size=(25, 4)), delimiter=",")
        manifest = tmp_path / "study.yaml"
        manifest.write_text(
            "schema_version: 1\n"
            "name: declared\n"
            "loader: TimeseriesDirectoryLoader\n"
            "paths:\n  dir_path: ts\n"
            "params:\n  pattern: '*.csv'\n"
            "preprocessing:\n"
            "  applied:\n"
            "    use_GSR: false\n"
            "    tr_seconds: 1.5\n"
            "  requested:\n"
            "    correlation: pearson\n"
            "    fisher_z: true\n"
        )
        ds = ManifestLoader(str(manifest)).load()
        record = ds.provenance["preprocessing"]
        assert record["gsr"] == {"value": False, "source": SOURCE_MANIFEST}
        assert record["tr_seconds"] == {"value": 1.5, "source": SOURCE_MANIFEST}
        # Raw timeseries loader: the estimator was not computed at load → unknown.
        assert "connectivity_method" not in record


# ---------------------------------------------------------------------------
# AC1 — analyze → provenance → bundle → reload
# ---------------------------------------------------------------------------

def _two_groups(seed=11, n=12, N=5):
    rng = np.random.default_rng(seed)
    a = np.tanh(rng.normal(0, 0.3, (n // 2, N, N)))
    b = np.tanh(rng.normal(0.05, 0.3, (n - n // 2, N, N)))
    out = []
    for g in (a, b):
        g = (g + g.transpose(0, 2, 1)) / 2
        g[:, np.arange(N), np.arange(N)] = 1.0
        out.append(g)
    return out


FAST = dict(method="tstat", n_permutations=30, rng=0, use_mp=False,
            two_tailed=True, acceleration=None)


class TestAnalyzeBundleRoundTrip:
    def test_record_reaches_provenance_and_bundle(self, tmp_path):
        g1, g2 = _two_groups()
        record = {"gsr": fact(False, SOURCE_MANIFEST),
                  "connectivity_method": fact("pearson", SOURCE_COMPUTED),
                  "tr_seconds": fact(2.0, SOURCE_MANIFEST)}
        res = analyze(group1=g1, group2=g2, input="r", preprocessing=record, **FAST)
        assert res.inference.analysis_provenance["preprocessing"] == \
            normalize_preprocessing_record(record)
        assert res.preprocessing == normalize_preprocessing_record(record)
        path = tmp_path / "bundle.pkl"
        res.to_bundle().save(path)
        reloaded = ResultBundle.load(path)
        assert reloaded.metadata["preprocessing"] == normalize_preprocessing_record(record)
        # explicit False survived the full round trip
        assert reloaded.metadata["preprocessing"]["gsr"]["value"] is False

    def test_no_record_stays_unknown_and_numbers_unchanged(self):
        g1, g2 = _two_groups()
        record = {"gsr": fact(True, SOURCE_FILE_TOKEN)}
        with_record = analyze(group1=g1, group2=g2, input="r", preprocessing=record, **FAST)
        without = analyze(group1=g1, group2=g2, input="r", **FAST)
        assert without.inference.analysis_provenance["preprocessing"] == empty_record()
        assert without.to_bundle().metadata["preprocessing"] == empty_record()
        np.testing.assert_array_equal(with_record["positive"], without["positive"])
        np.testing.assert_array_equal(with_record["negative"], without["negative"])
        np.testing.assert_array_equal(
            with_record.inference.stat_signed, without.inference.stat_signed)

    def test_one_sided_projection_keeps_record(self):
        g1, g2 = _two_groups()
        record = {"gsr": fact(False, SOURCE_MANIFEST)}
        res = analyze(group1=g1, group2=g2, input="r", preprocessing=record,
                      tested_tail="positive", **{**FAST, "two_tailed": None})
        assert res.preprocessing["gsr"]["value"] is False
        assert res.to_bundle().metadata["preprocessing"]["gsr"]["value"] is False

    def test_bundle_override_conflict_and_legacy_import(self):
        g1, g2 = _two_groups()
        record = {"gsr": fact(False, SOURCE_MANIFEST)}
        res = analyze(group1=g1, group2=g2, input="r", preprocessing=record, **FAST)
        with pytest.raises(ValueError):
            res.to_bundle(preprocessing={"gsr": fact(True, SOURCE_MANIFEST)})
        # An identical record is accepted; a legacy result can gain facts on import.
        bundle = res.to_bundle(preprocessing=record)
        assert bundle.metadata["preprocessing"]["gsr"]["value"] is False

    def test_invalid_record_rejected(self):
        g1, g2 = _two_groups()
        with pytest.raises(ValueError):
            analyze(group1=g1, group2=g2, input="r", preprocessing={"gsr": "maybe"}, **FAST)

    def test_cli_writes_record_to_provenance_json_and_bundle(self, tmp_path):
        from conninfpy.cli import run
        rng = np.random.default_rng(9)
        n, N = 12, 5
        r = np.tanh(rng.normal(0, 0.3, (n, N, N)))
        r = (r + r.transpose(0, 2, 1)) / 2
        r[:, np.arange(N), np.arange(N)] = 1.0
        np.save(tmp_path / "conn.npy", r)
        pd.DataFrame({"subject_id": [f"s{i:02d}" for i in range(n)],
                      "group": np.tile(["hc", "pt"], n // 2)}).to_csv(tmp_path / "pheno.csv", index=False)
        (tmp_path / "study.yaml").write_text(
            "schema_version: 1\n"
            "loader: NumpyLoader\n"
            "paths:\n  data_path: conn.npy\n  pheno_path: pheno.csv\n"
            "params:\n  data_kind: correlation\n"
            "preprocessing:\n  applied:\n    use_GSR: false\n    correlation: pearson\n"
        )
        (tmp_path / "analysis.yaml").write_text(json.dumps({
            "data": "study.yaml", "output": "out",
            "design": {"type": "two-sample", "group": "group", "levels": ["hc", "pt"]},
            "inference": dict(FAST),
        }) + "\n")
        written = run(tmp_path / "analysis.yaml")
        prov = json.loads((tmp_path / "out" / "provenance.json").read_text())
        record = prov["data"]["preprocessing"]
        assert record["gsr"] == {"value": False, "source": SOURCE_MANIFEST}
        assert record["connectivity_method"] == {"value": "pearson", "source": SOURCE_MANIFEST}
        bundle = ResultBundle.load(written["bundle"])
        assert bundle.metadata["preprocessing"]["gsr"]["value"] is False
        # n_valid_tr (frames) and delta_s (intensity) are exported as diagnostics,
        # never mapped onto tr_seconds.
        assert "tr_seconds" not in bundle.metadata["preprocessing"]


# ---------------------------------------------------------------------------
# AC3 — partial / native scale labels and coupling context
# ---------------------------------------------------------------------------

def _result_with_preprocessing(record, seed=13):
    g1, g2 = _two_groups(seed)
    return analyze(group1=g1, group2=g2, input="r", preprocessing=record, **FAST)


class TestScaleLabels:
    def test_partial_fisher_label_is_scale_transform(self):
        record = {"connectivity_method": fact("partial", SOURCE_COMPUTED)}
        res = _result_with_preprocessing(record)
        meta = describe_result_scale_and_stat(res)
        assert meta["scale_label"] == "Fisher z (atanh of partial r)"
        assert "not the exact standard z" in meta["scale_note"]
        assert "no" in meta["scale_note"] and "causal" in meta["scale_note"]
        # The estimator-aware default units are recorded on the result and exports.
        assert res.inference.analysis_provenance["units"] == "Fisher z (atanh of partial r)"
        assert res.to_bundle().effect_spec["units"] == "Fisher z (atanh of partial r)"

    def test_spearman_label(self):
        record = {"connectivity_method": fact("spearman", "app_rank_correlation_pipeline")}
        meta = describe_result_scale_and_stat(_result_with_preprocessing(record))
        assert meta["scale_label"] == "Fisher z (atanh of Spearman ρ)"
        assert "variance" in meta["scale_note"]

    def test_unknown_estimator_keeps_historical_labels(self):
        meta = describe_result_scale_and_stat(_result_with_preprocessing(None))
        assert meta["scale_label"] == "Fisher z"
        assert meta["effect_unit_label"] == "c' β (Δz, Fisher z units)"
        assert meta["scale_note"] == ""

    def test_run_plan_fallback_for_legacy_sessions(self):
        # Legacy shape: a pre-update analyze() result — scale captured in the
        # design/provenance, but no preprocessing record key at all.
        res = _result_with_preprocessing(None)
        del res.inference.analysis_provenance["preprocessing"]
        assert "preprocessing" not in res.inference.analysis_provenance
        plan = {"analysis_scale": "fisher_z",
                "preprocessing": {"connectivity_method": fact("partial", SOURCE_COMPUTED)}}
        meta = describe_result_scale_and_stat(res, run_plan=plan)
        assert meta["scale_label"] == "Fisher z (atanh of partial r)"
        # An analyze() result that recorded no facts stays unknown even with a plan.
        res2 = _result_with_preprocessing(None)
        meta2 = describe_result_scale_and_stat(res2, run_plan=plan)
        assert meta2["scale_label"] == "Fisher z"

    def test_coupling_note_mentions_gsr_context_without_causal_claims(self):
        res = _result_with_preprocessing({"gsr": fact(True, SOURCE_LOADER_DENOISING),
                                          "connectivity_method": fact("partial", SOURCE_COMPUTED)})
        from apps.utils.result_views import prepare_predicted_coupling_summary
        info = prepare_predicted_coupling_summary(res)
        if info["coupling_note"] is not None:
            note = info["coupling_note"]
            assert "do not by themselves imply a physiological or causal relationship" in note
            assert "Global-signal regression was applied" in note
            assert "partial correlation" in note

    def test_describe_preprocessing_facts_three_states(self):
        rows = describe_preprocessing_facts(None)
        values = {label: value for label, value, _ in rows}
        assert values["Global-signal regression (GSR)"] == "unknown"
        assert values["Connectivity estimator"] == "unknown"
        assert values["Repetition time (TR)"] == "unknown"
        rows = describe_preprocessing_facts({"gsr": fact(False, SOURCE_MANIFEST),
                                             "tr_seconds": fact(2.0, SOURCE_MANIFEST)})
        by_label = {label: (value, source) for label, value, source in rows}
        assert by_label["Global-signal regression (GSR)"] == ("no", "declared by the dataset manifest")
        assert by_label["Repetition time (TR)"] == ("2 s", "declared by the dataset manifest")


# ---------------------------------------------------------------------------
# AC4 — scale-relative degeneracy floor (behaviour oracles, value untouched)
# ---------------------------------------------------------------------------

class TestDegeneracyFloor:
    def test_floor_is_scale_relative(self):
        rng = np.random.default_rng(15)
        base = rng.normal(0, 0.2, (12, 5, 5))
        base = (base + base.transpose(0, 2, 1)) / 2
        idx = np.arange(5)
        base[:, idx, idx] = 0.0
        base[:, 0, 1] = base[:, 1, 0] = 0.3      # constant nonzero edge
        base[:, 2, 3] = base[:, 3, 2] = 0.0      # constant zero edge
        for scale in (1.0, 1000.0, 1e-4):
            from conninfpy.stats._degenerate import pairwise_infinite
            mask = pairwise_infinite(base * scale, None, "one-sample")
            assert mask[0, 1] and not mask[2, 3], f"scale {scale}"

    def test_float_noise_variance_is_treated_as_zero(self):
        # A constant edge with 1e-17 float noise: variance far below the
        # scale-relative floor → degenerate, not t ~ 1e16.
        g = np.zeros((10, 4, 4))
        g[:, 0, 1] = g[:, 1, 0] = 0.3 + np.array([0, 1e-17, -1e-17, 0, 0, 0, 0, 0, 0, 0])
        from conninfpy.stats._degenerate import pairwise_infinite
        assert pairwise_infinite(g, None, "one-sample")[0, 1]

    def test_floor_constant_is_documented_and_unchanged(self):
        from conninfpy.stats import _degenerate
        assert _degenerate.RTOL == 1e-10
        usage = (REPO / "docs" / "source" / "usage.rst").read_text()
        assert "scale-relative degeneracy floor" in usage
        assert "1e-10" in usage


# ---------------------------------------------------------------------------
# Owner reproducer defect 1 — performed computation overrides stale declarations
# ---------------------------------------------------------------------------

class TestPerformedFactsOverrideDeclarations:
    """Estimator facts state what the last computation actually resolved."""

    def _declared_timeseries_dataset(self):
        rng = np.random.default_rng(41)
        from conninfpy.loaders.base import LoadedDataset
        dataset = LoadedDataset(
            data=[rng.normal(size=(20, 5)) for _ in range(3)],
            pheno=pd.DataFrame({"subject_id": ["a", "b", "c"]}),
            data_kind="timeseries", provenance={},
        )
        dataset.provenance["preprocessing"] = record_from_manifest_applied({
            "connectivity_method": "partial", "shrinkage": "oas",
            "standardize": False, "use_GSR": False, "tr_seconds": 2,
        })
        return dataset

    def test_reconstruction_computed_facts_override_stale_declaration(self):
        dataset = self._declared_timeseries_dataset()
        connectivity_from_dataset(dataset, kind="correlation", shrinkage=None, standardize=True)
        record = dataset.provenance["preprocessing"]
        # Actually computed here: pearson, no shrinkage, standardization on.
        assert record["connectivity_method"] == {"value": "pearson", "source": SOURCE_COMPUTED}
        assert record["shrinkage"] == {"value": None, "source": SOURCE_COMPUTED}
        assert record["standardize"] == {"value": True, "source": SOURCE_COMPUTED}
        # Independent upstream facts and their declaration sources survive.
        assert record["gsr"] == {"value": False, "source": SOURCE_MANIFEST}
        assert record["tr_seconds"] == {"value": 2.0, "source": SOURCE_MANIFEST}
        # The legacy connectivity provenance is unchanged NETSTAT-14 behaviour.
        assert dataset.provenance["connectivity"] == {
            "kind": "correlation", "shrinkage": None, "standardize": True}

    def test_manifest_timeseries_path_actual_computation_wins(self, tmp_path):
        rng = np.random.default_rng(42)
        data_dir = tmp_path / "ts"
        data_dir.mkdir()
        for sub in ("sub-01", "sub-02"):
            np.savetxt(data_dir / f"{sub}.csv", rng.normal(size=(25, 4)), delimiter=",")
        manifest = tmp_path / "study.yaml"
        manifest.write_text(
            "schema_version: 1\n"
            "name: stale-declared\n"
            "loader: TimeseriesDirectoryLoader\n"
            "paths:\n  dir_path: ts\n"
            "params:\n  pattern: '*.csv'\n"
            "preprocessing:\n"
            "  applied:\n"
            "    correlation: partial\n"
            "    shrinkage: oas\n"
            "    standardize: false\n"
            "    use_GSR: false\n"
        )
        ds = ManifestLoader(str(manifest)).load()
        assert ds.data_kind == "timeseries"          # no manifest connectivity block
        record = ds.provenance["preprocessing"]
        # Nothing was computed at load: the declarations stand as declared.
        assert record["connectivity_method"] == {"value": "partial", "source": SOURCE_MANIFEST}
        # Now the ordinary reconstruction path computes with other settings.
        connectivity_from_dataset(ds, kind="correlation", shrinkage=None, standardize=True)
        record = ds.provenance["preprocessing"]
        assert record["connectivity_method"] == {"value": "pearson", "source": SOURCE_COMPUTED}
        assert record["shrinkage"] == {"value": None, "source": SOURCE_COMPUTED}
        assert record["standardize"] == {"value": True, "source": SOURCE_COMPUTED}
        assert record["gsr"] == {"value": False, "source": SOURCE_MANIFEST}


# ---------------------------------------------------------------------------
# Owner reproducer defect 4 — Fisher policy/clip report through analysis→bundle→CLI
# ---------------------------------------------------------------------------

class TestFisherProvenanceChannel:
    def _groups(self, seed=51, saturated=True):
        rng = np.random.default_rng(seed)
        N = 5
        def sym(x):
            x = (x + x.transpose(0, 2, 1)) / 2
            x[:, np.arange(N), np.arange(N)] = 1.0
            return x
        g1 = sym(np.tanh(rng.normal(0, 0.3, (6, N, N))))
        g2 = sym(np.tanh(rng.normal(0.05, 0.3, (6, N, N))))
        if saturated:
            g1[:, 0, 1] = g1[:, 1, 0] = 1.0     # observed saturated off-diagonal
        return g1, g2

    @staticmethod
    def _manual_prepare(r):
        r = np.array(r, copy=True)
        N = r.shape[-1]
        r[:, np.arange(N), np.arange(N)] = 0.0
        with np.errstate(divide="ignore"):
            return np.clip(np.arctanh(r), -5.0, 5.0)

    def test_analysis_records_actual_clip_z_with_counts(self):
        from conninfpy._prepare import prepare_connectivity
        g1, g2 = self._groups()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = analyze(group1=g1, group2=g2, input="r", **FAST)
        record = res.fisher
        assert [(e["stage"], e["scope"], e["policy"], e["parameters"]["max_z"])
                for e in record["applied"]] == [
            ("analysis_prepare", "group1", "clip_z", 5.0),
            ("analysis_prepare", "group2", "clip_z", 5.0)]
        counts = record["applied"][0]["counts"]
        # Observed saturated edge: 2 symmetric cells × 6 subjects; 1 unique edge.
        assert counts["off_diagonal"]["n_cells_clipped"] == 12
        assert counts["off_diagonal"]["n_unique_undirected_edges_clipped"] == 1
        # The structural diagonal was zeroed before arctanh: not counted.
        assert counts["diagonal"]["n_cells_clipped"] == 0
        assert counts["n_observations"] == 6
        assert record["upstream"] is None
        assert res.inference.analysis_provenance["fisher"] == record
        # prepare_connectivity reports the same entry (helper contract).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arr, entry = prepare_connectivity(g1, "r", "group1", return_report=True)
        assert entry["policy"] == "clip_z" and entry["counts"]["n_values_clipped"] == 12

    def test_analysis_numbers_bitwise_equal_to_pretransformed_input(self):
        g1, g2 = self._groups(seed=52)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            via_r = analyze(group1=g1, group2=g2, input="r", **FAST)
        via_z = analyze(group1=self._manual_prepare(g1),
                        group2=self._manual_prepare(g2), input="z", **FAST)
        np.testing.assert_array_equal(via_r["positive"], via_z["positive"])
        np.testing.assert_array_equal(via_r["negative"], via_z["negative"])
        np.testing.assert_array_equal(via_r.inference.stat_signed, via_z.inference.stat_signed)

    def test_input_z_without_history_is_upstream_unknown(self):
        g1, g2 = self._groups(seed=53, saturated=False)
        res = analyze(group1=self._manual_prepare(g1),
                      group2=self._manual_prepare(g2), input="z", **FAST)
        assert res.fisher["applied"] == []
        assert res.fisher["upstream"] == "unknown"

    def test_upstream_clip_r_history_passthrough_without_retransform(self):
        g1, g2 = self._groups(seed=54)
        # The historical UI policy applied upstream on the correlations.
        def ui_clip(r):
            r = np.array(r, copy=True)
            N = r.shape[-1]
            r[:, np.arange(N), np.arange(N)] = 0.0
            return fisher_r_to_z(r, clip_r=0.9999, return_report=True)
        z1, rep1 = ui_clip(g1)
        entry = make_fisher_entry(stage="app_design_button", scope="dataset_in_memory", report=rep1)
        res = analyze(group1=z1, group2=self._manual_prepare(g2),
                      input="z", fisher_history=[entry], **FAST)
        assert res.fisher["applied"] == []                    # no double transform
        assert [e["policy"] for e in res.fisher["upstream"]] == ["clip_r"]
        assert res.fisher["upstream"][0]["counts"]["off_diagonal"]["n_cells_clipped"] == 12
        # Numbers are exactly the input as given (no second transform/count).
        direct = analyze(group1=z1, group2=self._manual_prepare(g2), input="z", **FAST)
        np.testing.assert_array_equal(res["positive"], direct["positive"])

    def test_paired_interest_transforms_before_difference(self):
        a, b = self._groups(seed=55)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = analyze(group1=a, group2=b, interest=np.arange(6, dtype=float),
                          test_type="paired", input="r", **FAST)
        assert [(e["stage"], e["scope"]) for e in res.fisher["applied"]] == [
            ("analysis_prepare_before_difference", "group1"),
            ("analysis_prepare_before_difference", "group2")]
        # Fisher per condition, never on the difference: bitwise oracle.
        manual_diff = self._manual_prepare(b) - self._manual_prepare(a)
        oracle = analyze(Y=manual_diff, interest=np.arange(6, dtype=float), input="z", **FAST)
        np.testing.assert_array_equal(res["positive"], oracle["positive"])
        assert res.fisher["upstream"] is None

    def test_r_to_z_false_records_no_transform(self):
        g1, g2 = self._groups(seed=56, saturated=False)
        res = analyze(group1=g1, group2=g2, input="r", r_to_z=False, **FAST)
        assert res.fisher["applied"] == [] and res.fisher["upstream"] is None
        assert res.inference.design["scale"] == "correlation"

    def test_bundle_roundtrip_legacy_and_conflicts(self, tmp_path):
        g1, g2 = self._groups(seed=57)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = analyze(group1=g1, group2=g2, input="r", **FAST)
        path = tmp_path / "b.pkl"
        res.to_bundle().save(path)
        reloaded = ResultBundle.load(path)
        assert [e["policy"] for e in reloaded.metadata["fisher"]["applied"]] == ["clip_z", "clip_z"]
        assert reloaded.metadata["fisher"]["applied"][0]["counts"]["off_diagonal"]["n_cells_clipped"] == 12
        with pytest.raises(ValueError):
            res.to_bundle(fisher={"schema_version": 1, "applied": [], "upstream": None})
        # Pre-channel legacy bundle (no key) still loads without invention.
        legacy = analyze(group1=self._manual_prepare(g1),
                         group2=self._manual_prepare(g2), input="z", **FAST)
        legacy.inference.analysis_provenance.pop("fisher", None)
        legacy_path = tmp_path / "legacy.pkl"
        legacy.to_bundle().save(legacy_path)
        legacy_bundle = ResultBundle.load(legacy_path)
        assert legacy_bundle.metadata.get("fisher") is None   # nothing invented

    def test_cli_provenance_json_and_bundle_carry_fisher(self, tmp_path):
        from conninfpy.cli import run
        rng = np.random.default_rng(58)
        n, N = 12, 5
        r = np.tanh(rng.normal(0, 0.3, (n, N, N)))
        r = (r + r.transpose(0, 2, 1)) / 2
        r[:, 0, 2] = r[:, 2, 0] = 1.0          # observed saturated edge
        r[:, np.arange(N), np.arange(N)] = 1.0
        np.save(tmp_path / "conn.npy", r)
        pd.DataFrame({"subject_id": [f"s{i:02d}" for i in range(n)],
                      "group": np.tile(["hc", "pt"], n // 2)}).to_csv(tmp_path / "pheno.csv", index=False)
        (tmp_path / "study.yaml").write_text(
            "schema_version: 1\nloader: NumpyLoader\n"
            "paths:\n  data_path: conn.npy\n  pheno_path: pheno.csv\n"
            "params:\n  data_kind: correlation\n"
        )
        (tmp_path / "analysis.yaml").write_text(json.dumps({
            "data": "study.yaml", "output": "out",
            "design": {"type": "two-sample", "group": "group", "levels": ["hc", "pt"]},
            "inference": dict(FAST),
        }) + "\n")
        written = run(tmp_path / "analysis.yaml")
        prov = json.loads((tmp_path / "out" / "provenance.json").read_text())
        fisher = prov["results"]["result"]["fisher"]
        applied = fisher["applied"]
        assert [(e["scope"], e["policy"]) for e in applied] == [
            ("group1", "clip_z"), ("group2", "clip_z")]
        assert applied[0]["counts"]["off_diagonal"]["n_cells_clipped"] == 2 * (n // 2)
        assert applied[0]["counts"]["off_diagonal"]["n_unique_undirected_edges_clipped"] == 1
        assert applied[0]["counts"]["diagonal"]["n_cells_clipped"] == 0
        bundle = ResultBundle.load(written["bundle"])
        assert bundle.metadata["fisher"]["applied"][0]["policy"] == "clip_z"


# ---------------------------------------------------------------------------
# Owner reproducer defect 5 — multisite aggregation of per-site facts
# ---------------------------------------------------------------------------

def _site_config(tmp_path: Path, site: str, n_sub: int, token: str) -> dict:
    d = tmp_path / f"{site}_{token}"
    d.mkdir(exist_ok=True)
    rng = np.random.default_rng(abs(hash((site, token))) % 2**32)
    for cond in ("open", "close"):
        np.save(d / f"{site}_{cond}_Schaefer_strategy-4_{token}.npy",
                rng.normal(size=(n_sub, 15, 4)))
    (d / "order.txt").write_text("\n".join(f"{site}s{i}" for i in range(n_sub)))
    return {"open_path": str(d / f"{site}_open_Schaefer_strategy-4_{token}.npy"),
            "close_path": str(d / f"{site}_close_Schaefer_strategy-4_{token}.npy"),
            "subject_list_path": str(d / "order.txt")}


class TestMultisitePreprocessingAggregation:
    def test_heterogeneous_gsr_stays_unknown_globally_with_per_site_facts(self, tmp_path):
        from apps.utils.lab_loaders import MultiSiteOpenCloseLoader
        cfg = {"sitea": _site_config(tmp_path, "sitea", 2, "GSR"),
               "siteb": _site_config(tmp_path, "siteb", 2, "no_GSR")}
        ds = MultiSiteOpenCloseLoader(cfg).load()
        assert ds.data.shape[0] == 8                       # 2 sites × 2 subjects × 2 conditions
        record = ds.provenance["preprocessing"]
        assert "gsr" not in record                        # conflicting → unknown
        per_site = ds.provenance["preprocessing_per_site"]
        assert per_site["sitea"]["gsr"] == {"value": True, "source": SOURCE_FILE_TOKEN}
        assert per_site["siteb"]["gsr"] == {"value": False, "source": SOURCE_FILE_TOKEN}
        # Shared computed settings survive the aggregation.
        assert record["connectivity_method"] == {"value": "pearson", "source": SOURCE_COMPUTED}
        assert record["standardize"] == {"value": True, "source": SOURCE_COMPUTED}

    def test_homogeneous_sites_keep_known_gsr(self, tmp_path):
        from apps.utils.lab_loaders import MultiSiteOpenCloseLoader
        gsr = MultiSiteOpenCloseLoader({"a": _site_config(tmp_path, "ha", 2, "GSR"),
                                        "b": _site_config(tmp_path, "hb", 2, "GSR")}).load()
        assert gsr.provenance["preprocessing"]["gsr"] == {
            "value": True, "source": SOURCE_FILE_TOKEN}
        nogsr = MultiSiteOpenCloseLoader({"a": _site_config(tmp_path, "na", 2, "noGSR"),
                                          "b": _site_config(tmp_path, "nb", 2, "no-GSR")}).load()
        assert nogsr.provenance["preprocessing"]["gsr"] == {
            "value": False, "source": SOURCE_FILE_TOKEN}

    def test_known_plus_unknown_is_globally_unknown(self, tmp_path):
        from apps.utils.lab_loaders import MultiSiteOpenCloseLoader
        from conninfpy.preprocessing import aggregate_preprocessing_records
        # Site files with and without a GSR token: known+unknown → unknown.
        ds = MultiSiteOpenCloseLoader({"ka": _site_config(tmp_path, "ka", 2, "GSR"),
                                       "kb": _site_config(tmp_path, "kb", 2, "bold")}).load()
        assert "gsr" not in ds.provenance["preprocessing"]
        assert ds.provenance["preprocessing_per_site"]["ka"]["gsr"]["value"] is True
        assert "gsr" not in ds.provenance["preprocessing_per_site"]["kb"]
        # Unit-level: the aggregator never invents a common value.
        assert "gsr" not in aggregate_preprocessing_records([
            {"gsr": fact(True, SOURCE_FILE_TOKEN)}, {}])
        assert aggregate_preprocessing_records([]) == empty_record()

    def test_aggregate_mixed_sources_same_value(self):
        merged = aggregate_preprocessing_records([
            {"gsr": fact(False, SOURCE_MANIFEST)}, {"gsr": fact(False, SOURCE_FILE_TOKEN)}])
        assert merged["gsr"]["value"] is False
        assert merged["gsr"]["source"].startswith("mixed_sources(")


def _multisite_manifest(tmp_path: Path, sites: dict) -> Path:
    """A real manifest driving MultiSiteOpenCloseLoader via paths.site_configs."""
    import yaml
    manifest = tmp_path / f"ms_{'_'.join(sites.values())}.yaml"
    manifest.write_text(yaml.safe_dump({
        "schema_version": 1, "name": "mixed", "loader": "MultiSiteOpenCloseLoader",
        "paths": {"site_configs": {name: _site_config(tmp_path, name, 2, token)
                                    for name, token in sites.items()}},
    }, default_flow_style=False))
    return manifest


class TestPerSiteFactsThroughTheOrdinaryChain:
    """Per-site records must survive loader → build_analysis/analyze → bundle/CLI."""

    def test_mixed_sites_end_to_end_cli_and_bundle(self, tmp_path):
        import yaml
        from conninfpy.cli import build_analysis, run
        manifest = _multisite_manifest(tmp_path, {"a": "GSR", "b": "no_GSR"})
        ds = load(str(manifest))
        record = ds.provenance["preprocessing"]
        assert "gsr" not in record                      # disagreement → unknown global
        assert set(record["per_site"]) == {"a", "b"}
        cfg = {"data": manifest.name,
               "design": {"type": "paired", "condition": "condition",
                          "levels": ["open", "close"]}}
        kwargs, summary = build_analysis(ds, cfg)
        assert "per_site" in kwargs["preprocessing"]     # actual CLI kwargs channel
        assert "per_site" in summary["preprocessing"]
        res = analyze(**{k: v for k, v in kwargs.items() if k != "input"},
                      input=kwargs["input"], **FAST)
        assert "gsr" not in res.preprocessing
        assert res.preprocessing["per_site"]["a"]["gsr"] == {
            "value": True, "source": SOURCE_FILE_TOKEN}
        assert res.preprocessing["per_site"]["b"]["gsr"] == {
            "value": False, "source": SOURCE_FILE_TOKEN}
        path = tmp_path / "b.pkl"
        res.to_bundle().save(path)
        bundle = ResultBundle.load(path)
        assert bundle.metadata["preprocessing"]["per_site"]["b"]["gsr"]["value"] is False
        # The full CLI run writes both channels.
        (tmp_path / "analysis.yaml").write_text(yaml.safe_dump({
            "data": manifest.name, "output": "out", "design": cfg["design"],
            "inference": dict(FAST)}))
        written = run(tmp_path / "analysis.yaml")
        prov = json.loads((tmp_path / "out" / "provenance.json").read_text())
        assert set(prov["data"]["preprocessing"]["per_site"]) == {"a", "b"}
        cli_bundle = ResultBundle.load(written["bundle"])
        assert cli_bundle.metadata["preprocessing"]["per_site"]["a"]["gsr"]["value"] is True

    def test_homogeneous_false_and_known_plus_unknown(self, tmp_path):
        from apps.utils.lab_loaders import MultiSiteOpenCloseLoader
        both = MultiSiteOpenCloseLoader({
            "a": _site_config(tmp_path, "fa", 2, "noGSR"),
            "b": _site_config(tmp_path, "fb", 2, "no-GSR")}).load()
        record = both.provenance["preprocessing"]
        assert record["gsr"] == {"value": False, "source": SOURCE_FILE_TOKEN}
        assert all(r["gsr"]["value"] is False for r in record["per_site"].values())
        res = analyze(group1=_two_groups(seed=71)[0], group2=_two_groups(seed=72)[1],
                      input="r", preprocessing=record, **FAST)
        assert res.preprocessing["gsr"]["value"] is False
        assert set(res.preprocessing["per_site"]) == {"a", "b"}
        mixed = MultiSiteOpenCloseLoader({
            "a": _site_config(tmp_path, "ua", 2, "GSR"),
            "b": _site_config(tmp_path, "ub", 2, "bold")}).load()
        record = mixed.provenance["preprocessing"]
        assert "gsr" not in record                     # known + unknown → unknown
        assert record["per_site"]["a"]["gsr"]["value"] is True
        assert "gsr" not in record["per_site"]["b"]

    def test_record_per_site_validation_and_legacy_compat(self):
        good = normalize_preprocessing_record({
            "per_site": {"a": {"gsr": fact(True, SOURCE_FILE_TOKEN)}, "b": None}})
        assert good["per_site"]["a"]["gsr"]["value"] is True
        assert good["per_site"]["b"] == empty_record()
        with pytest.raises(ValueError):
            normalize_preprocessing_record({"per_site": ["a"]})
        with pytest.raises(ValueError):
            normalize_preprocessing_record({"per_site": {"": {}}})
        with pytest.raises(ValueError):
            normalize_preprocessing_record({"per_site": {"a": {"per_site": {}}}})  # no nesting
        # Legacy single-record paths unchanged; merge keeps the first per_site.
        legacy = normalize_preprocessing_record({"gsr": fact(False, SOURCE_MANIFEST)})
        assert "per_site" not in legacy
        merged = merge_preprocessing_records(
            {"gsr": fact(True, SOURCE_FILE_TOKEN), "per_site": {"a": {"gsr": fact(True, SOURCE_FILE_TOKEN)}}},
            {"gsr": fact(False, SOURCE_MANIFEST), "per_site": {"z": {}}})
        assert merged["gsr"]["value"] is True and set(merged["per_site"]) == {"a"}

    def test_gui_shows_per_site_context_without_common_claim(self):
        from apps.utils.helpers import describe_per_site_facts
        record = normalize_preprocessing_record({
            "per_site": {"a": {"gsr": fact(True, SOURCE_FILE_TOKEN)},
                         "b": {"gsr": fact(False, SOURCE_FILE_TOKEN)}}})
        rows = describe_per_site_facts(record)
        assert rows == [("a", "yes", "declared by the input file name"),
                        ("b", "no", "declared by the input file name")]
        assert describe_per_site_facts(None) == []


# ---------------------------------------------------------------------------
# Owner round-2 defect 7 — supplied Fisher histories cannot change origin/policy
# ---------------------------------------------------------------------------

class TestIncomingFisherHistoryGuards:
    @staticmethod
    def _report(policy="clip_r", **params):
        return {"policy": policy, "parameters": params or {"clip_r": 0.9999},
                "n_values_clipped": 0, "matrix_shape": None, "n_observations": None,
                "diagonal": None, "off_diagonal": None, "left_at_infinity": None,
                "count_units": "cells"}

    def test_missing_source_is_rejected_not_defaulted(self):
        entry = make_fisher_entry(stage="upstream", scope="dataset", report=self._report())
        entry.pop("source")
        with pytest.raises(ValueError, match="explicit nonempty source"):
            normalize_fisher_history([entry])
        with pytest.raises(ValueError, match="explicit nonempty source"):
            normalize_fisher_history([{**entry, "source": "   "}])

    def test_contradictory_policy_or_parameters_are_rejected(self):
        entry = make_fisher_entry(stage="upstream", scope="dataset", report=self._report())
        with pytest.raises(ValueError, match="policy contradicts"):
            normalize_fisher_history([{**entry, "policy": "clip_z"}])
        with pytest.raises(ValueError, match="parameters contradict"):
            normalize_fisher_history([{**entry, "parameters": {"clip_r": 0.5}}])
        # Agreement passes untouched.
        ok = normalize_fisher_history([entry])
        assert ok[0]["policy"] == "clip_r" and ok[0]["source"] == "conninfpy.fisher_r_to_z"

    def test_full_record_dict_cannot_bypass_schema_checks(self):
        entry = make_fisher_entry(stage="upstream", scope="dataset", report=self._report())
        with pytest.raises(ValueError, match="schema_version"):
            normalize_fisher_history({"schema_version": 2, "applied": [entry]})
        via_record = normalize_fisher_history(
            {"schema_version": 1, "applied": [entry], "upstream": "unknown"})
        assert [e["policy"] for e in via_record] == ["clip_r"]

    def test_entries_are_isolated_from_later_mutation(self):
        report = self._report()
        entry = make_fisher_entry(stage="s", scope="d", report=report)
        report["n_values_clipped"] = 999
        report["parameters"]["clip_r"] = 0.1
        assert entry["counts"]["n_values_clipped"] == 0
        assert entry["parameters"] == {"clip_r": 0.9999}

    def test_bundle_override_cannot_silently_change_origin_or_policy(self):
        g1, g2 = _two_groups(seed=71)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = analyze(group1=g1, group2=g2, input="r", **FAST)
        captured = res.fisher
        forged = {"schema_version": 1,
                  "applied": [{**captured["applied"][0], "policy": "clip_r"}],
                  "upstream": None}
        with pytest.raises(ValueError, match="contradicts"):
            res.to_bundle(fisher=forged)
        # Valid legacy import keeps working for results without a record.
        legacy = analyze(group1=g1, group2=g2, input="r", **FAST)
        legacy.inference.analysis_provenance.pop("fisher", None)
        entry = make_fisher_entry(stage="legacy_import", scope="Y", report=self._report("clip_z", max_z=5.0),
                                  source="legacy_statement")
        bundle = legacy.to_bundle(fisher={"schema_version": 1, "applied": [entry], "upstream": None})
        assert bundle.metadata["fisher"]["applied"][0]["source"] == "legacy_statement"




# ---------------------------------------------------------------------------
# AC1/AC2 — app paths (real Streamlit AppTest + worker)
# ---------------------------------------------------------------------------

class TestAppPaths:
    def test_apply_fisher_z_button_uses_shared_clip_r_policy(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(REPO / "apps" / "streamlit_app.py"), default_timeout=180)
        rng = np.random.default_rng(17)
        r = np.tanh(rng.normal(0, 0.3, (6, 5, 5)))
        r = (r + r.transpose(0, 2, 1)) / 2
        r[:, 0, 1] = r[:, 1, 0] = 0.99995          # saturating observed edge
        idx = np.arange(5)
        r[:, idx, idx] = 1.0                         # structural diagonal
        app.session_state.connectivity_data = r
        app.session_state.connectivity_data_kind = "correlation"
        app.session_state.pheno_df = pd.DataFrame(
            {"subject_id": [f"s{i}" for i in range(6)],
             "group": [0, 0, 0, 1, 1, 1]})
        app.run()
        app.session_state.active_tab = "2. Design & Inference"
        app.run()
        button = app.button(key="apply_fisher_z_btn")
        assert not app.exception
        button.click().run()
        assert not app.exception
        expected = np.arctanh(np.clip(r, -0.9999, 0.9999))
        np.testing.assert_allclose(app.session_state.connectivity_data, expected)
        assert app.session_state.connectivity_data_kind == "fisher_z"
        notes = [w.value for w in app.warning] + [s.value for s in app.success]
        assert any("off-diagonal matrix cell(s)" in n and "unique undirected" in n for n in notes)
        # The structured upstream entry (policy, scope, counts) is kept for
        # analyze() to carry into the result without re-transforming.
        history = app.session_state.get("dataset_fisher_history")
        assert history and len(history) == 1
        entry = history[0]
        assert (entry["stage"], entry["scope"], entry["policy"]) == (
            "app_design_button", "dataset_in_memory", "clip_r")
        assert entry["parameters"] == {"clip_r": 0.9999}
        assert entry["counts"]["off_diagonal"]["n_cells_clipped"] == 12

    def test_apply_fisher_z_rejects_invalid_input_without_touching_state(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(REPO / "apps" / "streamlit_app.py"), default_timeout=180)
        rng = np.random.default_rng(18)
        r = np.tanh(rng.normal(0, 0.3, (6, 5, 5)))
        r = (r + r.transpose(0, 2, 1)) / 2
        r[:, 0, 1] = r[:, 1, 0] = 1.4          # invalid correlation
        idx = np.arange(5)
        r[:, idx, idx] = 1.0
        app.session_state.connectivity_data = r
        app.session_state.connectivity_data_kind = "correlation"
        app.session_state.pheno_df = pd.DataFrame(
            {"subject_id": [f"s{i}" for i in range(6)], "group": [0] * 3 + [1] * 3})
        app.run()
        app.session_state.active_tab = "2. Design & Inference"
        app.run()
        app.button(key="apply_fisher_z_btn").click().run()
        assert not app.exception
        errors = [e.value for e in app.error]
        assert any("not applied" in e and "[-1, 1]" in e for e in errors)
        # Dataset, kind and metadata are untouched by the rejection.
        np.testing.assert_array_equal(app.session_state.connectivity_data, r)
        assert app.session_state.connectivity_data_kind == "correlation"
        assert app.session_state.get("dataset_fisher_history") is None

    def test_dataset_card_lists_known_and_unknown_facts(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(REPO / "apps" / "streamlit_app.py"), default_timeout=180).run()
        next(rd for rd in app.radio if rd.label == "Select Ingestion Method").set_value("Built-in data").run()
        next(b for b in app.button if b.label == "Load, Preprocess & Validate Data").click().run()
        assert not app.exception
        record = app.session_state.get("dataset_preprocessing")
        assert record is not None
        # ABIDE-mini: estimator computed by the library; GSR unknown.
        assert record["connectivity_method"]["value"] == "pearson"
        assert "gsr" not in record
        texts = " ".join(m.value for m in app.markdown) + " ".join(c.value for c in app.caption)
        assert "Global-signal regression (GSR)" in texts
        assert "Connectivity estimator" in texts
        # Loading a new dataset resets any older transform history; the
        # ingestion option's own entry (default: Fisher on) is the only one.
        history = app.session_state.get("dataset_fisher_history")
        assert history and len(history) == 1
        assert history[0]["stage"] == "app_ingestion_option"
        assert history[0]["policy"] == "clip_z"

    def test_worker_threads_record_into_run_plan_and_result(self):
        from apps.components.design_inference import InferenceTask, _run_inference_background
        g1, g2 = _two_groups(seed=21)
        pheno = pd.DataFrame({"subject_id": [f"s{i}" for i in range(12)],
                              "group": [0] * 6 + [1] * 6})
        data = np.concatenate([g1, g2])
        idx = np.arange(5)
        data[:, idx, idx] = 0.0
        # Pre-transform upstream with the historical UI policy and hand the
        # worker the captured entry, as the design-step button does.
        def ui(r):
            r = np.array(r, copy=True)
            r[:, idx, idx] = 0.0
            return fisher_r_to_z(r, clip_r=0.9999, return_report=True)
        z_data, ui_report = ui(data)
        ui_entry = make_fisher_entry(stage="app_design_button", scope="dataset_in_memory",
                                     report=ui_report)
        task = InferenceTask()
        record = {"gsr": fact(False, SOURCE_MANIFEST),
                  "connectivity_method": fact("pearson", SOURCE_COMPUTED)}
        _run_inference_background(
            task,
            Y=None, interest=None, confounds=None,
            group1=z_data[:6], group2=z_data[6:], test_type="two-sample", sites=None,
            harmonization_choice="auto", operator_choice="tstat", n_perms=20, seed_val=1,
            use_mp=False, acceleration_choice="none", op_kwargs={},
            run_sensitivity=False, comp_method=None, comp_kwargs={},
            sc_net_labels=None, use_sc_prior=False, atlas=None, annotation_atlas=None,
            alpha=0.05, question_choice="Group Difference", loaded_settings_hash="h",
            recipe_choice="auto", effective_recipe_choice="auto",
            group_col="group", ref_group="0", target_group="1", interest_var=None,
            subject_col="subject_id", condition_col=None, baseline_val=None, target_val=None,
            confound_vars=[], site_col=None, active_atlas_signature=None,
            data_kind="fisher_z", preprocessing=record, fisher_history=[ui_entry],
        )
        assert task.status == "success", task.error
        assert task.run_plan["preprocessing"] == record
        assert task.result.preprocessing["gsr"]["value"] is False
        # The captured upstream entry reaches the result and run plan without
        # a second transform (nothing applied here).
        assert task.run_plan["fisher_history"] == [ui_entry]
        assert task.result.fisher["applied"] == []
        assert task.result.fisher["upstream"] == [ui_entry]
        assert task.run_plan["fisher"]["upstream"] == [ui_entry]
