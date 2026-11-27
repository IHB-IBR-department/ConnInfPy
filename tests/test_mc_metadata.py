"""Monte-Carlo precision metadata: exceedance counts, Wilson intervals,
permutation space and RNG provenance (NETSTAT-32).

Synthetic only. The baseline test pins p-values bitwise against arrays
computed with the unmodified code before the metadata was added; the
metadata must never change an inference result.
"""
import json

import numpy as np
import pytest

from conninfpy import ResultBundle, compute_p_val
from conninfpy.stats._engine import _compute_p_values_from_null
from conninfpy.stats.acceleration import fit_gpd_tail
from tests.test_workspace_persistence import (
    _sync_workspace_checksums,
    _workspace_zip_files,
    _zip_from_files,
)

from conninfpy.stats._mc import (
    COUNT_KIND_EDGE_WISE,
    COUNT_KIND_JOINT_MAX,
    COUNT_KIND_MAX_STAT,
    COUNT_KIND_TWO_SIDED_SUM_STAT,
    COUNT_UNAVAILABLE,
    mark_untested,
    permutation_space,
    wilson_interval,
)

B_TWO_SAMPLE = 50


def _groups(n1=6, n2=7, n=4, seed=0):
    rng = np.random.default_rng(seed)
    g1 = rng.normal(0, 1, (n1, n, n))
    g2 = rng.normal(0.3, 1, (n2, n, n))
    for group in (g1, g2):
        for matrix in group:
            np.fill_diagonal(matrix, 0.0)
        group[:] = (group + group.transpose(0, 2, 1)) / 2
    return g1, g2


def _run(g1, g2, **kwargs):
    defaults = dict(
        n_permutations=B_TWO_SAMPLE,
        test_type="two-sample",
        method="tstat",
        use_mp=False,
        rng=42,
    )
    defaults.update(kwargs)
    return compute_p_val(g1, g2, **defaults)


# ---------------------------------------------------------------------------
# 1. p-values and counts are pinned (bitwise p unchanged by this change)
# ---------------------------------------------------------------------------

TWO_P = [[1.0, 0.6470588235294118, 1.0, 0.7647058823529411],
         [0.6470588235294118, 1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0, 1.0],
         [0.7647058823529411, 1.0, 1.0, 1.0]]
TWO_N = [[1.0] * 4 for _ in range(4)]
ONE_P = [[1.0, 0.39215686274509803, 1.0, 0.49019607843137253],
         [0.39215686274509803, 1.0, 0.9019607843137255, 1.0],
         [1.0, 0.9019607843137255, 1.0, 0.9607843137254902],
         [0.49019607843137253, 1.0, 0.9607843137254902, 1.0]]
BH_P = [[1.0, 0.47058823529411764, 1.0, 0.9117647058823528],
        [0.47058823529411764, 1.0, 0.9117647058823528, 1.0],
        [1.0, 0.9117647058823528, 1.0, 0.9215686274509803],
        [0.9117647058823528, 1.0, 0.9215686274509803, 1.0]]
PAIRED_P = [[1.0, 0.45161290322580644, 1.0, 0.6774193548387096],
            [0.45161290322580644, 1.0, 0.8064516129032258, 1.0],
            [1.0, 0.8064516129032258, 1.0, 1.0],
            [0.6774193548387096, 1.0, 1.0, 1.0]]

TWO_COUNTS_POS = [[-1, 32, 50, 38], [32, -1, 50, 50], [50, 50, -1, 50], [38, 50, 50, -1]]
TWO_COUNTS_NEG = [[-1, 50, 50, 50], [50, -1, 50, 50], [50, 50, -1, 50], [50, 50, 50, -1]]
ONE_COUNTS_POS = [[-1, 19, 50, 24], [19, -1, 45, 50], [50, 45, -1, 48], [24, 50, 48, -1]]
BH_COUNTS_POS = [[-1, 3, -1, 15], [3, -1, 30, -1], [-1, 30, -1, 46], [15, -1, 46, -1]]
PAIRED_COUNTS_POS = [[-1, 13, 30, 20], [13, -1, 24, 30], [30, 24, -1, 30], [20, 30, 30, -1]]


def test_two_sided_p_values_and_counts_are_pinned():
    g1, g2 = _groups()
    result = _run(g1, g2, two_tailed=True)
    # Golden p-values were computed with the unmodified code (base 6f823f4).
    assert np.array_equal(result["positive"], np.array(TWO_P))
    assert np.array_equal(result["negative"], np.array(TWO_N))
    assert np.array_equal(
        result.exceedance_counts["positive"], np.array(TWO_COUNTS_POS)
    )
    assert np.array_equal(
        result.exceedance_counts["negative"], np.array(TWO_COUNTS_NEG)
    )
    # Every counted cell is exactly the +1 estimate of its count; untested
    # (diagonal) cells keep the sentinel and were never "not significant".
    counts = result.exceedance_counts["positive"]
    counted = counts >= 0
    expected = (counts[counted] + 1.0) / (B_TWO_SAMPLE + 1.0)
    assert np.array_equal(result["positive"][counted], expected)
    assert np.all(np.diag(result["positive"]) == 1.0)
    assert np.all(np.diag(counts) == COUNT_UNAVAILABLE)


def test_one_sided_bh_and_paired_p_values_and_counts_are_pinned():
    g1, g2 = _groups()
    one = _run(g1, g2, two_tailed=False)
    assert np.array_equal(one["positive"], np.array(ONE_P))
    assert np.array_equal(one.exceedance_counts["positive"], np.array(ONE_COUNTS_POS))

    bh = _run(g1, g2, method="bh_fdr_perm", two_tailed=True)
    assert np.array_equal(bh["positive"], np.array(BH_P))
    assert np.array_equal(bh.exceedance_counts["positive"], np.array(BH_COUNTS_POS))

    paired = compute_p_val(
        g1[:6], g2[:6], n_permutations=30, test_type="paired",
        method="tstat", use_mp=False, rng=7, two_tailed=False,
    )
    assert np.array_equal(paired["positive"], np.array(PAIRED_P))
    assert np.array_equal(
        paired.exceedance_counts["positive"], np.array(PAIRED_COUNTS_POS)
    )


# ---------------------------------------------------------------------------
# 2. Known counts on tiny synthetic data
# ---------------------------------------------------------------------------


def test_known_counts_and_plus_one_estimate_on_tiny_data():
    observed = np.array([[0.0, 2.0], [2.0, 0.0]])
    null = np.array([0.5, 1.5, 2.0, 3.0])  # ties at 2.0 count as exceedances
    expected_counts = np.array([[4, 2], [2, 4]])
    p, counts = _compute_p_values_from_null(
        {"positive": observed}, {"positive": null}, return_counts=True
    )
    assert np.array_equal(counts["positive"], expected_counts)
    assert np.array_equal(p["positive"], (expected_counts + 1.0) / 5.0)
    # Default contract unchanged: p-values only.
    assert isinstance(
        _compute_p_values_from_null({"positive": observed}, {"positive": null}), dict
    )


# ---------------------------------------------------------------------------
# 3. Wilson interval against the hand-computed formula
# ---------------------------------------------------------------------------


def test_wilson_interval_matches_hand_computation():
    counts = np.array([0, 1, 5, 17, 40])
    n_draws = 40
    alpha = 0.05
    z = 1.959963984540054  # standard normal 0.975 quantile
    p_hat = counts / n_draws
    denom = 1 + z * z / n_draws
    center = (p_hat + z * z / (2 * n_draws)) / denom
    half = (z / denom) * np.sqrt(
        p_hat * (1 - p_hat) / n_draws + z * z / (4 * n_draws * n_draws)
    )
    low, high = wilson_interval(counts, n_draws, alpha)
    assert np.allclose(low, np.clip(center - half, 0.0, 1.0), rtol=0, atol=1e-15)
    assert np.allclose(high, np.clip(center + half, 0.0, 1.0), rtol=0, atol=1e-15)
    # The interval is for count/B, and it brackets the +1 estimate only loosely:
    # p = (count+1)/(B+1) is conservative, not the interval centre.
    assert np.all(low <= counts / n_draws + 1e-15)
    assert np.all(high >= counts / n_draws - 1e-15)


# ---------------------------------------------------------------------------
# 4. Count kinds for the two-sided / BH paths
# ---------------------------------------------------------------------------


def test_count_kind_labels_for_two_sided_and_bh_paths():
    g1, g2 = _groups()
    assert _run(g1, g2, two_tailed=True).count_kind == COUNT_KIND_JOINT_MAX
    assert _run(g1, g2, two_tailed=False).count_kind == COUNT_KIND_MAX_STAT
    assert (
        _run(g1, g2, method="bh_fdr_perm", two_tailed=True).count_kind
        == COUNT_KIND_TWO_SIDED_SUM_STAT
    )
    assert (
        _run(g1, g2, method="bh_fdr_perm", two_tailed=False).count_kind
        == COUNT_KIND_EDGE_WISE
    )
    # Two-sided: both tails count against the same joint null, so the intervals
    # are not independent (each tail counts against its own observed statistic).
    joint = _run(g1, g2, two_tailed=True)
    assert joint.count_kind == COUNT_KIND_JOINT_MAX
    report = joint.mc_report()
    assert report["count_kind"] == COUNT_KIND_JOINT_MAX
    assert any("same joint null" in note for note in report["notes"])
    # BH: the interval is declared to belong to the per-edge count, not to q.
    bh = _run(g1, g2, method="bh_fdr_perm", two_tailed=False)
    assert any("BH-adjusted q" in note for note in bh.mc_report()["notes"])


# ---------------------------------------------------------------------------
# 5. Accelerated results never get a Wilson interval
# ---------------------------------------------------------------------------


def test_accelerated_result_has_no_interval_with_the_actual_reason():
    g1, g2 = _groups()
    result = _run(g1, g2, acceleration="gpd")
    report = result.mc_report()
    for tail in ("positive", "negative"):
        entry = report["tails"][tail]
        assert entry["interval_status"] == "unknown"
        assert "acceleration=gpd" in entry["interval_reason"]
        assert entry["interval_low"] is None and entry["n_cells_interval"] == 0
        # The producer's empirical counts are still described (with the
        # replaced cells marked as having no usable count).
        assert entry["counts_available"] is True
        assert np.any(entry["count"] == COUNT_UNAVAILABLE)


def test_acceleration_diagnostics_are_optional_and_do_not_change_p():
    null = np.linspace(0.0, 10.0, 100)
    observed = np.array([0.5, 9.5, 12.0])
    plain = fit_gpd_tail(null, observed)
    p, counts, replaced = fit_gpd_tail(null, observed, return_diagnostics=True)
    assert np.array_equal(plain, p)
    assert counts.shape == observed.shape
    assert replaced.dtype == bool
    assert np.sum(replaced) <= counts.size


# ---------------------------------------------------------------------------
# 6. Legacy results and untested cells are never invented
# ---------------------------------------------------------------------------


def test_legacy_result_without_metadata_reports_unavailable():
    g1, g2 = _groups()
    result = _run(g1, g2, two_tailed=True)
    del result.exceedance_counts
    del result.count_kind
    del result.rng_provenance
    del result.permutation_space
    report = result.mc_report()
    assert report["permutation_space"] is None
    assert report["rng"] is None
    for entry in report["tails"].values():
        assert entry["counts_available"] is False
        assert entry["interval_status"] == "unknown"
        assert "not recorded" in entry["interval_reason"]


def test_untested_cells_keep_the_sentinel():
    g1, g2 = _groups()
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 1] = mask[1, 0] = True
    result = _run(g1, g2, edge_mask=mask, n_permutations=20)
    counts = result.exceedance_counts["positive"]
    assert counts[0, 1] >= 0 and counts[1, 0] >= 0
    assert np.all(counts[~mask] == COUNT_UNAVAILABLE)
    report = result.mc_report()
    entry = report["tails"]["positive"]
    assert entry["n_cells_counted"] == 2
    assert entry["n_cells_interval"] == 2
    sentinel = np.full((2, 2), 5, dtype=np.int64)
    marked = mark_untested(sentinel, np.array([[1.0, np.nan], [1.0, 1.0]]))
    assert marked[0, 1] == COUNT_UNAVAILABLE
    assert np.all(np.diag(marked) == COUNT_UNAVAILABLE)


# ---------------------------------------------------------------------------
# 7. Round-trip: pickle, bundle, CLI
# ---------------------------------------------------------------------------


def test_pickle_and_bundle_round_trip_preserve_counts_and_rng():
    import pickle

    g1, g2 = _groups()
    result = _run(g1, g2, two_tailed=True)
    clone = pickle.loads(pickle.dumps(result))
    assert np.array_equal(
        clone.exceedance_counts["positive"], result.exceedance_counts["positive"]
    )
    assert clone.count_kind == result.count_kind
    assert clone.rng_provenance == result.rng_provenance
    assert clone.permutation_space == result.permutation_space

    bundle = result.to_bundle()
    mc = bundle.metadata["mc"]
    assert mc["count_kind"] == COUNT_KIND_JOINT_MAX
    assert mc["rng"] == result.rng_provenance
    assert np.array_equal(
        mc["exceedance_counts"]["positive"], result.exceedance_counts["positive"]
    )
    assert mc["permutation_space"] == result.permutation_space
    assert mc["resolution"] == 1.0 / (B_TWO_SAMPLE + 1.0)


def test_bundle_round_trip_through_file_and_legacy_bundle_accepted(tmp_path):
    g1, g2 = _groups()
    result = _run(g1, g2, two_tailed=True)
    path = tmp_path / "bundle.pkl"
    result.to_bundle().save(path)
    loaded = ResultBundle.load(path)
    assert loaded.metadata["mc"]["rng"]["seed"] == 42
    # A bundle written before this metadata existed must still validate.
    legacy = ResultBundle.load(path)
    del legacy.metadata["mc"]
    legacy._validate()


def test_cli_provenance_contains_monte_carlo_keys(tmp_path):
    from tests.test_cli import FAST, _Study, _correlations, _pheno

    study = _Study(tmp_path, _pheno(24), _correlations(24))
    config = study.analysis({"type": "glm", "interest": "age", "confounds": ["sex"]})
    from conninfpy.cli import run

    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        written = run(config)
    prov = json.loads(written["provenance"].read_text())
    mc = prov["results"]["result"]["monte_carlo"]
    assert mc["n_permutations"] == FAST["n_permutations"]
    assert mc["resolution"] == 1.0 / (FAST["n_permutations"] + 1.0)
    assert mc["rng"]["seed"] == FAST["rng"]
    assert mc["rng"]["scheme"] == "legacy-randomstate-randint-uint32"
    assert mc["rng"]["bit_generator"] == "MT19937"
    # Freedman-Lane residuals: the assignment space is honestly unknown.
    assert mc["permutation_space"] is None
    assert mc["tails"]["positive"]["counts_available"] is True
    assert mc["tails"]["positive"]["interval_status"] == "available"


# ---------------------------------------------------------------------------
# 8. Serial and multiprocessing runs agree bitwise (p AND counts)
# ---------------------------------------------------------------------------


def test_serial_and_multiprocessing_share_p_values_and_counts():
    g1, g2 = _groups()
    serial = _run(g1, g2, n_permutations=15, two_tailed=True)
    parallel = _run(g1, g2, n_permutations=15, two_tailed=True,
                    use_mp=True, n_processes=2)
    for tail in ("positive", "negative"):
        # NaN marks untested available-case edges; equal_nan keeps that honest.
        assert np.array_equal(serial[tail], parallel[tail], equal_nan=True)
        assert np.array_equal(
            serial.exceedance_counts[tail], parallel.exceedance_counts[tail]
        )


# ---------------------------------------------------------------------------
# 9. Permutation space: exact integers, unknown where ambiguous
# ---------------------------------------------------------------------------


def test_permutation_space_is_exact_and_honest():
    assert permutation_space("paired", 6) == 2 ** 6
    assert permutation_space("one-sample", 5) == 2 ** 5
    assert permutation_space("two-sample", 3, 4) == 35
    strata = np.array([0, 0, 1, 1, 1, 0, 1])
    # Strata keep the per-stratum group totals: comb(3,2) * comb(4,1).
    assert permutation_space("two-sample", 3, 4, strata) == 3 * 4
    # Sign flips are not restricted by strata.
    assert permutation_space("paired", 4, strata=strata[:4]) == 16
    # Ambiguous / not a fixed relabelling space.
    assert permutation_space("glm", 5) is None
    assert permutation_space("two-sample", 3, 4, strata=np.array([0, 1])) is None


def test_report_flags_replacement_draws_without_suppressing_the_interval():
    g1, g2 = _groups()
    result = _run(g1, g2, n_permutations=20, two_tailed=True)
    # 3 + 3 subjects: space = comb(6, 3) = 20, so B = 50 exceeds it and the
    # informational "draws with replacement" flag must fire.
    small = compute_p_val(
        g1[:3], g2[:3], n_permutations=50, test_type="two-sample",
        method="tstat", use_mp=False, rng=1, two_tailed=True,
    )
    assert small.permutation_space == 20  # comb(6, 3)
    report = small.mc_report()
    assert report["duplicate_draws_possible"] is True
    assert "draws-exceed-permutation-space" in report["flags"]
    assert any("duplicate permutations are expected" in note for note in report["notes"])
    # Informational only: the interval is still emitted.
    assert report["tails"]["positive"]["interval_status"] == "available"
    assert result.permutation_space == 1716


# ---------------------------------------------------------------------------
# Rework R1 · no-pickle workspace v2 save → restore (explicit field contract)
# ---------------------------------------------------------------------------

MC_FIELDS = (
    "exceedance_counts", "count_kind", "rng_provenance", "permutation_space",
)


def _workspace_state(result, n_rois=4):
    return {
        "dataset_info": {
            "name": "synthetic-mc", "mode": "Generate Synthetic Dataset",
            "source": "synthetic", "observations": 13, "n_rois": n_rois,
            "loaded_at": "2026-01-01T00:00:00Z",
        },
        "run_plan": {"method": "tstat", "alpha": 0.05},
        "inference_result": result,
        "edges_df": None,
        "active_tab": "3. Inference Results",
    }


def _remove_mc_fields_from_state_doc(state_doc):
    """Simulate an archive written before the NETSTAT-32 attributes existed."""
    node = state_doc["results"]["primary"]["inference_result"]
    for key in MC_FIELDS:
        node["attrs"].pop(key, None)
        (node.get("extra") or {}).pop(key, None)
    return state_doc


def test_workspace_v2_round_trip_preserves_counts_rng_and_caption(tmp_path):
    import json

    from conninfpy.app.components.inference_results import _monte_carlo_caption
    from conninfpy.app.utils.workspace_io import (
        WORKSPACE_STATE_PATH,
        WORKSPACE_SCHEMA_VERSION,
        build_workspace_archive,
        restore_workspace_archive,
    )

    g1, g2 = _groups()
    result = _run(g1, g2, n_permutations=20, two_tailed=True)
    state = _workspace_state(result)
    zip_bytes, _ = build_workspace_archive(
        state, include_data=False, project_root=tmp_path
    )

    files = _workspace_zip_files(zip_bytes)
    state_doc = json.loads(files[WORKSPACE_STATE_PATH].decode("utf-8"))
    assert state_doc["workspace_schema_version"] == WORKSPACE_SCHEMA_VERSION
    node = state_doc["results"]["primary"]["inference_result"]
    # Explicit contract, not the `extra` catch-all.
    for key in MC_FIELDS:
        assert key in node["attrs"], key
        assert key not in (node.get("extra") or {}), key

    restored = {}
    restore_workspace_archive(zip_bytes, restored, project_root=tmp_path)
    back = restored["inference_result"]
    assert back.n_permutations == result.n_permutations
    assert back.rng_provenance == result.rng_provenance
    assert back.rng_provenance["seed"] == 42
    assert back.rng_provenance["scheme"] == "legacy-randomstate-randint-uint32"
    assert back.rng_provenance["schema_version"] == 1
    assert back.count_kind == result.count_kind == COUNT_KIND_JOINT_MAX
    assert back.permutation_space == result.permutation_space == 1716
    for tail in ("positive", "negative"):
        saved, loaded = result.exceedance_counts[tail], back.exceedance_counts[tail]
        assert loaded.dtype == np.int64 and saved.dtype == np.int64
        assert np.array_equal(loaded, saved)
        assert np.any(loaded == COUNT_UNAVAILABLE)  # sentinels survive
        # Eligibility (captured tested masks) survives with the counts.
        assert np.array_equal(
            back.analysis_provenance["tested"][tail],
            result.analysis_provenance["tested"][tail],
        )
    assert _monte_carlo_caption(back) == _monte_carlo_caption(result)
    assert back.mc_report()["tails"]["positive"]["interval_status"] == "available"


@pytest.mark.parametrize("schema_version", [2, 1])
def test_workspace_legacy_without_mc_fields_restores_as_unknown(
    tmp_path, schema_version
):
    import json

    from conninfpy.app.utils.workspace_io import (
        WORKSPACE_MANIFEST_PATH,
        WORKSPACE_SCHEMA_VERSION,
        WORKSPACE_STATE_PATH,
        build_workspace_archive,
        restore_workspace_archive,
    )

    g1, g2 = _groups()
    result = _run(g1, g2, n_permutations=20, two_tailed=True)
    zip_bytes, _ = build_workspace_archive(
        _workspace_state(result), include_data=False, project_root=tmp_path
    )
    files = _workspace_zip_files(zip_bytes)
    state_doc = json.loads(files[WORKSPACE_STATE_PATH].decode("utf-8"))
    state_doc = _remove_mc_fields_from_state_doc(state_doc)
    records = state_doc["results"]["primary"]["inference_result"]
    assert not (set(MC_FIELDS) & set(records["attrs"]))
    if schema_version != WORKSPACE_SCHEMA_VERSION:
        # Same archive, as a schema-v1 workspace written before the fields.
        state_doc["workspace_schema_version"] = schema_version
        ws_manifest = json.loads(files[WORKSPACE_MANIFEST_PATH].decode("utf-8"))
        ws_manifest["workspace_schema_version"] = schema_version
        files[WORKSPACE_MANIFEST_PATH] = json.dumps(
            ws_manifest, indent=2, sort_keys=True
        ).encode("utf-8")
        root = json.loads(files["manifest.json"].decode("utf-8"))
        if isinstance(root.get("workspace"), dict):
            root["workspace"]["workspace_schema_version"] = schema_version
        files["manifest.json"] = json.dumps(
            root, indent=2, sort_keys=True
        ).encode("utf-8")
    files[WORKSPACE_STATE_PATH] = json.dumps(
        state_doc, indent=2, sort_keys=True
    ).encode("utf-8")
    _sync_workspace_checksums(files)
    legacy_zip = _zip_from_files(files)

    target = {}
    info = restore_workspace_archive(legacy_zip, target, project_root=tmp_path)
    assert info.workspace_schema_version == schema_version
    legacy = target["inference_result"]
    assert legacy.exceedance_counts is None
    assert legacy.count_kind is None
    assert legacy.rng_provenance is None
    assert legacy.permutation_space is None
    report = legacy.mc_report()
    assert report["permutation_space"] is None
    assert report["rng"] is None
    for entry in report["tails"].values():
        assert entry["counts_available"] is False
        assert entry["interval_status"] == "unknown"
        assert "not recorded" in entry["interval_reason"]


# ---------------------------------------------------------------------------
# Rework R2 · available-case counts are real (sentinel where untested)
# ---------------------------------------------------------------------------


def _missing_groups(seed=3):
    rng = np.random.default_rng(seed)
    g1 = rng.normal(0, 1, (7, 4, 4))
    g2 = rng.normal(0.4, 1, (8, 4, 4))
    for group in (g1, g2):
        for matrix in group:
            np.fill_diagonal(matrix, 0.0)
        group[:] = (group + group.transpose(0, 2, 1)) / 2
    g1[:2, 0, 1] = g1[:2, 1, 0] = np.nan
    g2[0, 2, 3] = g2[0, 3, 2] = np.nan
    return g1, g2


def _ac(g1, g2, **kwargs):
    defaults = dict(
        n_permutations=20, test_type="two-sample", method="tstat",
        use_mp=False, rng=5, two_tailed=True, missing="available_case",
    )
    defaults.update(kwargs)
    return compute_p_val(g1, g2, **defaults)


def test_available_case_counts_are_real_with_sentinels_and_no_wilson():
    g1, g2 = _missing_groups()
    result = _ac(g1, g2)
    counts = result.exceedance_counts["positive"]
    assert counts.dtype == np.int64
    tested = result.missing_report["tested_mask"]
    assert np.array_equal(counts >= 0, tested)
    assert np.array_equal(np.diag(counts), np.full(4, COUNT_UNAVAILABLE))
    # The stored (N, N) cells are exactly the +1 estimate of their count.
    mask = counts >= 0
    assert np.any(mask)
    assert np.array_equal(result["positive"][mask], (counts[mask] + 1.0) / 21.0)
    assert result.count_kind == COUNT_KIND_JOINT_MAX
    report = result.mc_report()
    assert report["permutation_space"] is None
    assert "per-edge subject sets differ (available-case)" in (
        report["permutation_space_reason"]
    )
    for entry in report["tails"].values():
        assert entry["counts_available"] is True
        assert entry["interval_status"] == "unknown"
        assert "available-case" in entry["interval_reason"]
        assert entry["interval_low"] is None


def test_available_case_bh_and_accelerated_counts():
    g1, g2 = _missing_groups()
    bh = _ac(g1, g2, method="bh_fdr_perm")
    assert bh.count_kind == COUNT_KIND_TWO_SIDED_SUM_STAT
    bh_counts = bh.exceedance_counts["positive"]
    assert np.any(bh_counts == COUNT_UNAVAILABLE)  # untested/other-direction cells
    assert bh.mc_report()["tails"]["positive"]["interval_status"] == "unknown"

    one = _ac(g1, g2, method="bh_fdr_perm", two_tailed=False)
    assert one.count_kind == COUNT_KIND_EDGE_WISE
    assert np.array_equal(one.exceedance_counts["positive"] >= 0,
                          one.missing_report["tested_mask"])

    acc = _ac(g1, g2, acceleration="gpd")
    report = acc.mc_report()
    assert acc.count_kind == COUNT_KIND_JOINT_MAX
    for entry in report["tails"].values():
        assert entry["counts_available"] is True
        assert entry["interval_status"] == "unknown"
        assert "acceleration=gpd" in entry["interval_reason"]


def test_available_case_serial_and_mp_agree_bitwise():
    g1, g2 = _missing_groups()
    serial = _ac(g1, g2, n_permutations=15)
    parallel = _ac(g1, g2, n_permutations=15, use_mp=True, n_processes=2)
    for tail in ("positive", "negative"):
        # NaN marks untested available-case edges; equal_nan keeps that honest.
        assert np.array_equal(serial[tail], parallel[tail], equal_nan=True)
        assert np.array_equal(
            serial.exceedance_counts[tail], parallel.exceedance_counts[tail]
        )


def test_available_case_glm_counts_and_f_stat_kind():
    from conninfpy import compute_p_val_glm

    rng = np.random.default_rng(3)
    Y = rng.normal(0, 1, (12, 4, 4))
    Y = (Y + Y.transpose(0, 2, 1)) / 2
    for matrix in Y:
        np.fill_diagonal(matrix, 0.0)
    Y[:2, 0, 1] = Y[:2, 1, 0] = np.nan
    Y[3:, 2, 3] = Y[3:, 3, 2] = np.nan
    X = np.column_stack([np.ones(12), rng.normal(0, 1, 12)])

    result = compute_p_val_glm(
        Y, design_matrix=X, contrast=[0, 1], n_permutations=20, method="tstat",
        use_mp=False, rng=3, two_tailed=True, missing="available_case",
    )
    counts = result.exceedance_counts["positive"]
    tested = result.missing_report["tested_mask"]
    assert result.count_kind == COUNT_KIND_JOINT_MAX
    assert result.permutation_space is None
    assert np.array_equal(counts >= 0, tested)
    mask = counts >= 0
    assert np.array_equal(result["positive"][mask], (counts[mask] + 1.0) / 21.0)
    report = result.mc_report()
    assert report["tails"]["positive"]["interval_status"] == "unknown"
    assert "available-case" in report["tails"]["positive"]["interval_reason"]
    assert report["rng"]["seed"] == 3

    f_stat = compute_p_val_glm(
        Y, design_matrix=X, contrast=np.array([[0, 1]]), stat_type="fstat",
        n_permutations=20, method="tstat", use_mp=False, rng=3,
        missing="available_case",
    )
    assert f_stat.count_kind == COUNT_KIND_MAX_STAT
    omnibus_counts = f_stat.exceedance_counts["omnibus"]
    om_mask = omnibus_counts >= 0
    assert np.array_equal(f_stat["omnibus"][om_mask],
                          (omnibus_counts[om_mask] + 1.0) / 21.0)


def test_available_case_cnbs_reason_says_block_level():
    g1, g2 = _missing_groups()
    result = _ac(g1, g2, method="cnbs", net_labels=np.array([0, 0, 1, 1]),
                 n_permutations=20)
    assert result.exceedance_counts is None
    entry = result.mc_report()["tails"]["positive"]
    assert entry["interval_status"] == "unknown"
    assert "block-level" in entry["interval_reason"]


# ---------------------------------------------------------------------------
# Rework R3 · CLI bundle per-edge recovery + aggregate labeling
# ---------------------------------------------------------------------------


def _cli_study_run(tmp_path, design, **inference):
    import contextlib
    import io
    import yaml

    from tests.test_cli import _Study, _correlations, _pheno

    study = _Study(tmp_path, _pheno(24), _correlations(24))
    config = study.analysis(design)
    if inference:
        cfg = yaml.safe_load(config.read_text())
        cfg["inference"].update(inference)
        config.write_text(yaml.safe_dump(cfg))
    from conninfpy.cli import run

    with contextlib.redirect_stdout(io.StringIO()):
        return run(config)


def test_cli_bundle_exposes_per_edge_counts_and_labels_aggregates(tmp_path):
    written = _cli_study_run(
        tmp_path, {"type": "glm", "interest": "age", "confounds": ["sex"]}
    )
    bundle = ResultBundle.load(written["bundle"])
    mc = bundle.metadata["mc"]
    b = mc["n_permutations"]
    assert b > 0
    assert mc["level"] == "per_edge"
    assert mc["counts_are_per_edge"] is True
    assert mc["p_relation"] == "(count + 1) / (B + 1) on counted cells"
    assert mc["count_sentinel"] == -1
    assert mc["rng"]["seed"] == 0
    for tail in ("positive", "negative"):
        counts = np.asarray(mc["exceedance_counts"][tail])
        p = np.asarray(bundle.corrected_p[tail])
        mask = counts >= 0
        assert mask.any()
        # Per-edge B/count/+1 recovery straight from the CLI-written bundle.
        assert np.array_equal(p[mask], (counts[mask] + 1.0) / (b + 1.0))

    prov = json.loads(written["provenance"].read_text())
    summary = prov["results"]["result"]["monte_carlo"]
    # Run-level block, explicitly labelled, pointing at the per-edge sources.
    assert summary["level"] == "run" and summary["aggregate"] is True
    assert "result_bundle.zip" in summary["per_edge_source"]
    assert "edges.csv" in summary["per_edge_source"]
    assert summary["p_relation"] == "(count + 1) / (B + 1) on counted cells"
    assert summary["permutation_space"] is None
    assert "ambiguous" in summary["permutation_space_reason"]
    entry = summary["tails"]["positive"]
    assert entry["counts_available"] is True
    assert "count_min_aggregate" in entry and "count_max_aggregate" in entry
    assert entry["interval_status"] == "available"


def test_bh_bundle_adjusted_p_is_bh_of_the_empirical_counts():
    from conninfpy.stats._engine import _bh_fdr_correction

    g1, g2 = _groups()
    result = _run(g1, g2, method="bh_fdr_perm", two_tailed=False)
    bundle = result.to_bundle()
    mc = bundle.metadata["mc"]
    b = mc["n_permutations"]
    assert mc["count_kind"] == COUNT_KIND_EDGE_WISE
    triu = np.triu_indices(4, k=1)
    counts = np.asarray(mc["exceedance_counts"]["positive"])[triu]
    kept = counts >= 0
    empirical = (counts[kept] + 1.0) / (b + 1.0)
    rebuilt = _bh_fdr_correction(empirical)
    stored = np.asarray(bundle.corrected_p["positive"])[triu][kept]
    assert np.array_equal(stored, rebuilt)
    # The count is the raw empirical exceedance, never the adjusted q.
    assert not np.array_equal(empirical, stored)
