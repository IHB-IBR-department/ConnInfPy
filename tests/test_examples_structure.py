"""Bounded checks for refactored example paths, reuse and execution controls."""
from types import SimpleNamespace
import ast
import importlib.util
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conninfpy.results import InferenceResult
from examples._shared.paths import ROOT, config_path
from examples.open_close.atlas_stability import inference as cache
from examples.open_close.atlas_stability.metrics import draws, unit_summaries
from examples.open_close.paper_validation.regenerate_validation_results import commands
from examples.simulations.fpr.fpr_calibration import FPRConfig
from examples.simulations.glm.glm_validation import _load_config
from examples.simulations.power import power_analysis as power


def test_examples_compile_and_study_code_never_imports_runners():
    count = 0
    for path in (ROOT / "examples").rglob("*.py"):
        if any(part in path.parts for part in ("results", "__pycache__")):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not any(part.startswith("run_") for part in node.module.split(".")), path
            if isinstance(node, ast.Import):
                assert not any(part.startswith("run_") for name in node.names for part in name.name.split(".")), path
        count += 1
    assert count >= 70


def test_regeneration_uses_active_python_and_separates_model_workflows():
    paper, ml = commands("paper"), commands("ml")
    assert all(cmd[:2] == [sys.executable, "-m"] for cmd in paper + ml)
    assert not any("ml_transfer" in cmd[2] or "feature_stability" in cmd[2] for cmd in paper)
    assert any("run_state_methods" in cmd[2] for cmd in paper)
    assert any("run_ml_feature_selection" in cmd[2] for cmd in ml)
    assert all(importlib.util.find_spec(cmd[2]) for cmd in commands("all"))


def test_yaml_paths_are_relative_to_config_not_cwd(tmp_path, monkeypatch):
    config = tmp_path / "settings" / "run.yaml"
    config.parent.mkdir()
    config.write_text("output_dir: ../results/run\nmethods: [tstat]\n")
    monkeypatch.chdir(tmp_path)
    expected = tmp_path / "results/run"
    assert config_path("../results/run", config) == expected
    assert FPRConfig.from_yaml(config).output_dir == expected
    assert _load_config(config).output_dir == expected
    assert power.load_yaml_config(config)["output_dir"] == expected
    assert config_path(expected, config) == expected


def test_paired_unit_effects_match_explicit_edge_weighting():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(8, 4, 4))
    values = (values + values.transpose(0, 2, 1)) / 2
    for matrix in values:
        np.fill_diagonal(matrix, 0)
    membership = np.array([[1, 0], [.2, .8], [0, 1], [.6, .4]])
    summary = unit_summaries(values, membership, values.mean(0))
    i, j = np.triu_indices(4, 1)
    for column, (a, b) in enumerate(zip(*np.triu_indices(2))):
        weights = membership[i, a] * membership[j, b]
        if a != b:
            weights += membership[j, a] * membership[i, b]
        np.testing.assert_allclose(summary["E"][:, column], values[:, i, j] @ weights / weights.sum())
    bootstrap = draws(8, 20)
    np.testing.assert_allclose(bootstrap.sum(1), 1)
    assert np.array_equal(bootstrap, draws(8, 20))


def test_cache_reuse_checks_data_order_budget_settings_and_shape(tmp_path, monkeypatch):
    branch = SimpleNamespace(cohort="toy", name="atlas", subjects=["a", "b", "c"],
                             parcels=pd.DataFrame({"id": [1, 2], "name": ["x", "y"]}))
    eo = np.zeros((3, 2, 2)); ec = eo.copy()
    ec[:, 0, 1] = ec[:, 1, 0] = [.1, .2, .3]
    settings = dict(cache.TFNBS, use_mp=False)
    calls = []

    def analyze(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(inference=InferenceResult(
            np.array([[1., .2], [.2, 1.]]), np.ones((2, 2)), method="tfnbs",
            stat_positive=np.array([[0., 2.], [2., 0.]]), stat_negative=np.zeros((2, 2))))

    monkeypatch.setattr(cache, "analyze", analyze)
    original = cache.tfnbs(branch, eo, ec, 9, False, tmp_path, settings)
    reused = cache.tfnbs(branch, eo, ec, 9, True, tmp_path, settings)
    assert len(calls) == 1
    np.testing.assert_array_equal(original.positive, reused.positive)
    for changed_eo, changed_ec, budget, options in (
        (eo + .01, ec, 9, settings), (eo, ec + .01, 9, settings),
        (eo, ec, 99, settings), (eo, ec, 9, dict(settings, rng=99)),
    ):
        with pytest.raises(ValueError, match="stale cache"):
            cache.tfnbs(branch, changed_eo, changed_ec, budget, True, tmp_path, options)
    branch.subjects.reverse()
    with pytest.raises(ValueError, match="stale cache"):
        cache.tfnbs(branch, eo, ec, 9, True, tmp_path, settings)
    branch.subjects.reverse()
    branch.parcels = branch.parcels.iloc[::-1]
    with pytest.raises(ValueError, match="stale cache"):
        cache.tfnbs(branch, eo, ec, 9, True, tmp_path, settings)
    branch.parcels = branch.parcels.iloc[::-1]
    path = tmp_path / "tfnbs/toy__atlas.npz"
    with np.load(path) as saved:
        payload = {key: saved[key] for key in saved.files}
    payload["positive"] = np.ones((3, 3))
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="shape"):
        cache.tfnbs(branch, eo, ec, 9, True, tmp_path, settings)
    payload.pop("cache_identity")
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="Unverified"):
        cache.tfnbs(branch, eo, ec, 9, True, tmp_path, settings)
    assert len(calls) == 1


def test_real_tfnbs_cache_round_trip(tmp_path):
    rng = np.random.default_rng(19)
    eo = rng.normal(size=(6, 4, 4))
    eo = (eo + eo.transpose(0, 2, 1)) / 2
    ec = eo + rng.normal(size=eo.shape) * .1
    ec = (ec + ec.transpose(0, 2, 1)) / 2
    for array in (eo, ec):
        array[:, np.arange(4), np.arange(4)] = 0
    branch = SimpleNamespace(cohort="toy", name="atlas", subjects=list("abcdef"),
                             parcels=pd.DataFrame({"id": range(4), "name": list("wxyz")}))
    settings = dict(cache.TFNBS, use_mp=False)
    fresh = cache.tfnbs(branch, eo, ec, 9, False, tmp_path, settings)
    reused = cache.tfnbs(branch, eo, ec, 9, True, tmp_path, settings)
    np.testing.assert_array_equal(fresh.positive, reused.positive)
    np.testing.assert_array_equal(fresh.stat_positive, reused.stat_positive)


@pytest.mark.parametrize("run,kwargs", [
    (power.run_power_vs_effect_size, {"effect_sizes": [.15], "n_samples": 6}),
    (power.run_power_vs_sample_size, {"sample_sizes": [6], "effect_size": .15}),
])
def test_power_serial_control_does_not_start_pool(run, kwargs, monkeypatch):
    monkeypatch.setattr(power.mp, "get_context", lambda *_: SimpleNamespace(
        Pool=lambda **_: pytest.fail("n_jobs=1 started a pool")))
    result = run(methods=["tstat", "ni_tfnbs"], scenarios=["within_module_dense"],
                 n_repeats=1, n_nodes=6, n_modules=2, time_points=24, n_permutations=9,
                 n_jobs=1, method_configs={"tstat": {}, "ni_tfnbs": {"e": .3, "h": 3., "n": 3}},
                 **kwargs)
    assert len(result) == 2


@pytest.mark.parametrize("script", [
    "open_close/atlas_stability/run_atlas_stability.py",
    "open_close/feature_stability/run_feature_stability.py",
    "open_close/paper_validation/regenerate_validation_results.py",
    "abide/age/run_age_strategy_d.py",
    "benchmarks/performance/timing_benchmark.py",
    "simulations/power/power_analysis.py",
])
def test_entry_point_help_from_other_cwd(script, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "examples" / script), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
