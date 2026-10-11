"""Bounded checks for prediction leakage, feature definitions and resampling units."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.open_close.feature_stability import experiment as prediction
from examples.open_close.feature_stability import data as inputs, fitting, resampling
from examples.open_close.feature_stability.metrics import (
    agreement_icc, prediction_stability, test_participant_draws as participant_test_draws,
)
from examples.open_close.feature_stability import feature_stability as stability_analysis


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(17)
    atlas = pd.DataFrame({"name": ["a", "b", "c", "d"], "id": [1, 2, 3, 4],
                          "network": ["Visual", "Visual", "Default", "Default"]})
    annotations = pd.DataFrame({"roi": atlas.name, "level": ["subregion"] * 4,
                               "lobe": ["Frontal", "Frontal", "Temporal", "Temporal"],
                               "gyrus": ["g1", "g1", "g2", "g2"], "hemisphere": ["L", "L", "R", "R"]})
    rois, edges, spaces = inputs.feature_spaces(atlas, annotations, "lobe")
    participants, scans, values = [], [], []
    for site in ("ihb", "china"):
        for person in range(7):
            p = len(participants)
            identity = f"{site}:{person}"
            participants.append({"participant_index": p, "participant_id": identity,
                                 "subject_id": str(person), "site": site, "source_index": person})
            person_effect = rng.normal(0, 0.3, 6)
            for state in (0, 1):
                scans.append({"scan_index": len(scans), "participant_index": p,
                              "participant_id": identity, "subject_id": str(person), "site": site,
                              "state": state, "scan_id": f"{identity}:{state}", "close_run": 0})
                values.append(rng.normal(0, 0.3, 6) + person_effect + state * np.array([0.7, 0.3, 0, 0, 0, 0.3])
                              + (0.2 if site == "china" else 0))
    return inputs.PreparedData(np.asarray(values), pd.DataFrame(participants), pd.DataFrame(scans),
                                   rois, edges, spaces)


def test_block_means_use_all_edges_and_one_shared_mask():
    atlas = pd.DataFrame({"name": list("abcde"), "id": range(1, 6),
                          "network": ["A", "A", "B", "B", "B"]})
    annotations = pd.DataFrame({"roi": list("abcde"), "level": ["subregion", "gyrus", "lobe", "mixed", "subregion"],
                               "lobe": ["F", "F", "T", None, "T"], "hemisphere": ["L", "L", "R", None, "R"],
                               "gyrus": ["f1", "f1", None, None, "t1"]})
    rois, edges, spaces = inputs.feature_spaces(atlas, annotations.iloc[::-1], "lobe")
    assert rois.loc[rois.eligible, "roi"].tolist() == list("abce")
    assert len(edges) == 6
    values = np.arange(1, 7, dtype=float)[None, :]
    expected = {"A | A": 1, "A | B": 3.5, "B | B": 6}
    for label, value in zip(spaces["functional"].labels, spaces["functional"].transform(values)[0]):
        assert value == expected[label]
    gyrus_rois, gyrus_edges, gyrus_spaces = inputs.feature_spaces(atlas, annotations, "gyrus")
    assert gyrus_rois.loc[gyrus_rois.eligible, "roi"].tolist() == list("abe")
    assert len(gyrus_edges) == len(gyrus_spaces["edges"].labels) == 3
    assert "B | B" not in gyrus_spaces["functional"].labels  # Singleton group, no self-edge.
    with pytest.raises(ValueError, match="exactly"):
        inputs.feature_spaces(atlas, annotations.iloc[:-1], "lobe")


def test_shipped_crosswalk_counts():
    root = Path(__file__).resolve().parents[1]
    atlas = pd.read_csv(root / "datasets/open_close/schaefer_labels.csv")
    annotations = pd.read_csv(prediction.HERE / "roi_groups.csv")
    for level, count, n_edges, n_anatomical in (("lobe", 178, 15753, 78), ("gyrus", 165, 13530, 733)):
        rois, _, spaces = inputs.feature_spaces(atlas, annotations, level)
        assert rois.eligible.sum() == count
        assert [len(spaces[n].labels) for n in prediction.REPRESENTATIONS] == [n_edges, 27, n_anatomical]


def test_splits_folds_and_duplicate_draws_keep_pairs(toy_data, tmp_path):
    splits = resampling.split_participants(toy_data.participants, {"ihb": 2, "china": 2}, 42)
    splits["cv_fold"] = resampling.development_folds(splits, 2, 43)
    for site in ("ihb", "china"):
        assert ((splits.site == site) & (splits["split"] == "test")).sum() == 2
        assert set(splits.loc[(splits.site == site) & (splits["split"] == "development"), "cv_fold"]) == {0, 1}
    people = resampling.participant_bootstraps(splits, 10, 44)
    test_people = set(splits.loc[splits["split"] == "test", "participant_index"])
    for draw in people:
        assert not set(draw) & test_people
        indices = resampling.scan_indices(toy_data.scans, draw)
        assert len(indices) == 20
        assert toy_data.scans.iloc[indices].groupby("site").size().to_dict() == {"china": 10, "ihb": 10}
        for p in np.unique(draw):
            states = toy_data.scans.iloc[indices].loc[lambda x: x.participant_index == p, "state"].value_counts()
            assert states[0] == states[1] == np.sum(draw == p)
    assert any(len(np.unique(draw)) < len(draw) for draw in people)
    saved = tmp_path / "split.csv"
    splits.iloc[::-1].to_csv(saved, index=False)
    reused = resampling.split_participants(toy_data.participants, {}, 99, saved)
    assert reused["split"].equals(splits["split"])
    assert np.array_equal(people, resampling.participant_bootstraps(reused, 10, 44))


def test_training_boundaries_and_auditable_model_states(toy_data, tmp_path, monkeypatch):
    calls = []
    original_fit = fitting.combat_fit

    def checked_fit(y, *, sites, preserve):
        assert preserve is None
        calls.append(y.copy())
        return original_fit(y, sites=sites, preserve=preserve)

    monkeypatch.setattr(fitting, "combat_fit", checked_fit)
    original_transform = inputs.FeatureSpace.transform

    def only_edges_in_fitting(space, values):
        assert space.name == "edges", "Grouping must never change classifier input features."
        assert values.shape[1] == toy_data.edges.shape[1]
        return original_transform(space, values)

    monkeypatch.setattr(inputs.FeatureSpace, "transform", only_edges_in_fitting)
    with threadpool_limits(limits=1):
        config = prediction.run_experiment(toy_data, tmp_path / "run", n_bootstraps=3,
                                           n_test_bootstraps=4, n_folds=2,
                                           test_counts={"ihb": 2, "china": 2}, c_grid=(0.1, 1.0))
    assert config["status"] == "complete"
    assert set(config["selected_C"]) == {"edges_raw", "edges_combat"}
    assert config["fitted_representation"] == "individual_edges_only"
    folder = tmp_path / "run"
    splits = pd.read_csv(folder / "participant_splits.csv")
    development = resampling.scan_indices(toy_data.scans, splits.loc[splits["split"] == "development", "participant_index"].to_numpy())
    test = resampling.scan_indices(toy_data.scans, splits.loc[splits["split"] == "test", "participant_index"].to_numpy())
    assert len(calls) == 2 + 1 + 3  # One ComBat per CV fold, reference and bootstrap.
    for train in calls:
        for row in train:
            assert np.any(np.all(toy_data.edges[development] == row, axis=1))
            assert not np.any(np.all(toy_data.edges[test] == row, axis=1))
    draws = np.load(folder / "training_bootstrap_participant_indices.npy")
    probabilities = pd.read_csv(folder / "bootstrap_test_predictions.csv")
    for b, people in enumerate(draws):
        expected_train = resampling.scan_indices(toy_data.scans, people)
        panel = probabilities[probabilities.bootstrap == b]
        assert panel.groupby("pipeline").scan_id.apply(tuple).nunique() == 1
        for pipeline in config["selected_C"]:
            saved = np.load(folder / "models" / f"bootstrap_{b:04d}" / f"{pipeline}.npz")
            assert np.array_equal(saved["train_scan_indices"], expected_train)
            assert np.array_equal(saved["test_scan_indices"], test)
            edges = toy_data.edges[test]
            if pipeline.endswith("_combat"):
                state = np.load(folder / "models" / f"bootstrap_{b:04d}" / "combat.npz")
                combat = fitting.ComBatModel(site_labels=state["site_labels"].tolist(),
                                                **{k: state[k] for k in ("alpha", "sigma", "gamma_star", "delta_star")})
                assert int(state["preserve_n_cols"]) == 0
                edges = fitting.combat_apply(combat, edges, toy_data.scans.iloc[test].site.to_numpy())
            x = toy_data.spaces[pipeline.rsplit("_", 1)[0]].transform(edges)
            expected_probability = expit(x @ saved["coef_feature_units"] + saved["intercept_feature_units"])
            actual = panel.loc[panel.pipeline == pipeline, "probability_closed"].to_numpy()
            np.testing.assert_allclose(actual, expected_probability, atol=1e-12)
    report = pd.read_csv(folder / "feature_stability/feature_pattern_stability.csv")
    assert set(report.level) == {"edges", "functional", "anatomical"}
    assert set(report.n_model_edges) == {toy_data.edges.shape[1]}
    assert set(report.aggregation) == {"post_fit_same_edge_model"}


def test_evaluation_changes_do_not_change_fit(toy_data, tmp_path):
    train = np.r_[0:8, 14:22]
    test = np.r_[8:14, 22:28]
    c = {pipeline: 0.1 for pipeline in prediction.PIPELINES}
    with threadpool_limits(limits=1):
        _, weights = fitting.fit_panel(toy_data, train, test, c, 42, tmp_path / "first")
        changed = inputs.PreparedData(toy_data.edges.copy(), toy_data.participants, toy_data.scans,
                                          toy_data.rois, toy_data.edge_definitions, toy_data.spaces)
        changed.edges[test] += 100
        _, changed_weights = fitting.fit_panel(changed, train, test, c, 42, tmp_path / "changed")
    for pipeline in c:
        np.testing.assert_array_equal(weights[pipeline], changed_weights[pipeline])
        original = np.load(tmp_path / "first" / f"{pipeline}.npz")
        altered = np.load(tmp_path / "changed" / f"{pipeline}.npz")
        np.testing.assert_array_equal(original["scaler_mean"], altered["scaler_mean"])
    with pytest.raises(ValueError, match="not present at fit time"):
        fitting.transform_edges(toy_data.edges[train], toy_data.edges[test],
                                   toy_data.scans.iloc[train].site.to_numpy(), np.array(["unseen"] * len(test)), True)


def test_stability_and_test_uncertainty_use_pairs(toy_data):
    identical = np.tile(np.array([0.1, 0.2, 0.8, 0.9])[:, None], (1, 5))
    assert agreement_icc(identical) == pytest.approx(1)
    offsets = identical + np.arange(5)[None, :] * 0.01
    assert agreement_icc(offsets) < 1
    probabilities = np.array([[0.2, 0.6], [0.3, 0.4], [0.7, 0.8]])
    stability = prediction_stability(probabilities, np.array([0.1, 0.9]))
    np.testing.assert_allclose(stability.pairwise_decision_disagreement, [2/3, 2/3])
    np.testing.assert_allclose(stability.decision_flip_vs_reference, [1/3, 1/3])
    draws = participant_test_draws(toy_data.scans, 10, 123)
    for draw in draws:
        sample = toy_data.scans.iloc[draw]
        assert sample.groupby(["site", "state"]).size().to_dict() == {
            ("ihb", 0): 7, ("ihb", 1): 7, ("china", 0): 7, ("china", 1): 7}
        for _, person in sample.groupby("participant_index"):
            assert person.state.value_counts()[0] == person.state.value_counts()[1]


def test_postfit_importance_is_mean_absolute_not_absolute_mean(tmp_path):
    groups = stability_analysis.Grouping("anatomical", ["A", "B"], np.array([2, 1]), np.array([0, 0, 1]),
                                       inputs.csr_matrix([[0.5, 0], [0.5, 0], [0, 1]]))
    edge = stability_analysis.Grouping("edges", ["e1", "e2", "e3"], np.ones(3, dtype=int), None, None)
    effects = np.array([[4., -4., 1.], [2., -2., 1.], [3., -3., 1.]])
    stability_analysis.analyze_patterns(effects, {"edges": edge, "anatomical": groups}, pipeline="model",
                                        site="site", output=tmp_path, fractions=(0.5,))
    features = pd.read_csv(tmp_path / "model__site__anatomical_features.csv")
    assert features.coefficient_mean.tolist() == [0., 1.]  # Opposite signs cancel in the signed summary.
    assert features.importance_mean.tolist() == [3., 1.]  # Magnitude is retained in importance.
    assert features.total_absolute_mass_mean.tolist() == [6., 1.]
    assert features.top_50pct_selection_frequency.tolist() == [1., 0.]


def test_replacement_edges_can_give_stable_postfit_group_importance(tmp_path):
    groups = stability_analysis.Grouping("functional", ["A", "B"], np.array([2, 2]), np.array([0, 0, 1, 1]),
                                       inputs.csr_matrix([[0.5, 0], [0.5, 0], [0, 0.5], [0, 0.5]]))
    edge = stability_analysis.Grouping("edges", ["a1", "a2", "b1", "b2"], np.ones(4, dtype=int), None, None)
    effects = np.array([[8., 0., 1., 0.], [0., 8., 0., 1.], [8., 0., 0., 1.]])
    _, overlap, _ = stability_analysis.analyze_patterns(effects, {"edges": edge, "functional": groups},
                                                       pipeline="model", site="site", output=tmp_path, fractions=(0.25,))
    by_level = {r["level"]: r for r in overlap}
    assert by_level["functional"]["jaccard_mean"] == 1
    assert by_level["edges"]["jaccard_mean"] == pytest.approx(1 / 3)
    assert by_level["functional"]["kuncheva_mean"] == 1


def test_overlap_chance_correction_and_projection_capacity():
    selected = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]], dtype=bool)
    values = stability_analysis.overlaps(selected)
    np.testing.assert_allclose(values["dice"], [0.5, 0., 0.5])
    np.testing.assert_allclose(values["jaccard"], [1/3, 0., 1/3])
    np.testing.assert_allclose(values["kuncheva"], [0., -1., 0.])
    reverse = stability_analysis.overlaps(np.array([[1, 0], [1, 0]], dtype=bool), np.array([[1., 0.], [-1., 0.]]))
    assert reverse["dice"][0] == 1 and reverse["sign_concordant_dice"][0] == 0
    null = stability_analysis.presence_null(4, np.array([1, 3]), 1)
    np.testing.assert_allclose(null["hit_probability"], [0.25, 0.75])
    assert null["intersection_over_union_of_expectations"] == pytest.approx(0.625/1.375)
    all_selected = stability_analysis.overlaps(np.ones((3, 2), dtype=bool))
    assert np.isnan(all_selected["kuncheva"]).all()  # Saturated coarse overlap is uninformative.


def test_combat_weights_return_to_common_original_edge_units():
    weights = np.array([[4., -3.], [6., -8.]])
    delta = np.array([[2., 3.], [3., 4.]])
    np.testing.assert_allclose(stability_analysis.original_edge_weights(weights, delta), [[2., -1.], [2., -2.]])
    with pytest.raises(ValueError, match="positive"):
        stability_analysis.original_edge_weights(weights, np.zeros_like(delta))
