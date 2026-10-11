"""Fit EO/EC models on individual edges and summarize feature reproducibility.

See README.md for the design, reproduction commands and output definitions.
This example never uses a difference test or a significance-based feature mask.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from conninfpy.harmonize import ComBatModel, combat_apply, combat_fit  # noqa: E402
from examples.open_close.openclose_loader import OpenCloseDataset, DATA_DIR  # noqa: E402
from examples.open_close.prediction.metrics import (  # noqa: E402
    METRICS, agreement_icc, coefficient_correlations, paired_differences,
    prediction_stability, scores, sign_stability, test_participant_draws,
)
from examples.open_close.prediction.feature_stability import analyze_saved_edge_models  # noqa: E402

HERE = Path(__file__).resolve().parent
C_GRID = (0.01, 0.1, 1.0, 10.0)
REPRESENTATIONS = ("edges", "functional", "anatomical")
PIPELINES = ("edges_raw", "edges_combat")


@dataclass
class FeatureSpace:
    name: str
    labels: list[str]
    edge_groups: np.ndarray | None
    counts: np.ndarray
    projection: csr_matrix | None

    def transform(self, edges: np.ndarray) -> np.ndarray:
        return edges if self.projection is None else np.asarray(edges @ self.projection)


@dataclass
class PreparedData:
    edges: np.ndarray
    participants: pd.DataFrame
    scans: pd.DataFrame
    rois: pd.DataFrame
    edge_definitions: pd.DataFrame
    spaces: dict[str, FeatureSpace]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def block_space(name: str, roi_groups: np.ndarray, rows: np.ndarray,
                cols: np.ndarray) -> FeatureSpace:
    pairs = [tuple(sorted((str(roi_groups[i]), str(roi_groups[j])))) for i, j in zip(rows, cols)]
    unique = sorted(set(pairs))  # Only nonempty blocks; diagonal self-edges are excluded.
    lookup = {pair: i for i, pair in enumerate(unique)}
    groups = np.asarray([lookup[pair] for pair in pairs])
    counts = np.bincount(groups, minlength=len(unique))
    projection = csr_matrix((1.0 / counts[groups], (np.arange(len(groups)), groups)),
                            shape=(len(groups), len(unique)))
    return FeatureSpace(name, [f"{a} | {b}" for a, b in unique], groups, counts, projection)


def feature_spaces(atlas: pd.DataFrame, annotations: pd.DataFrame,
                   anatomical_level: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, FeatureSpace]]:
    """Align by ROI name, then use one anatomical eligibility mask for every space."""
    if annotations.roi.duplicated().any() or atlas.name.duplicated().any():
        raise ValueError("ROI names must be unique.")
    if set(annotations.roi) != set(atlas.name):
        raise ValueError("Annotations must cover exactly the source atlas ROI names.")
    aligned = annotations.set_index("roi").loc[atlas.name].reset_index()
    aligned["source_index"] = np.arange(len(atlas))
    aligned["atlas_id"] = atlas.id.to_numpy()
    aligned["functional"] = atlas.network.to_numpy()  # Source Yeo labels are authoritative.
    eligible_levels = ("subregion", "gyrus", "lobe") if anatomical_level == "lobe" else ("subregion", "gyrus")
    eligible = aligned.level.isin(eligible_levels)
    aligned["eligible"] = eligible
    if aligned.loc[eligible, [anatomical_level, "hemisphere"]].isna().any().any():
        raise ValueError("Anatomically eligible ROIs need a group and hemisphere.")
    if not aligned.loc[eligible, "hemisphere"].isin(["L", "R"]).all():
        raise ValueError("Anatomically eligible ROIs must have L/R hemisphere labels.")
    rois = aligned.loc[eligible].copy().reset_index(drop=True)
    if len(rois) < 2:
        raise ValueError("At least two anatomically assigned ROIs are required.")
    rois["anatomical"] = rois[anatomical_level] + "_" + rois.hemisphere
    rows, cols = np.triu_indices(len(rois), k=1)
    labels = [f"{rois.roi[i]} | {rois.roi[j]}" for i, j in zip(rows, cols)]
    spaces = {
        "edges": FeatureSpace("edges", labels, None, np.ones(len(labels), dtype=int), None),
        "functional": block_space("functional", rois.functional.to_numpy(), rows, cols),
        "anatomical": block_space("anatomical", rois.anatomical.to_numpy(), rows, cols),
    }
    edges = pd.DataFrame({
        "edge_index": np.arange(len(rows)), "roi_i": rois.roi.to_numpy()[rows],
        "roi_j": rois.roi.to_numpy()[cols], "atlas_id_i": rois.atlas_id.to_numpy()[rows],
        "atlas_id_j": rois.atlas_id.to_numpy()[cols],
    })
    for name in ("functional", "anatomical"):
        edges[name + "_feature_index"] = spaces[name].edge_groups
    return aligned, edges, spaces


def load_data(data_dir: Path, annotations_path: Path, anatomical_level: str,
              china_close_run: int) -> PreparedData:
    datasets = {site: OpenCloseDataset.load(site, data_dir=data_dir) for site in ("ihb", "china")}
    atlas = datasets["ihb"].atlas_df
    if not atlas.equals(datasets["china"].atlas_df):
        raise ValueError("Site atlases are not aligned.")
    annotated, edge_definitions, spaces = feature_spaces(atlas, pd.read_csv(annotations_path), anatomical_level)
    keep = annotated.loc[annotated.eligible, "source_index"].to_numpy()
    rows, cols = np.triu_indices(len(keep), k=1)
    participants, scans, all_edges = [], [], []
    for site, ds in datasets.items():
        # Same complete China participants for either run, including run0 sensitivity.
        subject_indices = np.flatnonzero(ds.retest_subject_mask())
        if len(set(ds.subject_ids)) != len(ds.subject_ids):
            raise ValueError(f"Duplicate subject IDs at {site}.")
        close_ts = ds.close_ts_run1 if site == "china" and china_close_run == 1 else ds.close_ts
        open_z = ds._ts_to_fisher_z(ds.open_ts[subject_indices][:, :, keep])
        close_z = ds._ts_to_fisher_z(close_ts[subject_indices][:, :, keep])
        for local, source_index in enumerate(subject_indices):
            p = len(participants)
            subject_id = ds.subject_ids[source_index]
            participant_id = f"{site}:{subject_id}"
            participants.append({"participant_index": p, "participant_id": participant_id,
                                 "subject_id": subject_id, "site": site, "source_index": int(source_index)})
            for state, z in ((0, open_z[local]), (1, close_z[local])):
                scans.append({"scan_index": len(scans), "participant_index": p,
                              "participant_id": participant_id, "subject_id": subject_id,
                              "site": site, "state": state,
                              "scan_id": participant_id + (":EO" if state == 0 else ":EC"),
                              "close_run": (china_close_run if site == "china" else 0) if state else -1})
                all_edges.append(z[rows, cols])
    edges = np.asarray(all_edges)
    if not np.isfinite(edges).all():
        raise ValueError("Nonfinite FC features; inspect time series before fitting.")
    return PreparedData(edges, pd.DataFrame(participants), pd.DataFrame(scans), annotated,
                        edge_definitions, spaces)


def split_participants(participants: pd.DataFrame, test_counts: dict[str, int],
                       seed: int, split_from: Path | None = None) -> pd.DataFrame:
    result = participants.copy()
    if split_from is not None:
        previous = pd.read_csv(split_from)
        if previous.participant_id.duplicated().any() or set(previous.participant_id) != set(result.participant_id):
            raise ValueError("Saved split must contain exactly the current participants, once each.")
        previous = previous.set_index("participant_id").loc[result.participant_id]
        if not np.array_equal(previous.site.to_numpy(), result.site.to_numpy()):
            raise ValueError("Saved split has inconsistent sites.")
        result["split"] = previous["split"].to_numpy()
        if not result["split"].isin(["development", "test"]).all():
            raise ValueError("Saved split must use development/test labels.")
    else:
        rng = np.random.default_rng(seed)
        result["split"] = "development"
        for site in sorted(result.site.unique()):
            pool = result.index[result.site == site].to_numpy()
            n = test_counts[site]
            if n < 2 or n >= len(pool):
                raise ValueError(f"Invalid test count for {site}: {n}.")
            result.loc[rng.choice(pool, n, replace=False), "split"] = "test"
    for site in result.site.unique():
        for split in ("development", "test"):
            if ((result.site == site) & (result["split"] == split)).sum() < 2:
                raise ValueError("Each site needs at least two participants in each split.")
    return result


def development_folds(participants: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    assignments = np.full(len(participants), -1, dtype=int)
    for site in sorted(participants.site.unique()):
        pool = np.flatnonzero((participants.site == site) & (participants["split"] == "development"))
        if len(pool) < n_folds:
            raise ValueError(f"Site {site} has fewer development participants than CV folds.")
        pool = rng.permutation(pool)
        assignments[pool] = np.arange(len(pool)) % n_folds
    return assignments


def participant_bootstraps(participants: pd.DataFrame, n_bootstraps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pools = [participants.loc[(participants.site == site) & (participants["split"] == "development"),
                              "participant_index"].to_numpy() for site in sorted(participants.site.unique())]
    return np.asarray([np.concatenate([rng.choice(pool, len(pool), replace=True) for pool in pools])
                       for _ in range(n_bootstraps)])


def scan_indices(scans: pd.DataFrame, people: np.ndarray) -> np.ndarray:
    """Expand draws in order, preserving duplicate draws and the original EO/EC pair."""
    lookup = {p: np.flatnonzero(scans.participant_index.to_numpy() == p)
              for p in np.unique(people)}
    for indices in lookup.values():
        if len(indices) != 2 or set(scans.iloc[indices].state) != {0, 1}:
            raise ValueError("Every participant must have exactly one EO and one EC scan.")
    return np.concatenate([lookup[p] for p in people])


def transform_edges(train: np.ndarray, evaluation: np.ndarray, train_sites: np.ndarray,
                    evaluation_sites: np.ndarray, harmonize: bool
                    ) -> tuple[np.ndarray, np.ndarray, ComBatModel | None]:
    if not harmonize:
        return train, evaluation, None
    # Never preserve the unknown target state. Fit only the specified training rows.
    model = combat_fit(train, sites=train_sites, preserve=None)
    return (combat_apply(model, train, sites=train_sites, preserve=None),
            combat_apply(model, evaluation, sites=evaluation_sites, preserve=None), model)


def fit_classifier(train: np.ndarray, y: np.ndarray, c: float, seed: int
                   ) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(train)
    # Dual liblinear is efficient for the p >> n edge representation.
    model = LogisticRegression(C=c, solver="liblinear", dual=train.shape[1] > train.shape[0],
                               max_iter=5000, tol=1e-6, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(scaler.transform(train), y)
    return scaler, model


def tune(data: PreparedData, splits: pd.DataFrame, n_folds: int, seed: int,
         c_grid: tuple[float, ...]) -> tuple[dict[str, float], pd.DataFrame]:
    folds = splits.cv_fold.to_numpy()[data.scans.participant_index.to_numpy()]
    y = data.scans.state.to_numpy()
    sites = data.scans.site.to_numpy()
    records = []
    for fold in range(n_folds):
        train_indices = np.flatnonzero((folds >= 0) & (folds != fold))
        validation_indices = np.flatnonzero(folds == fold)
        for harmonize in (False, True):
            train_edges, val_edges, _ = transform_edges(
                data.edges[train_indices], data.edges[validation_indices],
                sites[train_indices], sites[validation_indices], harmonize)
            for representation in ("edges",):
                space = data.spaces[representation]
                train, validation = space.transform(train_edges), space.transform(val_edges)
                pipeline = representation + ("_combat" if harmonize else "_raw")
                for c in c_grid:
                    scaler, model = fit_classifier(train, y[train_indices], c, seed)
                    probability = model.predict_proba(scaler.transform(validation))[:, 1]
                    records.append({"pipeline": pipeline, "fold": fold, "C": c,
                                    "n_train_scans": len(train_indices), "n_validation_scans": len(validation_indices),
                                    **scores(y[validation_indices], probability)})
    table = pd.DataFrame(records)
    selected = {}
    for pipeline, group in table.groupby("pipeline"):
        means = group.groupby("C").auc.mean().sort_index()
        # Deterministic: prefer the smaller C in an exact tie.
        selected[pipeline] = float(means.idxmax())
    return selected, table


def save_combat(path: Path, model: ComBatModel) -> None:
    np.savez_compressed(path, site_labels=np.asarray(model.site_labels), alpha=model.alpha,
                        sigma=model.sigma, gamma_star=model.gamma_star, delta_star=model.delta_star,
                        preserve_n_cols=model.preserve_n_cols, eb=model.eb)


def fit_panel(data: PreparedData, train_indices: np.ndarray, test_indices: np.ndarray,
              selected: dict[str, float], seed: int, output: Path
              ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """One participant draw; both classifiers use every individual edge."""
    output.mkdir(parents=True, exist_ok=True)
    sites, y = data.scans.site.to_numpy(), data.scans.state.to_numpy()
    predictions, weights = {}, {}
    for harmonize in (False, True):
        train_edges, test_edges, combat = transform_edges(
            data.edges[train_indices], data.edges[test_indices], sites[train_indices], sites[test_indices], harmonize)
        if combat is not None:
            save_combat(output / "combat.npz", combat)
        for representation in ("edges",):
            space = data.spaces[representation]
            pipeline = representation + ("_combat" if harmonize else "_raw")
            train, test = space.transform(train_edges), space.transform(test_edges)
            scaler, model = fit_classifier(train, y[train_indices], selected[pipeline], seed)
            predictions[pipeline] = model.predict_proba(scaler.transform(test))[:, 1]
            weights[pipeline] = model.coef_[0] / scaler.scale_
            unscaled_intercept = model.intercept_[0] - weights[pipeline] @ scaler.mean_
            np.savez_compressed(output / f"{pipeline}.npz", coef_standardized=model.coef_[0],
                                intercept_standardized=model.intercept_[0], scaler_mean=scaler.mean_,
                                scaler_scale=scaler.scale_, coef_feature_units=weights[pipeline],
                                intercept_feature_units=unscaled_intercept, C=selected[pipeline],
                                train_scan_indices=train_indices, test_scan_indices=test_indices,
                                solver="liblinear", dual=model.dual, n_iter=model.n_iter_)
    return predictions, weights


def metric_rows(probability: np.ndarray, scans: pd.DataFrame, **extra) -> list[dict]:
    records = []
    for site in ["pooled", *sorted(scans.site.unique())]:
        mask = np.ones(len(scans), dtype=bool) if site == "pooled" else scans.site.to_numpy() == site
        records.append({**extra, "site": site, "n_scans": int(mask.sum()),
                        **scores(scans.state.to_numpy()[mask], probability[mask])})
    return records


def comparison_pairs() -> list[tuple[str, str]]:
    return [("edges_combat", "edges_raw")]


def summarize(predictions: dict[str, np.ndarray], weights: dict[str, np.ndarray],
              reference: dict[str, np.ndarray], scans: pd.DataFrame, spaces: dict[str, FeatureSpace],
              test_draws: np.ndarray, output: Path) -> None:
    training_rows, stability_rows, scan_rows, coefficient_rows, test_rows = [], [], [], [], []
    for pipeline, panel in predictions.items():
        for bootstrap, probability in enumerate(panel):
            training_rows += metric_rows(probability, scans, pipeline=pipeline, bootstrap=bootstrap)
        per_scan = pd.concat([scans.reset_index(drop=True), prediction_stability(panel, reference[pipeline])], axis=1)
        per_scan["pipeline"] = pipeline
        scan_rows.append(per_scan)
        for site in ["pooled", *sorted(scans.site.unique())]:
            for state in ("both", "EO", "EC"):
                mask = np.ones(len(scans), dtype=bool) if site == "pooled" else scans.site.to_numpy() == site
                if state != "both":
                    mask &= scans.state.to_numpy() == (state == "EC")
                stability_rows.append({
                    "pipeline": pipeline, "site": site, "state": state, "n_scans": int(mask.sum()),
                    "icc_absolute_single": agreement_icc(panel[:, mask].T),
                    "mean_probability_sd": per_scan.loc[mask, "probability_sd"].mean(),
                    "mean_consensus_agreement": per_scan.loc[mask, "consensus_agreement"].mean(),
                    "mean_pairwise_decision_disagreement": per_scan.loc[mask, "pairwise_decision_disagreement"].mean(),
                    "mean_decision_flip_vs_reference": per_scan.loc[mask, "decision_flip_vs_reference"].mean(),
                })
        correlations = coefficient_correlations(weights[pipeline])
        coefficient_rows.append({"pipeline": pipeline, "n_features": weights[pipeline].shape[1],
                                 "pairwise_pattern_r_mean": np.nanmean(correlations),
                                 "pairwise_pattern_r_q025": np.nanquantile(correlations, 0.025),
                                 "pairwise_pattern_r_q975": np.nanquantile(correlations, 0.975)})
        np.save(output / f"{pipeline}_coefficient_pairwise_r.npy", correlations)
        signs = sign_stability(weights[pipeline])
        signs.insert(0, "feature", spaces[pipeline.rsplit("_", 1)[0]].labels)
        signs.to_csv(output / f"{pipeline}_coefficient_stability.csv", index=False)
        # Test-set uncertainty for the frozen full-development model, no refitting.
        for draw, indices in enumerate(test_draws):
            test_rows += metric_rows(reference[pipeline][indices], scans.iloc[indices].reset_index(drop=True),
                                     pipeline=pipeline, test_bootstrap=draw)
    training = pd.DataFrame(training_rows)
    training.to_csv(output / "training_bootstrap_metrics.csv", index=False)
    test = pd.DataFrame(test_rows)
    test.to_csv(output / "test_participant_bootstrap_metrics.csv", index=False)
    pd.concat(scan_rows).to_csv(output / "scan_prediction_stability.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(output / "prediction_stability.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(output / "coefficient_pattern_stability.csv", index=False)
    for table, label in ((training, "training_bootstrap"), (test, "test_participant_bootstrap")):
        rows = []
        for (pipeline, site), group in table.groupby(["pipeline", "site"]):
            for metric in METRICS:
                values = group[metric].to_numpy()
                rows.append({"pipeline": pipeline, "site": site, "metric": metric, "mean": values.mean(),
                             "sd": values.std(ddof=1), "q025": np.quantile(values, 0.025),
                             "q975": np.quantile(values, 0.975)})
        pd.DataFrame(rows).to_csv(output / f"{label}_summary.csv", index=False)
    paired_differences(training, comparison_pairs(), "bootstrap", "training_resampling").to_csv(
        output / "paired_training_metric_differences.csv", index=False)
    paired_differences(test, comparison_pairs(), "test_bootstrap", "finite_test_participants").to_csv(
        output / "paired_test_metric_differences.csv", index=False)
    # Paired training draws also describe stability contrasts on the same scans.
    differences = []
    per_scan_table = pd.concat(scan_rows)
    for first, second in comparison_pairs():
        a = per_scan_table[per_scan_table.pipeline == first].set_index("scan_id")
        b = per_scan_table[per_scan_table.pipeline == second].set_index("scan_id")
        for metric in ("probability_sd", "pairwise_decision_disagreement", "decision_flip_vs_reference"):
            for site in ("pooled", *sorted(scans.site.unique())):
                mask = np.ones(len(a), dtype=bool) if site == "pooled" else a.site.to_numpy() == site
                differences.append({"first": first, "second": second, "site": site, "metric": metric,
                                    "mean_difference": (a[metric] - b[metric]).to_numpy()[mask].mean()})
    pd.DataFrame(differences).to_csv(output / "paired_prediction_stability_differences.csv", index=False)


def run_experiment(data: PreparedData, output: Path, *, seed: int = 42, n_bootstraps: int = 200,
                   n_test_bootstraps: int = 1000, n_folds: int = 5,
                   test_counts: dict[str, int] | None = None, split_from: Path | None = None,
                   c_grid: tuple[float, ...] = C_GRID, provenance: dict | None = None) -> dict:
    if n_bootstraps < 2 or n_test_bootstraps < 2 or n_folds < 2:
        raise ValueError("At least two training/test bootstraps and CV folds are required.")
    if not c_grid or any(c <= 0 for c in c_grid):
        raise ValueError("C values must be positive.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    splits = split_participants(data.participants, test_counts or {"ihb": 25, "china": 14}, seed, split_from)
    splits["cv_fold"] = development_folds(splits, n_folds, seed + 1)
    splits.to_csv(output / "participant_splits.csv", index=False)
    scans = data.scans.merge(splits[["participant_index", "split", "cv_fold"]], on="participant_index", validate="many_to_one")
    scans.to_csv(output / "scans.csv", index=False)
    data.rois.to_csv(output / "roi_mask_and_groups.csv", index=False)
    data.edge_definitions.to_csv(output / "edge_definitions.csv", index=False)
    definitions = [{"representation": name, "feature_index": i, "label": label, "n_edges": int(space.counts[i])}
                   for name, space in data.spaces.items() for i, label in enumerate(space.labels)]
    pd.DataFrame(definitions).to_csv(output / "feature_definitions.csv", index=False)
    draws = participant_bootstraps(splits, n_bootstraps, seed + 2)
    np.save(output / "training_bootstrap_participant_indices.npy", draws)
    draw_rows = [{"bootstrap": b, "draw": j, "participant_index": int(p),
                  "participant_id": splits.iloc[p].participant_id, "site": splits.iloc[p].site}
                 for b, people in enumerate(draws) for j, p in enumerate(people)]
    pd.DataFrame(draw_rows).to_csv(output / "training_bootstrap_draws.csv", index=False)
    test_indices = scan_indices(data.scans, splits.loc[splits["split"] == "test", "participant_index"].to_numpy())
    test_scans = scans.iloc[test_indices].reset_index(drop=True)
    test_scans.to_csv(output / "test_scans.csv", index=False)
    test_draws = test_participant_draws(test_scans, n_test_bootstraps, seed + 3)
    np.save(output / "test_bootstrap_scan_indices.npy", test_draws)
    config = {**(provenance or {}), "seed": seed, "cv_seed": seed + 1,
              "training_bootstrap_seed": seed + 2, "test_bootstrap_seed": seed + 3,
              "n_bootstraps": n_bootstraps, "n_test_bootstraps": n_test_bootstraps, "n_folds": n_folds,
              "C_grid": list(c_grid), "feature_counts": {n: len(s.labels) for n, s in data.spaces.items()},
              "fitted_pipelines": list(PIPELINES), "fitted_representation": "individual_edges_only",
              "aggregation_stage": "post-fit feature interpretation only",
              "model_feature_count": len(data.spaces["edges"].labels),
              "n_eligible_rois": int(data.rois.eligible.sum()), "state_coding": {"EO": 0, "EC": 1},
              "split_from": str(split_from) if split_from is not None else None,
              "split_counts": {site: group["split"].value_counts().to_dict()
                               for site, group in splits.groupby("site")},
              "combat": {"batch": "cohort", "preserve": None, "fit_unit": "paired scan rows",
                         "stage": "training edges before edge-only classifier fitting", "eb": True,
                         "evaluation": "new participants from sites represented in training"},
              "bootstrap_tuning": "C selected inside development CV and frozen for training bootstrap",
              "coefficient_units": "Fisher-z feature units after ComBat when applicable; scaler undone",
              "software": {"python": platform.python_version(), "numpy": np.__version__,
                           "pandas": pd.__version__, "scipy": scipy.__version__,
                           "sklearn": sklearn_version},
              "status": "running"}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Tuning two edge-only pipelines; edges={config['model_feature_count']}; test people={len(test_scans) // 2}", flush=True)
    selected, cv = tune(data, splits, n_folds, seed, c_grid)
    cv.to_csv(output / "development_cv_scores.csv", index=False)
    (output / "selected_C.json").write_text(json.dumps(selected, indent=2) + "\n")
    train_indices = scan_indices(data.scans, splits.loc[splits["split"] == "development", "participant_index"].to_numpy())
    reference, _ = fit_panel(data, train_indices, test_indices, selected, seed, output / "models" / "reference")
    pd.DataFrame([row for pipeline, probability in reference.items()
                  for row in metric_rows(probability, test_scans, pipeline=pipeline)]).to_csv(
        output / "reference_metrics.csv", index=False)
    reference_predictions = test_scans.copy()
    for pipeline, probability in reference.items():
        reference_predictions[pipeline] = probability
    reference_predictions.to_csv(output / "reference_predictions.csv", index=False)
    predictions = {pipeline: [] for pipeline in selected}
    weights = {pipeline: [] for pipeline in selected}
    prediction_rows = []
    for bootstrap, people in enumerate(draws):
        indices = scan_indices(data.scans, people)
        probabilities, coefficients = fit_panel(data, indices, test_indices, selected, seed,
                                               output / "models" / f"bootstrap_{bootstrap:04d}")
        for pipeline in selected:
            predictions[pipeline].append(probabilities[pipeline])
            weights[pipeline].append(coefficients[pipeline])
            panel = test_scans.copy()
            panel["pipeline"], panel["bootstrap"], panel["probability_closed"] = pipeline, bootstrap, probabilities[pipeline]
            prediction_rows.append(panel)
        if bootstrap == 0 or (bootstrap + 1) % 10 == 0 or bootstrap + 1 == n_bootstraps:
            print(f"Training bootstrap {bootstrap + 1}/{n_bootstraps}; elapsed={time.monotonic() - start:.1f}s", flush=True)
    predictions = {p: np.asarray(v) for p, v in predictions.items()}
    weights = {p: np.asarray(v) for p, v in weights.items()}
    pd.concat(prediction_rows).to_csv(output / "bootstrap_test_predictions.csv", index=False)
    np.savez_compressed(output / "bootstrap_arrays.npz", **predictions,
                        **{p + "_coefficients": w for p, w in weights.items()})
    print("Summarizing training stability and separate test-participant uncertainty", flush=True)
    summarize(predictions, weights, reference, test_scans, data.spaces, test_draws, output)
    config.update(status="complete", selected_C=selected, elapsed_seconds=time.monotonic() - start)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("Summarizing the same fitted edge weights at edge, functional and anatomical levels", flush=True)
    analyze_saved_edge_models(output, output / "feature_stability")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--roi-annotations", type=Path, default=HERE / "roi_groups.csv")
    parser.add_argument("--anatomical-level", choices=("lobe", "gyrus"), default="lobe")
    parser.add_argument("--china-close-run", type=int, choices=(0, 1), default=1)
    parser.add_argument("--bootstraps", type=int, default=200)
    parser.add_argument("--test-bootstraps", type=int, default=1000)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ihb", type=int, default=25)
    parser.add_argument("--test-china", type=int, default=14)
    parser.add_argument("--split-from", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=1, help="BLAS thread limit")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    output = args.output or HERE / "results" / f"edge_models_{args.anatomical_level}_run{args.china_close_run}_b{args.bootstraps}_seed{args.seed}"
    input_paths = [args.roi_annotations, *[args.data_dir / f"{site}_{state}_Schaefer200_strategy-4_GSR.npy"
                                         for site in ("ihb", "china") for state in ("open", "close")],
                   args.data_dir / "schaefer_labels.csv", *[args.data_dir / f"subject_order_{site}.txt"
                                                          for site in ("ihb", "china")]]
    provenance = {"anatomical_level": args.anatomical_level, "china_close_run": args.china_close_run,
                  "preprocessing": "strategy-4_GSR (GSR already applied)",
                  "input_sha256": {str(p.resolve()): sha256(p) for p in input_paths},
                  "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
                      Path(__file__), HERE / "metrics.py", HERE / "feature_stability.py", ROOT / "conninfpy/harmonize.py",
                      ROOT / "examples/open_close/openclose_loader.py")},
                  "command": sys.argv, "threads": args.threads,
                  "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                 text=True, capture_output=True).stdout.strip()}
    with threadpool_limits(limits=args.threads):
        data = load_data(args.data_dir, args.roi_annotations, args.anatomical_level, args.china_close_run)
        config = run_experiment(data, output, seed=args.seed, n_bootstraps=args.bootstraps,
                                n_test_bootstraps=args.test_bootstraps, n_folds=args.cv_folds,
                                test_counts={"ihb": args.test_ihb, "china": args.test_china},
                                split_from=args.split_from, provenance=provenance)
    print(f"Complete in {config['elapsed_seconds']:.1f}s: {output}", flush=True)


if __name__ == "__main__":
    main()
