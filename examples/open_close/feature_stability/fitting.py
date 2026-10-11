"""Training-only ComBat, scaling and edge classifier fits."""
from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from conninfpy.harmonize import ComBatModel, combat_apply, combat_fit
from examples.open_close.feature_stability.data import PreparedData
from examples.open_close.feature_stability.metrics import scores


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
