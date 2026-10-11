"""Edge-model feature-stability orchestration; see README.md."""
from __future__ import annotations


import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy
from sklearn import __version__ as sklearn_version
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]

from examples.open_close._shared.data import DATA_DIR  # noqa: E402
from examples.open_close.feature_stability.metrics import (  # noqa: E402
    METRICS, agreement_icc, coefficient_correlations, paired_differences,
    prediction_stability, scores, sign_stability, test_participant_draws,
)
from examples.open_close.feature_stability.feature_stability import analyze_saved_edge_models  # noqa: E402

HERE = Path(__file__).resolve().parent
C_GRID = (0.01, 0.1, 1.0, 10.0)
REPRESENTATIONS = ("edges", "functional", "anatomical")
PIPELINES = ("edges_raw", "edges_combat")

from examples._shared.paths import sha256
from examples.open_close.feature_stability.data import FeatureSpace, PreparedData, load_data
from examples.open_close.feature_stability.resampling import split_participants, development_folds, participant_bootstraps, scan_indices
from examples.open_close.feature_stability.fitting import tune, fit_panel


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
                      Path(__file__), HERE / "data.py", HERE / "resampling.py", HERE / "fitting.py", HERE / "metrics.py", HERE / "feature_stability.py", ROOT / "conninfpy/harmonize/combat.py", ROOT / "conninfpy/harmonize/_arrays.py",
                      ROOT / "examples/open_close/_shared/data.py")},
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
