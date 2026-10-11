"""Post-fit reproducibility of the SAME edge-model features at several levels.

No classifier is fitted here, and no scan FC is aggregated before prediction.
Importance is a conditional logit coefficient for one fixed development SD.
"""
from __future__ import annotations


from pathlib import Path as _Path
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import gammaln
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
from examples.open_close.feature_stability.metrics import coefficient_correlations, sign_stability  # noqa: E402

TOP_FRACTIONS = (0.05, 0.10, 0.20)


@dataclass
class Grouping:
    name: str
    labels: list[str]
    counts: np.ndarray
    membership: np.ndarray | None
    projection: csr_matrix | None

    def mean(self, edge_values: np.ndarray) -> np.ndarray:
        return edge_values if self.projection is None else np.asarray(edge_values @ self.projection)


def load_groupings(source: Path) -> dict[str, Grouping]:
    definitions = pd.read_csv(source / "feature_definitions.csv")
    edges = pd.read_csv(source / "edge_definitions.csv")
    groupings = {}
    for name in ("edges", "functional", "anatomical"):
        table = definitions[definitions.representation == name].sort_values("feature_index")
        if not np.array_equal(table.feature_index, np.arange(len(table))):
            raise ValueError(f"Noncontiguous {name} feature definitions.")
        counts = table.n_edges.to_numpy(dtype=int)
        if name == "edges":
            if len(table) != len(edges):
                raise ValueError("Saved edge axis is inconsistent.")
            membership, projection = None, None
        else:
            membership = edges[name + "_feature_index"].to_numpy(dtype=int)
            if not np.array_equal(np.bincount(membership, minlength=len(counts)), counts):
                raise ValueError(f"Saved {name} membership/counts are inconsistent.")
            projection = csr_matrix((1 / counts[membership], (np.arange(len(edges)), membership)),
                                    shape=(len(edges), len(table)))
        groupings[name] = Grouping(name, table.label.tolist(), counts, membership, projection)
    return groupings


def original_edge_weights(weights: np.ndarray, delta: np.ndarray | None = None) -> np.ndarray:
    """Undo scaler upstream, then ComBat's per-site slope (delta is an SD)."""
    if delta is None:
        return weights
    if weights.shape != delta.shape or not np.isfinite(delta).all() or np.any(delta <= 0):
        raise ValueError("ComBat deltas must be finite, positive and match the edge weights.")
    return weights / delta


def top_mask(importance: np.ndarray, fraction: float) -> tuple[np.ndarray, int]:
    if not 0 < fraction < 1 or not np.isfinite(importance).all() or np.any(importance < 0):
        raise ValueError("Use finite nonnegative importance and a top fraction between zero and one.")
    k = int(np.ceil(importance.shape[1] * fraction))
    # Stable ties by feature index. Report tied cutoffs; deterministic ties are not evidence.
    order = np.argsort(-importance, axis=1, kind="stable")[:, :k]
    selected = np.zeros_like(importance, dtype=bool)
    np.put_along_axis(selected, order, True, axis=1)
    return selected, k


def overlaps(selected: np.ndarray, signs: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Pairwise set overlap. Kuncheva applies only to equal-size subsets."""
    selected = selected.astype(bool)
    n, p = selected.shape
    a, b = np.triu_indices(n, 1)
    indicator = selected.astype(float)
    intersection = (indicator @ indicator.T)[a, b]
    sizes = selected.sum(axis=1)
    sum_sizes = sizes[a] + sizes[b]
    union = sum_sizes - intersection
    result = {
        "dice": np.divide(2 * intersection, sum_sizes, out=np.full_like(intersection, np.nan), where=sum_sizes > 0),
        "jaccard": np.divide(intersection, union, out=np.full_like(intersection, np.nan), where=union > 0),
    }
    if np.all(sizes == sizes[0]) and 0 < sizes[0] < p:
        k = sizes[0]
        result["kuncheva"] = (intersection - k * k / p) / (k - k * k / p)
    else:
        result["kuncheva"] = np.full_like(intersection, np.nan)
    if signs is not None:
        same = np.zeros_like(intersection)
        for direction in (-1, 1):
            signed = (selected & (np.sign(signs) == direction)).astype(float)
            same += (signed @ signed.T)[a, b]
        result["sign_concordant_dice"] = np.divide(2 * same, sum_sizes, out=np.full_like(same, np.nan), where=sum_sizes > 0)
    return result


def presence_null(n_edges: int, counts: np.ndarray, k: int) -> dict[str, np.ndarray | float]:
    """Uniform-k-edge null: group-hit probabilities account for block capacities.

    The Jaccard value is a ratio of expected counts, not E[Jaccard]. It flags
    mechanical saturation when nearly every coarse block contains a top edge.
    """
    remaining = n_edges - counts
    can_miss = remaining >= k
    miss = np.zeros(len(counts))
    log_denominator = gammaln(n_edges + 1) - gammaln(k + 1) - gammaln(n_edges - k + 1)
    miss[can_miss] = np.exp(gammaln(remaining[can_miss] + 1) - gammaln(k + 1)
                           - gammaln(remaining[can_miss] - k + 1) - log_denominator)
    hit = np.clip(1 - miss, 0, 1)
    return {"hit_probability": hit, "mean_coverage": float(hit.mean()),
            "intersection_over_union_of_expectations": float(np.sum(hit**2) / np.sum(2 * hit - hit**2))}


def distribution(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {prefix + s: float("nan") for s in ("_mean", "_q025", "_q975")}
    return {prefix + "_mean": float(finite.mean()), prefix + "_q025": float(np.quantile(finite, 0.025)),
            prefix + "_q975": float(np.quantile(finite, 0.975))}


def analyze_patterns(edge_effects: np.ndarray, groupings: dict[str, Grouping],
                     *, pipeline: str, site: str, output: Path,
                     fractions: tuple[float, ...] = TOP_FRACTIONS) -> tuple[list[dict], list[dict], list[dict]]:
    """Aggregate learned effects only: signed mean, absolute density, total mass."""
    pattern_rows, overlap_rows, projection_rows = [], [], []
    n = len(edge_effects)
    pair_i, pair_j = np.triu_indices(n, 1)
    edge_importance = np.abs(edge_effects)
    edge_selections = {fraction: top_mask(edge_importance, fraction) for fraction in fractions}
    for name, grouping in groupings.items():
        signed = grouping.mean(edge_effects)
        importance = grouping.mean(edge_importance)  # Mean |edge effect|, not |mean edge effect|.
        total_mass = importance * grouping.counts
        ranks = rankdata(importance, axis=1, method="average")
        arrays = {"bootstrap_i": pair_i, "bootstrap_j": pair_j,
                  "signed_pattern_r": coefficient_correlations(signed),
                  "importance_pattern_r": coefficient_correlations(importance),
                  "importance_rank_r": coefficient_correlations(ranks)}
        common = {"pipeline": pipeline, "site": site, "level": name,
                  "n_features": len(grouping.labels), "n_model_edges": edge_effects.shape[1],
                  "n_models": n, "aggregation": "post_fit_same_edge_model"}
        pattern_rows.append({**common, **{key: value for metric, values in arrays.items()
                                        if metric.endswith("_r") for key, value in distribution(values, metric).items()}})
        features = sign_stability(signed)
        features.insert(0, "feature", grouping.labels)
        features.insert(1, "n_edges", grouping.counts)
        features["importance_mean"] = importance.mean(axis=0)
        features["importance_sd"] = importance.std(axis=0, ddof=1)
        features["total_absolute_mass_mean"] = total_mass.mean(axis=0)
        features["importance_rank_mean"] = ranks.mean(axis=0)
        for fraction in fractions:
            token = f"top_{int(round(100 * fraction)):02d}pct"
            selected, k = top_mask(importance, fraction)
            cutoff = np.min(np.where(selected, importance, np.inf), axis=1)
            tied = (importance == cutoff[:, None]).sum(axis=1)
            included_ties = ((importance == cutoff[:, None]) & selected).sum(axis=1)
            values = overlaps(selected, signed)
            for metric, value in values.items():
                arrays[token + "_" + metric] = value
            features[token + "_selection_frequency"] = selected.mean(axis=0)
            overlap_rows.append({**common, "top_fraction": fraction, "top_k": k,
                                 "realized_top_fraction": k / len(grouping.labels),
                                 "cutoff_tie_models_fraction": float((tied > included_ties).mean()),
                                 **{key: value for metric, values_ in values.items()
                                    for key, value in distribution(values_, metric).items()}})
            if grouping.membership is not None:
                edge_selected, edge_k = edge_selections[fraction]
                density = grouping.mean(edge_selected.astype(float))
                presence = density > 0
                projected = overlaps(presence)
                null = presence_null(edge_effects.shape[1], grouping.counts, edge_k)
                arrays[token + "_projected_presence_jaccard"] = projected["jaccard"]
                arrays[token + "_projected_density_r"] = coefficient_correlations(density)
                features[token + "_projected_edge_selection_density"] = density.mean(axis=0)
                features[token + "_projected_presence_frequency"] = presence.mean(axis=0)
                features[token + "_null_presence_probability"] = null["hit_probability"]
                projection_rows.append({**common, "top_edge_fraction": fraction, "n_selected_edges": edge_k,
                                        "mean_group_coverage": float(presence.mean()),
                                        "null_mean_group_coverage": null["mean_coverage"],
                                        "null_ratio_expected_intersection_union": null["intersection_over_union_of_expectations"],
                                        **distribution(projected["jaccard"], "presence_jaccard"),
                                        **distribution(arrays[token + "_projected_density_r"], "density_pattern_r")})
        prefix = f"{pipeline}__{site}__{name}"
        features.to_csv(output / f"{prefix}_features.csv", index=False)
        np.savez_compressed(output / f"{prefix}_pairwise.npz", **arrays)
        if grouping.membership is not None:
            np.savez_compressed(output / f"{prefix}_bootstrap_patterns.npz", signed_mean_effect=signed,
                                mean_absolute_effect=importance, total_absolute_mass=total_mass)
    return pattern_rows, overlap_rows, projection_rows


def analyze_saved_edge_models(source: Path, output: Path,
                              fractions: tuple[float, ...] = TOP_FRACTIONS) -> dict:
    """Reuse only edge fits, even if a legacy run also contains aggregate fits."""
    config = json.loads((source / "config.json").read_text())
    if config["status"] != "complete" or config["n_bootstraps"] < 2:
        raise ValueError("Feature analysis requires a completed run with at least two bootstrap edge models.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Feature output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    groupings = load_groupings(source)
    reference = np.load(source / "models/reference/edges_raw.npz")
    reference_scale = reference["scaler_scale"]
    p = len(groupings["edges"].labels)
    if reference_scale.shape != (p,) or np.any(reference_scale <= 0):
        raise ValueError("Reference raw development scale does not match the saved edge axis.")
    np.save(output / "fixed_development_edge_scale.npy", reference_scale)
    splits = pd.read_csv(source / "participant_splits.csv")
    sites = sorted(splits.site.unique())
    n = config["n_bootstraps"]
    weights = {pipeline: [] for pipeline in ("edges_raw", "edges_combat")}
    deltas = {site: [] for site in sites}
    digests = {}
    for b in range(n):
        folder = source / "models" / f"bootstrap_{b:04d}"
        train_indices = None
        for pipeline in weights:
            path = folder / f"{pipeline}.npz"
            state = np.load(path)
            coefficient = state["coef_feature_units"]
            if coefficient.shape != (p,) or not np.isfinite(coefficient).all():
                raise ValueError("Every fitted model must have the complete individual-edge axis.")
            if train_indices is not None and not np.array_equal(train_indices, state["train_scan_indices"]):
                raise ValueError("Raw and ComBat models used different participant draws.")
            train_indices = state["train_scan_indices"]
            weights[pipeline].append(coefficient)
            digests[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
        combat_path = folder / "combat.npz"
        combat = np.load(combat_path)
        if int(combat["preserve_n_cols"]) != 0:
            raise ValueError("ComBat must not preserve the prediction target.")
        labels = combat["site_labels"].tolist()
        for site in sites:
            deltas[site].append(combat["delta_star"][labels.index(site)])
        digests[str(combat_path.relative_to(source))] = hashlib.sha256(combat_path.read_bytes()).hexdigest()
    pattern_rows, overlap_rows, projection_rows = [], [], []
    for pipeline, values in weights.items():
        values = np.asarray(values)
        for site in sites:
            delta = np.asarray(deltas[site]) if pipeline == "edges_combat" else None
            effects = original_edge_weights(values, delta) * reference_scale
            rows = analyze_patterns(effects, groupings, pipeline=pipeline, site=site, output=output, fractions=fractions)
            pattern_rows.extend(rows[0]); overlap_rows.extend(rows[1]); projection_rows.extend(rows[2])
    pd.DataFrame(pattern_rows).to_csv(output / "feature_pattern_stability.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(output / "feature_overlap_summary.csv", index=False)
    pd.DataFrame(projection_rows).to_csv(output / "projected_edge_overlap_summary.csv", index=False)
    manifest = {"status": "complete", "source_run": str(source.resolve()), "n_bootstraps": n,
                "fitted_pipelines_used": ["edges_raw", "edges_combat"], "new_models_fitted": 0,
                "anatomical_level": config.get("anatomical_level", "unspecified"),
                "aggregation": "post_fit_same_edge_model", "top_fractions": list(fractions),
                "n_features": {name: len(grouping.labels) for name, grouping in groupings.items()},
                "importance": "mean absolute coefficient for one fixed original-development edge SD",
                "signed_pattern": "mean signed coefficient for one fixed original-development edge SD",
                "combat_coefficients": "undo scaler and per-site ComBat delta; original Fisher-z units",
                "reference_scale": "raw full-development scaler; fixed across models and sites; no test data",
                "model_state_sha256": digests,
                "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "analysis_config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completed run containing saved edge models")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    with threadpool_limits(limits=args.threads):
        config = analyze_saved_edge_models(args.input, args.output)
    print(f"Post-fit feature stability complete: {config['n_bootstraps']} shared edge fits; {args.output}")


if __name__ == "__main__":
    main()
