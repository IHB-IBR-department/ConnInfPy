"""Describe saved Brainnetome results and cross-cohort agreement without inference.

Reads the saved annotation/ML tables and computes ordinary Pearson FC from the
existing Open-Close time series. No permutations, feature selection, or model
training are performed. Pass --output to a supporting-records directory in the
project's Obsidian wiki; review notes do not belong in this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
BNA_DIRS = {"abide": ROOT / "examples/abide/results/brainnetome",
            "openclose": ROOT / "examples/open_close/results/brainnetome"}


def bna(name: str) -> Path:
    """Brainnetome interpretation results folder of an analysis (by its dataset prefix)."""
    return BNA_DIRS["abide" if name.startswith("abide") else "openclose"]
ML = ROOT / "examples/open_close/results/ml"
DATA = ROOT / "datasets/demo/open_close"
NAMES = ("abide_age", "abide_dx", "openclose_ihb", "openclose_china",
         "openclose_china_run0_n47", "openclose_china_run1")


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    return {"n": int(keep.sum()), "pearson": float(np.corrcoef(a[keep], b[keep])[0, 1]),
            "spearman": float(spearmanr(a[keep], b[keep]).statistic),
            "same_sign": int((np.sign(a[keep]) == np.sign(b[keep])).sum())}


def compare_detections(a, b):
    def ids(e, tail):
        return set(e.loc[e["tail"] == tail, "edge_id"].astype(int))
    out = {}
    for tail in ("positive", "negative"):
        x, y = ids(a, tail), ids(b, tail)
        out[tail] = {"n_a": len(x), "n_b": len(y), "intersection": len(x & y),
                     "dice": 2 * len(x & y) / (len(x) + len(y)),
                     "jaccard": len(x & y) / len(x | y)}
    out["opposite_direction"] = len(ids(a, "positive") & ids(b, "negative")) + len(ids(a, "negative") & ids(b, "positive"))
    return out


def fc(ts):
    """Replicate loader's Pearson correlation, clipping, and Fisher transform."""
    r = np.stack([np.corrcoef(t.T) for t in ts])
    idx = np.arange(r.shape[-1])
    r[:, idx, idx] = 0
    r = np.clip(r, -0.9999, 0.9999)
    return r, np.arctanh(r)


def analyze_results(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    tracked = []
    def read(path):
        tracked.append(path)
        return pd.read_csv(path)
    edges, overview, network_counts = {}, [], []
    for name in NAMES:
        e = read(bna(name) / name / "connections_annotated.csv")
        edges[name] = e
        relations = read(bna(name) / name / "relations_by_tail.csv")
        n_tested = int(relations.loc[relations.relation == "network_relation", "n_tested"].sum())
        assert not e.edge_id.duplicated().any()
        assert not e[["roi_i", "roi_j"]].duplicated().any()
        assert (e.roi_i < e.roi_j).all()
        assert (e.p_min <= 0.05).all()
        assert np.isfinite(e[["effect", "t_signed", "delta_r"]].to_numpy()).all()
        assert ((e.effect > 0) == (e["tail"] == "positive")).all()
        for tail in ("positive", "negative"):
            assert int((e["tail"] == tail).sum()) == int(relations.loc[relations.relation == "network_relation", f"n_{tail}"].sum())
        rois = read(bna(name) / ("rois_abide_schaefer100.csv" if name.startswith("abide") else "rois_openclose_schaefer200.csv"))
        assert (e.roi_i < len(rois)).all() and (e.roi_j < len(rois)).all()
        assert (e.i_roi.to_numpy() == rois.roi.iloc[e.roi_i].to_numpy()).all()
        assert (e.j_roi.to_numpy() == rois.roi.iloc[e.roi_j].to_numpy()).all()
        domains = read(bna(name) / name / "brainmap_domains.csv")
        qc = read(bna(name) / name / "distance_qc.csv")
        overview.append({"analysis": name, "n_tested": n_tested, "n_detected": len(e),
                         "positive": int((e["tail"] == "positive").sum()),
                         "negative": int((e["tail"] == "negative").sum()),
                         "endpoint_rois": len(set(e.roi_i) | set(e.roi_j)),
                         "max_abs_domain_share_difference": float((domains.share_detected_endpoint_rois - domains.share_all_rois).abs().max()),
                         **qc.iloc[0].to_dict()})
        for pair, g in e.groupby("network_relation"):
            opportunities = int(relations.loc[(relations.relation == "network_relation") & (relations.category == pair), "n_tested"].iloc[0])
            network_counts.append({"analysis": name, "network_pair": pair, "n_tested": opportunities,
                                   "n_positive": int((g["tail"] == "positive").sum()),
                                   "n_negative": int((g["tail"] == "negative").sum()),
                                   "mean_selected_delta_r": float(g.delta_r.mean())})
    pd.DataFrame(overview).to_csv(output / "analysis_overview.csv", index=False)
    pd.DataFrame(network_counts).to_csv(output / "detected_network_pairs.csv", index=False)
    comparisons = {}
    for a, b in [("openclose_ihb", "openclose_china"), ("openclose_ihb", "openclose_china_run0_n47"),
                 ("openclose_ihb", "openclose_china_run1"), ("openclose_china_run0_n47", "openclose_china_run1")]:
        comparisons[f"{a} vs {b}"] = compare_detections(edges[a], edges[b])

    # Compute unselected effects over all 16,471 aligned edge opportunities.
    labels = read(DATA / "schaefer_labels.csv")
    assert labels.id.is_unique and len(labels) == 182
    keep = labels.id.to_numpy(int) - 1
    iu = np.triu_indices(len(keep), 1)
    networks = labels.network.to_numpy()
    pair = np.array([" / ".join(sorted((networks[i], networks[j]))) for i, j in zip(*iu, strict=True)])
    states = {}
    for cohort in ("ihb", "china"):
        op = DATA / f"{cohort}_open_Schaefer200_strategy-4_GSR.npy"
        cl = DATA / f"{cohort}_close_Schaefer200_strategy-4_GSR.npy"
        tracked += [op, cl]
        o, c = np.load(op, mmap_mode="r"), np.load(cl, mmap_mode="r")
        if cohort == "china":
            subject_path = DATA / "subject_order_china.txt"
            tracked.append(subject_path)
            ids = subject_path.read_text().splitlines()
            subject_keep = np.array([s.strip() != "sub-3258811" for s in ids])
            assert subject_keep.sum() == 47
            o, c = o[subject_keep], c[subject_keep]
        r_open, z_open = fc(o[..., keep])
        for run in range(2 if cohort == "china" else 1):
            r_closed, z_closed = fc((c[..., run] if cohort == "china" else c)[..., keep])
            d = z_closed[:, iu[0], iu[1]] - z_open[:, iu[0], iu[1]]
            assert np.isfinite(d).all()
            states[f"{cohort}_run{run}"] = {"n_subjects": len(d), "delta_z": d.mean(axis=0),
                                            "t": d.mean(axis=0) / (d.std(axis=0, ddof=1) / np.sqrt(len(d))),
                                            "delta_r": (r_closed - r_open)[:, iu[0], iu[1]].mean(axis=0)}
            annotation_name = "openclose_ihb" if cohort == "ihb" else f"openclose_china_run{run}" + ("_n47" if run == 0 else "")
            detected = edges[annotation_name]
            state = states[f"{cohort}_run{run}"]
            assert np.allclose(detected.effect, state["delta_z"][detected.edge_id], atol=1e-12)
            assert np.allclose(detected.t_signed, state["t"][detected.edge_id], atol=1e-10)
            # Exported fc[...] values are Fisher-z means, not Pearson r.
            assert np.allclose(detected["fc[eyes open]"], z_open.mean(axis=0)[detected.roi_i, detected.roi_j], atol=1e-12)
            assert np.allclose(detected["fc[eyes closed]"], z_closed.mean(axis=0)[detected.roi_i, detected.roi_j], atol=1e-12)
            assert np.allclose(detected.delta_r, np.tanh(detected["fc[eyes closed]"]) - np.tanh(detected["fc[eyes open]"]), atol=1e-12)
    block_rows = []
    for category in np.unique(pair):
        mask = pair == category
        row = {"network_pair": category, "n_tested": int(mask.sum())}
        for name, state in states.items():
            for metric in ("delta_z", "delta_r", "t"):
                row[f"{name}_{metric}"] = float(state[metric][mask].mean())
        block_rows.append(row)
    blocks = pd.DataFrame(block_rows)
    blocks.to_csv(output / "all_edge_network_effects.csv", index=False)
    all_comparisons = {}
    for a, b in [("ihb_run0", "china_run0"), ("ihb_run0", "china_run1"), ("china_run0", "china_run1")]:
        all_comparisons[f"{a} vs {b}"] = {metric: {"edges": correlation(states[a][metric], states[b][metric]),
                                                  "blocks": correlation(blocks[f"{a}_{metric}"], blocks[f"{b}_{metric}"])}
                                              for metric in ("delta_z", "delta_r", "t")}
    # Fresh ML table is controlling evidence; historical derived files are checked.
    ml = read(ML / "ml_feature_selection.csv")
    key = ["direction", "kind", "selector", "alpha"]
    assert not ml.loc[ml.kind == "selector", key].duplicated().any()
    assert ml.auc.dropna().between(0, 1).all()
    assert ml.n_edges_used.between(0, len(pair)).all()
    ml_rows = []
    for row_idx, row in ml[(ml.kind == "selector") & (ml.alpha == 0.05)].iterrows():
        baseline = ml[(ml.direction == row.direction) & (ml.selector == "all_edges")].iloc[0]
        # Matching baseline records immediately follow each selector record.
        # Equal k across selectors can repeat baseline keys, especially random
        # draws, so an arbitrary first key match would use the wrong baseline.
        random, uni = ml.iloc[row_idx + 1], ml.iloc[row_idx + 2]
        for record, prefix in [(random, "random_matched_k"), (uni, "uni_topk")]:
            assert record.selector == f"{prefix}{int(row.n_edges_used)}"
            assert record.direction == row.direction and record.alpha == row.alpha
        ml_rows.append({"direction": row.direction, "selector": row.selector, "alpha": row.alpha,
                        "n_edges": int(row.n_edges_used), "auc": row.auc, "all_auc": baseline.auc,
                        "univariate_auc": uni.auc, "random_auc": random.auc,
                        "delta_vs_all": row.auc - baseline.auc, "delta_vs_univariate": row.auc - uni.auc,
                        "fraction_features_removed": 1 - row.n_edges_used / len(pair)})
    pd.DataFrame(ml_rows).to_csv(output / "current_ml_comparison.csv", index=False)
    fresh = ml[ml.kind == "selector"].groupby(["selector", "alpha"]).agg(auc=("auc", "mean"), n_edges_used=("n_edges_used", "mean"))
    old = read(ML / "average_cross_site_auc.csv").set_index(["selector", "alpha"])
    joined = old.join(fresh, rsuffix="_current")
    joined.to_csv(output / "ml_historical_summary_check.csv")
    timestamps = {p.name: datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat() for p in ML.glob("*.csv")}
    retest = []
    for path in sorted(ML.glob("pmap_china_retest_*.npz")):
        tracked.append(path)
        with np.load(path) as maps:
            counts = {}
            for tail, legacy in [("positive", "g2>g1"), ("negative", "g1>g2")]:
                p = maps[tail if tail in maps else legacy]
                assert p.shape == (182, 182)
                counts[tail] = int((p[iu] <= 0.05).sum())
            retest.append({"method": path.stem.removeprefix("pmap_china_retest_"), **counts})
    pd.DataFrame(retest).to_csv(output / "empirical_retest_control.csv", index=False)
    result = {"overview": overview, "detection_comparisons": comparisons,
              "unselected_comparisons": all_comparisons, "ml_alpha_005": ml_rows,
              "empirical_retest_control": retest,
              "ml_summary_max_abs_auc_difference": float((joined.auc - joined.auc_current).abs().max()),
              "ml_file_modified_utc": timestamps,
              "notes": ["Correlations across dependent edges/blocks are descriptive; no correlation p-values.",
                        "China run comparisons share subjects and the same eyes-open reference.",
                        "Sign concordance counts all nonzero estimated effects, irrespective of significance.",
                        "delta_r averages subject-level Pearson differences; annotation baselines are tanh of mean Fisher z.",
                        "Per-tail alpha is 0.05; a union of tails is not a joint 0.05 directional family."]}
    paths = sorted(set(tracked + [Path(__file__)]))
    result["sources"] = [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    (output / "checks.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_results(args.output)
    print(json.dumps({k: result[k] for k in ("detection_comparisons", "unselected_comparisons", "ml_summary_max_abs_auc_difference")}, indent=2))
