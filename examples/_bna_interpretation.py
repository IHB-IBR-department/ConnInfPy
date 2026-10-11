"""Shared steps of the Brainnetome interpretation (used by ``abide/`` and ``open_close/``).

The entry scripts are ``abide/brainnetome_interpretation.py`` and
``open_close/brainnetome_interpretation.py``. Each re-runs its dataset's paper
analyses with their own settings (5,000 permutations, seed 42, TFNBS E = 0.4,
H = 3, n = 10), checks the p-value maps against the stored ones, and describes
the ROIs and the detected connections with the Brainnetome atlas (BNA-246) via
``conninfpy.annotate``:

* ROIs by the tissue they sampled (the Schaefer parcel's overlap with BNA
  regions), assigned at the finest level holding >= 50 % of the parcel;
* connections by hemispheric relation, anatomical level, circuit class,
  distance, population structural connectivity and Yeo relation;
* direction against baseline coupling, from the coefficient maps (stronger /
  weaker positive coupling, stronger / weaker anticorrelation, sign change;
  both datasets used global-signal regression, which induces part of the
  anticorrelations);
* BrainMap behavioural domains of the endpoint regions (forward-inference
  literature context, not functions of the connections);
* Brainnetome-style connectograms (``connectogram_{gyrus,subregion}.{png,pdf}``),
  one panel per tail, chords coloured by the endpoints' Yeo networks.

Everything is descriptive: no test is added and the correction is unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conninfpy.annotate import summarize_annotation  # noqa: E402
from conninfpy.brainnetome import BrainnetomeReference  # noqa: E402
from conninfpy.interpret.evidence import build_annotation_evidence, describe_coverage  # noqa: E402

SCHAEFER = Path.home() / "nilearn_data" / "schaefer_2018"
TFNBS = dict(method="tfnbs", e=0.4, h=3.0, n=10, acceleration=None, rng=42, use_mp=True)


def reference() -> BrainnetomeReference:
    ref = BrainnetomeReference.load(data_dir=ROOT / "datasets" / "atlases")
    print("reference:", json.dumps({k: v for k, v in ref.provenance.items() if k != "profiles"}, default=str))
    return ref


def with_source_networks(rois, networks):
    """Network relations use the source atlas's own Yeo labels (authoritative for
    Schaefer parcels); the Brainnetome crosswalk is kept as ``yeo7_bna``."""
    rois.table["yeo7_bna"] = rois.table["yeo7"]
    rois.table["yeo7"] = [str(x) for x in networks]
    return rois


def reproduction(stored, inf) -> str:
    if stored is None or not Path(stored).exists():
        return "no stored map"
    s = np.load(stored)
    d = max(float(np.nanmax(np.abs(np.asarray(inf[t]) - s[t]))) for t in ("positive", "negative"))
    iu = np.triu_indices(s["positive"].shape[0], 1)
    same = all(np.array_equal(np.asarray(inf[t])[iu] < 0.05, s[t][iu] < 0.05) for t in ("positive", "negative"))
    return f"max |Δp| = {d:.3g}; same detections at α = 0.05: {same}"


def connectograms(name, inf, rois, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from conninfpy.plot import plot_connectogram

    for level in ("gyrus", "subregion"):
        fig, axes = plt.subplots(1, 2, figsize=(18, 9.2))
        for ax, tail in zip(axes, ("positive", "negative")):
            plot_connectogram(inf, rois, tail=tail, level=level, labels=level, ax=ax)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.04)
        fig.suptitle(f"{name}: Brainnetome {level} connectogram (α = 0.05 per tail)", fontsize=13)
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / name / f"connectogram_{level}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)


def describe(name, out, rois, labels, dataset, out_dir: Path):
    inf = out.inference
    if labels and inf.design is not None:
        prof = inf.design["profiles"]
        for old, new in labels.items():
            if old in prof:
                prof[new] = prof.pop(old)
        inf.design["comparison"] = [labels.get(x, x) for x in inf.design["comparison"]]
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    edges = inf.annotated_edges(rois, alpha=0.05)
    edges.to_csv(d / "connections_annotated.csv", index=False)
    summaries = {}
    for level in ("lobe", "network"):
        s = summarize_annotation(inf, rois, alpha=0.05, level=level)
        s["regions"].to_csv(d / f"groups_{level}.csv", index=False)
        summaries[level] = s
    s = summaries["lobe"]
    s["relations"].to_csv(d / "relations.csv", index=False)
    s["domains"].to_csv(d / "brainmap_domains.csv", index=False)
    s["distance_qc"].to_csv(d / "distance_qc.csv", index=False)
    by_tail = []
    for tail in ("positive", "negative"):
        sel = edges[edges[f"p_{tail}"] <= 0.05]
        counts = sel["coupling_change"].value_counts() if "coupling_change" in sel else pd.Series(dtype=int)
        for cls, k in counts.items():
            by_tail.append({"tail": tail, "coupling_change": cls, "n": int(k), "share": k / len(sel)})
    pd.DataFrame(by_tail).to_csv(d / "coupling_by_tail.csv", index=False)
    tested = s["edges"]
    tail_rows = []
    for col in ("circuit", "hemispheric_relation", "distance_band", "network_relation", "anatomical_relation"):
        for cat, g in tested.groupby(col):
            tail_rows.append({"relation": col, "category": cat, "n_tested": len(g),
                              "share_of_tested": len(g) / len(tested),
                              "n_positive": int((g["p_positive"] <= 0.05).sum()),
                              "n_negative": int((g["p_negative"] <= 0.05).sum())})
    by_rel = pd.DataFrame(tail_rows)
    for tail in ("positive", "negative"):
        tot = by_rel.groupby("relation")[f"n_{tail}"].transform("sum")
        by_rel[f"share_{tail}"] = np.where(tot > 0, by_rel[f"n_{tail}"] / tot, np.nan)
    by_rel.to_csv(d / "relations_by_tail.csv", index=False)
    s["by_tail"] = by_rel
    evidence = build_annotation_evidence(edges, rois, contrast_name=name, source_atlas=dataset,
                                         inference_unit=inf.inference_unit, summary=s)
    (d / "evidence.json").write_text(json.dumps(evidence, indent=1, default=str))
    sentence = describe_coverage(evidence)["report_sentence"] if evidence["rois"] else "no detections"
    return edges, s, pd.DataFrame(by_tail), sentence


def md_table(df, cols=None, n=None, fmt="{:.3f}"):
    df = df if cols is None else df[cols]
    df = df if n is None else df.head(n)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    rows = ["| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |" for _, r in df.iterrows()]
    return head + "\n".join(rows) + "\n"


def report_header(title: str, script: str, n_perm: int) -> str:
    return (f"# {title}\n\nGenerated by `{script}` ({n_perm} permutations, TFNBS E = 0.4, H = 3; α = 0.05 per "
            "tail). Descriptive: no test is added and the correction is unchanged. The data used global-signal "
            "regression, which induces part of the anticorrelations. BrainMap domains are forward-inference "
            "literature context of the endpoint regions, not functions of the connections.\n")


def report_section(name, dataset, repro, sentence, s, by_tail, inf) -> str:
    out = [f"\n## {name}\n\n{dataset}. Reproduction against the stored map: {repro}.\n\n{sentence}\n"
           f"\nConnectograms: `{name}/connectogram_gyrus.pdf`, `{name}/connectogram_subregion.pdf`.\n"]
    rel = s["by_tail"]
    for col, title in (("circuit", "Circuit class"), ("hemispheric_relation", "Hemispheric relation"),
                       ("anatomical_relation", "Anatomical level"), ("distance_band", "Distance"),
                       ("network_relation", "Yeo relation (source labels)")):
        t = rel[rel.relation == col][["category", "n_positive", "share_positive", "n_negative",
                                      "share_negative", "share_of_tested"]].sort_values("share_of_tested", ascending=False)
        out.append(f"\n**{title}**: detections per tail vs tested opportunities\n\n" + md_table(t))
    if not by_tail.empty:
        out.append("\n**Coupling change of the detected connections** (from the coefficient maps; "
                   f"comparison {inf.design['comparison'][0]} → {inf.design['comparison'][1]})\n\n"
                   + md_table(by_tail.sort_values(["tail", "n"], ascending=[True, False])))
    g = s["regions"].sort_values("n_detected", ascending=False)
    out.append("\n**Lobe groups** (ROIs not ≥ 50 % in one lobe are 'mixed')\n\n" + md_table(g, n=12))
    out.append("\n**BrainMap domains** (unique ROIs; overlapping memberships)\n\n" + md_table(s["domains"]))
    out.append("\n**Distance QC**\n\n" + md_table(s["distance_qc"]))
    return "".join(out)


def roi_section(key, rois) -> str:
    lv = rois.table["level"].value_counts().to_dict()
    return (f"\n## ROI annotation: {key}\n\nAssignment level of the {len(rois)} parcels: {lv}. "
            f"Median subregion purity {rois.table['region_purity'].median():.2f}.\n")
