"""Brainnetome interpretation of an inference result in one call.

:func:`interpret_result` takes a result and its :class:`~conninfpy.interpret.annotate.RoiAnnotation`
and returns an :class:`InterpretationReport`: the detected connections with
their endpoint anatomy and relations, detections against tested opportunities
by relation and by lobe / network group, coupling change against baseline,
BrainMap domains of the endpoints, a distance check, and the evidence packet
for the narrative. ``report.save(out_dir)`` writes the tables, the evidence
packet and Brainnetome-style connectograms; ``report.markdown()`` gives a report
section.

Network labels follow the annotation's ``network`` column: the source atlas's
own division (Schaefer: Yeo-7, via ``annotate_rois(..., networks=...)``) or the
Brainnetome division for Brainnetome regions (:func:`annotate_bna_regions`).

Everything is descriptive: nothing is tested, and the result's correction and
inference unit are unchanged.
"""
from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from conninfpy.interpret.annotate import RoiAnnotation
from conninfpy.interpret.annotate import summarize_annotation
from conninfpy.interpret.evidence import build_annotation_evidence
from conninfpy.interpret.evidence import describe_coverage

_RELATIONS = ("circuit", "hemispheric_relation", "distance_band", "network_relation", "anatomical_relation")
_TITLES = (("circuit", "Circuit class"), ("hemispheric_relation", "Hemispheric relation"),
           ("anatomical_relation", "Anatomical level"), ("distance_band", "Distance"),
           ("network_relation", "Network relation"))


@contextmanager
def _display_labels(result: Any, labels: Optional[Mapping[str, str]]):
    """Temporarily rename the design's profiles / comparison (e.g. 'A' -> 'eyes open')."""
    design = getattr(result, "design", None)
    if not labels or design is None:
        yield
        return
    renamed = copy.deepcopy(design)
    prof = renamed.get("profiles") or {}
    for old, new in labels.items():
        if old in prof:
            prof[new] = prof.pop(old)
    if renamed.get("comparison"):
        renamed["comparison"] = [labels.get(x, x) for x in renamed["comparison"]]
    result.design = renamed
    try:
        yield
    finally:
        result.design = design


@dataclass
class InterpretationReport:
    """Brainnetome interpretation of one result (see :func:`interpret_result`)."""

    name: str
    source_atlas: str
    alpha: float
    edges: pd.DataFrame
    relations: pd.DataFrame
    relations_by_tail: pd.DataFrame
    groups: Dict[str, pd.DataFrame]
    domains: pd.DataFrame
    distance_qc: pd.DataFrame
    coupling_by_tail: pd.DataFrame
    evidence: Dict[str, Any]
    sentence: str
    comparison: Optional[Tuple[str, str]]
    result: Any = field(repr=False)
    rois: RoiAnnotation = field(repr=False)
    labels: Optional[Dict[str, str]] = field(default=None, repr=False)

    def save(self, out_dir: Any, *, figures: bool = True, levels: Sequence[str] = ("gyrus", "subregion")) -> Path:
        """Write the tables (CSV), the evidence packet (JSON) and connectograms (PNG, PDF) to ``out_dir``."""
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.edges.to_csv(d / "connections_annotated.csv", index=False)
        for level, table in self.groups.items():
            table.to_csv(d / f"groups_{level}.csv", index=False)
        self.relations.to_csv(d / "relations.csv", index=False)
        self.relations_by_tail.to_csv(d / "relations_by_tail.csv", index=False)
        self.domains.to_csv(d / "brainmap_domains.csv", index=False)
        self.distance_qc.to_csv(d / "distance_qc.csv", index=False)
        self.coupling_by_tail.to_csv(d / "coupling_by_tail.csv", index=False)
        (d / "evidence.json").write_text(json.dumps(self.evidence, indent=1, default=str))
        if figures:
            self.connectograms(d, levels=levels)
        return d

    def connectograms(self, out_dir: Any, levels: Sequence[str] = ("gyrus", "subregion")) -> None:
        """Brainnetome-style connectograms, both tails side by side, per level (PNG and PDF)."""
        import matplotlib
        if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "cairo"):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from conninfpy.plot import plot_connectogram

        with _display_labels(self.result, self.labels):
            for level in levels:
                fig, axes = plt.subplots(1, 2, figsize=(18, 9.2))
                for ax, tail in zip(axes, ("positive", "negative")):
                    plot_connectogram(self.result, self.rois, tail=tail, level=level, labels=level,
                                      alpha=self.alpha, ax=ax)
                fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.04)
                fig.suptitle(f"{self.name}: Brainnetome {level} connectogram (α = {self.alpha:g} per tail)", fontsize=13)
                for ext in ("png", "pdf"):
                    fig.savefig(Path(out_dir) / f"connectogram_{level}.{ext}", dpi=150, bbox_inches="tight")
                plt.close(fig)

    def markdown(self, reproduction: Optional[str] = None) -> str:
        """Report section: relations, coupling change, lobe groups, domains and the distance check."""
        repro = f" Reproduction against the stored map: {reproduction}." if reproduction else ""
        out = [f"\n## {self.name}\n\n{self.source_atlas}.{repro}\n\n{self.sentence}\n"
               f"\nConnectograms: `{self.name}/connectogram_gyrus.pdf`, `{self.name}/connectogram_subregion.pdf`.\n"]
        rel = self.relations_by_tail
        for col, title in _TITLES:
            t = rel[rel.relation == col][["category", "n_positive", "share_positive", "n_negative",
                                          "share_negative", "share_of_tested"]].sort_values("share_of_tested", ascending=False)
            out.append(f"\n**{title}**: detections per tail vs tested opportunities\n\n" + md_table(t))
        if not self.coupling_by_tail.empty and self.comparison:
            out.append("\n**Coupling change of the detected connections** (from the coefficient maps; "
                       f"comparison {self.comparison[0]} → {self.comparison[1]})\n\n"
                       + md_table(self.coupling_by_tail.sort_values(["tail", "n"], ascending=[True, False])))
        g = self.groups["lobe"].sort_values("n_detected", ascending=False)
        out.append("\n**Lobe groups** (ROIs not ≥ 50 % in one lobe are 'mixed')\n\n" + md_table(g, n=12))
        out.append("\n**BrainMap domains** (unique ROIs; overlapping memberships)\n\n" + md_table(self.domains))
        out.append("\n**Distance QC**\n\n" + md_table(self.distance_qc))
        return "".join(out)


def interpret_result(
    result: Any,
    rois: RoiAnnotation,
    *,
    name: str = "result",
    source_atlas: str = "",
    alpha: float = 0.05,
    labels: Optional[Mapping[str, str]] = None,
    param_idx: Optional[int] = None,
) -> InterpretationReport:
    """Describe a result's detected connections with the Brainnetome atlas.

    Parameters
    ----------
    result : InferenceResult
        Any result with p-value maps (``analyze(...).inference``).
    rois : RoiAnnotation
        Annotation of the result's ROIs in matrix order: :func:`annotate_rois`
        for another atlas (pass its own network labels as ``networks``) or
        :func:`annotate_bna_regions` for Brainnetome regions.
    name, source_atlas : str
        Used in the evidence packet and the report.
    alpha : float
        Detection threshold per tail.
    labels : mapping, optional
        Display names for the design's profiles and comparison (e.g.
        ``{'A': 'eyes open', 'B': 'eyes closed'}``); the result is not changed.
    param_idx : int, optional
        (E, H) cell of a grid result.
    """
    with _display_labels(result, labels):
        design = getattr(result, "design", None) or {}
        comparison = tuple(design["comparison"]) if design.get("comparison") else None
        edges = result.annotated_edges(rois, alpha=alpha)
        summaries = {level: summarize_annotation(result, rois, alpha=alpha, level=level, param_idx=param_idx)
                     for level in ("lobe", "network")}
    s = summaries["lobe"]

    coupling = []
    for tail in ("positive", "negative"):
        sel = edges[edges[f"p_{tail}"] <= alpha]
        counts = sel["coupling_change"].value_counts() if "coupling_change" in sel else pd.Series(dtype=int)
        for cls, k in counts.items():
            coupling.append({"tail": tail, "coupling_change": cls, "n": int(k), "share": k / len(sel)})

    tested = s["edges"]
    rows = []
    for col in _RELATIONS:
        for cat, g in tested.groupby(col):
            rows.append({"relation": col, "category": cat, "n_tested": len(g), "share_of_tested": len(g) / len(tested),
                         "n_positive": int((g["p_positive"] <= alpha).sum()),
                         "n_negative": int((g["p_negative"] <= alpha).sum())})
    by_tail = pd.DataFrame(rows)
    for tail in ("positive", "negative"):
        tot = by_tail.groupby("relation")[f"n_{tail}"].transform("sum")
        by_tail[f"share_{tail}"] = np.where(tot > 0, by_tail[f"n_{tail}"] / tot, np.nan)

    evidence = build_annotation_evidence(edges, rois, contrast_name=name, source_atlas=source_atlas,
                                         inference_unit=getattr(result, "inference_unit", "edge"), summary=s)
    sentence = describe_coverage(evidence)["report_sentence"] if evidence["rois"] else "no detections"
    return InterpretationReport(
        name=name, source_atlas=source_atlas, alpha=alpha, edges=edges, relations=s["relations"],
        relations_by_tail=by_tail, groups={lv: summaries[lv]["regions"] for lv in summaries},
        domains=s["domains"], distance_qc=s["distance_qc"], coupling_by_tail=pd.DataFrame(coupling),
        evidence=evidence, sentence=sentence, comparison=comparison, result=result, rois=rois,
        labels=dict(labels) if labels else None)


def roi_markdown(rois: RoiAnnotation, title: str) -> str:
    """Report section on how the ROIs were assigned (levels, purity, network source)."""
    lv = rois.table["level"].value_counts().to_dict()
    source = rois.settings.get("network_source", "")
    return (f"\n## ROI annotation: {title}\n\nAssignment level of the {len(rois)} ROIs: {lv}. "
            f"Median subregion purity {rois.table['region_purity'].median():.2f}. Network labels: {source}.\n")


def reproduction_note(result: Any, stored: Any, alpha: float = 0.05) -> str:
    """Compare a result's p-value maps with a stored ``.npz`` (keys 'positive', 'negative')."""
    if stored is None or not Path(stored).exists():
        return "no stored map"
    s = np.load(stored)
    d = max(float(np.nanmax(np.abs(np.asarray(result[t]) - s[t]))) for t in ("positive", "negative"))
    iu = np.triu_indices(s["positive"].shape[0], 1)
    same = all(np.array_equal(np.asarray(result[t])[iu] < alpha, s[t][iu] < alpha) for t in ("positive", "negative"))
    return f"max |Δp| = {d:.3g}; same detections at α = {alpha:g}: {same}"


def md_table(df: pd.DataFrame, n: Optional[int] = None, fmt: str = "{:.3f}") -> str:
    df = df if n is None else df.head(n)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    rows = ["| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |"
            for _, r in df.iterrows()]
    return head + "\n".join(rows) + "\n"


__all__ = ["InterpretationReport", "interpret_result", "roi_markdown", "reproduction_note"]
