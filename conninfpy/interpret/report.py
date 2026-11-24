"""Brainnetome interpretation of an inference result in one call.

:func:`interpret_result` takes a result and its :class:`~conninfpy.interpret.annotate.RoiAnnotation`
and returns an :class:`InterpretationReport`: the detected connections with
their endpoint anatomy and relations, detections against tested opportunities
by relation and by lobe / network group, coupling change against baseline,
BrainMap domains of the endpoints, a distance check, and the evidence packet
for downstream use. ``report.save(out_dir)`` writes the tables, the evidence
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
        sidecar = getattr(self.result, "motion_qc", None)
        if sidecar:
            out.append(motion_qc_markdown(sidecar, level=2))
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


def _adjustment_type_label(adjustment: Mapping[str, Any]) -> str:
    """Plain label for the adjustment state (partial Spearman is never claimed)."""
    mode = str(adjustment.get("mode"))
    if mode == "fc_residualized_not_partial":
        return "edge-value residualisation only (NOT partial Spearman)"
    if mode == "none":
        return "unadjusted (no nuisance matrix supplied)"
    return f"unavailable ({adjustment.get('reason') or mode})"


def _motion_qc_arm_markdown(diag: Mapping[str, Any], title: str, *, max_rows: int = 8) -> str:
    """One descriptive arm: provenance, per-observation table, QC-FC, distance."""
    out: list = []
    if title:
        out.append(f"\n**{title}**\n")
    status = str(diag.get("status"))
    reason = diag.get("status_reason")
    out.append(f"\nStatus: `{status}`" + (f" — {reason}" if reason and status != "available" else "") + ".\n")
    fd = diag.get("fd") or {}
    source = fd.get("source") or {}
    inputs = diag.get("input") or {}
    provenance = [
        ("FD source", f"{source.get('kind', 'unavailable')}"
                      + (f" (`{source.get('name')}`)" if source.get("name") else "")),
        ("FD units", str(fd.get("units") or "unknown")),
        ("FD convention", str(fd.get("convention") or "unknown")),
        ("FD alignment", str(fd.get("id_alignment") or "not applicable")),
        ("Input scale", f"{inputs.get('scale')} (source: {inputs.get('scale_source')})"),
        ("Per-observation mean field", str(inputs.get("mean_label"))),
        ("Transform", str(inputs.get("transform"))),
        ("Observations / ROIs / edges", f"{diag.get('n_observations')} / {diag.get('n_rois')} / {diag.get('n_edges')}"),
        ("Adjustment", str((diag.get("adjustment") or {}).get("mode"))),
        ("Adjustment type", _adjustment_type_label((diag.get("adjustment") or {}))),
    ]
    out.append("\n| Provenance | Value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in provenance) + "\n")

    subjects = diag.get("subjects") or {}
    mean_label = str(inputs.get("mean_label") or "mean_abs_value")
    if subjects.get(mean_label):
        ids = subjects.get("subject_ids")
        cond = subjects.get("condition")
        index_values = subjects.get("index_values")
        rows = []
        for idx, value in enumerate(subjects[mean_label][:max_rows]):
            rows.append({
                "row": idx if not index_values else index_values[idx],
                "subject": "" if not ids else ids[idx],
                "condition": "" if not cond else cond[idx],
                "fd_mm": (subjects.get("fd") or [None] * len(subjects[mean_label]))[idx],
                mean_label: value,
                "nan_share": (subjects.get("nan_share") or [None] * len(subjects[mean_label]))[idx],
                "negative_share": (subjects.get("negative_share") or [None] * len(subjects[mean_label]))[idx],
                "negative_denominator": (subjects.get("negative_denominator") or [None] * len(subjects[mean_label]))[idx],
            })
        table = pd.DataFrame(rows).astype(object).where(pd.DataFrame(rows).notna(), "")
        out.append("\nPer-observation table (first rows; NaN denominator = n_rois*(n_rois-1)/2 = "
                   f"{subjects.get('nan_denominator')}, negative denominator = finite edges)\n\n"
                   + md_table(table, fmt="{:.4f}"))
        out.append(f"\nShowing {min(max_rows, len(rows))} of {len(subjects[mean_label])} observation(s).\n")

    qc = diag.get("qc_fc") or {}
    q = qc.get("quantiles") or {}
    f = lambda v: "—" if v is None else f"{float(v):.3f}"
    section = f"""
| QC-FC (descriptive Spearman, no p-values) | Value |
|---|---|
| Edges with a valid association | {qc.get('n_edges_with_association')} / {qc.get('n_edges_total')} |
| Insufficient complete pairs (< {qc.get('min_complete_pairs')}) | {qc.get('n_edges_insufficient_pairs')} |
| Constant FD on complete pairs | {qc.get('n_edges_constant_fd')} |
| Constant edge on complete pairs | {qc.get('n_edges_constant_edge')} |
| rho median (q25 / q75) | {f(q.get('median'))} ({f(q.get('q25'))} / {f(q.get('q75'))}) |
| rho q10 / q90 | {f(q.get('q10'))} / {f(q.get('q90'))} |
| rho min / max | {f(q.get('min'))} / {f(q.get('max'))} |
| Subject mean |r| vs FD (rho, n) | {f((qc.get('subject_level') or {}).get('rho'))} ({(qc.get('subject_level') or {}).get('n')}) |
"""
    out.append(section)
    roi_nf = diag.get("roi_nonfinite") or {}
    if roi_nf.get("nan_count"):
        bad = [
            (idx, nan, inf, share)
            for idx, (nan, inf, share) in enumerate(zip(
                roi_nf.get("nan_count") or [],
                roi_nf.get("inf_count") or [],
                roi_nf.get("nonfinite_share") or [],
            ))
            if (nan or inf)
        ]
        out.append(
            "\nPer-ROI non-finite entries (non-diagonal edges, "
            f"denominator {roi_nf.get('denominator')} = observations × (ROIs − 1); "
            "NaN and Inf counted separately).\n"
        )
        if bad:
            table = pd.DataFrame([
                {"roi_index": idx, "nan": nan, "inf": inf,
                 "nonfinite_share": share if share is not None else ""}
                for idx, nan, inf, share in bad[:max_rows]
            ])
            out.append("\n" + md_table(table) + "\n")
            out.append(f"Showing {min(max_rows, len(bad))} of {len(bad)} ROI(s) with non-finite entries.\n")
        else:
            out.append("No non-finite entries on any ROI.\n")

    dist = diag.get("distance") or {}
    qc_dist = dist.get("qc_fc_vs_distance") or {}
    out.append("\nDistance (known coordinates only): "
               f"status `{dist.get('status')}`"
               + (f" — {dist.get('reason')}" if dist.get("status") != "available" else "")
               + ". Two separate descriptive measures:\n"
               f"- edge level `spearman_mean_vs_distance` = "
               f"{f(dist.get('spearman_mean_vs_distance'))} over {dist.get('n_edges_used')} edges "
               "(mean edge value vs distance; NOT the distance dependence of QC-FC);\n"
               f"- QC-FC vs distance `qc_fc_vs_distance` = {f(qc_dist.get('rho'))} "
               f"over {qc_dist.get('n_pairs')} edge(s) with a valid association "
               f"(status `{qc_dist.get('status')}`"
               + (f", {qc_dist.get('reason')}" if qc_dist.get("status") not in ("available", None) and qc_dist.get("reason") else "")
               + ").\n")
    adjustment = diag.get("adjustment") or {}
    residualization = adjustment.get("residualization") or {}
    if adjustment.get("mode") == "fc_residualized_not_partial" and residualization:
        out.append(
            "\nResidualisation (FC-only, **not** partial Spearman): rank "
            f"{residualization.get('matrix_rank')}/{residualization.get('n_columns')}, "
            f"residual df {residualization.get('residual_df_min')}–"
            f"{residualization.get('residual_df_max')} on each edge's complete rows; "
            f"{residualization.get('n_edges_insufficient_df')} edge(s) below 2 residual df, "
            f"{residualization.get('n_edges_zero_residual_variance')} with zero residual "
            "variance (both reported, never interpreted).\n"
        )
    unavailable = diag.get("unavailable") or []
    if unavailable:
        out.append("\nExplicitly unavailable:\n" + "".join(
            f"- `{item.get('field')}`: {item.get('status')} — {item.get('reason')}\n"
            for item in unavailable
        ))
    for note in diag.get("notes") or []:
        out.append(f"\n- {note}\n")
    return "".join(out)


def motion_qc_markdown(
    diag: Optional[Mapping[str, Any]],
    *,
    heading: str = "Motion / QC-FC side-car (descriptive, before inference)",
    level: int = 2,
    max_rows: int = 8,
) -> str:
    """Markdown section for a saved ``motion_qc`` diagnostic (or ``""``).

    Reports the FD provenance (source, units, unrecorded convention), the
    native-scale handling, the per-observation table with denominators, the
    descriptive QC-FC and distance associations, and every explicit
    unavailable reason. No p-values are produced and nothing is tested.
    """
    if not diag:
        return ""
    out = [f"\n{'#' * int(level)} {heading}\n"]
    out.append(
        "\nDescriptive only: Spearman rank agreement without p-values, no null "
        "distribution, no exclusions, and no effect on inference.\n"
    )
    if str(diag.get("kind")) == "motion_qc_fc_paired":
        pooling = diag.get("pooling") or {}
        out.append(f"\nPooling: `{pooling.get('status')}` — {pooling.get('note') or ''}\n")
        for name, arm in (diag.get("conditions") or {}).items():
            out.append(_motion_qc_arm_markdown(arm, f"Condition {name}", max_rows=max_rows))
        if diag.get("pooled"):
            out.append(_motion_qc_arm_markdown(diag["pooled"], "Pooled (stacked conditions, verified identity)", max_rows=max_rows))
        difference = diag.get("difference") or {}
        out.append(f"\nCondition difference: `{difference.get('status')}` — {difference.get('reason') or ''}\n")
    else:
        out.append(_motion_qc_arm_markdown(diag, "", max_rows=max_rows))
    return "".join(out)


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


__all__ = ["InterpretationReport", "interpret_result", "roi_markdown", "reproduction_note",
           "motion_qc_markdown"]
