"""Evidence packets for interpretation text, built from Brainnetome annotation.

The packet holds only what the analysis and the atlas support:

* ``query`` - the contrast, the source atlas, the detected connections and
  their inference unit (edge, component or block);
* ``reference`` - Brainnetome atlas and BrainMap provenance, ROI support;
* ``rois`` - anatomy of every endpoint ROI (assignment level, region, lobe,
  hemisphere, structure, cytoarchitectonic description, Yeo network,
  BrainMap domains, flags) and how many detected connections touch it;
* ``terms`` - BrainMap terms of the BNA regions each endpoint covers, with the
  published forward-inference likelihood ratio (never combined across
  regions or endpoints);
* ``relations`` - optional detection counts by circuit class, hemispheric
  relation and distance band, next to the tested opportunities;
* ``caveats``.

No term is ranked as "the" function of a region, and nothing here is a test.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from conninfpy.atlas.brainnetome import BD_DOMAINS

DEFAULT_CAVEATS = [
    "BrainMap profiles are forward-inference associations from published activation studies: "
    "they describe which kinds of tasks engage a region, not what the region or the connection does.",
    "A region without published terms has too few BrainMap experiments; this is not evidence of no function.",
    "Left/right differences in profiles partly reflect lateralized task literatures (e.g. language); "
    "read homologous regions together.",
    "Detections are interpreted at the method's inference unit: an edge of a significant component or "
    "block is not individually significant.",
    "ROIs are described by the tissue they sampled (sphere or parcel overlap); 'mixed' and 'unassigned' "
    "ROIs carry no single regional label.",
]

_ROI_FIELDS = ("assigned", "level", "region", "lobe", "hemisphere", "structure", "description",
               "yeo7", "domains", "profile_status", "flags")


def build_annotation_evidence(
    edges: pd.DataFrame,
    rois: Any,
    *,
    contrast_name: str = "Unknown",
    source: str = "conninfpy_edges",
    source_atlas: Optional[str] = None,
    inference_unit: str = "edge",
    summary: Optional[Dict[str, Any]] = None,
    caveats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compile detected connections and their Brainnetome annotation into an evidence packet.

    Parameters
    ----------
    edges : DataFrame
        Detected connections with ``roi_i`` / ``roi_j`` (matrix indices),
        e.g. ``result.significant_edges(...)`` or ``annotate_edges(...)``.
    rois : RoiAnnotation
        From :func:`conninfpy.interpret.annotate.annotate_rois`, in matrix order.
    contrast_name, source : str
        Labels for the query.
    source_atlas : str, optional
        Name of the atlas the ROIs came from (default: the annotation's).
    inference_unit : {'edge', 'component', 'block'}
        Unit of the corrected inference (``result.inference_unit``).
    summary : dict, optional
        Output of :func:`conninfpy.interpret.annotate.summarize_annotation`; its
        relation counts are added.
    caveats : list of str, optional
        Replaces :data:`DEFAULT_CAVEATS`.
    """
    t = rois.table
    if not edges.empty:
        ends = pd.concat([edges["roi_i"], edges["roi_j"]]).astype(int)
        burden = ends.value_counts().to_dict()
    else:
        burden = {}
    endpoint_ids = sorted(int(k) for k in burden)

    roi_entries = []
    for k in endpoint_ids:
        r = t.iloc[k]
        entry = {"roi_index": k, "roi_name": str(r["roi"]), "n_detected_edges": int(burden[k])}
        entry.update({f: (None if pd.isna(r[f]) else str(r[f])) for f in _ROI_FIELDS})
        roi_entries.append(entry)

    terms: List[Dict[str, Any]] = []
    if rois.reference.profiles is not None and endpoint_ids:
        long = rois.profiles("BD")
        long = long[long["roi_index"].isin(endpoint_ids)]
        for _, row in long.iterrows():
            terms.append({"roi_index": int(row["roi_index"]), "roi_name": str(row["roi"]),
                          "region": str(row["region"]), "region_weight": round(float(row["region_weight"]), 3),
                          "term": str(row["term"]), "domain": str(row["domain"]),
                          "likelihood_ratio": float(row["likelihood_ratio"])})

    pairs: Dict[tuple, int] = {}
    if not edges.empty:
        for a, b in zip(edges["roi_i"].astype(int), edges["roi_j"].astype(int)):
            key = tuple(sorted((str(t.iloc[a]["assigned"]), str(t.iloc[b]["assigned"]))))
            pairs[key] = pairs.get(key, 0) + 1
    region_pairs = [{"source": s, "target": g, "n_edges": n} for (s, g), n in sorted(pairs.items())]

    relations = []
    if summary is not None:
        rel = summary["relations"]
        for _, row in rel[rel["n_detected"] > 0].iterrows():
            relations.append({"relation": str(row["relation"]), "category": str(row["category"]),
                              "n_detected": int(row["n_detected"]), "n_tested": int(row["n_tested"]),
                              "share_of_detections": round(float(row["share_of_detections"]), 3),
                              "share_of_tested": round(float(row["share_of_tested"]), 3)})

    settings = rois.settings
    prov = settings.get("provenance", {})
    return {
        "query": {
            "source": source, "contrast": contrast_name,
            "atlas": source_atlas or f"{len(t)} ROIs ({settings.get('support')} support)",
            "roi_ids": endpoint_ids, "edge_count": int(len(edges)),
            "inference_unit": inference_unit, "region_pairs": region_pairs,
        },
        "reference": {
            "atlas": prov.get("atlas", "Brainnetome BNA-246 (Fan et al. 2016)"),
            "profiles": (prov.get("profiles") or {}).get("source", "not available"),
            "support": settings.get("support"), "space": settings.get("space"),
            "purity": settings.get("purity"),
        },
        "rois": roi_entries,
        "terms": terms,
        "relations": relations,
        "caveats": list(DEFAULT_CAVEATS if caveats is None else caveats),
    }


def validate_evidence(evidence: Dict[str, Any]) -> None:
    """Check the structure of an evidence packet.

    Raises
    ------
    ValueError
        If a required field is missing or malformed.
    """
    required = {"query", "reference", "rois", "terms", "caveats"}
    missing = required - set(evidence)
    if missing:
        raise ValueError(f"Missing required top-level evidence fields: {sorted(missing)}")
    query = evidence["query"]
    if not isinstance(query, dict):
        raise ValueError("Field 'query' must be a dictionary.")
    for k in ("source", "atlas", "inference_unit"):
        if not query.get(k):
            raise ValueError(f"Field 'query.{k}' is required and cannot be empty.")
    if not isinstance(evidence["reference"], dict) or not evidence["reference"].get("atlas"):
        raise ValueError("Field 'reference.atlas' is required.")
    rois = evidence["rois"]
    if not isinstance(rois, list) or not rois:
        raise ValueError("Field 'rois' must be a non-empty list: no detected connection to describe.")
    for idx, r in enumerate(rois):
        for k in ("roi_index", "roi_name", "assigned", "level"):
            if k not in r:
                raise ValueError(f"ROI entry {idx} is missing '{k}'.")
    if not isinstance(evidence["terms"], list):
        raise ValueError("Field 'terms' must be a list (it may be empty).")
    for idx, term in enumerate(evidence["terms"]):
        for k in ("term", "domain", "likelihood_ratio", "roi_index"):
            if k not in term:
                raise ValueError(f"Term entry {idx} is missing '{k}'.")
        if not term["term"]:
            raise ValueError(f"Term entry {idx} has an empty 'term'.")
    if not isinstance(evidence["caveats"], list):
        raise ValueError("Field 'caveats' must be a list of strings.")


def summarize_terms(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """BrainMap domains and terms among the endpoint ROIs.

    Each ROI counts once per domain or term, however many connections touch
    it (a hub is not counted repeatedly); ``n_detected_edges`` per ROI is
    reported separately. Memberships overlap, so shares need not sum to 1.
    """
    rois = evidence["rois"]
    n = len(rois)
    by_domain: Dict[str, set] = {d: set() for d in BD_DOMAINS}
    by_term: Dict[str, Dict[str, Any]] = {}
    for t in evidence["terms"]:
        by_domain.setdefault(t["domain"], set()).add(t["roi_index"])
        e = by_term.setdefault(t["term"], {"term": t["term"], "domain": t["domain"], "rois": set(),
                                           "max_likelihood_ratio": 0.0})
        e["rois"].add(t["roi_index"])
        e["max_likelihood_ratio"] = max(e["max_likelihood_ratio"], t["likelihood_ratio"])
    domains = [{"domain": d, "n_rois": len(s), "share_rois": len(s) / n if n else np.nan}
               for d, s in by_domain.items()]
    terms = sorted(({"term": e["term"], "domain": e["domain"], "n_rois": len(e["rois"]),
                     "max_likelihood_ratio": e["max_likelihood_ratio"]} for e in by_term.values()),
                   key=lambda e: (-e["n_rois"], e["term"]))
    hubs = sorted(({"roi_name": r["roi_name"], "assigned": r["assigned"], "n_detected_edges": r["n_detected_edges"]}
                   for r in rois), key=lambda e: -e["n_detected_edges"])[:5]
    return {"domains": domains, "terms": terms, "top_endpoint_rois": hubs}


def describe_coverage(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Factual description of what the annotation can support (no quality label)."""
    rois = evidence["rois"]
    n = len(rois)
    with_terms = sum(1 for r in rois if r.get("profile_status") == "terms")
    unassigned = sum(1 for r in rois if r.get("level") == "unassigned")
    mixed = sum(1 for r in rois if r.get("level") == "mixed")
    present = [d["domain"] for d in summarize_terms(evidence)["domains"] if d["n_rois"]]
    sentence = (
        f"The {evidence['query']['edge_count']} detected connections (inference unit: "
        f"{evidence['query']['inference_unit']}) involve {n} ROIs; {with_terms} of them cover Brainnetome "
        f"regions with published BrainMap terms"
        + (f" (domains: {', '.join(present)})" if present else "")
        + f", {mixed} are anatomically mixed and {unassigned} unassigned."
    )
    return {"n_endpoint_rois": n, "n_with_terms": with_terms, "n_mixed": mixed,
            "n_unassigned": unassigned, "domains_present": present, "report_sentence": sentence}


__all__ = ["DEFAULT_CAVEATS", "build_annotation_evidence", "validate_evidence", "summarize_terms",
           "describe_coverage"]
