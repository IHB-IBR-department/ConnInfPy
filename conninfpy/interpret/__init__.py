"""Interpretation of inference results: Brainnetome annotation, reports, evidence and narration.

* :mod:`conninfpy.interpret.annotate`: ROI and connection annotation with the
  Brainnetome atlas (:func:`annotate_rois`, :func:`annotate_bna_regions`,
  :func:`summarize_annotation`, :func:`aggregate_connections`);
* :mod:`conninfpy.interpret.report`: one-call interpretation of a result
  (:func:`interpret_result`), tables, evidence packet and connectograms;
* :mod:`conninfpy.interpret.evidence` and :mod:`conninfpy.interpret.llm_narrative`:
  the evidence packet and its optional LLM narration.
"""
from .annotate import (RoiAnnotation, aggregate_connections, annotate_bna_regions, annotate_edges,
                       annotate_rois, bna_region_ids, summarize_annotation)
from .report import InterpretationReport, interpret_result

__all__ = ["RoiAnnotation", "annotate_rois", "annotate_bna_regions", "bna_region_ids", "annotate_edges",
           "summarize_annotation", "aggregate_connections", "InterpretationReport", "interpret_result"]
