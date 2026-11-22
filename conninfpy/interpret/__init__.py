"""Interpretation of inference results: Brainnetome annotation, reports and evidence.

* :mod:`conninfpy.interpret.annotate`: ROI and connection annotation with the
  Brainnetome atlas (:func:`annotate_rois`, :func:`annotate_bna_regions`,
  :func:`summarize_annotation`, :func:`aggregate_connections`);
* :mod:`conninfpy.interpret.report`: one-call interpretation of a result
  (:func:`interpret_result`), tables, evidence packet and connectograms;
* :mod:`conninfpy.interpret.evidence`: the evidence packet built from the
  annotation.
"""
from conninfpy.interpret.annotate import RoiAnnotation
from conninfpy.interpret.annotate import aggregate_connections
from conninfpy.interpret.annotate import annotate_bna_regions
from conninfpy.interpret.annotate import annotate_edges
from conninfpy.interpret.annotate import annotate_rois
from conninfpy.interpret.annotate import bna_region_ids
from conninfpy.interpret.annotate import summarize_annotation
from conninfpy.interpret.report import InterpretationReport
from conninfpy.interpret.report import interpret_result

__all__ = ["RoiAnnotation", "annotate_rois", "annotate_bna_regions", "bna_region_ids", "annotate_edges",
           "summarize_annotation", "aggregate_connections", "InterpretationReport", "interpret_result"]
