"""Interpretation of inference results: Brainnetome annotation, reports and evidence.

* :mod:`conninfpy.interpret.annotate`: ROI and connection annotation with the
  Brainnetome atlas (:func:`annotate_rois`, :func:`annotate_bna_regions`,
  :func:`summarize_annotation`, :func:`aggregate_connections`);
* :mod:`conninfpy.interpret.report`: one-call interpretation of a result
  (:func:`interpret_result`), tables, evidence packet and connectograms;
* :mod:`conninfpy.interpret.evidence`: the evidence packet built from the
  annotation;
* :mod:`conninfpy.interpret.motion_qc`: additive, opt-in descriptive motion
  (framewise displacement) and QC-FC side-car diagnostics computed before
  inference (:func:`motion_qc_fc`, :func:`motion_qc_fc_paired`,
  :func:`resolve_fd`), with a Markdown section
  (:func:`~conninfpy.interpret.report.motion_qc_markdown`) for reports.
"""
from conninfpy.interpret.annotate import RoiAnnotation
from conninfpy.interpret.annotate import aggregate_connections
from conninfpy.interpret.annotate import annotate_bna_regions
from conninfpy.interpret.annotate import annotate_edges
from conninfpy.interpret.annotate import annotate_rois
from conninfpy.interpret.annotate import bna_region_ids
from conninfpy.interpret.annotate import summarize_annotation
from conninfpy.interpret.motion_qc import (
    detect_fd_column,
    motion_qc_fc,
    motion_qc_fc_paired,
    resolve_fd,
)
from conninfpy.interpret.report import InterpretationReport
from conninfpy.interpret.report import interpret_result
from conninfpy.interpret.report import motion_qc_markdown

__all__ = ["RoiAnnotation", "annotate_rois", "annotate_bna_regions", "bna_region_ids", "annotate_edges",
           "summarize_annotation", "aggregate_connections", "InterpretationReport", "interpret_result",
           "motion_qc_fc", "motion_qc_fc_paired", "resolve_fd", "detect_fd_column",
           "motion_qc_markdown"]
