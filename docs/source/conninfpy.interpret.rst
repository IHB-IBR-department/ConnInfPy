conninfpy.interpret
===================

Anatomical interpretation of inference results with the Brainnetome atlas
(BNA-246; Fan et al., 2016). Everything here describes a result: nothing adds
a test, and the p-values, their correction and the inference unit (edge,
component or block) are unchanged. The rules (tissue overlap, purity, network
labels, declared template) and worked calls are in :ref:`usage-interpret`.

- :func:`~conninfpy.interpret.annotate_rois` describes the ROIs of any atlas
  or coordinate set by the BNA regions their sphere, parcel or point covers;
  :func:`~conninfpy.interpret.annotate_bna_regions` annotates data
  parcellated with BNA-246 itself.
- :func:`~conninfpy.interpret.annotate_edges` (also
  :meth:`~conninfpy.results.InferenceResult.annotated_edges`),
  :func:`~conninfpy.interpret.summarize_annotation` and
  :func:`~conninfpy.interpret.aggregate_connections` describe the connections
  against the tested opportunities.
- :func:`~conninfpy.interpret.interpret_result` does all of it in one call and
  returns an :class:`~conninfpy.interpret.InterpretationReport` (tables,
  evidence packet, connectograms, Markdown section).

The optional narrative (``conninfpy.interpret.llm_narrative``) writes text
from the evidence packet only.

.. automodule:: conninfpy.interpret
   :members:
   :exclude-members: table, weights, membership, reference, settings
   :show-inheritance:

Evidence packet
---------------

.. automodule:: conninfpy.interpret.evidence
   :members:
