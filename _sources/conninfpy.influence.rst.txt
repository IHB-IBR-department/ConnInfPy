conninfpy.influence
===================

Descriptive per-subject leave-one-out (LOO) influence for an explicitly chosen
set of edges of a finished result. It refits the *native* model once per
subject on one fewer observation and reports ``Δ = c'β_full − c'β_(−s)``, the
recomputed native statistic, its df and n, the leverage of *that edge's*
design, the edge's MAD and a status per subject × edge. There is no
permutation, no random number generation, no p-value and no selected-edge
search; the source result is never modified.

.. automodule:: conninfpy.influence
   :members: subject_influence
   :show-inheritance:

Example
-------

.. code-block:: python

   import numpy as np
   from conninfpy import analyze
   from conninfpy.influence import subject_influence

   result = analyze(group1=condition_a, group2=condition_b,
                    test_type="paired", method="tstat").inference

   report = subject_influence(
       result,
       edges=[(4, 17), (23, 61)],          # explicitly chosen ROI pairs
       group1=condition_a,                 # the tested inputs, analysis scale
       group2=condition_b,
   )
   report["summary"]["top_subject_by_max_abs_delta"]
   report["edges"][0]["effect"], report["edges"][0]["stat"]
   [row for row in report["per_subject_edge"] if row["status"] != "ok"]

Limits
------

* **Inputs are the caller's responsibility.** A result carries coefficient
  maps and a design description, not the data. Row order, subject IDs, the
  analysis scale (Fisher z already applied) and the GLM design matrix must be
  the ones that produced the result. The report refits the full model and
  requires it to reproduce ``result.beta`` at ``1e-5``/``1e-8``; a mismatch is
  ``unavailable`` with the reason "inputs do not reproduce the tested model",
  not a repaired answer.
* **Paired alignment.** With a recorded non-identity
  ``design['paired_alignment']``, either supply ``ids_group1``/``ids_group2``
  (verified against the record and applied exactly once) or pass the aligned
  input with ``paired_inputs_aligned=True``. A second reorder is never
  applied. For a paired design the mean ``B − A`` is invariant to the row
  pairing, so the observed-statistic check, not the beta check, catches a
  wrong order.
* **Only explicit edges.** At most :data:`~conninfpy.influence.MAX_EDGES` (64)
  unique off-diagonal pairs, each inside the producer's ``estimable`` and
  ``tested`` provenance masks; anything else raises ``ValueError`` before any
  computation. There is no ``select='significant'``. The report is capped at
  :data:`~conninfpy.influence.MAX_RECORDS` (4096) subject × edge records: the
  refit grows with the number of subjects, so a larger request returns an
  ``unavailable`` report with the reason instead of running.
* **Leverage is per edge.** It uses the rows that edge actually uses:
  ``1/n_edge`` of the relevant group for a pairwise equivalent design, and the
  hat-matrix diagonal of the edge's coverage-row design for a GLM (one entry
  at a time, never an ``n × n`` hat matrix). A row outside an edge's coverage
  has no leverage there and reports ``None`` with ``not_in_coverage``; no
  full-design leverage is mixed into the same column.
* **Inference unit.** For NBS/cNBS/TFNBS results the diagnostic is allowed at
  explicit edges but the report says that these are not independent edge
  discoveries and that the p-value map belongs to the component or block.
* **Native units and no hypotheses.** ``Δ`` and the effect stay in the
  response units of the model. The recomputed t values are descriptive, not
  tests, and are not corrected for multiplicity.
* **Statuses instead of numbers.** Constant values, non-estimable
  contrasts, no residual df, no coverage and all-NaN edges are reported with
  ``delta``/``t`` set to ``None`` and a status; ``zero_mad`` is kept distinct
  from ``constant`` because a zero MAD alone can be majority ties with one
  outlying value. ``rank_loss`` means the reduced design rank fell below the
  full fit's rank **on the same edge** (not ``rank < p``: a design may already
  be rank deficient); such a row keeps its ``delta`` but is excluded from the
  per-subject ``max|Δ|``, the ranking and the ``ok`` counts.
* **Harmonization.** ComBat and other sample-dependent transforms are fixed:
  the diagnostic is conditional on the fitted harmonization and never refits
  it.
