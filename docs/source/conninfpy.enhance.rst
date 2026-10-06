conninfpy.enhance
=================

Network enhancement: the ``apply_*`` wrappers (shared ``stat_dict → score_dict``
transformations used by both the t-test pipeline (:mod:`conninfpy.pairwise_stats`) and the GLM pipeline
(:mod:`conninfpy.glm_stats`)) and the raw scoring functions
(``get_*_score``, ``nbs_bct``).

These five wrappers are pure, key-agnostic transformations: a statistic
dictionary keyed by effect direction (``'g2>g1'/'g1>g2'`` for the
two-sample / paired t-test, ``'positive'/'negative'`` for the GLM, or
``'omnibus'`` for the F-contrast pipeline) maps to an enhanced score
dictionary with the same keys. Both the input statistic map and the
output enhanced map are non-negative ``(N, N)`` symmetric matrices with
zero diagonal.

The mathematical definitions of each operator are in the corresponding
sections of the methods chapter.

.. automodule:: conninfpy.enhance
   :members:
   :undoc-members:
   :show-inheritance:
