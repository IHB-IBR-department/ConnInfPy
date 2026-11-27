conninfpy.enhance
=================

Network enhancement: the ``apply_*`` wrappers (shared ``stat_dict → score_dict``
transformations used by both the t-test pipeline (:mod:`conninfpy.stats.pairwise`) and the GLM pipeline
(:mod:`conninfpy.stats.glm`)) and the raw scoring functions
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

TFNBS — :mod:`conninfpy.enhance.tfnbs`
--------------------------------------

.. automodule:: conninfpy.enhance.tfnbs
   :members:
   :undoc-members:
   :show-inheritance:

NBS — :mod:`conninfpy.enhance.nbs`
----------------------------------

.. automodule:: conninfpy.enhance.nbs
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: CNBSBlocks

Network-informed TFNBS — :mod:`conninfpy.enhance.ni_tfnbs`
----------------------------------------------------------

.. automodule:: conninfpy.enhance.ni_tfnbs
   :members:
   :show-inheritance:

Functional-block clustering TFNBS — :mod:`conninfpy.enhance.fbc_tfnbs`
-----------------------------------------------------------------------

.. automodule:: conninfpy.enhance.fbc_tfnbs
   :members:
   :show-inheritance:
