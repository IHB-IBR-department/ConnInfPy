conninfpy.stats
===============

Advanced inference API — the permutation pipelines behind
:func:`conninfpy.analyze`: two-sample / paired / one-sample tests
(:func:`~conninfpy.pairwise_stats.compute_p_val`), the Freedman–Lane GLM
with t / β / F contrasts (:func:`~conninfpy.glm_stats.compute_p_val_glm`,
:func:`~conninfpy.glm_stats.compute_p_val_glm_multi`), the paired-difference
GLM, and the raw null distribution. Inputs are expected on the analysis
scale (e.g. Fisher z); every function returns an
:class:`~conninfpy.results.InferenceResult`.

The implementations live in :mod:`conninfpy.pairwise_stats` and
:mod:`conninfpy.glm_stats`.

.. automodule:: conninfpy.stats
   :members:
   :imported-members:
