conninfpy.stats
===============

Advanced inference API — the permutation pipelines behind
:func:`conninfpy.analyze`. Inputs are expected on the analysis scale (e.g.
Fisher z); every pipeline returns an
:class:`~conninfpy.results.InferenceResult`.

Permutation pipelines:

- :func:`~conninfpy.stats.pairwise.compute_p_val` — two-sample, paired and one-sample t-test designs
- :func:`~conninfpy.stats.glm.compute_p_val_glm` — Freedman–Lane GLM with t / β / F contrasts
- :func:`~conninfpy.stats.glm.compute_p_val_glm_multi` — several contrasts under one nuisance model, one permutation pass
- :func:`~conninfpy.stats.glm.compute_p_val_paired_glm` — paired A vs B with condition-varying confounds
- :func:`~conninfpy.stats.pairwise.compute_null_dist` — the max-statistic null distribution alone

Edge-wise statistics and designs:
:func:`~conninfpy.stats.pairwise.compute_t_stat`,
:func:`~conninfpy.stats.pairwise.compute_t_stat_diff`,
:func:`~conninfpy.stats.glm.compute_glm_stat`,
:func:`~conninfpy.stats.glm.build_design_matrix`.

Enums: :class:`~conninfpy.stats.pairwise.StatMethod`,
:class:`~conninfpy.stats.pairwise.TestType`,
:class:`~conninfpy.stats.glm.GLMStatType`.

Permutation acceleration — :mod:`conninfpy.stats.acceleration`
--------------------------------------------------------------

Set ``acceleration='gpd'`` (or ``'gamma'``) on either pipeline to replace
the p-values that fewer than 10 permutations reach with a fitted tail
(Generalized Pareto Distribution, Winkler 2016b); all other p-values stay
exact. A goodness-of-fit guard (Anderson–Darling on tail exceedances) falls
back to permutation p-values when the GPD does not fit cleanly.

Package surface
---------------

.. automodule:: conninfpy.stats
   :members:
   :undoc-members:
   :show-inheritance:

t-test designs — :mod:`conninfpy.stats.pairwise`
------------------------------------------------

.. automodule:: conninfpy.stats.pairwise
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: StatMethod, TestType, GLMStatType

GLM designs — :mod:`conninfpy.stats.glm`
----------------------------------------

.. automodule:: conninfpy.stats.glm
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: StatMethod, TestType, GLMStatType

Tail approximation — :mod:`conninfpy.stats.acceleration`
--------------------------------------------------------

.. automodule:: conninfpy.stats.acceleration
   :members:
   :undoc-members:
   :show-inheritance:
