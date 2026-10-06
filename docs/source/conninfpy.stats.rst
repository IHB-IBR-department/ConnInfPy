conninfpy.stats
===============

Advanced inference API — the permutation pipelines behind
:func:`conninfpy.analyze`. Inputs are expected on the analysis scale (e.g.
Fisher z); every pipeline returns an
:class:`~conninfpy.results.InferenceResult`.

Permutation pipelines:

- :func:`~conninfpy.pairwise_stats.compute_p_val` — two-sample, paired and one-sample t-test designs
- :func:`~conninfpy.glm_stats.compute_p_val_glm` — Freedman–Lane GLM with t / β / F contrasts
- :func:`~conninfpy.glm_stats.compute_p_val_glm_multi` — several contrasts under one nuisance model, one permutation pass
- :func:`~conninfpy.glm_stats.compute_p_val_paired_glm` — paired A vs B with condition-varying confounds
- :func:`~conninfpy.pairwise_stats.compute_null_dist` — the max-statistic null distribution alone

Edge-wise statistics and designs:
:func:`~conninfpy.pairwise_stats.compute_t_stat`,
:func:`~conninfpy.pairwise_stats.compute_t_stat_diff`,
:func:`~conninfpy.glm_stats.compute_glm_stat`,
:func:`~conninfpy.glm_stats.build_design_matrix`.

Enums: :class:`~conninfpy.pairwise_stats.StatMethod`,
:class:`~conninfpy.pairwise_stats.TestType`,
:class:`~conninfpy.glm_stats.GLMStatType`.

Full documentation lives with the implementations in
:mod:`conninfpy.pairwise_stats` and :mod:`conninfpy.glm_stats`.
