Usage
=====

Start with :func:`conninfpy.analyze`: it takes per-subject connectivity and a
design, prepares the input, optionally harmonizes sites, and runs permutation
inference. The pipelines it is built from live in :mod:`conninfpy.stats` and
share one contract: edge-wise statistics → topology-aware enhancement →
permutation null → FWER / FDR-corrected p-values, with positive and negative
effects tested separately.

Quick reference:

============================================================  ============================================================
Entry point                                                    Use case
============================================================  ============================================================
:func:`conninfpy.analyze`                                      Main entry point: input preparation → optional ComBat → GLM / t-test inference → :class:`~conninfpy.analysis.AnalyzeResult`
:func:`conninfpy.stats.compute_p_val`                          Group / paired / one-sample t-test
:func:`conninfpy.stats.compute_p_val_glm`                      Custom design matrix and t / β / F contrasts (Freedman–Lane)
:func:`conninfpy.stats.compute_p_val_glm_multi`                Several contrasts under a shared nuisance model in **one** permutation pass
:func:`conninfpy.stats.compute_p_val_paired_glm`               Paired A vs B with condition-varying confounds
============================================================  ============================================================

Helper modules:

- :mod:`conninfpy.harmonize` — multi-site ComBat harmonization + design diagnostics
- :mod:`conninfpy.acceleration` — GPD/gamma tail approximation for permutation acceleration
- :mod:`conninfpy.enhance` — enhancement without permutation (``apply_*``, ``get_*_score``, ``nbs_bct``)
- :mod:`conninfpy.sim` — synthetic connectivity and topology scenarios
- :mod:`conninfpy.loaders` — manifest-based data loading (:func:`conninfpy.load`)

Input conventions
-----------------

- Connectivity tensors: shape ``(n_subjects, N, N)``. Directed (asymmetric)
  connectivity is supported. The diagonal is zeroed automatically on a copy —
  your arrays are never modified.
- :func:`conninfpy.analyze` takes ``input='r'`` (correlations; Fisher r→z is
  applied once) or ``input='z'`` (already transformed). Left as ``None`` it
  detects the scale: a unit diagonal means correlations, any value beyond ±1
  means Fisher z; data that could be either raise an error rather than risk a
  second transform.
- The advanced pipelines in :mod:`conninfpy.stats` expect data already on the
  analysis scale (e.g. Fisher z via :func:`conninfpy.fisher_r_to_z`).
- Results are :class:`~conninfpy.results.InferenceResult` objects, which
  behave as ``{'positive', 'negative'}`` dicts and additionally carry
  ``positive`` / ``negative`` / ``method`` / ``n_permutations`` /
  ``acceleration`` / ``wall_time_s`` attributes and edge-table exports.
  ``analyze()`` wraps one in an :class:`~conninfpy.analysis.AnalyzeResult`
  (``.inference``, plus ``flags`` and ComBat diagnostics). The legacy v1.x
  keys ``'g2>g1'`` / ``'g1>g2'`` remain readable with a
  :class:`DeprecationWarning` until v2.1. F-stat omnibus tests return
  ``{'omnibus': arr}``.
- Every entry point shares the defaults in :mod:`conninfpy.defaults`
  (1000 permutations, exact empirical p-values).

t-test pipeline — :func:`compute_p_val`
---------------------------------------

The simplest entry point. Computes per-edge t-statistics, applies an
enhancement operator, builds a permutation null of per-tail max
statistics, and returns FWER- or FDR-corrected p-maps with the +1
Phipson–Smyth correction.

.. code-block:: python

   from conninfpy import compute_p_val, fisher_r_to_z

   group1_z = fisher_r_to_z(group1_corr)   # (n1, N, N)
   group2_z = fisher_r_to_z(group2_corr)

   p = compute_p_val(
       group1_z, group2_z,
       test_type='two-sample',             # or 'paired', 'one-sample'
       method='tfnbs',
       n_permutations=1000,
       e=0.3, h=3.0, n=10,                 # FDR-calibrated regime
       use_mp=True, rng=42,
   )
   # p['positive'] : (N, N) p-map for group2 > group1
   # p['negative'] : (N, N) p-map for group1 > group2

GLM pipeline — :func:`compute_p_val_glm`
----------------------------------------

Edge-wise GLM with Freedman–Lane permutation. Use this when you have a
continuous predictor and want to control for nuisance regressors
(motion, age, sex, site).

Convenience API (recommended):

.. code-block:: python

   import numpy as np
   from conninfpy import compute_p_val_glm, fisher_r_to_z

   Y = fisher_r_to_z(connectivity_matrices)              # (n, N, N)
   age = np.array([25, 30, 28])                          # (n,)
   confounds = np.column_stack([motion, sex])            # (n, p)

   p = compute_p_val_glm(
       Y, interest=age, confounds=confounds,
       stat_type='tstat',                                 # or 'beta', 'fstat'
       method='tfnbs',
       n_permutations=1000,
       e=0.3, h=3.0, n=10,
       use_mp=True, rng=42,
   )
   # p['positive'] : edges where age ↑ → connectivity ↑
   # p['negative'] : edges where age ↑ → connectivity ↓

Advanced API (full design matrix + contrast):

.. code-block:: python

   from conninfpy import build_design_matrix, compute_p_val_glm

   X, contrast = build_design_matrix(interest=age, confounds=motion_sex)
   p = compute_p_val_glm(
       Y, design_matrix=X, contrast=contrast,
       stat_type='tstat', method='tfnbs', n_permutations=1000,
   )

F-contrast (omnibus) for joint multi-row tests — e.g. ≥3-condition
designs or testing several predictors jointly:

.. code-block:: python

   X = np.column_stack([np.ones(n), age, age**2, sex, motion])
   contrast = np.array([
       [0, 1, 0, 0, 0],       # beta_age
       [0, 0, 1, 0, 0],       # beta_age_squared
   ])
   p = compute_p_val_glm(
       Y, design_matrix=X, contrast=contrast,
       stat_type='fstat', method='tfnbs', n_permutations=1000,
   )
   # F-pipeline returns a single non-negative tail:
   # p['omnibus'] : (N, N) FWER-corrected p-map

Multi-contrast GLM in one pass — :func:`compute_p_val_glm_multi`
----------------------------------------------------------------

If you need to test several contrasts of interest under the same
nuisance model — e.g. ``age``, ``sex``, and ``mean_fd`` separately
while treating the others as nuisance — calling
:func:`compute_p_val_glm` once per contrast wastes work: the
reduced-model residual fit and the per-permutation
``X_pinv @ Y_perm`` matrix multiplication are identical. The
multi-contrast wrapper does it in one pass.

.. code-block:: python

   from conninfpy import compute_p_val_glm_multi

   X = np.column_stack([np.ones(n), age, sex, mean_fd])  # (n, 4)
   contrasts = {
       "age":     np.array([0.0, 1.0, 0.0, 0.0]),
       "sex":     np.array([0.0, 0.0, 1.0, 0.0]),
       "motion":  np.array([0.0, 0.0, 0.0, 1.0]),
   }

   results = compute_p_val_glm_multi(
       Y, design_matrix=X, contrasts=contrasts,
       method="tfnbs", n_permutations=5000,
       acceleration="gpd", rng=42,
   )
   # results = {'age': InferenceResult, 'sex': ..., 'motion': ...}
   results["age"].positive          # (N, N) FWER p-map
   results["age"].n_significant(0.05)

For ``K`` contrasts the wall-time is roughly that of a single
:func:`compute_p_val_glm`` call rather than ``K`` calls — typically a
3× speedup for the canonical age + sex + motion design.

By default the reduced model excludes any column touched by *any*
contrast in the dictionary. Pass ``nuisance_contrast=`` to override
explicitly, e.g. when you want sex and motion treated as nuisance for
the age contrast even though you also test them separately. F-stat /
multi-row contrasts are unsupported in this wrapper — call
:func:`compute_p_val_glm` once per omnibus test.

Paired A vs B with Δ-level confounds — :func:`compute_p_val_paired_glm`
------------------------------------------------------------------------

When you have a within-subject contrast (task A vs B) and confounds
that differ between conditions (e.g. condition-level motion):

.. code-block:: python

   from conninfpy import compute_p_val_paired_glm

   p = compute_p_val_paired_glm(
       Y_task_A, Y_task_B,                            # both (n, N, N)
       confounds_A=np.column_stack([motion_A, drowsiness_A]),
       confounds_B=np.column_stack([motion_B, drowsiness_B]),
       method='tfnbs', n_permutations=1000,
       e=0.3, h=3.0, n=10,
   )

With no confounds, this delegates to ``compute_p_val(test_type='paired')``
(sign-flip permutation, the exact non-asymptotic null). With Δ-level
confounds it constructs Δ_Y = Y^B − Y^A, Δ_C = C^B − C^A and runs a
one-sample GLM on the differences. On both routes ``positive`` means
Y^B > Y^A.

Methods (enhancement operators)
-------------------------------

``compute_p_val(..., method=...)`` and ``compute_p_val_glm(..., method=...)``
both accept:

==================  ================================================================
``method``          Operator
==================  ================================================================
``'tstat'``         No enhancement; per-tail max-statistic FWER on raw t / β / F
``'tfnbs'``         Threshold-Free NBS (Baggio 2018)
``'nbs'``           Classical Network-Based Statistic (Zalesky 2010); supply
                    ``threshold=2.0`` and ``nbs_stat='extent'`` or ``'intensity'``
``'cnbs'``          Constrained NBS (Noble & Scheinost 2020); supply ``net_labels=``
``'ni_tfnbs'``      Network-Informed TFNBS (this work); soft block-density prior;
                    supply ``net_labels=``
``'fbc_tfnbs'``     Functional-Block-Clustering TFNBS (this work); hard block
                    prior; supply ``net_labels=``, ``min_cluster_size=3``
``'bh_fdr'``        Parametric Benjamini–Hochberg FDR (no permutation)
``'bh_fdr_perm'``   Empirical edge-wise BH-FDR via permutation null
==================  ================================================================

Multi-site harmonization — :mod:`conninfpy.harmonize`
-----------------------------------------------------

Native NumPy implementation of parametric empirical-Bayes ComBat.
No ``neuroHarmonize`` or ``neurocombat`` dependency.

End-to-end with confound-aware GLM:

.. code-block:: python

   import numpy as np
   from conninfpy import (
       fisher_r_to_z,
       combat_harmonize, design_diagnostics,
       compute_p_val_glm,
   )

   # Y_corr : (n, N, N) per-subject Pearson correlation matrices
   # sites  : (n,) acquisition-site labels
   # age, sex, mean_fd : per-subject phenotype
   Y = fisher_r_to_z(Y_corr)

   # 1. Harmonize site effects, preserving age + sex + motion as biology
   preserve = np.column_stack([age, sex, mean_fd])
   Y = combat_harmonize(Y, sites=sites, preserve=preserve).Y_adjusted

   # 2. Sanity-check the design before running anything expensive
   X = np.column_stack([np.ones_like(age), age, sex, mean_fd])
   diag = design_diagnostics(X, names=["intercept", "age", "sex", "mean_fd"])
   # reports condition number + per-column VIF + plain-English flags

   # 3. TFNBS on age, partialling out sex + motion (Freedman–Lane)
   p = compute_p_val_glm(
       Y, interest=age,
       confounds=np.column_stack([sex, mean_fd]),
       stat_type='tstat', method='tfnbs',
       e=0.3, h=3.0, n=10,
       n_permutations=200, acceleration='gpd',
       use_mp=True, rng=42,
   )

For cross-site machine-learning transfer (fit ComBat on training
cohorts, freeze, apply to held-out sites at test time), use
:func:`combat_fit` / :func:`combat_apply` separately.

Permutation acceleration — :mod:`conninfpy.acceleration`
--------------------------------------------------------

Set ``acceleration='gpd'`` (or ``'gamma'``) on either pipeline to
replace the empirical permutation p-value formula with a fitted
parametric tail (Generalized Pareto Distribution, Winkler 2016b).
Reduces the perm budget from ~5000 to ~200 with ~25× wall-clock saving.
Reproduces the empirical FWER-corrected p-values to within
``|Δ(-log10 p)| ≤ 0.001`` on >99% of edges in the ConnInfPy ABIDE Age
validation. A goodness-of-fit guard (Anderson–Darling on tail
exceedances) falls back to empirical p-values when the GPD does not
fit cleanly.

Topology-aware default integration steps
----------------------------------------

The TFCE integral is evaluated at ``n`` thresholds. Defaults:

- ``n=100`` for direct scoring (single-shot, high resolution; via
  :func:`conninfpy.get_tfnbs_score`)
- ``n=10`` inside the permutation loop (Hao 2024 reports n=10 is
  sufficient for FDR control on network data)

The exponents ``(e, h)`` accept either scalars or equal-length lists.
Lists are zipped pairwise into a single 3-D batched call, so a 16×11
(E, H) grid runs 176 combos in one ``compute_p_val`` invocation without
multiplying the runtime.

See :mod:`conninfpy.defaults` for the full list of constants and
citations (Smith & Nichols 2009, Baggio 2018, Vinokur 2023, Hao 2024).


Synthetic data and scenarios — :mod:`conninfpy.sim`
---------------------------------------------------

Planted-effect data for tutorials, power analysis and method comparison:

.. code-block:: python

   import conninfpy

   # Two groups of correlation matrices with a planted difference
   # (needs the [full] extra for scikit-learn)
   g1, g2, _ = conninfpy.sim.generate_fc_matrices(
       N=60, effect_size=0.3, n_samples_group1=30, n_samples_group2=30, seed=0)

   # 20 controlled effect topologies (hub, rich club, chain, gradient, ...)
   print(conninfpy.sim.list_scenarios())
   gen = conninfpy.sim.TopologyDatasetGenerator(n_nodes=100, n_modules=7, seed=42)
   ds = gen.generate(conninfpy.sim.get_scenario("within_module_dense"),
                     effect_size=0.3, n_samples=30, time_points=120)
   # ds.group1 / ds.group2: correlations with a zeroed diagonal → input='r'
   out = conninfpy.analyze(group1=ds.group1, group2=ds.group2, input="r",
                           test_type="two-sample", method="tfnbs")

Enhancement without permutation — :mod:`conninfpy.enhance`
----------------------------------------------------------

Apply an enhancement operator to a statistic map you computed yourself
(scores only, no p-values):

.. code-block:: python

   from conninfpy import enhance, stats

   t = stats.compute_t_stat(group1_z, group2_z, test_type="two-sample")
   scores = enhance.apply_tfnbs(t, e=0.3, h=3.0, n=100)   # {'positive', 'negative'}


Command line — ``conninfpy run``
--------------------------------

An analysis file (YAML) names a data manifest, the design and the inference
settings; ``conninfpy run`` loads the data, calls :func:`conninfpy.analyze`
and writes ``edges.csv`` (significant edges, atlas-annotated when an atlas is
known), ``pvals.npz`` and ``provenance.json`` (package version, analysis file,
data summary, warnings, run time):

.. code-block:: yaml

   data: ../../datasets/abide_mini.yaml   # manifest, relative to this file
   design:
     type: glm                            # glm | two-sample | paired | one-sample
     interest: Age                        # a list tests several predictors in one pass
     confounds: [Sex, Motion_FD]          # text columns are one-hot encoded
     sites: site
   inference:                             # any conninfpy.analyze() keyword
     method: tfnbs
     n_permutations: 200
     acceleration: gpd
     rng: 42
   atlas: schaefer_100_yeo7               # built-in name or CSV path
   output: results/abide_mini_age

.. code-block:: bash

   conninfpy run examples/configs/abide_mini_age.yaml [--out DIR] [--n-permutations N] [--no-mp]
   conninfpy info

Two-sample designs use ``group:`` and ``levels: [group1, group2]``
(``positive`` = second level higher); paired designs use ``condition:``,
``levels:`` and ``subject:`` (default ``subject_id``) and pair rows by subject.
Rows with a missing value in any column the design uses are dropped and
counted in ``provenance.json``. ``python -m conninfpy`` is equivalent.

