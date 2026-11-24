How Inference Works
===================

This page explains the **approach** behind ConnInfPy — the reasoning that
connects "connectivity matrices and a hypothesis" to "these edges are
significant." It is conceptual rather than formal: the aim is to make clear
*why* each step is present. For the recipes (which function to call) see
:doc:`workflows`; for the equations and citations see :doc:`references`.

The package applies a single inferential scheme across designs:

.. code-block:: text

   a statistic at each edge  →  network enhancement  →
   a null distribution from relabelling  →  comparison of observed to null

The remainder of this page covers each of these four steps and the reasoning
behind it.


Rationale for permutation testing
---------------------------------

A connectome has many edges (:math:`N(N{-}1)/2`), and they are **not
independent**: neighbouring regions covary, hubs participate in many edges, and
the correlations are bounded and non-normal. A parametric per-edge p-value
assumes a known distribution and independent tests; neither holds here, and
across thousands of edges even a small per-edge error rate yields many false
positives.

Permutation testing avoids both assumptions. Rather than *assume* a null
distribution, the method **constructs** one: the data are relabelled so that
the effect is removed, the full statistic is recomputed, and the process is
repeated. The only requirement is **exchangeability** — under the null, the
relabelled data are as probable as the observed data. This is a substantially
weaker assumption than normality, and the resulting null automatically reflects
the empirical covariance among edges.


Step 1 — a statistic at each edge
---------------------------------

At every edge a single statistic quantifies the strength and reliability of the
effect. The choice of statistic follows the design:

- **Two groups** → a *t*-statistic of the difference in means: Welch's by
  default (separate group variances), or Student's with ``var_equal=True``
  (pooled variance, as in the MATLAB NBS toolbox). The group-label
  permutation is the same for both; with equal group sizes they coincide.
- **Same subjects, two conditions** → a one-sample *t* of the within-subject
  differences.
- **A predictor with confounds** (e.g. age, controlling for motion and sex) →
  a linear model fit at each edge, reading off the *t* for the predictor of
  interest.

Direction is meaningful, so ConnInfPy always splits the statistic into two
**separate** tails — ``positive`` (e.g. group2 > group1, or predictor ↑ →
connectivity ↑) and ``negative`` — and propagates them independently. Testing a
single two-sided statistic would allow a positive and a negative block to
cancel.

.. code-block:: python

   from conninfpy import compute_t_stat, fisher_r_to_z
   t = compute_t_stat(fisher_r_to_z(g1), fisher_r_to_z(g2),
                      test_type='two-sample')
   # t['g2>g1'] is the per-edge statistic for "group2 > group1"

Step 1 thus reduces the data to one strength-map per direction.


Step 2 — network enhancement
----------------------------

A connectivity effect rarely appears as a single isolated edge; it typically
appears as a **connected set** of edges that covary. An isolated edge with a
large statistic may be noise, whereas a coherent set of moderate edges is more
convincing evidence. **Enhancement** rewrites each edge's score to reflect the
network structure it participates in; the operators differ in how they define
and summarise that structure.

A useful way to picture enhancement: take the strength-map from step 1 and
progressively raise a threshold (a "water level"). At a low threshold most
edges are supra-threshold and form large connected components; as the threshold
rises, only the strongest edges remain, in small components. The operators
differ in how they read these components.

No enhancement — ``tstat``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each edge is tested on its own, with no network aggregation. Useful as a
baseline and for strong focal effects, but insensitive to the distributed weak
effects typical of brain connectivity — which the operators below are designed
to recover.

Fixed threshold — ``nbs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classical Network-Based Statistics. Fix a single threshold, identify the
connected components above it, and score each edge by the size of its component
(extent) or by the summed excess of the statistic over the threshold,
:math:`\sum (t - \tau)`, within it (intensity — only the excess above
:math:`\tau` counts, so an edge just above the threshold adds almost nothing
and a small cluster of strong edges can outrank a large cluster of weak ones).
It is most powerful when the effect forms a large connected component above
the chosen threshold.
The threshold is arbitrary, however, and results can vary substantially with it
(Vinokur et al. 2023 report up to 75-fold variation in detected edges across
reasonable thresholds); report results at several thresholds when using it.

The inference unit is the component. Its statistic is compared with the
permutation distribution of the *largest* component statistic (family-wise
error control over components; see Step 4): with ``two_tailed=True`` (the ``analyze()`` default) the two
tails share one null, the larger of the two tails' maxima in every
permutation; with separate one-sided families (the low-level pipelines'
default, ``two_tailed=False``) each tail has its own null of maxima. Every
edge of a component carries the
component's p-value, so a significant component is a claim about the set, not
about each of its edges. ``result.component_report`` (and
``result.component_table(atlas)``) lists every observed component of each tail:
its ``component_id`` (numbered within the tail by decreasing statistic), edges,
ROIs, extent or intensity, and ``p_fwer``; with an atlas also the ROI names and
the networks and network pairs it spans. Components that are not significant
are listed too (``significant`` marks ``p_fwer <= alpha``); a single
suprathreshold edge is a component of one edge. The ``component_id`` column of
``significant_edges()`` and of the CSV export gives the component of each edge.
The p-values are those of the p-value map: the report recomputes nothing.
``result.node_summary(atlas)`` describes the same detections per ROI (which ROIs
gather them): for edge-wise methods the significant edges of each ROI out of its
*tested* incident edges; for ``nbs`` and ``cnbs``, whose edges in a component or
block share one p-value, the significant components or blocks containing the ROI
and how many of its edges lie in them, never a count of "significant edges" or a
mass of -log10 p. It is descriptive and adds no significance.

Threshold-free — ``tfnbs`` (the default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rather than fix a single threshold, TFNBS integrates the cluster evidence over
the full range of thresholds:

.. math::

   \text{score}(e) \;=\; \int_{h_0}^{w_e}
       (\text{edge count in the component containing } e \text{ at } h)^{E}\, h^{H}\, dh

At low thresholds an edge accrues weight from belonging to a spatially extended
component; at high thresholds from its own magnitude. Summing across thresholds
rewards an edge that is extended *or* strong, removing the dependence on any
single cutoff. ``E`` controls the contribution of component size and ``H`` that
of statistic height; they act as sensitivity parameters rather than fixed
constants, and can be swept in one permutation pass to report stability
(:ref:`wf-eh-stability`). This is the recommended default when there is no
strong prior about where the effect is located.

Using a known network partition — ``cnbs``, ``ni_tfnbs``, ``fbc_tfnbs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given an atlas partition (e.g. Yeo-7 networks, lobes, or modules), the operator
can be directed to expect effects that respect those blocks:

- ``cnbs`` (constrained NBS; Noble & Scheinost, 2020) — averages the
  **signed** edge statistic over every edge of each within- and
  between-network block, tests each block against its own permutation null
  (the same block's mean in every permuted map), and corrects the block
  p-values across blocks (``cnbs_correction='bonferroni'`` for FWER, the
  default, or ``'bh'`` for FDR). The inference unit is the block: every
  edge of a significant block carries the block's p-value, and
  ``result.block_report`` lists the blocks. A significant block means its
  *mean* effect is nonzero, not that each of its edges is affected; edges
  that go in opposite directions cancel. Most powerful for block-aligned
  effects such as within-network hypoconnectivity; it can miss effects that
  are small parts of large blocks. With K blocks and Bonferroni the
  smallest attainable p-value is K/(B+1), so use enough permutations.
- ``ni_tfnbs`` (network-informed, **soft** prior) — integrates an edge-level
  block-density score over thresholds, without connected-component clustering.
  At threshold h, k_b(h) eligible edges are active in block b and the extent
  weight is k_b(h)/sqrt(M_b) by default (``normalization='linear'`` uses
  k_b(h)/M_b; ``'none'`` uses k_b(h)). M_b is the fixed number of eligible
  undirected opportunities: n_a*n_b between networks and n_a*(n_a-1)/2 within
  a network when all pairs are eligible. Available-case inference excludes
  untested pairs from both capacity and active counts. The same capacity is
  used for observed and permuted maps. Its behavior depends on the partition;
  a misspecified partition does not generally reproduce plain TFNBS.
- ``fbc_tfnbs`` (functional-block-clustering, **hard** prior) — treats each
  block as a single atomic cluster and requires a minimum number of active
  edges per block. It targets diffuse, block-confined effects that are spread
  too thinly to form a strong connected component.

Choosing an operator
~~~~~~~~~~~~~~~~~~~~~~

No operator is universally optimal: each encodes an assumption about the
*spatial shape* of the effect, and is strongest when that assumption holds and
weakest otherwise.

.. list-table::
   :header-rows: 1
   :widths: 18 30 30 22

   * - Operator
     - Assumes the effect is…
     - Best for
     - Prior needed
   * - ``tstat``
     - focal / very strong
     - baseline reference
     - none
   * - ``nbs``
     - one large connected component
     - strong connected effects
     - a threshold
   * - ``tfnbs``
     - connected at *some* scale
     - mixed focal + extended (default)
     - none
   * - ``cnbs``
     - a block's mean effect (inference per block)
     - block-wide effects, high power
     - ``net_labels``
   * - ``ni_tfnbs``
     - block-concentrated (soft)
     - low false-discovery priority
     - ``net_labels``
   * - ``fbc_tfnbs``
     - diffuse within a block (hard)
     - thin, block-confined effects
     - ``net_labels``

Because the choice encodes an assumption, the recommended practice is to report
two operators: a topology-agnostic reference (``tfnbs``) and the network-aware
operator matching the prior. Agreement between them strengthens a finding;
disagreement is itself informative about the effect's spatial shape.


Step 3 — the null distribution from relabelling (permutation schemes)
---------------------------------------------------------------------

Given the enhanced score-map from the observed data, its significance is
assessed against the distribution expected under the null. That distribution is
constructed by relabelling the data so as to remove the effect while preserving
all other structure, rerunning steps 1–2, and repeating many times to obtain a
collection of null score-maps.

The principle underlying every scheme is **exchangeability**: under the null,
certain labels are interchangeable, and a valid permutation is any reshuffle
that respects that interchangeability. The correct scheme is the relabelling
that removes exactly the tested effect while leaving every nuisance structure
intact — too little randomisation leaves the effect in the null (low power);
too much scrambles structure that should be held fixed (mis-calibrated
false-positive rate). Each design has its own scheme.

Two groups — shuffle the group labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Removes:* the association between subject and group label. *Preserves:* each
subject's connectivity matrix. Under the null the two groups are exchangeable,
so the observed assignment is one of many equally probable assignments; the
null is generated by drawing other assignments and recomputing.

Paired conditions — sign-flip each subject's difference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Removes:* the direction of each within-subject change. *Preserves:* the
pairing and each subject's magnitude. The response is the per-subject
difference ``B − A`` (``group2 − group1``) on the analysis scale — for
correlations, Fisher z of each condition before the subtraction. If the two
conditions are interchangeable within each subject, the labelling of each
subject's difference as "A−B" or "B−A" is equiprobable, so each subject's
difference matrix is randomly negated, subject by subject, independently.

What the null needs is an **invariance of the joint law**: the stacked
difference matrices of all subjects must have the same distribution after
any pattern of per-subject sign flips. Independent subjects whose difference
matrices are centrally symmetric (each matrix distributed like its own
negation) satisfy this; within-subject exchangeability of A and B is one way
to get there. Unequal variances of A and B do not by themselves break it — a
symmetric difference can have asymmetric condition marginals — and nothing
else about the marginal laws is assumed. Without the invariance, exactness
and Type-I control of the sign-flip test are not guaranteed, and a
condition-varying covariate does not repair that (the Freedman–Lane route
below rests on its own, approximate, premise). With all :math:`2^n` sign
patterns enumerated the test is exact under the invariance; the engine
samples patterns at random, which gives the Monte Carlo form of Step 4. For
edge-wise methods (``tstat``, the TFNBS family) the hypothesis is about the
**mean** difference: a significant edge means that the average within-subject
change differs from zero, not that every subject changes.

Predictor with confounds — Freedman–Lane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A natural but incorrect approach is to shuffle the predictor of interest
directly. When the predictor is correlated with the confounds (e.g. age with
motion), shuffling it also destroys that correlation, so the permuted data no
longer match the model under the null and the test is mis-calibrated.
Freedman–Lane permutes only the part of the data the confounds cannot account
for:

#. Fit the reduced model (confounds only). Decompose each edge's data into the
   fitted part (explained by the confounds) and the residual.
#. Permute the residuals, then add the fitted part back, unchanged.
#. Recompute the predictor's statistic on the reconstructed data.

*Removes:* any genuine predictor effect (the residuals are now in random
subject order). *Preserves:* the confound structure, held fixed. Only the
residual — the component of the data the predictor of interest could explain —
is permuted, so the resulting null isolates the tested effect while controlling
the confounds.

Repeated measures with condition-varying confounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The within-subject difference is formed first. Any quantity constant per
subject — including additive **site** effects — cancels in the subtraction.
Freedman–Lane is then applied to the differences to partial out the component
of the confound that varies between conditions. The two adjustments are applied
in the order that makes each straightforward.

Several predictors at once
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each predictor is tested against its own Freedman–Lane null: its reduced
model keeps the other predictors as nuisance, so a real effect of one
predictor stays in the fitted part while another is tested. The permutations
are shared across predictors, so each result equals a separate single-predictor
run with the same seed (see :ref:`wf-multi-interest`). A single reduced model
that removes every predictor at once would test the joint null "all of them
are zero" instead, which is conservative when one of them has an effect.

Multiple sites — permute within site
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subjects from different scanners are **not** freely exchangeable, since a site
can shift the connectivity baseline. Passing ``sites=`` confines every
permutation to occur within each site block, never across blocks. *Removes:*
the effect, within each site. *Preserves:* the between-site differences, which
therefore remain in the null rather than leaking into the effect and inflating
false positives. This is the permutation component of multi-site handling;
ComBat (below) is the complementary harmonization component.

.. code-block:: python

   from conninfpy import compute_p_val, compute_p_val_glm, fisher_r_to_z
   import numpy as np

   # two groups: group labels are permuted internally
   compute_p_val(g1, g2, test_type='two-sample', method='tfnbs', rng=42)

   # paired: each subject's difference is sign-flipped internally (Monte Carlo sign-flip null)
   compute_p_val(rest, task, test_type='paired', method='tfnbs', rng=42)

   # predictor + confounds: Freedman–Lane handles the relabelling
   compute_p_val_glm(fisher_r_to_z(Y), interest=age,
                     confounds=np.column_stack([sex, motion]),
                     method='tfnbs', rng=42)

   # multi-site: pass sites= and permutation stays within each site block
   compute_p_val_glm(fisher_r_to_z(Y), interest=age,
                     confounds=np.column_stack([sex, motion]),
                     sites=site, method='tfnbs', rng=42)


Step 4 — comparison of observed to null
---------------------------------------

The max-statistic methods — ``tstat``, the TFNBS family and ``nbs`` — retain
a single value from each relabelled null map: the maximum enhanced score over
all edges (for ``nbs`` the enhancement has already copied each component's
statistic onto its edges, so this is the largest component statistic). Across
permutations these maxima form the null distribution of the largest score
attributable to noise alone. A unit is declared significant when its
statistic reaches this null-maximum distribution at the chosen α, and its
p-value is written onto every edge of the unit. The unit is
``result.inference_unit``: ``'edge'`` for ``tstat`` and the TFNBS family,
``'component'`` for ``nbs``. The p-value of a unit is the proportion of
permutation maxima that match or exceed its statistic:

.. math::

   p(e) \;=\; \frac{1 + \#\{\text{permutations whose max} \ge \text{score}(e)\}}
                    {1 + (\text{number of permutations})}

Two methods stand outside this formula. ``cnbs`` compares every block's
statistic with that block's *own* permutation null (the same block's value
in every permuted map, in absolute value under ``two_tailed=True``) and then
corrects the block p-values across blocks with Bonferroni (FWER over blocks)
or Benjamini–Hochberg (FDR over blocks); its unit is ``'block'``.
``bh_fdr_perm`` keeps a per-edge marginal permutation p-value and applies
Benjamini–Hochberg across the eligible edges; ``bh_fdr`` and ``bonferroni``
are parametric and use no permutations.

Three design choices are embedded in the formula:

- **Why the maximum?** Comparing every unit to the distribution of the
  *largest* null score keeps the probability of **any** false positive at the
  inference unit at most α **under the global null** (no effect anywhere) —
  weak family-wise error control — without requiring knowledge of the
  inter-edge covariance, which is already reflected in the permutations.
  Whether a p-value ≤ α means this in the data at hand — here and for the
  FDR variants — depends on the permutation scheme being valid: the
  relabelling of Step 3 must leave the data, under the null, distributed as
  the observed data, and the reconstructed null maps must reproduce the
  sampling law of the statistics. Sign-flip and group-label permutation do
  this under their exchangeability premises; Freedman–Lane is an
  approximation (Winkler et al. 2014), so every GLM p-value inherits that
  approximation. **Strong** control (the same bound under any mix of true
  and false nulls) needs more: the *joint* distribution of the true-null
  units' statistics must be the same whatever the other units are — subset
  pivotality (Westfall & Troendle 2008; Rempala & Yang 2013) — on top of a
  valid scheme. Invariance of each marginal null law alone is not enough,
  and an edge-wise *t* by itself does not establish it for these statistics;
  for the TFNBS family an edge's enhanced score depends on its neighbours'
  statistics, and for ``nbs`` the unit is the component, so a significant
  component is a claim about the set and never about each of its edges. The
  package asserts strong control for none of its methods; read detections at
  ``result.inference_unit``, which ``component_report`` / ``component_table``
  and ``node_summary`` keep.
- **Why the** ``+1``\ **?** The observed statistic is itself one admissible
  relabelling and is counted (Phipson & Smyth 2010; Hemerik & Goeman 2017).
  With relabellings drawn at random with replacement the test is a Monte
  Carlo test: the ``+1`` rule keeps it valid at level α given a valid scheme
  and keeps the p-value strictly positive, whereas the naive proportion is
  slightly liberal. It does not turn a sampled null into the enumerated one —
  exhaustive enumeration of all relabellings is a different procedure, exact
  under its premises, and is not what the engine does.

To control the *false-discovery rate* instead (greater sensitivity at the cost
of some false positives), ``method='bh_fdr_perm'`` replaces the max-statistic
step with a per-edge permutation p-value and a Benjamini–Hochberg procedure.


Auxiliary components: harmonization and acceleration
----------------------------------------------------

**ComBat (multi-site harmonization).** Different scanners introduce additive
and multiplicative offsets that can exceed the biological signal. ComBat
estimates and removes these per-site offsets *before* step 1, while preserving
the covariates of interest, and excludes the *tested* variable from its model
so that the harmonization never incorporates the labels the permutation will
later shuffle (which would otherwise inflate false positives). The site
strategies and their trade-offs are described in :ref:`wf-glm-sites`.

**Acceleration (GPD / gamma tail, opt-in).** With J permutations the empirical
p-value of a unit rests on how many null maxima reach its statistic, and its
resolution is 1/(J+1). ``acceleration='gpd'`` (default ``None``) fits a
generalized Pareto distribution to the exceedances over a sequence of
thresholds, starting at the upper quartile of the null maxima and moving
upwards until a method-of-moments fit passes an Anderson–Darling check
(Winkler et al., 2016; Knijnenburg et al., 2009); ``acceleration='gamma'``
fits a gamma distribution to all null maxima by their moments. Either replaces
the p-value **only of units that fewer than 10 null maxima reach** (for the
GPD, only units at or above the fitted threshold); every other p-value stays
the empirical permutation p-value, a failed fit leaves the empirical p in
place, and no binomial interval is reported for an accelerated result.
The replaced values are model-based tail estimates that can be smaller than
1/(J+1); their accuracy is not established by the number of permutations, and
the option exists for exploration with a few hundred permutations — reported
results use the empirical default. Known limits of the 2.0.0 implementation:
a replaced p-value is capped at min(1, (min_exceedances + 1)/(J + 1)), which
can be *larger* than the empirical p-value of the same unit — the replacement
can raise a p, for GPD and gamma alike; the gamma branch has no
goodness-of-fit check and no eligibility filter, and a statistic below its
fitted support gets a tail survival function of 1, recorded as that cap —
which is 1 when J equals min_exceedances (reproduced in a helper-level example
with J = 10; not observed at J = 200/1000 in a limited check); with a bounded
(ξ < 0) GPD fit, one unit beyond the fitted endpoint cancels the fit for the
whole matrix at that threshold, and when no threshold is accepted the
empirical p-values are returned unchanged; in the GLM and missing-data paths
the acceleration is applied to every statistic type without a mode check,
unlike the pairwise max-statistic path; and with ``method='cnbs'`` it is not
applied (a warning is logged). The Anderson–Darling threshold 2.5 is a lenient
heuristic filter that has not been calibrated for the method-of-moments fit,
and the moment conditions on the shape parameter are not checked. No FWER,
power or anti-conservativeness claim is made for accelerated p-values.

**Directional families.** ``two_tailed=True`` declares one two-sided family
for pairwise and GLM inference. Max-statistic methods pool positive/negative
null maxima, cNBS tests absolute block means before across-block correction,
and pairwise BH methods correct two-sided marginal p-values across eligible
edges. Directional maps retain the p-value in the observed direction and 1
in the opposite direction. :func:`~conninfpy.analysis.analyze` defaults to
this two-sided family: one nominal α applies to the error measure of the
chosen method — FWER for the max-statistic methods and ``bonferroni``, FDR
for the BH variants — over both directions jointly, and only under that
measure's premises (above). Its ``two_tailed=False`` keeps the legacy
separate one-sided families, each corrected at the nominal α; two families
at α = 0.05 do not promise a combined-direction FWER of 0.05 — for FWER,
α/2 per tail bounds the combined FWER at α by Bonferroni (under the same
premises); this is not an FDR statement. The low-level
:func:`~conninfpy.compute_p_val` / :func:`~conninfpy.compute_p_val_glm`
entry-points keep ``two_tailed=False`` as their own default.

For a single prespecified direction (e.g. controls > patients, as in the
MATLAB NBS toolbox's t-test), declare it before the run:
``analyze(..., tested_tail='positive')`` (or ``'negative'``). The run uses
per-tail nulls, the tested tail keeps its own-null p-values at the nominal
alpha, the other direction is reported as not tested (p = 1), and the
result records ``tail_policy = 'one-sided-positive'`` with
``tail_selection = 'prespecified'``, which provenance, exports and the app
carry through. Choosing a direction only after inspecting a finished run is
a post hoc selection: ``result.one_sided('positive')`` performs it and is
marked ``tail_selection = 'post-hoc'`` instead, so a restriction made after
seeing the p-values is never reported as a hypothesis fixed in advance.
Each enhancement setting and contrast is corrected separately; selecting
among settings or contrasts requires an additional declared correction.


End-to-end example
------------------

A single ``analyze()`` call runs all four steps, with ComBat and within-site
permutation included:

.. code-block:: python

   from conninfpy import analyze
   import numpy as np

   out = analyze(
       Y,                                   # raw correlations (Fisher-z applied)
       interest=diagnosis,                  # step 1: a t at each edge
       confounds=np.column_stack([age, sex, mean_fd]),
       sites=site,                          # ComBat + within-site permutation
       harmonize='combat_site_dummies_glm',
       method='tfnbs',                      # step 2: threshold-free enhancement
       n_permutations=5000, acceleration=None,   # steps 3–4: empirical null
       rng=42,
   )
   out.inference['positive']                # FWER p-map, diagnosis ↑ → conn ↑

The call composes the four steps: a *t*-statistic at each edge (controlling for
age, sex, and motion) → TFNBS enhancement over connected components → a
within-site Freedman–Lane null on the harmonized data → a family-wise-corrected
p-value from the max-statistic distribution with the ``+1`` correction. See
:doc:`workflows` for the design variants and :doc:`references` for the
supporting literature.


After inference: anatomical reading
-----------------------------------

The four steps end with corrected p-values on the parcellation the signals
were measured with. Reading the result anatomically
(:mod:`conninfpy.interpret`; :ref:`usage-interpret`) is a descriptive layer
that changes none of them:

- **The tissue, not the centre point.** Each ROI is described by its overlap
  with the 246 regions of the Brainnetome atlas (Fan et al., 2016) over the
  sphere or parcel that produced its signal. A 6 mm sphere usually covers 2–4
  Brainnetome regions and a Schaefer parcel often spans several subregions,
  so a label is given only at the finest level of the hierarchy (subregion →
  gyrus → lobe) that holds at least half of the ROI's labelled tissue; below
  that the ROI is ``mixed``.
- **The inference unit stays.** Grouping detections by lobe, gyrus, circuit
  class, hemispheric or network relation re-describes them. It adds no test,
  and an edge of a significant component or block does not become
  individually significant.
- **Against what was tested.** Detections are shown next to the tested
  opportunities in each category, because categories differ in size (there
  are far more cortico-cortical than thalamo-cortical pairs).
- **Literature context, not function.** BrainMap behavioural-domain terms are
  forward-inference associations of each region in published activation
  studies (Laird et al., 2005); they are listed per region and never ranked
  into "the function" of a region or a connection.

Measuring the signals in Brainnetome regions (native parcellation) is a
different analysis: it changes the signals and the tested edge family.
``examples/open_close/atlas_stability`` compares the two on the Open-Close
contrast.
