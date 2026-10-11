# Edge-model feature reproducibility

Fit eyes-open/eyes-closed classifiers on **individual connectivity edges**,
then ask which learned features are reproducible across participant bootstraps.
Summarize the **same fitted edge patterns** at individual-edge, functional
network-pair and anatomical region-pair levels. Aggregation is applied after
fitting, so it changes the resolution of interpretation while predictions and
fitted models stay fixed.

There are two classifiers per bootstrap: raw edges and ComBat-adjusted edges.
There are no classifiers trained on functional/anatomical means, difference
tests, significance masks, or perturbation experiments. Top-feature sets are
used to audit interpretation; they do not select inputs to the classifiers.

## Data, fitting and participant bootstrap

Use the existing `datasets/open_close/` Schaefer-200 `strategy-4_GSR` time
series and loader. Each participant contributes one EO=0 and one EC=1 scan.
Connectivity is Pearson correlation followed by Fisher z; GSR is already
applied. IHB has 84 participants. China uses the same 47 complete participants
for either EC run, excluding `sub-3258811`; the counterbalanced EC run1 is
primary, and earlier always-closed run0 is a sensitivity.

1. Fix one site-stratified participant split, seed 42: development = 59 IHB
   + 33 China; test = 25 IHB + 14 China (39 people, 78 scans). Both scans of
   each participant stay together.
2. Tune L2 logistic C from `{0.01, 0.1, 1, 10}` separately for raw and ComBat
   models in five participant/site-balanced development folds. Select mean
   validation AUC; exact ties choose smaller C. All learned transforms are
   fitted on each fold's training scans.
3. Fit full-development reference models, then draw 200 site-stratified
   development-participant bootstraps with replacement. Copy both scans for
   every draw, including duplicates. Use identical draws for the two models;
   refit ComBat, scaler and classifier, keeping selected C fixed.
4. Save every model's weights and probabilities for the fixed test panel.
   Prediction stability is a companion outcome, separate from the primary
   feature-stability question. A separate 1,000 paired test-participant
   bootstrap describes test uncertainty for frozen reference models.

ComBat uses cohort as batch with `preserve=None`, so the unknown target state
is never used in adjustment. This example fits/applies ComBat on edges and
uses no earlier pooled/state-preserving harmonization output. Evaluation
uses new participants from represented sites; standard `combat_apply` rejects
unseen sites. ComBat treats paired scans as rows; participant splitting and
resampling account for pairing, but its empirical Bayes fit does not model
within-person covariance explicitly.

## Post-fit feature importance and aggregation

For bootstrap b and original edge e, undo the fitted scaler to obtain the
coefficient `w[b,e]`. For ComBat, also divide by that fitted site's
`delta_star[b,site,e]` to return to **original Fisher-z units**. ComBat offsets
affect the intercept, not the feature coefficients. Thus the ComBat
interpretation is site-specific; raw coefficients are identical across sites.

Multiply these original-unit coefficients by one **fixed raw-development edge
SD**, taken from the full-development raw scaler. This common reference is
estimated without test data or labels and is held fixed for all bootstraps
and both model types. The resulting `effect[b,e]` is the conditional logit
coefficient for one reference SD of an edge.

For each fixed block G, compute from those learned edge effects:

- **Signed pattern:** `mean(effect[b,e] for e in G)`.
- **Importance density:** `mean(abs(effect[b,e]) for e in G)`.
- **Total absolute mass:** `sum(abs(effect[b,e]) for e in G)`.

Importance density is the primary group-ranking score, preventing larger
blocks from ranking highly simply because they contain more edges. Total
mass is saved separately. Taking absolute values **before** averaging retains
opposite-signed large coefficients that would cancel in a signed average.

All three levels use the same fitted edge models, reference scaling and
edge coverage within a run:

| Interpretation level | Primary lobe coverage | Gyrus sensitivity coverage |
|---|---:|---:|
| Eligible ROIs in fitted edge models | 178 | 165 |
| Fitted individual edges | 15,753 | 13,530 |
| Functional Yeo7 pair summaries | 27 | 27 |
| Anatomical pair summaries | 78 | 733 |

`roi_groups.csv` contains fixed Schaefer-to-Brainnetome parcel-overlap
annotations, assigned at the finest level with at least 50% of labelled
support. Lobe coverage excludes four mixed parcels; gyrus coverage also
excludes parcels assigned only at lobe level. Anatomical groups retain
hemisphere: 12 lobe/hemisphere or 38 gyrus/hemisphere groups. These annotate
Schaefer signals. The source space is generic MNI152, with approximate FSL
MNI152 correspondence preserved in the crosswalk flags/provenance.

Functional groups use the source Schaefer Yeo labels. Every eligible upper
triangle edge belongs to exactly one group pair; self-edges and empty blocks
are omitted. The single eligible Limbic ROI has no within-Limbic edge.
The optional gyrus run changes coverage, so compare resolutions within each
run and use its matching edge baseline.

## Stability and overlap

Report pairwise signed-pattern Pearson r, importance Pearson r and
importance-rank Spearman r, plus per-feature coefficient/sign variability,
importance variability, rank and selection frequency across models.

L2 models are dense; overlap of all nonzero weights is uninformative. At each
resolution select the top **5%, 10%, 20%** of importance, using ceil for the
feature count. Compare sets with Dice, Jaccard and the equal-size
[Kuncheva chance-adjusted index](https://research.bangor.ac.uk/en/publications/a-stability-index-for-feature-selection/).
Also report sign-concordant Dice for selected features. These are audit sets;
all model inputs remain individual edges. Deterministic boundary ties are
flagged. Pairwise quantiles describe a distribution of dependent model pairs,
not independent-pair confidence intervals.

A second view projects the **same top-edge set** into functional/anatomical
groups. Save projected group-presence Jaccard and selected-edge-density
correlations. Large coarse groups can all be touched even by random edge
sets, producing apparently perfect overlap. A uniform-k-edge null accounts
for block sizes: `P(group hit)=1-C(P-m,k)/C(P,k)`, for P edges and group size m.
Save expected group coverage and the ratio of expected intersection/union
counts. This ratio is a saturation diagnostic, not exact expected Jaccard or
a p-value. High projected presence overlap alone is insufficient evidence.

## Run or reuse existing edge fits

From the repository root with `conninfpy` and development dependencies:

```bash
# Corrected two-model smoke run, including post-fit feature analysis.
python examples/open_close/feature_stability/run_feature_stability.py \
  --bootstraps 10 --test-bootstraps 100 \
  --output examples/open_close/feature_stability/results/edge_only_smoke

# Full edge-only primary run; post-fit analysis is automatic.
python examples/open_close/feature_stability/run_feature_stability.py \
  --output examples/open_close/feature_stability/results/edge_only_run1

# EC run0, same participant split and seeds.
python examples/open_close/feature_stability/run_feature_stability.py \
  --china-close-run 0 \
  --split-from examples/open_close/feature_stability/results/edge_only_run1/participant_splits.csv \
  --output examples/open_close/feature_stability/results/edge_only_run0

# Reuse saved edge models from a previous complete run: NO fitting.
# Legacy aggregate-input models in that folder are ignored.
python examples/open_close/feature_stability/feature_stability.py \
  --input examples/open_close/feature_stability/results/lobe_run1 \
  --output examples/open_close/feature_stability/results/edge_features_run1

python -m pytest -q tests/test_openclose_prediction.py
```

Use the same `--seed` to reuse CV assignments/bootstrap draws.
`--split-from` validates participants and sites. Nonempty output directories
are rejected. Outputs are regeneratable and ignored by Git. `--threads 1`
is the default BLAS limit. `--anatomical-level gyrus` supports the matching
165-ROI finer-anatomy sensitivity; both classifiers still fit individual edges.

The crosswalk ships with the example and needs no inference run. Regenerate
from local atlas images with `build_roi_groups.py --parcel-image /path/to/...`.

## Outputs

Prediction runs save configuration/input/source hashes, selected C/fold
scores, fixed person/scan splits, feature memberships, shared participant
draws, every held-out probability, prediction/metric stability, paired test
uncertainty and fitted model/ComBat states. Their `feature_stability/` folder
contains the same outputs as the standalone feature-analysis command:

| File | Meaning |
|---|---|
| `analysis_config.json` | Source model-state hashes, grouping/scaling definitions, fractions and number of new models fitted (zero) |
| `fixed_development_edge_scale.npy` | Common raw training reference SD, used for both model types |
| `feature_pattern_stability.csv` | Signed/absolute/rank pattern stability of the same edge fits, at each resolution and site |
| `feature_overlap_summary.csv` | Top-feature Dice/Jaccard/chance-adjusted/sign-concordant overlap at each resolution |
| `projected_edge_overlap_summary.csv` | Projected edge-set presence/density stability and block-capacity null diagnostics |
| `*__features.csv` | Feature sign/importance/rank/selection stability, block sizes and projected edge coverage |
| `*__pairwise.npz` | Underlying paired-bootstrap correlations and overlap values |
| `*__bootstrap_patterns.npz` | Every bootstrap's grouped signed/absolute patterns, all derived from the same edge weights |

The previous six-model aggregate-input experiment answers a separate
prediction-performance question. Its aggregate-model metrics do not answer
the feature-reproducibility question. Only its saved **edge** fits are reused.

Bootstrap stability describes sensitivity of this model's features to the
training sample, conditional on one split and selected C. Correlated edges
can exchange coefficients. Stable classifier weights remain conditional model
features; interpreting them as neural sources requires additional analysis
([Haufe et al., 2014](https://pubmed.ncbi.nlm.nih.gov/24239590/)). This example
measures internal feature reproducibility; independent replication is a
separate experiment.

## Code organization

`data.py` defines individual-edge inputs and reporting memberships; `resampling.py` defines participant splits/folds/draws; `fitting.py` owns training-only transforms and edge fits; `experiment.py` coordinates the existing workflow. `run_feature_stability.py` is its entry point. `feature_stability.py` analyzes saved edge fits; `metrics.py` contains numerical summaries. `build_roi_groups.py` creates the atlas crosswalk using canonical resource/interpretation APIs.

Saved crosswalk hashes and older run provenance are preserved. Newly generated provenance points at the current implementation sources. No model-fitting code is invoked by the atlas contrast-stability runner.
