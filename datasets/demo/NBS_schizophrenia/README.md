# NBS schizophrenia example (AAL-74)

The example data distributed with the MATLAB **NBS toolbox v1.2**
(`SchizophreniaExample/`), converted for ConnInfPy by
[scripts/build_nbs_schizophrenia_demo.py](../../../scripts/build_nbs_schizophrenia_demo.py).

> Zalesky A, Fornito A, Bullmore ET. Network-based statistic: identifying
> differences in brain networks. *NeuroImage* 2010; 53(4):1197-1207.

NBS describes the data as resting-state functional connectivity, measured as
the correlation between wavelet-decomposed time series, in 12 people with
schizophrenia and 15 healthy controls (NBS `help/examples.txt`).

## Files

| File | Content |
|---|---|
| `nbs_schizophrenia.npy` | (27, 74, 74) float64. Raw correlations, diagonal 1, no Fisher z. Subjects in NBS order: `subject01`-`subject12` schizophrenia, `subject13`-`subject27` controls. |
| `pheno.csv` | `subject_id`, `group` (`SZ` / `HC`), from NBS `designMatrix.txt` (column 1 = SZ, column 2 = HC). |
| `aal74_atlas.csv` | `name`, `network` (empty), `x`, `y`, `z` (MNI, mm), `hemisphere`, `nbs_label`. From NBS `nodeLabels.txt` and `COG.txt`. |
| `LICENSE-GPL-3.0.txt` | GNU GPL v3 text. |

Two corrections to the NBS node table, both checked by the build script:

* **x is flipped to MNI.** In `COG.txt` every `*_L` node has x > 0 and every
  `*_R` node x < 0 (radiological orientation). With x negated, the centres
  agree with the AAL centroids of the labelled hemisphere in x, y and z,
  e.g. Precentral_R (42, -10, 50) against AAL (41.4, -8.2, 52.1).
* **Subcortical names.** NBS labels nodes 61-68 all `Subcortical`. Their order
  and centres match AAL 71-78, so they are named `Caudate_L`, `Caudate_R`,
  `Putamen_L`, `Putamen_R`, `Pallidum_L`, `Pallidum_R`, `Thalamus_L`,
  `Thalamus_R`. The original label stays in `nbs_label`.

## Reference result

NBS v1.2 with the inputs its GUI pre-fills (t-test, contrast `[-1,1]` =
controls > patients, threshold 3.1, 5000 permutations, `rng(1)`):

| Component size | Components | Edges | p |
|---|---|---|---|
| Extent | 1 | 40 | 0.0398 |
| Intensity | 1 | 40 | 0.0462 |

The run is reproduced by
[scripts/nbs_schizophrenia_matlab_reference.m](../../../scripts/nbs_schizophrenia_matlab_reference.m)
and stored in `tests/data/nbs_schizophrenia_matlab_reference.json`;
`tests/test_nbs_schizophrenia_reference.py` checks ConnInfPy against it.
To match NBS, use Student's t (pooled variance), on the raw correlations, with
`two_tailed=False`, and keep the positive tail (`result.one_sided("positive")`):
NBS tests the contrast direction only. Two equivalent routes:
`compute_p_val(sz, hc, test_type="two-sample", var_equal=True, ...)` (the
default two-sample statistic is Welch's t) or `compute_p_val_glm` with the HC
indicator. Both component statistics follow NBS: extent counts edges,
intensity sums t − threshold.

In the Streamlit app: Group Difference (`SZ` reference, `HC` target, no
covariates), *Group variances* = Equal (Student's t, the default),
*Hypothesis direction* = one-sided HC > SZ, NBS with tau 3.1, raw
correlations (the app does not apply Fisher z on its own). This gives the
40-edge component with p ≈ 0.04; two-sided, the same component has p ≈ 0.07.
On Fisher z the component has 39 edges, but not 39 of the 40: 34 are shared,
6 fall below 3.1 and 5 others rise above it. The variance of r depends on its
level, (1 − ρ²)²/T, so the patients' lower correlations are more variable on
the r scale (variance ratio HC/SZ on the 40 edges: median 0.58 on r, 0.85 on
z); Fisher z removes most of that, which shifts edge t-values by up to ±0.7
and moves edges near the threshold. The same heteroscedasticity is why
Welch's t differs from the pooled t on r (28 against 40 edges) but not on z
(39 and 39). How each choice moves the result (5000 permutations, seed 1):

| Variance | Scale | Tail | Edges above 3.1 | Component p |
|---|---|---|---|---|
| pooled (NBS) | r | one-sided | 40 | 0.041 |
| pooled | r | two-sided | 40 | 0.068 |
| pooled | Fisher z | one-sided | 39 | 0.043 |
| pooled | Fisher z | two-sided | 39 | 0.077 |
| Welch | r | one-sided | 28 | 0.046 |
| Welch | Fisher z | one-sided | 39 | 0.041 |

## Licence

NBS, including this example, is distributed under the GNU General Public
License, version 3 or later (NBS `ReadMe.txt`). These converted files are
distributed under the same licence (`LICENSE-GPL-3.0.txt`); this differs from
the MIT licence of the ConnInfPy code.
