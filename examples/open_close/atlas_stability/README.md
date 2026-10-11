# Open-Close contrast stability across parcellations

Does the eyes-closed − eyes-open (EC − EO) connectivity contrast depend on the
parcellation? This folder compares two things:

* **native Brainnetome parcellation**: signals measured in BNA-246 regions;
* **Brainnetome annotation**: the same Schaefer-200 result, labelled with BNA anatomy.

Plan: vault note `openclose_atlas_contrast_stability_plan_2026-10-10`.

## Branches

All matched branches share the same participants and runs, `strategy-4_GSR`,
and a fixed common cortical support.

| Branch | Signals |
|---|---|
| `schaefer` | Schaefer-200 parcels (all 182 retained parcels pass the support rule) |
| `bna` | native BNA-246 cortical regions on the common support (178) |
| `bna_gyrus` | volume-weighted mean of the `bna` signals within gyrus × hemisphere (40). Resolution control: same data, coarser parcels |
| `schaefer_full`, `bna_full` | full sets (182; 242 incl. subcortex), descriptive only |

**Cohorts:**
* IHB (84 participants);
* China EO vs counterbalanced EC1 (47 participants; primary);
* China EC0 (the paper's run; sensitivity).

**Common support rule** (`adapter.common_support`, fixed before any contrast is examined):
* drop the BNA dropout regions 101, 102, 109, 118;
* drop BNA regions mostly inside the 18 Schaefer parcels removed for IHB coverage;
* drop Schaefer parcels mostly inside BNA dropout regions.

**Provenance check** (`adapter.provenance`, blocking): per scan, the median correlation
of each BNA region with its most-overlapping Schaefer parcel. It confirms that both
extraction products come from the same scans.

## Analyses

* **Inference:** paired TFNBS with the paper settings (per-tail, E = 0.4, H = 3, n = 10,
  5,000 permutations, seed 42). The Schaefer full set reproduces the stored paper maps.
* **Annotation only:** the Schaefer result's arrays are unchanged by annotation.
  Also reported: attribution levels of the detected edges.
* **Common-unit summaries:** participant-level weighted mean signed Δz per unit pair
  at four levels:
  * BNA lobe × hemisphere (primary);
  * gyrus × hemisphere;
  * the Brainnetome Yeo-7 crosswalk (`yeo_ref`);
  * each atlas's own labels (`yeo_own`).

  Each also gets a baseline split (strong vs weak EO coupling) and a between-edge
  heterogeneity τ.
* **Bootstrap:** 1,000 participant draws per cohort (seed 42), the same draws for
  every branch.
  * Primary endpoint: IHB vs China agreement of unit effects per atlas (Lin's CCC,
    Pearson, Spearman, sign agreement), with paired atlas differences.
  * Secondary: within-atlas edge stability with parcel volume, cross-atlas agreement,
    detection density, China EC0.

## Run

```
python examples/open_close/atlas_stability/run_atlas_stability.py            # ~5 min
python examples/open_close/atlas_stability/run_atlas_stability.py --quick     # 500 permutations, 200 draws
python examples/open_close/atlas_stability/run_atlas_stability.py --reuse     # reuse saved TFNBS maps
```

Outputs go to `results/` (git-ignored): `report.md`, CSV tables, `tfnbs/*.npz`,
`fig_cross_cohort_units.png`, `fig_primary_bootstrap.png`.

Everything is descriptive: no new hypothesis test, and the TFNBS correction is unchanged.
