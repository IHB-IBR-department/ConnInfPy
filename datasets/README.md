# Study data

This directory holds **public demo** study inputs and manifests tracked in the repository:

| Path | Contents |
|---|---|
| `demo/ABIDE_mini/` + `abide_mini.yaml` | 204-subject, 5-site Schaefer-100 ABIDE subset (`abide_mini.npz`, pickle-free) for CLI, Streamlit, and README demos |
| `demo/NBS_schizophrenia/` + `nbs_schizophrenia.yaml` | MATLAB NBS v1.2 example (27 subjects, 74 AAL regions, GPL v3+) |
| `demo/open_close/` + `open_close.yaml` | Eyes-Open vs. Eyes-Closed IHB Schaefer-200 arrays (CC BY 4.0), labels, and subject list |

Full external datasets — ABIDE I PCP CC200 (`../local_datasets/ABIDE_pcp/` or `~/nilearn_data/ABIDE_pcp/`), Beijing EOEC (China) derivatives, local Brainnetome Open-Close series, and personal fMRIPrep or clinical cohorts — are stored **outside** the repository (e.g., under `$CONNINFPY_DATA` or an external study directory) and passed via CLI flags (`--data-dir`) or a manifest YAML.

Atlas resources have separate owners:

| Resource | Location |
|---|---|
| Bundled MIT parcellation tables (`schaefer*_yeo7.csv`) and provenance manifests | Installed `conninfpy/atlas_data/`, accessed through `conninfpy.atlas.resources` |
| On-demand Brainnetome BNA-246 & SUIT v1.2 atlases | `~/.cache/conninfpy/atlases/` (or `$CONNINFPY_ATLAS_DIR`), populated via `scripts/download_atlases.py` |
| Optional Brainnetome probabilistic/tractography originals | `~/conninfpy_data/brainnetome/`, or a configured data directory |
| Original Brainnetome hierarchy-building tables | The external directory's `sources/` subfolder |
| ENIGMA Schaefer structural-connectivity simulation fixtures | [examples/simulations/sc_prior/data/](../examples/simulations/sc_prior/data/) |
| Derived Brainnetome arrays | `~/.cache/conninfpy/bna246/`, or the configured cache |

## NBS schizophrenia example (bundled, GPL v3)

`demo/NBS_schizophrenia/` holds the example data of the MATLAB NBS toolbox
v1.2 (Zalesky, Fornito & Bullmore 2010): 12 people with schizophrenia and 15
healthy controls, 74 AAL regions, raw wavelet correlations. It is read by
`nbs_schizophrenia.yaml` (`NumpyLoader` with an atlas CSV) and offered in the
app as the built-in template `NBS-Schizophrenia`. Unlike the rest of the
repository these files are under the GNU GPL v3 or later; provenance, the two
corrections to the NBS node table and the MATLAB reference result are in
[demo/NBS_schizophrenia/README.md](demo/NBS_schizophrenia/README.md).

| File | Shape, dtype | Bytes | SHA-256 |
|---|---|---|---|
| `nbs_schizophrenia.npy` | (27, 74, 74), float64 | 1 182 944 | `b33d3cee753c81ba55af6a7afbe7f468bccc7385e8fbc6a3f63d9a824c473f6e` |
| `pheno.csv` | 27 rows | 368 | `0bda04c86213d670f623845b1b4c638f5f727c6e8f70abef97f874bfc4d9848c` |
| `aal74_atlas.csv` | 74 rows | 3 535 | `4cfcd4966763a976f68d21315643a9f7402473091a1647147491b2c8d8aa49aa` |

Rebuild with `python scripts/build_nbs_schizophrenia_demo.py --nbs-dir <NBS1.2>`.

## Beijing EOEC (China) Schaefer-200 derivatives (external / link only)

Due to their Creative Commons Attribution-NonCommercial (CC BY-NC) licence, the
Beijing EOEC derivatives (`china_open_Schaefer200_strategy-4_GSR.npy`,
`china_close_Schaefer200_strategy-4_GSR.npy`, `subject_order_china.txt`) are
**not** tracked in this repository.

1. Download the preprocessed derivatives archive from Medvedeva et al. (2026):
   `https://disk.yandex.ru/d/fNz33QPwJQj7HQ`
   (Upstream cohort: `https://fcon_1000.projects.nitrc.org/indi/retro/BeijingEOEC.html`).
2. Prepare and verify shapes/checksums with:
   `python scripts/prepare_china_open_close.py --source-dir <downloaded_dir>`

See [demo/open_close/NOTICE_BEIJING_EOEC.txt](demo/open_close/NOTICE_BEIJING_EOEC.txt)
and [NOTICE](../NOTICE) for mandatory attribution and grant acknowledgements.

## Open-Close Brainnetome series and Beijing EOEC phenotype (external / local only)

These nine files are **not** stored in the repository; keep them in an external study directory (e.g., `$CONNINFPY_DATA/open_close/` or `../local_datasets/open_close/`) and pass `--data-dir` (or an external path) rather than placing them inside `datasets/`. Provenance recorded in the repository: BenchmarkingEOEC pre-processing pipeline ``strategy-4`` (Medvedeva et al. 2026, *Brain Informatics* 13:18, https://doi.org/10.1186/s40708-026-00305-1; preprocessed data archive: https://disk.yandex.ru/d/fNz33QPwJQj7HQ), see the docstring of [examples/open_close/_shared/data.py](../examples/open_close/_shared/data.py). For third-party licence and attribution terms, see [demo/open_close/NOTICE_BEIJING_EOEC.txt](demo/open_close/NOTICE_BEIJING_EOEC.txt) and the root [NOTICE](../NOTICE).

Each ``{experiment}_{state}_Brainnetome_strategy-4_{GSR,noGSR}.npy`` file holds
the same participants as the Schaefer-200 series (same subject order; IHB: 84
subjects × 120 TRs, China: 48 subjects × 240 TRs) parcellated with BNA-246,
with and without global-signal regression; the China close products carry two
independent closed-eyes runs on the last axis (test-retest). Labels come from
`bna246_hierarchy.csv` (in the on-demand atlas cache); `drop_missing_rois` removes
`BNA_DROPOUT_IDS` (242 regions retained).

| File | Shape, dtype | Bytes | SHA-256 |
|---|---|---|---|
| `china_close_Brainnetome_strategy-4_GSR.npy` | (48, 240, 246, 2), float32 | 22 671 488 | `28548a2e78ddc4942628ff09f5c5e24f25bab0e63925554ba0c7c3ba48077c59` |
| `china_close_Brainnetome_strategy-4_noGSR.npy` | (48, 240, 246, 2), float32 | 22 671 488 | `1eca5d4957234e35e4ab9f59a9404871cb45bb9509ae6d42cdab0512cfa65308` |
| `china_open_Brainnetome_strategy-4_GSR.npy` | (48, 240, 246), float32 | 11 335 808 | `94cc3e32a442ffebed7fd936f7bae117a1bfab54a38f76e56ddff3d8b649b457` |
| `china_open_Brainnetome_strategy-4_noGSR.npy` | (48, 240, 246), float32 | 11 335 808 | `3e63cbc310fd00a62d2f7026158bd0a9556eb918d8e68325f51bdcfe67a6b8fc` |
| `ihb_close_Brainnetome_strategy-4_GSR.npy` | (84, 120, 246), float32 | 9 918 848 | `a85a2778c84f8d23101d2714a46050a691ddd3936372317069ff1cd2092842f6` |
| `ihb_close_Brainnetome_strategy-4_noGSR.npy` | (84, 120, 246), float32 | 9 918 848 | `31a277a81faaae5ef73115e930b7179df973bea8c142abe355d090a38f6b1144` |
| `ihb_open_Brainnetome_strategy-4_GSR.npy` | (84, 120, 246), float32 | 9 918 848 | `70b59c9bad1bec70b82d3582b1f80c98409730109c785c80254f122aed1e51ab` |
| `ihb_open_Brainnetome_strategy-4_noGSR.npy` | (84, 120, 246), float32 | 9 918 848 | `22eef7e484bf296c728f41e278cbb573bd808e3b4c0a4d6fedd3453e3816f3bd` |

Read by `OpenCloseDataset.load(..., atlas="brainnetome", gsr=...)` in
[examples/open_close/_shared/data.py](../examples/open_close/_shared/data.py);
`tests/test_openclose_loader.py::TestBrainnetomeSeries` skips without them.

`BeijingEOEC.csv` (1 980 bytes, SHA-256
`70a443e6310a0f2de534c3cf03c884e55e65370cae606379d17b82bb3b29d317`, 48 data
rows) is the Beijing EOEC phenotype: columns `SubjectID`, `Age`, `Sex`,
`Session_1`–`Session_3` (per-session eyes state). Not read by the package code.

Run order in the Beijing EOEC protocol matters for the Open-Close contrasts:
scan 1 is always eyes closed and precedes eyes open, while scans 2 and 3 are
eyes open / eyes closed in random order. Close run 0 (EC₀, the paper contrast)
is therefore always acquired first and is confounded with scan order, whereas
close run 1 (EC₁) is counterbalanced with eyes open. The EC₁ − EC₀ repeat-run
contrast is not an empirical null (556 positive / 416 negative edges in the
2.0 pipeline). The sensitivity arms (`EC₁ − EO` vs `EC₀ − EO` on the 47
participants with both closed runs) and the evidence are summarised in
[paper/README.md](../paper/README.md#open-close-beijing-repeat-run-and-scan-order).

Set `CONNINFPY_ATLAS_DIR` for the on-demand Brainnetome/SUIT atlas cache
(default `~/.cache/conninfpy/atlases/`). Set `CONNINFPY_DATA` to a data root;
optional BNA probabilistic/tractography maps are resolved in its `brainnetome/`
subfolder. `CONNINFPY_BNA_DIR` remains a direct override; explicit `data_dir=` takes
precedence. The retired `datasets/atlases/` location is not searched implicitly.
See the [resource documentation](../docs/source/architecture.rst).
