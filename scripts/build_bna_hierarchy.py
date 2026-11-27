"""Thin wrapper around ``conninfpy.atlas.download.build_bna_hierarchy_csv``.

Build ``bna246_hierarchy.csv`` from the official Brainnetome files
(atlas.brainnetome.org/download.html):
  BNA_subregions.xlsx                       Fan et al. (2016) Table 1
  subregion_func_network_Yeo_updated.csv    BNA subregion -> Yeo 7 / 17 network
and the cached ``bna246.csv`` (names, voxel-mass centroids).

The derivation itself lives in the package (``conninfpy.atlas.download``) and
is also run automatically by ``conninfpy download-atlases``. The output is
redistributed under the Brainnetome terms (see
conninfpy/atlas_data/NOTICE_BRAINNETOME.txt).

Usage:
    python scripts/build_bna_hierarchy.py [--source-dir ~/conninfpy_data/brainnetome/sources]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from conninfpy.atlas.download import build_hierarchy_main

if __name__ == "__main__":
    raise SystemExit(build_hierarchy_main())