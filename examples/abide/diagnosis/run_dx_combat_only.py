"""ABIDE Diagnosis — ComBat Only (No Site Dummies in GLM).

Isolates the effect of the ComBat transformation:
- ComBat is fit on residuals of interest (nuisance-only).
- NO site dummies are included in the GLM.
- NO stratified permutations (standard shuffle).

This test determines if the ComBat data transformation is sufficient
by itself to neutralize site effects in ABIDE.
"""
from __future__ import annotations


from pathlib import Path as _Path
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from conninfpy import AtlasInfo, analyze
from examples.abide._shared.data import load_prepared, build_atlas
from conninfpy.harmonize import combat_harmonize

HERE = Path(__file__).resolve().parents[1]
DATA_FILE = HERE / "results" / "abide_prepared.npz"
OUT_DIR = HERE / "results" / "diagnosis" / "combat_only"


def _load(path: Path = DATA_FILE) -> dict:
    return load_prepared(path)


def _build_atlas(data: dict) -> AtlasInfo:
    return build_atlas(data)


def run_dx_combat_only(data: dict, atlas: AtlasInfo, n_perm: int, seed: int) -> None:
    print("\n" + "=" * 72)
    print("Diagnosis Contrast: ASD vs. HC (ComBat ONLY - No Site Dummies)")
    print("=" * 72)

    Y = data["connectivity_z"]
    group = data["group"].astype(float)
    confounds = np.column_stack([
        data["age"].astype(float),
        data["sex"].astype(float),
        data["mean_fd"].astype(float)
    ])
    sites = data["site"]

    # 1. Manually apply ComBat (Strategy D style: preserve bio, not diagnosis)
    print("  Applying ComBat (nuisance-only)...")
    combat_out = combat_harmonize(Y, sites, preserve=confounds)
    Y_harm = combat_out.Y_adjusted

    # 2. Run analysis on harmonized data WITHOUT site info
    t0 = time.time()
    out = analyze(
        Y_harm,
        interest=group,
        confounds=confounds,
        sites=None,      # No site dummies
        harmonize=None,  # Already harmonized
        fisher_z=False,
        method="tfnbs",
        e=0.4, h=3.0, n=10,  # prespecified (Hao et al.), not the package default
        n_permutations=n_perm,
        rng=seed,
        use_mp=True,
    )
    print(f"  time: {time.time() - t0:.1f}s")
    
    nsig = out.inference.n_significant()
    print(f"  sig@α=0.05: positive={nsig['positive']} (ASD>HC), "
          f"negative={nsig['negative']} (HC>ASD)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(
        OUT_DIR / "dx_combat_only_edges.csv",
        atlas=atlas,
        alpha=0.05,
        sort="network_pair",
    )
    print(f"  saved edge table: {OUT_DIR / 'dx_combat_only_edges.csv'}")


def main() -> int:
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-file", type=Path, default=DATA_FILE)
    ap.add_argument("--output", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    OUT_DIR = args.output.resolve()
    n_perm = 500 if args.quick else 5000

    data = _load(args.data_file)
    atlas = _build_atlas(data)
    run_dx_combat_only(data, atlas, n_perm, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
