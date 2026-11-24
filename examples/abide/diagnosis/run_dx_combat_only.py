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
from examples.abide._shared.data import add_prepared_cli, archive_result, build_atlas, describe_selection, load_selected
from examples.abide.age._inference_cli import add_inference_cli, add_run_cli, describe_contract, resolve_n_perm, tfnbs_kwargs, write_metadata
from conninfpy.harmonize import combat_harmonize

HERE = Path(__file__).resolve().parents[1]
DATA_FILE = HERE / "results" / "abide_prepared.npz"
OUT_DIR = HERE / "results" / "diagnosis" / "combat_only"


def _build_atlas(data: dict) -> AtlasInfo:
    return build_atlas(data)


def run_dx_combat_only(data: dict, atlas: AtlasInfo, n_perm: int, seed: int,
                      tfnbs_n=None, two_tailed: bool = True) -> None:
    print("\n" + "=" * 72)
    print("Diagnosis Contrast: ASD vs. HC (ComBat ONLY - No Site Dummies)")
    print(f"  {describe_contract(tfnbs_n, two_tailed)}, n_perm={n_perm}, seed={seed}")
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
    print("  Applying ComBat (nuisance-only, missing='available_case')...")
    combat_out = combat_harmonize(Y, sites, preserve=confounds, missing="available_case")  # NaN edges fitted per available case, never zero-imputed
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
        **tfnbs_kwargs("tfnbs", tfnbs_n, two_tailed),  # 2.0.0: exact integration, one two-sided family
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
    np.savez(
        OUT_DIR / "dx_combat_only.npz",
        positive=out.inference["positive"],
        negative=out.inference["negative"],
        beta=np.asarray(getattr(out.inference, "beta")),
        subject_ids=np.asarray(data.get("subject_ids", []), dtype=str),
        roi_names=np.asarray([str(r) for r in data["roi_names"]]),
    )
    archive = archive_result(out, OUT_DIR, "dx_combat_only")
    print(f"  saved: {OUT_DIR / archive['bundle']} / {archive['result']}  "
          f"(bundle_ok={archive['bundle_ok']}, result_ok={archive['result_ok']}, coupling_ok={archive['coupling_ok']})")
    write_metadata(
        OUT_DIR / "dx_combat_only_meta.json", script=Path(__file__).name,
        output="dx_combat_only.npz", method="tfnbs", contrast="ASD vs HC",
        harmonize="combat (nuisance-only), no site dummies",
        tfnbs_n=tfnbs_n, two_tailed=two_tailed,
        n_permutations=n_perm, seed=seed, n_subjects=int(Y.shape[0]), n_rois=int(Y.shape[1]),
        atlas=atlas.source, roi_names=list(map(str, data["roi_names"])), net_labels=list(map(int, data["net_labels"])),
        archive=archive,
    )


def main() -> int:
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    add_prepared_cli(ap, DATA_FILE)
    add_run_cli(ap, OUT_DIR)
    add_inference_cli(ap)
    args = ap.parse_args()
    OUT_DIR = args.output.resolve()
    n_perm = resolve_n_perm(args)

    data = load_selected(args)
    atlas = _build_atlas(data)
    print(describe_selection(args, data))
    run_dx_combat_only(data, atlas, n_perm, args.seed, tfnbs_n=args.tfnbs_n, two_tailed=args.two_tailed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
