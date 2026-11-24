"""ABIDE Diagnosis Strategy E — No ComBat, but Site Dummies in GLM.

Baseline for Plot 7:
- No ComBat harmonization.
- Site dummies ARE included in GLM.
- Stratified permutations.
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

HERE = Path(__file__).resolve().parents[1]
DATA_FILE = HERE / "results" / "abide_prepared.npz"
OUT_DIR = HERE / "results" / "diagnosis" / "strategy_e"


def _build_atlas(data: dict) -> AtlasInfo:
    return build_atlas(data)


def run_dx_strat_e(data: dict, atlas: AtlasInfo, n_perm: int, seed: int,
                   tfnbs_n=None, two_tailed: bool = True) -> None:
    print("\n" + "=" * 72)
    print("Diagnosis Contrast: ASD vs. HC (Strategy E - No ComBat)")
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
    roi_ids = list(map(str, data["roi_names"]))

    t0 = time.time()
    out = analyze(
        Y,
        interest=group,
        confounds=confounds,
        sites=sites,
        harmonize=None,  # Strategy E (No ComBat)
        fisher_z=False,
        method="tfnbs",
        atlas=atlas,
        roi_ids=roi_ids,
        **tfnbs_kwargs("tfnbs", tfnbs_n, two_tailed),  # 2.0.0: exact integration, one two-sided family
        n_permutations=n_perm,
        rng=seed,
        use_mp=True,
    )
    print(f"  time: {time.time() - t0:.1f}s")

    nsig = out.inference.n_significant()
    print(f"  sig@α=0.05: positive={nsig['positive']}, negative={nsig['negative']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / "dx_strategy_e.npz",
        positive=out.inference["positive"],
        negative=out.inference["negative"],
        beta=np.asarray(getattr(out.inference, "beta")),
        subject_ids=np.asarray(data.get("subject_ids", []), dtype=str),
        roi_names=np.asarray(roi_ids),
    )
    archive = archive_result(out, OUT_DIR, "dx_strategy_e")
    print(f"  saved: {OUT_DIR / archive['bundle']} / {archive['result']}  "
          f"(bundle_ok={archive['bundle_ok']}, result_ok={archive['result_ok']}, coupling_ok={archive['coupling_ok']})")
    out.to_csv(
        OUT_DIR / "dx_strategy_e_edges.csv",
        atlas=atlas,
        alpha=0.05,
        sort="network_pair",
    )
    print(f"  saved edge table: {OUT_DIR / 'dx_strategy_e_edges.csv'}")
    write_metadata(
        OUT_DIR / "dx_strategy_e_meta.json", script=Path(__file__).name,
        output="dx_strategy_e.npz", method="tfnbs", contrast="ASD vs HC",
        harmonize=None, site_dummies_in_glm=True, missing="available_case",
        tfnbs_n=tfnbs_n, two_tailed=two_tailed,
        n_permutations=n_perm, seed=seed, n_subjects=int(Y.shape[0]), n_rois=int(Y.shape[1]),
        atlas=atlas.source, roi_names=roi_ids, net_labels=list(map(int, data["net_labels"])),
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
    run_dx_strat_e(data, atlas, n_perm, args.seed, tfnbs_n=args.tfnbs_n, two_tailed=args.two_tailed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
