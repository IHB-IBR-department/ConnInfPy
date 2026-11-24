"""ABIDE age and motion positive controls under ComBat Strategy D.

This is the canonical ABIDE positive-control pipeline. ComBat is fit with
``preserve=confounds`` only; the tested variable is deliberately omitted from
the ComBat preserve design and site dummies are appended to the downstream GLM.
That removes the Nygaard-style label leak risk that Strategy B can introduce
when the tested variable is correlated with site.

Outputs go to ``results/age/`` as strategy-explicit p-map ``.npz``
files plus atlas-annotated edge tables.

Usage:
    python examples/abide/age/run_age_strategy_d.py --variant age
    python examples/abide/age/run_age_strategy_d.py --variant all
"""
from __future__ import annotations


from pathlib import Path as _Path
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from conninfpy import AtlasInfo, analyze
from examples.abide._shared.data import add_prepared_cli, archive_result, build_atlas, describe_selection, load_selected
from examples.abide.age._inference_cli import add_inference_cli, add_run_cli, describe_contract, resolve_n_perm, tfnbs_kwargs, write_metadata

HERE = Path(__file__).resolve().parents[1]
DATA_FILE = HERE / "results" / "abide_prepared.npz"
OUT_DIR = HERE / "results" / "age"


def _build_atlas(data: dict) -> AtlasInfo:
    return build_atlas(data)


def _summary(label: str, result, alpha: float = 0.05) -> None:
    nsig = result.n_significant(alpha=alpha)
    print(f"  {label} sig@α={alpha}: positive={nsig['positive']}, "
          f"negative={nsig['negative']}")
    diag = result.combat_diagnostics
    if diag is not None:
        strategy = diag.get("strategy", "?")
        rb = diag.get("between_site_variance_ratio_before")
        ra = diag.get("between_site_variance_ratio_after")
        if rb is not None and ra is not None:
            print(f"  ComBat (strategy {strategy}) "
                  f"between-site var: {rb:.4f} → {ra:.4f}")


def run_variant(
    data: dict,
    atlas: AtlasInfo,
    *,
    interest_name: str,
    confound_names: list[str],
    label: str,
    n_perm: int,
    seed: int,
    tfnbs_n=None,
    two_tailed: bool = True,
) -> None:
    print("\n" + "=" * 72)
    print(f"{label}: interest={interest_name}, confounds={confound_names}")
    print(f"  {describe_contract(tfnbs_n, two_tailed)}, n_perm={n_perm}, seed={seed}")
    print(f"  harmonize='combat_site_dummies_glm' (Strategy D), site dummies in GLM")
    print("=" * 72)

    Y = data["connectivity_z"]
    interest = data[interest_name].astype(float)
    confounds = np.column_stack(
        [data[c].astype(float) for c in confound_names]
    )
    sites = data["site"]
    print(f"  N={Y.shape[0]}, confounds shape {confounds.shape}, "
          f"K sites={len(np.unique(sites))}")

    t0 = time.time()
    out = analyze(
        Y,
        interest=interest,
        confounds=confounds,
        sites=sites,
        harmonize="combat_site_dummies_glm",                   # Strategy D
        fisher_z=False,
        method="tfnbs",
        **tfnbs_kwargs("tfnbs", tfnbs_n, two_tailed),   # 2.0.0: exact integration, one two-sided family
        n_permutations=n_perm,
        acceleration=None,
        rng=seed,
        use_mp=True,
    )
    print(f"  time: {time.time() - t0:.1f}s")
    _summary(label, out.inference)
    for flag in out.flags:
        print(f"  flag: {flag}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / f"{label}.npz",
        positive=out.inference["positive"],
        negative=out.inference["negative"],
        beta=np.asarray(getattr(out.inference, "beta")),
        subject_ids=np.asarray(data.get("subject_ids", []), dtype=str),
        roi_names=np.asarray([str(r) for r in data["roi_names"]]),
    )
    archive = archive_result(out, OUT_DIR, label)
    print(f"  saved: {OUT_DIR / archive['bundle']} / {archive['result']}  "
          f"(bundle_ok={archive['bundle_ok']}, result_ok={archive['result_ok']}, coupling_ok={archive['coupling_ok']})")
    write_metadata(
        OUT_DIR / f"{label}_meta.json", script=Path(__file__).name,
        output=f"{label}.npz", method="tfnbs", interest=interest_name, confounds=confound_names,
        harmonize="combat_site_dummies_glm", tfnbs_n=tfnbs_n, two_tailed=two_tailed,
        n_permutations=n_perm, seed=seed, n_subjects=int(Y.shape[0]), n_rois=int(Y.shape[1]),
        atlas=atlas.source, roi_names=list(data["roi_names"]), net_labels=list(map(int, data["net_labels"])),
        archive=archive,
    )
    out.to_csv(
        OUT_DIR / f"{label}_edges.csv",
        atlas=atlas,
        alpha=0.05,
        sort="network_pair",
    )
    print(f"  saved: {OUT_DIR / f'{label}.npz'}")


def main() -> int:
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--variant", choices=["age", "age_dx", "fd", "all"],
                    default="all")
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

    if args.variant in ("age", "all"):
        run_variant(data, atlas,
                    interest_name="age", confound_names=["sex", "mean_fd"],
                    label="age_strategy_d", n_perm=n_perm, seed=args.seed,
                    tfnbs_n=args.tfnbs_n, two_tailed=args.two_tailed)
    if args.variant in ("age_dx", "all"):
        run_variant(data, atlas,
                    interest_name="age",
                    confound_names=["sex", "mean_fd", "group"],
                    label="age_controlling_dx_strategy_d",
                    n_perm=n_perm, seed=args.seed,
                    tfnbs_n=args.tfnbs_n, two_tailed=args.two_tailed)
    if args.variant in ("fd", "all"):
        run_variant(data, atlas,
                    interest_name="mean_fd",
                    confound_names=["age", "sex", "group"],
                    label="fd_strategy_d", n_perm=n_perm, seed=args.seed,
                    tfnbs_n=args.tfnbs_n, two_tailed=args.two_tailed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
