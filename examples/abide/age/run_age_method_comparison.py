"""ABIDE method comparison on the age → FC positive control.

Runs each method on the same Strategy D GLM design (interest=age,
confounds=sex, mean_fd) and writes p-maps plus atlas-aware edge tables to
``results/age/methods/``. The downstream agreement audit is
produced by :mod:`audit_age_methods`.

We intentionally run on age (which has strong signal) rather than ADOS
(honest null) — the method comparison is only informative when all
methods have something to find.
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
from examples.abide.age._inference_cli import add_inference_cli, add_run_cli, describe_contract, resolve_n_perm, write_metadata

HERE = Path(__file__).resolve().parents[1]
DATA_FILE = HERE / "results" / "abide_prepared.npz"
OUT_DIR = HERE / "results" / "age" / "methods"


def _build_atlas(data: dict) -> AtlasInfo:
    return build_atlas(data)


def run_method(data, atlas, method_key: str, n_perm: int, seed: int,
               tfnbs_n=None, two_tailed: bool = True):
    print("\n" + "=" * 72)
    print(f"METHOD: {method_key}")
    print("=" * 72)
    print(f"  {describe_contract(tfnbs_n, two_tailed)}, n_perm={n_perm}, seed={seed}")

    Y = data["connectivity_z"]
    interest = data["age"].astype(float)
    confounds = np.column_stack([data["sex"].astype(float), data["mean_fd"].astype(float)])
    sites = data["site"]

    kwargs = dict(
        Y=Y, interest=interest, confounds=confounds, sites=sites,
        harmonize="combat_site_dummies_glm",  # Strategy D
        fisher_z=False, n_permutations=n_perm, rng=seed, use_mp=True,
        two_tailed=two_tailed,   # 2.0.0: one two-sided family (per-tail nulls with --one-tailed)
        missing="available_case",  # NaN edges fitted per available case; never zero-imputed
    )

    t0 = time.time()
    if method_key == "tstat":
        out = analyze(**kwargs, method="tstat")
    elif method_key == "tfnbs":
        out = analyze(**kwargs, method="tfnbs", e=0.4, h=3.0, n=tfnbs_n, start_thres=1.65)
    elif method_key.startswith("nbs_"):
        thr = float(method_key.split("_")[1])
        out = analyze(**kwargs, method="nbs", threshold=thr)
    elif method_key == "cnbs":
        out = analyze(**kwargs, method="cnbs", net_labels=data["net_labels"])
    elif method_key == "ni_tfnbs":
        out = analyze(**kwargs, method="ni_tfnbs", net_labels=data["net_labels"],
                      e=0.4, h=3.0, n=tfnbs_n, start_thres=1.65)
    elif method_key == "fbc_tfnbs":
        out = analyze(**kwargs, method="fbc_tfnbs", net_labels=data["net_labels"],
                     e=0.4, h=3.0, n=tfnbs_n, start_thres=1.65, min_cluster_size=3)
    else:
        raise ValueError(f"Unknown method_key: {method_key}")

    print(f"  time: {time.time() - t0:.1f}s")
    nsig = out.inference.n_significant()
    print(f"  sig@α=0.05: positive={nsig['positive']}, negative={nsig['negative']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / f"age_{method_key}.npz",
        positive=out.inference["positive"],
        negative=out.inference["negative"],
        beta=np.asarray(getattr(out.inference, "beta")),
        subject_ids=np.asarray(data.get("subject_ids", []), dtype=str),
        roi_names=np.asarray([str(r) for r in data["roi_names"]]),
    )
    archive = archive_result(out, OUT_DIR, f"age_{method_key}")
    print(f"  saved: {OUT_DIR / archive['bundle']} / {archive['result']}  "
          f"(bundle_ok={archive['bundle_ok']}, result_ok={archive['result_ok']}, coupling_ok={archive['coupling_ok']})")
    write_metadata(
        OUT_DIR / f"age_{method_key}_meta.json", script=Path(__file__).name,
        output=f"age_{method_key}.npz", method=out.inference.method if hasattr(out.inference, "method") else method_key,
        interest="age", confounds=["sex", "mean_fd"], harmonize="combat_site_dummies_glm",
        tfnbs_n=tfnbs_n if method_key in ("tfnbs", "ni_tfnbs", "fbc_tfnbs") else None,
        two_tailed=two_tailed, n_permutations=n_perm, seed=seed,
        n_subjects=int(Y.shape[0]), n_rois=int(Y.shape[1]),
        atlas=atlas.source, roi_names=list(map(str, data["roi_names"])), net_labels=list(map(int, data["net_labels"])),
        archive=archive,
    )
    out.to_csv(
        OUT_DIR / f"age_{method_key}_edges.csv",
        atlas=atlas, alpha=0.05, sort="network_pair"
    )
    print(f"  saved p-map: {OUT_DIR / f'age_{method_key}.npz'}")
    print(f"  saved edge table: {OUT_DIR / f'age_{method_key}_edges.csv'}")


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--methods", nargs="+",
        # The seven methods of the manuscript's method-spread table. (The GLM
        # pipeline has no parametric BH-FDR, so it is not offered here.)
        default=["tstat", "nbs_2.0", "nbs_3.0", "tfnbs", "cnbs", "ni_tfnbs", "fbc_tfnbs"],
    )
    add_prepared_cli(ap, DATA_FILE)
    add_run_cli(ap, OUT_DIR)
    add_inference_cli(ap)
    args = ap.parse_args()
    OUT_DIR = args.output.resolve()
    n_perm = resolve_n_perm(args)

    data = load_selected(args)
    atlas = _build_atlas(data)
    print(describe_selection(args, data))
    for m in args.methods:
        run_method(data, atlas, m, n_perm=n_perm, seed=args.seed,
                   tfnbs_n=args.tfnbs_n, two_tailed=args.two_tailed)


if __name__ == "__main__":
    main()
