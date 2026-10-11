"""ABIDE method comparison on the age → FC positive control.

Runs each method on the same Strategy D GLM design (interest=age,
confounds=sex, mean_fd) and writes p-maps plus atlas-aware edge tables to
``results/age_development/methods/``. The downstream agreement audit is
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
from examples.abide._shared.data import load_prepared, build_atlas

HERE = Path(__file__).resolve().parents[1]
DATA_FILE = HERE / "results" / "abide_prepared.npz"
OUT_DIR = HERE / "results" / "age_development" / "methods"


def _load(path: Path = DATA_FILE) -> dict:
    return load_prepared(path)


def _build_atlas(data: dict) -> AtlasInfo:
    return build_atlas(data)


def run_method(data, atlas, method_key: str, n_perm: int, seed: int):
    print("\n" + "=" * 72)
    print(f"METHOD: {method_key}")
    print("=" * 72)

    Y = data["connectivity_z"]
    interest = data["age"].astype(float)
    confounds = np.column_stack([data["sex"].astype(float), data["mean_fd"].astype(float)])
    sites = data["site"]

    kwargs = dict(
        Y=Y, interest=interest, confounds=confounds, sites=sites,
        harmonize="combat_site_dummies_glm",  # Strategy D
        fisher_z=False, n_permutations=n_perm, rng=seed, use_mp=True,
    )

    t0 = time.time()
    if method_key == "tstat":
        out = analyze(**kwargs, method="tstat")
    elif method_key == "tfnbs":
        out = analyze(**kwargs, method="tfnbs", e=0.4, h=3.0, n=10)
    elif method_key.startswith("nbs_"):
        thr = float(method_key.split("_")[1])
        out = analyze(**kwargs, method="nbs", threshold=thr)
    elif method_key == "cnbs":
        out = analyze(**kwargs, method="cnbs", net_labels=data["net_labels"])
    elif method_key == "ni_tfnbs":
        out = analyze(**kwargs, method="ni_tfnbs", net_labels=data["net_labels"],
                      e=0.4, h=3.0, n=10)
    elif method_key == "fbc_tfnbs":
        out = analyze(**kwargs, method="fbc_tfnbs", net_labels=data["net_labels"],
                     e=0.4, h=3.0, n=10, min_cluster_size=3)
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
    ap.add_argument("--data-file", type=Path, default=DATA_FILE)
    ap.add_argument("--output", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    OUT_DIR = args.output.resolve()
    n_perm = 500 if args.quick else 5000

    data = _load(args.data_file)
    atlas = _build_atlas(data)
    for m in args.methods:
        run_method(data, atlas, m, n_perm=n_perm, seed=args.seed)


if __name__ == "__main__":
    main()
