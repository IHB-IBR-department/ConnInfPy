"""
Per-method run of the China close-run0 vs close-run1 paired contrast.

Counterbalanced ordering across subjects ⇒ this contrast acts as a
matched-design empirical null for the open-vs-close state contrast.
Running every method in the SELECTORS bundle on this contrast lets us
build a specificity panel (state vs empirical-null per-method edge
counts at FWER α = 0.05).

The state-contrast pmaps already live in ``results/paper_validation/methods/pmap_china_*.npz``;
this script writes the matching retest pmaps as
``results/paper_validation/methods/pmap_china_retest_{method}.npz`` so the downstream plot
script can pair them up by method name.

Subject mask: ``ds.retest_subject_mask()`` (excludes ``sub-3258811``
whose run-3 is incomplete; see openclose_loader docstring).

Runtime: ~1–2 min per perm-based method × 6 methods ≈ 10 min total on
M5 with use_mp=True at n_permutations=1000.

ConnInfPy 2.0.0 contract: TFNBS-family selectors use exact integration
(``--tfnbs-n none``) and every selector runs as one two-sided family
(``--two-tailed``); ``--tfnbs-n 10 --one-tailed`` reproduces the legacy cell
for the 2x2 control. The resolved parameters are written to
``pmap_china_retest_<method>_meta.json`` next to every p-map.
"""
from __future__ import annotations


from pathlib import Path as _Path
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import time
from pathlib import Path

import numpy as np

from conninfpy import compute_p_val
from examples.open_close._shared.data import OpenCloseDataset, DATA_DIR
from examples.open_close._shared.paths import METHOD_MAPS
from examples.open_close._shared.inference import (
    TFNBS_200, add_inference_cli, apply_contract, describe_contract, write_metadata,
)


MAPS_DIR = METHOD_MAPS

# Matches examples/open_close/ml_transfer/run_ml_feature_selection.py.
# TFNBS entries carry the 2.0.0 cell; ``n`` and ``two_tailed`` are resolved from the CLI.
SELECTORS = {
    "tstat":    dict(method="tstat"),
    "tfnbs":    dict(method="tfnbs", **TFNBS_200),
    "nbs_20":   dict(method="nbs", threshold=2.0),
    "nbs_30":   dict(method="nbs", threshold=3.0),
    "cnbs":     dict(method="cnbs"),
    "ni_tfnbs": dict(method="ni_tfnbs", **TFNBS_200),
    "fbc_tfnbs":dict(method="fbc_tfnbs", **TFNBS_200),
    "bh_fdr":   dict(method="bh_fdr"),
}


def main() -> None:
    global MAPS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", nargs="+", default=list(SELECTORS.keys()))
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--output", type=Path, default=MAPS_DIR)
    add_inference_cli(ap)
    args = ap.parse_args()
    MAPS_DIR = args.output.resolve()

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    write_metadata(
        MAPS_DIR / "pmap_china_retest_meta.json", script=Path(__file__).name,
        contrast="china close run0 vs run1 (paired)", methods=args.methods,
        n_permutations=args.n_permutations, seed=args.seed,
        tfnbs_e=TFNBS_200["e"], tfnbs_h=TFNBS_200["h"], tfnbs_start_thres=TFNBS_200["start_thres"],
        tfnbs_n=args.tfnbs_n, two_tailed=args.two_tailed,
    )
    print(f"{describe_contract(args.tfnbs_n, args.two_tailed)}")

    ds = OpenCloseDataset.load("china", data_dir=args.data_dir)
    mask = ds.retest_subject_mask()
    excluded = [s for s, m in zip(ds.subject_ids, mask) if not m]
    print(f"China retest — excluding {(~mask).sum()} subject(s): {excluded}")
    _, c0 = ds.connectivity_z(run=0)
    _, c1 = ds.connectivity_z(run=1)
    c0 = c0[mask]
    c1 = c1[mask]
    print(f"n_paired={c0.shape[0]}, N={c0.shape[1]}, n_perm={args.n_permutations}")
    net_labels = ds.net_labels

    for name in args.methods:
        if name not in SELECTORS:
            print(f"skip unknown method: {name}")
            continue
        kwargs = apply_contract(SELECTORS[name], args.tfnbs_n, args.two_tailed)
        if kwargs["method"] in ("cnbs", "ni_tfnbs", "fbc_tfnbs"):
            kwargs["net_labels"] = net_labels

        t0 = time.time()
        p = compute_p_val(
            c0, c1,
            test_type="paired",
            n_permutations=args.n_permutations,
            use_mp=True,
            random_state=args.seed,
            **kwargs,
        )
        dt = time.time() - t0
        keys = list(p.keys())
        iu = np.triu_indices(c0.shape[1], k=1)
        n_sig_a = int((p[keys[0]][iu] < 0.05).sum())
        n_sig_b = int((p[keys[1]][iu] < 0.05).sum())
        print(f"  {name:10s}  {dt:5.1f}s  "
              f"n_sig(α=0.05) {keys[0]}={n_sig_a}, {keys[1]}={n_sig_b}")

        out_path = MAPS_DIR / f"pmap_china_retest_{name}.npz"
        np.savez(out_path, **{k: p[k] for k in p})
        write_metadata(
            MAPS_DIR / f"pmap_china_retest_{name}_meta.json", script=Path(__file__).name,
            output=out_path.name,
            kwargs={k: v for k, v in kwargs.items() if k != "net_labels"},
            tfnbs_n=args.tfnbs_n, two_tailed=args.two_tailed,
            n_permutations=args.n_permutations, seed=args.seed, test_type="paired",
            n_paired=int(c0.shape[0]), n_rois=int(c0.shape[1]),
        )
        print(f"    saved: {out_path}")


if __name__ == "__main__":
    main()
