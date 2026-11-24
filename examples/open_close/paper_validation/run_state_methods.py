"""Compute paired state-contrast maps for method-agreement plots; no classifiers.

Runs every selector of ``examples.open_close._shared.inference.SELECTORS`` on
the IHB and China open-vs-close paired contrast under the ConnInfPy 2.0.0
contract: exact TFNBS integration (``--tfnbs-n none``) and one two-sided family
(``--two-tailed``). ``--tfnbs-n 10 --one-tailed`` reproduces the legacy settings
for the 2x2 control. The resolved parameters (``tfnbs_n``, ``two_tailed``,
``n_permutations``, ``seed``, ``git_sha``) are written to ``pmaps_meta.json``
next to the p-maps.
"""
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
from pathlib import Path

import numpy as np

from examples.open_close._shared.data import DATA_DIR, OpenCloseDataset
from examples.open_close._shared.inference import (
    SELECTORS, TFNBS_200, add_inference_cli, compute_selector_pmaps, describe_contract, write_metadata,
)
from examples.open_close._shared.paths import METHOD_MAPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=METHOD_MAPS)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    add_inference_cli(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_metadata(
        args.output / "pmaps_meta.json", script=Path(__file__).name,
        cohorts=["ihb", "china"], selectors=list(SELECTORS), test_type="paired",
        n_permutations=args.n_perm, seed=args.seed,
        tfnbs_e=TFNBS_200["e"], tfnbs_h=TFNBS_200["h"],
        tfnbs_start_thres=TFNBS_200["start_thres"],
        tfnbs_n=args.tfnbs_n, two_tailed=args.two_tailed,
    )
    print(f"{describe_contract(args.tfnbs_n, args.two_tailed)}, n_perm={args.n_perm}, seed={args.seed}")
    for cohort in ("ihb", "china"):
        ds = OpenCloseDataset.load(cohort, data_dir=args.data_dir)
        for name, maps in compute_selector_pmaps(
            ds, args.n_perm, args.seed, tfnbs_n=args.tfnbs_n, two_tailed=args.two_tailed
        ).items():
            stem = name.replace("@", "_").replace(".", "")
            np.savez(args.output / f"pmap_{cohort}_{stem}.npz", **maps)


if __name__ == "__main__":
    main()
