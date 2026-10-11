"""Compute paired state-contrast maps for method-agreement plots; no classifiers."""
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import numpy as np
from examples.open_close._shared.data import DATA_DIR, OpenCloseDataset
from examples.open_close._shared.inference import compute_selector_pmaps
from examples.open_close._shared.paths import RESULTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS / "ml")
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for cohort in ("ihb", "china"):
        ds = OpenCloseDataset.load(cohort, data_dir=args.data_dir)
        for name, maps in compute_selector_pmaps(ds, args.n_perm, args.seed).items():
            stem = name.replace("@", "_").replace(".", "")
            np.savez(args.output / f"pmap_{cohort}_{stem}.npz", **maps)


if __name__ == "__main__":
    main()
