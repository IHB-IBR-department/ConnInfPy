
from pathlib import Path as _Path
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from examples.open_close._shared.data import OpenCloseDataset

HERE = Path(__file__).resolve().parents[2]
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"

# Σ −log10 p per network pair, from conninfpy.results.summarize_by_network
# (the aggregation users get from result.network_summary(atlas)).
from examples.open_close._shared.agreement import block_mass as _block_mass

def main():
    ihb = np.load(RESULTS / "ihb_paired_tfnbs.npz")
    china = np.load(RESULTS / "china_paired_tfnbs_run0.npz")
    china_retest = np.load(RESULTS / "china_retest_tfnbs.npz")
    ds = OpenCloseDataset.load("ihb"); lab = ds.net_labels; names = ds.get_label_names(); K = len(names)
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    for r, tail in enumerate(["positive", "negative"]):
        ma = _block_mass(ihb[tail], lab)
        mb = _block_mass(china[tail], lab)
        mc = _block_mass(china_retest[tail], lab)
        vmax = max(ma.max(), mb.max(), mc.max())
        for c, (M, title) in enumerate([(ma, f"IHB {tail}"), (mb, f"Beijing China State {tail}"), (mc, f"Beijing China Retest (Null) {tail}")]):
            im = axes[r, c].imshow(M, cmap="viridis", vmin=0, vmax=vmax)
            axes[r, c].set_xticks(range(K)); axes[r, c].set_xticklabels(names, rotation=60, ha="right", fontsize=8)
            axes[r, c].set_yticks(range(K)); axes[r, c].set_yticklabels(names, fontsize=8)
            axes[r, c].set_title(title); fig.colorbar(im, ax=axes[r, c], fraction=0.046)
    plt.suptitle("Block-Mass Matrix Convergence: Cross-Cohort Replication & Specificity (TF-NBS: E=0.4, H=3.0)\nComparison of topological integration patterns (Σ −log10 p)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(PLOTS / "plot2_block_mass_convergence.png", dpi=150); plt.close()

if __name__ == "__main__":
    main()
