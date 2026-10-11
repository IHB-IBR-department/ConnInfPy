"""
ABIDE (E, H) sensitivity sweep — Multiverse Hierarchy.

Sweeps a 6×6 grid of (E, H) values for three methods:
  1. TFNBS (Unrestricted)
  2. NI-TFNBS (Soft Network Prior)
  3. FBC-TFNBS (Bayesian Control)

This demonstrates how discovery counts vary across the TF-NBS multiverse
on the ABIDE Age developmental contrast, and how the network-informed
priors constrain the result across different topological regimes.

Grid
----
E ∈ {0.20, 0.40, 0.50, 0.75, 1.00, 1.30}  (6 values)
H ∈ {1.00, 2.00, 3.00, 5.00, 7.00, 10.00} (6 values)

Design: the same model as the age map (run_age_strategy_d.py --variant age):
age with sex and motion as covariates, ComBat + site dummies, permutation
within site, exact p-values from 5,000 permutations (no tail approximation).
The (0.4, 3.0) cell therefore reproduces the age map.

Output
------
results/age_development/sensitivity/eh/
  grid_summary.csv                    # per-method per-cell n_sig (p <= 0.05)
  pmaps_<method>.npz                  # p-maps, (N, N, 36) per tail
  spearman_matrix[_neg].csv           # TFNBS: pairwise Spearman of -log10 p across cells
  spearman_below_ceiling_matrix[_neg].csv  # same, edges with p < 1 in either cell
  jaccard_matrix[_neg].csv            # TFNBS: pairwise Jaccard of significant edge sets
  multiverse_summary.csv              # median / range of the above (Table tfnbs-multiverse)
results/plots/plot5_tfnbs_grid_sensitivity.png
"""
from __future__ import annotations


from pathlib import Path as _Path
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from conninfpy import analyze
from examples.abide._shared.data import load_prepared

HERE = Path(__file__).resolve().parents[1]
DATA_FILE = HERE / "results" / "abide_prepared.npz"
OUT_DIR = HERE / "results" / "age_development" / "sensitivity" / "eh"
PLOTS_DIR = HERE / "results" / "plots"

E_GRID = (0.20, 0.40, 0.50, 0.75, 1.00, 1.30)
H_GRID = (1.00, 2.00, 3.00, 5.00, 7.00, 10.00)

PUBLISHED_DEFAULTS = {
    (0.40, 3.00): "Hao 2024",
    (0.50, 2.00): "Smith & Nichols 2009",
    (0.75, 3.00): "Baggio 2018",
}


def _load(path: Path = DATA_FILE) -> dict:
    return load_prepared(path)


def _pairwise(pmaps: np.ndarray, labels: list, alpha: float) -> tuple:
    """Similarity between cells: Spearman of -log10 p and Jaccard of significant sets.

    Most edges sit at the ceiling p = 1 in every cell, and these ties push the
    all-edge Spearman towards 1. The third matrix is the Spearman over edges
    below the ceiling in at least one of the two cells.
    """
    neglog = -np.log10(pmaps)                      # (n_edges, K)
    rho = spearmanr(neglog).correlation
    K = pmaps.shape[1]
    rho_below = np.ones((K, K))
    for a in range(K):
        for b in range(a + 1, K):
            keep = (pmaps[:, a] < 1) | (pmaps[:, b] < 1)
            rho_below[a, b] = rho_below[b, a] = spearmanr(pmaps[keep, a], pmaps[keep, b]).correlation
    sig = pmaps <= alpha
    inter = sig.T.astype(int) @ sig.astype(int)
    union = sig.sum(0)[:, None] + sig.sum(0)[None, :] - inter
    jac = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    frame = lambda m: pd.DataFrame(m, index=labels, columns=labels)
    return frame(rho), frame(rho_below), frame(jac)


def _make_eh_panels(
    summary_df: pd.DataFrame,
    out_path: str,
):
    """6-panel (E, H) sensitivity figure comparing TFNBS, NI, and FBC."""
    import matplotlib.pyplot as plt

    nE, nH = len(E_GRID), len(H_GRID)

    def _pivot(method: str, tail: str):
        sub = summary_df[summary_df["method"] == method]
        col = f"n_sig_{tail}"
        return (
            sub.pivot_table(index="E", columns="H", values=col, aggfunc="first")
            .reindex(index=E_GRID, columns=H_GRID)
        )

    fig = plt.figure(figsize=(18, 10))
    panels = [
        (_pivot("tfnbs", "pos"), "YlGn", "TFNBS Detection (+)", "#"),
        (_pivot("ni_tfnbs", "pos"), "YlGn", "NI-TFNBS Detection (+)", "#"),
        (_pivot("fbc_tfnbs", "pos"), "YlGn", "FBC-TFNBS Detection (+)", "#"),

        (_pivot("tfnbs", "neg"), "YlGn", "TFNBS Detection (-)", "#"),
        (_pivot("ni_tfnbs", "neg"), "YlGn", "NI-TFNBS Detection (-)", "#"),
        (_pivot("fbc_tfnbs", "neg"), "YlGn", "FBC-TFNBS Detection (-)", "#"),
    ]
    layout = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for (gi, gj), (data, cmap, title, label) in zip(layout, panels):
        ax = plt.subplot2grid((2, 3), (gi, gj))
        im = ax.imshow(data.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(nH))
        ax.set_yticks(range(nE))
        ax.set_xticklabels([f"{h:.1f}" for h in H_GRID], fontsize=8)
        ax.set_yticklabels([f"{e:.2f}" for e in E_GRID], fontsize=8)
        ax.set_xlabel("H (height exponent)", fontsize=9)
        ax.set_ylabel("E (extent exponent)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label, fontsize=8)

        # Annotate counts
        for i in range(nE):
            for j in range(nH):
                v = data.values[i, j]
                if not np.isfinite(v):
                    continue
                txt_color = "white" if v > data.values.max() * 0.5 else "black"
                ax.text(j, i, f"{int(v)}", ha="center", va="center",
                        color=txt_color, fontsize=8)

        # Mark defaults
        for (e, h), _name in PUBLISHED_DEFAULTS.items():
            if e in E_GRID and h in H_GRID:
                ax.add_patch(plt.Rectangle(
                    (H_GRID.index(h) - 0.45, E_GRID.index(e) - 0.45),
                    0.90, 0.90,
                    fill=False, edgecolor="red", linewidth=1.5,
                ))

    fig.suptitle(
        "Multiverse Hierarchy Sensitivity Sweep — ABIDE Age (GLM)\n"
        "Comparison of discovery consistency across topological regimes and network-informed priors",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-permutations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-file", type=Path, default=DATA_FILE)
    ap.add_argument("--output", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    OUT_DIR = args.output.resolve()
    alpha = 0.05

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    data = _load(args.data_file)
    Y = data["connectivity_z"]
    confounds = np.column_stack([
        data["sex"].astype(float),
        data["mean_fd"].astype(float)
    ])
    net_labels = data["net_labels"]
    print(f"Loaded ABIDE: n={Y.shape[0]}, N={Y.shape[1]}")

    cells = list(product(E_GRID, H_GRID))
    e_list = [e for e, _ in cells]
    h_list = [h for _, h in cells]
    labels = [f"E{e:.2f}_H{h:.2f}" for e, h in cells]
    K = len(cells)
    iu = np.triu_indices(Y.shape[1], k=1)

    methods = ["tfnbs", "ni_tfnbs", "fbc_tfnbs"]
    all_rows = []

    for mname in methods:
        print(f"\nRunning {mname} sweep ({K} cells) ...")
        t0 = time.time()
        kwargs = {"net_labels": net_labels} if mname in ("ni_tfnbs", "fbc_tfnbs") else {}
        out = analyze(
            Y, interest=data["age"].astype(float), confounds=confounds,
            sites=data["site"], harmonize="combat_site_dummies_glm",  # as the age map
            fisher_z=False, method=mname,
            e=e_list, h=h_list, n=10,
            n_permutations=args.n_permutations, acceleration=None,
            rng=args.seed, use_mp=True,
            **kwargs
        )
        print(f"  Done in {time.time()-t0:.1f}s")
        pos = np.asarray(out.inference["positive"])
        neg = np.asarray(out.inference["negative"])
        np.savez_compressed(OUT_DIR / f"pmaps_{mname}.npz", positive=pos, negative=neg,
                            cells=np.array(labels))

        pos_edges, neg_edges = pos[iu], neg[iu]
        for k, (e, h) in enumerate(cells):
            all_rows.append(dict(
                method=mname, E=e, H=h,
                n_sig_pos=int((pos_edges[:, k] <= alpha).sum()),
                n_sig_neg=int((neg_edges[:, k] <= alpha).sum()),
            ))

        if mname == "tfnbs":
            summary = []
            for tail, edges, suffix in (("positive", pos_edges, ""), ("negative", neg_edges, "_neg")):
                rho, rho_below, jac = _pairwise(edges, labels, alpha)
                rho.to_csv(OUT_DIR / f"spearman_matrix{suffix}.csv")
                rho_below.to_csv(OUT_DIR / f"spearman_below_ceiling_matrix{suffix}.csv")
                jac.to_csv(OUT_DIR / f"jaccard_matrix{suffix}.csv")
                tri = np.triu_indices(K, k=1)
                summary.append(dict(tail=tail, measure="share_of_edges_at_p1", n_pairs=K,
                                    median=float(np.median((edges >= 1).mean(0))),
                                    min=float((edges >= 1).mean(0).min()),
                                    max=float((edges >= 1).mean(0).max())))
                for name, m in (("spearman_neglog10p", rho),
                                ("spearman_below_ceiling", rho_below), ("jaccard", jac)):
                    v = m.to_numpy()[tri]
                    summary.append(dict(tail=tail, measure=name, n_pairs=len(v),
                                        median=float(np.median(v)), min=float(v.min()),
                                        max=float(v.max())))
            summary = pd.DataFrame(summary)
            summary.to_csv(OUT_DIR / "multiverse_summary.csv", index=False)
            print(summary.round(3).to_string(index=False))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "grid_summary.csv", index=False)

    plot_path = PLOTS_DIR / "plot5_tfnbs_grid_sensitivity.png"
    _make_eh_panels(df, str(plot_path))
    print(f"\nFig saved: {plot_path}")


if __name__ == "__main__":
    main()
