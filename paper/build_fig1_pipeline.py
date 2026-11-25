"""Build Main Figure 1: ConnInfPy 2.0.0 end-to-end workflow diagram (>= 300 DPI).

Outputs:
    - paper/figures/figures/fig1_pipeline.png
    - ../PaperNN/ConnInfPy__connectome_inference/figs/figures/fig1_pipeline.png
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from conninfpy import combat_harmonize
from conninfpy.enhance import (
    get_cnbs_score,
    get_fbc_tfnbs_score,
    get_nbs_score,
    get_network_informed_tfnbs_score,
    get_tfnbs_score,
)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "figure.dpi": 150,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_REPO = REPO_ROOT / "paper" / "figures" / "figures" / "fig1_pipeline.png"
OUT_PAPER = (
    REPO_ROOT.parent
    / "PaperNN"
    / "ConnInfPy__connectome_inference"
    / "figs"
    / "figures"
    / "fig1_pipeline.png"
)


def _make_modular_mask(n_nodes: int = 24, n_modules: int = 3, effect_blocks=(0,)) -> np.ndarray:
    mod_size = n_nodes // n_modules
    mask = np.zeros((n_nodes, n_nodes), dtype=int)
    for m in effect_blocks:
        sl = slice(m * mod_size, (m + 1) * mod_size)
        mask[sl, sl] = 1
    np.fill_diagonal(mask, 0)
    return mask


def _make_subject_fc(
    n_subj: int = 9,
    n_nodes: int = 24,
    site_effects: np.ndarray | None = None,
    planted_mask: np.ndarray | None = None,
    planted_strength: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_modules = 3
    mod_size = n_nodes // n_modules
    base = np.full((n_nodes, n_nodes), 0.05)
    for m in range(n_modules):
        sl = slice(m * mod_size, (m + 1) * mod_size)
        base[sl, sl] = 0.30
    np.fill_diagonal(base, 1.0)
    base = 0.5 * (base + base.T)
    fc_stack = np.zeros((n_subj, n_nodes, n_nodes))
    for i in range(n_subj):
        noise = rng.normal(0, 0.04, (n_nodes, n_nodes))
        noise = 0.5 * (noise + noise.T)
        mat = base + noise
        if site_effects is not None:
            mat += site_effects[i]
        if planted_mask is not None:
            mat += planted_strength * planted_mask
        np.fill_diagonal(mat, 1.0)
        fc_stack[i] = mat
    return fc_stack


def _box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    fc: str = "#f4f7fb",
    ec: str = "#3b5978",
    title_color: str = "#16293d",
    lw: float = 1.0,
    title_size: float = 7.6,
    body_size: float = 6.5,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        fc=fc,
        ec=ec,
        lw=lw,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.02 * w + 0.012,
        y + h - 0.025,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        zorder=3,
    )
    ax.text(
        x + 0.02 * w + 0.012,
        y + h - 0.145,
        body,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=body_size,
        color="#23303d",
        linespacing=1.28,
        zorder=3,
    )


def _arrow(ax, x0: float, y0: float, x1: float, y1: float, color: str = "#3b5978"):
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=10,
        lw=1.1,
        color=color,
        zorder=4,
    )
    ax.add_patch(arr)


def panel_A_inputs(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Left: miniature subject FC stack
    fig = ax.figure
    sub = ax.get_subplotspec().subgridspec(1, 2, width_ratios=[1.05, 1.35], wspace=0.08)
    sub_left = sub[0].subgridspec(2, 2, wspace=0.10, hspace=0.12)
    mask = _make_modular_mask(24, 3, (0,))
    site_eff = np.array([-0.06, 0.0, 0.06, 0.0])
    fc = _make_subject_fc(4, 24, site_effects=site_eff, planted_mask=mask, planted_strength=0.18)
    site_cols = ["#0072B2", "#E69F00", "#009E73", "#0072B2"]
    for k in range(4):
        a = fig.add_subplot(sub_left[k // 2, k % 2])
        a.imshow(fc[k], cmap="viridis", vmin=-0.05, vmax=0.45)
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_edgecolor(site_cols[k])
            sp.set_linewidth(1.8)
    a_r = fig.add_subplot(sub[1])
    a_r.axis("off")
    _box(
        a_r,
        0.02,
        0.05,
        0.96,
        0.90,
        "Inputs & Pre-Inference QC",
        "• Time series → compute_connectivity()\n"
        "  (Pearson, Ledoit–Wolf, OAS, n_valid_tr)\n"
        "• Connectivity matrices (N × P × P)\n"
        "• missing='available_case' (constant ROIs)\n"
        "• motion_qc(): QC–FC r vs Euclidean dist.",
        fc="#eef5fb",
        ec="#0072B2",
    )


def panel_B_design(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(
        ax,
        0.01,
        0.53,
        0.98,
        0.43,
        "Path 1: t-test (no confounds / no ComBat)",
        "• analyze(group1, group2, test_type='two-sample'|'paired')\n"
        "• Welch / Student / paired t; exact sign-flip or label perm.",
        fc="#f3f6fa",
        ec="#4d648d",
    )
    _box(
        ax,
        0.01,
        0.04,
        0.98,
        0.43,
        "Path 2: Freedman–Lane GLM (confounds & multi-site)",
        "• analyze(Y, interest=x, confounds=C, interactions=..., strata=s)\n"
        "• Group × Condition, multi-contrast in 1 pass (compute_p_val_glm_multi)",
        fc="#edf7f1",
        ec="#009E73",
    )


def panel_C_multisite(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig = ax.figure
    sub = ax.get_subplotspec().subgridspec(1, 2, width_ratios=[0.95, 1.50], wspace=0.08)
    sub_im = sub[0].subgridspec(1, 2, wspace=0.10)
    rng = np.random.default_rng(7)
    sites = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    site_eff = np.array([(s - 1) * 0.08 for s in sites])
    fc = _make_subject_fc(9, 24, site_effects=site_eff, planted_mask=_make_modular_mask(24, 3, (0,)), planted_strength=0.18)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc_harm = combat_harmonize(fc, sites=sites).Y_adjusted
    a0 = fig.add_subplot(sub_im[0])
    a1 = fig.add_subplot(sub_im[1])
    a0.imshow(fc.mean(axis=0), cmap="viridis", vmin=-0.05, vmax=0.45)
    a1.imshow(fc_harm.mean(axis=0), cmap="viridis", vmin=-0.05, vmax=0.45)
    a0.set_title("Before", fontsize=6.8, pad=2)
    a1.set_title("ComBat", fontsize=6.8, pad=2)
    for a in (a0, a1):
        a.set_xticks([])
        a.set_yticks([])

    a_r = fig.add_subplot(sub[1])
    a_r.axis("off")
    _box(
        a_r,
        0.02,
        0.05,
        0.96,
        0.90,
        "Multi-Site Handling (GLM Path Only)",
        "• Site dummies in X (Strategy E)\n"
        "• ComBat EB (nuisance_only, fixed-pt)\n"
        "• ComBat + site dummies (Strategy D)\n"
        "• Site-blocked perm. (strata=sites)\n"
        "• Note: skipped on two-sample path",
        fc="#fff6eb",
        ec="#D55E00",
        title_size=7.0,
        body_size=6.2,
    )


def panel_D_operators(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig = ax.figure
    sub = ax.get_subplotspec().subgridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.06)

    # Left: 6 actual conninfpy.enhance operator outputs on a signed synthetic t-matrix
    sub_ops = sub[0].subgridspec(1, 6, wspace=0.14)
    rng = np.random.default_rng(11)
    n_nodes, n_mod = 24, 3
    mod_sz = n_nodes // n_mod
    mask = _make_modular_mask(n_nodes, n_mod, (0,))
    T = rng.normal(0, 0.9, (n_nodes, n_nodes))
    T = 0.5 * (T + T.T) + 2.6 * mask
    np.fill_diagonal(T, 0.0)
    bl = np.repeat(np.arange(n_mod), mod_sz)

    nbs_ext = get_nbs_score(T, threshold=2.0, stat_type="extent")
    nbs_int = get_nbs_score(T, threshold=2.0, stat_type="intensity")
    tfnbs_sc = get_tfnbs_score(T, e=0.4, h=3.0, n=None, start_thres=1.65)
    cnbs_sc = get_cnbs_score(T, net_labels=bl)
    ni_sc = get_network_informed_tfnbs_score(T, net_labels=bl, e=0.4, h=3.0, n=None, start_thres=1.65)
    fbc_sc = get_fbc_tfnbs_score(T, net_labels=bl, e=0.4, h=3.0, n=None, start_thres=1.65, min_cluster_size=3)

    ops = [
        (nbs_ext, r"NBS Ext ($\tau=2$)"),
        (nbs_int, r"NBS Int $\Sigma(t-\tau)$"),
        (tfnbs_sc, "Exact TFNBS\n(n=None default)"),
        (cnbs_sc, "cNBS\n(signed block)"),
        (ni_sc, "NI-TFNBS\n(fixed $M_B$)"),
        (fbc_sc, r"FBC-TFNBS" + "\n" + r"($m_{\min}=3$)"),
    ]
    for k, (mat, ttl) in enumerate(ops):
        a = fig.add_subplot(sub_ops[k])
        vmax = float(np.max(mat)) if np.any(mat > 0) else 1.0
        vmin = float(np.min(mat)) if np.any(mat < 0) else 0.0
        cmap = "RdBu_r" if vmin < 0 else "viridis"
        if vmin < 0:
            vlim = max(abs(vmin), abs(vmax), 1e-6)
            a.imshow(mat, cmap=cmap, vmin=-vlim, vmax=vlim)
        else:
            a.imshow(mat, cmap=cmap, vmin=0, vmax=vmax)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(ttl, fontsize=6.6, pad=3)
        for kk in range(1, n_mod):
            a.axvline(kk * mod_sz - 0.5, color="white", lw=0.4, alpha=0.65)
            a.axhline(kk * mod_sz - 0.5, color="white", lw=0.4, alpha=0.65)

    a_r = fig.add_subplot(sub[1])
    a_r.axis("off")
    _box(
        a_r,
        0.02,
        0.04,
        0.96,
        0.92,
        "Permutation FWER Engine",
        "• Default: exact piecewise TFNBS (n=None)\n"
        "  & empirical permutation p = (b+1)/(B+1)\n"
        "  (acceleration=None; GPD/gamma opt-in)\n"
        "• analyze() default: two_tailed=True\n"
        "  joint max-null M = max(M+, M-) + tested_tail",
        fc="#f3f1fa",
        ec="#6b4c9a",
    )


def panel_E_interpretation(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(
        ax,
        0.005,
        0.06,
        0.315,
        0.88,
        "1. Unit-Preserving Summaries",
        "• ResultBundle & *_meta.json\n"
        "• component_report() (connected comp.)\n"
        "• node_summary() (hub degree / mass)\n"
        "• block_report() (network-pair mass)\n"
        "• subject_influence(): LOO stability",
        fc="#f0f6fa",
        ec="#0072B2",
    )
    _arrow(ax, 0.323, 0.50, 0.343, 0.50, "#0072B2")
    _box(
        ax,
        0.346,
        0.06,
        0.315,
        0.88,
        "2. Tissue-First BNA-246 + SUIT",
        "• annotate_rois(): sphere / parcel / point\n"
        "• purity >= 0.50; explicit mixed / unassigned\n"
        "• Opt-in cerebellum='anatom' (SUIT)\n"
        "• Coordinate space provenance\n"
        "• BrainMap BD/PC (descriptive only)",
        fc="#edf7f1",
        ec="#009E73",
    )
    _arrow(ax, 0.664, 0.50, 0.684, 0.50, "#009E73")
    _box(
        ax,
        0.687,
        0.06,
        0.308,
        0.88,
        "3. Neuroanatomical Visuals",
        "• plot_connectogram(): L/R lobes + CBM\n"
        "  sector & crossed cerebello-cortical\n"
        "• coupling() & plot_coupling_classes():\n"
        "  stronger pos. / weaker neg. / sign flip\n"
        "• Automatic BNA homolog_id markers",
        fc="#fdf3eb",
        ec="#CC79A7",
    )


def main():
    fig = plt.figure(figsize=(13.5, 9.2), constrained_layout=False)
    gs = GridSpec(
        nrows=3,
        ncols=3,
        figure=fig,
        height_ratios=[1.05, 0.95, 0.85],
        width_ratios=[1.05, 1.0, 1.05],
        left=0.03,
        right=0.98,
        top=0.90,
        bottom=0.03,
        wspace=0.14,
        hspace=0.36,
    )

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])
    ax_D = fig.add_subplot(gs[1, :])
    ax_E = fig.add_subplot(gs[2, :])

    panel_A_inputs(ax_A)
    panel_B_design(ax_B)
    panel_C_multisite(ax_C)
    panel_D_operators(ax_D)
    panel_E_interpretation(ax_E)

    # Panel headers
    fig.text(0.03, 0.922, "A.  Inputs & Motion QC", fontsize=9.5, fontweight="bold", color="#16293d")
    fig.text(0.37, 0.922, "B.  Statistical Design", fontsize=9.5, fontweight="bold", color="#16293d")
    fig.text(0.68, 0.922, "C.  Multi-Site Harmonization", fontsize=9.5, fontweight="bold", color="#16293d")
    fig.text(0.03, 0.595, "D.  Topology-Aware Enhancement & Permutation FWER", fontsize=9.5, fontweight="bold", color="#16293d")
    fig.text(0.03, 0.285, "E.  Neuroanatomical Interpretation Layer (Tissue-First BNA-246 + Opt-in SUIT Cerebellum)", fontsize=9.5, fontweight="bold", color="#16293d")

    fig.suptitle(
        "ConnInfPy 2.0.0 End-to-End Connectome Inference & Neuroanatomical Interpretation Workflow",
        fontsize=11.5,
        fontweight="bold",
        y=0.975,
        color="#102030",
    )

    for out in (OUT_REPO, OUT_PAPER):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
