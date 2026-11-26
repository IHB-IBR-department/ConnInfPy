"""Build Main Figure 1: ConnInfPy 2.0 end-to-end workflow diagram (600 DPI).

Outputs:
    - paper/figures/figures/fig1_pipeline.png
    - ../PaperNN/ConnInfPy__connectome_inference/figs/figures/fig1_pipeline.png
"""

from __future__ import annotations

import argparse
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
    linespacing: float = 1.28,
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
        linespacing=linespacing,
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
        "Connectivity Inputs",
        "• Time series → compute_connectivity()\n"
        "  (Pearson, Ledoit–Wolf, OAS, n_valid_tr)\n"
        "• Connectivity matrices (N × P × P)\n"
        "• missing='available_case'\n"
        "  (handles constant / missing ROIs)\n"
        "• Opt-in motion_qc: descriptive QC–FC\n"
        "  (Spearman r vs FD/dist.; no filtering)",
        fc="#eef5fb",
        ec="#0072B2",
        title_size=8.3,
        body_size=7.0,
        linespacing=1.50,
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
        title_size=8.3,
        body_size=7.1,
        linespacing=1.42,
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
        title_size=8.3,
        body_size=7.0,
        linespacing=1.42,
    )


def panel_C_multisite(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rng = np.random.default_rng(7)
    n_per_site = 8
    sites = np.repeat([0, 1, 2], n_per_site)
    site_shifts = np.array([-0.10, 0.0, 0.10])
    site_scales = np.array([0.65, 1.0, 1.45])
    subj_var = rng.normal(0.0, 0.025, size=sites.size) * site_scales[sites]
    site_eff = site_shifts[sites] + subj_var
    fc = _make_subject_fc(
        sites.size,
        24,
        site_effects=site_eff,
        planted_mask=_make_modular_mask(24, 3, (0,)),
        planted_strength=0.18,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc_harm = combat_harmonize(fc, sites=sites).Y_adjusted

    iu = np.triu_indices(24, 1)
    mean_before = fc[:, iu[0], iu[1]].mean(axis=1)
    mean_after = fc_harm[:, iu[0], iu[1]].mean(axis=1)

    _box(
        ax,
        0.01,
        0.04,
        0.98,
        0.92,
        "Multi-Site Handling (GLM Path Only)",
        "• Site dummies in GLM (no ComBat)\n"
        "• ComBat only (preserve confounds)\n"
        "• ComBat + site dummies (default)\n"
        "• Site-blocked perm. (strata=sites)\n"
        "• Note: skipped on two-sample path",
        fc="#fff6eb",
        ec="#D55E00",
        title_size=8.0,
        body_size=6.8,
        linespacing=1.28,
    )

    a0 = ax.inset_axes([0.18, 0.10, 0.30, 0.27], zorder=4)
    a1 = ax.inset_axes([0.60, 0.10, 0.30, 0.27], zorder=4)
    site_cols = ["#0072B2", "#E69F00", "#009E73"]
    all_vals = np.concatenate([mean_before, mean_after])
    pad = 0.03
    y_lo, y_hi = float(all_vals.min() - pad), float(all_vals.max() + pad)
    grand_mean = float(mean_after.mean())

    for sub_ax, vals, title in ((a0, mean_before, "Before"), (a1, mean_after, "ComBat")):
        sub_ax.set_facecolor("white")
        sub_ax.axhline(grand_mean, color="#9aa5b1", ls="--", lw=0.7, zorder=1)
        data = [vals[sites == s] for s in range(3)]
        bp = sub_ax.boxplot(
            data,
            positions=[0, 1, 2],
            widths=0.50,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#16293d", "lw": 1.1},
            whiskerprops={"color": "#52606d", "lw": 0.8},
            capprops={"color": "#52606d", "lw": 0.8},
        )
        for patch, col in zip(bp["boxes"], site_cols):
            patch.set_facecolor(col)
            patch.set_alpha(0.35)
            patch.set_edgecolor(col)
            patch.set_linewidth(0.9)
        for s, col in enumerate(site_cols):
            xs = s + rng.uniform(-0.14, 0.14, size=n_per_site)
            sub_ax.scatter(xs, data[s], s=8, color=col, alpha=0.85, linewidths=0, zorder=3)
        sub_ax.set_ylim(y_lo, y_hi)
        sub_ax.set_xticks([0, 1, 2])
        sub_ax.set_xticklabels(["S1", "S2", "S3"], fontsize=6.4)
        sub_ax.set_title(title, fontsize=7.0, pad=2)
        sub_ax.tick_params(axis="y", labelsize=6.0, length=2, pad=1)
        sub_ax.tick_params(axis="x", length=2, pad=1)
        for sp in ("top", "right"):
            sub_ax.spines[sp].set_visible(False)

    a0.set_ylabel("Mean FC", fontsize=6.4, labelpad=2)
    a1.set_yticklabels([])
    _arrow(ax, 0.50, 0.24, 0.58, 0.24, "#D55E00")


def panel_D_operators(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig = ax.figure
    sub = ax.get_subplotspec().subgridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.06)

    # Left: 6 actual conninfpy.enhance operator outputs on a signed synthetic t-matrix
    # Unequal functional blocks [4, 8, 12] + disjoint/bridge edges so TFNBS
    # (topological), cNBS (signed block), NI-TFNBS (block capacity M_B) and
    # FBC-TFNBS (within-block support >= m_min) are all visually distinct.
    sub_ops = sub[0].subgridspec(1, 6, wspace=0.14)
    rng = np.random.default_rng(11)
    n_nodes = 24
    block_sizes = [4, 8, 12]
    boundaries = np.cumsum(block_sizes)[:-1]
    bl = np.concatenate([np.full(sz, idx) for idx, sz in enumerate(block_sizes)])
    T = rng.normal(0, 0.45, (n_nodes, n_nodes))
    T = 0.5 * (T + T.T)
    # 1) Dense cluster in small block 0 (4 nodes, 6 possible edges; 5 active -> high density)
    for i in range(4):
        for j in range(i + 1, 4):
            if not (i == 2 and j == 3):
                T[i, j] = T[j, i] = 3.1
    # 2) Negative block in medium block 1 (nodes 4..11) -> visible in signed cNBS
    for i in range(4, 12):
        for j in range(i + 1, 12):
            T[i, j] = T[j, i] = -2.4
    # 3) 6 disjoint edges in large block 2 (nodes 12..23, no shared nodes):
    #    weak in TFNBS (extent=1), diluted by sqrt(M_B) in NI-TFNBS, boosted in FBC-TFNBS (k=6)
    for i, j in ((12, 13), (14, 15), (16, 17), (18, 19), (20, 21), (22, 23)):
        T[i, j] = T[j, i] = 2.9
    # 4) 2 cross-block edges in (0, 2) bridging block 0 to (12, 13):
    #    pulled into block 0's topological cluster in TFNBS, suppressed in FBC-TFNBS (< 3)
    for i, j in ((0, 12), (1, 14)):
        T[i, j] = T[j, i] = 2.9
    np.fill_diagonal(T, 0.0)

    nbs_ext = get_nbs_score(T, threshold=2.0, stat_type="extent")
    nbs_int = get_nbs_score(T, threshold=2.0, stat_type="intensity")
    tfnbs_sc = get_tfnbs_score(T, e=0.4, h=3.0, start_thres=1.65)
    cnbs_sc = get_cnbs_score(T, net_labels=bl)
    ni_sc = get_network_informed_tfnbs_score(T, net_labels=bl, e=0.4, h=3.0, start_thres=1.65)
    fbc_sc = get_fbc_tfnbs_score(T, net_labels=bl, e=0.4, h=3.0, start_thres=1.65, min_cluster_size=3)

    ops = [
        (nbs_ext, r"NBS Ext ($\tau=2$)"),
        (nbs_int, r"NBS Int $\Sigma(t-\tau)$"),
        (tfnbs_sc, "TFNBS\n(E=0.4, H=3.0)"),
        (cnbs_sc, "cNBS\n(signed block)"),
        (ni_sc, "NI-TFNBS\n(block density)"),
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
        a.set_title(ttl, fontsize=7.3, pad=3)
        for b in boundaries:
            a.axvline(b - 0.5, color="white", lw=0.4, alpha=0.65)
            a.axhline(b - 0.5, color="white", lw=0.4, alpha=0.65)

    a_r = fig.add_subplot(sub[1])
    a_r.axis("off")
    _box(
        a_r,
        0.02,
        0.04,
        0.96,
        0.92,
        "Permutation FWER Engine",
        "• Default: TFNBS — step-free integration\n"
        "  (no grid-step hyperparameter; legacy n kept)\n"
        "• Accepts single (E, H) pair or full (E, H) grid\n"
        "• Empirical perm. p = (b+1)/(B+1); opt-in GPD/gamma\n"
        "• two_tailed=True (joint max-null) or tested_tail",
        fc="#f3f1fa",
        ec="#6b4c9a",
        title_size=8.6,
        body_size=7.4,
        linespacing=1.50,
    )


def panel_E_interpretation(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig = ax.figure
    sub = ax.get_subplotspec().subgridspec(1, 2, width_ratios=[1.65, 1.0], wspace=0.04)

    # Left: 3-stage interpretation pipeline cards
    ax_left = fig.add_subplot(sub[0])
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)
    ax_left.axis("off")
    _box(
        ax_left,
        0.012,
        0.05,
        0.296,
        0.88,
        "1. Unit-Preserving",
        "• ResultBundle & *_meta.json\n"
        "• component_report()\n"
        "  (connected components)\n"
        "• node_summary() (hub mass)\n"
        "• block_report() (net-pairs)\n"
        "• subject_influence() (LOO)",
        fc="#f0f6fa",
        ec="#0072B2",
        title_size=7.3,
        body_size=6.35,
        linespacing=1.35,
    )
    _arrow(ax_left, 0.322, 0.49, 0.338, 0.49, "#0072B2")
    _box(
        ax_left,
        0.352,
        0.05,
        0.296,
        0.88,
        "2. BNA-246 + SUIT Cerebellum",
        "• Tissue-first annotate_rois()\n"
        "  (sphere / parcel / point)\n"
        "• purity >= 0.50; explicit\n"
        "  mixed / unassigned ROIs\n"
        "• Opt-in cerebellum='anatom'\n"
        "• Coord. space & BrainMap",
        fc="#edf7f1",
        ec="#009E73",
        title_size=7.0,
        body_size=6.35,
        linespacing=1.35,
    )
    _arrow(ax_left, 0.662, 0.49, 0.678, 0.49, "#009E73")
    _box(
        ax_left,
        0.692,
        0.05,
        0.296,
        0.88,
        "3. Neuroanatomical Visuals",
        "• plot_connectogram():\n"
        "  L/R lobes + CBM sector &\n"
        "  crossed cerebello-cortical\n"
        "• coupling() & class matrix:\n"
        "  stronger + / weaker − / flip\n"
        "• BNA homolog_id markers",
        fc="#fdf3eb",
        ec="#CC79A7",
        title_size=7.3,
        body_size=6.35,
        linespacing=1.35,
    )

    # Right: Connectogram + Coupling-Classes preview using ConnInfPy palettes
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import PathPatch, Wedge
    from matplotlib.path import Path as MplPath
    from conninfpy.plot.connectograms import _LOBES, _YEO7_COLORS
    from conninfpy.plot.coupling import COUPLING_CLASS_COLORS

    ax_vis = fig.add_subplot(sub[1])
    ax_vis.axis("off")

    # 3a) Miniature BNA-style connectogram (L/R lobes + CBM sector)
    ax_cg = ax_vis.inset_axes([0.00, 0.03, 0.50, 0.80])
    ax_cg.set_aspect("equal")
    ax_cg.set_xlim(-1.28, 1.28)
    ax_cg.set_ylim(-1.28, 1.24)
    ax_cg.axis("off")
    ax_cg.set_title("plot_connectogram()\nL/R lobes + CBM", fontsize=7.1, pad=2)

    lobe_order = ["Frontal", "Insular", "Limbic", "Temporal", "Parietal", "Occipital", "Subcortical"]
    net_cols = list(_YEO7_COLORS.values())
    # Right side: 90 -> -72 deg (clockwise from top); Left side: 90 -> 252 deg; CBM at bottom (-108 -> -72)
    span_each = 160.0 / len(lobe_order)
    node_angles = {}
    for side, sign in (("R", -1.0), ("L", 1.0)):
        for idx, lobe in enumerate(lobe_order):
            code, col = _LOBES[lobe]
            t0 = 90.0 + sign * (4.0 + idx * span_each)
            t1 = 90.0 + sign * (4.0 + (idx + 1) * span_each - 1.8)
            a_lo, a_hi = min(t0, t1), max(t0, t1)
            ax_cg.add_patch(Wedge((0, 0), 1.02, a_lo, a_hi, width=0.16, fc=col, ec="white", lw=0.6))
            ax_cg.add_patch(Wedge((0, 0), 0.84, a_lo, a_hi, width=0.10, fc=net_cols[idx % len(net_cols)], ec="white", lw=0.5, alpha=0.88))
            mid = 0.5 * (a_lo + a_hi)
            node_angles[(side, lobe)] = np.deg2rad(mid)
            if lobe in ("Frontal", "Temporal", "Parietal", "Occipital", "Subcortical"):
                lx, ly = 1.15 * np.cos(np.deg2rad(mid)), 1.15 * np.sin(np.deg2rad(mid))
                ax_cg.text(lx, ly, code, fontsize=5.5, ha="center", va="center", color="#23303d")

    # Bottom Cerebellum (CBM) sector
    _, cbm_col = _LOBES["Cerebellum"]
    ax_cg.add_patch(Wedge((0, 0), 1.02, -108.0, -72.0, width=0.16, fc=cbm_col, ec="white", lw=0.7))
    ax_cg.add_patch(Wedge((0, 0), 0.84, -108.0, -72.0, width=0.10, fc=net_cols[5], ec="white", lw=0.5, alpha=0.88))
    ax_cg.text(0.0, -1.16, "CBM", fontsize=5.8, fontweight="bold", ha="center", va="center", color="#23303d")
    ax_cg.text(-1.12, 1.06, "L", fontsize=6.8, fontweight="bold", ha="center", va="center", color="#16293d")
    ax_cg.text(1.12, 1.06, "R", fontsize=6.8, fontweight="bold", ha="center", va="center", color="#16293d")
    node_angles[("M", "CBM_L")] = np.deg2rad(-98.0)
    node_angles[("M", "CBM_R")] = np.deg2rad(-82.0)

    chords = [
        (("L", "Frontal"), ("R", "Frontal"), "#CD3E4E", 1.5),
        (("L", "Frontal"), ("L", "Parietal"), "#E69422", 1.3),
        (("R", "Frontal"), ("R", "Temporal"), "#4682B4", 1.1),
        (("L", "Parietal"), ("R", "Parietal"), "#00760E", 1.2),
        (("M", "CBM_L"), ("R", "Frontal"), "#781286", 1.5),
        (("M", "CBM_R"), ("L", "Parietal"), "#781286", 1.3),
        (("L", "Subcortical"), ("R", "Frontal"), "#C43AFA", 0.9),
    ]
    r_in = 0.72
    for src, dst, col, lw in chords:
        a1, a2 = node_angles[src], node_angles[dst]
        p0 = (r_in * np.cos(a1), r_in * np.sin(a1))
        p2 = (r_in * np.cos(a2), r_in * np.sin(a2))
        path = MplPath([p0, (0.0, 0.0), p2], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
        ax_cg.add_patch(PathPatch(path, fc="none", ec=col, lw=lw, alpha=0.78))

    # 3b) Miniature coupling-classes matrix with BNA homolog_id markers
    ax_cp = ax_vis.inset_axes([0.53, 0.12, 0.45, 0.71])
    ax_cp.set_title("plot_coupling_classes()\nclasses + homologs", fontsize=7.1, pad=2)
    class_names = [
        "no change",
        "stronger positive coupling",
        "weaker positive coupling",
        "stronger anticorrelation",
        "weaker anticorrelation",
        "sign change: positive to negative",
        "sign change: negative to positive",
    ]
    cmap_cp = ListedColormap([COUPLING_CLASS_COLORS[c] for c in class_names])
    n_cp = 14
    mat = np.zeros((n_cp, n_cp), dtype=float)
    # Plant structured blocks of coupling classes (symmetric)
    mat[0:4, 0:4] = 1        # stronger positive
    mat[4:7, 4:7] = 2        # weaker positive
    mat[0:4, 8:11] = 3       # stronger anticorrelation
    mat[8:11, 0:4] = 3
    mat[7:11, 7:11] = 4      # weaker anticorrelation
    mat[2:5, 11:14] = 5      # sign flip + -> -
    mat[11:14, 2:5] = 5
    mat[11:14, 11:14] = 6    # sign flip - -> +
    np.fill_diagonal(mat, np.nan)
    cmap_cp.set_bad("#f5f5f5")
    ax_cp.imshow(mat, cmap=cmap_cp, vmin=-0.5, vmax=len(class_names) - 0.5, interpolation="nearest")
    # Overlay BNA homolog_id pairs (L/R paired indices: (i, i+7))
    hom_i = np.array([0, 1, 2, 3, 4, 5, 6])
    hom_j = hom_i + 7
    ax_cp.scatter(
        np.concatenate([hom_j, hom_i]),
        np.concatenate([hom_i, hom_j]),
        s=10,
        facecolors="white",
        edgecolors="#16293d",
        linewidths=0.6,
        zorder=4,
    )
    ax_cp.set_xticks([])
    ax_cp.set_yticks([])
    # Compact inline legend beneath coupling matrix
    legend_items = [
        ("#2166ac", "stronger +"),
        ("#b2182b", "stronger −"),
        ("#762a83", "sign flip"),
        ("white", "homolog"),
    ]
    for idx, (col, lbl) in enumerate(legend_items):
        row = idx // 2
        col_i = idx % 2
        x_pt = 0.04 + col_i * 0.52
        y_pt = -0.10 - row * 0.11
        ax_cp.scatter(
            [x_pt],
            [y_pt],
            s=18,
            fc=col,
            ec="#16293d" if col == "white" else col,
            lw=0.6,
            transform=ax_cp.transAxes,
            clip_on=False,
        )
        ax_cp.text(
            x_pt + 0.06,
            y_pt,
            lbl,
            transform=ax_cp.transAxes,
            ha="left",
            va="center",
            fontsize=6.0,
            color="#23303d",
            clip_on=False,
        )


def build_detailed_figure():
    fig = plt.figure(figsize=(13.5, 9.2), constrained_layout=False)
    gs = GridSpec(
        nrows=3,
        ncols=3,
        figure=fig,
        height_ratios=[1.02, 0.92, 0.96],
        width_ratios=[1.12, 1.08, 0.80],
        left=0.03,
        right=0.98,
        top=0.90,
        bottom=0.05,
        wspace=0.14,
        hspace=0.34,
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
    fig.text(0.03, 0.922, "A.  Connectivity Inputs", fontsize=9.5, fontweight="bold", color="#16293d")
    fig.text(0.395, 0.922, "B.  Statistical Design", fontsize=9.5, fontweight="bold", color="#16293d")
    fig.text(0.748, 0.922, "C.  Multi-Site Harmonization", fontsize=9.5, fontweight="bold", color="#16293d")
    fig.text(0.03, ax_D.get_position().y1 + 0.014, "D.  Topology-Aware Enhancement & Permutation FWER", fontsize=9.5, fontweight="bold", color="#16293d")
    fig.text(0.03, ax_E.get_position().y1 + 0.014, "E.  Neuroanatomical Interpretation Layer (Tissue-First BNA-246 + Opt-in SUIT Cerebellum)", fontsize=9.5, fontweight="bold", color="#16293d")

    fig.suptitle(
        "ConnInfPy 2.0 End-to-End Connectome Inference & Neuroanatomical Interpretation Workflow",
        fontsize=11.5,
        fontweight="bold",
        y=0.975,
        color="#102030",
    )

    return fig


def build_manuscript_figure():
    fig, ax = plt.subplots(figsize=(7.15, 5.25))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def card(x, y, w, h, heading, lines, color="#285477"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.012",
            fc="#f5f8fb", ec=color, lw=1.1,
        ))
        ax.text(x + 0.023, y + h - 0.035, heading, fontsize=11, weight="bold", va="top", color=color)
        ax.text(x + 0.023, y + h - 0.084, lines, fontsize=9.7, va="top", linespacing=1.35)

    def link(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#40566c"))

    card(0.05, 0.82, 0.90, 0.14, "a   Connectivity inputs",
         "Time series or connectivity matrices  ·  Design variables and motion QC")
    card(0.05, 0.60, 0.42, 0.15, "b   Two-sample or paired t-test",
         "Label permutations or paired sign flips")
    card(0.53, 0.60, 0.42, 0.15, "Freedman–Lane GLM",
         "Nuisance covariates and contrasts")
    card(0.53, 0.40, 0.42, 0.15, "c   Multisite options within GLM",
         "Site indicators, ComBat sensitivity arms\nand site-restricted permutations", "#215f52")
    card(0.05, 0.21, 0.90, 0.14, "d   Enhancement and permutation inference",
         "Operator scores (exact TFNBS integration)  →  Monte Carlo permutation p-values")
    card(0.05, 0.02, 0.90, 0.14, "e   Neuroanatomical interpretation",
         "Unit-preserving summaries  ·  BNA-246 and optional SUIT annotation")

    for start, end in [
        ((0.26, 0.817), (0.26, 0.756)),
        ((0.74, 0.817), (0.74, 0.756)),
        ((0.74, 0.597), (0.74, 0.555)),
        ((0.50, 0.574), (0.50, 0.354)),
        ((0.26, 0.597), (0.26, 0.354)),
        ((0.74, 0.397), (0.74, 0.354)),
        ((0.50, 0.206), (0.50, 0.164)),
    ]:
        link(start, end)
    ax.plot([0.525, 0.50], [0.63, 0.63], color="#40566c", lw=1.1)
    ax.plot([0.50, 0.50], [0.63, 0.574], color="#40566c", lw=1.1)
    ax.text(0.575, 0.574, "optional", va="center", fontsize=8.5, color="#215f52")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--repo-only", action="store_true", help="Write only the repository PNG")
    outputs.add_argument("--detailed-output", type=Path, help="Render the detailed workflow to the given path")
    args = parser.parse_args()
    if args.detailed_output is not None:
        fig = build_detailed_figure()
        destinations = (args.detailed_output,)
    else:
        fig = build_manuscript_figure()
        destinations = (OUT_REPO,) if args.repo_only else (OUT_REPO, OUT_PAPER)
    for out in destinations:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300 if args.detailed_output else 600, bbox_inches="tight", facecolor="white")
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
