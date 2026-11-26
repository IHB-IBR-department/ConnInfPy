"""Build the 600 DPI manuscript figures for ConnInfPy 2.0.0.

Outputs are written to BOTH:
  - paper/figures/<subdir>/<filename>
  - ../PaperNN/ConnInfPy__connectome_inference/figs/<subdir>/<filename>

Usage:
  python paper/build_manuscript_figures.py --group all
  python paper/build_manuscript_figures.py --group sim
  python paper/build_manuscript_figures.py --group abide
  python paper/build_manuscript_figures.py --group openclose
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend import Legend
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conninfpy.atlas import AtlasInfo, BrainnetomeReference
from conninfpy.harmonize import combat_harmonize
from conninfpy.interpret import annotate_bna_regions, annotate_rois
from conninfpy.plot import plot_connectogram, plot_coupling_classes
from examples.open_close._shared.agreement import block_mass as _block_mass
from examples.open_close._shared.data import OpenCloseDataset

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
})

OUT_REPO = REPO_ROOT / "paper" / "figures"
OUT_PAPER = (
    REPO_ROOT.parent
    / "PaperNN"
    / "ConnInfPy__connectome_inference"
    / "figs"
)
PAPER_DATA = REPO_ROOT / "paper" / "data"
OUTPUT_DIRS = (OUT_REPO, OUT_PAPER)


def _save_both(fig: plt.Figure, rel_path: str, dpi: int = 600) -> None:
    for base in OUTPUT_DIRS:
        dest = base / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Wrote {dest}")
    plt.close(fig)


def _place_connectogram_legend(ax, legend_ax, ncol: int) -> None:
    """Move the network and chord-width keys into their own space."""
    handles, labels = [], []
    for legend in list(ax.get_children()):
        if isinstance(legend, Legend):
            handles.extend(legend.legend_handles)
            labels.extend(text.get_text() for text in legend.get_texts())
            legend.remove()
    for label in list(ax.texts):
        if "ROIs placed" in label.get_text():
            label.remove()
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", ncol=ncol, frameon=False)


def _clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


# =============================================================================
# SIMULATION & TIMING FIGURES (Figs 2, 3, 7, S4, S5)
# =============================================================================

SIM_METHOD_ORDER = [
    "tstat",
    "bonferroni",
    "bh_fdr",
    "nbs_extent_2.0",
    "nbs_intensity_2.0",
    "nbs_extent_3.0",
    "nbs_intensity_3.0",
    "cnbs",
    "tfnbs_exact",
    "ni_tfnbs_exact",
    "fbc_tfnbs_exact",
]

SIM_METHOD_LABELS = {
    "tstat": "Max-t (perm.)",
    "bonferroni": "Bonferroni",
    "bh_fdr": "BH-FDR",
    "nbs_extent_2.0": r"NBS Ext ($\tau=2$)",
    "nbs_intensity_2.0": r"NBS Int ($\tau=2$, $\Sigma(t-\tau)$)",
    "nbs_extent_3.0": r"NBS Ext ($\tau=3$)",
    "nbs_intensity_3.0": r"NBS Int ($\tau=3$, $\Sigma(t-\tau)$)",
    "cnbs": "cNBS",
    "tfnbs_exact": "TFNBS",
    "ni_tfnbs_exact": "NI-TFNBS",
    "fbc_tfnbs_exact": "FBC-TFNBS",
}

GLM_METHOD_ORDER = [
    "tstat",
    "nbs_extent_2.0",
    "nbs_intensity_2.0",
    "nbs_extent_3.0",
    "nbs_intensity_3.0",
    "cnbs",
    "tfnbs",
    "ni_tfnbs",
    "fbc_tfnbs",
]

GLM_METHOD_LABELS = {
    "tstat": "Max-t (perm.)",
    "nbs_extent_2.0": r"NBS Ext ($\tau=2$)",
    "nbs_intensity_2.0": r"NBS Int ($\tau=2$)",
    "nbs_extent_3.0": r"NBS Ext ($\tau=3$)",
    "nbs_intensity_3.0": r"NBS Int ($\tau=3$)",
    "cnbs": "cNBS",
    "tfnbs": "TFNBS",
    "ni_tfnbs": "NI-TFNBS",
    "fbc_tfnbs": "FBC-TFNBS",
}

TOPOLOGY_ORDER = [
    "within_module_dense",
    "between_modules_dense",
    "partial_bipartite_between_modules",
    "gradient_core_periphery_within_module",
    "scattered_cross_block",
    "hub",
    "rich_club",
    "cross_block_connected_chain",
    "chain",
    "fragmented_within_module",
]

TOPOLOGY_LABELS = {
    "within_module_dense": "Within-module dense",
    "between_modules_dense": "Between-modules dense",
    "partial_bipartite_between_modules": "Partial bipartite",
    "gradient_core_periphery_within_module": "Core-periphery",
    "scattered_cross_block": "Scattered cross-block",
    "hub": "Hub",
    "rich_club": "Rich-club",
    "cross_block_connected_chain": "Cross-block chain",
    "chain": "Chain",
    "fragmented_within_module": "Fragmented",
}


def build_fig2_fwer_calibration() -> None:
    sim_res = REPO_ROOT / "examples" / "simulations" / "results"
    df_arch = pd.read_csv(
        sim_res / "_archive" / "pre_nbs_intensity_fix_2026-11-24" / "fpr_calibration" / "fpr_results.csv"
    )
    df_arch = df_arch[
        ~df_arch["method"].isin(["nbs_intensity_2.0", "nbs_intensity_3.0", "tfnbs", "ni_tfnbs", "fbc_tfnbs"])
    ].copy()
    df_nbs = pd.read_csv(sim_res / "fpr_calibration_nbs_int_v2" / "fpr_results.csv")
    df_exact = pd.read_csv(sim_res / "fpr_calibration_tfnbs_exact" / "fpr_results.csv")
    df_ni_fbc_exact = pd.read_csv(sim_res / "fpr_calibration_ni_fbc_exact" / "fpr_results.csv")
    df_pw = pd.concat([df_arch, df_nbs, df_exact, df_ni_fbc_exact], ignore_index=True)

    # Compute per-method, per-n max-tail FWER + simultaneous 95% CI via 97.5% per-tail CP
    pw_rows = []
    for (m, n), grp in df_pw.groupby(["method", "n_samples"]):
        k_pos = int(grp["any_significant_pos"].sum())
        k_neg = int(grp["any_significant_neg"].sum())
        tot = len(grp)
        lo_pos, hi_pos = _clopper_pearson(k_pos, tot, alpha=0.025)
        lo_neg, hi_neg = _clopper_pearson(k_neg, tot, alpha=0.025)
        pw_rows.append({
            "method": m,
            "n": int(n),
            "fwer": max(k_pos / tot, k_neg / tot),
            "lo": max(lo_pos, lo_neg),
            "hi": max(hi_pos, hi_neg),
        })
    pw_sum = pd.DataFrame(pw_rows)

    df_glm = pd.read_csv(sim_res / "glm" / "glm_validation_2_0_1000" / "glm_fwer_summary.csv")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.0, 4.8), gridspec_kw={"width_ratios": [1.18, 1.0]})

    # Exact binomial quantiles for observed frequency at p=0.05, N=1000
    binom_lo, binom_hi = stats.binom.ppf([0.025, 0.975], 1000, 0.05) / 1000.0
    band_label = "95% reference band\nBinomial(1,000, 0.05)"
    sample_styles = {
        10: ("#0072B2", "o"),
        30: ("#E69F00", "s"),
        50: ("#009E73", "^"),
        60: ("#D55E00", "P"),
        100: ("#CC79A7", "D"),
    }

    # Panel A: Pairwise/Two-sample
    n_vals_a = [10, 30, 50, 100]
    y_pos_a = np.arange(len(SIM_METHOD_ORDER))
    offsets_a = np.linspace(-0.26, 0.26, len(n_vals_a))

    axA.axvspan(binom_lo, binom_hi, color="#e8eef5", alpha=0.7, zorder=0, label=band_label)
    axA.axvline(0.05, color="#c0392b", ls="--", lw=1.1, zorder=1, label=r"Nominal $\alpha=0.05$")

    for j, n in enumerate(n_vals_a):
        sub = pw_sum[pw_sum["n"] == n].set_index("method")
        xs, ys, xerr_lo, xerr_hi = [], [], [], []
        for i, m in enumerate(SIM_METHOD_ORDER):
            if m in sub.index:
                r = sub.loc[m]
                xs.append(r["fwer"])
                ys.append(i + offsets_a[j])
                xerr_lo.append(r["fwer"] - r["lo"])
                xerr_hi.append(r["hi"] - r["fwer"])
        color, marker = sample_styles[n]
        axA.errorbar(
            xs, ys, xerr=[xerr_lo, xerr_hi], fmt=marker, markersize=4.0,
            color=color, elinewidth=1.0, capsize=2, label=f"n = {n}", zorder=3
        )

    axA.set_yticks(y_pos_a)
    axA.set_yticklabels([SIM_METHOD_LABELS[m] for m in SIM_METHOD_ORDER])
    axA.invert_yaxis()
    axA.set_xlim(0.0, 0.10)
    axA.set_xlabel("Max-tail FWER\n95% CI from simultaneous per-tail intervals")
    axA.set_title("a   Two-sample null", fontweight="bold", loc="left")
    axA.grid(axis="x", ls=":", alpha=0.4)
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=True, fontsize=6.8, ncol=3)

    # Panel B: GLM 2.0.0 Joint Two-Sided
    n_vals_b = [30, 60, 100]
    y_pos_b = np.arange(len(GLM_METHOD_ORDER))
    offsets_b = np.linspace(-0.22, 0.22, len(n_vals_b))

    axB.axvspan(binom_lo, binom_hi, color="#e8eef5", alpha=0.7, zorder=0, label=band_label)
    axB.axvline(0.05, color="#c0392b", ls="--", lw=1.1, zorder=1, label=r"Nominal $\alpha=0.05$")

    for j, n in enumerate(n_vals_b):
        sub = df_glm[df_glm["n_samples"] == n].set_index("method")
        xs, ys, xerr_lo, xerr_hi = [], [], [], []
        for i, m in enumerate(GLM_METHOD_ORDER):
            if m in sub.index:
                r = sub.loc[m]
                val = float(r["fwer_two_sided_joint"])
                lo = float(r["fwer_two_sided_joint_ci_low"])
                hi = float(r["fwer_two_sided_joint_ci_high"])
                xs.append(val)
                ys.append(i + offsets_b[j])
                xerr_lo.append(val - lo)
                xerr_hi.append(hi - val)
        color, marker = sample_styles[n]
        axB.errorbar(
            xs, ys, xerr=[xerr_lo, xerr_hi], fmt=marker, markersize=4.2,
            color=color, elinewidth=1.0, capsize=2, label=f"n = {n}", zorder=3
        )

    axB.set_yticks(y_pos_b)
    axB.set_yticklabels([GLM_METHOD_LABELS[m] for m in GLM_METHOD_ORDER])
    axB.invert_yaxis()
    axB.set_xlim(0.0, 0.10)
    axB.set_xlabel("Joint two-sided FWER\n95% Clopper–Pearson CI")
    axB.set_title("b   Freedman–Lane GLM null", fontweight="bold", loc="left")
    axB.grid(axis="x", ls=":", alpha=0.4)
    axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=True, fontsize=6.8, ncol=3)

    plt.tight_layout()
    _save_both(fig, "simulation/fig2_fwer_calibration.png")


def _parse_eh(mp: object) -> tuple[float | None, float | None]:
    if not isinstance(mp, str):
        return None, None
    me = re.search(r"\be=([\d.]+)", mp)
    mh = re.search(r"\bh=([\d.]+)", mp)
    return (float(me.group(1)) if me else None, float(mh.group(1)) if mh else None)


def build_fig3_power_heatmap() -> None:
    sim_res = REPO_ROOT / "examples" / "simulations" / "results"
    df_arch = pd.read_csv(
        sim_res / "_archive" / "pre_nbs_intensity_fix_2026-11-24"
        / "power_analysis_1000_30" / "power_vs_effect_size_summary.csv"
    )
    # Exclude stale nbs_intensity and legacy n=10 TFNBS family runs
    keep_arch = ~df_arch["method"].isin(["nbs_intensity_2.0", "nbs_intensity_3.0", "tfnbs", "ni_tfnbs", "fbc_tfnbs"])
    df_arch = df_arch[keep_arch].copy()

    df_nbs = pd.read_csv(sim_res / "power" / "power_analysis_1000_30_nbs_int_v2" / "power_vs_effect_size_summary.csv")
    df_ex = pd.read_csv(sim_res / "power" / "power_tfnbs_exact_04_30" / "power_vs_effect_size_summary.csv")
    df_ex = df_ex[df_ex["method"] == "tfnbs_exact"].copy()
    df_ni_fbc_ex = pd.read_csv(sim_res / "power" / "power_ni_fbc_exact_04_30" / "power_vs_effect_size_summary.csv")

    df = pd.concat([df_arch, df_nbs, df_ex, df_ni_fbc_ex], ignore_index=True)
    df = df[np.isclose(df["effect_size"], 0.125) & df["scenario"].isin(TOPOLOGY_ORDER)].copy()

    tpr_piv = df.pivot(index="method", columns="scenario", values="TPR_mean").reindex(
        index=SIM_METHOD_ORDER, columns=TOPOLOGY_ORDER
    )
    fdp_piv = df.pivot(index="method", columns="scenario", values="FDR_mean").reindex(
        index=SIM_METHOD_ORDER, columns=TOPOLOGY_ORDER
    )

    row_labels = [SIM_METHOD_LABELS[m] for m in SIM_METHOD_ORDER]
    col_labels = [TOPOLOGY_LABELS[t] for t in TOPOLOGY_ORDER]
    tpr_piv.index = row_labels
    tpr_piv.columns = col_labels
    fdp_piv.index = row_labels
    fdp_piv.columns = col_labels

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 6.8), sharex=True)
    sns.heatmap(
        tpr_piv, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.0, vmax=1.0,
        linewidths=0.4, cbar_kws={"label": "Mean TPR"}, ax=ax1, annot_kws={"size": 7.2}
    )
    ax1.set_title("a   Mean edge-level true positive rate", fontweight="bold", loc="left")
    ax1.set_ylabel("")
    ax1.set_xlabel("")

    fdp_max = np.ceil(10 * np.nanmax(fdp_piv.to_numpy())) / 10
    sns.heatmap(
        fdp_piv, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0.0, vmax=max(0.1, fdp_max),
        linewidths=0.4, cbar_kws={"label": "Mean FDP"}, ax=ax2, annot_kws={"size": 7.2}
    )
    ax2.set_title("b   Mean edge-level false discovery proportion", fontweight="bold", loc="left")
    ax2.set_ylabel("")
    ax2.set_xlabel("Planted Signal Topology")
    ax2.set_xticklabels(col_labels, rotation=28, ha="right")

    plt.tight_layout()
    _save_both(fig, "simulation/fig3_power_heatmap.png")


def build_fig7_timing_benchmark() -> None:
    src = REPO_ROOT / "examples" / "benchmarks" / "res_benchmarks" / "timing_benchmark.csv"
    benchmark = pd.read_csv(src)
    benchmark = benchmark[
        (benchmark["n_permutations"] == 100)
        & (benchmark["n_samples"] == 30)
        & (benchmark["topology"] == "within_module_dense")
    ]
    fig, ax = plt.subplots(figsize=(5.5, 3.25), layout="constrained")
    for method, label, marker, color in [
        ("nbs", "NBS", "s", "#D55E00"),
        ("tfnbs", "TFNBS · exact", "o", "#0072B2"),
        ("tfnbs_n10", "TFNBS · 10-point grid", "^", "#009E73"),
        ("glm_tfnbs", "GLM TFNBS · exact", "D", "#CC79A7"),
        ("glm_tfnbs_n10", "GLM TFNBS · 10-point grid", "v", "#6A4C93"),
    ]:
        sub = benchmark[benchmark["method"] == method].sort_values("n_nodes")
        ax.loglog(sub["n_nodes"], sub["per_perm_ms"], marker=marker, color=color, label=label)
    nodes = sorted(benchmark["n_nodes"].unique())
    ax.set_xticks(nodes, [str(n) for n in nodes])
    ax.tick_params(axis="x", which="minor", labelbottom=False)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Mean wall-clock time ÷ 100 (ms per permutation)")
    ax.grid(axis="y", which="both", ls=":", alpha=0.45)
    ax.legend(loc="upper left")
    _save_both(fig, "figures/fig7_timing_benchmark.png")


def build_figS4_topologies_gallery() -> None:
    from examples.simulations.topology_gallery import plot_topology_gallery
    with TemporaryDirectory(prefix="conninf-topology-") as tmp_dir:
        out = plot_topology_gallery(
            effect_size=0.125, n_samples=30, seed=42, output_dir=Path(tmp_dir), dpi=600,
        )
        for base in OUTPUT_DIRS:
            dest = base / "simulation" / "figS4_topologies_gallery.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out, dest)
            print(f"Wrote {dest}")


def build_figS5_tfnbs_eh_sweep_and_paired_diff() -> None:
    sim_res = REPO_ROOT / "examples" / "simulations" / "results"
    df_arch = pd.read_csv(
        sim_res / "_archive" / "pre_nbs_intensity_fix_2026-11-24"
        / "power_analysis_1000_30" / "power_vs_effect_size_summary.csv"
    )
    df_tf = df_arch[(df_arch["method"] == "tfnbs") & np.isclose(df_arch["effect_size"], 0.125)].copy()
    eh = df_tf["method_params"].map(_parse_eh)
    df_tf["e"] = [x[0] for x in eh]
    df_tf["h"] = [x[1] for x in eh]
    df_tf = df_tf.dropna(subset=["e", "h"])

    tpr_grid = df_tf.groupby(["h", "e"])["TPR_mean"].mean().unstack("e").sort_index(ascending=False)
    fdp_grid = df_tf.groupby(["h", "e"])["FDR_mean"].mean().unstack("e").sort_index(ascending=False)

    df_paired = pd.read_csv(PAPER_DATA / "exact_grid_paired_comparison.csv")

    fig = plt.figure(figsize=(7.2, 5.6), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    ax1, ax2, ax3 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    for ax, values, title, palette, precision in [
        (ax1, tpr_grid, "a   Mean TPR", "YlGnBu", ".2f"),
        (ax2, fdp_grid, "b   Mean FDP", "YlOrRd", ".3f"),
    ]:
        sns.heatmap(values, cmap=palette, ax=ax)
        ax.set_title(title, fontweight="bold", loc="left")
        ax.set_xlabel("Extent exponent E")
        ax.set_ylabel("Height exponent H")
        if 0.4 in values.columns and 3.0 in values.index:
            col = values.columns.get_loc(0.4)
            row = values.index.get_loc(3.0)
            ax.add_patch(plt.Rectangle((col, row), 1, 1, fill=False, edgecolor="#0072B2", lw=1.4))
            ax.annotate(
                f"{values.iloc[row, col]:{precision}}",
                (col + 0.5, row + 0.5), xytext=(13, 8), textcoords="offset points",
                bbox={"facecolor": "white", "edgecolor": "#0072B2", "pad": 1.5},
                arrowprops={"arrowstyle": "-", "color": "#0072B2"},
            )

    # Panel C: Paired exact - grid10 MC 95% CI
    drs = sorted(df_paired["delta_r"].unique())
    x = np.arange(len(drs))
    w = 0.28
    for idx, (metric, col, lbl) in enumerate([("TPR", "#0072B2", r"$\Delta$TPR (exact $-$ grid10)"), ("mean_FDP", "#D55E00", r"$\Delta$FDP (exact $-$ grid10)")]):
        sub = df_paired[df_paired["metric"] == metric].set_index("delta_r").loc[drs]
        vals = sub["delta_exact_minus_grid"].values
        err_lo = vals - sub["mc95_low"].values
        err_hi = sub["mc95_high"].values - vals
        ax3.errorbar(
            x + (idx - 0.5) * w, vals, yerr=[err_lo, err_hi], fmt="o", capsize=4,
            color=col, markersize=5.5, lw=1.4, label=lbl
        )
    ax3.axhline(0.0, color="gray", ls="--", lw=0.9)
    ax3.set_xticks(x)
    ax3.set_xticklabels([rf"$\Delta r = {d}$" for d in drs])
    ax3.set_ylabel("Paired Difference (Exact vs 10-Point Grid)")
    ax3.set_title("c   Paired differences between integration schemes (95% Monte Carlo CI)", fontweight="bold", loc="left")
    ax3.legend(loc="upper left", frameon=True, fontsize=7.0)
    ax3.grid(axis="y", ls=":", alpha=0.4)

    _save_both(fig, "simulation/figS5_tfnbs_eh_sweep_and_paired_diff.png")


# =============================================================================
# ABIDE I CC200 FIGURES (Figs 4, 5, S1, S2)
# =============================================================================

def _load_cc200_data() -> dict:
    d = np.load(REPO_ROOT / "examples" / "abide" / "results" / "cc200" / "abide_cc200_prepared.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def _load_cc200_result(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    return getattr(obj, "inference", obj)


def build_fig4_abide_cc200_workflow() -> None:
    data = _load_cc200_data()
    net_labels = data["net_labels"]
    network_order = list(data["network_order"])
    n_nets = len(network_order)
    cc200_dir = REPO_ROOT / "examples" / "abide" / "results" / "cc200"

    def _agg_edges(csv_path: Path) -> tuple[np.ndarray, int, int]:
        df = pd.read_csv(csv_path)
        mat = np.zeros((n_nets, n_nets), dtype=int)
        n_pos = int((df["tail"] == "positive").sum()) if "tail" in df.columns else 0
        n_neg = int((df["tail"] == "negative").sum()) if "tail" in df.columns else 0
        for _, r in df.iterrows():
            ni = int(net_labels[int(r["roi_i"])])
            nj = int(net_labels[int(r["roi_j"])])
            mat[ni, nj] += 1
            if ni != nj:
                mat[nj, ni] += 1
        return mat, n_pos, n_neg

    # Short network labels for readability
    short_nets = [
        s.replace("Somatomotor", "SMN")
        .replace("Dorsal Attention", "DAN")
        .replace("Ventral Attention", "VAN")
        .replace("DorsalAttn", "DAN")
        .replace("VentralAttn", "VAN")
        .replace("Frontoparietal", "FPN")
        .replace("Subcortical", "Subcort")
        .replace("Cerebellum", "CBM")
        for s in network_order
    ]

    strategies = [
        ("Naive (no age, sex, FD or site)", "dx_naive"),
        ("ComBat only (no site indicators)", "dx_combat_only"),
        ("Site GLM (Strategy E)", "dx_strategy_e"),
        ("ComBat + site GLM (Strategy D)", "dx_strategy_d"),
    ]
    counts = [_agg_edges(cc200_dir / arm / f"{arm}_edges.csv") for _, arm in strategies]
    vmax_a = max(int(mat.max()) for mat, _, _ in counts)

    fig = plt.figure(figsize=(7.4, 9.2), layout="constrained")
    grid = fig.add_gridspec(5, 3, height_ratios=[0.08, 1, 1, 0.08, 1], width_ratios=[1, 1, 0.07])
    for row, title in [(0, "a   ASD–HC site-handling strategies"), (3, "b   Age contrast (Strategy D)")]:
        heading = fig.add_subplot(grid[row, :])
        heading.set_axis_off()
        heading.text(0, 0.5, title, weight="bold", va="center")
    mask = np.triu(np.ones((n_nets, n_nets), dtype=bool), k=1)

    def draw(ax, mat, title, vmax, cmap):
        sns.heatmap(
            mat, mask=mask, annot=True, fmt="d", cmap=cmap, vmin=0, vmax=vmax,
            xticklabels=short_nets, yticklabels=short_nets, cbar=False,
            ax=ax, annot_kws={"size": 7.1},
        )
        ax.set_title(title, fontsize=9)
        ax.set_xticklabels(short_nets, rotation=45, ha="right")
        ax.set_yticklabels(short_nets, rotation=0)
        return ax.collections[0]

    for idx, ((title, _), (mat, n_pos, n_neg)) in enumerate(zip(strategies, counts)):
        ax = fig.add_subplot(grid[1 + idx // 2, idx % 2])
        scale_a = draw(ax, mat, f"{title}\n+{n_pos} / −{n_neg} edges", vmax_a, "Blues")
    fig.colorbar(scale_a, cax=fig.add_subplot(grid[1:3, 2]), label="Detected edges (both tails)")

    # Panel B: Age Strategy D 11-block positive vs negative profile
    age_df = pd.read_csv(cc200_dir / "age" / "age_strategy_d_edges.csv")
    mat_pos = np.zeros((n_nets, n_nets), dtype=int)
    mat_neg = np.zeros((n_nets, n_nets), dtype=int)
    for _, r in age_df.iterrows():
        ni = int(net_labels[int(r["roi_i"])])
        nj = int(net_labels[int(r["roi_j"])])
        target = mat_pos if r["tail"] == "positive" else mat_neg
        target[ni, nj] += 1
        if ni != nj:
            target[nj, ni] += 1

    vmax_b = max(int(mat_pos.max()), int(mat_neg.max()))
    for column, (mat_b, tail, sign) in enumerate([
        (mat_pos, "positive", "+"),
        (mat_neg, "negative", "−"),
    ]):
        ax = fig.add_subplot(grid[4, column])
        n = int((age_df["tail"] == tail).sum())
        scale_b = draw(ax, mat_b, f"{tail.title()} tail ({sign}{n} edges)", vmax_b, "Purples")
    fig.colorbar(scale_b, cax=fig.add_subplot(grid[4, 2]), label="Detected edges per tail")
    _save_both(fig, "abide/fig4_abide_cc200_workflow.png")


def _permute_inference_result(res, order: np.ndarray):
    """Return a shallow copy of InferenceResult with N x N matrices reordered by `order`."""
    import copy
    out = copy.copy(res)
    n = len(order)
    idx = np.asarray(order, dtype=int)
    for key in list(out.keys()):
        val = out[key]
        if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape == (n, n):
            out[key] = val[np.ix_(idx, idx)]
    for attr in ("stat_positive", "stat_negative"):
        val = getattr(res, attr, None)
        if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape == (n, n):
            setattr(out, attr, val[np.ix_(idx, idx)])
    beta = getattr(res, "beta", None)
    if isinstance(beta, np.ndarray):
        if beta.ndim == 3 and beta.shape[1:] == (n, n):
            out.beta = beta[:, idx, :][:, :, idx]
        elif beta.ndim == 2 and beta.shape == (n, n):
            out.beta = beta[np.ix_(idx, idx)]
    return out


def build_fig5_abide_cc200_bna_suit_showcase() -> None:
    data = _load_cc200_data()
    roi_names = list(data["roi_names"])
    net_labels = data["net_labels"]
    network_order = list(data["network_order"])
    roi_networks = [network_order[int(i)] for i in net_labels]

    cc200_dir = REPO_ROOT / "examples" / "abide" / "results" / "cc200"
    dx_res = _load_cc200_result(cc200_dir / "dx_strategy_d" / "dx_strategy_d.result.pkl")
    # Ensure original CC200 ROI names are bound
    dx_res.atlas = AtlasInfo(
        labels=roi_names,
        networks=roi_networks,
        coords=data["roi_centroids"],
        source="CC200",
    )

    ref_suit = BrainnetomeReference.load(cerebellum="anatom")
    cc200_img = "/Users/irina/nilearn_data/ABIDE_pcp/cc200_roi_atlas.nii.gz"
    rois_suit = annotate_rois(
        reference=ref_suit,
        names=roi_names,
        space="MNI152NLin6Asym",
        support="parcel",
        parcel_image=cc200_img,
        parcel_labels=list(range(1, 201)),
        networks=roi_networks,
    )

    with open(cc200_dir / "annotation" / "cerebellum_coverage_diff.json") as f:
        cov = json.load(f)

    fig = plt.figure(figsize=(7.3, 9.7), layout="constrained")
    gs = fig.add_gridspec(4, 2, height_ratios=[1.6, 0.76, 1.0, 0.25])

    chord_grid = gs[0, :].subgridspec(2, 1, height_ratios=[4.1, 0.7])
    axA = fig.add_subplot(chord_grid[0])
    plot_connectogram(
        dx_res, rois_suit, tail="positive", level="gyrus",
        max_links=80, labels="none", ax=axA,
        title="a   ASD > HC connectogram (Strategy D)"
    )
    _place_connectogram_legend(axA, fig.add_subplot(chord_grid[1]), ncol=5)
    for label in axA.texts:
        x_pos, y_pos = label.get_position()
        if label.get_text() == "CBM-L":
            label.set_x(x_pos - 0.10)
        elif label.get_text() == "CBM-R":
            label.set_x(x_pos + 0.10)
        elif label.get_text() == "CBM-M":
            label.set_y(y_pos - 0.11)

    # Panel B: Descriptive BNA-only vs BNA+SUIT coverage bar chart
    axC = fig.add_subplot(gs[1, :])
    cats = [
        "Unassigned ROIs\n(out of 200)",
        "Age sig. edges\nwith unassigned end\n(out of 881)",
        "ASD–HC sig. edges\nwith unassigned end\n(out of 1,144)",
    ]
    bna_only_vals = [
        cov["roi_level"]["bna_only"]["n_unassigned"],
        cov["edge_level"]["age_strategy_d"]["either"]["bna_only"]["n_edges_with_unassigned_endpoint"],
        cov["edge_level"]["dx_strategy_d_asd_vs_hc"]["either"]["bna_only"]["n_edges_with_unassigned_endpoint"],
    ]
    bna_suit_vals = [
        cov["roi_level"]["bna_suit"]["n_unassigned"],
        cov["edge_level"]["age_strategy_d"]["either"]["bna_suit"]["n_edges_with_unassigned_endpoint"],
        cov["edge_level"]["dx_strategy_d_asd_vs_hc"]["either"]["bna_suit"]["n_edges_with_unassigned_endpoint"],
    ]
    x = np.arange(len(cats))
    w = 0.34
    b1 = axC.bar(x - w / 2, bna_only_vals, w, label="BNA-246 only", color="#999999", edgecolor="black", lw=0.6)
    b2 = axC.bar(x + w / 2, bna_suit_vals, w, label="BNA-246 + SUIT (cerebellum='anatom')", color="#009E73", edgecolor="black", lw=0.6)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            axC.text(bar.get_x() + bar.get_width() / 2, h + 3, f"{int(h)}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    axC.set_xticks(x)
    axC.set_xticklabels(cats, fontsize=7.5)
    axC.set_ylabel("Count")
    axC.set_ylim(0, max(bna_only_vals) * 1.18)
    axC.set_title("b   Atlas coverage", fontweight="bold", loc="left")
    axC.legend(loc="upper left", frameon=True)
    axC.grid(axis="y", ls=":", alpha=0.4)

    # Panel C: Coupling-class matrix on ASD-HC Strategy D, reordered by 11 network blocks
    axD = fig.add_subplot(gs[2, 0])
    net_order = np.argsort(net_labels, kind="stable")
    dx_res_ord = _permute_inference_result(dx_res, net_order)
    plot_coupling_classes(
        dx_res_ord,
        ax=axD,
        title="c   Coupling classes (HC → ASD)",
    )
    coupling_legend = axD.get_legend()
    legend_ax = fig.add_subplot(gs[3, :])
    legend_ax.axis("off")
    legend_items = [
        (handle, txt.get_text())
        for handle, txt in zip(coupling_legend.legend_handles, coupling_legend.get_texts())
        if not txt.get_text().startswith("homotopy")
    ]
    handles, labels = zip(*legend_items)
    legend_ax.legend(handles, labels, loc="center", ncol=3, frameon=False)
    coupling_legend.remove()

    ax_mask = fig.add_subplot(gs[2, 1])
    detected = dx_res_ord.significant_edges(alpha=0.05, tail="both")
    significant = np.zeros((len(net_order), len(net_order)), dtype=bool)
    i = detected["roi_i"].to_numpy(dtype=int)
    j = detected["roi_j"].to_numpy(dtype=int)
    significant[i, j] = significant[j, i] = True
    ax_mask.imshow(significant, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax_mask.set_title("Significant edges (α = 0.05)")
    sorted_nets = net_labels[net_order]
    short_nets = [
        s.replace("Somatomotor", "SMN")
        .replace("Dorsal Attention", "DAN")
        .replace("Ventral Attention", "VAN")
        .replace("Frontoparietal", "FPN")
        .replace("Subcortical", "Subcort")
        .replace("Cerebellum", "CBM")
        for s in network_order
    ]
    boundaries = np.where(sorted_nets[:-1] != sorted_nets[1:])[0] + 0.5
    centers, tick_lbls = [], []
    for k, lbl in enumerate(short_nets):
        idx_k = np.where(sorted_nets == k)[0]
        if len(idx_k):
            centers.append(float(idx_k.mean()))
            tick_lbls.append(lbl)
    for ax, line_color in [(axD, "white"), (ax_mask, "#aab4bf")]:
        for b in boundaries:
            ax.axhline(b, color=line_color, lw=0.65)
            ax.axvline(b, color=line_color, lw=0.65)
        ax.set_xticks(centers, tick_lbls, rotation=45, ha="right")
        ax.set_yticks(centers, tick_lbls)
        ax.set_xlabel("CC200 ROIs by network")
        ax.set_ylabel("")
    _save_both(fig, "abide/fig5_abide_cc200_bna_suit_showcase.png")


def build_figS1_abide_cc200_combat_qc() -> None:
    data = _load_cc200_data()
    Y_raw = data["connectivity_z"]
    sites = data["site"]
    confounds = np.column_stack([data["age"].astype(float), data["sex"].astype(float), data["mean_fd"].astype(float)])

    # Pass raw NaN-preserving Y into ComBat with missing="available_case"
    combat_out = combat_harmonize(Y_raw, sites, preserve=confounds, missing="available_case")
    diag = combat_out.diagnostics
    r_before = float(diag["between_site_variance_ratio_before"])
    r_after = float(diag["between_site_variance_ratio_after"])
    r_ratio = float(diag["between_site_variance_ratio_after_over_before"])

    iu = np.triu_indices(Y_raw.shape[1], k=1)
    Y_raw_flat = Y_raw[:, iu[0], iu[1]]
    Y_harm_flat = combat_out.Y_adjusted[:, iu[0], iu[1]]
    # Exact same valid-observation mask before and after ComBat (no imputation)
    valid_mask = np.isfinite(Y_raw_flat) & np.isfinite(Y_harm_flat)

    def _resid_masked(Y_flat, X, mask):
        X_full = np.column_stack([np.ones(X.shape[0]), X])
        resid = np.full_like(Y_flat, np.nan, dtype=float)
        for e in range(Y_flat.shape[1]):
            m = mask[:, e]
            if m.sum() > X_full.shape[1]:
                Xe = X_full[m]
                ye = Y_flat[m, e]
                beta, *_ = np.linalg.lstsq(Xe, ye, rcond=None)
                resid[m, e] = ye - Xe @ beta
        return resid

    resid_raw = _resid_masked(Y_raw_flat, confounds, valid_mask)
    resid_harm = _resid_masked(Y_harm_flat, confounds, valid_mask)

    with open(REPO_ROOT / "examples" / "abide" / "results" / "cc200" / "motion_qc_summary.json") as f:
        mqc = json.load(f)

    fig = plt.figure(figsize=(7.3, 5.8), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0])
    ax1 = fig.add_subplot(grid[0])
    qc = fig.add_subplot(grid[1])
    qc.axis("off")

    # Panel A: Site-level mean residual FC before vs after ComBat (over identical valid-edge mask)
    df_box = pd.DataFrame({
        "Site": np.concatenate([sites, sites]),
        "Mean Residual FC": np.concatenate([np.nanmean(resid_raw, axis=1), np.nanmean(resid_harm, axis=1)]),
        "Status": ["Before ComBat"] * len(sites) + ["After ComBat (nuisance_only)"] * len(sites),
    })
    sns.boxplot(data=df_box, x="Site", y="Mean Residual FC", hue="Status", palette=["#E69F00", "#009E73"], ax=ax1, fliersize=2)
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    ax1.tick_params(axis="x", labelrotation=38, labelsize=6.5)
    ax1.set_title("a   Mean residual connectivity by site", fontweight="bold", loc="left")
    ax1.set_xlabel(
        f"Between-site / total feature variance (ComBat diagnostics): "
        f"{r_before:.4f} → {r_after:.4f} (×{r_ratio:.3f})"
    )
    ax1.legend(loc="upper right", frameon=True, fontsize=7.0)

    # Panel B: Pre-inference Motion QC summary metrics (from motion_qc_summary.json)
    all_s = mqc["all_saved_subjects"]
    metrics = [
        ("Mean |QC–FC r|", all_s["abs_qc_fc_mean"]),
        ("Share |QC–FC r| > 0.2", all_s["share_abs_r_gt"]["0.2"]),
        ("Share BH q < 0.05", all_s["share_bh_q_lt_0.05"]),
        ("Short-range |r| (Q1)", all_s["distance_dependence"]["mean_abs_r_shortest_quartile"]),
        ("Long-range |r| (Q4)", all_s["distance_dependence"]["mean_abs_r_longest_quartile"]),
        ("Mean FD ~ Mean FC (ρ)", all_s["subject_level"]["spearman_mean_fd_vs_mean_fc_rho"]),
    ]
    qc.set_title("b   Motion diagnostics before inference", fontweight="bold", loc="left")
    qc.text(
        0, 0.91,
        f"Mean framewise displacement: ASD {all_s['mean_fd_asd_mm']:.3f} mm, "
        f"HC {all_s['mean_fd_hc_mm']:.3f} mm (p = {all_s['fd_asd_vs_hc_p']:.2g})",
        va="top",
    )
    for x_pos, heading, ids in [
        (0.00, "Edge correlations", (0, 3, 4)),
        (0.35, "Fractions of edges", (1, 2)),
        (0.70, "Across participants", (5,)),
    ]:
        qc.text(x_pos, 0.67, heading, weight="bold", va="top")
        for row, idx in enumerate(ids):
            label, value = metrics[idx]
            display = f"{value:.1%}" if idx in (1, 2) else f"{value:.3f}"
            qc.text(x_pos, 0.49 - row * 0.17, f"{label}   {display}", va="top")
    _save_both(fig, "abide/figS1_abide_cc200_combat_qc.png")


def build_figS2_abide_cc200_eh_multiverse() -> None:
    eh_dir = REPO_ROOT / "examples" / "abide" / "results" / "cc200" / "eh_sensitivity"
    sp = pd.read_csv(eh_dir / "spearman_matrix.csv", index_col=0)
    jc = pd.read_csv(eh_dir / "jaccard_matrix.csv", index_col=0)
    gs = pd.read_csv(eh_dir / "grid_summary.csv")

    fig = plt.figure(figsize=(7.3, 7.9), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0])
    heatmaps = grid[0].subgridspec(1, 2)
    settings = gs[["E", "H"]].drop_duplicates().sort_values(["E", "H"])
    cell_names = [f"E{e:.2f}_H{h:.2f}" for e, h in settings.itertuples(index=False, name=None)]
    groups = settings.groupby("E").size()
    boundaries = np.r_[0, groups.cumsum()]
    centers = (boundaries[1:] + boundaries[:-1]) / 2
    group_labels = [f"{e:g}" for e in groups.index]
    for j, (matrix, title, cmap, low, metric) in enumerate([
        (sp, "a   Positive-tail rank stability", "viridis", min(0.5, float(sp.min().min())), "Spearman ρ"),
        (jc, "b   Detected-edge overlap", "magma", 0.0, "Jaccard"),
    ]):
        ax = fig.add_subplot(heatmaps[0, j])
        sns.heatmap(
            matrix.loc[cell_names, cell_names], cmap=cmap, vmin=low, vmax=1,
            square=True, ax=ax, cbar_kws={"label": metric, "shrink": 0.80},
        )
        for line in boundaries[1:-1]:
            ax.axhline(line, color="white", lw=0.65)
            ax.axvline(line, color="white", lw=0.65)
        ax.set_xticks(centers, group_labels, rotation=0)
        ax.set_yticks(centers, group_labels, rotation=0)
        ax.set(xlabel="Extent exponent E", ylabel="Extent exponent E")
        ax.set_title(title, fontweight="bold", loc="left")

    gs["total_sig"] = gs["n_sig_pos"] + gs["n_sig_neg"]
    counts = grid[1].subgridspec(4, 3, height_ratios=[0.19, 1, 1, 0.22])
    title_ax = fig.add_subplot(counts[0, :])
    title_ax.axis("off")
    title_ax.text(0, 0.5, "c   Detected edges by E and H (both tails, α = 0.05)", weight="bold", va="center")
    first_ax = None
    h_values = sorted(settings["H"].unique())
    max_count = gs["total_sig"].max()
    for idx, (e, by_e) in enumerate(gs.groupby("E")):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(counts[1 + row, col], sharex=first_ax, sharey=first_ax)
        if first_ax is None:
            first_ax = ax
        for method, color, marker, label in [
            ("tfnbs", "#0072B2", "o", "TFNBS"),
            ("ni_tfnbs", "#009E73", "^", "NI-TFNBS"),
            ("fbc_tfnbs", "#D55E00", "s", "FBC-TFNBS"),
        ]:
            sub = by_e[by_e["method"] == method].sort_values("H")
            ax.plot(sub["H"], sub["total_sig"], marker=marker, ms=3.2, color=color, label=label)
        ax.set_xticks(h_values)
        ax.set_title(f"E = {e:g}")
        ax.grid(axis="y", ls=":", alpha=0.45)
        ax.tick_params(labelbottom=row == 1, labelleft=col == 0)
        ax.set_ylim(0, max_count * 1.05)
        if row == 1:
            ax.set_xlabel("Height exponent H")
        if col == 0:
            ax.set_ylabel("Detected edges")
    legend_ax = fig.add_subplot(counts[3, :])
    legend_ax.axis("off")
    handles, labels = first_ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", ncol=3, frameon=False)
    _save_both(fig, "abide/figS2_abide_cc200_eh_multiverse.png")


# =============================================================================
# OPEN-CLOSE EO/EC FIGURES (Figs 6, S3)
# =============================================================================

def build_fig6_openclose_bna242_showcase() -> None:
    oc_dir = REPO_ROOT / "examples" / "open_close" / "results" / "paper_validation" / "v2_0_0"
    ihb = np.load(oc_dir / "ihb_paired_tfnbs.npz")
    china = np.load(oc_dir / "china_paired_tfnbs_run0.npz")
    ds = OpenCloseDataset.load("ihb")
    lab = ds.net_labels
    names = ds.get_label_names()
    K = len(names)

    agr = pd.read_csv(PAPER_DATA / "openclose_agreement_summary.csv")
    agr_v2 = agr[(agr["source"] == "v2") & (agr["comparison"] == "IHB vs China run0")].set_index("tail")

    # Native BNA-242 result + annotation
    bna_dir = oc_dir / "bna242"
    with open(bna_dir / "ihb_paired_tfnbs.result.pkl", "rb") as f:
        bna_res_obj = pickle.load(f)
    bna_res = getattr(bna_res_obj, "inference", bna_res_obj)
    with open(bna_dir / "ihb_paired_tfnbs_bna_atlas.json") as f:
        bna_atlas_meta = json.load(f)
    ref_bna = BrainnetomeReference.load()
    bna_atlas = AtlasInfo(
        labels=bna_atlas_meta.get("labels", [str(x) for x in bna_atlas_meta["label_ids"]]),
        networks=bna_atlas_meta.get("networks", ["BNA"] * len(bna_atlas_meta["label_ids"])),
        source="BNA-242",
        label_ids=np.asarray(bna_atlas_meta["label_ids"], dtype=np.int64),
    )
    bna_res.atlas = bna_atlas
    rois_bna = annotate_bna_regions(bna_atlas_meta["label_ids"], reference=ref_bna)

    fig = plt.figure(figsize=(8.1, 10.1), layout="constrained")
    gs = fig.add_gridspec(5, 2, height_ratios=[1.1, 0.26, 1.5, 0.25, 0.46])

    # Panel A: Schaefer-200 block sums (IHB vs Beijing positive tail)
    ma = _block_mass(ihb["positive"], lab)
    mb = _block_mass(china["positive"], lab)
    vmax = max(float(ma.max()), float(mb.max()))
    short_n = [
        n.replace("Default Mode", "DMN")
        .replace("Default", "DMN")
        .replace("Dorsal Attention", "DAN")
        .replace("DorsalAttn", "DAN")
        .replace("Ventral Attention", "VAN")
        .replace("VentralAttn", "VAN")
        .replace("Frontoparietal Control", "FPN")
        .replace("Frontoparietal", "FPN")
        .replace("Somatomotor", "SMN")
        for n in names
    ]
    block_axes = []
    for c, (M, ttl) in enumerate([
        (ma, f"a   IHB EC > EO ({int(agr_v2.loc['positive', 'n_sig_a']):,} edges)"),
        (mb, f"Beijing EC > EO ({int(agr_v2.loc['positive', 'n_sig_b']):,} edges)"),
    ]):
        ax = fig.add_subplot(gs[0, c])
        block_axes.append(ax)
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xticks(range(K), short_n, rotation=45, ha="right")
        ax.set_yticks(range(K), short_n)
        ax.set_title(ttl, fontweight="bold")
    fig.colorbar(im, ax=block_axes, shrink=0.86, label=r"Yeo-7 block sum of $-\log_{10}p$")
    agreement_ax = fig.add_subplot(gs[1, :])
    agreement_ax.axis("off")
    r_pos, r_neg = agr_v2.loc[["positive", "negative"], "pearson_of_block_sum_neglog10p"]
    j_pos, j_neg = agr_v2.loc[["positive", "negative"], "jaccard"]
    agreement_ax.text(
        0.5, 0.68, "Schaefer-200  ·  Cross-cohort agreement by tail",
        weight="bold", va="center", ha="center",
    )
    agreement_ax.text(
        0.5, 0.22,
        f"Yeo-7 block Pearson r: {r_pos:.3f} (positive), {r_neg:.3f} (negative)     "
        f"Edge Jaccard: {j_pos:.3f} (positive), {j_neg:.3f} (negative)",
        va="center", ha="center",
    )

    # Panel B: Native BNA-242 Connectogram (IHB EC > EO)
    chord_grid = gs[2, 0].subgridspec(2, 1, height_ratios=[3, 0.85])
    axB = fig.add_subplot(chord_grid[0])
    n_bna_positive = int((bna_res.positive[np.triu_indices_from(bna_res.positive, k=1)] < 0.05).sum())
    plot_connectogram(
        bna_res, rois_bna, tail="positive", level="gyrus",
        max_links=90, labels="none", ax=axB,
        title=f"b   Native BNA-242 connectogram\nIHB EC > EO ({n_bna_positive:,} edges)"
    )
    _place_connectogram_legend(axB, fig.add_subplot(chord_grid[1]), ncol=3)

    # Panel C + 1D zoomed homotopic strip: Native BNA-242 Coupling Classes
    # Reorder nodes by lobe order [Frontal -> Insular -> Limbic -> Temporal -> Parietal -> Occipital -> Subcortical]
    # as [all Left (121: 120 paired + ID 117), all Right (121: 120 matched + ID 110)]
    from conninfpy.plot.coupling import COUPLING_CLASS_COLORS
    from matplotlib.colors import to_rgb

    lobe_order = ["Frontal", "Insular", "Limbic", "Temporal", "Parietal", "Occipital", "Subcortical"]
    lobe_short = {
        "Frontal": "FRO", "Insular": "INS", "Limbic": "LIM",
        "Temporal": "TEM", "Parietal": "PAR", "Occipital": "OCC", "Subcortical": "SUB",
    }
    lobe_rank = {lb: r for r, lb in enumerate(lobe_order)}

    lids = np.asarray(bna_atlas_meta["label_ids"], dtype=int)
    lid_to_idx = {int(rid): idx for idx, rid in enumerate(lids)}
    paired_entries = []
    unpaired_left, unpaired_right = [], []
    used = set()
    for idx, rid in enumerate(lids):
        if idx in used or int(rid) not in ref_bna.regions.index:
            continue
        reg = ref_bna.regions.loc[int(rid)]
        hemi = str(reg["hemisphere"]).upper()
        hid = reg["homolog_id"]
        if hemi in ("L", "LEFT") and pd.notna(hid) and int(hid) in lid_to_idx:
            jdx = lid_to_idx[int(hid)]
            rk = lobe_rank.get(str(reg["lobe"]), 99)
            paired_entries.append((rk, int(rid), idx, jdx, str(reg["lobe"])))
            used.add(idx)
            used.add(jdx)

    paired_entries.sort(key=lambda t: (t[0], t[1]))
    left_indices = [t[2] for t in paired_entries]
    right_indices = [t[3] for t in paired_entries]
    pair_lobes = [t[4] for t in paired_entries]

    for idx, rid in enumerate(lids):
        if idx not in used:
            reg = ref_bna.regions.loc[int(rid)]
            hemi = str(reg["hemisphere"]).upper()
            if hemi in ("L", "LEFT"):
                unpaired_left.append(idx)
            else:
                unpaired_right.append(idx)
            used.add(idx)

    left_indices.extend(unpaired_left)
    right_indices.extend(unpaired_right)

    assert len(left_indices) == 121, f"Expected 121 Left ROIs, got {len(left_indices)}"
    assert len(right_indices) == 121, f"Expected 121 Right ROIs, got {len(right_indices)}"
    bna_order = np.array(left_indices + right_indices, dtype=int)
    assert len(set(bna_order.tolist())) == 242, "Duplicate ROI index in BNA-242 order"

    bna_res_ord = _permute_inference_result(bna_res, bna_order)
    bna_atlas_ord = AtlasInfo(
        labels=[bna_atlas.labels[i] for i in bna_order],
        networks=[bna_atlas.networks[i] for i in bna_order],
        source="BNA-242",
        label_ids=lids[bna_order],
    )

    axC = fig.add_subplot(gs[2, 1])
    plot_coupling_classes(
        bna_res_ord,
        atlas=bna_atlas_ord,
        reference=ref_bna,
        ax=axC,
        title="c   Native BNA-242 coupling classes (EO → EC)",
    )
    homotopy_pairs = axC.attrs["homotopy_pairs"]
    assert len(homotopy_pairs) == 120, f"Expected 120 homotopy pairs, got {len(homotopy_pairs)}"
    expected_pairs = {(k, k + 121) for k in range(120)}
    actual_pairs = {(min(i, j), max(i, j)) for (i, j) in homotopy_pairs}
    assert actual_pairs == expected_pairs, "Homotopy pairs do not land strictly at (k, k+121)"

    # Remove the 240 dense black circles so the blue homotopic band stays clearly visible
    for coll in list(axC.collections):
        coll.remove()

    # Subtle dashed flanking guide lines along the two homotopic off-diagonals
    k_arr = np.arange(120, dtype=float)
    for offset in (-2.0, 2.0):
        axC.plot(k_arr + 121, k_arr + offset, ls="--", lw=0.6, color="white", alpha=0.85, zorder=3)
        axC.plot(k_arr, k_arr + 121 + offset, ls="--", lw=0.6, color="white", alpha=0.85, zorder=3)

    # Mark ONLY the statistically significant (alpha = 0.05) homotopic pairs
    sig_pos_k = [k for k in range(120) if bna_res_ord.positive[k, k + 121] < 0.05]
    sig_neg_k = [k for k in range(120) if bna_res_ord.negative[k, k + 121] < 0.05]
    sig_all_k = sig_pos_k + sig_neg_k
    xs_sig = [k + 121 for k in sig_all_k] + sig_all_k
    ys_sig = sig_all_k + [k + 121 for k in sig_all_k]
    axC.scatter(xs_sig, ys_sig, s=34, facecolors="none", edgecolors="white", linewidths=1.6, zorder=4)
    axC.scatter(xs_sig, ys_sig, s=34, facecolors="none", edgecolors="black", linewidths=0.95, zorder=5)

    leg_c = axC.get_legend()
    if leg_c is not None:
        legend_ax = fig.add_subplot(gs[3, :])
        legend_ax.axis("off")
        labels = [
            "Significant homolog (α = 0.05)" if "homotopy pair" in txt.get_text() else txt.get_text()
            for txt in leg_c.get_texts()
        ]
        legend_ax.legend(leg_c.legend_handles, labels, ncol=4, loc="center", frameon=False)
        leg_c.remove()

    n_left = len(left_indices)
    axC.axhline(n_left - 0.5, color="white", lw=0.9, alpha=0.9)
    axC.axvline(n_left - 0.5, color="white", lw=0.9, alpha=0.9)
    axC.set_xticks([n_left / 2, n_left + len(right_indices) / 2])
    axC.set_xticklabels([f"Left (n={n_left})", f"Right (n={len(right_indices)})"], rotation=0, fontsize=7.5)
    axC.set_yticks([n_left / 2, n_left + len(right_indices) / 2])
    axC.set_yticklabels([f"Left (n={n_left})", f"Right (n={len(right_indices)})"], fontsize=7.5)
    axC.set_xlabel("BNA-242 regions (dashed: homotopic diagonals)")
    axC.set_ylabel("BNA-242 regions")

    # 1D zoomed homotopic strip right below Panel C
    axStrip = fig.add_subplot(gs[4, :])
    classes = np.asarray(bna_res_ord.coupling()["classes"], dtype=object)
    hom_classes = [str(classes[k, k + 121]) for k in range(120)]
    n_strong = sum(c == "stronger positive coupling" for c in hom_classes)
    n_weak = sum(c == "weaker positive coupling" for c in hom_classes)
    strip_rgb = np.array([to_rgb(COUPLING_CLASS_COLORS[c]) for c in hom_classes])[None, :, :]
    axStrip.imshow(strip_rgb, aspect="auto", interpolation="nearest", extent=[-0.5, 119.5, 0.5, -0.5])

    # Lobe dividers and labels
    lobe_centers, lobe_lbls = [], []
    for lb in lobe_order:
        idxs = [k for k, pl in enumerate(pair_lobes) if pl == lb]
        if idxs:
            lobe_centers.append(float(np.mean(idxs)))
            lobe_lbls.append(lobe_short[lb])
            if idxs[0] > 0:
                axStrip.axvline(idxs[0] - 0.5, color="white", lw=1.1, zorder=3)

    # Mark significant homotopic pairs above strip
    if sig_pos_k:
        axStrip.plot(sig_pos_k, [-0.85] * len(sig_pos_k), linestyle="none", marker="^", ms=6.2,
                     color="#08306b", mec="white", mew=0.5, clip_on=False, label=f"EC > EO ({len(sig_pos_k)})")
    if sig_neg_k:
        axStrip.plot(sig_neg_k, [-0.85] * len(sig_neg_k), linestyle="none", marker="v", ms=6.2,
                     color="#6baed6", mec="black", mew=0.5, clip_on=False, label=f"EO > EC ({len(sig_neg_k)})")

    axStrip.set_ylim(0.5, -1.35)
    axStrip.set_xlim(-0.5, 119.5)
    axStrip.set_yticks([])
    axStrip.set_xticks(lobe_centers)
    axStrip.set_xticklabels(lobe_lbls, fontsize=7.2, fontweight="bold")
    axStrip.set_title(
        f"120 homotopic pairs positive in both states: {n_strong} stronger, {n_weak} weaker in EC\n"
        f"Significant pairs: {len(sig_pos_k)} EC > EO (▲), {len(sig_neg_k)} EO > EC (▼)",
        fontsize=8, loc="left",
    )
    _save_both(fig, "openclose/fig6_openclose_bna242_showcase.png")


def build_figS3_openclose_retest_agreement() -> None:
    maps_dir = REPO_ROOT / "examples" / "open_close" / "results" / "paper_validation" / "v2_0_0" / "methods"
    methods = ["tstat", "tfnbs", "ni_tfnbs", "fbc_tfnbs", "nbs_20", "nbs_30", "bh_fdr"]
    pretty = {
        "tstat": "Max-t",
        "tfnbs": "TFNBS (exact)",
        "ni_tfnbs": "NI-TFNBS",
        "fbc_tfnbs": "FBC-TFNBS",
        "nbs_20": r"NBS extent $\tau=2$",
        "nbs_30": r"NBS extent $\tau=3$",
        "bh_fdr": "BH-FDR",
    }

    def _nsig(path: Path) -> int:
        d = np.load(path)
        arrs = [d[k] for k in d.files]
        p_min = arrs[0].copy()
        for a in arrs[1:]:
            p_min = np.minimum(p_min, a)
        iu = np.triu_indices(p_min.shape[0], k=1)
        return int((p_min[iu] < 0.05).sum())

    state_n = [_nsig(maps_dir / f"pmap_china_{m}.npz") for m in methods]
    retest_n = [_nsig(maps_dir / f"pmap_china_retest_{m}.npz") for m in methods]

    agr = pd.read_csv(PAPER_DATA / "openclose_agreement_summary.csv")
    ret = agr[(agr["source"] == "v2") & (agr["comparison"] == "China run0 paired vs China retest")].set_index("tail")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.7), layout="constrained")
    x = np.arange(len(methods))
    w = 0.36
    ax1.bar(x - w / 2, state_n, w, label="State (EC₀ − EO)", color="#009E73", edgecolor="black", lw=0.6)
    ax1.bar(x + w / 2, retest_n, w, label="Repeat run (EC₁ − EC₀)", color="#D55E00", edgecolor="black", lw=0.6, hatch="//")
    ax1.set_xticks(x)
    ax1.set_xticklabels([pretty[m] for m in methods], rotation=28, ha="right")
    ax1.set_ylabel("Detected edges (either tail, α = 0.05)")
    ax1.set_title("a   Beijing state and repeat-run contrasts share EC₀", fontweight="bold", loc="left")
    ax1.legend(loc="upper left", frameon=True, fontsize=7.5)
    ax1.grid(axis="y", ls=":", alpha=0.4)
    ax1.set_xlabel("NBS counts the member edges of detected components")

    # Panel B: Spatial agreement between contrasts
    tails = ["Positive tail", "Negative tail"]
    r_vals = [float(ret.loc["positive", "pearson_of_block_sum_neglog10p"]), float(ret.loc["negative", "pearson_of_block_sum_neglog10p"])]
    j_vals = [float(ret.loc["positive", "jaccard"]), float(ret.loc["negative", "jaccard"])]
    sp_vals = [float(ret.loc["positive", "spearman_neglog10p"]), float(ret.loc["negative", "spearman_neglog10p"])]

    xb = np.arange(len(tails))
    wb = 0.24
    ax2.bar(xb - wb, r_vals, wb, label=r"Block-mass Pearson $r$", color="#0072B2", edgecolor="black", lw=0.6)
    ax2.bar(xb, sp_vals, wb, label=r"Edge Spearman $\rho$ of $-\log_{10} p$", color="#CC79A7", edgecolor="black", lw=0.6, hatch="//")
    ax2.bar(xb + wb, j_vals, wb, label="Edge Jaccard", color="#E69F00", edgecolor="black", lw=0.6, hatch="..")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(tails)
    ax2.set_ylim(-0.22, 0.15)
    ax2.set_ylabel("Agreement")
    ax2.set_title("b   Spatial agreement between state and repeat-run contrasts (TFNBS)", fontweight="bold", loc="left")
    ax2.legend(loc="lower center", ncol=3, frameon=True, fontsize=7.2)
    ax2.grid(axis="y", ls=":", alpha=0.4)

    _save_both(fig, "openclose/figS3_openclose_retest_orthogonality.png")


def main() -> None:
    global OUTPUT_DIRS
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["all", "sim", "abide", "openclose"], default="all")
    parser.add_argument("--repo-only", action="store_true", help="Write only the repository PNGs")
    args = parser.parse_args()
    OUTPUT_DIRS = (OUT_REPO,) if args.repo_only else (OUT_REPO, OUT_PAPER)

    if args.group in ("all", "sim"):
        build_fig2_fwer_calibration()
        build_fig3_power_heatmap()
        build_fig7_timing_benchmark()
        build_figS4_topologies_gallery()
        build_figS5_tfnbs_eh_sweep_and_paired_diff()

    if args.group in ("all", "abide"):
        build_fig4_abide_cc200_workflow()
        build_fig5_abide_cc200_bna_suit_showcase()
        build_figS1_abide_cc200_combat_qc()
        build_figS2_abide_cc200_eh_multiverse()

    if args.group in ("all", "openclose"):
        build_fig6_openclose_bna242_showcase()
        build_figS3_openclose_retest_agreement()


if __name__ == "__main__":
    main()
