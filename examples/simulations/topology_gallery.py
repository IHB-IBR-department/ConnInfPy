"""
Topology gallery for the 10 representative scenarios used in power/FPR analysis.

For each scenario, displays:
  - Ground truth: population correlation difference between the groups
    (Delta r on the planted edges, after the positive-definite projection)
  - Uncorrected signed t-statistic from simulated two-sample data

Generation parameters match fpr_config_1000.yaml / sweep_config_power_1000.yaml:
  n_nodes=60, n_modules=4, intra_corr=0.3, inter_corr=0.05, noise_level=0.05,
  30 time points per subject

Usage:
    python examples/simulations/topology_gallery.py
    python examples/simulations/topology_gallery.py --output-dir examples/simulations/results/topologies
    python examples/simulations/topology_gallery.py --effect-size 0.25 --n-samples 50
"""
from __future__ import annotations


from pathlib import Path as _Path
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from examples._shared.paths import ROOT, config_path
import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


from conninfpy.sim.topologies import TopologyDatasetGenerator
from examples.simulations._plotting import draw_module_boundaries as _draw_module_boundaries
from conninfpy.stats.pairwise import compute_t_stat

# The 10 scenarios used in power analysis and FPR calibration configs.
SCENARIOS: List[str] = [
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

# Short display names for plot titles.
DISPLAY_NAMES = {
    "within_module_dense": "Within-module dense",
    "between_modules_dense": "Between-modules dense",
    "partial_bipartite_between_modules": "Partial bipartite",
    "gradient_core_periphery_within_module": "Core-periphery gradient",
    "scattered_cross_block": "Scattered cross-block",
    "hub": "Hub (star)",
    "rich_club": "Rich club",
    "cross_block_connected_chain": "Cross-block chain",
    "chain": "Chain",
    "fragmented_within_module": "Fragmented within-module",
}


def plot_topology_gallery(
    n_nodes: int = 60,
    n_modules: int = 4,
    intra_corr: float = 0.3,
    inter_corr: float = 0.05,
    noise_level: float = 0.05,
    effect_size: float = 0.125,
    n_samples: int = 30,
    seed: int = 42,
    output_dir: Path = (ROOT / "examples/output"),
    dpi: int = 300,
) -> Path:
    """Generate a 10x2 gallery figure: ground truth + t-stat for each scenario."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen = TopologyDatasetGenerator(
        n_nodes=n_nodes,
        n_modules=n_modules,
        intra_corr=intra_corr,
        inter_corr=inter_corr,
        noise_level=noise_level,
        seed=seed,
    )

    panels = []
    for scenario_name in SCENARIOS:
        ds = gen.generate(
            scenario_name,
            effect_size=effect_size,
            n_samples=n_samples,
            time_points=30,
        )

        g1_z, g2_z = ds.fisher_z()
        t_dict = compute_t_stat(g1_z, g2_z, test_type="two-sample")
        t_signed = t_dict["positive"] - t_dict["negative"]

        effect_gt = ds.population_diff
        n_edges = int(np.sum(ds.effect_mask != 0) // 2)
        display_name = DISPLAY_NAMES.get(scenario_name, scenario_name)
        panels.append((display_name, n_edges, effect_gt, t_signed, ds.net_labels))

    fig = plt.figure(figsize=(7.2, 9.2), layout="constrained")
    grid = fig.add_gridspec(6, 4, height_ratios=[1, 1, 1, 1, 1, 0.06])
    axes = np.array([[fig.add_subplot(grid[row, col]) for col in range(4)] for row in range(5)])
    max_gt = max(np.max(np.abs(panel[2])) for panel in panels) or 1.0
    max_t = max(np.max(np.abs(panel[3])) for panel in panels) or 1.0
    for idx, (display_name, n_edges, effect_gt, t_signed, net_labels) in enumerate(panels):
        row = idx // 2
        col_base = (idx % 2) * 2

        ax_gt = axes[row, col_base]
        im_gt = ax_gt.imshow(effect_gt, cmap="seismic", vmin=-max_gt, vmax=max_gt)
        ax_gt.set_title(f"{chr(97 + idx)}   {display_name}\n{n_edges} planted edges", fontsize=8)
        _draw_module_boundaries(ax_gt, net_labels)

        ax_t = axes[row, col_base + 1]
        im_t = ax_t.imshow(t_signed, cmap="seismic", vmin=-max_t, vmax=max_t)
        _draw_module_boundaries(ax_t, net_labels)

        for ax in (ax_gt, ax_t):
            ax.set_xticks([])
            ax.set_yticks([])

    fig.colorbar(
        im_gt, cax=fig.add_subplot(grid[5, :2]), orientation="horizontal",
        label=r"Population $\Delta r$ (left matrix)",
    )
    fig.colorbar(
        im_t, cax=fig.add_subplot(grid[5, 2:]), orientation="horizontal",
        label=r"Uncorrected $t$ (right matrix)",
    )
    fig.supxlabel(
        rf"$n={n_samples}$ per group, {n_nodes} nodes, {n_modules} modules, nominal $\Delta r={effect_size}$",
        fontsize=8,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"topologies_gallery_dr{effect_size:g}_n{n_samples}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot ground-truth masks and t-stats for the 10 representative topologies."
    )
    parser.add_argument("--n-nodes", type=int, default=60)
    parser.add_argument("--n-modules", type=int, default=4)
    parser.add_argument("--intra-corr", type=float, default=0.3)
    parser.add_argument("--inter-corr", type=float, default=0.05)
    parser.add_argument("--noise-level", type=float, default=0.05)
    parser.add_argument("--effect-size", type=float, default=0.125,
                        help="Delta r: shift of the population correlation on planted edges")
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results" / "topologies",
    )
    args = parser.parse_args(argv)

    out_path = plot_topology_gallery(
        n_nodes=args.n_nodes,
        n_modules=args.n_modules,
        intra_corr=args.intra_corr,
        inter_corr=args.inter_corr,
        noise_level=args.noise_level,
        effect_size=args.effect_size,
        n_samples=args.n_samples,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
