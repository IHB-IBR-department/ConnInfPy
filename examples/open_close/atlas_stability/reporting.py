"""Figures and optional Markdown report from saved contrast-stability quantities."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from examples.open_close.atlas_stability import adapter as A
from examples.open_close.atlas_stability.metrics import agreement, SEED

ALPHA, PRIMARY = 0.05, "lobe"

def figures(full, boot, common, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {"schaefer": "Schaefer-200", "bna": "BNA-246 (native)", "bna_gyrus": "BNA-gyrus (resolution control)"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
    for row, level in enumerate(("lobe", "gyrus")):
        for ax, name in zip(axes[row], A.MATCHED):
            x, y = full[("ihb", name, level)], full[("china_ec1", name, level)]
            lim = np.nanmax(np.abs(np.r_[x, y])) * 1.08
            ax.axhline(0, color="#bbb", lw=0.6)
            ax.axvline(0, color="#bbb", lw=0.6)
            ax.plot([-lim, lim], [-lim, lim], color="#999", lw=0.6, ls="--")
            ax.scatter(x, y, s=12 if level == "lobe" else 5, alpha=0.7, color="#2b6cb0")
            ccc = agreement(x, y)["ccc"][0]
            ax.set_title(f"{titles[name]}\n{level} × hemisphere pairs: CCC {ccc:.2f}", fontsize=9)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel("IHB mean Δz (EC − EO)", fontsize=8)
            ax.set_ylabel("China (EC1) mean Δz", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("Cross-cohort agreement of common-unit EC−EO effects, per atlas", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "fig_cross_cohort_units.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, level in zip(axes, ("lobe", "gyrus")):
        vals = [agreement(boot[("ihb", n, level)], boot[("china_ec1", n, level)])["ccc"] for n in A.MATCHED]
        ax.violinplot(vals, showmedians=True)
        ax.set_xticks([1, 2, 3], [titles[n].split(" (")[0] for n in A.MATCHED], fontsize=8)
        ax.set_ylabel("IHB–China concordance (CCC)", fontsize=8)
        ax.set_title(f"{level} × hemisphere: bootstrap distribution (n={len(common[level][0])} unit pairs)", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "fig_primary_bootstrap.png", dpi=150)
    plt.close(fig)

def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)

def write_report(args, n_perm, n_boot, support_rec, support, prov, branches, results, repro, ann, prim, diffs,
                 xa, stab, dprof, units_df, common, diag, report_path: Path):
    L = ["# Open-Close EC−EO contrast stability: Schaefer-200 vs native BNA-246 vs BNA-gyrus\n",
         f"Plan: `openclose_atlas_contrast_stability_plan_2026-10-10` (vault). {n_perm} permutations, "
         f"{n_boot} bootstrap draws, seed {SEED}; per-tail TFNBS (E = 0.4, H = 3, n = 10). Descriptive comparisons.\n"]

    excl = support_rec[support_rec.excluded != ""].groupby(["atlas", "excluded"]).size().reset_index(name="n")
    L.append(f"\n## 1. Common cortical support and provenance\n\nKept: {len(support['schaefer'])} Schaefer parcels, "
             f"{len(support['bna'])} BNA cortical regions; BNA-gyrus = "
             f"{len(branches[('ihb', 'bna_gyrus')].parcels)} gyrus × hemisphere signals.\n\n" + _md(excl) + "\n")
    q = prov.groupby(["experiment", "scan"])["median_r"].agg(["min", "median"]).reset_index()
    L.append("\nProvenance (per scan, median correlation of each BNA region with its most-overlapping "
             "Schaefer parcel; blocking threshold 0.5):\n\n" + _md(q) + "\n")

    rows = []
    for (cohort, name), br in branches.items():
        r = results[(cohort, name)]
        rows.append({"cohort": cohort, "branch": name, "parcels": br.eo.shape[2], "subjects": br.eo.shape[0],
                     "edges": br.eo.shape[2] * (br.eo.shape[2] - 1) // 2,
                     "EC>EO": int((r.positive < ALPHA).sum() // 2), "EO>EC": int((r.negative < ALPHA).sum() // 2)})
    L.append("\n## 2. Paired TFNBS per branch (α = 0.05 per tail)\n\nDetection counts are not comparable across "
             "branches (different edge families); see §6 for common-unit densities. Reproduction of the paper's "
             f"Schaefer map, IHB (max |Δp|): {json.dumps(repro)}. China branches use the matched 47 participants; "
             "counts must be compared against matching participant/run configurations, rather than the paper's 48-participant run-0 map.\n\n" + _md(pd.DataFrame(rows)) + "\n")

    L.append("\n## 3. Annotation only (Schaefer full set)\n\nArrays must be unchanged by annotation; attribution "
             "levels of the detected edges (worst endpoint):\n\n" + _md(ann) + "\n")

    p = prim[(prim.china == "china_ec1") & (prim.metric.isin(["ccc", "pearson", "sign_agreement"]))]
    L.append("\n## 4. Primary endpoint: IHB vs China (EC1) agreement of common-unit effects\n\n"
             "Full-sample value and 95 % bootstrap percentile interval (independent IHB and China draws).\n\n"
             + _md(p.drop(columns="china")) + "\n")
    d = diffs[(diffs.china == "china_ec1") & (diffs.metric == "ccc")]
    L.append("\n**Paired atlas differences in CCC** (same draws; descriptive percentile intervals without multiplicity adjustment). "
             "The Yeo reference is the Brainnetome subregion crosswalk: native BNA matches it exactly while Schaefer "
             "parcels are mapped through it, so `yeo_ref` differences favour BNA by construction; `yeo_own` (each "
             "atlas's own labels) is the fair functional comparison.\n\n"
             + _md(d.drop(columns=["china", "metric"])) + "\n")
    L.append("\n**Post hoc diagnostic (not prespecified):** agreement after excluding within-occipital pairs "
             "is shown as a sensitivity, separately from the primary analysis:\n\n" + _md(diag) + "\n")
    d0 = diffs[(diffs.china == "china_ec0") & (diffs.metric == "ccc") & (diffs.level == PRIMARY)]
    p0 = prim[(prim.china == "china_ec0") & (prim.metric == "ccc")]
    L.append("\n**China EC0 sensitivity** (run choice):\n\n" + _md(p0.drop(columns=["china", "metric"])) + "\n\n"
             + _md(d0.drop(columns=["china", "metric"])) + "\n")

    L.append("\n## 5. Cross-atlas agreement within cohort (same draws)\n\n"
             + _md(xa[xa.metric == "ccc"].drop(columns="metric")) + "\n")
    L.append("\n## 6. Secondary: edge stability, parcel volume, detection density\n\n"
             "Edge sign frequency is roughly Φ(t); it reflects per-edge SNR (parcel volume), not anatomical "
             "correctness.\n\n" + _md(stab) + "\n\nDetection-density profile agreement (Spearman over common "
             "unit pairs):\n\n" + _md(dprof) + "\n")

    key = units_df[(units_df.cohort.isin(["ihb", "china_ec1"])) & (units_df.level == "lobe")
                   & (units_df.unit_pair.str.count("Occipital") == 2)]
    key = key[["cohort", "atlas", "unit_pair", "mean_dz", "ci_lo", "ci_hi", "mean_dz_strong_baseline",
               "mean_dz_weak_baseline", "tau"]].sort_values(["unit_pair", "cohort", "atlas"])
    L.append("\n## 7. Within-occipital lobe pairs: signed effect, baseline split and heterogeneity\n\n"
             "Strong/weak = edges above/below the unit pair's weighted median EO coupling; τ = between-edge SD "
             "of true effects (debiased).\n\n" + _md(key) + "\n")
    L.append("\nFull tables: `unit_effects.csv`, `primary_*.csv`, `cross_atlas_agreement.csv`, `edge_stability.csv`, "
             "`detection_density*.csv`, `common_support.csv`, `provenance.csv`, `annotation_evaluation.csv`; "
             "figures `fig_cross_cohort_units.png`, `fig_primary_bootstrap.png`.\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(L))
