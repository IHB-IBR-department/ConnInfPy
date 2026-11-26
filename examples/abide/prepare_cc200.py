"""ABIDE I CC200 (Craddock 200) preparation with NaN-preserving missingness, a deterministic
11-block network partition, and an optional real-data pilot.

Inputs (ABIDE Preprocessed Connectomes Project, CPAC 'cpac/filt_global' strategy):
    <roi_dir>/*_rois_cc200.1D             ROI time series (T x 200), one per subject
    Phenotypic_V1_0b_preprocessed1.csv    phenotypes / QC (SITE_ID, DX_GROUP, AGE_AT_SCAN, SEX, func_mean_fd)
    cc200_roi_atlas.nii.gz                200-label MNI image (parcel support for the annotation)
    CC200_ROI_labels.csv                  (optional) anatomical label probabilities per ROI
Outputs (explicit ``--output``; the Schaefer-100 ``abide_prepared.npz`` is never touched):
    <output>.npz                   connectivity_z (NaN kept), phenotypes, net_labels, masks, provenance
    <output stem>_roi_table.csv    200-row ROI table (annotation + partition block)
    <output stem>_meta.json        QC bookkeeping and partition rule ID

Differences from examples/abide/prepare_data.py (Schaefer-100):
  * constant / all-zero ROI time series give NaN correlations that are KEPT (never
    0-imputed); the runners pass ``missing='available_case'`` so those edges are fitted
    on the subjects who have them and left untested where coverage is too low;
  * an explicit 11-block ``net_labels`` partition (rule below) with a rule ID, plus masks
    for the assigned ROIs (Yeo-7 + Cerebellum + Subcortical) and for complete-case subjects
    (no constant ROI).

Partition rule ``cc200_annotate_rois_bna246_cerebellum_anatom_v1`` — no atlas resampling; the
200 parcels are annotated against the Brainnetome reference shipped with ConnInfPy
(``annotate_rois(..., reference=BrainnetomeReference.load(cerebellum="anatom"),
support="parcel", space="MNI152NLin6Asym")``, see the CC200 audit §4):
  1. ``lobe == "Cerebellum"``                              -> 'Cerebellum'            (13)
  2. ``lobe == "Subcortical"`` and ``yeo7 == "not applicable"`` -> 'Subcortical'      (15)
  3. ``level == "unassigned"``                             -> 'Unassigned'             (7)
  4. ``yeo7 == "mixed"``                                   -> 'Mixed'                  (27)
  5. otherwise ``yeo7`` in the seven Yeo networks          -> that network           (138)
Blocks 0-6 are the Yeo-7 networks in the order below, then Cerebellum, Subcortical, Mixed,
Unassigned (indices 7-10). The annotation's own purity threshold (0.5) decides 4 vs 5.

Usage:
    python examples/abide/prepare_cc200.py --output results/abide_cc200_prepared.npz
    python examples/abide/prepare_cc200.py --output <dir>/abide_cc200_prepared.npz --pilot \
        --pilot-dir <dir> --pilot-n-perm 20
"""
from __future__ import annotations


from pathlib import Path as _Path
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from conninfpy import AtlasInfo
from conninfpy.atlas.brainnetome import BrainnetomeReference
from conninfpy.interpret.annotate import annotate_rois
from examples.abide._shared.data import load_prepared

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
_LOCAL_ABIDE_PCP = REPO_ROOT.parent / "local_datasets" / "ABIDE_pcp"
DEFAULT_ABIDE_PCP_DIR = _LOCAL_ABIDE_PCP if _LOCAL_ABIDE_PCP.exists() else (Path.home() / "nilearn_data" / "ABIDE_pcp")
DEFAULT_ROI_DIR = DEFAULT_ABIDE_PCP_DIR / "cpac" / "filt_global"
DEFAULT_PHENO = DEFAULT_ABIDE_PCP_DIR / "Phenotypic_V1_0b_preprocessed1.csv"
DEFAULT_ATLAS = DEFAULT_ABIDE_PCP_DIR / "cc200_roi_atlas.nii.gz"
DEFAULT_ROI_TABLE = DEFAULT_ABIDE_PCP_DIR / "CC200_ROI_labels.csv"

# QC thresholds mirror examples/abide/prepare_data.py (Schaefer-100) for comparability.
MIN_TS_LENGTH = 100          # minimum time points
MAX_MEAN_FD = 0.5            # maximum mean framewise displacement (mm)
MIN_SUBJECTS_PER_SITE = 10   # minimum subjects per site per DX group
MISSING_VALUE = -9999        # ABIDE missing value code

YEO7 = ["Visual", "Somatomotor", "Dorsal Attention", "Ventral Attention", "Limbic", "Frontoparietal", "Default"]
EXTRA_BLOCKS = ["Cerebellum", "Subcortical", "Mixed", "Unassigned"]
NETWORK_ORDER = YEO7 + EXTRA_BLOCKS            # 11 blocks, indices 0..10
PARTITION_RULE_ID = "cc200_annotate_rois_bna246_cerebellum_anatom_v1"
EXPECTED_BLOCK_SIZES = {"Visual": 29, "Somatomotor": 25, "Dorsal Attention": 20, "Ventral Attention": 11,
                        "Limbic": 8, "Frontoparietal": 18, "Default": 27, "Cerebellum": 13,
                        "Subcortical": 15, "Mixed": 27, "Unassigned": 7}
CEREBELLUM_BLOCK = NETWORK_ORDER.index("Cerebellum")
SUBCORTICAL_BLOCK = NETWORK_ORDER.index("Subcortical")
MIXED_BLOCK = NETWORK_ORDER.index("Mixed")
UNASSIGNED_BLOCK = NETWORK_ORDER.index("Unassigned")


def _git_sha() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=str(HERE.parents[1]),
        ).stdout.strip()
    except Exception:
        return None


# =============================================================================
# PHENOTYPES AND TIME SERIES
# =============================================================================

def load_phenotypic(path: Path) -> pd.DataFrame:
    pheno = pd.read_csv(path)
    pheno = pheno.replace(MISSING_VALUE, np.nan)
    pheno["FILE_ID"] = pheno["FILE_ID"].astype(str)
    return pheno


def list_subjects(roi_dir: Path) -> Dict[str, Path]:
    return {p.name[: -len("_rois_cc200.1D")]: p for p in sorted(roi_dir.glob("*_rois_cc200.1D"))}


def _read_1d(path: Path) -> np.ndarray:
    """Read a CPP .1D time-series file (whitespace separated, '#'-comment header) as (T, 200) float."""
    ts = pd.read_csv(path, sep=r"\s+", header=None, engine="c", comment="#").to_numpy(dtype=float)
    return np.asarray(ts, dtype=float)


def pearson_with_nan(ts: np.ndarray) -> np.ndarray:
    """Pearson r with NaN for constant / all-zero / NaN columns (0/0), never 0-imputed."""
    tc = ts - ts.mean(axis=0)
    sd = tc.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (tc.T @ tc) / (tc.shape[0] - 1) / np.outer(sd, sd)
    np.fill_diagonal(r, 0.0)
    return np.clip(r, -0.999, 0.999)      # np.clip keeps NaN


def compute_connectivity(
    subjects: Dict[str, Path], sub_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fisher-z connectivity (n, 200, 200), TS lengths, and the per-subject constant-ROI mask (n, 200).

    The constant mask is read off the TIME SERIES (zero variance or NaN column), not off the
    connectivity matrix, where one constant ROI blanks its whole row and column.
    """
    n = len(sub_ids)
    sample = _read_1d(next(iter(subjects.values())))
    n_rois = sample.shape[1]
    conn = np.full((n, n_rois, n_rois), np.nan)
    ts_len = np.zeros(n, dtype=int)
    const = np.ones((n, n_rois), dtype=bool)
    for i, file_id in enumerate(sub_ids):
        ts = _read_1d(subjects[file_id])
        ts_len[i] = ts.shape[0]
        if ts.shape[0] >= 2 and ts.shape[1] == n_rois:
            with np.errstate(invalid="ignore"):
                const[i] = (np.nanstd(ts, axis=0) == 0) | np.isnan(ts).all(axis=0)
            conn[i] = np.arctanh(pearson_with_nan(ts))
    return conn, ts_len, const


def apply_qc(
    pheno: pd.DataFrame, ts_len: np.ndarray, const: np.ndarray, min_ts: int, max_fd: float,
    min_per_site: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """QC mirroring prepare_data.py (TS length, mean FD, site size per DX group) + all-constant files."""
    valid = ts_len >= min_ts
    steps: Dict[str, object] = {"subjects_with_file": int(len(ts_len)), "excluded_short_ts": int((~valid).sum())}
    all_const = const.all(axis=1)
    steps["excluded_all_constant_rois"] = int((valid & all_const).sum())
    valid &= ~all_const
    fd = pheno["func_mean_fd"].to_numpy(dtype=float)
    motion_ok = (fd <= max_fd) & ~np.isnan(fd)
    steps["excluded_motion"] = int((valid & ~motion_ok).sum())
    valid &= motion_ok
    dx = pheno["DX_GROUP"].to_numpy()
    sites = pheno["SITE_ID"].to_numpy()
    counts = {}
    for s, g in zip(sites[valid], dx[valid]):
        counts[(s, g)] = counts.get((s, g), 0) + 1
    keep_sites = {s for (s, g) in counts
                  if counts.get((s, 1), 0) >= min_per_site and counts.get((s, 2), 0) >= min_per_site}
    site_ok = np.array([s in keep_sites for s in sites])
    steps["excluded_small_sites"] = int((valid & ~site_ok).sum())
    steps["sites_dropped"] = sorted({str(s) for s in set(sites) - keep_sites})
    valid &= site_ok
    steps["final_after_qc"] = int(valid.sum())
    sex = pheno["SEX"].to_numpy(dtype=float)
    cov_ok = (~np.isnan(pheno["AGE_AT_SCAN"].to_numpy(dtype=float)) & ~np.isnan(sex) & (sex > 0)
              & ~np.isnan(fd))
    return valid, {"qc_mask": valid, "covariate_complete": valid & cov_ok, "steps": steps}


# =============================================================================
# 11-BLOCK PARTITION
# =============================================================================

def partition_networks(atlas_path: Path, roi_table_path: Optional[Path] = None):
    """Annotate the 200 parcels against the Brainnetome reference and derive the 11 blocks.

    Returns (net_labels[200] in 0..10, centroids[200, 3] MNI mm, ROI info table).
    """
    ann = annotate_rois(
        reference=BrainnetomeReference.load(cerebellum="anatom"), support="parcel",
        space="MNI152NLin6Asym", parcel_image=str(atlas_path), parcel_labels=list(range(1, 201)),
    )
    t = ann.table.copy()
    yeo = t["yeo7"].astype(str)
    lobe = t["lobe"].fillna("").astype(str)
    level = t["level"].astype(str)
    net_labels = np.full(len(t), UNASSIGNED_BLOCK, dtype=int)
    rule = np.full(len(t), "", dtype=object)
    is_cereb = lobe == "Cerebellum"
    is_sub = (lobe == "Subcortical") & (yeo == "not applicable")
    is_unass = level == "unassigned"
    rest = ~is_cereb & ~is_sub & ~is_unass            # steps 4-5 never touch the dedicated blocks
    is_mixed = rest & (yeo == "mixed")
    net_labels[is_cereb] = CEREBELLUM_BLOCK
    rule[is_cereb] = "lobe == Cerebellum"
    net_labels[is_sub] = SUBCORTICAL_BLOCK
    rule[is_sub] = "lobe == Subcortical & yeo7 not applicable"
    net_labels[is_unass] = UNASSIGNED_BLOCK
    rule[is_unass] = "annotation level == unassigned"
    net_labels[is_mixed] = MIXED_BLOCK
    rule[is_mixed] = "yeo7 == mixed (annotation purity < 0.5)"
    for k, name in enumerate(YEO7):
        m = rest & (yeo == name)
        net_labels[m] = k
        rule[m] = f"yeo7 == {name}"
    t["block"] = [NETWORK_ORDER[b] for b in net_labels]
    t["block_index"] = net_labels
    t["rule"] = rule
    t["roi_name"] = [f"CC200_{k:03d}" for k in range(1, len(t) + 1)]
    if roi_table_path is not None and Path(roi_table_path).exists():   # optional anatomical cross-check
        tab = pd.read_csv(roi_table_path)
        if "AAL" in tab.columns and len(tab) == len(t):
            dom = []
            for cell in tab["AAL"]:
                pairs = {m.group(1): float(m.group(2)) for m in re.finditer(r'\["([^"]+)":\s*([0-9.]+)\]', str(cell))}
                dom.append(max(pairs, key=pairs.get) if pairs else "")
            t["aal_dominant"] = dom
    centroids = t[["x", "y", "z"]].to_numpy(dtype=float)
    return net_labels, centroids, t


# =============================================================================
# PREPARE
# =============================================================================

def prepare(args) -> Path:
    t0 = time.time()
    output = Path(args.output).resolve()
    if output.name == "abide_prepared.npz" or output == (HERE / "results" / "abide_prepared.npz").resolve():
        raise SystemExit("Refusing to overwrite the Schaefer-100 abide_prepared.npz; pass another --output.")
    roi_dir = Path(args.roi_dir).expanduser()
    pheno = load_phenotypic(Path(args.pheno).expanduser())
    subjects = list_subjects(roi_dir)
    pheno = pheno[pheno["FILE_ID"].isin(subjects)].sort_values("SUB_ID").reset_index(drop=True)
    print(f"ROI time series found: {len(subjects)} | phenotypic rows matched: {len(pheno)}")
    sub_ids = pheno["FILE_ID"].to_numpy()

    print("Computing Fisher-z connectivity (NaN for constant ROIs, never 0-imputed)...")
    conn_z, ts_len, const = compute_connectivity(subjects, sub_ids)
    valid, qc = apply_qc(pheno, ts_len, const, args.min_ts_length, args.max_mean_fd, args.min_subjects_per_site)
    print(json.dumps(qc["steps"]))

    print(f"Partition (rule {PARTITION_RULE_ID})...")
    net_labels, centroids, roi_info = partition_networks(
        Path(args.atlas).expanduser(), Path(args.roi_table).expanduser() if args.roi_table else None)
    block_sizes = np.bincount(net_labels, minlength=11)
    sizes = {NETWORK_ORDER[i]: int(block_sizes[i]) for i in range(11)}
    print("Block sizes:", sizes)
    if sizes != EXPECTED_BLOCK_SIZES:
        print(f"WARNING: block sizes differ from the audited partition {EXPECTED_BLOCK_SIZES}")
    assigned = ~np.isin(net_labels, [MIXED_BLOCK, UNASSIGNED_BLOCK])
    print(f"Assigned ROIs (Yeo-7 + Cerebellum + Subcortical): {int(assigned.sum())} / 200")

    keep = valid
    conn_keep = conn_z[keep]
    constant = const[keep]
    complete_case = ~constant.any(axis=1)
    ph = pheno[keep].reset_index(drop=True)
    cov_ok = qc["covariate_complete"][keep]
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.with_suffix("")
    roi_info.to_csv(stem.parent / f"{stem.name}_roi_table.csv", index=False)

    dx = ph["DX_GROUP"].to_numpy(dtype=float)
    fd = ph["func_mean_fd"].to_numpy(dtype=float)
    np.savez_compressed(
        output,
        connectivity_z=conn_keep.astype(np.float64),
        subject_ids=ph["FILE_ID"].to_numpy().astype(str),
        sub_id=ph["SUB_ID"].to_numpy(),
        site=ph["SITE_ID"].to_numpy().astype(str),
        dx_group=dx, group=(dx == 1).astype(float),
        age=ph["AGE_AT_SCAN"].to_numpy(dtype=float), sex=ph["SEX"].to_numpy(dtype=float), mean_fd=fd,
        roi_names=np.array([f"CC200_{k:03d}" for k in range(1, 201)]),
        roi_centroids=centroids, net_labels=net_labels, network_order=np.array(NETWORK_ORDER),
        partition_rule_id=np.array(PARTITION_RULE_ID), assigned_roi_mask=assigned,
        constant_roi_mask=constant, complete_case_mask=complete_case, covariate_complete_mask=cov_ok,
        qc_steps=np.array(json.dumps(qc["steps"])),
    )
    meta = {
        "script": "examples/abide/prepare_cc200.py", "git_sha": _git_sha(),
        "python_version": platform.python_version(), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": {"roi_dir": str(roi_dir), "pheno": args.pheno, "atlas": args.atlas, "roi_table": args.roi_table},
        "qc_thresholds": {"min_ts_length": args.min_ts_length, "max_mean_fd": args.max_mean_fd,
                          "min_subjects_per_site": args.min_subjects_per_site},
        "qc_steps": qc["steps"],
        "n_saved": int(keep.sum()), "n_covariate_complete": int(cov_ok.sum()),
        "n_complete_case": int(complete_case.sum()),
        "n_subjects_with_any_constant_roi": int((~complete_case).sum()),
        "constant_roi_rate_per_roi_mean": float(constant.mean(axis=0).mean()),
        "partition_rule_id": PARTITION_RULE_ID, "block_sizes": sizes,
        "block_sizes_match_audit": sizes == EXPECTED_BLOCK_SIZES,
        "n_assigned_rois": int(assigned.sum()),
        "missing_policy": "NaN kept for constant ROIs (never 0-imputed); analyse with missing='available_case'",
        "elapsed_s": round(time.time() - t0, 1),
    }
    (stem.parent / f"{stem.name}_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"Saved {output}  ({int(keep.sum())} subjects, {time.time() - t0:.1f}s)")
    return output


# =============================================================================
# PILOT
# =============================================================================

def _contrast_beta(inf) -> np.ndarray:
    """The contrast effect map cᵀβ (``beta`` is (N, N) or (p, N, N); ``design['contrast']`` is c)."""
    beta = np.asarray(getattr(inf, "beta"))
    design = getattr(inf, "design", None) or {}
    if beta.ndim == 3:
        c = np.asarray(design.get("contrast"), dtype=float).ravel()
        if c.size == beta.shape[0]:
            return np.einsum("p,pnm->nm", c, beta)
        return beta[-1]
    return beta


def run_pilot(npz_path: Path, pilot_dir: Path, n_perm: int, max_subjects: int, seed: int) -> dict:
    import pickle
    from conninfpy import analyze
    from conninfpy.post_analysis import ResultBundle

    pilot_dir.mkdir(parents=True, exist_ok=True)
    d = load_prepared(npz_path)
    Y = d["connectivity_z"]
    net_labels = d["net_labels"].astype(int)
    order = list(d["network_order"])
    keep = d["covariate_complete_mask"].astype(bool)
    subset_note = None
    if max_subjects and keep.sum() > max_subjects:
        idx = np.flatnonzero(keep)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_subjects, replace=False))
        keep = np.zeros(len(keep), dtype=bool)
        keep[idx] = True
        subset_note = f"pilot subset: {max_subjects} of {int(d['covariate_complete_mask'].sum())} covariate-complete subjects (seed {seed})"
    Yk, site, group = Y[keep], d["site"][keep], d["group"][keep].astype(float)
    age, sex, fd = d["age"][keep].astype(float), d["sex"][keep].astype(float), d["mean_fd"][keep].astype(float)
    const = d["constant_roi_mask"][keep].astype(bool)
    n_const = const.sum(axis=1)
    rep: Dict[str, object] = {"npz": str(npz_path), "subset_note": subset_note,
                              "n_saved": int(len(Y)), "n_covariate_complete": int(d["covariate_complete_mask"].sum()),
                              "n_final_analysis": int(keep.sum()), "n_rois": int(Y.shape[1]),
                              "qc_steps": json.loads(str(d["qc_steps"]))}
    print(f"N saved={len(Y)}  covariate-complete={rep['n_covariate_complete']}  analysed={rep['n_final_analysis']}"
          + (f"  ({subset_note})" if subset_note else ""))

    # --- sample composition and missingness ---------------------------------
    dx_name = pd.Series(np.where(group == 1, "ASD", "HC"), name="dx")
    comp = pd.crosstab(pd.Series(site, name="site"), dx_name)
    print("Site x DX (analysed subjects):\n" + comp.to_string())
    rep["site_dx_table"] = comp.to_dict()
    rep["n_sites"] = int(len(np.unique(site)))
    rep["site_sizes_min_max"] = [int(comp.sum(axis=1).min()), int(comp.sum(axis=1).max())]
    miss_site = pd.Series(n_const).groupby(pd.Series(site, name="site")).agg(["mean", "max", "count"]).round(3)
    miss_dx = pd.Series(n_const).groupby(dx_name).agg(["mean", "max", "count"]).round(3)
    print("Constant (NaN) ROIs per subject by site:\n" + miss_site.to_string())
    print("Constant (NaN) ROIs per subject by DX:\n" + miss_dx.to_string())
    rep["constant_rois_per_subject"] = {
        "mean": float(n_const.mean()), "max": int(n_const.max()),
        "share_subjects_with_any": float((n_const > 0).mean()),
        "by_dx_mean": {str(k): float(v) for k, v in miss_dx["mean"].items()},
        "by_site_mean": {str(k): float(v) for k, v in miss_site["mean"].items()},
        "corr_with_age": float(np.corrcoef(n_const, age)[0, 1]) if n_const.std() > 0 else None,
        "corr_with_fd": float(np.corrcoef(n_const, fd)[0, 1]) if n_const.std() > 0 else None,
    }
    roi_missing = const.mean(axis=0)
    rep["rois_ever_constant"] = int((roi_missing > 0).sum())
    rep["roi_missing_rate_top"] = {f"CC200_{i + 1:03d}": round(float(roi_missing[i]), 3)
                                   for i in np.argsort(-roi_missing)[:10] if roi_missing[i] > 0}
    print(f"Constant-ROI count: mean {n_const.mean():.2f}/subject, max {n_const.max()}; "
          f"corr(age)={rep['constant_rois_per_subject']['corr_with_age']}, "
          f"corr(FD)={rep['constant_rois_per_subject']['corr_with_fd']}; ROIs ever constant: {rep['rois_ever_constant']}")

    # --- ComBat harmonizability -----------------------------------------------
    sites, counts = np.unique(site, return_counts=True)
    rep["combat_harmonizability"] = {
        "n_sites": int(len(sites)), "sites_with_lt2_subjects": [str(s) for s in sites[counts < 2]],
        "min_site_size": int(counts.min()), "median_site_size": float(np.median(counts)),
        "design_columns": 1 + 3 + (len(sites) - 1) + 1,
        "note": "intercept + age, sex, mean_fd + site dummies (K-1) + interest (group); stratified permutations within site",
    }
    print(f"ComBat/GLM: {len(sites)} sites, min site size {counts.min()}, design columns {rep['combat_harmonizability']['design_columns']}")

    # --- analysis: ComBat + site-dummy GLM + available_case ------------------
    atlas = AtlasInfo(labels=[str(r) for r in d["roi_names"]], networks=[order[i] for i in net_labels],
                      source="ABIDE I CC200 — 11-block partition " + str(d["partition_rule_id"]))
    confounds = np.column_stack([age, sex, fd])
    t0 = time.time()
    out = analyze(
        Yk, interest=group, confounds=confounds, sites=site,
        harmonize="combat_site_dummies_glm", fisher_z=False, method="tfnbs",
        e=0.4, h=3.0, n=None, start_thres=1.65, two_tailed=True,
        n_permutations=n_perm, acceleration=None, rng=seed, use_mp=False,
        missing="available_case", atlas=atlas, roi_ids=atlas.labels,
    )
    t_an = time.time() - t0
    inf = out.inference
    mr = inf.missing_report
    iu = np.triu_indices(Y.shape[1], k=1)
    pos = np.asarray(inf["positive"])
    tested = ~np.isnan(pos[iu])
    edge_n_full = np.asarray(mr["edge_n"], dtype=float)
    edge_n = edge_n_full[iu] if edge_n_full.shape == pos.shape else edge_n_full.ravel()
    n_design = len(inf.design["columns"]) if isinstance(inf.design, dict) and "columns" in inf.design else np.nan
    diag = getattr(out, "combat_diagnostics", None)
    reasons = mr.get("untested_reasons")
    rep["analysis"] = {
        "elapsed_s": round(t_an, 2), "n_permutations": n_perm,
        "missing_policy": str(mr.get("policy")), "min_coverage": float(mr.get("min_coverage", np.nan)),
        "n_missingness_patterns": int(mr.get("n_patterns", 0)),
        "n_edges_total": int(iu[0].size), "n_edges_tested": int(tested.sum()),
        "n_edges_untested": int((~tested).sum()), "n_edges_untested_report": int(mr.get("n_edges_untested", -1)),
        "untested_reasons": {str(k): int(v) for k, v in reasons.items()} if isinstance(reasons, dict) else str(reasons)[:200],
        "edge_n_min_max": [float(np.nanmin(edge_n)), float(np.nanmax(edge_n))] if edge_n.size else None,
        "edge_df_min_max": [float(np.nanmin(edge_n) - n_design), float(np.nanmax(edge_n) - n_design)] if edge_n.size and not np.isnan(n_design) else None,
        "n_design_columns": None if np.isnan(n_design) else int(n_design),
        "n_significant_pos_neg": {k: int(v) for k, v in inf.n_significant(alpha=0.05).items()},
        "combat_diagnostics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in (diag or {}).items()} if isinstance(diag, dict) else str(diag),
    }
    a = rep["analysis"]
    print(f"GLM analysis {t_an:.1f}s: edges tested {a['n_edges_tested']}/{a['n_edges_total']} (untested {a['n_edges_untested']}); "
          f"edge n in {a['edge_n_min_max']}, df in {a['edge_df_min_max']} ({a['n_design_columns']} design columns); sig {a['n_significant_pos_neg']}")
    print("ComBat diagnostics:", {k: v for k, v in a["combat_diagnostics"].items() if "ratio" in k or k == "strategy"})

    # --- block sizes ---------------------------------------------------------
    sizes = np.bincount(net_labels, minlength=11)
    rep["block_sizes"] = {order[i]: int(sizes[i]) for i in range(11)}
    assigned = d["assigned_roi_mask"].astype(bool)
    rep["n_assigned_rois"] = int(assigned.sum())
    print("Block sizes (11):", rep["block_sizes"])

    # --- save -> reload round-trip (GLM bundle + pickled result) --------------
    bundle_path = pilot_dir / "cc200_dx_strategy_d.bundle.pkl"
    result_path = pilot_dir / "cc200_dx_strategy_d.result.pkl"
    out.to_bundle().save(bundle_path)
    with open(result_path, "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    bundle = ResultBundle.load(bundle_path)
    with open(result_path, "rb") as fh:
        reloaded = pickle.load(fh)
    eff_b = np.asarray(bundle.effect, dtype=float)
    eff_r = np.asarray(_contrast_beta(reloaded.inference), dtype=float)
    off = ~np.eye(eff_b.shape[0], dtype=bool)          # the diagonal is not an edge (NaN vs 0 by convention)
    eff_diff = float(np.nanmax(np.abs(eff_b[off] - eff_r[off])))
    eff_equal = bool(np.allclose(eff_b[off], eff_r[off], equal_nan=True, rtol=1e-10, atol=1e-12))
    assert eff_equal, f"ResultBundle.effect and the reloaded cᵀβ differ on off-diagonal edges (max|Δ|={eff_diff})"
    rt: Dict[str, object] = {"glm": {
        "bundle_reload_ok": True, "result_reload_ok": True,
        "bundle_effect_shape": list(eff_b.shape),
        "bundle_n_roi_ids": len(bundle.roi_ids),
        "effect_equal_after_reload": eff_equal, "effect_max_abs_diff_offdiag": eff_diff,
        "corrected_p_equal_after_reload": bool(np.array_equal(np.asarray(bundle.corrected_p["positive"]), np.asarray(reloaded.inference["positive"]))),
    }}
    try:
        coupling = reloaded.inference.coupling()
        rt["glm"]["coupling_ok"] = True
    except Exception as exc:            # reported, not hidden: coupling() needs design['comparison']
        rt["glm"]["coupling_ok"] = False
        rt["glm"]["coupling_error"] = f"{type(exc).__name__}: {exc}"[:200]
    print("GLM round-trip:", json.dumps(rt["glm"], default=str))

    # --- coupling() is defined for two-profile designs: two-group ASD vs HC ------
    cc = d["complete_case_mask"][keep].astype(bool)
    asd, hc = Yk[cc & (group == 1)], Yk[cc & (group == 0)]
    try:
        t0 = time.time()
        out2 = analyze(
            group1=asd, group2=hc, test_type="two-sample", fisher_z=False, method="tfnbs",
            e=0.4, h=3.0, n=None, start_thres=1.65, two_tailed=True,
            n_permutations=min(5, n_perm), acceleration=None, rng=seed, use_mp=False,
            atlas=atlas, roi_ids=atlas.labels,
        )
        b2 = pilot_dir / "cc200_asd_vs_hc_twosample.bundle.pkl"
        r2p = pilot_dir / "cc200_asd_vs_hc_twosample.result.pkl"
        out2.to_bundle().save(b2)
        with open(r2p, "wb") as fh:
            pickle.dump(out2, fh, protocol=pickle.HIGHEST_PROTOCOL)
        ResultBundle.load(b2)
        with open(r2p, "rb") as fh:
            re2 = pickle.load(fh)
        coupling = re2.inference.coupling()
        classes, cnt = np.unique(np.asarray(coupling["classes"]), return_counts=True)
        rt["two_group"] = {
            "design": "ASD (group1) vs HC (group2), two-sample, complete-case subjects, no covariates/ComBat",
            "n_asd": int(len(asd)), "n_hc": int(len(hc)), "elapsed_s": round(time.time() - t0, 1),
            "bundle_reload_ok": True, "result_reload_ok": True, "coupling_ok": True,
            "coupling_profiles": [str(p) for p in np.atleast_1d(coupling["profiles"])],
            "design_comparison": [str(c) for c in np.atleast_1d(re2.inference.design.get("comparison", []))],
            "coupling_classes": {str(c): int(v) for c, v in zip(classes, cnt)},
        }
    except Exception as exc:
        rt["two_group"] = {"coupling_ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    print("Two-group coupling check:", json.dumps(rt["two_group"], default=str))
    rep["round_trip"] = rt
    (pilot_dir / "cc200_pilot_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"Saved {pilot_dir / 'cc200_pilot_report.json'}")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True, help="npz path for the CC200 dataset (never abide_prepared.npz).")
    ap.add_argument("--roi-dir", type=Path, default=DEFAULT_ROI_DIR)
    ap.add_argument("--pheno", type=Path, default=DEFAULT_PHENO)
    ap.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    ap.add_argument("--roi-table", type=Path, default=DEFAULT_ROI_TABLE, help="Optional CC200_ROI_labels.csv (AAL cross-check column).")
    ap.add_argument("--min-ts-length", type=int, default=MIN_TS_LENGTH)
    ap.add_argument("--max-mean-fd", type=float, default=MAX_MEAN_FD)
    ap.add_argument("--min-subjects-per-site", type=int, default=MIN_SUBJECTS_PER_SITE)
    ap.add_argument("--pilot", action="store_true", help="Run the real-data pilot on the prepared dataset.")
    ap.add_argument("--pilot-only", action="store_true", help="Skip preparation; --output must already exist.")
    ap.add_argument("--pilot-dir", type=Path, default=None, help="Directory for pilot artefacts (default: <output dir>/pilot).")
    ap.add_argument("--pilot-n-perm", type=int, default=20)
    ap.add_argument("--pilot-max-subjects", type=int, default=0, help="0 = all covariate-complete subjects.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    output = args.output.resolve()
    if not args.pilot_only:
        prepare(args)
    if args.pilot or args.pilot_only:
        if not output.exists():
            raise SystemExit(f"{output} does not exist; run without --pilot-only first.")
        run_pilot(output, args.pilot_dir or (output.parent / "pilot"), args.pilot_n_perm, args.pilot_max_subjects, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
