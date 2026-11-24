"""Open-Close paired TFNBS under analyze(test_type='paired').

Canonical paired Open-Close validation. With no ``sites=`` argument, the
wrapper skips ComBat and runs the paired sign-flip permutation. Outputs are
atlas-aware p-maps, a full result archive and edge tables under
``results/paper_validation/``.

Sign convention (recorded in every ``*_meta.json``): ``analyze(group1, group2,
test_type='paired')`` tests the paired difference ``group2 - group1`` (``B_minus_A``).
EO/EC analyses: cond1=EO (open), cond2=EC (close), contrast ``EC - EO`` -> ``positive``
= **EC > EO**, ``negative`` = **EO > EC**; the China test-retest (close run 0 vs run 1):
cond1=EC0, cond2=EC1, contrast ``EC1 - EC0`` -> ``positive`` = **EC1 > EC0**. The effect
map (``beta``) is the paired mean difference on the Fisher-z scale. The package names the
design columns generically (``condition_A_mean``, ``B_minus_A``); the A/B -> cond1/cond2
mapping is written to prints, metadata (``design_label_map``) and the ``.npz``.

Archive per analysis (``<out_stem>``):
    <out_stem>.npz            positive / negative p-maps, beta, subject pair IDs
    <out_stem>.bundle.pkl     ResultBundle (conninfpy.post_analysis): beta, design,
                              corrected p-maps, masks, provenance — feeds
                              plot_connectogram / plot_coupling_classes / summarize
    <out_stem>.result.pkl     pickled AnalyzeResult (has ``.inference.coupling()``)
    <out_stem>_edges.csv      atlas-aware edge table
    <out_stem>_meta.json      resolved contract, sign convention, pair IDs, provenance

Usage:
    python examples/open_close/paper_validation/run_paired_tfnbs.py --quick
    python examples/open_close/paper_validation/run_paired_tfnbs.py --analyses ihb
    python examples/open_close/paper_validation/run_paired_tfnbs.py --analyses ihb --n-perm 20
    # 2x2 control, legacy cell: 10-point grid x per-tail families
    python examples/open_close/paper_validation/run_paired_tfnbs.py --analyses ihb \
        --tfnbs-n 10 --one-tailed --output results/paper_validation/legacy_n10_onetail

ConnInfPy 2.0.0 defaults: exact TFNBS integration (``--tfnbs-n none``) and one
two-sided family (``--two-tailed``); ``--tfnbs-n 10 --one-tailed`` reproduces the
legacy settings for the 2x2 control.
"""
from __future__ import annotations


from pathlib import Path as _Path
import sys as _sys

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import pickle
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from conninfpy import AtlasInfo, analyze
from examples.open_close._shared.data import OpenCloseDataset, DATA_DIR
from examples.open_close._shared.paths import PAPER
from examples.open_close._shared.inference import add_inference_cli, describe_contract, write_metadata


OUT_ROOT = PAPER

# analyze(group1, group2, test_type="paired") tests group2 - group1: `positive` = cond2 > cond1.
# Each analysis passes explicit condition labels (full name, short name) and a contrast label;
# the package names the design columns generically (condition_A_mean, B_minus_A), so the
# mapping A -> cond1, B -> cond2 is recorded in prints, metadata and the .npz.
DEFAULT_COND1 = ("EO (open)", "EO")
DEFAULT_COND2 = ("EC (close)", "EC")
DEFAULT_CONTRAST = "EC - EO"


def _build_atlas(net_labels: np.ndarray, names: list[str]) -> AtlasInfo:
    """Construct a Schaefer-200/182 AtlasInfo from the loader output.

    The loader exposes integer-coded `net_labels` plus a `names` list
    indexed by those integers; `AtlasInfo` wants per-ROI network names.
    """
    net_labels = np.asarray(net_labels, dtype=int)
    networks_per_roi = [names[i] for i in net_labels]
    return AtlasInfo(
        labels=[f"roi_{i:03d}" for i in range(len(net_labels))],
        networks=networks_per_roi,
        source="Open-Close Schaefer-200 / 182 ROIs (loader-provided net_labels)",
    )


def _summary(label: str, result, positive_tail: str, negative_tail: str, alpha: float = 0.05) -> None:
    nsig = result.n_significant(alpha=alpha)
    print(f"  {label} sig@α={alpha}: positive [{positive_tail}]={nsig['positive']}, "
          f"negative [{negative_tail}]={nsig['negative']}")


def _archive(out, out_stem: str) -> dict:
    """Save the full result and prove the reload: ResultBundle (official API) + pickled result.

    ``ResultBundle`` carries beta / design / masks / corrected p-maps for the
    reporting and plotting layer; it has no ``coupling()``, so the AnalyzeResult is
    pickled next to it for ``result.inference.coupling()``. Both are reloaded here.
    """
    from conninfpy.post_analysis import ResultBundle

    bundle_path = OUT_ROOT / f"{out_stem}.bundle.pkl"
    result_path = OUT_ROOT / f"{out_stem}.result.pkl"
    out.to_bundle().save(bundle_path)
    with open(result_path, "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)

    bundle = ResultBundle.load(bundle_path)
    with open(result_path, "rb") as fh:
        reloaded = pickle.load(fh)
    coupling = reloaded.inference.coupling()
    classes, counts = np.unique(np.asarray(coupling["classes"]), return_counts=True)
    return {
        "bundle": bundle_path.name,
        "result": result_path.name,
        "bundle_effect_shape": list(np.asarray(bundle.effect).shape),
        "bundle_n_roi_ids": len(bundle.roi_ids),
        "bundle_reload_ok": True,
        "result_reload_ok": True,
        "reload_coupling_profiles": [str(p) for p in np.atleast_1d(coupling["profiles"])],
        "reload_coupling_classes": {str(c): int(n) for c, n in zip(classes, counts)},
    }


def _run_paired(
    g1: np.ndarray,
    g2: np.ndarray,
    atlas: AtlasInfo,
    label: str,
    out_stem: str,
    n_perm: int,
    seed: int,
    tfnbs_n=None,
    two_tailed: bool = True,
    subject_ids: Optional[Sequence[str]] = None,
    cond1: Tuple[str, str] = DEFAULT_COND1,
    cond2: Tuple[str, str] = DEFAULT_COND2,
    contrast_label: str = DEFAULT_CONTRAST,
) -> None:
    subject_ids = [str(s) for s in subject_ids] if subject_ids is not None else None
    positive_tail = f"{cond2[1]} > {cond1[1]}"
    negative_tail = f"{cond1[1]} > {cond2[1]}"
    design_label_map = {
        "A": cond1[0], "B": cond2[0],
        "condition_A_mean": f"mean({cond1[1]})",
        "B_minus_A": f"{cond2[1]} - {cond1[1]}  (= {contrast_label})",
    }
    print(f"\n{'=' * 72}\n{label} — {g1.shape[0]} subjects, "
          f"{g1.shape[1]}² edges\n{'=' * 72}")
    print(f"  cond1={cond1[0]}, cond2={cond2[0]}; contrast {contrast_label} "
          f"(paired difference = cond2 - cond1)")
    print(f"  positive tail = {positive_tail}; negative tail = {negative_tail}")
    print(f"  design labels: {design_label_map}")
    print(f"  {describe_contract(tfnbs_n, two_tailed)}, e=0.4, h=3.0, start_thres=1.65, "
          f"n_perm={n_perm}, seed={seed}")
    t0 = time.time()
    out = analyze(
        group1=g1, group2=g2,
        test_type="paired",
        fisher_z=False,                              # caller already Fisher-z'd
        method="tfnbs",
        e=0.4, h=3.0, n=tfnbs_n, start_thres=1.65,  # 2.0.0: n=None = exact integration
        two_tailed=two_tailed,                       # 2.0.0: one two-sided family
        n_permutations=n_perm,
        acceleration=None,
        rng=seed,
        use_mp=True,
        atlas=atlas,
        roi_ids=atlas.labels,
    )
    print(f"  time: {time.time() - t0:.1f}s")
    _summary(label, out.inference, positive_tail, negative_tail)
    for flag in out.flags:
        print(f"  flag: {flag}")
    # Sign check on the effect map: the contrast coefficient (B - A = EC - EO) must track
    # mean(EC - EO), not mean(EO - EC). beta is (p, N, N) with design['columns'] naming p.
    beta_full = np.asarray(getattr(out.inference, "beta"))
    cols = list((getattr(out.inference, "design", None) or {}).get("columns", []))
    if beta_full.ndim == 3:
        beta = beta_full[cols.index("B_minus_A")] if "B_minus_A" in cols else beta_full[-1]
    else:
        beta = beta_full
    iu = np.triu_indices(g1.shape[1], k=1)
    d_ec_eo = np.mean(g2 - g1, axis=0)[iu]
    print(f"  contrast check: corr(beta[B_minus_A], mean(EC-EO)) = {np.corrcoef(beta[iu], d_ec_eo)[0, 1]:+.4f} "
          f"(vs mean(EO-EC): {np.corrcoef(beta[iu], -d_ec_eo)[0, 1]:+.4f}); "
          f"max|beta - mean(EC-EO)| = {np.max(np.abs(beta[iu] - d_ec_eo)):.2e}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_ROOT / f"{out_stem}.npz",
        positive=out.inference["positive"],
        negative=out.inference["negative"],
        beta=beta_full,
        beta_contrast=beta,
        subject_ids=np.asarray(subject_ids if subject_ids is not None else [], dtype=str),
        design_labels=np.array([cond1[0], cond2[0], contrast_label, positive_tail, negative_tail]),
    )
    out.to_csv(
        OUT_ROOT / f"{out_stem}_edges.csv",
        atlas=atlas,
        alpha=0.05,
        sort="network_pair",
    )
    print(f"  saved: {OUT_ROOT / f'{out_stem}.npz'}")
    print(f"  saved: {OUT_ROOT / f'{out_stem}_edges.csv'}")
    archive = _archive(out, out_stem)
    print(f"  saved: {OUT_ROOT / archive['bundle']}  (ResultBundle, reload OK)")
    print(f"  saved: {OUT_ROOT / archive['result']}  (AnalyzeResult, reload OK, "
          f"coupling classes: {archive['reload_coupling_classes']})")
    design = getattr(out.inference, "design", None)
    write_metadata(
        OUT_ROOT / f"{out_stem}_meta.json", script=Path(__file__).name,
        output=f"{out_stem}.npz", method="tfnbs", test_type="paired",
        cond1=cond1[0], cond2=cond2[0], contrast=contrast_label,
        positive_tail=positive_tail, negative_tail=negative_tail, design_label_map=design_label_map,
        e=0.4, h=3.0, n=tfnbs_n, tfnbs_n=tfnbs_n, start_thres=1.65, two_tailed=two_tailed,
        n_permutations=n_perm, seed=seed, acceleration=None,
        n_subjects=int(g1.shape[0]), n_rois=int(g1.shape[1]),
        subject_pair_ids=subject_ids,
        subject_pair_ids_available=subject_ids is not None,
        design=design, label=label, archive=archive,
    )


def main() -> None:
    global OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="500 perms for smoke testing (~30 s each)")
    ap.add_argument("--n-perm", type=int, default=None,
                    help="Permutations per test (overrides --quick / the 5000 default).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--analyses", nargs="+",
                    choices=["ihb", "china", "china_retest", "pooled", "all"],
                    default=["all"])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--output", type=Path, default=OUT_ROOT)
    add_inference_cli(ap)
    args = ap.parse_args()
    OUT_ROOT = args.output.resolve()
    if args.n_perm is not None:
        n_perm = args.n_perm
    else:
        n_perm = 500 if args.quick else 5000
    targets = ["ihb", "china", "china_retest", "pooled"] if "all" in args.analyses \
        else args.analyses

    print(f"Output → {OUT_ROOT}")
    print(f"n_permutations = {n_perm}")
    print(f"{describe_contract(args.tfnbs_n, args.two_tailed)}, seed = {args.seed}")
    print("Sign convention (per analysis): positive = cond2 > cond1, negative = cond1 > cond2 "
          "(paired difference = cond2 - cond1)")

    # ---- §3.5 IHB paired open vs close --------------------------------
    if "ihb" in targets:
        ds = OpenCloseDataset.load("ihb", data_dir=args.data_dir)
        atlas = _build_atlas(ds.net_labels, ds.get_label_names())
        o_z, c_z = ds.connectivity_z(run=0)
        _run_paired(
            o_z, c_z, atlas,
            label="§3.5 IHB paired open vs close",
            out_stem="ihb_paired_tfnbs",
            n_perm=n_perm, seed=args.seed,
            tfnbs_n=args.tfnbs_n, two_tailed=args.two_tailed,
            subject_ids=list(ds.subject_ids),
        )

    # ---- §3.6 China paired (run 0) -----------------------------------
    if "china" in targets:
        ds = OpenCloseDataset.load("china", data_dir=args.data_dir)
        atlas = _build_atlas(ds.net_labels, ds.get_label_names())
        o_z, c_z = ds.connectivity_z(run=0)
        _run_paired(
            o_z, c_z, atlas,
            label="§3.6 China paired (close run 0)",
            out_stem="china_paired_tfnbs_run0",
            n_perm=n_perm, seed=args.seed,
            tfnbs_n=args.tfnbs_n, two_tailed=args.two_tailed,
            subject_ids=list(ds.subject_ids),
        )

    # ---- §3.6 China test-retest: close run 0 vs run 1 ----------------
    if "china_retest" in targets:
        ds = OpenCloseDataset.load("china", data_dir=args.data_dir)
        atlas = _build_atlas(ds.net_labels, ds.get_label_names())
        mask = ds.retest_subject_mask()
        excluded = [s for s, m in zip(ds.subject_ids, mask) if not m]
        print(f"\n§3.6 China test-retest — excluding "
              f"{(~mask).sum()} subject(s): {excluded}")
        _, c0 = ds.connectivity_z(run=0)
        _, c1 = ds.connectivity_z(run=1)
        c0 = c0[mask]
        c1 = c1[mask]
        _run_paired(
            c0, c1, atlas,
            label="§3.6 China test-retest (close run 0 vs run 1)",
            out_stem="china_retest_tfnbs",
            n_perm=n_perm, seed=args.seed,
            tfnbs_n=args.tfnbs_n, two_tailed=args.two_tailed,
            subject_ids=[s for s, m in zip(ds.subject_ids, mask) if m],
            cond1=("EC0 (close run 0)", "EC0"), cond2=("EC1 (close run 1)", "EC1"),
            contrast_label="EC1 - EC0",
        )

    # ---- Pooled Harmonized ------------------------------------------
    if "pooled" in targets:
        harm_path = OUT_ROOT / "openclose_harmonized.npz"
        if not harm_path.exists():
            print(f"\nSkipping 'pooled' analysis: Harmonized data not found at {harm_path}.")
            print("Please run `python -m examples.open_close.paper_validation.harmonize_pooled_cohorts` first.")
        else:
            ds = OpenCloseDataset.load("ihb", data_dir=args.data_dir)  # Load for atlas info
            atlas = _build_atlas(ds.net_labels, ds.get_label_names())

            d = np.load(harm_path)
            o_ihb = d["Y_ihb_open_harm"]
            c_ihb = d["Y_ihb_close_harm"]
            o_china = d["Y_china_open_harm"]
            c_china = d["Y_china_close_harm"]

            o_pooled = np.concatenate([o_ihb, o_china], axis=0)
            c_pooled = np.concatenate([c_ihb, c_china], axis=0)

            _run_paired(
                o_pooled, c_pooled, atlas,
                label="Pooled Harmonized (IHB + China) paired open vs close",
                out_stem="pooled_paired_tfnbs",
                n_perm=n_perm, seed=args.seed,
                tfnbs_n=args.tfnbs_n, two_tailed=args.two_tailed,
                subject_ids=None,   # the harmonized archive carries no IDs
            )


if __name__ == "__main__":
    main()
