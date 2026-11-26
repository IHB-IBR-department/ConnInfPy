"""End-to-end fMRIPrep + nilearn workflow example for ConnInfPy.

Demonstrates how to go from fMRIPrep derivatives and on-demand nilearn
volumetric atlases (Schaefer-100 or Craddock-200) to:
1. Confound-regressed ROI time series via ``FmriprepDerivativesLoader``
2. Connectivity estimation via ``compute_connectivity()``
3. TFNBS inference with motion QC via ``conninfpy.analyze(..., motion_qc=...)``
4. Volumetric parcel-based Brainnetome + SUIT annotation (``support="parcel"``),
   connectogram rendering, and Methods boilerplate export.

No third-party NIfTI atlases or fMRI datasets are bundled with ``conninfpy``;
``nilearn`` is an optional dependency (``pip install "conninfpy[app]"``).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import conninfpy
from conninfpy import AtlasInfo, analyze, compute_connectivity
from conninfpy.atlas.brainnetome import BrainnetomeReference
from conninfpy.interpret import annotate_rois
from conninfpy.loaders import ManifestLoader
from conninfpy.plot import plot_connectogram
from conninfpy.reporting import methods_boilerplate


def _require_nilearn() -> None:
    try:
        import nibabel  # noqa: F401
        import nilearn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "This workflow requires the optional 'nilearn' and 'nibabel' packages. "
            "Install them with: pip install \"conninfpy[app]\" (or: pip install nilearn)"
        ) from exc


def _mean_fd_from_confounds(confounds_path: Path) -> float:
    df = pd.read_csv(confounds_path, sep="\t", na_values=["n/a", "N/A", "NA"])
    for col in ("framewise_displacement", "FramewiseDisplacement", "FD"):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0:
                return float(vals.mean())
    return 0.0


def _fetch_volumetric_atlas(atlas_name: str, data_dir: Path, out_dir: Path) -> tuple[Path, Any, list[str] | None]:
    from nilearn import datasets, image

    if atlas_name == "schaefer100":
        bunch = datasets.fetch_atlas_schaefer_2018(
            n_rois=100, yeo_networks=7, resolution_mm=2, data_dir=str(data_dir)
        )
        raw_labels = [
            lbl.decode("utf-8") if isinstance(lbl, bytes) else str(lbl)
            for lbl in bunch.labels
        ]
        # nilearn 0.13+ may omit Background, older versions include it at index 0
        if raw_labels and raw_labels[0].lower() == "background":
            raw_labels = raw_labels[1:]
        lut = {i + 1: name for i, name in enumerate(raw_labels)}
        networks = [
            name.split("_")[2] if len(name.split("_")) >= 3 else "Unassigned"
            for name in raw_labels
        ]
        return Path(bunch.maps), lut, networks

    if atlas_name == "craddock200":
        import nibabel as nib

        bunch = datasets.fetch_atlas_craddock_2012(data_dir=str(data_dir))
        img_4d = nib.load(bunch.scorr_mean)
        # Pick the 3D volume closest to 200 non-zero parcels
        best_idx = 0
        best_diff = float("inf")
        for idx in range(img_4d.shape[3]):
            vol = np.asarray(image.index_img(img_4d, idx).dataobj)
            n_labels = int(np.sum(np.unique(vol) > 0))
            diff = abs(n_labels - 200)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx
        img_3d = image.index_img(img_4d, best_idx)
        atlas_3d_path = out_dir / "craddock200_3d.nii.gz"
        img_3d.to_filename(str(atlas_3d_path))
        return atlas_3d_path, None, None

    raise ValueError(f"Unsupported atlas: {atlas_name}")


def _prepare_real_nilearn_inputs(
    n_subjects: int, atlas_name: str, data_dir: Path, out_dir: Path
) -> tuple[Path, Path, dict[str, Any]]:
    from nilearn import datasets

    dev = datasets.fetch_development_fmri(n_subjects=n_subjects, data_dir=str(data_dir))
    atlas_path, lut, networks = _fetch_volumetric_atlas(atlas_name, data_dir, out_dir)

    func_paths = [Path(p) for p in dev.func]
    conf_paths = [Path(p) for p in dev.confounds]
    deriv_dir = func_paths[0].parent

    pheno_rows = []
    if hasattr(dev, "phenotypic") and dev.phenotypic is not None:
        pheno_df = pd.DataFrame(dev.phenotypic).reset_index(drop=True)
    else:
        pheno_df = pd.DataFrame()

    for idx, (bf, cf) in enumerate(zip(func_paths, conf_paths)):
        sub_id = bf.name.split("_")[0].replace("sub-", "")
        age = float(pheno_df.loc[idx, "Age"]) if "Age" in pheno_df.columns else float(idx + 8)
        child_adult = (
            str(pheno_df.loc[idx, "Child_Adult"])
            if "Child_Adult" in pheno_df.columns
            else ("adult" if age >= 18 else "child")
        )
        pheno_rows.append({
            "subject_id": sub_id,
            "age": age,
            "is_adult": 1.0 if child_adult.lower() == "adult" else 0.0,
            "mean_fd": _mean_fd_from_confounds(cf),
        })

    pheno_csv = out_dir / "pheno.csv"
    pd.DataFrame(pheno_rows).to_csv(pheno_csv, index=False)

    manifest_path = out_dir / "manifest.yaml"
    manifest_doc = {
        "schema_version": 1,
        "name": f"nilearn-development-fmri-{atlas_name}",
        "loader": "FmriprepDerivativesLoader",
        "paths": {
            "derivatives_dir": str(deriv_dir),
            "pheno_path": str(pheno_csv),
            "atlas_image_path": str(atlas_path),
        },
        "metadata": {
            "cohort": "nilearn-development-fmri",
            "atlas_name": atlas_name,
            "space": "MNI152NLin2009cAsym",
        },
        "params": {
            "bold_pattern": "*desc-preproc_bold.nii.gz",
            "confounds_pattern": "*desc-confounds*.tsv",
            "subject_regex": "sub-([^_/]+)",
            "denoising": {"strategy": 1, "use_GSR": False},
            "cache_dir": str(out_dir / "loader_cache"),
        },
        "checks": {
            "min_subjects": 2,
            "expected_data_kind": "timeseries",
        },
    }
    manifest_path.write_text(yaml.safe_dump(manifest_doc, sort_keys=False))
    return manifest_path, atlas_path, {"lut": lut, "networks": networks}


def _prepare_synthetic_inputs(out_dir: Path, n_subjects: int = 6) -> tuple[Path, Path, dict[str, Any]]:
    import nibabel as nib

    rng = np.random.default_rng(42)
    synth_root = out_dir / "_synthetic_bids"
    deriv_dir = synth_root / "derivatives"
    deriv_dir.mkdir(parents=True, exist_ok=True)

    # 6x6x6 volume in MNI152NLin2009cAsym-like affine covering BNA-246 cortex
    affine = np.array([
        [-20.0, 0.0, 0.0, 50.0],
        [0.0, 20.0, 0.0, -60.0],
        [0.0, 0.0, 20.0, -10.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    atlas_arr = np.zeros((6, 6, 6), dtype=np.int32)
    parcel_coords = [
        (0, 0, 2), (1, 1, 2), (2, 2, 2),
        (3, 3, 2), (4, 4, 2), (5, 2, 3),
    ]
    for label_id, (x, y, z) in enumerate(parcel_coords, start=1):
        atlas_arr[x, y, z] = label_id
    atlas_path = synth_root / "synthetic_atlas.nii.gz"
    nib.save(nib.Nifti1Image(atlas_arr, affine), str(atlas_path))

    n_tp = 60
    pheno_rows = []
    for i in range(n_subjects):
        sub_id = f"s{i + 1:02d}"
        func_dir = deriv_dir / f"sub-{sub_id}" / "func"
        func_dir.mkdir(parents=True, exist_ok=True)
        bold_path = func_dir / f"sub-{sub_id}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        conf_path = func_dir / f"sub-{sub_id}_task-rest_desc-confounds_timeseries.tsv"

        age = 8.0 + 3.0 * i
        vol_data = rng.normal(scale=0.5, size=(6, 6, 6, n_tp)).astype(np.float32)
        # Inject age-modulated coupling between parcel 1 and parcel 2
        shared = rng.normal(size=n_tp).astype(np.float32)
        x1, y1, z1 = parcel_coords[0]
        x2, y2, z2 = parcel_coords[1]
        vol_data[x1, y1, z1, :] += (0.5 + 0.1 * i) * shared
        vol_data[x2, y2, z2, :] += (0.5 + 0.1 * i) * shared
        nib.save(nib.Nifti1Image(vol_data, affine), str(bold_path))

        motion_cols = {}
        for name in ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"):
            base = rng.normal(scale=0.05, size=n_tp)
            deriv = np.concatenate([[0.0], np.diff(base)])
            motion_cols[name] = base
            motion_cols[f"{name}_derivative1"] = deriv
            motion_cols[f"{name}_power2"] = base ** 2
            motion_cols[f"{name}_derivative1_power2"] = deriv ** 2
        motion_cols["framewise_displacement"] = ["n/a"] + list(rng.uniform(0.05, 0.25, size=n_tp - 1))
        conf_df = pd.DataFrame(motion_cols)
        conf_df.to_csv(conf_path, sep="\t", index=False)

        pheno_rows.append({
            "subject_id": sub_id,
            "age": age,
            "mean_fd": _mean_fd_from_confounds(conf_path),
        })

    pheno_csv = out_dir / "pheno.csv"
    pd.DataFrame(pheno_rows).to_csv(pheno_csv, index=False)

    manifest_path = out_dir / "manifest.yaml"
    manifest_doc = {
        "schema_version": 1,
        "name": "synthetic-fmriprep-smoke",
        "loader": "FmriprepDerivativesLoader",
        "paths": {
            "derivatives_dir": str(deriv_dir),
            "pheno_path": str(pheno_csv),
            "atlas_image_path": str(atlas_path),
        },
        "metadata": {
            "cohort": "synthetic-fmriprep",
            "atlas_name": "synthetic-6roi",
            "space": "MNI152NLin2009cAsym",
        },
        "params": {
            "bold_pattern": "sub-*/func/*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
            "confounds_pattern": "sub-*/func/*desc-confounds_timeseries.tsv",
            "subject_regex": "sub-([^_/]+)",
            "denoising": {"strategy": 1, "use_GSR": False},
            "cache_dir": str(out_dir / "loader_cache"),
        },
        "checks": {
            "expected_rois": len(parcel_coords),
            "min_subjects": 2,
            "expected_data_kind": "timeseries",
        },
    }
    manifest_path.write_text(yaml.safe_dump(manifest_doc, sort_keys=False))

    lut = {i + 1: f"ROI_{i + 1:02d}" for i in range(len(parcel_coords))}
    networks = ["Default", "Default", "Visual", "Visual", "Somatomotor", "Somatomotor"]
    return manifest_path, atlas_path, {"lut": lut, "networks": networks}


def run_workflow(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    if args.dry_run:
        print(
            f"[dry-run] atlas={args.atlas}, n_subjects={args.n_subjects}, "
            f"n_permutations={args.n_permutations}, data_dir={args.data_dir}, out_dir={out_dir}"
        )
        return {"status": "dry_run"}

    _require_nilearn()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        manifest_path, atlas_path, atlas_kw = _prepare_synthetic_inputs(
            out_dir, n_subjects=min(args.n_subjects, 6)
        )
        atlas_source = "Synthetic-6ROI"
    else:
        manifest_path, atlas_path, atlas_kw = _prepare_real_nilearn_inputs(
            args.n_subjects, args.atlas, Path(args.data_dir).expanduser(), out_dir
        )
        atlas_source = args.atlas

    dataset = ManifestLoader(str(manifest_path)).load()
    corr, diag = compute_connectivity(dataset.data, kind="correlation", return_diagnostics=True)
    print(
        f"Loaded {corr.shape[0]} subjects x {corr.shape[1]} ROIs | "
        f"n_valid_tr={diag['n_valid_tr'].tolist()} | delta_s_max={float(np.max(diag['delta_s'])):.4f}"
    )

    atlas = AtlasInfo.from_nifti(
        str(atlas_path),
        lut=atlas_kw["lut"],
        networks=atlas_kw["networks"],
        source=atlas_source,
    )

    pheno = dataset.pheno
    age = pheno["age"].to_numpy(dtype=float)
    mean_fd = pheno["mean_fd"].to_numpy(dtype=float)

    out = analyze(
        corr,
        input="r",
        interest=age,
        confounds=mean_fd[:, None],
        motion_qc={
            "fd": mean_fd,
            "subject_ids": dataset.subject_ids,
            "atlas": atlas,
            "space": "MNI152NLin2009cAsym",
        },
        atlas=atlas,
        method="tfnbs",
        n_permutations=args.n_permutations,
        acceleration=None,
        use_mp=False,
        rng=42,
    )
    print("Significant edges:", out.inference.n_significant(alpha=0.05))

    ref = BrainnetomeReference.load(profiles=False, resolution_mm=1, cerebellum="anatom")
    rois = annotate_rois(
        coords=atlas.coords,
        reference=ref,
        names=atlas.labels,
        networks=atlas.networks,
        space="MNI152NLin2009cAsym",
        support="parcel",
        parcel_image=str(atlas_path),
        parcel_labels=list(atlas.label_ids),
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    plot_connectogram(
        out.inference,
        rois,
        tail="positive",
        title=f"Age effect ({atlas_source}, parcel support)",
        ax=ax,
    )
    connectogram_path = out_dir / "connectogram_positive.png"
    fig.savefig(connectogram_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    boilerplate = methods_boilerplate(out, atlas=atlas, cerebellum="anatom")
    boilerplate_path = out_dir / "methods_boilerplate.md"
    boilerplate_path.write_text(boilerplate)

    return {
        "status": "ok",
        "manifest_path": manifest_path,
        "connectogram_path": connectogram_path,
        "boilerplate_path": boilerplate_path,
        "n_subjects": int(corr.shape[0]),
        "n_rois": int(corr.shape[1]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end fMRIPrep + nilearn volumetric workflow for ConnInfPy."
    )
    parser.add_argument("--n-subjects", type=int, default=12, help="Number of subjects to fetch (default: 12).")
    parser.add_argument(
        "--atlas",
        choices=["schaefer100", "craddock200"],
        default="schaefer100",
        help="On-demand nilearn volumetric atlas (default: schaefer100).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "nilearn_data",
        help="Directory for nilearn downloads (default: ~/nilearn_data).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Output directory for manifest, connectogram, and methods boilerplate.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=100,
        help="Number of TFNBS permutations (default: 100).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run offline with tiny synthetic fMRIPrep derivatives instead of downloading.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved settings and exit without downloading or running inference.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_workflow(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
