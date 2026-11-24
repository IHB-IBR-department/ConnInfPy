"""ABIDE prepared-data, atlas and archive adapter shared by the age and diagnosis workflows.

Works for the legacy Schaefer-100 ``results/abide_prepared.npz`` and for CC200 datasets built by
``examples/abide/prepare_cc200.py`` (which additionally carry ``partition_rule_id``,
``roi_names``, ``assigned_roi_mask`` and ``complete_case_mask``). Runners select the file with
``--prepared-npz`` and may restrict it with ``--roi-subset assigned`` (Yeo-7 + Cerebellum +
Subcortical ROIs) and/or ``--complete-case`` (subjects without constant, NaN, ROIs).
"""
from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from conninfpy import AtlasInfo
from examples._shared.paths import ROOT

STUDY = ROOT / "examples" / "abide"
DATA_FILE = STUDY / "results" / "abide_prepared.npz"
LEGACY_ATLAS_SOURCE = "ABIDE I — Schaefer-100 / Yeo-7 (from abide_prepared.npz)"
CC200_ATLAS_SOURCE = "ABIDE I — CC200 / BNA-246+SUIT"

# Per-subject arrays that follow the subject axis when a subset is selected.
_SUBJECT_KEYS = (
    "subject_ids", "sub_id", "site", "dx_group", "group", "age", "sex", "mean_fd",
    "constant_roi_mask", "complete_case_mask", "covariate_complete_mask",
)
# Per-ROI arrays that follow the ROI axis.
_ROI_KEYS = ("roi_names", "net_labels", "roi_centroids", "assigned_roi_mask")


def load_prepared(path: Path = DATA_FILE) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run python -m examples.abide.prepare_data (Schaefer-100) or "
            f"examples/abide/prepare_cc200.py --output <path> (CC200) first."
        )
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def atlas_source(data: dict) -> str:
    """Atlas / partition label recorded with every result (legacy files carry no metadata)."""
    names = [str(r) for r in np.atleast_1d(data.get("roi_names", []))]
    rule = str(data.get("partition_rule_id", "")) if "partition_rule_id" in data else ""
    if rule.startswith("cc200") or (names and names[0].startswith("CC200")):
        return f"{CC200_ATLAS_SOURCE} (partition {rule})" if rule else CC200_ATLAS_SOURCE
    return LEGACY_ATLAS_SOURCE


def build_atlas(data: dict) -> AtlasInfo:
    networks = list(data["network_order"])
    labels = np.asarray(data["net_labels"], dtype=int)
    return AtlasInfo(labels=[str(x) for x in data["roi_names"]],
                     networks=[networks[i] for i in labels],
                     source=atlas_source(data))


def select_subset(data: dict, roi_subset: str = "all", complete_case: bool = False) -> dict:
    """Reduced copy of ``data``: assigned ROIs only and/or complete-case subjects."""
    out = dict(data)
    if roi_subset == "assigned":
        if "assigned_roi_mask" not in data:
            raise ValueError("--roi-subset assigned needs a prepared file with assigned_roi_mask (prepare_cc200.py).")
        keep = np.asarray(data["assigned_roi_mask"], dtype=bool)
        out["connectivity_z"] = data["connectivity_z"][:, keep][:, :, keep]
        for key in _ROI_KEYS:
            if key in data:
                out[key] = data[key][keep]
        out["roi_subset"] = np.array("assigned")
    if complete_case:
        if "complete_case_mask" not in data:
            raise ValueError("--complete-case needs a prepared file with complete_case_mask (prepare_cc200.py).")
        sub = np.asarray(data["complete_case_mask"], dtype=bool)
        out["connectivity_z"] = out["connectivity_z"][sub]
        for key in _SUBJECT_KEYS:
            if key in data:
                out[key] = data[key][sub]
        out["complete_case"] = np.array(True)
    return out


def add_prepared_cli(ap, default_npz: Path = DATA_FILE) -> None:
    """``--prepared-npz`` (alias of the old ``--data-file``) plus the optional subsetting flags."""
    ap.add_argument("--prepared-npz", "--data-file", dest="prepared_npz", type=Path, default=default_npz,
                    help="Prepared dataset .npz (Schaefer-100 abide_prepared.npz or a CC200 file from prepare_cc200.py).")
    ap.add_argument("--roi-subset", choices=["all", "assigned"], default="all",
                    help="'assigned' keeps only ROIs with a network block (Yeo-7 + Cerebellum + Subcortical).")
    ap.add_argument("--complete-case", action="store_true",
                    help="Keep only subjects without constant (NaN) ROIs.")


def load_selected(args) -> dict:
    """Load the prepared file chosen on the CLI and apply ``--roi-subset`` / ``--complete-case``."""
    data = load_prepared(Path(args.prepared_npz))
    return select_subset(data, getattr(args, "roi_subset", "all"), getattr(args, "complete_case", False))


def describe_selection(args, data: dict) -> str:
    y = data["connectivity_z"]
    return (f"prepared: {args.prepared_npz} -> N={y.shape[0]}, ROIs={y.shape[1]}, "
            f"roi_subset={getattr(args, 'roi_subset', 'all')}, complete_case={getattr(args, 'complete_case', False)}, "
            f"atlas: {atlas_source(data)}")


def archive_result(out, out_dir, stem: str) -> dict:
    """Save the full result next to the p-maps and prove the reload.

    ``<stem>.bundle.pkl``: ``ResultBundle`` through the package API (beta, design, corrected
    p-maps, masks, provenance; feeds ``plot_connectogram`` / ``plot_coupling_classes`` /
    ``summarize``). ``<stem>.result.pkl``: the pickled ``AnalyzeResult``, because the bundle
    has no ``coupling()``. Both are reloaded immediately; ``coupling()`` is attempted and its
    outcome reported (it needs a two-profile design, i.e. it may be undefined for GLM runs).
    """
    from conninfpy.post_analysis import ResultBundle

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info: Dict[str, object] = {"bundle": f"{stem}.bundle.pkl", "result": f"{stem}.result.pkl"}
    try:
        out.to_bundle().save(out_dir / info["bundle"])
        bundle = ResultBundle.load(out_dir / info["bundle"])
        info.update(bundle_ok=True, bundle_effect_shape=list(np.asarray(bundle.effect).shape),
                    bundle_n_roi_ids=len(bundle.roi_ids))
    except Exception as exc:
        info.update(bundle_ok=False, bundle_error=f"{type(exc).__name__}: {exc}"[:200])
    with open(out_dir / info["result"], "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_dir / info["result"], "rb") as fh:
        reloaded = pickle.load(fh)
    info["result_ok"] = True
    try:
        coupling = reloaded.inference.coupling()
        # GLM designs (no two profiles) can yield unclassified edges (None): count by string.
        counts = Counter(str(c) for c in np.asarray(coupling["classes"]).ravel())
        info.update(coupling_ok=True, reload_coupling_classes=dict(sorted(counts.items())),
                    reload_coupling_profiles=[str(p) for p in np.atleast_1d(coupling["profiles"])])
    except Exception as exc:
        info.update(coupling_ok=False, coupling_error=f"{type(exc).__name__}: {exc}"[:200])
    return info
