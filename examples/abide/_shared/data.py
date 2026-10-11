"""ABIDE prepared-data and atlas adapter shared by age and diagnosis workflows."""
from pathlib import Path
import numpy as np
from conninfpy import AtlasInfo
from examples._shared.paths import ROOT

STUDY = ROOT / "examples" / "abide"
DATA_FILE = STUDY / "results" / "abide_prepared.npz"


def load_prepared(path: Path = DATA_FILE) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run python -m examples.abide.prepare_data first.")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def build_atlas(data: dict) -> AtlasInfo:
    networks = list(data["network_order"])
    labels = np.asarray(data["net_labels"], dtype=int)
    return AtlasInfo(labels=[str(x) for x in data["roi_names"]],
                     networks=[networks[i] for i in labels],
                     source="ABIDE I — Schaefer-100 / Yeo-7 (from abide_prepared.npz)")
