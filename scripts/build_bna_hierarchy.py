"""Build ``conninfpy/atlas_data/bna246_hierarchy.csv`` from the official Brainnetome files.

Inputs (atlas.brainnetome.org/download.html):
  BNA_subregions.xlsx                  Fan et al. (2016) Table 1: lobe, gyrus, subregion code,
                                       left/right label IDs, cytoarchitectonic description
  subregion_func_network_Yeo_updated.csv   BNA subregion -> Yeo 7 / 17 network (0 = not applicable)
and the bundled ``conninfpy/atlas_data/bna246.csv`` (names, voxel-mass centroids).

Every name implied by the subregion table is checked against the bundled names
before anything is written. The output is redistributed under the Brainnetome
terms (see conninfpy/atlas_data/NOTICE_BRAINNETOME.txt).

Usage:
    python scripts/build_bna_hierarchy.py [datasets/atlases]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "datasets" / "atlases"
OUT = ROOT / "conninfpy" / "atlas_data" / "bna246_hierarchy.csv"

STRUCTURE = {"Amyg": "amygdala", "Hipp": "hippocampus", "BG": "basal_ganglia", "Tha": "thalamus"}
YEO7 = {1: "Visual", 2: "Somatomotor", 3: "Dorsal Attention", 4: "Ventral Attention",
        5: "Limbic", 6: "Frontoparietal", 7: "Default"}
YEO17 = {1: "Visual peripheral", 2: "Visual central", 3: "Somato-motor A", 4: "Somato-motor B",
         5: "Dorsal attention A", 6: "Dorsal attention B", 7: "Ventral attention", 8: "Salience",
         9: "Limbic-1", 10: "Limbic-2", 11: "Control C", 12: "Control A", 13: "Control B",
         14: "Default D (Auditory)", 15: "Default C", 16: "Default A", 17: "Default B"}


def _table1(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None).iloc[1:]
    raw[0] = raw[0].ffill()
    raw[1] = raw[1].ffill()
    rows = []
    for _, r in raw.iterrows():
        lobe = str(r[0]).replace(" Lobe", "").replace(" lobe", "").strip()
        lobe = "Subcortical" if lobe.lower().startswith("subcortical") else lobe
        gyrus, gyrus_name = [s.strip() for s in str(r[1]).split(",", 1)]
        code = re.sub(r"\s+", "", str(r[2]))                       # "LOcC _L(R)_4_2" -> "LOcC_L(R)_4_2"
        description = re.sub(r"\s+", " ", str(r[5])).strip()
        for hemi, rid in (("L", int(r[3])), ("R", int(r[4]))):
            rows.append({
                "roi_id": rid, "hemisphere": hemi,
                "code": code.replace("L(R)", hemi), "description": description,
                "gyrus": gyrus, "gyrus_name": gyrus_name, "lobe": lobe,
                "structure": STRUCTURE.get(gyrus, "cortex"),
            })
    return pd.DataFrame(rows)


def _yeo(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=1, usecols=[0, 1, 2, 3, 4])
    df.columns = ["roi_id", "short", "region", "yeo7_id", "yeo17_id"]
    df = df.dropna(subset=["roi_id"]).astype({"roi_id": int, "yeo7_id": int, "yeo17_id": int})
    df["yeo7"] = df["yeo7_id"].map(YEO7).fillna("not applicable")
    df["yeo17"] = df["yeo17_id"].map(YEO17).fillna("not applicable")
    return df


def main() -> None:
    bundled = pd.read_csv(ROOT / "conninfpy" / "atlas_data" / "bna246.csv")
    t1 = _table1(SRC / "BNA_subregions.xlsx")
    yeo = _yeo(SRC / "subregion_func_network_Yeo_updated.csv")
    out = bundled[["roi_id", "name", "x", "y", "z"]].merge(t1, on="roi_id", validate="1:1")
    out = out.merge(yeo[["roi_id", "short", "region", "yeo7_id", "yeo7", "yeo17_id", "yeo17"]],
                    on="roi_id", validate="1:1")

    # Cross-checks: the IDs, hemispheres and names of the three sources agree
    assert len(out) == 246 and sorted(out.roi_id) == list(range(1, 247))
    assert (out.hemisphere == out.roi_id.map(lambda i: "L" if i % 2 else "R")).all()
    assert (out.name.str[-2:] == "_" + out.hemisphere).all()
    assert (out.name.str[:-2] == out.short).all(), out[out.name.str[:-2] != out.short]
    assert (out.code == out.region).all(), out[out.code != out.region][["roi_id", "code", "region"]]
    assert ((out.structure == "cortex") == (out.yeo7_id > 0)).all(), "Yeo networks exist for cortex only"
    out["homolog_id"] = out.roi_id.map(lambda i: i + 1 if i % 2 else i - 1)

    cols = ["roi_id", "name", "hemisphere", "homolog_id", "code", "description", "gyrus", "gyrus_name",
            "lobe", "structure", "yeo7_id", "yeo7", "yeo17_id", "yeo17", "x", "y", "z"]
    out[cols].to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} regions")


if __name__ == "__main__":
    main()
