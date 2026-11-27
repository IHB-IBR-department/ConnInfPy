"""Download, verify and reproducibly derive third-party Brainnetome/SUIT atlas caches.

ConnInfPy's code and Schaefer 2018 tables are MIT-licensed. The Brainnetome
(BNA-246) and DiedrichsenLab SUIT cerebellar atlases are third-party datasets
with separate licences and are not bundled in the repository or wheel.

The derived tables ``bna246.csv`` and ``bna246_hierarchy.csv`` are rebuilt
byte-for-byte from the official Brainnetome originals (SHA-256-verified against
``conninfpy/atlas_data/brainnetome_external.json``):

* ``bna246.csv`` — region names from ``BN_Atlas_246_LUT.txt``, seven-lobe
  grouping from ``BNA_subregions.xlsx`` (Fan et al. 2016, Table 1), and
  voxel-mass centroids of ``BN_Atlas_246_1mm.nii.gz`` in MNI mm (2 decimals,
  fixed ``%.2f`` formatting, CRLF line endings — the published byte layout).
* ``bna246_hierarchy.csv`` — the same Table 1 plus the official
  ``subregion_func_network_Yeo_updated.csv`` Yeo 7/17 mapping, merged onto
  ``bna246.csv`` with cross-source consistency checks (LF line endings).

Usage (package CLI; ``scripts/download_atlases.py`` is a compatible wrapper):
    conninfpy download-atlases --accept-licenses
    conninfpy download-atlases --atlas suit --accept-licenses
    conninfpy download-atlases --atlas bna --from-dir ~/Downloads/bna --accept-licenses
    conninfpy download-atlases --verify-only
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from conninfpy.atlas.resources import (
    atlas_cache_dir,
    atlas_resource_path,
    brainnetome_data_dir,
    bundled_path,
    verify_brainnetome_resource,
)

LICENSE_NOTICE = """
Third-Party Atlas Licences & Attribution Terms
==============================================
1. Brainnetome Atlas (BNA-246) & BrainMap behavioral decoding profiles:
   - Source: https://atlas.brainnetome.org/download.html
   - Terms: Non-commercial research use only, with proper attribution and
     share-alike (see conninfpy/atlas_data/NOTICE_BRAINNETOME.txt).
   - Cite: Fan et al. (2016) Cerebral Cortex 26(8):3508-3526;
     Laird et al. (2005, 2011) for BrainMap profiles.

2. DiedrichsenLab SUIT cerebellar atlases (v1.2):
   - Source: https://github.com/DiedrichsenLab/cerebellar_atlases/releases/tag/v1.2
   - Terms: Creative Commons Attribution 4.0 International (CC BY 4.0).
   - Cite: Diedrichsen et al. (2009), Buckner et al. (2011),
     Nettekoven et al. (2024).
""".strip()

# Hierarchy cross-check constants (Fan et al. 2016, Table 1; Yeo 2011 networks).
_STRUCTURE = {"Amyg": "amygdala", "Hipp": "hippocampus", "BG": "basal_ganglia", "Tha": "thalamus"}
_YEO7 = {1: "Visual", 2: "Somatomotor", 3: "Dorsal Attention", 4: "Ventral Attention",
         5: "Limbic", 6: "Frontoparietal", 7: "Default"}
_YEO17 = {1: "Visual peripheral", 2: "Visual central", 3: "Somato-motor A", 4: "Somato-motor B",
          5: "Dorsal attention A", 6: "Dorsal attention B", 7: "Ventral attention", 8: "Salience",
          9: "Limbic-1", 10: "Limbic-2", 11: "Control C", 12: "Control A", 13: "Control B",
          14: "Default D (Auditory)", 15: "Default C", 16: "Default A", 17: "Default B"}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _deterministic_gzip(raw: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9, mtime=0) as gz:
        gz.write(raw)
    return buf.getvalue()


def _download_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "conninfpy/2.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _write_atomic(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)


def _bna_manifest() -> dict:
    return json.loads(bundled_path("brainnetome_external.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Reproducible derivations (pure: verified inputs in, deterministic bytes out)
# ---------------------------------------------------------------------------

def _lut_names(path: Path) -> Dict[int, str]:
    """FreeSurfer-style LUT: ``id name R G B A`` lines; background id 0 skipped."""
    names: Dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) > 0:
            names[int(parts[0])] = parts[1]
    if not names:
        raise ValueError(f"{path}: no labelled regions found in the LUT")
    return names


def _table1(path: Path):
    """Fan et al. (2016) Table 1 (BNA_subregions.xlsx) as one row per ROI id."""
    import pandas as pd

    raw = pd.read_excel(path, header=None).iloc[1:]
    raw[0] = raw[0].ffill()
    raw[1] = raw[1].ffill()
    rows = []
    for _, r in raw.iterrows():
        lobe = str(r[0]).replace(" Lobe", "").replace(" lobe", "").strip()
        lobe = "Subcortical" if lobe.lower().startswith("subcortical") else lobe
        gyrus, gyrus_name = [s.strip() for s in str(r[1]).split(",", 1)]
        code = re.sub(r"\s+", "", str(r[2]))               # "LOcC _L(R)_4_2" -> "LOcC_L(R)_4_2"
        description = re.sub(r"\s+", " ", str(r[5])).strip()
        for hemi, rid in (("L", int(r[3])), ("R", int(r[4]))):
            rows.append({
                "roi_id": rid, "hemisphere": hemi,
                "code": code.replace("L(R)", hemi), "description": description,
                "gyrus": gyrus, "gyrus_name": gyrus_name, "lobe": lobe,
                "structure": _STRUCTURE.get(gyrus, "cortex"),
            })
    return pd.DataFrame(rows)


def _yeo(path: Path):
    """Official subregion -> Yeo 7/17 network table (0 = not applicable)."""
    import pandas as pd

    df = pd.read_csv(path, skiprows=1, usecols=[0, 1, 2, 3, 4])
    df.columns = ["roi_id", "short", "region", "yeo7_id", "yeo17_id"]
    df = df.dropna(subset=["roi_id"]).astype({"roi_id": int, "yeo7_id": int, "yeo17_id": int})
    df["yeo7"] = df["yeo7_id"].map(_YEO7).fillna("not applicable")
    df["yeo17"] = df["yeo17_id"].map(_YEO17).fillna("not applicable")
    return df


def build_bna246_csv(nifti_path: Path, lut_path: Path, subregions_path: Path) -> bytes:
    """Derive ``bna246.csv`` bytes from the official Brainnetome originals.

    Names come from the LUT, the ``network`` column is the seven-lobe grouping
    of Table 1, and ``x, y, z`` are voxel-mass centroids of the 1 mm label
    image transformed to MNI mm. The byte layout (``%.2f`` coordinates, CRLF
    line endings) reproduces the published table's SHA-256 exactly.
    """
    import nibabel as nib
    import numpy as np
    import pandas as pd

    img: Any = nib.load(str(nifti_path))
    data = np.asanyarray(img.dataobj)
    affine = img.affine
    names = _lut_names(Path(lut_path))
    t1 = _table1(Path(subregions_path))
    lobe = dict(zip(t1.roi_id, t1.lobe, strict=True))
    rows = []
    for rid in sorted(names):
        vox = np.argwhere(data == rid)
        if len(vox) == 0:
            raise ValueError(f"{nifti_path}: label {rid} ({names[rid]}) has no voxels")
        centroid = affine @ np.append(vox.mean(axis=0), 1.0)
        name = names[rid]
        if name[-2:] not in ("_L", "_R"):
            raise ValueError(f"{lut_path}: label {rid} name {name!r} lacks an _L/_R suffix")
        rows.append({
            "roi_id": rid, "name": name, "network": lobe[rid],
            "hemisphere": name[-1],
            "x": round(float(centroid[0]), 2),
            "y": round(float(centroid[1]), 2),
            "z": round(float(centroid[2]), 2),
        })
    out = pd.DataFrame(rows, columns=["roi_id", "name", "network", "hemisphere", "x", "y", "z"])
    return out.to_csv(index=False, float_format="%.2f", lineterminator="\r\n").encode("utf-8")


def build_bna_hierarchy_csv(bna_csv_path: Path, subregions_path: Path, yeo_path: Path) -> bytes:
    """Derive ``bna246_hierarchy.csv`` bytes from ``bna246.csv`` + Table 1 + Yeo.

    Every name implied by the subregion table is cross-checked against the
    ``bna246.csv`` names (IDs, hemispheres, short names, region codes) before
    anything is returned. LF line endings, the published byte layout.
    """
    import pandas as pd

    bundled = pd.read_csv(bna_csv_path)
    t1 = _table1(Path(subregions_path))
    yeo = _yeo(Path(yeo_path))
    out = bundled[["roi_id", "name", "x", "y", "z"]].merge(t1, on="roi_id", validate="1:1")
    out = out.merge(yeo[["roi_id", "short", "region", "yeo7_id", "yeo7", "yeo17_id", "yeo17"]],
                    on="roi_id", validate="1:1")

    # Cross-checks: the IDs, hemispheres and names of the three sources agree.
    n = len(bundled)
    assert len(out) == n and sorted(out.roi_id) == list(range(1, n + 1))
    assert (out.hemisphere == out.roi_id.map(lambda i: "L" if i % 2 else "R")).all()
    assert (out.name.str[-2:] == "_" + out.hemisphere).all()
    assert (out.name.str[:-2] == out.short).all(), out[out.name.str[:-2] != out.short]
    assert (out.code == out.region).all(), out[out.code != out.region][["roi_id", "code", "region"]]
    assert ((out.structure == "cortex") == (out.yeo7_id > 0)).all(), "Yeo networks exist for cortex only"
    out["homolog_id"] = out.roi_id.map(lambda i: i + 1 if i % 2 else i - 1)

    cols = ["roi_id", "name", "hemisphere", "homolog_id", "code", "description", "gyrus", "gyrus_name",
            "lobe", "structure", "yeo7_id", "yeo7", "yeo17_id", "yeo17", "x", "y", "z"]
    return out[cols].to_csv(index=False, lineterminator="\n").encode("utf-8")


# ---------------------------------------------------------------------------
# Source resolution for the derived tables
# ---------------------------------------------------------------------------

def _find_bna_source(relname: str, from_dir: Optional[Path], dest_dir: Path) -> Optional[Path]:
    """Locate a SHA-256-verified official source (``sources/...`` manifest key).

    Search order: explicit ``--from-dir`` (nested or flat), the cache directory
    (nested or flat), then ``brainnetome_data_dir()/sources``. A present but
    corrupt candidate raises ``ValueError`` instead of being skipped silently.
    """
    candidates: List[Path] = []
    if from_dir is not None:
        candidates += [from_dir / relname, from_dir / Path(relname).name]
    candidates += [dest_dir / relname, dest_dir / Path(relname).name,
                   brainnetome_data_dir() / relname]
    for cand in candidates:
        if cand.is_file():
            return verify_brainnetome_resource(cand, relname)
    return None


def _find_bna_nifti(from_dir: Optional[Path], dest_dir: Path) -> Optional[Path]:
    """Locate the verified 1 mm label image needed to derive ``bna246.csv``."""
    fname = "BN_Atlas_246_1mm.nii.gz"
    expected = _bna_manifest()["atlas_files"][fname]["sha256"]
    candidates = [dest_dir / fname]
    if from_dir is not None:
        candidates.append(from_dir / fname)
    for cand in candidates:
        if cand.is_file():
            if _sha256_file(cand) != expected:
                raise ValueError(f"{cand}: SHA-256 mismatch (expected {expected})")
            return cand
    return None


def _build_derived(fname: str, dest_dir: Path, from_dir: Optional[Path]) -> Tuple[Optional[bytes], List[str]]:
    """Build a derived table's bytes; ``None`` + missing-input list when impossible."""
    xlsx = _find_bna_source("sources/BNA_subregions.xlsx", from_dir, dest_dir)
    if fname == "bna246.csv":
        nifti = _find_bna_nifti(from_dir, dest_dir)
        lut = _find_bna_source("sources/BN_Atlas_246_LUT.txt", from_dir, dest_dir)
        missing = [label for label, path in (
            ("BN_Atlas_246_1mm.nii.gz", nifti),
            ("sources/BN_Atlas_246_LUT.txt", lut),
            ("sources/BNA_subregions.xlsx", xlsx),
        ) if path is None]
        if missing:
            return None, missing
        assert nifti is not None and lut is not None and xlsx is not None
        return build_bna246_csv(nifti, lut, xlsx), []
    if fname == "bna246_hierarchy.csv":
        bna_csv = dest_dir / "bna246.csv"
        yeo = _find_bna_source("sources/subregion_func_network_Yeo_updated.csv", from_dir, dest_dir)
        missing = [label for label, path in (
            ("bna246.csv (cache)", bna_csv if bna_csv.is_file() else None),
            ("sources/BNA_subregions.xlsx", xlsx),
            ("sources/subregion_func_network_Yeo_updated.csv", yeo),
        ) if path is None]
        if missing:
            return None, missing
        assert xlsx is not None and yeo is not None
        return build_bna_hierarchy_csv(bna_csv, xlsx, yeo), []
    raise ValueError(f"Unknown derived atlas file: {fname!r}")


# ---------------------------------------------------------------------------
# Download / copy / derive drivers
# ---------------------------------------------------------------------------

def _download_suit(dest_dir: Path, *, from_dir: Optional[Path] = None, force: bool = False) -> Dict[str, str]:
    manifest = json.loads(bundled_path("cerebellar_atlases_provenance.json").read_text(encoding="utf-8"))
    files = manifest["atlas_set"]["files"]
    status: Dict[str, str] = {}

    for fname, meta in files.items():
        target = dest_dir / fname
        expected_pkg = meta["packaged_sha256"]
        if target.is_file() and not force and _sha256_file(target) == expected_pkg:
            status[fname] = "cached"
            continue

        if from_dir is not None:
            cand = from_dir / fname
            if not cand.is_file():
                cand = from_dir / Path(meta["upstream_path"]).name
            if cand.is_file():
                raw = cand.read_bytes()
                if fname.endswith(".nii.gz") and cand.name.endswith(".nii"):
                    if _sha256_bytes(raw) != meta["upstream_sha256"]:
                        raise ValueError(f"{cand}: SHA-256 mismatch against upstream_sha256")
                    raw = _deterministic_gzip(raw)
                if _sha256_bytes(raw) != expected_pkg:
                    raise ValueError(f"{cand}: SHA-256 mismatch against packaged_sha256 ({expected_pkg})")
                _write_atomic(target, raw)
                status[fname] = "copied"
                continue

        raw = _download_bytes(meta["upstream_url"])
        if _sha256_bytes(raw) != meta["upstream_sha256"]:
            raise ValueError(f"Downloaded {meta['upstream_url']} failed upstream_sha256 check")
        if fname.endswith(".nii.gz"):
            raw = _deterministic_gzip(raw)
        if _sha256_bytes(raw) != expected_pkg:
            raise ValueError(f"Packaged {fname} failed packaged_sha256 check ({expected_pkg})")
        _write_atomic(target, raw)
        status[fname] = "downloaded"

    return status


def _download_bna(dest_dir: Path, *, from_dir: Optional[Path] = None, force: bool = False) -> Dict[str, str]:
    manifest = _bna_manifest()
    atlas_files = manifest["atlas_files"]
    status: Dict[str, str] = {}

    for fname, meta in atlas_files.items():
        target = dest_dir / fname
        expected = meta["sha256"]
        if target.is_file() and not force and _sha256_file(target) == expected:
            status[fname] = "cached"
            continue

        if from_dir is not None:
            cand = from_dir / fname
            if cand.is_file():
                if _sha256_file(cand) != expected:
                    raise ValueError(f"{cand}: SHA-256 mismatch (expected {expected})")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cand, target)
                status[fname] = "copied"
                continue

        if meta.get("download_method") == "direct":
            raw = _download_bytes(meta["url"])
            if _sha256_bytes(raw) != expected:
                raise ValueError(f"Downloaded {meta['url']} failed SHA-256 check ({expected})")
            _write_atomic(target, raw)
            status[fname] = "downloaded"
        elif meta.get("download_method") == "derived":
            derived, missing = _build_derived(fname, dest_dir, from_dir)
            if derived is None:
                status[fname] = (
                    "manual (derived table; missing verified inputs: "
                    + ", ".join(missing) + ")"
                )
                continue
            if _sha256_bytes(derived) != expected:
                raise ValueError(
                    f"Derived {fname} failed SHA-256 check ({expected}); "
                    "the verified inputs did not reproduce the published table."
                )
            _write_atomic(target, derived)
            status[fname] = "derived"
        else:
            status[fname] = f"manual ({meta.get('url') or meta.get('source_page')})"

    return status


def verify_cache(dest_dir: Path) -> int:
    """Print verification status of all Brainnetome and SUIT cache files."""
    bna_manifest = _bna_manifest()["atlas_files"]
    suit_manifest = json.loads(bundled_path("cerebellar_atlases_provenance.json").read_text(encoding="utf-8"))["atlas_set"]["files"]

    expected: Dict[str, str] = {}
    for k, v in bna_manifest.items():
        expected[k] = v["sha256"]
    for k, v in suit_manifest.items():
        expected[k] = v["packaged_sha256"]

    missing = 0
    mismatched = 0
    print(f"Atlas cache directory: {dest_dir}")
    for fname, digest in expected.items():
        path = dest_dir / fname
        if not path.is_file():
            print(f"  [MISSING] {fname}")
            missing += 1
        elif _sha256_file(path) != digest:
            print(f"  [SHA256 MISMATCH] {fname}")
            mismatched += 1
        else:
            print(f"  [OK] {fname}")
    print(f"Summary: {len(expected) - missing - mismatched}/{len(expected)} valid, {missing} missing, {mismatched} mismatched.")
    return 0 if (missing == 0 and mismatched == 0) else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="conninfpy download-atlases",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--atlas", choices=("all", "suit", "bna"), default="all",
                        help="Which atlas suite to download/prepare (default: all).")
    parser.add_argument("--dest", type=Path, default=None,
                        help="Target cache directory (default: CONNINFPY_ATLAS_DIR or ~/.cache/conninfpy/atlases).")
    parser.add_argument("--from-dir", type=Path, default=None,
                        help="Optional local folder containing pre-downloaded atlas files to verify and copy.")
    parser.add_argument("--accept-licenses", action="store_true",
                        help="Acknowledge third-party Brainnetome (NC-SA) and SUIT (CC BY 4.0) terms.")
    parser.add_argument("--force", action="store_true", help="Re-download/re-copy even if valid cache exists.")
    parser.add_argument("--verify-only", action="store_true", help="Check existing cache files without downloading.")
    args = parser.parse_args(argv)

    dest_dir = atlas_cache_dir(args.dest)
    if args.verify_only:
        return verify_cache(dest_dir)

    print(LICENSE_NOTICE)
    print()
    if not args.accept_licenses:
        if not sys.stdin.isatty():
            print(
                "Refusing to download without explicit consent. Re-run with --accept-licenses.",
                file=sys.stderr,
            )
            return 2
        ans = input("Accept these third-party terms and proceed? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Cancelled.")
            return 2

    from_dir = args.from_dir.expanduser().resolve() if args.from_dir else None
    results: Dict[str, str] = {}
    if args.atlas in ("all", "suit"):
        results.update(_download_suit(dest_dir, from_dir=from_dir, force=args.force))
    if args.atlas in ("all", "bna"):
        results.update(_download_bna(dest_dir, from_dir=from_dir, force=args.force))

    print(f"Cache directory: {dest_dir}")
    manual_items = []
    for fname, state in results.items():
        print(f"  {fname}: {state}")
        if state.startswith("manual "):
            manual_items.append((fname, state))

    if manual_items:
        print(
            "\nNote: Some Brainnetome files are hosted on CAS Cloud Box share pages "
            "(https://atlas.brainnetome.org/download.html) that require manual browser download.\n"
            f"Download them from the links above, place them in a folder (or directly in {dest_dir}), "
            "and run:\n"
            "  conninfpy download-atlases --atlas bna --from-dir <folder> --accept-licenses\n"
            "The derived tables (bna246.csv, bna246_hierarchy.csv) are rebuilt automatically "
            "once their verified inputs (the 1 mm label image and the official source tables "
            "under sources/) are available in the cache, --from-dir, or CONNINFPY_DATA/brainnetome."
        )
    return 0


def build_hierarchy_main(argv: Optional[List[str]] = None) -> int:
    """CLI of the legacy ``scripts/build_bna_hierarchy.py`` (now a thin wrapper)."""
    parser = argparse.ArgumentParser(
        prog="build_bna_hierarchy",
        description="Build bna246_hierarchy.csv from the official Brainnetome source tables.",
    )
    parser.add_argument("source_dir", nargs="?", type=Path, help="Legacy positional source directory")
    parser.add_argument("--source-dir", dest="source_override", type=Path,
                        help="Directory containing original hierarchy tables")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    source = (args.source_override or args.source_dir or brainnetome_data_dir() / "sources").expanduser().resolve()
    output = args.output or atlas_cache_dir() / "bna246_hierarchy.csv"
    bna_csv = atlas_resource_path("bna246.csv")
    t1_path = verify_brainnetome_resource(source / "BNA_subregions.xlsx", "sources/BNA_subregions.xlsx")
    yeo_path = verify_brainnetome_resource(source / "subregion_func_network_Yeo_updated.csv",
                                           "sources/subregion_func_network_Yeo_updated.csv")
    raw = build_bna_hierarchy_csv(bna_csv, t1_path, yeo_path)
    _write_atomic(output, raw)
    print(f"wrote {output}: {len(raw.splitlines()) - 1} regions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())