"""Download and verify third-party Brainnetome and SUIT atlases into the local cache.

ConnInfPy's code and Schaefer 2018 tables are MIT-licensed. The Brainnetome
(BNA-246) and DiedrichsenLab SUIT cerebellar atlases are third-party datasets
with separate licences and are not bundled in the repository or wheel.

Usage:
    python scripts/download_atlases.py --accept-licenses
    python scripts/download_atlases.py --atlas suit --accept-licenses
    python scripts/download_atlases.py --atlas bna --from-dir ~/Downloads/bna --accept-licenses
    python scripts/download_atlases.py --verify-only
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from conninfpy.atlas.resources import atlas_cache_dir, bundled_path


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
    manifest = json.loads(bundled_path("brainnetome_external.json").read_text(encoding="utf-8"))
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
        else:
            status[fname] = f"manual ({meta.get('url') or meta.get('source_page')})"

    return status


def verify_cache(dest_dir: Path) -> int:
    """Print verification status of all Brainnetome and SUIT cache files."""
    bna_manifest = json.loads(bundled_path("brainnetome_external.json").read_text(encoding="utf-8"))["atlas_files"]
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
            f"  python scripts/download_atlases.py --atlas bna --from-dir <folder> --accept-licenses"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
