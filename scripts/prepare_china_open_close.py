"""Prepare and verify Beijing EOEC (China) Open-Close derivatives from an external download.

The Beijing EOEC derivatives are licensed under Creative Commons
Attribution-NonCommercial (CC BY-NC) and are not bundled in the repository.

Upstream sources:
  - Cohort page: https://fcon_1000.projects.nitrc.org/indi/retro/BeijingEOEC.html
  - Preprocessed archive (Medvedeva et al., 2026): https://disk.yandex.ru/d/fNz33QPwJQj7HQ

Mandatory attribution:
  - Liu D, Dong Z, Zuo X, Wang J, Zang Y (2013). Neuroinformatics 11(4):469-476.
    doi:10.1007/s12021-013-9187-0
  - Funding: NSFC grant 30770594; 863 Program grant 2008AA02Z405.
  - Medvedeva T, Knyazeva I, Masharipov R, Korotkov A, Cherednichenko D,
    Kireev M (2026). Brain Informatics 13:18. doi:10.1186/s40708-026-00305-1

Usage:
    python scripts/prepare_china_open_close.py --source-dir /path/to/downloaded_archive
    python scripts/prepare_china_open_close.py --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "datasets" / "demo" / "open_close"

EXPECTED_FILES = {
    "china_open_Schaefer200_strategy-4_GSR.npy": {
        "sha256": "e0760e21098c50ed6968b4f46fd83e6bf43d237f47dca8c3777981716a33375d",
        "shape": (48, 240, 200),
    },
    "china_close_Schaefer200_strategy-4_GSR.npy": {
        "sha256": "10d836cd84d98408cd0f317887a25751456ab860f40bfccc4ac93f1383787f9e",
        "shape": (48, 240, 200, 2),
    },
    "subject_order_china.txt": {
        "sha256": "b4070b128ebb8b1e8b8e61335cafdb17a39df4978782745d30082f638f5b3134",
        "n_lines": 48,
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_file(path: Path, fname: str, spec: dict) -> None:
    if fname.endswith(".npy"):
        arr = np.load(path, allow_pickle=False)
        if tuple(arr.shape) != spec["shape"]:
            raise ValueError(f"{path}: expected shape {spec['shape']}, got {arr.shape}")
    else:
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) != spec["n_lines"]:
            raise ValueError(f"{path}: expected {spec['n_lines']} subject IDs, got {len(lines)}")
    digest = _sha256(path)
    if digest != spec["sha256"]:
        raise ValueError(f"{path}: SHA-256 mismatch ({digest} != {spec['sha256']})")


def _find_source_file(source_dir: Path, fname: str) -> Optional[Path]:
    direct = source_dir / fname
    if direct.is_file():
        return direct
    matches = sorted(source_dir.rglob(fname))
    return matches[0] if matches else None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=None,
                        help="Directory containing downloaded Beijing EOEC derivative files.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"Destination directory (default: {DEFAULT_DEST}).")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify files in --dest without copying.")
    args = parser.parse_args(argv)

    dest = args.dest.expanduser().resolve()
    if args.verify_only:
        missing = 0
        for fname, spec in EXPECTED_FILES.items():
            target = dest / fname
            if not target.is_file():
                print(f"  [MISSING] {target}")
                missing += 1
            else:
                _verify_file(target, fname, spec)
                print(f"  [OK] {target}")
        return 0 if missing == 0 else 1

    if args.source_dir is None:
        print(
            "Provide --source-dir pointing to your downloaded Beijing EOEC archive.\n"
            "Download links:\n"
            "  - Preprocessed archive: https://disk.yandex.ru/d/fNz33QPwJQj7HQ\n"
            "  - Upstream cohort:      https://fcon_1000.projects.nitrc.org/indi/retro/BeijingEOEC.html\n"
            "Licence: Creative Commons Attribution-NonCommercial (CC BY-NC).\n"
            "See datasets/demo/open_close/NOTICE_BEIJING_EOEC.txt for attribution terms.",
            file=sys.stderr,
        )
        return 2

    src_root = args.source_dir.expanduser().resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    dest.mkdir(parents=True, exist_ok=True)
    for fname, spec in EXPECTED_FILES.items():
        src = _find_source_file(src_root, fname)
        if src is None:
            raise FileNotFoundError(f"Could not find {fname!r} under {src_root}")
        _verify_file(src, fname, spec)
        target = dest / fname
        if src.resolve() != target.resolve():
            shutil.copy2(src, target)
        print(f"  [OK] {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
