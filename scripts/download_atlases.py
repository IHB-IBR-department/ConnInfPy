"""Thin wrapper around ``conninfpy download-atlases`` (conninfpy.atlas.download).

Download and verify third-party Brainnetome and SUIT atlases into the local
cache. ConnInfPy's code and Schaefer 2018 tables are MIT-licensed; the
Brainnetome (BNA-246) and DiedrichsenLab SUIT cerebellar atlases are
third-party datasets with separate licences, not bundled in the repository
or wheel.

Prefer the package CLI:
    conninfpy download-atlases --accept-licenses
    conninfpy download-atlases --atlas suit --accept-licenses
    conninfpy download-atlases --atlas bna --from-dir ~/Downloads/bna --accept-licenses
    conninfpy download-atlases --verify-only

This script remains as a compatible entry point with identical arguments.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from conninfpy.atlas.download import main

if __name__ == "__main__":
    raise SystemExit(main())