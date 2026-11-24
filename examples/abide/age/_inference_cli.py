"""ConnInfPy 2.0.0 inference-contract CLI + provenance helpers for the ABIDE runners.

Shared by the ``examples/abide/age`` and ``examples/abide/diagnosis`` scripts so
that ``--tfnbs-n`` / ``--two-tailed`` and the resolved-parameter ``*_meta.json``
sidecars are defined once. It lives under ``age/`` because that is the write
scope of the 2.0.0 re-computation pass; ``examples/abide/_shared/`` is its
natural long-term home.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Optional

# ConnInfPy 2.0.0 TFNBS contract: exact integration (n=None) inside the
# permutation loop and one two-sided family against the joint max-null.
# ``n=10`` is the legacy pre-2.0.0 grid, reachable through ``--tfnbs-n 10`` for
# the 2x2 (grid x per-tail) control.
TFNBS_200 = {"e": 0.4, "h": 3.0, "n": None, "start_thres": 1.65}
TFNBS_FAMILY = ("tfnbs", "ni_tfnbs", "fbc_tfnbs")
# Missing connectivity (constant / all-zero ROI time series) is handled per
# available case by the package; it is never zero-imputed before the analysis.
MISSING_POLICY = "available_case"


def parse_tfnbs_n(text) -> Optional[int]:
    """``none`` / ``null`` / ``exact`` -> None (exact integration); else a grid size such as ``10``."""
    if str(text).strip().lower() in {"none", "null", "exact"}:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid --tfnbs-n {text!r}: use an integer or 'none'") from exc


def add_inference_cli(ap, default_two_tailed: bool = True) -> None:
    """Add ``--tfnbs-n`` and ``--two-tailed`` / ``--one-tailed`` (2.0.0 defaults) to a parser."""
    ap.add_argument("--tfnbs-n", type=parse_tfnbs_n, default=None, metavar="N|none",
                    help="TFNBS threshold grid size; none (default) = exact integration "
                         "(ConnInfPy 2.0.0); 10 = legacy grid for the 2x2 control")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--two-tailed", dest="two_tailed", action="store_true", default=default_two_tailed,
                   help="one two-sided family against the joint max-null (ConnInfPy 2.0.0 default)")
    g.add_argument("--one-tailed", dest="two_tailed", action="store_false",
                   help="legacy per-tail nulls: two one-sided families, each at alpha")


def tfnbs_kwargs(method: str, tfnbs_n: Optional[int], two_tailed: bool) -> dict:
    """Keywords to merge into ``analyze(...)`` / ``compute_p_val(...)``.

    ``two_tailed`` and ``missing='available_case'`` for every method; ``e, h,
    start_thres`` and the resolved ``n`` for the TFNBS family only (``n=None`` =
    exact integration).
    """
    kw = {"two_tailed": bool(two_tailed), "missing": MISSING_POLICY}
    if method in TFNBS_FAMILY:
        kw.update(TFNBS_200, n=tfnbs_n)
    return kw


def describe_contract(tfnbs_n: Optional[int], two_tailed: bool) -> str:
    kind = "exact integration" if tfnbs_n is None else f"legacy {tfnbs_n}-point grid"
    return f"tfnbs n={tfnbs_n} ({kind}), two_tailed={two_tailed}"


def add_run_cli(ap, default_output) -> None:
    """``--n-perm`` and ``--output`` / ``--output-dir`` shared by the production ABIDE runners."""
    ap.add_argument("--n-perm", type=int, default=None,
                    help="Permutations per test (overrides --quick / the 5000 default).")
    ap.add_argument("--output", "--output-dir", dest="output", type=Path, default=default_output,
                    help="Output directory (use a separate one for CC200 runs).")


def resolve_n_perm(args, quick_default: int = 500, default: int = 5000) -> int:
    n = getattr(args, "n_perm", None)
    return int(n) if n is not None else (quick_default if getattr(args, "quick", False) else default)


def git_sha() -> Optional[str]:
    root = Path(__file__).resolve().parents[3]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=str(root)).stdout.strip()
    except Exception:
        return None


def write_metadata(path, script=None, **fields) -> dict:
    """Write ``path`` (a ``*_meta.json``) with the resolved parameters plus git/python provenance."""
    path = Path(path)
    meta = {"git_sha": git_sha(), "python_version": platform.python_version(),
            "script": script, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
    path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"  saved metadata: {path}")
    return meta
