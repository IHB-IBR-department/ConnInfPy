"""Paired edge method maps shared by inference and the separate ML transfer workflow.

Also hosts the ConnInfPy 2.0.0 inference-contract helpers used by the
``paper_validation`` runner scripts: the ``--tfnbs-n`` / ``--two-tailed`` CLI,
the TFNBS keyword bundle and the resolved-parameter ``*_meta.json`` sidecar, so
every p-map on disk records how it was produced.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from conninfpy import compute_p_val
from examples._shared.paths import ROOT
from examples.open_close._shared.data import OpenCloseDataset

# ConnInfPy 2.0.0 TFNBS contract: exact integration (n=None) inside the
# permutation loop and one two-sided family against the joint max-null.
# ``n=10`` is the legacy pre-2.0.0 grid, reachable through ``--tfnbs-n 10`` for
# the 2x2 (grid x per-tail) control.
TFNBS_200 = {"e": 0.4, "h": 3.0, "n": None, "start_thres": 1.65}
TFNBS_FAMILY = ("tfnbs", "ni_tfnbs", "fbc_tfnbs")

SELECTORS = {
    "tstat":     dict(method="tstat"),
    "tfnbs":     dict(method="tfnbs", **TFNBS_200),
    "nbs@2.0":   dict(method="nbs", threshold=2.0),
    "nbs@3.0":   dict(method="nbs", threshold=3.0),
    "cnbs":      dict(method="cnbs"),               # net_labels filled in later
    "ni_tfnbs":  dict(method="ni_tfnbs", **TFNBS_200),
    "fbc_tfnbs": dict(method="fbc_tfnbs", **TFNBS_200),   # net_labels filled in later
    "bh_fdr":    dict(method="bh_fdr"),             # parametric, no permutations
}


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


def apply_contract(kwargs: dict, tfnbs_n: Optional[int], two_tailed: bool) -> dict:
    """Return ``kwargs`` with the resolved TFNBS ``n`` (TFNBS family only) and ``two_tailed``."""
    out = dict(kwargs)
    if out.get("method") in TFNBS_FAMILY:
        out["n"] = tfnbs_n
    out["two_tailed"] = bool(two_tailed)
    return out


def git_sha() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=str(ROOT)).stdout.strip()
    except Exception:
        return None


def write_metadata(path, **fields) -> dict:
    """Write ``<path>`` (a ``*_meta.json``) with the resolved parameters plus git/python provenance."""
    path = Path(path)
    meta = {"git_sha": git_sha(), "python_version": platform.python_version(),
            "numpy_version": np.__version__, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
    path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"  saved metadata: {path}")
    return meta


def describe_contract(tfnbs_n: Optional[int], two_tailed: bool) -> str:
    kind = "exact integration" if tfnbs_n is None else f"legacy {tfnbs_n}-point grid"
    return f"tfnbs n={tfnbs_n} ({kind}), two_tailed={two_tailed}"


def compute_selector_pmaps(
    ds: OpenCloseDataset,
    n_perm: int,
    seed: int,
    tfnbs_n: Optional[int] = None,
    two_tailed: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run every selector on this cohort; return {method_key: {'g1>g2':.., 'g2>g1':..}}.

    TFNBS-family selectors use ``tfnbs_n`` (None = exact integration) and every
    selector runs with ``two_tailed`` (True = one two-sided family, the 2.0.0 default).
    """
    o_z, c_z = ds.connectivity_z(run=0)
    net_labels = ds.net_labels

    out = {}
    for name, base_kwargs in SELECTORS.items():
        kwargs = apply_contract(base_kwargs, tfnbs_n, two_tailed)
        if kwargs["method"] in ("cnbs", "ni_tfnbs", "fbc_tfnbs"):
            kwargs["net_labels"] = net_labels

        t0 = time.time()
        p = compute_p_val(
            o_z, c_z,
            test_type="paired",
            n_permutations=n_perm,
            use_mp=True,
            random_state=seed,
            **kwargs,
        )
        dt = time.time() - t0
        n_sig = int((p[list(p.keys())[0]][np.triu_indices(ds.n_rois, k=1)] < 0.05).sum())
        print(f"  [{ds.experiment}] {name:10s}  {dt:5.1f}s  n_sig(0.05, g2>g1)={n_sig}")
        out[name] = p
    return out
