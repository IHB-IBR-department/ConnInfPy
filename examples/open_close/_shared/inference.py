"""Paired edge method maps shared by inference and the separate ML transfer workflow."""
from __future__ import annotations
import time
from typing import Dict
import numpy as np
from conninfpy import compute_p_val
from examples.open_close._shared.data import OpenCloseDataset

SELECTORS = {
    "tstat":     dict(method="tstat"),
    "tfnbs":     dict(method="tfnbs", e=0.4, h=3.0, n=10),
    "nbs@2.0":   dict(method="nbs", threshold=2.0),
    "nbs@3.0":   dict(method="nbs", threshold=3.0),
    "cnbs":      dict(method="cnbs"),               # net_labels filled in later
    "ni_tfnbs":  dict(method="ni_tfnbs", e=0.4, h=3.0, n=10),
    "fbc_tfnbs": dict(method="fbc_tfnbs", e=0.4, h=3.0, n=10),          # net_labels filled in later
    "bh_fdr":    dict(method="bh_fdr"),             # parametric, no permutations
}


def compute_selector_pmaps(
    ds: OpenCloseDataset, n_perm: int, seed: int
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run every selector on this cohort; return {method_key: {'g1>g2':.., 'g2>g1':..}}."""
    o_z, c_z = ds.connectivity_z(run=0)
    net_labels = ds.net_labels

    out = {}
    for name, base_kwargs in SELECTORS.items():
        kwargs = dict(base_kwargs)
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
