"""Tests for edge-max permutation equivalence, no-numba fallback, and backend provenance."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from conninfpy import compute_p_val, fisher_r_to_z, generate_fc_matrices
from conninfpy.enhance import (
    apply_cnbs,
    apply_edge_mask,
    apply_fbc_tfnbs,
    apply_nbs,
    apply_ni_tfnbs,
    apply_tfnbs,
)
from conninfpy.enhance._backend import HAS_NUMBA, _enhance_call_scope
import conninfpy.enhance.nbs as nbs_mod
from conninfpy.enhance.nbs import get_nbs_score, nbs_components
from conninfpy.enhance.tfnbs import get_tfnbs_score
from conninfpy.stats._types import enhancement_provenance
import conninfpy.stats.pairwise as pw_mod


class _WarningRecorder(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))


def _sample_data():
    g1, g2, _ = generate_fc_matrices(
        N=14, effect_size=0.4, n_samples_group1=8, n_samples_group2=8, seed=11
    )
    z1 = fisher_r_to_z(g1)
    z2 = fisher_r_to_z(g2)
    net_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    edge_mask = np.ones((14, 14), dtype=bool)
    edge_mask[:3, 10:] = False
    edge_mask[10:, :3] = False
    return z1, z2, net_labels, edge_mask


def test_enhancement_provenance_backward_compatible():
    legacy = enhancement_provenance(0.4, 3.0, None, 1.65, method="tfnbs")
    assert legacy["backend"] == "auto"
    assert "actual_backend" not in legacy

    tracked = enhancement_provenance(
        0.4, 3.0, None, 1.65, method="ni_tfnbs", normalization="sqrt",
        actual_backend="numpy",
    )
    assert tracked["backend"] == "auto"
    assert tracked["actual_backend"] == "numpy"
    assert tracked["normalization"] == "sqrt"


@pytest.mark.parametrize("use_mp", [False, True])
@pytest.mark.parametrize(
    "case_kwargs",
    [
        dict(test_type="two-sample", method="tfnbs"),
        dict(test_type="paired", method="tfnbs", e=[0.0, 0.4], h=[0.0, 3.0]),
        dict(test_type="one-sample", method="tfnbs", two_tailed=True),
        dict(test_type="two-sample", method="ni_tfnbs", normalization="sqrt"),
        dict(test_type="paired", method="ni_tfnbs", normalization="linear", e=[0.0, 0.5], h=[0.0, 2.0]),
        dict(test_type="two-sample", method="fbc_tfnbs", min_cluster_size=2),
        dict(test_type="paired", method="fbc_tfnbs", min_cluster_size=1, e=[0.0, 0.4], h=[0.0, 3.0]),
        dict(test_type="two-sample", method="nbs", threshold=1.8, nbs_stat="extent"),
        dict(test_type="paired", method="nbs", threshold=1.8, nbs_stat="intensity", two_tailed=True),
    ],
)
def test_edge_max_matches_matrix_oracle_and_multiprocessing(case_kwargs, use_mp):
    z1, z2, net_labels, edge_mask = _sample_data()
    kw = dict(
        group1=z1,
        group2=None if case_kwargs["test_type"] == "one-sample" else z2,
        n_permutations=12,
        rng=123,
        use_mp=use_mp,
        n_processes=2 if use_mp else None,
        **case_kwargs,
    )
    if case_kwargs["method"] in ("ni_tfnbs", "fbc_tfnbs"):
        kw["net_labels"] = net_labels
    if case_kwargs["method"] in ("tfnbs", "ni_tfnbs", "nbs") and "e" not in case_kwargs:
        kw["edge_mask"] = edge_mask

    fast_res = compute_p_val(**kw)

    # Compare against forced matrix path (_build_edge_max_spec -> None) in single-process
    kw_oracle = dict(kw, use_mp=False, n_processes=None)
    with patch.object(pw_mod, "_build_edge_max_spec", return_value=None):
        oracle_res = compute_p_val(**kw_oracle)

    np.testing.assert_array_equal(fast_res.positive, oracle_res.positive)
    np.testing.assert_array_equal(fast_res.negative, oracle_res.negative)
    np.testing.assert_array_equal(fast_res.stat_positive, oracle_res.stat_positive)
    np.testing.assert_array_equal(fast_res.stat_negative, oracle_res.stat_negative)

    if case_kwargs["method"] in ("tfnbs", "ni_tfnbs", "fbc_tfnbs"):
        enh = fast_res.design["enhancement"]
        assert enh["backend"] == "auto"
        expected_backend = "numba" if HAS_NUMBA else (
            "python" if case_kwargs["method"] == "tfnbs" else "numpy"
        )
        assert enh["actual_backend"] == expected_backend
    else:
        assert "enhancement" not in fast_res.design


def test_legacy_grid_n_equivalence_and_actual_backend():
    z1, z2, net_labels, _ = _sample_data()
    for method in ("tfnbs", "ni_tfnbs", "fbc_tfnbs"):
        kw = dict(
            group1=z1,
            group2=z2,
            test_type="two-sample",
            method=method,
            n=8,
            n_permutations=10,
            rng=5,
            use_mp=False,
        )
        if method != "tfnbs":
            kw["net_labels"] = net_labels
        res = compute_p_val(**kw)
        enh = res.design["enhancement"]
        assert enh["integration"] == "right-grid"
        assert enh["n"] == 8
        assert enh["backend"] == "auto"
        if method == "tfnbs":
            assert enh["actual_backend"] == ("numba" if HAS_NUMBA else "scipy")
        else:
            assert enh["actual_backend"] == "numpy"


def test_directed_nbs_ties_empty_and_validation_both_backends():
    mat_sym = np.array([
        [0.0, 3.0, 2.0, 0.0, 0.0],
        [3.0, 0.0, 2.0, 1.7, 0.0],
        [2.0, 2.0, 0.0, 1.65, 2.5],
        [0.0, 1.7, 1.65, 0.0, 2.5],
        [0.0, 0.0, 2.5, 2.5, 0.0],
    ])
    mat_dir = mat_sym.copy()
    mat_dir[1, 0] = 2.5
    mat_dir[4, 2] = 1.8

    mat_ties = np.array([
        [0.0, 2.2, 2.2, 0.0, 0.0, 0.0],
        [2.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 2.2, 2.2],
        [0.0, 0.0, 0.0, 2.2, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2.2, 0.0, 0.0],
    ])
    mat_empty = np.zeros((5, 5))

    ref_dir_ext = get_nbs_score(mat_dir, threshold=1.8, stat_type="extent")
    ref_dir_int = get_nbs_score(mat_dir, threshold=1.8, stat_type="intensity")
    ref_comp_ties = nbs_components(mat_ties, threshold=1.8, stat_type="extent")

    # Asymmetric counting check: (0,1) and (1,0) are distinct directed suprathreshold entries
    assert ref_dir_ext[0, 1] == ref_dir_ext[1, 0] == 9.0
    assert ref_dir_ext[2, 4] == 9.0
    assert ref_dir_ext[4, 2] == 0.0  # 1.8 is not > 1.8

    with pytest.raises(ValueError, match="symmetric"):
        nbs_components(mat_dir, threshold=1.8)
    with pytest.raises(ValueError, match="stat_type"):
        get_nbs_score(mat_sym, stat_type="bad")
    with pytest.raises(ValueError, match="square"):
        get_nbs_score(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="Diagonal"):
        get_nbs_score(np.ones((3, 3)))
    with pytest.raises(ValueError, match="backend"):
        get_tfnbs_score(mat_sym, 0.4, 3.0, backend="invalid")

    with patch.object(nbs_mod, "HAS_NUMBA", False):
        np.testing.assert_array_equal(
            get_nbs_score(mat_dir, threshold=1.8, stat_type="extent"), ref_dir_ext
        )
        np.testing.assert_array_equal(
            get_nbs_score(mat_dir, threshold=1.8, stat_type="intensity"), ref_dir_int
        )
        np.testing.assert_array_equal(
            get_nbs_score(mat_empty, threshold=1.8, stat_type="extent"), mat_empty
        )
        comp_scipy = nbs_components(mat_ties, threshold=1.8, stat_type="extent")
        np.testing.assert_array_equal(comp_scipy.labels, ref_comp_ties.labels)
        np.testing.assert_array_equal(comp_scipy.n_edges, ref_comp_ties.n_edges)
        np.testing.assert_array_equal(comp_scipy.statistic, ref_comp_ties.statistic)


def test_apply_wrapper_identities_and_single_call_warning_dedup():
    from conninfpy.enhance import (
        apply_edge_mask as m_mask,
        apply_fbc_tfnbs as m_fbc,
        apply_nbs as m_nbs,
        apply_ni_tfnbs as m_ni,
        apply_tfnbs as m_tfnbs,
    )

    assert m_tfnbs is apply_tfnbs
    assert m_ni is apply_ni_tfnbs
    assert m_fbc is apply_fbc_tfnbs
    assert m_nbs is apply_nbs
    assert m_mask is apply_edge_mask
    assert apply_cnbs is not None

    enh_logger = logging.getLogger("conninfpy.enhance")
    rec = _WarningRecorder()
    enh_logger.addHandler(rec)
    try:
        mat = np.array([
            [0.0, 2.5, 0.0],
            [2.5, 0.0, 2.0],
            [0.0, 2.0, 0.0],
        ])
        stat_dict = {"positive": mat, "negative": mat.copy()}
        with patch.object(nbs_mod, "HAS_NUMBA", False):
            # Two tails (positive + negative) inside one apply_nbs call -> 1 warning
            apply_nbs(stat_dict, threshold=1.8)
            assert len(rec.messages) == 1
            rec.messages.clear()
            # Two separate user calls -> 1 warning each
            with _enhance_call_scope():
                get_nbs_score(mat, threshold=1.8)
                get_nbs_score(mat, threshold=1.8)
            assert len(rec.messages) == 1
    finally:
        enh_logger.removeHandler(rec)


def test_isolated_subprocess_without_numba():
    code = """
import importlib.abc
import io
import logging
import sys
from contextlib import redirect_stdout
import numpy as np

class _BlockNumba(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "numba" or fullname.startswith("numba."):
            raise ImportError("Blocked numba for isolated test")
        return None

sys.meta_path.insert(0, _BlockNumba())

from conninfpy import compute_p_val, generate_fc_matrices, fisher_r_to_z
from conninfpy.cli import main as cli_main
from conninfpy.enhance._backend import HAS_NUMBA
from conninfpy.enhance.tfnbs import get_tfnbs_score
from conninfpy.enhance.ni_tfnbs import get_network_informed_tfnbs_score
from conninfpy.enhance.fbc_tfnbs import get_fbc_tfnbs_score
from conninfpy.enhance.nbs import get_nbs_score

assert HAS_NUMBA is False

buf = io.StringIO()
with redirect_stdout(buf):
    assert cli_main(["info"]) == 0
assert "Python/NumPy/SciPy fallback" in buf.getvalue()

class _CountHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append(self.format(record))

logger = logging.getLogger("conninfpy.enhance")
handler = _CountHandler()
logger.addHandler(handler)
logger.setLevel(logging.WARNING)

g1, g2, _ = generate_fc_matrices(N=10, effect_size=0.4, n_samples_group1=6, n_samples_group2=6, seed=3)
z1, z2 = fisher_r_to_z(g1), fisher_r_to_z(g2)
labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

for method, extra, expected_backend in [
    ("tfnbs", {}, "python"),
    ("tfnbs", {"n": 6}, "scipy"),
    ("ni_tfnbs", {"net_labels": labels}, "numpy"),
    ("fbc_tfnbs", {"net_labels": labels, "min_cluster_size": 2}, "numpy"),
    ("nbs", {"threshold": 1.8}, None),
]:
    handler.records.clear()
    res = compute_p_val(
        z1, z2, test_type="two-sample", method=method,
        n_permutations=15, rng=9, use_mp=False, **extra
    )
    assert len(handler.records) == 1, f"{method} {extra}: {handler.records}"
    if expected_backend is not None:
        assert res.design["enhancement"]["backend"] == "auto"
        assert res.design["enhancement"]["actual_backend"] == expected_backend

# Also verify use_mp=True in no-numba env does not spam parent or fail pickling
handler.records.clear()
res_mp = compute_p_val(
    z1, z2, test_type="paired", method="ni_tfnbs", net_labels=labels,
    n_permutations=10, rng=9, use_mp=True, n_processes=2
)
assert res_mp.design["enhancement"]["actual_backend"] == "numpy"
assert len(handler.records) == 1

# Smoke raw scoring functions without numba
m = np.zeros((4, 4))
m[0, 1] = m[1, 0] = 2.5
l4 = np.array([0, 0, 1, 1])
assert get_tfnbs_score(m, 0.4, 3.0, n=None)[0, 1] > 0
assert get_tfnbs_score(m, 0.4, 3.0, n=5)[0, 1] > 0
assert get_network_informed_tfnbs_score(m, l4, 0.4, 3.0, n=None)[0, 1] > 0
assert get_fbc_tfnbs_score(m, l4, 0.4, 3.0, n=None, min_cluster_size=1)[0, 1] > 0
assert get_nbs_score(m, threshold=1.8)[0, 1] == 1.0
print("ISOLATED_NO_NUMBA_OK")
"""
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        }
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert "ISOLATED_NO_NUMBA_OK" in proc.stdout
