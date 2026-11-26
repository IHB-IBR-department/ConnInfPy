"""NETSTAT-27 R1-5 round 2: read-only diagnostics, caching preserved on runs.

Owner-side complement to the writer's info() fix: enhancement kernels are
decorated without ``cache=True`` and numba caching is enabled lazily on the
first real enhancement dispatch (``conninfpy/enhance/_backend.py``). These
isolated-subprocess tests prove both halves of the contract:

* ``import conninfpy`` and ``conninfpy info`` never create the configured
  ``NUMBA_CACHE_DIR`` (the parent round-1 process-level probe);
* a real inference run still activates caching and writes cache entries, so
  the user's cache settings and behaviour are unchanged.

Synthetic data only; no real user cache settings are touched.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("numba")


def _run(code, cache_dir, timeout=300):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        cwd=os.getcwd(),
    )


def test_package_import_creates_no_cache_directory(tmp_path):
    cache = tmp_path / "import-cache"
    proc = _run("import conninfpy\nprint('IMPORT_OK')", cache)
    assert proc.returncode == 0, proc.stderr
    assert "IMPORT_OK" in proc.stdout
    assert not cache.exists(), "import conninfpy must not create NUMBA_CACHE_DIR"


def test_info_subprocess_creates_no_cache_directory(tmp_path):
    cache = tmp_path / "info-cache"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    env["NUMBA_CACHE_DIR"] = str(cache)
    proc = subprocess.run(
        [sys.executable, "-m", "conninfpy.cli", "info"],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        cwd=os.getcwd(),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""
    assert not cache.exists(), "conninfpy info must not create NUMBA_CACHE_DIR"
    assert "absent: no cache written" in proc.stdout
    assert "cache hits unknown/not measured" in proc.stdout


def test_real_run_still_enables_caching_and_writes_entries(tmp_path):
    cache = tmp_path / "run-cache"
    code = """
import numpy as np, conninfpy
rng = np.random.default_rng(0)
def stack(n):
    out = np.zeros((n, 6, 6))
    for i in range(n):
        a = rng.standard_normal((6, 12))
        c = np.corrcoef(a)
        np.fill_diagonal(c, 1.0)
        out[i] = c
    return out
res = conninfpy.analyze(group1=stack(6), group2=stack(6), method="tfnbs",
                        n_permutations=8, rng=1, use_mp=False)
print("RUN_OK", res.inference.method)
"""
    proc = _run(code, cache)
    assert proc.returncode == 0, proc.stderr
    assert "RUN_OK" in proc.stdout
    assert cache.is_dir(), "a real run must activate numba caching"
    entries = [
        n
        for _root, _dirs, names in os.walk(cache)
        for n in names
        if n.endswith((".nbi", ".nbc"))
    ]
    assert entries, "a real run must write numba cache entries (caching preserved)"