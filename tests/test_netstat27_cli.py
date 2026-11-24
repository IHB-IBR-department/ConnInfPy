"""NETSTAT-27: ``conninfpy info`` reports resources and honest unknowns.

Isolated subprocess with its own temporary environment: the command must not
create a numba cache for diagnostics, must not change anything on the user's
system, and must print exit code 0 with explicit ``unknown/not measured``
wording where no real record exists.
"""

from __future__ import annotations

import os
import subprocess
import sys


def _run_info(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    # Point numba at a directory that does not exist: diagnostics must not
    # create it, and an absent cache must be reported as absent.
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "numba-cache")
    proc = subprocess.run(
        [sys.executable, "-m", "conninfpy.cli", "info"],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        cwd=os.getcwd(),
    )
    return proc, env


def test_info_exit_code_and_versions(tmp_path):
    proc, _ = _run_info(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""
    text = proc.stdout
    assert "conninfpy" in text and "python" in text and "numpy" in text


def test_info_reports_worker_context_without_starting_a_pool(tmp_path):
    proc, _ = _run_info(tmp_path)
    text = proc.stdout
    assert "workers" in text
    assert "threadpoolctl" in text
    assert "requested" in text
    assert "unknown/not measured" in text


def test_info_cache_presence_is_not_a_hit_and_writes_no_cache(tmp_path):
    """``info`` creates no cache directory, writes no entries, claims no hit.

    NETSTAT-27 R1-5 (round 2): enhancement kernels enable numba caching
    lazily on first real use, so neither the package import nor this command
    creates the configured cache directory. With a nonexistent
    ``NUMBA_CACHE_DIR`` the directory must still be absent afterwards, and
    the output must say so honestly; hits stay unknown/not measured.
    """
    proc, env = _run_info(tmp_path)
    text = proc.stdout
    assert "numba cache" in text
    assert "unknown/not measured" in text
    # Presence of files (or their absence) is never reported as a cache hit.
    assert "cache hits unknown/not measured" in text
    cache_dir = env["NUMBA_CACHE_DIR"]
    assert not os.path.exists(cache_dir), (
        "conninfpy info (including the package import) must not create the "
        "numba cache directory"
    )
    assert "absent: no cache written" in text


def test_info_does_not_guess_the_executed_backend(tmp_path):
    proc, _ = _run_info(tmp_path)
    text = proc.stdout
    # A numba import is not evidence that a JIT kernel ran.
    assert "unknown/not measured" in text
    assert "which kernel actually executed" in text or "not importable" in text
