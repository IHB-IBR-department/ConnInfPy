"""Progress-reporting helpers for permutation pipelines.

Wraps the permutation pool with an optional progress bar. Uses
``tqdm.auto`` if available; otherwise prints a plain-text update every
``log_every`` permutations. Pass ``verbose=False`` (the default) to
keep the previous silent behaviour.

The helper handles the multiprocessing decision internally so call
sites collapse from ten lines to one.
"""
from __future__ import annotations

import multiprocessing
import sys
from typing import Callable, List, Optional, Sequence, TypeVar

T = TypeVar("T")

try:
    from tqdm.auto import tqdm

    _HAS_TQDM = True
except ImportError:  # pragma: no cover (env-specific)
    _HAS_TQDM = False


def run_permutations(
    task_func: Callable[[int], T],
    seeds: Sequence[int],
    *,
    use_mp: bool = True,
    n_processes: Optional[int] = None,
    verbose: bool = False,
    desc: str = "perm",
    log_every: int = 50,
) -> List[T]:
    """Apply ``task_func`` to each seed; optionally show a progress bar.

    Parameters
    ----------
    task_func : callable
        Single-permutation worker, taking an int seed and returning
        whatever the caller's pipeline needs.
    seeds : sequence of int
        Per-permutation seeds (length = n_permutations).
    use_mp : bool, default ``True``
        Use a multiprocessing :class:`Pool`. Disabled automatically
        inside worker processes by the caller.
    n_processes : int, optional
        Pool size. ``None`` lets ``Pool`` pick the default.
    verbose : bool, default ``False``
        Show a progress bar (tqdm) or plain-text update.
    desc : str
        Progress-bar prefix.
    log_every : int
        Plain-text fallback log frequency (only used when tqdm absent).
    """
    n = len(seeds)

    if not use_mp:
        if not verbose:
            return [task_func(s) for s in seeds]
        if _HAS_TQDM:
            return [
                task_func(s)
                for s in tqdm(seeds, total=n, desc=desc, leave=False)
            ]
        out: List[T] = []
        for i, s in enumerate(seeds, start=1):
            out.append(task_func(s))
            if i % log_every == 0 or i == n:
                _print_progress(i, n, desc)
        return out

    # Multiprocessing path. Use forkserver (spawn on platforms without it)
    # instead of the default fork start method: the parent process may be
    # multi-threaded (numba JIT threads), and forking a threaded process
    # can deadlock the child (DeprecationWarning since Python 3.12).
    ctx: multiprocessing.context.BaseContext
    try:
        ctx = multiprocessing.get_context("forkserver")
    except ValueError:  # pragma: no cover (platform-specific)
        ctx = multiprocessing.get_context("spawn")
    if n_processes is not None:
        pool = ctx.Pool(processes=n_processes)
    else:
        pool = ctx.Pool()
    with pool as p:
        if not verbose:
            return list(p.map(task_func, seeds))
        if _HAS_TQDM:
            out2: List[T] = []
            for r in tqdm(
                p.imap(task_func, seeds),
                total=n, desc=desc, leave=False,
            ):
                out2.append(r)
            return out2
        out3: List[T] = []
        for i, r in enumerate(p.imap(task_func, seeds), start=1):
            out3.append(r)
            if i % log_every == 0 or i == n:
                _print_progress(i, n, desc)
        return out3


def _print_progress(i: int, n: int, desc: str) -> None:
    """Print an inline ``i/n`` progress line."""
    sys.stdout.write(f"\r{desc} {i}/{n} ({100 * i / n:.0f}%)")
    sys.stdout.flush()
    if i == n:
        sys.stdout.write("\n")


__all__ = ["run_permutations"]
