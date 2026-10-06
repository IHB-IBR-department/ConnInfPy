"""Progress-reporting helpers for permutation pipelines.

Wraps the permutation pool with an optional progress bar. Uses
``tqdm.auto`` if available; otherwise prints a plain-text update every
``log_every`` permutations. Pass ``verbose=False`` (the default) to
keep the previous silent behaviour.

The helper handles the multiprocessing decision internally so call
sites collapse from ten lines to one.
"""
from __future__ import annotations

import contextlib
import functools
import multiprocessing
import sys
from typing import Any, Callable, Iterator, List, Optional, Sequence, TypeVar

T = TypeVar("T")

try:
    from tqdm.auto import tqdm

    _HAS_TQDM = True
except ImportError:  # pragma: no cover (env-specific)
    _HAS_TQDM = False


def _limit_worker_threads() -> None:
    """Pool initializer: one BLAS/OpenMP thread per worker process.

    The pool already runs one permutation per core. Left alone, every
    worker's BLAS also starts one thread per core, oversubscribing the
    machine (measured on 18 cores: a small TFNBS run took 9.3 s with the
    default threading and 2.6 s with one thread per worker).
    """
    try:
        import numpy  # noqa: F401  (load BLAS first so threadpoolctl sees it)
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover (threadpoolctl is a core dep)
        return
    threadpool_limits(limits=1)


def _references_main(obj: Any, _depth: int = 0) -> bool:
    """True if ``obj`` (a task, its arguments, ...) needs the ``__main__`` module.

    Pickled functions and classes are looked up by module name in the
    worker; anything defined in the caller's script lives in ``__main__``.
    """
    if _depth > 4:
        return False
    if isinstance(obj, functools.partial):
        parts = [obj.func, *obj.args, *(obj.keywords or {}).values()]
        return any(_references_main(p, _depth + 1) for p in parts)
    if isinstance(obj, (list, tuple, set)):
        return any(_references_main(p, _depth + 1) for p in obj)
    if isinstance(obj, dict):
        return any(_references_main(p, _depth + 1) for p in obj.values())
    if callable(obj) and getattr(obj, "__module__", None) == "__main__":
        return True
    return type(obj).__module__ == "__main__"


@contextlib.contextmanager
def _workers_skip_main_script(task_func: Callable[..., Any]) -> Iterator[None]:
    """Start pool workers without re-running the caller's script.

    With the spawn / forkserver start methods every new worker imports the
    main script so it can unpickle functions defined there, which re-runs
    the script's top-level code (the whole analysis) once per worker unless
    the script has an ``if __name__ == "__main__":`` guard. Permutation
    tasks are conninfpy functions and never need the script, so while the
    workers start we hide the script's path from multiprocessing; workers
    then import only conninfpy. Tasks that do reference ``__main__`` (a
    user function passed to ``compute_null_dist``) keep the standard
    behaviour.
    """
    main = sys.modules.get("__main__")
    saved = {k: main.__dict__[k] for k in ("__file__", "__spec__")
             if main is not None and k in main.__dict__}
    if not saved or _references_main(task_func):
        yield
        return
    main.__dict__.pop("__file__", None)
    if "__spec__" in saved:
        main.__spec__ = None
    try:
        yield
    finally:
        main.__dict__.update(saved)


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
    # processes=None lets Pool use every core. Workers start inside the
    # context manager, so they never re-run an unguarded caller script.
    with _workers_skip_main_script(task_func):
        pool = ctx.Pool(processes=n_processes, initializer=_limit_worker_threads)
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
