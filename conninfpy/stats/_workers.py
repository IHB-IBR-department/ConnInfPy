"""Process execution and progress reporting for permutation pipelines.

Wraps the permutation pool with an optional progress bar. Uses
``tqdm.auto`` if available; otherwise prints a plain-text update every
``log_every`` permutations. Pass ``verbose=False`` (the default) to
keep the previous silent behaviour.

The helper handles the multiprocessing decision internally so call
sites collapse from ten lines to one.

The data-bearing task is serialized once into a private temporary file and
each worker loads it from there when it starts. Pool initializer arguments
travel through each worker's start-up channel one worker at a time (spawn /
forkserver), so a large task passed there serialized worker start-up: about
0.8 s per worker for an 11.5 MB task, roughly 15 s for 18 workers. A path is
a few bytes, and the workers read the file from the page cache in parallel.
"""

from __future__ import annotations

import contextlib
import functools
import multiprocessing
import multiprocessing.reduction
import operator
import os
import pickle
import sys
import tempfile
import threading
from typing import Any, Callable, Iterator, List, Optional, Sequence, TypeVar

from conninfpy.stats._engine import _is_worker_process, get_available_cores

T = TypeVar("T")
_main_script_lock = threading.RLock()
_worker_task: Optional[Callable[[int], Any]] = None
_worker_thread_limiter: Any = None

try:
    from tqdm.auto import tqdm

    _HAS_TQDM = True
except ImportError:  # pragma: no cover (env-specific)
    _HAS_TQDM = False


def _limit_worker_threads() -> None:
    """Pool initializer: one BLAS/OpenMP thread per worker process.

    Pool workers provide the parallelism. Keep native libraries from
    independently starting a full thread pool in each worker. The parent
    process's thread settings are left unchanged.
    """
    global _worker_thread_limiter
    try:
        import numpy  # noqa: F401  (load BLAS first so threadpoolctl sees it)
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover (threadpoolctl is a core dep)
        return
    _worker_thread_limiter = threadpool_limits(limits=1)


def _initialize_worker_task(task_path: str) -> None:
    """Load the data-bearing task once per worker from the parent's task file."""
    global _worker_task
    _limit_worker_threads()
    with open(task_path, "rb") as stream:
        _worker_task = pickle.load(stream)


@contextlib.contextmanager
def _task_file(task_func: Callable[..., Any]) -> Iterator[str]:
    """Serialize the task once into a private temporary file; yield its path.

    The file (mode 0600, in ``tempfile.gettempdir()``) is removed when the
    block exits, after the pool has been shut down, on success or failure.
    ``ForkingPickler`` keeps multiprocessing's own reducers, as for initargs.
    """
    fd, path = tempfile.mkstemp(prefix="conninfpy-task-", suffix=".pkl")
    try:
        with os.fdopen(fd, "wb") as stream:
            multiprocessing.reduction.ForkingPickler(stream, pickle.HIGHEST_PROTOCOL).dump(task_func)
        yield path
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


def _run_worker_seed(seed: int) -> Any:
    assert _worker_task is not None
    return _worker_task(seed)


def _worker_count(n_processes: Optional[int], n_jobs: int) -> int:
    """Resolve pool size once for every inference path."""
    if n_processes is None:
        return min(get_available_cores(), n_jobs)
    try:
        count = operator.index(n_processes)
    except TypeError:
        raise ValueError("n_processes must be a positive integer or None.") from None
    if isinstance(n_processes, bool) or count < 1:
        raise ValueError("n_processes must be a positive integer or None.")
    return min(count, n_jobs)


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
    # Background UI threads can start analyses concurrently. Serialize only
    # pool startup so their temporary changes to __main__ cannot interleave.
    with _main_script_lock:
        main = sys.modules.get("__main__")
        saved = {
            k: main.__dict__[k]
            for k in ("__file__", "__spec__")
            if main is not None and k in main.__dict__
        }
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
        whatever the caller's pipeline needs. The task is reused across
        seeds in each process and must not mutate its fixed input data.
    seeds : sequence of int
        Per-permutation seeds (length = n_permutations).
    use_mp : bool, default ``True``
        Use a multiprocessing :class:`Pool`. Disabled automatically
        inside child processes, with one worker, or with fewer than two jobs.
    n_processes : int, optional
        Positive pool size, capped at the number of jobs. ``None`` uses the
        available logical CPU count. Each worker uses one BLAS/OpenMP thread.
    verbose : bool, default ``False``
        Show a progress bar (tqdm) or plain-text update.
    desc : str
        Progress-bar prefix.
    log_every : int
        Plain-text fallback log frequency (only used when tqdm absent).
    """
    n = len(seeds)
    if not n:
        return []
    use_mp = use_mp and not _is_worker_process()
    workers = _worker_count(n_processes, n) if use_mp else 1

    if workers == 1:
        if not verbose:
            return [task_func(s) for s in seeds]
        if _HAS_TQDM:
            return [task_func(s) for s in tqdm(seeds, total=n, desc=desc, leave=False)]
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
    # Only seeds go through the task queue; the partial's data/design arrays
    # are copied once per worker at startup. Balanced chunks also avoid one
    # queue message per permutation when progress reporting is enabled.
    chunksize = max(1, (n + 4 * workers - 1) // (4 * workers))
    with _task_file(task_func) as task_path:
        with _workers_skip_main_script(task_func):
            pool = ctx.Pool(
                processes=workers,
                initializer=_initialize_worker_task,
                initargs=(task_path,),
            )
        with pool as p:
            if not verbose:
                return list(p.map(_run_worker_seed, seeds, chunksize=chunksize))
            if _HAS_TQDM:
                out2: List[T] = []
                for r in tqdm(
                    p.imap(_run_worker_seed, seeds, chunksize=chunksize),
                    total=n,
                    desc=desc,
                    leave=False,
                ):
                    out2.append(r)
                return out2
            out3: List[T] = []
            for i, r in enumerate(
                p.imap(_run_worker_seed, seeds, chunksize=chunksize), start=1
            ):
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
