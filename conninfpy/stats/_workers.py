"""Process execution and progress reporting for permutation pipelines.

Wraps the permutation pool with an optional progress bar. Uses
``tqdm.auto`` if available; otherwise prints a plain-text update every
``log_every`` permutations. Pass ``verbose=False`` (the default) to
keep the previous silent behaviour.

The helper handles the multiprocessing decision internally so call
sites collapse from ten lines to one.

``run_permutations_reduce`` is a streaming variant for permutation
p-value counting: workers fold a chunk of per-seed results into a partial
sum (e.g. per-edge exceedance counts) and only that partial travels back
to the parent, so the parent never holds ``B`` per-seed results. The
number of issued-but-unfolded chunks is bounded by construction at
``2 * workers`` (bounded ``apply_async`` issuance: a new chunk is issued
only after a previous one has been folded), and inside a chunk workers
fold in batches of ``_FOLD_BATCH`` seeds. Because every per-seed
contribution is a 0/1 indicator summed as float64 (exact below 2**53)
and chunks fold in seed order, the totals are bitwise identical to a
serial fold over the same seeds.

Progress and cancellation (NETSTAT-27) are parent-side only: the parent
reports ``(done, total)`` at chunk boundaries for permutations that were
actually folded, and checks a parent-side cancel callable before handing
out the next chunk. Nothing but ordinary per-seed / per-chunk results
crosses back from the workers, and no callback, event or UI object is ever
serialized into a worker (see :class:`ProgressSession` and
:func:`prepare_run`).

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
import logging
import multiprocessing
import multiprocessing.reduction
import operator
import os
import pickle
import sys
import tempfile
import threading
from collections import deque
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, TypeVar

from conninfpy.stats._engine import _is_worker_process, get_available_cores

T = TypeVar("T")
_logger = logging.getLogger(__name__)
_main_script_lock = threading.RLock()
_worker_task: Optional[Callable[[int], Any]] = None
_worker_thread_limiter: Any = None

# Per-seed results inside a chunk fold are processed in batches of this many
# seeds: peak live memory stays at O(batch * per-seed result) in the worker
# (and in the parent's serial path) instead of O(chunk * per-seed result).
# Summed 0/1 contributions are exact regardless of batch grouping.
_FOLD_BATCH = 32

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


def _fold_seeds_in_batches(
    task: Callable[[int], Any],
    chunk_seeds: List[int],
    reduce_func: Callable[[List[Any]], Any],
) -> Any:
    """Fold a chunk's seeds in order, in batches of ``_FOLD_BATCH`` seeds.

    The batch of per-seed results is dropped (``del``) right after it is
    reduced, before the next batch is built, so at most ``_FOLD_BATCH``
    per-seed results are alive at once. The only other transient is the
    accumulator fold ``total + partial`` — two ``O(E)`` reduction-size
    arrays, never per-seed data.
    """
    total: Any = None
    for start in range(0, len(chunk_seeds), _FOLD_BATCH):
        batch = [task(seed) for seed in chunk_seeds[start : start + _FOLD_BATCH]]
        partial = reduce_func(batch)
        del batch
        total = partial if total is None else total + partial
    return total


def _run_worker_chunk_reduce(
    chunk_seeds: List[int], reduce_func: Callable[[List[Any]], Any]
) -> Any:
    """Fold one chunk of seeds inside the worker; return only the partial sum.

    The per-seed results never leave the worker process, and they are folded
    in batches of ``_FOLD_BATCH`` seeds, so the worker never holds a whole
    chunk of per-seed results either. The returned value is a ``reduce_func``
    partial reduction over the chunk's seeds (in order).
    """
    assert _worker_task is not None
    return _fold_seeds_in_batches(_worker_task, chunk_seeds, reduce_func)


def _result_nbytes(partial: Any) -> int:
    """Approximate parent-side footprint of one chunk reduction."""
    nbytes = getattr(partial, "nbytes", None)
    return int(nbytes) if nbytes is not None else 0


class _ReductionStats:
    """Counters proving the structural bound on unreduced chunk results.

    ``issued`` counts chunks handed to the pool (or the serial loop) whose
    result has not been folded yet; ``arrived``/``arrived_bytes`` count
    results that have reached the parent but are not folded yet (mp only,
    tracked from the pool's result-handler thread). ``limit`` is the
    structural bound the implementation must never exceed.
    """

    def __init__(self, limit: int) -> None:
        self._lock = threading.Lock()
        self.limit = limit
        self.issued = 0
        self.arrived = 0
        self.arrived_bytes = 0
        self.max_issued = 0
        self.max_arrived = 0
        self.max_arrived_bytes = 0
        self.max_result_bytes = 0
        self.chunks_issued = 0
        self.chunks_folded = 0

    def on_issue(self) -> None:
        with self._lock:
            self.issued += 1
            self.chunks_issued += 1
            self.max_issued = max(self.max_issued, self.issued)
            assert self.issued <= self.limit, "bounded issuance exceeded"

    def on_arrival(self, nbytes: int) -> None:
        with self._lock:
            self.arrived += 1
            self.arrived_bytes += nbytes
            self.max_result_bytes = max(self.max_result_bytes, nbytes)
            self.max_arrived = max(self.max_arrived, self.arrived)
            self.max_arrived_bytes = max(self.max_arrived_bytes, self.arrived_bytes)
            assert self.arrived <= self.limit, "bounded buffer exceeded"

    def on_fold(self, nbytes: int) -> None:
        with self._lock:
            self.issued -= 1
            self.arrived -= 1
            self.arrived_bytes -= nbytes
            self.chunks_folded += 1

    def snapshot(self, mode: str, workers: int, chunk_size: int, n_chunks: int) -> Dict[str, Any]:
        with self._lock:
            return {
                "mode": mode,
                "workers": workers,
                "chunk_size": chunk_size,
                "n_chunks": n_chunks,
                "issued_outstanding_limit": self.limit,
                "max_issued_outstanding": self.max_issued,
                "max_arrived_outstanding": self.max_arrived,
                "max_arrived_outstanding_bytes": self.max_arrived_bytes,
                "max_chunk_result_bytes": self.max_result_bytes,
                "chunks_issued": self.chunks_issued,
                "chunks_folded": self.chunks_folded,
            }


class InferenceCancelled(RuntimeError):
    """A permutation pipeline was cancelled cooperatively by its caller.

    Raised by the engine at a permutation chunk boundary (never from inside
    a worker), so no partial reduction, no valid
    :class:`~conninfpy.results.InferenceResult` and no p-values are produced
    from an incomplete null distribution.
    """


ProgressCallback = Callable[[int, int], None]
CancelCallback = Callable[[], bool]


class ProgressSession:
    """Parent-side progress/cancel aggregator for one analysis call.

    A session is created by the caller (``analyze()`` or a UI job) and
    passed down through the pipelines as ``progress=session`` and
    ``cancel=<callable>``. Each permutation loop announces itself to the
    session before it runs (see :func:`prepare_run`), so the ``total`` the
    session reports is the number of permutation executions planned so far
    across every loop/stage of the call — never an invented figure.

    Contract
    --------
    ``callback(done, total)`` is called in the parent process only, at chunk
    boundaries for permutations that were actually folded:

    * ``done`` is monotonic and never exceeds ``total``;
    * ``total`` is monotonic non-decreasing and equals the sum of the
      permutation executions announced so far. For every current pipeline
      a single permutation loop of ``n_permutations`` seeds is run per
      ``analyze()`` call, and that loop is announced before its first seed,
      so ``total`` is exact from the first callback. Pipelines that run
      several loops (e.g. the available-case multi-contrast route, which
      runs one loop per contrast) grow ``total`` as each loop starts;
      ``done == total`` is only reported when all planned work has been
      redeemed and fully reduced;
    * the last callback of a successful call has ``done == total``.

    Cancellation is a parent-side decision: ``cancel`` is a zero-argument
    callable evaluated in the parent process at chunk boundaries. A
    :class:`threading.Event` (or any UI callback) is therefore never
    pickled into a worker task payload. When it returns true the engine
    stops issuing chunks, terminates and joins the pool, removes its
    private task file and raises :class:`InferenceCancelled`; the last
    ``done`` is not forced to ``total``.
    """

    def __init__(
        self,
        callback: Optional[ProgressCallback] = None,
        cancel: Optional[CancelCallback] = None,
    ) -> None:
        self._callback = callback
        self._cancel = cancel
        self._lock = threading.RLock()
        self._total = 0
        self._done = 0
        self._stage: Optional[str] = None
        self._run_offset = 0
        self._run_size = 0
        self._pending: List[tuple[int, Optional[str]]] = []
        self.cancelled_early = False
        self.unused_plan = 0
        self.last: tuple[int, int] = (0, 0)

    # -- planning ------------------------------------------------------
    def expect(self, n: int, label: Optional[str] = None) -> None:
        """Announce an upcoming loop of ``n`` executions before it starts.

        Used by callers that know a later stage will run (a sensitivity
        companion) so the reported ``total`` is exact from the first
        callback instead of growing when that stage starts. The next
        :func:`prepare_run` of the same size redeems the announcement
        instead of adding it twice.
        """
        n = int(n)
        if n <= 0:
            return
        with self._lock:
            self._total += n
            self._pending.append((n, label))
            done, total = self._done, self._total
        self._emit(done, total)

    def _plan(self, n: int, label: Optional[str] = None) -> None:
        """Start a loop of ``n`` executions, redeeming an announcement if any."""
        n = int(n)
        with self._lock:
            announced: Optional[str] = None
            for i, (size, lab) in enumerate(self._pending):
                if size == n:
                    _, announced = self._pending.pop(i)
                    break
            else:
                self._total += n
            self._run_offset = self._done
            self._run_size = n
            stage = label or announced
            if stage:
                self._stage = stage
            done, total = self._done, self._total
        self._emit(done, total)

    def _advance(self, done_local: int) -> None:
        with self._lock:
            done = self._run_offset + max(0, min(int(done_local), self._run_size))
            # Persist the high-water mark so a later finalize() (e.g. on
            # cancellation) reports the actually folded count, not a stale 0.
            self._done = max(self._done, done)
            total = self._total
        self._emit(done, total)

    def _finish_run(self) -> None:
        with self._lock:
            self._done = self._run_offset + self._run_size
            done, total = self._done, self._total
        self._emit(done, total)

    def finalize(self) -> None:
        """Drop announcements that were never redeemed, then report.

        Honest final state: an announced stage that did not run is removed
        from ``total`` (recorded in ``unused_plan``) rather than being left
        as a permanent shortfall.
        """
        with self._lock:
            if self._pending:
                dropped = sum(size for size, _ in self._pending)
                self._pending.clear()
                self.unused_plan += dropped
                self._total = max(self._done, self._total - dropped)
            done, total = self._done, self._total
        self._emit(done, total)

    # -- state ----------------------------------------------------------
    @property
    def done(self) -> int:
        with self._lock:
            return self._done

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    @property
    def stage(self) -> Optional[str]:
        with self._lock:
            return self._stage

    def snapshot(self) -> Dict[str, Any]:
        """Plain-data view for run plans and tests (no callables)."""
        with self._lock:
            return {
                "done": self._done,
                "total": self._total,
                "stage": self._stage,
                "unused_plan": self.unused_plan,
            }

    # -- cancellation ---------------------------------------------------
    def cancelled(self) -> bool:
        """True if the caller asked to cancel (parent-side check)."""
        if self._cancel is None:
            return False
        return bool(self._cancel())

    def check_cancel(self) -> None:
        """Raise :class:`InferenceCancelled` if cancellation was requested."""
        if self.cancelled():
            raise InferenceCancelled(
                "inference cancelled by the caller at a permutation chunk boundary"
            )

    def _emit(self, done: int, total: int) -> None:
        self.last = (done, total)
        if self._callback is None:
            return
        # Never call user code (Streamlit, tqdm, ...) while holding our lock.
        self._callback(done, total)


class RunProgress:
    """Per-run progress/cancel channel handed to the permutation leaves."""

    def __init__(
        self,
        session: Optional[ProgressSession],
        callback: Optional[ProgressCallback],
        cancel: Optional[CancelCallback],
        total: int,
    ) -> None:
        self._session = session
        self._callback = callback
        self._cancel = cancel
        self.total = int(total)
        self.done = 0

    def update(self, done: int, total: Optional[int] = None) -> None:
        self.done = int(done)
        if self._session is not None:
            self._session._advance(done)
        elif self._callback is not None:
            self._callback(int(done), self.total if total is None else int(total))

    def cancelled(self) -> bool:
        return bool(self._cancel()) if self._cancel is not None else False

    def check_cancel(self) -> None:
        if self.cancelled():
            raise InferenceCancelled(
                "inference cancelled by the caller at a permutation chunk boundary"
            )

    def finish(self) -> None:
        """Mark this loop fully reduced (only called after a clean return)."""
        if self._session is not None:
            self._session._finish_run()
        elif self._callback is not None:
            self._callback(self.total, self.total)


def prepare_run(
    progress: Any,
    cancel: Optional[CancelCallback],
    total: int,
    label: Optional[str] = None,
) -> RunProgress:
    """Build the per-run progress/cancel channel used by the permutation leaves.

    ``progress`` may be a :class:`ProgressSession` (aggregated over all
    loops of one call), a plain ``(done, total)`` callable (reported per
    loop), or ``None``. ``cancel`` is only consulted in the parent process;
    it is never serialized.
    """
    total = int(total)
    session = progress if isinstance(progress, ProgressSession) else None
    callback = None
    if session is not None:
        if total > 0:
            session._plan(total, label)
        cancel_fn: Optional[CancelCallback] = session.cancelled
    else:
        callback = progress if callable(progress) else None
        cancel_fn = (lambda: bool(cancel())) if cancel is not None else None
    if cancel is not None and session is not None:
        cancel_fn = session.cancelled
    return RunProgress(session, callback, cancel_fn, total)


def _shutdown_pool(pool: Any) -> None:
    """Terminate and join a pool; cleanup failures are logged, never hidden."""
    try:
        pool.terminate()
    except Exception as exc:  # pragma: no cover - defensive
        _logger.warning("pool terminate failed during cleanup: %s", exc)
    try:
        pool.join()
    except Exception as exc:  # pragma: no cover - defensive
        _logger.warning("pool join failed during cleanup: %s", exc)


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


def describe_worker_policy(
    *,
    use_mp: bool = True,
    n_processes: Optional[int] = None,
    n_permutations: Optional[int] = None,
) -> Dict[str, Any]:
    """Requested vs resolved worker/thread facts, with honest unknowns.

    Single source of truth for the run plan and ``conninfpy info``. The
    requested values are what the caller asked for; the effective worker
    count is what the engine would resolve for ``n_permutations`` jobs
    (``None`` when the number of jobs is unknown, so the cap cannot be
    applied). Which BLAS/OpenMP library and how many native threads the
    parent process actually runs are *not* measured here and are reported
    as ``unknown/not measured`` rather than guessed.
    """
    available = get_available_cores()
    workers: Optional[int]
    error: Optional[str] = None
    if not use_mp:
        workers = 1
    elif n_permutations is None:
        workers = None
    else:
        try:
            workers = _worker_count(n_processes, max(1, int(n_permutations)))
        except (TypeError, ValueError) as exc:
            workers = None
            error = str(exc)
    try:
        start_method = multiprocessing.get_context("forkserver").get_start_method()
    except ValueError:  # pragma: no cover (platform-specific)
        start_method = multiprocessing.get_context("spawn").get_start_method()
    return {
        "use_mp_requested": bool(use_mp),
        "n_processes_requested": n_processes,
        "n_permutations": None if n_permutations is None else int(n_permutations),
        "workers_effective": workers,
        "workers_default_budget": available,
        "worker_start_method": start_method,
        "blas_threads_per_worker": 1 if (workers is None or workers > 1) else "parent default (unchanged)",
        "thread_limiter": (
            "threadpoolctl limits=1 in each worker"
            if (workers is None or workers > 1)
            else "not applied (no pool)"
        ),
        "blas_library": "unknown/not measured",
        "enhancement_backend": "unknown/not measured",
        "request_error": error,
    }


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
    progress: Optional[ProgressCallback] = None,
    cancel: Optional[CancelCallback] = None,
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
    progress : callable, optional
        Parent-side ``progress(done, total)`` callback, invoked in the parent
        process only, at chunk boundaries, for permutations already
        completed. ``total`` is this run's ``len(seeds)``; pass a
        :class:`ProgressSession` through :func:`prepare_run` to aggregate
        several loops. Nothing is sent to or called inside the workers.
    cancel : callable, optional
        Zero-argument parent-side predicate. Evaluated before each chunk is
        handed out; when it returns true the engine stops issuing chunks,
        joins the pool and raises :class:`InferenceCancelled`. Cancellation
        is cooperative: at most one in-flight chunk per worker is allowed to
        finish before the pool is terminated.
    """
    n = len(seeds)
    if not n:
        return []

    def _check_cancel() -> None:
        if cancel is not None and cancel():
            raise InferenceCancelled(
                "inference cancelled by the caller at a permutation chunk boundary"
            )

    _check_cancel()
    use_mp = use_mp and not _is_worker_process()
    workers = _worker_count(n_processes, n) if use_mp else 1
    bar = _make_chunk_progress(n, verbose, desc, log_every)

    def _emit(done: int) -> None:
        bar(done)
        if progress is not None:
            progress(done, n)

    if workers == 1:
        out: List[T] = []
        for i, s in enumerate(seeds, start=1):
            _check_cancel()
            out.append(task_func(s))
            if verbose or progress is not None:
                _emit(i)
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
        out2: List[T] = []
        try:
            iterator = pool.imap(_run_worker_seed, seeds, chunksize=chunksize)
            done = 0
            while True:
                _check_cancel()  # before blocking on the next completed chunk
                try:
                    result = next(iterator)
                except StopIteration:
                    break
                out2.append(result)
                done += 1
                if done % chunksize == 0 or done == n:
                    _emit(done)
        finally:
            # Always terminate + join, on success, error and cancellation:
            # no worker process or result-handler thread is left behind.
            _shutdown_pool(pool)
        return out2



def _make_chunk_progress(
    n: int, verbose: bool, desc: str, log_every: int
) -> Callable[[int], None]:
    """Per-permutation progress advanced once per folded chunk."""
    if not verbose:
        return lambda done: None
    if _HAS_TQDM:
        bar = tqdm(total=n, desc=desc, leave=False)
        state = {"done": 0}

        def update(done: int) -> None:
            bar.update(done - state["done"])
            state["done"] = done

        return update

    state = {"next": min(log_every, n)}

    def update(done: int) -> None:
        if done >= state["next"] or done == n:
            _print_progress(done, n, desc)
            state["next"] = done + log_every

    return update


def run_permutations_reduce(
    task_func: Callable[[int], T],
    seeds: Sequence[int],
    *,
    reduce_func: Callable[[List[T]], Any],
    use_mp: bool = True,
    n_processes: Optional[int] = None,
    verbose: bool = False,
    desc: str = "perm",
    log_every: int = 50,
    stats_out: Optional[Dict[str, Any]] = None,
    progress: Optional[ProgressCallback] = None,
    cancel: Optional[CancelCallback] = None,
):
    """Map ``task_func`` over seeds with worker-side chunk reduction.

    Instead of returning one result per seed, seeds are processed in chunks;
    each worker folds its chunk's per-seed results with ``reduce_func`` and
    returns only the partial reduction (e.g. a per-edge count vector). The
    parent sums the partials in seed order. This keeps ``B`` per-seed
    results out of the parent: peak parent memory is ``O(reduction size)``
    plus at most ``2 * workers`` outstanding chunk results.

    Parameters
    ----------
    task_func : callable
        Single-permutation worker, as in :func:`run_permutations`. The task
        still travels through the private task file, once per worker.
    seeds : sequence of int
        Per-permutation seeds (length = n_permutations).
    reduce_func : callable
        ``list_of_per_seed_results -> partial reduction`` (an ndarray, or
        something supporting in-place ``+=`` with other chunk reductions).
        Must be defined at module level so pool workers can unpickle it by
        reference (spawn / forkserver). Called once per chunk, in seed order.
        With per-seed 0/1 float64 contributions the summed counts are exact
        (well below 2**53), hence bitwise identical to a serial fold.
    use_mp, n_processes, verbose, desc, log_every
        As in :func:`run_permutations`. Progress counts permutations, not
        chunks; it advances as each chunk completes.
    stats_out : dict, optional
        Written with instrumentation counters: ``max_issued_outstanding``
        (peak chunks issued but not folded; structurally ≤
        ``issued_outstanding_limit`` = ``2 * workers``, or 1 serial),
        ``max_arrived_outstanding`` / ``max_arrived_outstanding_bytes``
        (peak parent-held unreduced results and their bytes), chunking
        details. Intended for tests and memory audits.
    progress, cancel
        Parent-side ``progress(done, total)`` callback at folded-chunk
        boundaries and a zero-argument cancel predicate, exactly as in
        :func:`run_permutations`. Cancellation stops issuing chunks, drops
        the in-flight partials (no partial reduction is returned or
        published), terminates and joins the pool and raises
        :class:`InferenceCancelled`.

    Returns
    -------
    The accumulated reduction over all seeds (``None`` for empty ``seeds``).
    Worker exceptions propagate; the pool and task file are cleaned up on
    success, failure and cancellation exactly as in :func:`run_permutations`.
    """
    n = len(seeds)
    if not n:
        return None

    def _check_cancel() -> None:
        if cancel is not None and cancel():
            raise InferenceCancelled(
                "inference cancelled by the caller at a permutation chunk boundary"
            )

    _check_cancel()
    use_mp = use_mp and not _is_worker_process()
    workers = _worker_count(n_processes, n) if use_mp else 1
    # Same chunk shape as run_permutations' balanced pool chunks.
    chunk_size = max(1, (n + 4 * workers - 1) // (4 * workers))
    chunks = [list(seeds[i : i + chunk_size]) for i in range(0, n, chunk_size)]
    bar = _make_chunk_progress(n, verbose, desc, log_every)
    total: Any = None

    def _emit(done: int) -> None:
        bar(done)
        if progress is not None:
            progress(done, n)

    if workers == 1:
        stats = _ReductionStats(limit=1)
        done = 0
        for chunk in chunks:
            _check_cancel()
            stats.on_issue()
            partial = _fold_seeds_in_batches(task_func, list(chunk), reduce_func)
            stats.on_arrival(_result_nbytes(partial))
            total = partial if total is None else total + partial
            stats.on_fold(_result_nbytes(partial))
            done += len(chunk)
            _emit(done)
        if stats_out is not None:
            stats_out.update(stats.snapshot("serial", 1, chunk_size, len(chunks)))
        return total

    # Multiprocessing path: identical pool setup and start-method policy as
    # run_permutations. Structural memory bound: a chunk is issued only
    # while fewer than 2 * workers issued-but-unfolded chunks exist, so at
    # most that many partial reductions can ever sit in the parent.
    ctx: multiprocessing.context.BaseContext
    try:
        ctx = multiprocessing.get_context("forkserver")
    except ValueError:  # pragma: no cover (platform-specific)
        ctx = multiprocessing.get_context("spawn")
    chunk_limit = 2 * workers
    stats = _ReductionStats(limit=chunk_limit)
    pending: deque = deque()
    with _task_file(task_func) as task_path:
        with _workers_skip_main_script(task_func):
            pool = ctx.Pool(
                processes=workers,
                initializer=_initialize_worker_task,
                initargs=(task_path,),
            )
        try:
            next_chunk = 0
            done = 0

            def _issue() -> None:
                nonlocal next_chunk
                stats.on_issue()
                result = pool.apply_async(
                    _run_worker_chunk_reduce,
                    (chunks[next_chunk], reduce_func),
                    callback=lambda partial: stats.on_arrival(_result_nbytes(partial)),
                )
                next_chunk += 1
                pending.append(result)

            for _ in range(len(chunks)):
                _check_cancel()
                while next_chunk < len(chunks) and len(pending) < chunk_limit:
                    _issue()
                _check_cancel()
                partial = pending.popleft().get()  # propagates worker errors
                stats.on_fold(_result_nbytes(partial))
                total = partial if total is None else total + partial
                done += len(chunks[stats.chunks_folded - 1])
                _emit(done)
        finally:
            _shutdown_pool(pool)

    if stats_out is not None:
        stats_out.update(stats.snapshot("mp", workers, chunk_size, len(chunks)))
    return total


def _print_progress(i: int, n: int, desc: str) -> None:
    """Print an inline ``i/n`` progress line."""
    sys.stdout.write(f"\r{desc} {i}/{n} ({100 * i / n:.0f}%)")
    sys.stdout.flush()
    if i == n:
        sys.stdout.write("\n")


__all__ = [
    "run_permutations",
    "run_permutations_reduce",
    "InferenceCancelled",
    "ProgressSession",
    "RunProgress",
    "prepare_run",
    "describe_worker_policy",
]
