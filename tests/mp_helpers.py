"""Importable, picklable tasks for real process-worker regression checks."""

import os
import time

import numpy as np
from threadpoolctl import threadpool_info


class CountTaskPickles:
    """Record parent-side task serialization without worker-side file writes."""

    def __init__(self, log):
        self.log = str(log)

    def __getstate__(self):
        with open(self.log, "a") as stream:
            stream.write("pickle\n")
        return self.__dict__

    def __call__(self, seed):
        return seed * seed


def worker_threads(seed):
    # Exercise both NumPy and SciPy BLAS before inspecting their limits.
    from scipy.linalg import solve

    solve(np.eye(4), np.ones(4))
    np.eye(4) @ np.ones(4)
    time.sleep(0.01)
    return {"pid": os.getpid(), "seed": seed, "pools": threadpool_info()}


def nested_permutations(seed):
    from functools import partial

    from conninfpy.stats._workers import run_permutations

    return os.getpid(), run_permutations(partial(pow, seed), [1, 2], n_processes=2)


def fail_on_seed(seed):
    if seed == 2:
        raise ValueError("deliberate worker failure")
    return seed


def slow_first_chunk(seed):
    """Boundedness fixture: sleep on the seeds of a size-4 first chunk.

    Module-level constants (not instance state) so forkserver / spawn
    workers see the same schedule after importing this module fresh.
    """
    if seed < 4:
        time.sleep(0.6)
    return seed * seed


def sum_squares_chunk(results):
    """Chunk reduction for the boundedness fixture (partial sum, ndarray)."""
    return np.sum(np.asarray(results, dtype=np.float64))


def nested_reduction(seed):
    """Reduce path nested inside a pool worker must stay serial (no sub-pool)."""
    from functools import partial

    from conninfpy.stats._workers import run_permutations_reduce

    return os.getpid(), run_permutations_reduce(
        partial(pow, seed), [1, 2], reduce_func=sum_squares_chunk, n_processes=2
    )


# -- NETSTAT-27: cancellation fixtures -------------------------------------
def slow_square_logging_pid(log_path, seed):
    """Sleep per seed and log the worker PID (terminate/join assertions).

    The log path is bound with ``functools.partial``, so it travels in the
    picklable task payload itself: forkserver workers reuse a long-lived
    server process whose environment predates the test, so environment
    variables are not a reliable channel. No parent callback or event is ever
    part of the payload.
    """
    try:
        with open(log_path, "a") as stream:
            stream.write(f"{os.getpid()}\n")
    except OSError:
        pass
    time.sleep(0.05)
    return seed * seed
