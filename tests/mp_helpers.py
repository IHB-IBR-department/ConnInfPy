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
