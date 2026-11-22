"""Process lifecycle, thread limits, task transport and worker-count policy."""

import functools
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import pickle
import sys
from unittest.mock import patch

import numpy as np
import pytest
from threadpoolctl import threadpool_info

from conninfpy.stats import _workers
from conninfpy.stats import _engine
from conninfpy.stats._workers import run_permutations_reduce
from tests.mp_helpers import (
    CountTaskPickles,
    fail_on_seed,
    nested_permutations,
    nested_reduction,
    sum_squares_chunk,
    worker_threads,
)

START_METHODS = [
    name
    for name in ("spawn", "forkserver")
    if name in multiprocessing.get_all_start_methods()
]


@pytest.fixture(params=START_METHODS)
def worker_context(request):
    # Force both supported safe contexts, rather than only this OS's default.
    ctx = multiprocessing.get_context(request.param)
    with patch.object(_workers.multiprocessing, "get_context", return_value=ctx):
        yield ctx


def test_empty_jobs_do_not_start_workers():
    with patch.object(_workers.multiprocessing, "get_context") as factory:
        assert _workers.run_permutations(abs, []) == []
    factory.assert_not_called()


@pytest.mark.parametrize("count", [1, np.int64(1)])
def test_one_worker_uses_serial_execution(count):
    with patch.object(_workers.multiprocessing, "get_context") as factory:
        assert _workers.run_permutations(abs, [-3, -1], n_processes=count) == [3, 1]
    factory.assert_not_called()


@pytest.mark.parametrize("count", [0, -2, 1.5, True, "2"])
def test_invalid_worker_count_is_rejected(count):
    with pytest.raises(ValueError, match="positive integer"):
        _workers.run_permutations(abs, [-1, -2], n_processes=count)


@pytest.mark.parametrize("verbose", [False, True])
def test_task_is_serialized_once_for_all_workers(worker_context, tmp_path, verbose):
    log = tmp_path / "pickles.txt"
    out = _workers.run_permutations(
        CountTaskPickles(log),
        list(range(32)),
        n_processes=2,
        verbose=verbose,
    )
    assert out == [seed * seed for seed in range(32)]
    assert log.read_text().splitlines() == ["pickle"]


def test_worker_startup_carries_only_the_task_path(worker_context, tmp_path):
    # A large task must not travel through the per-worker start-up channel,
    # which spawn / forkserver serve one worker at a time.
    payload = np.arange(2_000_000, dtype=np.float64)                  # 16 MB
    task = functools.partial(np.take, payload)
    with patch.object(_workers.tempfile, "tempdir", str(tmp_path)):
        with patch.object(worker_context, "Pool", wraps=worker_context.Pool) as pool:
            out = _workers.run_permutations(task, [0, 5, 1_999_999], n_processes=2)
    assert out == [0.0, 5.0, 1_999_999.0]
    (path,) = pool.call_args.kwargs["initargs"]
    assert isinstance(path, str) and len(pickle.dumps(pool.call_args.kwargs["initargs"])) < 1024
    assert list(tmp_path.iterdir()) == []                              # task file removed


def test_task_file_is_removed_after_a_worker_failure(worker_context, tmp_path):
    with patch.object(_workers.tempfile, "tempdir", str(tmp_path)):
        with pytest.raises(ValueError, match="deliberate worker failure"):
            _workers.run_permutations(fail_on_seed, [1, 2, 3], n_processes=2)
    assert list(tmp_path.iterdir()) == []


def test_thread_caps_apply_only_in_workers(worker_context):
    before = threadpool_info()
    out = _workers.run_permutations(worker_threads, list(range(8)), n_processes=2)
    assert [r["seed"] for r in out] == list(range(8))
    assert all(r["pools"] for r in out)
    assert all(p["num_threads"] == 1 for r in out for p in r["pools"])
    assert threadpool_info() == before


def test_default_pool_size_is_capped_by_job_count(worker_context):
    with patch.object(_workers, "get_available_cores", return_value=100):
        with patch.object(worker_context, "Pool", wraps=worker_context.Pool) as pool:
            assert _workers.run_permutations(abs, [-1, -2]) == [1, 2]
    assert pool.call_args.kwargs["processes"] == 2


def test_nested_worker_runs_serially(worker_context):
    out = _workers.run_permutations(nested_permutations, [2, 3], n_processes=2)
    assert [r[1] for r in out] == [[2, 4], [3, 9]]


def test_worker_error_cleans_up_and_next_pool_succeeds(worker_context):
    main_before = dict(sys.modules["__main__"].__dict__)
    children_before = {p.pid for p in multiprocessing.active_children()}
    with pytest.raises(ValueError, match="deliberate worker failure"):
        _workers.run_permutations(fail_on_seed, [1, 2, 3], n_processes=2)
    assert {p.pid for p in multiprocessing.active_children()} == children_before
    main = sys.modules["__main__"].__dict__
    for key in ("__file__", "__spec__"):
        assert (key in main, main.get(key)) == (
            key in main_before,
            main_before.get(key),
        )
    assert _workers.run_permutations(
        functools.partial(pow, 2), [1, 2], n_processes=2
    ) == [2, 4]


def test_plain_progress_fallback(worker_context, capsys):
    with patch.object(_workers, "_HAS_TQDM", False):
        assert _workers.run_permutations(
            abs,
            [-1, -2, -3],
            n_processes=2,
            verbose=True,
            log_every=2,
        ) == [1, 2, 3]
    assert "3/3 (100%)" in capsys.readouterr().out


def test_concurrent_background_threads_restore_main(worker_context):
    main = sys.modules["__main__"].__dict__
    before = {k: main[k] for k in ("__file__", "__spec__") if k in main}
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [
            executor.submit(
                _workers.run_permutations,
                functools.partial(pow, base),
                [1, 2, 3],
                n_processes=2,
            )
            for base in (2, 3)
        ]
        assert [job.result(timeout=30) for job in jobs] == [[2, 4, 8], [3, 9, 27]]
    assert {k: main[k] for k in ("__file__", "__spec__") if k in main} == before


def test_cpu_count_uses_process_limits_when_available():
    with patch.object(_engine, "_cpu_count", return_value=12):
        with patch.object(_engine.os, "process_cpu_count", return_value=3, create=True):
            assert _engine.get_available_cores() == 3


@pytest.mark.parametrize("process_limit", [None, 12])
def test_cpu_count_respects_available_cpu_budget(process_limit):
    with patch.object(_engine, "_cpu_count", return_value=2):
        with patch.object(
            _engine.os, "process_cpu_count", return_value=process_limit, create=True
        ):
            assert _engine.get_available_cores() == 2


def test_cpu_count_is_positive():
    with patch.object(_engine, "_cpu_count", return_value=0):
        with patch.object(
            _engine.os, "process_cpu_count", return_value=None, create=True
        ):
            assert _engine.get_available_cores() == 1


def test_cpu_soft_limit_is_honored():
    with patch.dict(_engine.os.environ, {"LOKY_MAX_CPU_COUNT": "2"}):
        assert 1 <= _engine.get_available_cores() <= 2


# ---------------------------------------------------------------------------
# run_permutations_reduce: same lifecycle contract as run_permutations.
# ---------------------------------------------------------------------------


def test_reduce_worker_count_policy_matches_run_permutations(worker_context):
    """One worker stays serial and starts no pool; None caps at job count."""
    with patch.object(_workers.multiprocessing, "get_context") as factory:
        out = run_permutations_reduce(
            abs, [-3, -1], reduce_func=sum_squares_chunk, n_processes=1
        )
        assert out == 4.0
    factory.assert_not_called()
    with patch.object(_workers, "get_available_cores", return_value=100):
        with patch.object(worker_context, "Pool", wraps=worker_context.Pool) as pool:
            run_permutations_reduce(
                abs, [-1, -2], reduce_func=sum_squares_chunk, n_processes=2
            )
    assert pool.call_args.kwargs["processes"] == 2


def test_reduce_task_file_removed_after_success_and_failure(worker_context, tmp_path):
    """The private task file is cleaned up on success and on worker errors."""
    # The pickle-count log lives outside tmp_path so the patched tempdir
    # (which also hosts the forkserver socket) stays short.
    log = tmp_path.parent / f"{tmp_path.name}-pickles.txt"
    with patch.object(_workers.tempfile, "tempdir", str(tmp_path)):
        out = run_permutations_reduce(
            CountTaskPickles(log),
            list(range(8)),
            reduce_func=sum_squares_chunk,
            n_processes=2,
        )
        assert out == float(sum(s * s for s in range(8)))
        assert list(tmp_path.iterdir()) == []
    with patch.object(_workers.tempfile, "tempdir", str(tmp_path)):
        with pytest.raises(ValueError, match="deliberate worker failure"):
            run_permutations_reduce(
                fail_on_seed,
                [1, 2, 3],
                reduce_func=sum_squares_chunk,
                n_processes=2,
            )
        assert list(tmp_path.iterdir()) == []


def test_reduce_worker_error_leaves_no_pool_and_next_pool_succeeds(worker_context):
    children_before = {p.pid for p in multiprocessing.active_children()}
    with pytest.raises(ValueError, match="deliberate worker failure"):
        run_permutations_reduce(
            fail_on_seed, [1, 2, 3], reduce_func=sum_squares_chunk, n_processes=2
        )
    assert {p.pid for p in multiprocessing.active_children()} == children_before
    assert _workers.run_permutations(
        functools.partial(pow, 2), [1, 2], n_processes=2
    ) == [2, 4]


def test_reduce_task_is_serialized_once_across_chunks(worker_context, tmp_path):
    """Chunked reduction reuses the per-worker task like the list path."""
    log = tmp_path / "pickles.txt"
    out = run_permutations_reduce(
        CountTaskPickles(log),
        list(range(32)),
        reduce_func=sum_squares_chunk,
        n_processes=2,
    )
    assert out == float(sum(s * s for s in range(32)))
    assert log.read_text().splitlines() == ["pickle"]


def test_reduce_plain_progress_fallback_counts_permutations(worker_context, capsys):
    """Without tqdm the fallback logs at permutation granularity (total n)."""
    with patch.object(_workers, "_HAS_TQDM", False):
        out = run_permutations_reduce(
            abs,
            [-1, -2, -3, -4],
            reduce_func=sum_squares_chunk,
            n_processes=2,
            verbose=True,
            log_every=2,
        )
    assert out == 10.0
    out_text = capsys.readouterr().out
    assert "4/4 (100%)" in out_text


def test_reduce_nested_worker_runs_serially(worker_context):
    """Inside a pool worker the reduce path must not spawn another pool."""
    out = _workers.run_permutations(nested_reduction, [2, 3], n_processes=2)
    assert [r[1] for r in out] == [6.0, 12.0]


def test_reduce_batch_fold_never_holds_two_batches(worker_context):
    """NETSTAT-33 rework: the previous batch is dead before the next is built.

    Weakref counter: while seeds of batch k+1 are produced, batch k must
    already be folded and dropped, so at most ``_FOLD_BATCH`` per-seed
    results are ever alive (a missing ``del`` would drive the counter to
    almost 2 * _FOLD_BATCH). tracemalloc cross-checks the payload peak.
    """
    import gc
    import tracemalloc
    import weakref

    batch_size = 4
    refs: list = []
    probe = {"max_alive": 0}
    payload_nbytes = 200_000

    def task(seed):
        arr = np.full(payload_nbytes // 8, float(seed))
        alive = sum(1 for ref in refs if ref() is not None)
        probe["max_alive"] = max(probe["max_alive"], alive)
        refs.append(weakref.ref(arr))
        return arr

    def reduce_func(results):
        acc = np.zeros(1)
        for r in results:
            acc += r[0]
        return acc

    seeds = list(range(2 * batch_size + 3))
    with patch.object(_workers, "_FOLD_BATCH", batch_size):
        tracemalloc.start(1)
        total = run_permutations_reduce(task, seeds, reduce_func=reduce_func, use_mp=False)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    gc.collect()

    assert float(total[0]) == float(sum(seeds))
    assert probe["max_alive"] <= batch_size
    per_seed = payload_nbytes
    # Batch data plus fold transients stay well below two live batches.
    assert peak < 2 * batch_size * per_seed
