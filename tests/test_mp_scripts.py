"""Scripts need no ``if __name__ == "__main__":`` guard.

With the forkserver / spawn start methods, each pool worker normally
re-imports the caller's script, re-running its top-level code (the whole
analysis) once per worker. conninfpy hides the script while its workers
start, because permutation tasks never need it. Measured on 18 cores: an
unguarded README-style script ran 19 times and took 87 s; with the fix it
runs once in about 3 s.
"""

import functools
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from conninfpy._progress import _references_main, run_permutations
from conninfpy.pairwise_stats import compute_t_stat

REPO = Path(__file__).resolve().parents[1]


def _run_script(code: str, log: Path, start_method=None) -> subprocess.CompletedProcess:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        if start_method is not None:
            f.write("import multiprocessing\n")
            f.write(f"_ctx = multiprocessing.get_context({start_method!r})\n")
            f.write("multiprocessing.get_context = lambda *args: _ctx\n")
        f.write(textwrap.dedent(code))
        path = f.name
    try:
        return subprocess.run(
            [sys.executable, path],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "RUN_LOG": str(log), "PYTHONWARNINGS": "ignore"},
        )
    finally:
        os.unlink(path)


class TestUnguardedScripts(unittest.TestCase):
    def test_unguarded_spawn_script_runs_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "runs.txt"
            proc = _run_script(
                """
                import os
                from functools import partial
                from conninfpy.stats._workers import run_permutations

                with open(os.environ["RUN_LOG"], "a") as f:
                    f.write("run\\n")
                print(run_permutations(partial(pow, 2), [1, 2, 3], n_processes=2))
            """,
                log,
                start_method="spawn",
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            self.assertEqual(log.read_text().count("run"), 1)
            self.assertIn("[2, 4, 8]", proc.stdout)

    def test_main_task_reaches_spawn_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = _run_script(
                """
                from conninfpy.stats._workers import run_permutations

                def square(seed):
                    return seed * seed

                if __name__ == "__main__":
                    print(run_permutations(square, [1, 2, 3], n_processes=2))
            """,
                Path(tmp) / "unused.txt",
                start_method="spawn",
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            self.assertIn("[1, 4, 9]", proc.stdout)

    def test_unguarded_script_runs_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "runs.txt"
            proc = _run_script(
                """
                import os
                import numpy as np
                import conninfpy

                with open(os.environ["RUN_LOG"], "a") as f:   # top-level side effect
                    f.write("run\\n")

                rng = np.random.default_rng(0)
                r = np.tanh(rng.normal(0, 0.3, (20, 8, 8)))
                r = (r + r.transpose(0, 2, 1)) / 2
                out = conninfpy.analyze(group1=r[:10], group2=r[10:], input="r",
                                        test_type="two-sample", method="tstat",
                                        n_permutations=40, rng=0, n_processes=3)
                print("done", out.inference.n_permutations)
            """,
                log,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            self.assertEqual(log.read_text().count("run"), 1)
            self.assertEqual(proc.stdout.count("done"), 1)

    def test_task_defined_in_script_still_reaches_workers(self):
        """A user function from __main__ keeps the standard (guarded) path."""
        with tempfile.TemporaryDirectory() as tmp:
            proc = _run_script(
                """
                import functools
                from conninfpy._progress import run_permutations

                def square_plus(seed, offset):
                    return seed * seed + offset

                if __name__ == "__main__":
                    out = run_permutations(functools.partial(square_plus, offset=1),
                                           list(range(6)), use_mp=True, n_processes=2)
                    print(out)
            """,
                Path(tmp) / "unused.txt",
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            self.assertIn("[1, 2, 5, 10, 17, 26]", proc.stdout)


class TestMainDetection(unittest.TestCase):
    def test_conninfpy_tasks_do_not_need_main(self):
        task = functools.partial(
            compute_t_stat, np.zeros((3, 4, 4)), test_type="one-sample"
        )
        self.assertFalse(_references_main(task))

    def test_function_from_main_is_detected(self):
        def f(seed):
            return seed

        f.__module__ = "__main__"
        self.assertTrue(_references_main(functools.partial(f)))
        self.assertTrue(_references_main(functools.partial(abs, key=f)))

    def test_main_module_is_restored_after_the_pool(self):
        main = sys.modules["__main__"]
        before = (main.__dict__.get("__file__"), main.__dict__.get("__spec__"))
        out = run_permutations(
            functools.partial(pow, 2), [1, 2, 3], use_mp=True, n_processes=2
        )
        self.assertEqual(out, [2, 4, 8])
        self.assertEqual(
            (main.__dict__.get("__file__"), main.__dict__.get("__spec__")), before
        )


if __name__ == "__main__":
    unittest.main()
