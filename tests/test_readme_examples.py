"""The README's runnable examples must keep running.

Each example is the first ```python block after an HTML marker comment
(``<!-- readme-quickstart -->``, ``<!-- readme-real-data -->``), so the
test exercises exactly the code a reader copies.
"""
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"


def _example(marker: str) -> str:
    text = README.read_text(encoding="utf-8")
    match = re.search(
        rf"<!-- {re.escape(marker)} -->\s*```python\n(.*?)```", text, re.DOTALL
    )
    if match is None:
        raise AssertionError(f"README has no python block after <!-- {marker} -->")
    return match.group(1)


def _run_as_script(code: str, timeout: int = 300) -> str:
    """Run code the way a reader would: as a script, from the repo root."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path], cwd=REPO, capture_output=True, text=True,
            timeout=timeout, env={**os.environ, "PYTHONWARNINGS": "ignore"},
        )
    finally:
        os.unlink(path)
    if proc.returncode != 0:
        raise AssertionError(f"README example failed:\n{proc.stderr[-3000:]}")
    return proc.stdout


class TestReadmeExamples(unittest.TestCase):

    def test_quickstart_recovers_the_planted_effect(self):
        out = _run_as_script(_example("readme-quickstart"))
        found, planted = map(int, re.search(r"planted edges found: (\d+) of (\d+)", out).groups())
        false_pos = int(re.search(r"false positives:\s+(\d+)", out).group(1))
        self.assertGreaterEqual(found, 0.8 * planted)     # README shows 45 of 45
        self.assertLessEqual(false_pos, 2)                # README shows 0

    @unittest.skipUnless((REPO / "datasets" / "abide_mini.yaml").exists(),
                         "bundled ABIDE-mini dataset not present")
    def test_real_data_example_runs(self):
        out = _run_as_script(_example("readme-real-data"))
        self.assertIn("AnalyzeResult(", out)


if __name__ == "__main__":
    unittest.main()
