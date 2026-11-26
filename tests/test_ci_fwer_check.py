"""Decision-rule and workflow tests for ``scripts/ci_fwer_check.py``.

The nightly FWER check itself runs in ``.github/workflows/fwer-calibration.yml``
and takes about a minute of simulation; these tests cover everything that can
be verified without simulating:

1. The exact binomial decision statistics (count band, one-sided p-value,
   Clopper-Pearson interval, Holm adjustment) and the pass/fail aggregation.
2. The workflow file parses, is triggered only by ``schedule`` and
   ``workflow_dispatch`` (never ``pull_request`` / ``push``), and points at the
   script that exists.
3. The script's command line parses and ``--help`` works.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

import yaml
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ci_fwer_check.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "fwer-calibration.yml"


def _load_script():
    spec = importlib.util.spec_from_file_location("ci_fwer_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cfc = _load_script()


class TestDecisionStatistics(unittest.TestCase):
    def test_exact_binomial_interval_brackets_alpha_mean(self):
        for n in (100, 200, 1000):
            lo, hi = cfc.exact_binomial_interval(n)
            self.assertLess(lo, cfc.ALPHA * n)
            self.assertGreater(hi, cfc.ALPHA * n)
        # Values used by the default 200-null run.
        self.assertEqual(cfc.exact_binomial_interval(200), (3, 19))

    def test_one_sided_p_matches_scipy(self):
        for k in (0, 7, 19, 20, 200):
            self.assertAlmostEqual(
                cfc.one_sided_p_above(k, 200), float(stats.binom.sf(k - 1, 200, 0.05)), places=12
            )

    def test_clopper_pearson_matches_scipy_beta(self):
        low, high = cfc.clopper_pearson(10, 200)
        self.assertAlmostEqual(low, float(stats.beta.ppf(0.005, 10, 191)), places=12)
        self.assertAlmostEqual(high, float(stats.beta.ppf(0.995, 11, 190)), places=12)

    def test_fail_bound_is_the_99_5_percent_quantile(self):
        # k at the 99.5 % quantile of Binomial(200, 0.05) passes; one more fails.
        self.assertFalse(cfc.summarize_arm("tstat", 200, 19, 0.0)["fail"])
        self.assertTrue(cfc.summarize_arm("tstat", 200, 20, 0.0)["fail"])
        self.assertGreaterEqual(cfc.summarize_arm("tstat", 200, 19, 0.0)["p_hi"], cfc.FAIL_P)
        self.assertLess(cfc.summarize_arm("tstat", 200, 20, 0.0)["p_hi"], cfc.FAIL_P)

    def test_clopper_pearson_edges_and_consistency(self):
        self.assertEqual(cfc.clopper_pearson(0, 200)[0], 0.0)
        self.assertEqual(cfc.clopper_pearson(200, 200)[1], 1.0)
        low, high = cfc.clopper_pearson(10, 200)
        self.assertLess(low, 0.05)
        self.assertGreater(high, 0.05)

    def test_summarize_arm_fields(self):
        row = cfc.summarize_arm("tfnbs", 200, 11, 4.1, k_pos=5, k_neg=6)
        self.assertEqual(row["fwer_joint"], 11 / 200)
        self.assertTrue(row["inside_binom99"])
        self.assertFalse(row["conservative"])  # CP upper bound above alpha
        self.assertEqual((row["k_pos"], row["k_neg"], row["seconds"]), (5, 6, 4.1))
        self.assertTrue(json.dumps(row))  # JSON-serialisable for --json

    def test_holm_adjust(self):
        self.assertEqual(cfc.holm_adjust([0.001, 0.04, 0.9]), [0.003, 0.08, 0.9])
        self.assertEqual(cfc.holm_adjust([0.5, 0.5]), [1.0, 1.0])
        self.assertEqual(cfc.holm_adjust([1.0]), [1.0])

    def test_finalize_verdict_and_holm_column(self):
        good = [cfc.summarize_arm(a, 200, k, 0.0) for a, k in zip(cfc.ALL_ARMS, (8, 11, 7, 12, 12))]
        ok, rows = cfc.finalize(good)
        self.assertTrue(ok)
        self.assertTrue(all(r["p_holm"] >= r["p_hi"] for r in rows))
        self.assertEqual(len(rows), len(cfc.ALL_ARMS))
        bad = [cfc.summarize_arm(a, 200, k, 0.0) for a, k in zip(cfc.ALL_ARMS, (8, 11, 25, 12, 12))]
        ok_bad, rows_bad = cfc.finalize(bad)
        self.assertFalse(ok_bad)
        self.assertEqual([r["arm"] for r in rows_bad if r["fail"]], ["nbs"])

    def test_arms_and_contract(self):
        self.assertEqual(cfc.ALL_ARMS, ("tstat", "tfnbs", "nbs", "cnbs", "glm_tfnbs"))
        self.assertTrue(cfc.TWO_SIDED)
        self.assertIsNone(cfc.TFNBS_KWARGS["n"])  # exact integration
        self.assertEqual(cfc.ALPHA, 0.05)


class TestWorkflowFile(unittest.TestCase):
    def setUp(self):
        self.doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    def test_triggers_are_schedule_and_dispatch_only(self):
        # PyYAML (YAML 1.1) reads the bare `on` key as boolean True.
        triggers = self.doc.get("on", self.doc.get(True))
        self.assertEqual(set(triggers), {"schedule", "workflow_dispatch"})
        self.assertNotIn("pull_request", triggers)
        self.assertNotIn("push", triggers)
        self.assertRegex(triggers["schedule"][0]["cron"], r"^\d+ \d+ \* \* \*$")
        self.assertIn("inputs", triggers["workflow_dispatch"])

    def test_permissions_read_only_and_single_job(self):
        self.assertEqual(self.doc["permissions"], {"contents": "read"})
        self.assertEqual(list(self.doc["jobs"]), ["calibration"])
        job = self.doc["jobs"]["calibration"]
        self.assertIn("timeout-minutes", job)
        runs = [s.get("run", "") for s in job["steps"] if "run" in s]
        self.assertTrue(any("scripts/ci_fwer_check.py" in r for r in runs))
        self.assertTrue(SCRIPT.exists())
        self.assertTrue(all("pull_request" not in r for r in runs))


class TestCommandLine(unittest.TestCase):
    def test_help(self):
        proc = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--n-null", proc.stdout)
        self.assertIn("--seed", proc.stdout)

    def test_rejects_degenerate_sizes(self):
        proc = subprocess.run([sys.executable, str(SCRIPT), "--n-null", "1"], capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("must be >= 2", proc.stderr)


if __name__ == "__main__":
    unittest.main()
