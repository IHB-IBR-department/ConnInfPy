"""Command line: ``conninfpy run analysis.yaml`` and ``conninfpy info``.

Each test writes a small synthetic study (NumPy connectivity + phenotype
CSV + manifest) and an analysis file into a temporary directory, so the
whole path — manifest → design → analyze() → output files — is exercised.
"""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from conninfpy import __version__
from conninfpy.cli import main, run

REPO = Path(__file__).resolve().parents[1]
FAST = {"method": "tstat", "n_permutations": 20, "use_mp": False, "rng": 0}


def _correlations(n, N=6, seed=0):
    rng = np.random.default_rng(seed)
    r = np.tanh(rng.normal(0, 0.3, (n, N, N)))
    r = (r + r.transpose(0, 2, 1)) / 2
    idx = np.arange(N)
    r[:, idx, idx] = 1.0
    return r


class _Study:
    """A throwaway study directory with data, phenotype and manifest."""

    def __init__(self, tmp: Path, pheno: pd.DataFrame, data: np.ndarray):
        self.dir = tmp
        np.save(tmp / "conn.npy", data)
        pheno.to_csv(tmp / "pheno.csv", index=False)
        (tmp / "study.yaml").write_text(yaml.safe_dump({
            "schema_version": 1, "loader": "NumpyLoader",
            "paths": {"data_path": "conn.npy", "pheno_path": "pheno.csv"},
        }))

    def analysis(self, design, **extra):
        cfg = {"data": "study.yaml", "design": design, "inference": dict(FAST),
               "output": "out", **extra}
        path = self.dir / "analysis.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path


def _pheno(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "subject_id": [f"s{i:02d}" for i in range(n)],
        "age": rng.normal(30, 8, n),
        "sex": rng.choice(["F", "M"], n),       # text column: one-hot encoded
        "site": np.repeat(["a", "b"], n // 2),
        "group": np.tile(["hc", "pt"], n // 2),
    })


class TestRun(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _quiet_run(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run(*args, **kwargs)

    def test_glm_writes_edges_pvals_and_provenance(self):
        study = _Study(self.tmp, _pheno(24), _correlations(24))
        cfg = study.analysis({"type": "glm", "interest": "age",
                              "confounds": ["sex"], "sites": "site"})
        written = self._quiet_run(cfg)
        for key in ("edges", "pvals", "provenance"):
            self.assertTrue(written[key].exists(), key)
        with np.load(written["pvals"]) as z:
            self.assertEqual(z["positive"].shape, (6, 6))
        prov = json.loads(written["provenance"].read_text())
        self.assertEqual(prov["conninfpy_version"], __version__)
        self.assertEqual(prov["data"]["n_subjects_used"], 24)
        self.assertEqual(prov["data"]["input"], "r")      # from the manifest's data_kind
        self.assertIn("flags", prov["results"]["result"])
        self.assertEqual(prov["results"]["result"]["tail_policy"], "two-sided")
        self.assertEqual(prov["results"]["result"]["correction"], "permutation_max")
        self.assertEqual(prov["results"]["result"]["inference_unit"], "edge")

    def test_explicit_one_sided_config_preserves_directional_family(self):
        study = _Study(self.tmp, _pheno(12), _correlations(12))
        cfg = study.analysis({"type": "glm", "interest": "age"})
        config = yaml.safe_load(cfg.read_text())
        config["inference"]["two_tailed"] = False
        cfg.write_text(yaml.safe_dump(config))
        written = self._quiet_run(cfg)
        prov = json.loads(written["provenance"].read_text())
        self.assertEqual(prov["results"]["result"]["tail_policy"], "separate-one-sided")

    def test_rows_with_missing_values_are_dropped(self):
        pheno = _pheno(24)
        pheno.loc[3, "age"] = np.nan
        study = _Study(self.tmp, pheno, _correlations(24))
        written = self._quiet_run(study.analysis({"type": "glm", "interest": "age"}))
        prov = json.loads(written["provenance"].read_text())
        self.assertEqual(prov["data"]["n_subjects_used"], 23)

    def test_several_predictors_write_one_set_of_files_each(self):
        study = _Study(self.tmp, _pheno(24), _correlations(24))
        written = self._quiet_run(study.analysis({"type": "glm", "interest": ["age", "sex"]}))
        self.assertIn("edges_age", written)
        self.assertIn("edges_sex", written)

    def test_two_sample_design(self):
        study = _Study(self.tmp, _pheno(24), _correlations(24))
        written = self._quiet_run(study.analysis(
            {"type": "two-sample", "group": "group", "levels": ["hc", "pt"]}))
        prov = json.loads(written["provenance"].read_text())
        self.assertEqual(prov["data"]["n_subjects_used"], 24)

    def test_paired_design_aligns_subjects(self):
        n = 10
        pheno = pd.DataFrame({
            "subject_id": [f"s{i}" for i in range(n)] * 2,
            "condition": ["rest"] * n + ["task"] * n,
            "fd": np.random.default_rng(1).normal(0.2, 0.05, 2 * n),
        }).iloc[::-1].reset_index(drop=True)              # shuffled order on purpose
        study = _Study(self.tmp, pheno, _correlations(2 * n))
        written = self._quiet_run(study.analysis(
            {"type": "paired", "condition": "condition", "levels": ["rest", "task"],
             "confounds": ["fd"]}))
        prov = json.loads(written["provenance"].read_text())
        self.assertEqual(prov["data"]["n_subjects_used"], n)

    def test_out_argument_overrides_output(self):
        study = _Study(self.tmp, _pheno(24), _correlations(24))
        written = self._quiet_run(study.analysis({"type": "one-sample"}),
                                  out_dir=self.tmp / "elsewhere")
        self.assertEqual(written["edges"].parent, self.tmp / "elsewhere")

    def test_unknown_column_is_a_clean_error(self):
        study = _Study(self.tmp, _pheno(24), _correlations(24))
        cfg = study.analysis({"type": "glm", "interest": "AGE"})
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = main(["run", str(cfg)])
        self.assertEqual(code, 1)
        self.assertIn("not in the phenotype table", err.getvalue())
        self.assertIn("age", err.getvalue())               # lists available columns

    def test_unknown_top_level_key_is_rejected(self):
        study = _Study(self.tmp, _pheno(24), _correlations(24))
        cfg = study.analysis({"type": "glm", "interest": "age"}, desing={"typo": 1})
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["run", str(cfg)]), 1)


class TestCommand(unittest.TestCase):

    def test_info(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["info"]), 0)
        self.assertIn(__version__, out.getvalue())

    @unittest.skipUnless((REPO / "datasets" / "abide_mini.yaml").exists(),
                         "bundled ABIDE-mini dataset not present")
    def test_bundled_example_runs_as_a_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, "-m", "conninfpy", "run",
                 "examples/configs/abide_mini_age.yaml", "--out", tmp,
                 "--n-permutations", "50"],
                cwd=REPO, capture_output=True, text=True, timeout=600,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            # 204 in the file; one subject has a region without signal and is
            # dropped by the manifest (params.missing: drop_subjects)
            self.assertIn("203 subjects × 100 regions", proc.stdout)
            self.assertTrue((Path(tmp) / "edges.csv").exists())


if __name__ == "__main__":
    unittest.main()
