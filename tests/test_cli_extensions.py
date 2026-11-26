"""Tests for CLI upgrades: selftest, run flags, interpret support/format, influence."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from conninfpy.cli import ConfigError, main, run

FAST = {"method": "tstat", "n_permutations": 20, "use_mp": False, "rng": 0}


def _correlations(n: int, N: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.tanh(rng.normal(0, 0.3, (n, N, N)))
    r = (r + r.transpose(0, 2, 1)) / 2
    idx = np.arange(N)
    r[:, idx, idx] = 1.0
    return r


def _pheno(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "subject_id": [f"s{i:02d}" for i in range(n)],
        "age": rng.normal(30, 8, n),
        "sex": rng.choice(["F", "M"], n),
        "site": np.repeat(["a", "b"], n // 2),
        "group": np.tile(["hc", "pt"], n // 2),
    })


def _make_study(tmp: Path, pheno: pd.DataFrame, data: np.ndarray, design: dict, **extra) -> Path:
    np.save(tmp / "conn.npy", data)
    pheno.to_csv(tmp / "pheno.csv", index=False)
    (tmp / "study.yaml").write_text(yaml.safe_dump({
        "schema_version": 1,
        "loader": "NumpyLoader",
        "paths": {"data_path": "conn.npy", "pheno_path": "pheno.csv"},
    }), encoding="utf-8")
    cfg = {"data": "study.yaml", "design": design, "inference": dict(FAST), "output": "out", **extra}
    cfg_path = tmp / "analysis.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _write_coords_atlas(tmp: Path, n_rois: int = 6) -> Path:
    # MNI coordinates inside cerebral cortex so BNA-246 annotation succeeds
    coords = [
        (-10.0, 50.0, 20.0),
        (10.0, 50.0, 20.0),
        (-40.0, -20.0, 50.0),
        (40.0, -20.0, 50.0),
        (-25.0, -70.0, 10.0),
        (25.0, -70.0, 10.0),
    ][:n_rois]
    df = pd.DataFrame({
        "name": [f"ROI_{i}" for i in range(n_rois)],
        "network": ["DMN", "DMN", "SMN", "SMN", "Vis", "Vis"][:n_rois],
        "x": [c[0] for c in coords],
        "y": [c[1] for c in coords],
        "z": [c[2] for c in coords],
    })
    path = tmp / "atlas.csv"
    df.to_csv(path, index=False)
    return path


class TestSelftest(unittest.TestCase):

    def test_selftest_text_and_json(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["selftest"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("PASS", out)
        self.assertIn("tfnbs_mp", out)

        buf_json = io.StringIO()
        with contextlib.redirect_stdout(buf_json):
            rc_j = main(["selftest", "--json"])
        self.assertEqual(rc_j, 0)
        payload = json.loads(buf_json.getvalue())
        self.assertTrue(payload["ok"])
        self.assertIn("actual_backend", payload["backend"])
        self.assertEqual(payload["checks"]["synthetic_data"]["status"], "PASS")
        self.assertEqual(payload["checks"]["tfnbs_mp"]["status"], "PASS")
        self.assertEqual(payload["checks"]["bundled_atlas"]["status"], "PASS")


class TestRunFlags(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_dry_run_writes_no_files(self):
        cfg = _make_study(self.tmp, _pheno(16), _correlations(16), {"type": "glm", "interest": "age", "confounds": ["sex"]})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["run", str(cfg), "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertIn("dry-run", buf.getvalue())
        self.assertFalse((self.tmp / "out").exists())

    def test_n_jobs_method_seed_overrides(self):
        cfg = _make_study(self.tmp, _pheno(16), _correlations(16), {"type": "glm", "interest": "age"})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["run", str(cfg), "--n-jobs", "2", "--method", "tstat", "--seed", "7"])
        self.assertEqual(rc, 0)
        prov = json.loads((self.tmp / "out" / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(prov["results"]["result"]["method"], "tstat")

    def test_no_mp_and_n_jobs_gt_1_raises(self):
        cfg = _make_study(self.tmp, _pheno(16), _correlations(16), {"type": "glm", "interest": "age"})
        with self.assertRaises(ConfigError):
            run(cfg, use_mp=False, n_processes=2)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = main(["run", str(cfg), "--no-mp", "--n-jobs", "2"])
        self.assertEqual(rc, 1)
        self.assertIn("--no-mp", err.getvalue())


class TestInterpretFlags(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_interpret_sphere_pdf_and_point(self):
        atlas_path = _write_coords_atlas(self.tmp, n_rois=6)
        cfg = _make_study(
            self.tmp,
            _pheno(16),
            _correlations(16, N=6),
            {"type": "glm", "interest": "age"},
            atlas=str(atlas_path),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            run(cfg)

        out_pdf = self.tmp / "interp_pdf"
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main([
                "interpret", str(self.tmp / "out"),
                "--support", "sphere",
                "--radius-mm", "5",
                "--space", "MNI152NLin2009cAsym",
                "--format", "pdf",
                "--dpi", "150",
                "--out", str(out_pdf),
            ])
        self.assertEqual(rc, 0)
        self.assertTrue((out_pdf / "summary_figure.pdf").is_file())
        self.assertTrue((out_pdf / "annotated_edges.csv").is_file())

        out_pt = self.tmp / "interp_point"
        with contextlib.redirect_stdout(io.StringIO()):
            rc_pt = main([
                "interpret", str(self.tmp / "out"),
                "--support", "point",
                "--no-plots",
                "--out", str(out_pt),
            ])
        self.assertEqual(rc_pt, 0)
        self.assertTrue((out_pt / "annotated_edges.csv").is_file())
        self.assertFalse((out_pt / "summary_figure.png").exists())


class TestInfluenceCLI(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_influence_edges_and_edges_file(self):
        cfg = _make_study(
            self.tmp,
            _pheno(16),
            _correlations(16, N=6),
            {"type": "glm", "interest": "age", "confounds": ["sex"]},
        )
        with contextlib.redirect_stdout(io.StringIO()):
            run(cfg)

        # Missing --edges and --edges-file must fail cleanly
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc_missing = main(["influence", str(self.tmp / "out"), "--config", str(cfg)])
        self.assertEqual(rc_missing, 1)

        # Explicit --edges
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main([
                "influence", str(self.tmp / "out"),
                "--config", str(cfg),
                "--edges", "0:1,2:5",
                "--json",
            ])
        self.assertEqual(rc, 0)
        rep = json.loads(buf.getvalue())
        self.assertEqual(rep["unavailable"], [])
        self.assertEqual(rep["n_edges"], 2)
        infl_dir = self.tmp / "out" / "influence"
        self.assertTrue((infl_dir / "subject_influence.json").is_file())
        self.assertTrue((infl_dir / "subject_influence.csv").is_file())

        # Explicit --edges-file
        edges_csv = self.tmp / "chosen_edges.csv"
        pd.DataFrame({"roi_i": [0, 1], "roi_j": [2, 3]}).to_csv(edges_csv, index=False)
        out_file_dir = self.tmp / "infl_file"
        with contextlib.redirect_stdout(io.StringIO()):
            rc2 = main([
                "influence", str(self.tmp / "out"),
                "--config", str(cfg),
                "--edges-file", str(edges_csv),
                "--out", str(out_file_dir),
            ])
        self.assertEqual(rc2, 0)
        rep2 = json.loads((out_file_dir / "subject_influence.json").read_text(encoding="utf-8"))
        self.assertEqual(rep2["unavailable"], [])
        self.assertEqual(rep2["n_edges"], 2)


class TestInitManifestAndBidsPheno(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_bids_tsv_na_participant_id_end_to_end(self):
        from conninfpy.loaders import load

        data_dir = self.tmp / "fc_tsv"
        data_dir.mkdir()
        mats = _correlations(6, N=5, seed=11)
        for i in range(6):
            np.savetxt(data_dir / f"sub-{i+1:02d}_fc.tsv", mats[i], delimiter="\t", fmt="%.6f")

        pheno_path = self.tmp / "participants.tsv"
        pheno_path.write_text(
            "participant_id\tage\tgroup\tnotes\n"
            "sub-01\t25.0\thc\tok\n"
            "sub-02\tn/a\tpt\tmissing_age\n"
            "sub-03\t31.5\thc\tn/a\n"
            "sub-04\t28.0\tpt\tok\n"
            "sub-05\t35.0\thc\tok\n"
            "sub-06\t40.0\tpt\tok\n",
            encoding="utf-8",
        )

        manifest_path = self.tmp / "manifest.yaml"
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main([
                "init-manifest",
                "--data-dir", str(data_dir),
                "--pheno", str(pheno_path),
                "--out", str(manifest_path),
            ])
        self.assertEqual(rc, 0)
        self.assertTrue(manifest_path.is_file())

        ds = load(manifest_path)
        self.assertEqual(ds.data.shape, (6, 5, 5))
        self.assertEqual(ds.data_kind, "correlation")
        self.assertTrue(pd.api.types.is_numeric_dtype(ds.pheno["age"]))
        self.assertTrue(pd.isna(ds.pheno.loc[1, "age"]))
        self.assertTrue(pd.isna(ds.pheno.loc[2, "notes"]))

        analysis_path = self.tmp / "analysis.yaml"
        with contextlib.redirect_stdout(io.StringIO()):
            rc_init = main([
                "init",
                "--manifest", str(manifest_path),
                "--out", str(analysis_path),
            ])
        self.assertEqual(rc_init, 0)
        self.assertTrue(analysis_path.is_file())

        with contextlib.redirect_stdout(io.StringIO()):
            rc_qc = main(["qc", str(analysis_path)])
        self.assertEqual(rc_qc, 0)

    def test_bids_sub_prefix_mismatch_with_unprefixed_npy(self):
        from conninfpy.loaders import load

        data_dir = self.tmp / "fc_npy"
        data_dir.mkdir()
        mats = _correlations(4, N=4, seed=5) * 0.5  # off-diag in (-0.5, 0.5), diag 0.5 -> fisher_z or corr
        mats[:, np.arange(4), np.arange(4)] = 1.5  # >1 so auto-detects fisher_z
        for i in range(4):
            np.save(data_dir / f"{i+1:02d}.npy", mats[i])

        pheno_path = self.tmp / "participants.tsv"
        pheno_path.write_text(
            "participant_id\tage\n"
            "sub-01\t22\n"
            "sub-02\tn/a\n"
            "sub-03\t29\n"
            "sub-04\t33\n",
            encoding="utf-8",
        )

        manifest_path = self.tmp / "manifest_npy.yaml"
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main([
                "init-manifest",
                "--data-dir", str(data_dir),
                "--pheno", str(pheno_path),
                "--out", str(manifest_path),
            ])
        self.assertEqual(rc, 0)
        ds = load(manifest_path)
        self.assertEqual(ds.data_kind, "fisher_z")
        self.assertEqual(ds.pheno["age"].dropna().tolist(), [22.0, 29.0, 33.0])

    def test_init_manifest_3d_npy_and_timeseries_dir(self):
        from conninfpy.loaders import load

        npy_file = self.tmp / "stack.npy"
        np.save(npy_file, _correlations(5, N=6, seed=3))
        m_npy = self.tmp / "stack_manifest.yaml"
        with contextlib.redirect_stdout(io.StringIO()):
            rc1 = main(["init-manifest", "--data-file", str(npy_file), "--out", str(m_npy)])
        self.assertEqual(rc1, 0)
        ds1 = load(m_npy)
        self.assertEqual(ds1.data.shape, (5, 6, 6))
        self.assertEqual(ds1.data_kind, "correlation")

        ts_dir = self.tmp / "ts"
        ts_dir.mkdir()
        rng = np.random.default_rng(9)
        for i in range(3):
            np.savetxt(ts_dir / f"sub-{i+1:02d}_ts.csv", rng.normal(0, 1, (30, 5)), delimiter=",")
        m_ts = self.tmp / "ts_manifest.yaml"
        with contextlib.redirect_stdout(io.StringIO()):
            rc2 = main(["init-manifest", "--data-dir", str(ts_dir), "--out", str(m_ts)])
        self.assertEqual(rc2, 0)
        ds2 = load(m_ts)
        self.assertEqual(ds2.data_kind, "timeseries")
        self.assertEqual(ds2.data.shape, (3, 30, 5))


if __name__ == "__main__":
    unittest.main()
