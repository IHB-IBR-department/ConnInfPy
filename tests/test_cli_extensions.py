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

    @unittest.skipUnless(
        __import__("conninfpy.atlas.resources", fromlist=["atlas_resource_available"]).atlas_resource_available("BN_Atlas_246_1mm.nii.gz")
        and __import__("conninfpy.atlas.resources", fromlist=["atlas_resource_available"]).atlas_resource_available("bna246_hierarchy.csv")
        and __import__("conninfpy.atlas.resources", fromlist=["atlas_resource_available"]).atlas_resource_available("bna246.csv"),
        "Brainnetome atlas files not in local cache; run scripts/download_atlases.py",
    )
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


class TestReportingAndBrainNet(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_methods_boilerplate_api_and_cli_conditional_refs(self):
        from conninfpy import analyze
        from conninfpy.atlas.metadata import AtlasInfo

        # Single-site run: ComBat must NOT be cited; SUIT cited only when cerebellum != none
        r = _correlations(16, N=6, seed=1)
        grp = np.array(["hc"] * 8 + ["pt"] * 8)
        res = analyze(
            group1=r[:8],
            group2=r[8:],
            input="r",
            test_type="two-sample",
            method="tfnbs",
            n_permutations=20,
            rng=0,
            use_mp=False,
        )
        md_single = res.methods_boilerplate()
        self.assertIn("2.0", md_single)
        self.assertIn("inference_unit", md_single)
        self.assertIn("`edge`", md_single)
        self.assertIn("tail_policy", md_single)
        self.assertIn("tfnbs", md_single)
        self.assertIn("Baggio", md_single)
        self.assertIn("Phipson", md_single)
        self.assertNotIn("Fortin", md_single)
        self.assertNotIn("Diedrichsen", md_single)

        # CLI interpret --methods with cerebellum=anatom and multi-site ComBat
        atlas_path = _write_coords_atlas(self.tmp, n_rois=6)
        cfg = _make_study(
            self.tmp,
            _pheno(16),
            _correlations(16, N=6, seed=2),
            {
                "type": "glm",
                "interest": "age",
                "confounds": ["sex"],
                "sites": "site",
                "harmonize": "combat_site_dummies_glm",
            },
            atlas=str(atlas_path),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            run(cfg)

        out_interp = self.tmp / "interp_methods"
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main([
                "interpret", str(self.tmp / "out"),
                "--methods",
                "--no-plots",
                "--out", str(out_interp),
            ])
        self.assertEqual(rc, 0)
        methods_file = out_interp / "methods_boilerplate.md"
        self.assertTrue(methods_file.is_file())
        md_cli = methods_file.read_text(encoding="utf-8")
        self.assertIn("2.0", md_cli)
        self.assertIn("combat_site_dummies_glm", md_cli)
        self.assertIn("Fortin", md_cli)

        from conninfpy.atlas.resources import atlas_resource_available
        if all(
            atlas_resource_available(n)
            for n in (
                "BN_Atlas_246_1mm.nii.gz",
                "bna246_hierarchy.csv",
                "bna246.csv",
                "atl-Anatom_space-MNI_dseg.nii.gz",
                "atl-Anatom_space-MNI_dseg.tsv",
            )
        ):
            out_interp_cb = self.tmp / "interp_methods_cb"
            with contextlib.redirect_stdout(io.StringIO()):
                rc_cb = main([
                    "interpret", str(self.tmp / "out"),
                    "--methods",
                    "--cerebellum", "anatom",
                    "--no-plots",
                    "--out", str(out_interp_cb),
                ])
            self.assertEqual(rc_cb, 0)
            md_cb = (out_interp_cb / "methods_boilerplate.md").read_text(encoding="utf-8")
            self.assertIn("Diedrichsen", md_cb)

        # AtlasInfo without coords check is independent; test BNA-246 reference trigger
        bna_mock = AtlasInfo(
            labels=[f"R{i}" for i in range(6)],
            networks=["Net1"] * 6,
            source="Brainnetome Atlas (BNA-246)",
        )
        md_bna = res.methods_boilerplate(atlas=bna_mock)
        self.assertIn("Fan, L.", md_bna)

    def test_n_effective_in_significant_edges_and_csv(self):
        from conninfpy import analyze

        r = _correlations(16, N=6, seed=4)
        # Inject NaNs on edge (0, 1) for 3 subjects -> n_effective = 13
        r[0, 0, 1] = r[0, 1, 0] = np.nan
        r[1, 0, 1] = r[1, 1, 0] = np.nan
        r[8, 0, 1] = r[8, 1, 0] = np.nan

        res = analyze(
            group1=r[:8],
            group2=r[8:],
            input="r",
            test_type="two-sample",
            method="tstat",
            n_permutations=20,
            missing="available_case",
            rng=0,
            use_mp=False,
        )
        df = res.significant_edges(include_nonsig=True)
        self.assertIn("n_effective", df.columns)
        row_01 = df[(df["roi_i"] == 0) & (df["roi_j"] == 1)].iloc[0]
        row_23 = df[(df["roi_i"] == 2) & (df["roi_j"] == 3)].iloc[0]
        self.assertEqual(int(row_01["n_effective"]), 13)
        self.assertEqual(int(row_23["n_effective"]), 16)

        csv_path = self.tmp / "edges_avail.csv"
        res.to_csv(csv_path, include_nonsig=True)
        df_csv = pd.read_csv(csv_path)
        self.assertIn("n_effective", df_csv.columns)

        # Complete-case result must NOT have n_effective
        res_cc = analyze(
            group1=_correlations(16, N=6, seed=4)[:8],
            group2=_correlations(16, N=6, seed=4)[8:],
            input="r",
            test_type="two-sample",
            method="tstat",
            n_permutations=20,
            rng=0,
            use_mp=False,
        )
        self.assertNotIn("n_effective", res_cc.significant_edges(include_nonsig=True).columns)

    def test_export_brainnet_api_cli_and_missing_coords(self):
        from conninfpy.atlas.metadata import AtlasInfo
        from conninfpy.results import InferenceResult

        N = 6
        p_pos = np.ones((N, N), dtype=float)
        p_neg = np.ones((N, N), dtype=float)
        p_pos[0, 1] = p_pos[1, 0] = 0.01
        p_neg[2, 3] = p_neg[3, 2] = 0.02
        stat_pos = np.zeros((N, N), dtype=float)
        stat_neg = np.zeros((N, N), dtype=float)
        stat_pos[0, 1] = stat_pos[1, 0] = 3.5
        stat_neg[2, 3] = stat_neg[3, 2] = 2.8

        res = InferenceResult(
            p_pos,
            p_neg,
            method="tstat",
            n_permutations=20,
            stat_positive=stat_pos,
            stat_negative=stat_neg,
        )
        coords = np.array([
            [-10.0, 50.0, 20.0],
            [10.0, 50.0, 20.0],
            [-40.0, -20.0, 50.0],
            [40.0, -20.0, 50.0],
            [-25.0, -70.0, 10.0],
            [25.0, -70.0, 10.0],
        ])
        atlas = AtlasInfo(
            labels=[f"ROI {i}" for i in range(N)],
            networks=["DMN", "DMN", "SMN", "SMN", "Vis", "Vis"],
            coords=coords,
        )
        paths = res.to_brainnet(atlas, self.tmp / "bn_test", alpha=0.05)
        self.assertTrue(paths["node"].is_file())
        self.assertTrue(paths["edge"].is_file())

        node_lines = paths["node"].read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(node_lines), N)
        for idx, line in enumerate(node_lines):
            cols = line.split("\t")
            self.assertEqual(len(cols), 6)
            self.assertNotIn(" ", cols[5])
            self.assertEqual(cols[5], f"ROI_{idx}")

        edge_mat = np.loadtxt(paths["edge"], delimiter="\t")
        self.assertEqual(edge_mat.shape, (N, N))
        self.assertTrue(np.allclose(edge_mat, edge_mat.T))
        self.assertTrue(np.allclose(np.diag(edge_mat), 0.0))
        self.assertAlmostEqual(edge_mat[0, 1], 3.5, places=5)
        self.assertAlmostEqual(edge_mat[2, 3], -2.8, places=5)
        self.assertEqual(int(np.sum(edge_mat != 0.0)), 4)  # 2 undirected edges * 2

        # Missing coords raises ValueError in API
        atlas_no_coords = AtlasInfo(labels=[f"ROI_{i}" for i in range(N)], networks=["A"] * N)
        with self.assertRaises(ValueError):
            res.to_brainnet(atlas_no_coords, self.tmp / "bn_fail")

        # CLI interpret --export-brainnet with coords
        atlas_path = _write_coords_atlas(self.tmp, n_rois=N)
        cfg = _make_study(
            self.tmp,
            _pheno(16),
            _correlations(16, N=N, seed=5),
            {"type": "glm", "interest": "age"},
            atlas=str(atlas_path),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            run(cfg)
        out_bn = self.tmp / "interp_bn"
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main([
                "interpret", str(self.tmp / "out"),
                "--export-brainnet",
                "--no-plots",
                "--out", str(out_bn),
            ])
        self.assertEqual(rc, 0)
        self.assertTrue((out_bn / "brainnet.node").is_file())
        self.assertTrue((out_bn / "brainnet.edge").is_file())

        # CLI interpret --export-brainnet without coords skips gracefully
        no_coord_csv = self.tmp / "atlas_nocoord.csv"
        pd.DataFrame({"name": [f"R{i}" for i in range(N)], "network": ["Net"] * N}).to_csv(
            no_coord_csv, index=False
        )
        out_skip = self.tmp / "interp_bn_skip"
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc_skip = main([
                "interpret", str(self.tmp / "out"),
                "--atlas", str(no_coord_csv),
                "--export-brainnet",
                "--no-plots",
                "--out", str(out_skip),
            ])
        self.assertEqual(rc_skip, 0)
        self.assertFalse((out_skip / "brainnet.node").exists())
        self.assertIn("skipping BrainNet export", err.getvalue())


if __name__ == "__main__":
    unittest.main()
