"""Tests for BrainNet Viewer scene export, label sanitization, and headless rendering."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from conninfpy.atlas.metadata import AtlasInfo
from conninfpy.cli import main
from conninfpy.reporting import export_brainnet, render_brainnet
from conninfpy.results import InferenceResult


def _make_sample_result_and_atlas(labels=None):
    n = 4
    p_pos = np.ones((n, n), dtype=float)
    p_neg = np.ones((n, n), dtype=float)
    p_pos[0, 1] = p_pos[1, 0] = 0.01
    p_neg[2, 3] = p_neg[3, 2] = 0.02
    stat_pos = np.zeros((n, n), dtype=float)
    stat_neg = np.zeros((n, n), dtype=float)
    stat_pos[0, 1] = stat_pos[1, 0] = 3.2
    stat_neg[2, 3] = stat_neg[3, 2] = 2.5

    res = InferenceResult(
        p_pos,
        p_neg,
        method="tstat",
        n_permutations=20,
        stat_positive=stat_pos,
        stat_negative=stat_neg,
    )
    coords = np.array(
        [
            [-12.0, 48.0, 18.0],
            [12.0, 48.0, 18.0],
            [-36.0, -22.0, 52.0],
            [36.0, -22.0, 52.0],
        ],
        dtype=float,
    )
    atlas = AtlasInfo(
        labels=labels if labels is not None else [f"ROI {i}" for i in range(n)],
        networks=["DMN", "DMN", "SMN", "SMN"],
        coords=coords,
    )
    return res, atlas


class BrainNetExportAndSceneTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_export_brainnet_writes_valid_scene_json_and_hashes(self) -> None:
        res, atlas = _make_sample_result_and_atlas()
        paths = export_brainnet(res, atlas, self.tmp / "network_out", alpha=0.05, write_scene=True)

        self.assertIn("scene", paths)
        self.assertTrue(paths["node"].is_file())
        self.assertTrue(paths["edge"].is_file())
        self.assertTrue(paths["scene"].is_file())

        scene = json.loads(paths["scene"].read_text(encoding="utf-8"))
        self.assertEqual(scene["schema_version"], "0.1")
        self.assertEqual(scene["loadfile"], 7)
        self.assertEqual(scene["map_mode"], 0)
        self.assertEqual(scene["coordinate_space"], "MNI")

        inputs = scene["inputs"]
        for role in ("surface", "node", "edge"):
            rel = inputs[role]["relative_path"]
            self.assertTrue(rel)
            self.assertFalse(rel.startswith("/"))
            self.assertNotIn("..", rel)

        node_sha = hashlib.sha256(paths["node"].read_bytes()).hexdigest()
        edge_sha = hashlib.sha256(paths["edge"].read_bytes()).hexdigest()
        self.assertEqual(inputs["node"]["sha256"], node_sha)
        self.assertEqual(inputs["edge"]["sha256"], edge_sha)
        self.assertEqual(inputs["surface"]["relative_path"], "BrainMesh_ICBM152_smoothed.nv")
        self.assertEqual(inputs["surface"]["sha256"], "")

        vis = scene["options"]["visualization_settings"]
        self.assertEqual(vis["schema_version"], "1.22.0")
        self.assertEqual(vis["layout"]["preset"], 1)
        self.assertEqual(vis["layout"]["view_direction"], 1)
        self.assertAlmostEqual(vis["surface"]["opacity"], 0.3)
        self.assertEqual(vis["node"]["color_mode"], "modular")
        self.assertEqual(vis["node"]["selection_mode"], "connected")
        self.assertEqual(vis["node"]["size_mode"], "mapped")
        self.assertFalse(vis["edge"]["directed"])
        self.assertEqual(vis["edge"]["selection_mode"], "all")
        self.assertEqual(vis["edge"]["color_mode"], "threshold")
        self.assertEqual(vis["edge"]["threshold_value_mode"], "signed")
        self.assertEqual(vis["edge"]["color_threshold"], 0.0)

    def test_export_brainnet_copies_surface_path_and_computes_sha256(self) -> None:
        res, atlas = _make_sample_result_and_atlas()
        src_dir = self.tmp / "surf_src"
        src_dir.mkdir()
        surf_file = src_dir / "custom_mesh.nv"
        surf_file.write_text("dummy surface\n", encoding="utf-8")

        out_dir = self.tmp / "out"
        paths = export_brainnet(
            res,
            atlas,
            out_dir / "bn",
            write_scene=True,
            surface_path=surf_file,
        )
        copied_surf = out_dir / "custom_mesh.nv"
        self.assertTrue(copied_surf.is_file())
        scene = json.loads(paths["scene"].read_text(encoding="utf-8"))
        self.assertEqual(scene["inputs"]["surface"]["relative_path"], "custom_mesh.nv")
        self.assertEqual(
            scene["inputs"]["surface"]["sha256"],
            hashlib.sha256(copied_surf.read_bytes()).hexdigest(),
        )

    def test_node_label_sanitization_strips_leading_hash(self) -> None:
        res, atlas = _make_sample_result_and_atlas(
            labels=["#ROI 0", "###ROI_1", "#", "Normal#Label"]
        )
        paths = export_brainnet(res, atlas, self.tmp / "sanitized")
        lines = paths["node"].read_text(encoding="utf-8").strip().splitlines()
        labels = [line.split("\t")[5] for line in lines]
        self.assertEqual(labels, ["ROI_0", "ROI_1", "ROI_2", "Normal#Label"])
        for lbl in labels:
            self.assertFalse(lbl.startswith("#"))

    def test_render_brainnet_missing_executable_raises_clear_error(self) -> None:
        res, atlas = _make_sample_result_and_atlas()
        paths = export_brainnet(res, atlas, self.tmp / "missing_exe", write_scene=True)
        with mock.patch("conninfpy.reporting.shutil.which", return_value=None):
            with self.assertRaises(FileNotFoundError) as ctx:
                render_brainnet(paths["scene"], self.tmp / "out.png")
        self.assertIn("brainnet-render", str(ctx.exception))
        self.assertIn("BrainNet Viewer", str(ctx.exception))

    def test_render_brainnet_invokes_subprocess_with_expected_flags(self) -> None:
        res, atlas = _make_sample_result_and_atlas()
        surf_src = self.tmp / "BrainMesh_ICBM152_smoothed.nv"
        surf_src.write_text("mesh\n", encoding="utf-8")
        out_dir = self.tmp / "render_mock"
        paths = export_brainnet(res, atlas, out_dir / "bn", write_scene=True)

        fake_exe = self.tmp / "bin" / "brainnet-render"
        fake_exe.parent.mkdir(parents=True, exist_ok=True)
        fake_exe.write_text("#!/bin/sh\n", encoding="utf-8")

        out_png = out_dir / "bn.png"

        def _fake_run(cmd, capture_output=True, text=True, check=False):
            out_png.write_bytes(b"\x89PNG\r\n\x1a\n")
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch("conninfpy.reporting.subprocess.run", side_effect=_fake_run) as run_mock:
            result_path = render_brainnet(
                paths["scene"],
                out_png,
                executable=fake_exe,
                surface_path=surf_src,
                width=800,
                height=600,
                dpi=100,
                overwrite=True,
            )

        self.assertEqual(result_path, out_png)
        self.assertTrue((out_dir / "BrainMesh_ICBM152_smoothed.nv").is_file())
        called_cmd = run_mock.call_args[0][0]
        self.assertEqual(
            called_cmd,
            [
                str(fake_exe),
                "--scene",
                str(paths["scene"]),
                "--output",
                str(out_png),
                "--width",
                "800",
                "--height",
                "600",
                "--dpi",
                "100",
                "--overwrite",
            ],
        )

    def test_cli_interpret_export_brainnet_writes_scene_and_warns_when_renderer_missing(self) -> None:
        n = 6
        rng = np.random.default_rng(7)
        n_subj = 16
        pheno = pd.DataFrame({"subject": [f"s{i}" for i in range(n_subj)], "age": np.linspace(20, 60, n_subj)})
        pheno_path = self.tmp / "pheno.csv"
        pheno.to_csv(pheno_path, index=False)

        mats = rng.normal(0.0, 0.2, size=(n_subj, n, n))
        mats = 0.5 * (mats + np.swapaxes(mats, 1, 2))
        for i in range(n_subj):
            np.fill_diagonal(mats[i], 0.0)
        mats_path = self.tmp / "mats.npy"
        np.save(mats_path, mats)

        atlas_df = pd.DataFrame(
            {
                "name": [f"ROI_{i}" for i in range(n)],
                "network": ["DMN", "DMN", "SMN", "SMN", "Vis", "Vis"],
                "x": [-10.0, 10.0, -35.0, 35.0, -20.0, 20.0],
                "y": [45.0, 45.0, -20.0, -20.0, -70.0, -70.0],
                "z": [15.0, 15.0, 50.0, 50.0, 10.0, 10.0],
            }
        )
        atlas_path = self.tmp / "atlas.csv"
        atlas_df.to_csv(atlas_path, index=False)

        manifest_path = self.tmp / "dataset.yaml"
        manifest_path.write_text(
            "\n".join(
                [
                    "schema_version: 1",
                    "loader: NumpyLoader",
                    "paths:",
                    f"  data_path: {mats_path}",
                    f"  pheno_path: {pheno_path}",
                ]
            ),
            encoding="utf-8",
        )

        out_run = self.tmp / "run_out"
        cfg_path = self.tmp / "study.yaml"
        cfg_path.write_text(
            "\n".join(
                [
                    f"data: {manifest_path}",
                    "design:",
                    "  type: glm",
                    "  interest: age",
                    "inference:",
                    "  method: tstat",
                    "  n_permutations: 20",
                    "  rng: 42",
                    f"atlas: {atlas_path}",
                    f"output: {out_run}",
                ]
            ),
            encoding="utf-8",
        )

        with contextlib.redirect_stdout(io.StringIO()):
            rc_run = main(["run", str(cfg_path)])
        self.assertEqual(rc_run, 0)

        out_interp = self.tmp / "interp_out"
        err = io.StringIO()
        with mock.patch("conninfpy.reporting.shutil.which", return_value=None):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                rc_interp = main(
                    [
                        "interpret",
                        str(out_run),
                        "--export-brainnet",
                        "--brainnet-render",
                        "--no-plots",
                        "--out",
                        str(out_interp),
                    ]
                )
        self.assertEqual(rc_interp, 0)
        self.assertTrue((out_interp / "brainnet.node").is_file())
        self.assertTrue((out_interp / "brainnet.edge").is_file())
        self.assertTrue((out_interp / "brainnet.scene.json").is_file())
        self.assertFalse((out_interp / "brainnet.png").exists())
        self.assertIn("skipping BrainNet render", err.getvalue())

    @unittest.skipIf(shutil.which("brainnet-render") is None, "brainnet-render not on PATH")
    def test_render_brainnet_headless_integration(self) -> None:
        res, atlas = _make_sample_result_and_atlas()
        out_dir = self.tmp / "bnv_integration"
        paths = export_brainnet(res, atlas, out_dir / "brainnet", alpha=0.05, write_scene=True)
        png_path = render_brainnet(paths["scene"], out_dir / "brainnet.png", width=600, height=450, dpi=100)
        self.assertTrue(png_path.is_file())
        self.assertGreater(png_path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
