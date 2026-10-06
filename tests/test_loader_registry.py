"""Loader registry: how a manifest's ``loader:`` value becomes a class.

- Generic loaders ship registered with the package.
- Projects add their own with ``register_loader``; a manifest may also
  name any class as ``"package.module:ClassName"``.
- The lab-specific loaders live in ``apps/utils/lab_loaders.py`` and are
  not part of the core package.
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from conninfpy.loaders import (
    ManifestLoader,
    NumpyLoader,
    get_loader,
    load,
    register_loader,
)

REPO = Path(__file__).resolve().parents[1]


class TestLoaderRegistry(unittest.TestCase):

    def test_generic_loaders_are_registered(self):
        self.assertIs(get_loader("NumpyLoader"), NumpyLoader)

    def test_module_colon_class_path(self):
        self.assertIs(get_loader("conninfpy.loaders.builtins:NumpyLoader"), NumpyLoader)

    def test_unknown_name_lists_registered_loaders(self):
        with self.assertRaisesRegex(ValueError, "Registered loaders: .*NumpyLoader"):
            get_loader("NoSuchLoader")

    def test_registered_project_loader_is_used_by_manifests(self):
        @register_loader
        class ProjectLoader(NumpyLoader):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.save(tmp / "conn.npy", np.tile(np.eye(4), (5, 1, 1)))
            pd.DataFrame({"subject_id": [f"s{i}" for i in range(5)],
                          "age": np.arange(5.0)}).to_csv(tmp / "pheno.csv", index=False)
            (tmp / "study.yaml").write_text(
                "schema_version: 1\nloader: ProjectLoader\n"
                "paths:\n  data_path: conn.npy\n  pheno_path: pheno.csv\n"
            )
            self.assertIsInstance(ManifestLoader(str(tmp / "study.yaml")), ManifestLoader)
            data = load(tmp / "study.yaml")
        self.assertEqual(data.data.shape, (5, 4, 4))

    def test_lab_loaders_are_not_in_the_core_package(self):
        """Without the app, the lab loader names are unknown (fresh process)."""
        code = ("from conninfpy.loaders import get_loader\n"
                "try:\n    get_loader('OpenCloseLoader')\nexcept ValueError:\n    print('unknown')\n")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, cwd=REPO, timeout=120)
        self.assertIn("unknown", out.stdout)

    def test_app_module_registers_lab_loaders(self):
        sys.path.insert(0, str(REPO))
        try:
            import apps.utils.lab_loaders as lab
        finally:
            sys.path.remove(str(REPO))
        self.assertIs(get_loader("OpenCloseLoader"), lab.OpenCloseLoader)


if __name__ == "__main__":
    unittest.main()
