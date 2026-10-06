"""The public surface: what ``conninfpy`` advertises and where things live.

- ``__all__`` is the short recommended surface (front door, results,
  data-in helpers, modules).
- Advanced functions are reachable through their module
  (``conninfpy.stats``, ``conninfpy.enhance``, ``conninfpy.sim``).
- Names imported from the top level in earlier versions stay importable.
"""
import importlib
import unittest

import conninfpy

MODULE_EXPORTS = {
    "stats": ["compute_p_val", "compute_p_val_glm", "compute_p_val_glm_multi",
              "compute_p_val_paired_glm", "compute_null_dist", "compute_t_stat",
              "build_design_matrix", "StatMethod", "TestType", "GLMStatType"],
    "enhance": ["apply_tfnbs", "apply_nbs", "apply_cnbs", "apply_ni_tfnbs",
                "apply_fbc_tfnbs", "get_tfnbs_score", "nbs_bct",
                "create_prior_weights"],
    "sim": ["generate_fc_matrices", "generate_multisite_glm_dataset",
            "TopologyDatasetGenerator", "get_scenario", "list_scenarios"],
    "harmonize": ["combat_harmonize", "combat_fit", "combat_apply",
                  "design_diagnostics"],
    "loaders": ["load", "ManifestLoader", "LoadedDataset"],
}

# Top-level names that examples, tests and notebooks import today.
LEGACY_TOP_LEVEL = [
    "compute_p_val", "compute_p_val_glm", "compute_p_val_glm_multi",
    "compute_p_val_paired_glm", "compute_t_stat", "compute_t_stat_diff",
    "apply_tfnbs", "apply_ni_tfnbs", "nbs_bct", "get_tfnbs_score",
    "generate_fc_matrices", "generate_multisite_glm_dataset",
    "TopologyDatasetGenerator", "combat_harmonize", "block_mass",
    "fit_gpd_tail", "build_design_matrix", "design_diagnostics",
    "TailResult", "EEGData",
]


class TestPublicSurface(unittest.TestCase):

    def test_all_is_short_and_resolvable(self):
        self.assertLessEqual(len(conninfpy.__all__), 20)
        for name in conninfpy.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(conninfpy, name))

    def test_front_door_is_advertised(self):
        for name in ("analyze", "AnalyzeResult", "InferenceResult", "load"):
            self.assertIn(name, conninfpy.__all__)

    def test_modules_export_their_functions(self):
        for module, names in MODULE_EXPORTS.items():
            mod = importlib.import_module(f"conninfpy.{module}")
            for name in names:
                with self.subTest(module=module, name=name):
                    self.assertTrue(hasattr(mod, name))
                    if hasattr(mod, "__all__"):
                        self.assertIn(name, mod.__all__)

    def test_legacy_top_level_names_still_import(self):
        for name in LEGACY_TOP_LEVEL:
            with self.subTest(name=name):
                self.assertTrue(hasattr(conninfpy, name))

    def test_module_and_top_level_names_are_the_same_objects(self):
        self.assertIs(conninfpy.stats.compute_p_val, conninfpy.compute_p_val)
        self.assertIs(conninfpy.enhance.apply_tfnbs, conninfpy.apply_tfnbs)
        self.assertIs(conninfpy.sim.generate_fc_matrices,
                      conninfpy.generate_fc_matrices)

    def test_analysis_module_is_not_shadowed(self):
        """`conninfpy.analysis` stays a module; the function is `conninfpy.analyze`."""
        import conninfpy.analysis as analysis
        self.assertTrue(hasattr(analysis, "AnalyzeResult"))
        self.assertTrue(callable(conninfpy.analyze))


if __name__ == "__main__":
    unittest.main()
