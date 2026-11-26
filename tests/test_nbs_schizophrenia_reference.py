"""NBS schizophrenia example: ConnInfPy against the MATLAB NBS v1.2 reference.

The dataset is the example distributed with the MATLAB NBS toolbox
(Zalesky, Fornito & Bullmore 2010): 12 people with schizophrenia, 15 healthy
controls, 74 AAL regions. ``tests/data/nbs_schizophrenia_matlab_reference.json``
holds the NBS v1.2 output for the inputs its GUI pre-fills (t-test, contrast
[-1,1] = controls > patients, threshold 3.1, 5000 permutations), written by
``scripts/nbs_schizophrenia_matlab_reference.m``. MATLAB is not needed here.

NBS's t-test is Student's t (pooled variance): the comparison goes through
``compute_p_val_glm`` (intercept + group indicator) and through the two-sample
permutation test with ``var_equal=True``; the default two-sample statistic is
Welch's t.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from conninfpy import compute_p_val_glm
from conninfpy.loaders.manifest import ManifestLoader

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # the [app] extra is not installed
    AppTest = None

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "conninfpy" / "app" / "streamlit_app.py"
MANIFEST = REPO / "datasets" / "nbs_schizophrenia.yaml"
REFERENCE = REPO / "tests" / "data" / "nbs_schizophrenia_matlab_reference.json"
N_NODES = 74
THRESHOLD = 3.1


def _load():
    """Connectivity (zero diagonal), HC indicator, atlas, MATLAB reference."""
    dataset = ManifestLoader(str(MANIFEST)).load()
    Y = np.array(dataset.data, dtype=np.float64)
    idx = np.arange(Y.shape[1])
    Y[:, idx, idx] = 0.0
    hc = (dataset.pheno["group"] == "HC").to_numpy(dtype=np.float64)
    reference = json.loads(REFERENCE.read_text())
    return Y, hc, dataset, reference


def _edge_set(edges) -> set[tuple[int, int]]:
    """0-based (i, j), i < j, from the reference edge list(s)."""
    return {(int(i), int(j)) for component in edges for i, j in np.asarray(component).reshape(-1, 2)}


class TestNBSSchizophreniaDataset(unittest.TestCase):
    def test_manifest_loads_with_atlas(self):
        Y, hc, dataset, _ = _load()
        self.assertEqual(Y.shape, (27, N_NODES, N_NODES))
        self.assertEqual(dataset.data_kind, "correlation")
        self.assertEqual(int(hc.sum()), 15)
        self.assertEqual(list(dataset.pheno["group"][:12]), ["SZ"] * 12)
        atlas = dataset.atlas
        self.assertIsNotNone(atlas)
        self.assertEqual(len(atlas), N_NODES)
        self.assertEqual(len(set(atlas.labels)), N_NODES)
        # MNI orientation: left hemisphere at negative x.
        left = np.array([h == "L" for h in atlas.hemisphere])
        self.assertTrue(np.all(atlas.coords[left, 0] < 0))
        self.assertTrue(np.all(atlas.coords[~left, 0] > 0))


class TestNBSSchizophreniaAgainstMatlab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Y, cls.hc, cls.dataset, cls.ref = _load()
        cls.iu = np.triu_indices(N_NODES, 1)
        # NBS tests the contrast direction only: one-sided, per-tail null.
        cls.results = {
            stat: compute_p_val_glm(
                cls.Y, interest=cls.hc, method="nbs", threshold=THRESHOLD,
                nbs_stat=stat, n_permutations=2000, acceleration=None,
                rng=1, use_mp=False, two_tailed=False,
            )
            for stat in ("extent", "intensity")
        }
        cls.result = cls.results["extent"]

    def test_edge_t_statistics_match(self):
        t = self.result.stat_signed[self.iu]
        np.testing.assert_allclose(t, np.asarray(self.ref["t_upper"]), rtol=0, atol=1e-9)

    def test_component_matches(self):
        t_ref = np.asarray(self.ref["t_upper"])
        # No edge sits at the threshold, so the comparison is not fragile.
        self.assertGreater(np.min(np.abs(t_ref - THRESHOLD)), 1e-6)
        expected = _edge_set(self.ref["extent"]["edges"])
        self.assertEqual(self.ref["extent"]["n_components"], 1)
        self.assertEqual(len(expected), 40)
        significant = {
            (int(i), int(j)) for i, j in zip(*self.iu, strict=True)
            if self.result.positive[i, j] < 0.05
        }
        self.assertEqual(significant, expected)

    def test_component_p_values_within_monte_carlo_error(self):
        for stat in ("extent", "intensity"):
            with self.subTest(stat=stat):
                p_ref = float(np.atleast_1d(self.ref[stat]["p"])[0])
                edges = _edge_set(self.ref[stat]["edges"])
                p = {float(self.results[stat].positive[i, j]) for i, j in edges}
                self.assertEqual(len(p), 1)
                # SE of the difference of two independent estimates near
                # p = 0.04 (2000 and 5000 permutations) is about 0.005;
                # allow four SEs.
                self.assertAlmostEqual(p.pop(), p_ref, delta=0.02)

    def test_two_sample_student_path_matches_matlab(self):
        """The two-sample permutation test with Student's t is NBS's t-test."""
        from conninfpy import compute_p_val

        hc = self.hc.astype(bool)
        res = compute_p_val(
            self.Y[~hc], self.Y[hc], test_type="two-sample", var_equal=True,
            method="nbs", threshold=THRESHOLD, nbs_stat="extent",
            n_permutations=2000, acceleration=None, rng=1, use_mp=False,
            two_tailed=False,
        ).one_sided("positive")
        np.testing.assert_allclose(res.stat_signed[self.iu], np.asarray(self.ref["t_upper"]),
                                   rtol=0, atol=1e-9)
        edges = _edge_set(self.ref["extent"]["edges"])
        significant = {(int(i), int(j)) for i, j in zip(*self.iu, strict=True) if res.positive[i, j] < 0.05}
        self.assertEqual(significant, edges)
        p = {float(res.positive[i, j]) for i, j in edges}
        self.assertEqual(len(p), 1)
        self.assertAlmostEqual(p.pop(), float(np.atleast_1d(self.ref["extent"]["p"])[0]), delta=0.02)
        self.assertEqual(res.design["t_statistic"], "student")
        # the component report of this path is the same MATLAB component
        row = res.component_table().iloc[0]
        self.assertEqual((row["tail"], row["n_edges"]), ("positive", 40))
        self.assertAlmostEqual(row["p_fwer"], float(np.atleast_1d(self.ref["extent"]["p"])[0]), delta=0.02)

    def test_component_report_is_the_matlab_component(self):
        """NETSTAT-15 AC #1: one component of 40 edges; statistic and p as in NBS v1.2."""
        t_ref = np.asarray(self.ref["t_upper"])
        edges = _edge_set(self.ref["extent"]["edges"])
        nodes = sorted({n for edge in edges for n in edge})
        for stat in ("extent", "intensity"):
            with self.subTest(stat=stat):
                res = self.results[stat]
                report = res.component_report
                self.assertEqual(report["n_components"], {"positive": 1, "negative": 0})
                self.assertEqual(report["stat_type"], stat)
                self.assertEqual(report["threshold"], THRESHOLD)
                table = res.component_table()
                self.assertEqual(len(table), 1)
                row = table.iloc[0]
                self.assertEqual((row["tail"], row["component_id"]), ("positive", 1))
                self.assertEqual(row["n_edges"], len(edges))
                self.assertEqual(row["n_nodes"], len(nodes))
                self.assertEqual(row["nodes"], nodes)
                labelled = {(int(i), int(j)) for i, j in zip(*self.iu, strict=True)
                            if report["labels"]["positive"][i, j] == 1}
                self.assertEqual(labelled, edges)
                if stat == "extent":
                    self.assertEqual(row["statistic"], 40.0)
                else:
                    self.assertAlmostEqual(
                        row["statistic"], float(np.sum(t_ref[t_ref > THRESHOLD] - THRESHOLD)), places=9)
                # the component's p is the p of every one of its edges in the map ...
                self.assertEqual({float(res.positive[i, j]) for i, j in edges}, {row["p_fwer"]})
                # ... and MATLAB's within Monte-Carlo error (2000 vs 5000 permutations)
                self.assertAlmostEqual(row["p_fwer"], float(np.atleast_1d(self.ref[stat]["p"])[0]), delta=0.02)
                self.assertEqual(bool(row["significant"]), row["p_fwer"] <= 0.05)

    def test_component_id_in_edge_table_and_roi_names(self):
        res = self.results["extent"]
        edges = res.significant_edges(atlas=self.dataset.atlas)
        self.assertEqual(len(edges), 40)
        self.assertTrue((edges["component_id"] == 1).all())
        self.assertTrue((edges["component_tail"] == "positive").all())
        table = res.component_table(self.dataset.atlas)
        names = table.iloc[0]["roi_names"]
        self.assertEqual(len(names), 29)
        self.assertEqual(names[0], "Precentral_L")
        # the AAL-74 example carries no network labels: nothing is invented
        self.assertNotIn("networks", table.columns)
        self.assertNotIn("network_pairs", table.columns)

    def test_intensity_is_mass_above_threshold(self):
        """NBS 'Intensity' = sum of (t - threshold) over the component."""
        t_ref = np.asarray(self.ref["t_upper"])
        expected = float(np.sum(t_ref[t_ref > THRESHOLD] - THRESHOLD))
        from conninfpy.enhance.nbs import get_nbs_score
        T = self.result.stat_signed.copy()
        T[T < 0] = 0.0
        score = get_nbs_score(T, threshold=THRESHOLD, stat_type="intensity")
        i, j = next(iter(_edge_set(self.ref["intensity"]["edges"])))
        self.assertAlmostEqual(float(score[i, j]), expected, places=9)


@unittest.skipIf(AppTest is None, "streamlit is not installed")
class TestBuiltinNBSSchizophrenia(unittest.TestCase):

    def test_builtin_template_loads_raw_correlations_with_its_atlas(self):
        at = AppTest.from_file(str(APP), default_timeout=300).run()
        next(r for r in at.radio if r.label == "Select Ingestion Method").set_value("Dataset Manifest (data.yaml)").run()
        manifests = next(s for s in at.selectbox if s.label == "Select Ingestion Manifest File (.yaml)")
        manifests.set_value("datasets/nbs_schizophrenia.yaml").run()
        next(b for b in at.button if b.label == "Load, Preprocess & Validate Data").click().run()
        self.assertFalse(at.exception)
        self.assertFalse([e.value for e in at.error])
        Y = at.session_state["connectivity_data"]
        self.assertEqual(Y.shape, (27, N_NODES, N_NODES))
        # Stored as raw correlations (the NBS scale); the diagonal is zeroed on loading.
        self.assertEqual(at.session_state["connectivity_data_kind"], "correlation")
        self.assertTrue(np.all(np.abs(Y) <= 1.0))
        # The dataset atlas wins over the sidebar atlas and keeps its own coordinates.
        atlas = at.session_state["dataset_atlas"]
        self.assertEqual(len(atlas), N_NODES)
        self.assertEqual(atlas.labels[0], "Precentral_L")
        self.assertLess(atlas.coords[0, 0], 0)


if __name__ == "__main__":
    unittest.main()
