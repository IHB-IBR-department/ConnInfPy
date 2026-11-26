"""Streamlit data step: the selected dataset is the one that is loaded.

The app keeps the draft ``current_settings_hash`` separate from the signature
of the loaded dataset. Every selected source must be represented in that draft
signature, while loaded data survive edits. The built-in
ABIDE-mini demo follows its manifest (``params.missing: drop_subjects``). No
inference runs here.
"""
import unittest
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # the [app] extra is not installed
    AppTest = None

APP = Path(__file__).resolve().parents[1] / "conninfpy" / "app" / "streamlit_app.py"


@unittest.skipIf(AppTest is None, "streamlit is not installed")
class TestManifestSettingsHash(unittest.TestCase):

    def test_switching_manifest_changes_settings_hash(self):
        at = AppTest.from_file(str(APP), default_timeout=120).run()
        self.assertFalse(at.exception)
        mode = next(r for r in at.radio if r.label == "Select Ingestion Method")
        mode.set_value("Dataset Manifest (data.yaml)").run()
        manifest = next(s for s in at.selectbox if s.label == "Select Ingestion Manifest File (.yaml)")
        first = manifest.value
        before = at.session_state["current_settings_hash"]
        self.assertIn(first, before)
        other = next(o for o in manifest.options if o != first)
        manifest.set_value(other).run()
        after = at.session_state["current_settings_hash"]
        self.assertNotEqual(before, after)
        self.assertIn(other, after)


@unittest.skipIf(AppTest is None, "streamlit is not installed")
class TestBuiltinAbideMini(unittest.TestCase):

    def test_builtin_demo_loads_without_the_subject_with_missing_connectivity(self):
        at = AppTest.from_file(str(APP), default_timeout=300).run()
        next(r for r in at.radio if r.label == "Select Ingestion Method").set_value("Built-in data").run()
        picker = next(s for s in at.selectbox if s.label == "Select Template Structure")
        self.assertEqual(picker.value, "ABIDE-mini")
        self.assertEqual(picker.options, ["ABIDE-mini", "OpenClose"])
        next(b for b in at.button if b.label == "Load, Preprocess & Validate Data").click().run()
        self.assertFalse(at.exception)
        self.assertFalse([e.value for e in at.error])
        # 204 subjects; one has ROI 82 without signal (as in datasets/abide_mini.yaml).
        self.assertEqual(at.session_state["connectivity_data"].shape, (203, 100, 100))
        notes = at.session_state["dataset_load_notes"]
        self.assertTrue(any("ROI 82" in note for note in notes))
        self.assertTrue(any("n_valid_tr" in note for note in notes))
        self.assertTrue(any("ROI 82" in i.value for i in at.info))      # shown to the user


if __name__ == "__main__":
    unittest.main()
