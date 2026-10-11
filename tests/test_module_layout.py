"""Compatibility and resource-lifetime contracts for the domain packages."""
from contextlib import ExitStack
import importlib
import pickle
from pathlib import Path
import zipfile

import nibabel as nib
import pytest

import conninfpy
from conninfpy.atlas import resources
from conninfpy.loaders.manifest import resolve_path


@pytest.mark.parametrize("legacy,canonical", [
    ("pairwise_stats", "stats.pairwise"),
    ("glm_stats", "stats.glm"),
    ("_engine", "stats._engine"),
    ("_progress", "stats._workers"),
    ("_available_case", "stats._missing_pairwise"),
    ("_available_case_glm", "stats._missing_glm"),
    ("_estimates", "stats._estimates"),
    ("_cnbs", "stats._cnbs"),
    ("_degenerate", "stats._degenerate"),
    ("acceleration", "stats.acceleration"),
    ("synth_datasets", "sim.datasets"),
    ("topologies", "sim.topologies"),
    ("utils", "_shared.arrays"),
    ("_rng", "_shared.rng"),
    ("tfnbs_score", "enhance.tfnbs"),
    ("nbs_score", "enhance.nbs"),
    ("brainnetome", "atlas.brainnetome"),
    ("annotate", "interpret.annotate"),
])
def test_legacy_modules_share_objects_and_globals(legacy, canonical):
    old = importlib.import_module(f"conninfpy.{legacy}")
    new = importlib.import_module(f"conninfpy.{canonical}")
    assert old is new


@pytest.mark.parametrize("module,name,current", [
    ("conninfpy.atlas", "AtlasInfo", conninfpy.AtlasInfo),
    ("conninfpy.harmonize", "ComBatModel", conninfpy.ComBatModel),
    ("conninfpy.harmonize", "CombatResult", conninfpy.CombatResult),
    ("conninfpy.pairwise_stats", "TestType", conninfpy.TestType),
    ("conninfpy.pairwise_stats", "StatMethod", conninfpy.StatMethod),
    ("conninfpy.glm_stats", "GLMStatType", conninfpy.GLMStatType),
    ("conninfpy.brainnetome", "BrainnetomeReference", conninfpy.BrainnetomeReference),
    ("conninfpy.annotate", "RoiAnnotation", conninfpy.RoiAnnotation),
])
def test_saved_objects_can_resolve_old_class_paths(module, name, current):
    # A protocol-0 GLOBAL exercises the exact lookup used by older pickles.
    assert pickle.loads(f"c{module}\n{name}\n.".encode()) is current


def test_tfnbs_family_has_explicit_implementations():
    ni = importlib.import_module("conninfpy.enhance.ni_tfnbs")
    fbc = importlib.import_module("conninfpy.enhance.fbc_tfnbs")
    assert conninfpy.get_network_informed_tfnbs_score is ni.get_network_informed_tfnbs_score
    assert conninfpy.get_fbc_tfnbs_score is fbc.get_fbc_tfnbs_score


@pytest.mark.parametrize("token", ["bna_246_1mm", "bna_246_2mm", "bna246.csv",
                                   "schaefer100", "schaefer200", "schaefer400"])
def test_manifest_and_atlas_use_the_same_resource_path(token, tmp_path):
    resolved = Path(resolve_path(f"BUILTIN:{token}", tmp_path))
    assert resolved == resources.builtin_atlas_path(token)
    assert resolved.is_file()


def test_zip_resource_survives_getter_for_lazy_nifti_read(tmp_path, monkeypatch):
    filename = "BN_Atlas_246_2mm.nii.gz"
    original = resources.bundled_path(filename)
    archive = tmp_path / "atlas.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(original, filename)
    resources.bundled_path.cache_clear()
    with zipfile.ZipFile(archive) as z, ExitStack() as stack:
        monkeypatch.setattr(resources, "_RESOURCE_CONTEXTS", stack)
        monkeypatch.setattr(resources.resources, "files", lambda package: zipfile.Path(z))
        extracted = resources.get_bna_246_nifti_path(2)
        image = nib.load(str(extracted))
        assert extracted.is_file()
        assert image.get_fdata().max() == 246
        assert resources.get_bna_246_nifti_path(2) == extracted
    resources.bundled_path.cache_clear()


def test_hierarchy_readers_return_independent_tables():
    first = resources.read_bna_hierarchy()
    second = resources.read_bna_hierarchy()
    original = second.loc[1, "name"]
    first.loc[1, "name"] = "changed by caller"
    assert second.loc[1, "name"] == original
