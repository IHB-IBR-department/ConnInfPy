"""Canonical module layout and resource-lifetime contracts for the domain packages."""
from contextlib import ExitStack
import importlib
import pickle
from pathlib import Path
import zipfile

import pytest

import conninfpy
from conninfpy.atlas import resources
from conninfpy.loaders.manifest import resolve_path


# Root ``sys.modules`` aliases removed with the v2 layout: every name now
# lives only in its canonical subpackage.
LEGACY_ROOT_MODULES = [
    "pairwise_stats", "glm_stats", "_engine", "_progress",
    "_available_case", "_available_case_glm", "_estimates", "_cnbs",
    "_degenerate", "acceleration", "synth_datasets", "topologies",
    "utils", "_rng", "tfnbs_score", "nbs_score", "brainnetome",
    "annotate",
]
CANONICAL_MODULES = [
    "conninfpy.stats.pairwise", "conninfpy.stats.glm",
    "conninfpy.stats._engine", "conninfpy.stats._workers",
    "conninfpy.stats._missing_pairwise", "conninfpy.stats._missing_glm",
    "conninfpy.stats._estimates", "conninfpy.stats._cnbs",
    "conninfpy.stats._degenerate", "conninfpy.stats.acceleration",
    "conninfpy.sim.datasets", "conninfpy.sim.topologies",
    "conninfpy._shared.arrays", "conninfpy._shared.rng",
    "conninfpy.enhance.tfnbs", "conninfpy.enhance.nbs",
    "conninfpy.atlas.brainnetome", "conninfpy.interpret.annotate",
]


@pytest.mark.parametrize("legacy", LEGACY_ROOT_MODULES)
def test_legacy_root_modules_are_gone(legacy):
    # No root file and no ``sys.modules`` alias: the name is importable
    # only through its canonical subpackage.
    dotted = f"conninfpy.{legacy}"
    assert not (Path(conninfpy.__file__).parent / f"{legacy}.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(dotted)


@pytest.mark.parametrize("canonical", CANONICAL_MODULES)
def test_canonical_modules_import(canonical):
    module = importlib.import_module(canonical)
    assert module.__name__ == canonical


@pytest.mark.parametrize("module,name,current", [
    ("conninfpy.atlas", "AtlasInfo", conninfpy.AtlasInfo),
    ("conninfpy.harmonize", "ComBatModel", conninfpy.ComBatModel),
    ("conninfpy.harmonize", "CombatResult", conninfpy.CombatResult),
    ("conninfpy.stats.pairwise", "TestType", conninfpy.TestType),
    ("conninfpy.stats.pairwise", "StatMethod", conninfpy.StatMethod),
    ("conninfpy.stats.glm", "GLMStatType", conninfpy.GLMStatType),
    ("conninfpy.atlas.brainnetome", "BrainnetomeReference", conninfpy.BrainnetomeReference),
    ("conninfpy.interpret.annotate", "RoiAnnotation", conninfpy.RoiAnnotation),
])
def test_saved_objects_resolve_canonical_class_paths(module, name, current):
    # A protocol-0 GLOBAL exercises the exact lookup used by pickles; the
    # classes must be reachable at their canonical, documented paths.
    assert pickle.loads(f"c{module}\n{name}\n.".encode()) is current


def test_tfnbs_family_has_explicit_implementations():
    ni = importlib.import_module("conninfpy.enhance.ni_tfnbs")
    fbc = importlib.import_module("conninfpy.enhance.fbc_tfnbs")
    assert conninfpy.get_network_informed_tfnbs_score is ni.get_network_informed_tfnbs_score
    assert conninfpy.get_fbc_tfnbs_score is fbc.get_fbc_tfnbs_score


@pytest.mark.parametrize("token", ["schaefer100", "schaefer200", "schaefer400"])
def test_manifest_and_atlas_use_the_same_resource_path(token, tmp_path):
    resolved = Path(resolve_path(f"BUILTIN:{token}", tmp_path))
    assert resolved == resources.builtin_atlas_path(token)
    assert resolved.is_file()


def test_zip_resource_survives_getter_for_lazy_read(tmp_path, monkeypatch):
    filename = "schaefer100_yeo7.csv"
    original = resources.bundled_path(filename)
    archive = tmp_path / "atlas.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(original, filename)
    resources.bundled_path.cache_clear()
    with zipfile.ZipFile(archive) as z, ExitStack() as stack:
        monkeypatch.setattr(resources, "_RESOURCE_CONTEXTS", stack)
        monkeypatch.setattr(resources.resources, "files", lambda package: zipfile.Path(z))
        extracted = resources.bundled_path(filename)
        assert extracted.is_file()
        assert extracted.read_text(encoding="utf-8").startswith("roi_id,")
        assert resources.bundled_path(filename) == extracted
    resources.bundled_path.cache_clear()


@pytest.mark.skipif(
    not resources.atlas_resource_available("bna246_hierarchy.csv"),
    reason="Brainnetome cache not downloaded; run python scripts/download_atlases.py",
)
def test_hierarchy_readers_return_independent_tables():
    first = resources.read_bna_hierarchy()
    second = resources.read_bna_hierarchy()
    original = second.loc[1, "name"]
    first.loc[1, "name"] = "changed by caller"
    assert second.loc[1, "name"] == original
