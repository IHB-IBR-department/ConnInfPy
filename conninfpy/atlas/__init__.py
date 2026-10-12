"""Atlas metadata, Brainnetome reference data and resource access."""

from conninfpy.atlas.brainnetome import BrainnetomeReference
from conninfpy.atlas.metadata import AtlasInfo, assign_coordinates
from conninfpy.atlas.resources import (
    bna_profile_paths,
    brainnetome_data_dir,
    brainnetome_resource_path,
    fetch_bna_behavioral_decoding,
    get_bna_246_nifti_path,
)

__all__ = [
    "AtlasInfo",
    "assign_coordinates",
    "BrainnetomeReference",
    "get_bna_246_nifti_path",
    "bna_profile_paths",
    "brainnetome_data_dir",
    "brainnetome_resource_path",
    "fetch_bna_behavioral_decoding",
]


def __getattr__(name):
    from conninfpy.atlas import metadata, resources

    for module in (metadata, resources):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
