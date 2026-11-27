conninfpy.atlas
===============

Atlas metadata and reference data.

- :class:`~conninfpy.atlas.AtlasInfo` holds per-region labels, networks,
  coordinates and hemispheres: the bundled Schaefer-100/200/400 (Yeo-7) and
  Brainnetome BNA-246 atlases, or any CSV, label image or nilearn atlas.
  ``AtlasInfo.bna_246()`` groups its regions by the Brainnetome Yeo-7 division
  (``networks='yeo17'`` or ``'lobe'`` for the other groupings). Changed in
  October 2026: the default was the lobe grouping.
- :func:`~conninfpy.atlas.assign_coordinates` places MNI coordinates in a
  reference atlas.
- :class:`~conninfpy.atlas.BrainnetomeReference` is the BNA-246 reference used
  by :mod:`conninfpy.interpret`: hierarchy (subregion → gyrus → lobe,
  homologs, cytoarchitectonic descriptions, Yeo crosswalk), label images,
  BrainMap profiles and the optional probabilistic and tractography maps.

Data locations, the integrity manifest and caches are described in
:doc:`architecture`. Brainnetome data are for non-commercial use with
attribution and share-alike (``conninfpy/atlas_data/NOTICE_BRAINNETOME.txt``);
cite Fan et al. (2016) and, for the BrainMap profiles, Laird et al. (2005,
2011) (:doc:`references`).

.. automodule:: conninfpy.atlas
   :members:
   :exclude-members: labels, networks, coords, hemisphere, source, image, label_ids, assignment
   :show-inheritance:
