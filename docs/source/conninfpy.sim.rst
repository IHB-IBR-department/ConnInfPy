conninfpy.sim
=============

Synthetic connectivity data with planted effects and the 20-scenario
topology library used by the examples, simulation validation and benchmarks.

- :func:`~conninfpy.synth_datasets.generate_fc_matrices` — two groups (or paired conditions) of correlation matrices with a masked effect
- :func:`~conninfpy.synth_datasets.generate_multisite_glm_dataset` — multi-site GLM data with site offsets and site/regressor imbalance
- :class:`~conninfpy.synth_datasets.ModularDatasetGenerator` — modular network generator
- :class:`~conninfpy.topologies.TopologyDatasetGenerator`, :func:`~conninfpy.topologies.get_scenario`,
  :func:`~conninfpy.topologies.list_scenarios` — controlled effect topologies (hub, rich club, chain, gradient, ...)

Full documentation lives with the implementations in
:mod:`conninfpy.synth_datasets` and :mod:`conninfpy.topologies`.
