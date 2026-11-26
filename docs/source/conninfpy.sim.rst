conninfpy.sim
=============

Synthetic connectivity data with planted effects and the 20-scenario
topology library used by the examples, simulation validation and benchmarks.

- :func:`~conninfpy.sim.datasets.generate_fc_matrices` — two groups (or paired conditions) of correlation matrices with a masked effect
- :func:`~conninfpy.sim.datasets.generate_multisite_glm_dataset` — multi-site GLM data with site offsets and site/regressor imbalance
- :class:`~conninfpy.sim.datasets.ModularDatasetGenerator` — modular network generator

Synthetic datasets — :mod:`conninfpy.sim.datasets`
--------------------------------------------------

.. automodule:: conninfpy.sim.datasets
   :members:
   :undoc-members:
   :show-inheritance:

Topology scenarios — :mod:`conninfpy.sim.topologies`
----------------------------------------------------

Named topology scenarios for benchmarking enhancement methods on
synthetic connectivity. Promoted from ``examples/`` to the public API
on 2026-04-18.

Provides 19 named scenarios covering the topologies most likely to
arise in real connectivity data: within-module-dense,
between-modules-dense, hub, rich-club, scattered, chain,
gradient-core-periphery, partial-bipartite, fragmented-within-module,
cross-block-connected-chain, plus secondary variants.

Public surface:

- :class:`~conninfpy.sim.topologies.TopologyDatasetGenerator` — main generator class
- :class:`~conninfpy.sim.topologies.TopologyDataset` — output container (``.group1``, ``.group2``,
  ``.mask``, ``.net_labels``, ``.scenario``)
- :class:`~conninfpy.sim.topologies.TopologyScenario` — scenario specification
- :func:`~conninfpy.sim.topologies.list_scenarios` — return all scenario names
- :func:`~conninfpy.sim.topologies.get_scenarios` — return all scenario specs
- :func:`~conninfpy.sim.topologies.get_scenario` — look up a specific scenario by name

Power benchmarks (``examples/simulations/power/``) compare methods across
the topologies selected in each YAML configuration.

.. automodule:: conninfpy.sim.topologies
   :members:
   :undoc-members:
   :show-inheritance:
