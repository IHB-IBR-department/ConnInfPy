"""Synthetic connectivity data and the topology scenario library.

Generators for modular networks with planted effects, used by the
examples, the simulation validation and the benchmarks.
"""

from conninfpy.sim.datasets import (
    ModularDatasetGenerator,
    generate_fc_matrices,
    generate_multisite_glm_dataset,
)
from conninfpy.sim.topologies import (
    TopologyDataset,
    TopologyDatasetGenerator,
    TopologyScenario,
    get_scenario,
    get_scenarios,
    list_scenarios,
)

__all__ = [
    "generate_fc_matrices",
    "generate_multisite_glm_dataset",
    "ModularDatasetGenerator",
    "TopologyDataset",
    "TopologyDatasetGenerator",
    "TopologyScenario",
    "get_scenario",
    "get_scenarios",
    "list_scenarios",
]
