"""Synthetic connectivity data and the topology scenario library.

Generators for modular networks with planted effects, used by the
examples, the simulation validation and the benchmarks.
"""
from .synth_datasets import (
    ModularDatasetGenerator,
    generate_fc_matrices,
    generate_multisite_glm_dataset,
)
from .topologies import (
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
