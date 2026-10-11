"""ComBat harmonization and design diagnostics."""

from conninfpy.harmonize._arrays import flatten_upper, unflatten_upper
from conninfpy.harmonize.combat import (
    ComBatModel,
    CombatResult,
    combat_apply,
    combat_fit,
    combat_harmonize,
)
from conninfpy.harmonize.diagnostics import block_mass, compute_vif, design_diagnostics

__all__ = [
    "ComBatModel",
    "CombatResult",
    "combat_harmonize",
    "combat_fit",
    "combat_apply",
    "compute_vif",
    "design_diagnostics",
    "flatten_upper",
    "unflatten_upper",
    "block_mass",
]


def __getattr__(name):
    from conninfpy.harmonize import combat

    return getattr(combat, name)
