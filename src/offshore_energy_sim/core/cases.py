"""Case configuration models for hydroelastic simulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MasterNodeRule:
    """Rule for generating one-based master/control node IDs."""

    first_node: int
    node_interval: int
    count: int


@dataclass(frozen=True)
class StructuralMatrixPaths:
    """Mass and stiffness matrix file paths."""

    mass: Path
    stiffness: Path


@dataclass(frozen=True)
class RodmFrequencyCase:
    """Inputs needed by the current RODM frequency-domain workflow."""

    case_id: str
    total_nodes: int
    full_dofs_per_node: int
    retained_dofs_per_node: int
    removed_full_dofs_zero_based: tuple[int, ...]
    master_node_rule: MasterNodeRule
    hydrodynamic_dataset: Path
    structural_matrices: StructuralMatrixPaths
    hydrodynamic_nodes: int = 10
    hydrodynamic_dof_to_remove_zero_based: int = 5
    mass_blend_beta: float = 0.0
    use_hydrostatic: bool = True
    frequency_index: int = 0
    reverse_hydrodynamic_node_order: bool = False
