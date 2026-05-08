"""Configuration loading helpers for simulation cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from offshore_energy_sim.core.cases import (
    MasterNodeRule,
    RodmFrequencyCase,
    StructuralMatrixPaths,
)


def load_case_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML case configuration file.

    The current repository uses YAML for documented benchmark cases. This
    helper centralizes reading so later command-line runners and validation
    scripts do not duplicate parser setup.
    """

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Case config must be a mapping: {path}")
    return data


def build_rodm_frequency_case_from_config(
    config_path: str | Path,
    *,
    reverse_hydrodynamic_node_order: bool | None = None,
    use_hydrostatic: bool | None = None,
    frequency_index: int | None = None,
) -> RodmFrequencyCase:
    """Build a ``RodmFrequencyCase`` from a YAML case configuration.

    Numerical-result expectation: unchanged relative to the hard-coded
    reference-case builder when the same configuration values are used.
    """

    config = load_case_config(config_path)
    case_config = config["case"]
    geometry = config["geometry"]
    inputs = config.get("inputs", {})
    hydrodynamics = config.get("hydrodynamics", {})
    structure = config.get("structure", {})
    solver = config.get("solver", {})
    master_rule = geometry["master_node_rule"]

    hydrodynamic_dataset = hydrodynamics.get(
        "dataset",
        inputs.get("hydrodynamic_dataset"),
    )
    mass_matrix = structure.get(
        "mass_matrix",
        inputs.get("structural_mass_matrix"),
    )
    stiffness_matrix = structure.get(
        "stiffness_matrix",
        inputs.get("structural_stiffness_matrix"),
    )
    if hydrodynamic_dataset is None or mass_matrix is None or stiffness_matrix is None:
        raise KeyError("RODM config must define hydrodynamic and structural matrix paths.")

    return RodmFrequencyCase(
        case_id=str(case_config["id"]),
        total_nodes=int(geometry["total_nodes"]),
        full_dofs_per_node=int(geometry["full_dofs_per_node"]),
        retained_dofs_per_node=int(geometry["retained_dofs_per_node"]),
        removed_full_dofs_zero_based=tuple(int(value) for value in geometry["removed_dofs_zero_based"]),
        master_node_rule=MasterNodeRule(
            first_node=int(master_rule["first_node"]),
            node_interval=int(master_rule["node_interval"]),
            count=int(master_rule["count"]),
        ),
        hydrodynamic_dataset=Path(hydrodynamic_dataset),
        structural_matrices=StructuralMatrixPaths(
            mass=Path(mass_matrix),
            stiffness=Path(stiffness_matrix),
        ),
        hydrodynamic_nodes=int(hydrodynamics.get("hydrodynamic_nodes", master_rule["count"])),
        hydrodynamic_dof_to_remove_zero_based=int(
            hydrodynamics.get(
                "hydrodynamic_dof_to_remove_zero_based",
                int(geometry["full_dofs_per_node"]) - 1,
            )
        ),
        mass_blend_beta=float(structure.get("mass_blend_beta", 0.0)),
        use_hydrostatic=bool(
            solver.get("use_hydrostatic", True)
            if use_hydrostatic is None
            else use_hydrostatic
        ),
        frequency_index=int(
            solver.get("frequency_index", 0)
            if frequency_index is None
            else frequency_index
        ),
        reverse_hydrodynamic_node_order=bool(
            hydrodynamics.get("reverse_hydrodynamic_node_order", False)
            if reverse_hydrodynamic_node_order is None
            else reverse_hydrodynamic_node_order
        ),
    )
