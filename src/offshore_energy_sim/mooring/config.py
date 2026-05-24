"""Configuration helpers for linear mooring models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from offshore_energy_sim.core.cases import RodmFrequencyCase
from offshore_energy_sim.mooring.linear import (
    LinearMooringMatrix,
    NodalMooringAttachment,
    ReducedMooringTerms,
    build_nodal_mooring_reduced_terms,
)
from offshore_energy_sim.structure import StructuralReductionResult


MooringProvider = Callable[[RodmFrequencyCase, StructuralReductionResult], ReducedMooringTerms | None]


def mooring_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the ``mooring`` section from a full config or the section itself."""

    section = config.get("mooring", config)
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise ValueError("mooring config section must be a mapping")
    return section


def is_mooring_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether a mooring config section requests active mooring terms."""

    section = mooring_section(config)
    return bool(section.get("enabled", False))


def retained_full_dofs_from_case(case: RodmFrequencyCase) -> tuple[int, ...]:
    """Return full-DOF indices retained by the RODM structural reduction."""

    removed = set(int(value) for value in case.removed_full_dofs_zero_based)
    return tuple(index for index in range(case.full_dofs_per_node) if index not in removed)


def retained_full_dofs_from_config(
    config: Mapping[str, Any],
    case: RodmFrequencyCase,
) -> tuple[int, ...]:
    """Return retained full DOFs from mooring config or infer them from case."""

    section = mooring_section(config)
    values = section.get("retained_full_dofs_zero_based")
    if values is None:
        return retained_full_dofs_from_case(case)
    retained = tuple(int(value) for value in values)
    if not retained:
        raise ValueError("mooring retained_full_dofs_zero_based must not be empty")
    if len(set(retained)) != len(retained):
        raise ValueError("mooring retained_full_dofs_zero_based must be unique")
    if min(retained) < 0 or max(retained) >= case.full_dofs_per_node:
        raise ValueError("mooring retained_full_dofs_zero_based is outside full DOF range")
    return retained


def build_mooring_attachments_from_config(
    config: Mapping[str, Any],
    *,
    dof_count: int = 6,
) -> tuple[NodalMooringAttachment, ...]:
    """Build linear nodal mooring attachments from a YAML-like mapping."""

    section = mooring_section(config)
    if not is_mooring_enabled(section):
        return ()
    model = str(section.get("model", "linear_matrix")).lower()
    if model != "linear_matrix":
        raise ValueError("only mooring.model='linear_matrix' is currently supported")
    attachments_config = section.get("attachments", ())
    if not isinstance(attachments_config, Sequence) or isinstance(attachments_config, (str, bytes)):
        raise ValueError("mooring.attachments must be a list")
    attachments: list[NodalMooringAttachment] = []
    for index, item in enumerate(attachments_config):
        if not isinstance(item, Mapping):
            raise ValueError(f"mooring attachment {index} must be a mapping")
        node = int(item["node_one_based"])
        name = str(item.get("name", f"mooring_{index + 1}"))
        matrix = LinearMooringMatrix(
            stiffness=_matrix_from_config(item.get("stiffness"), dof_count, "stiffness"),
            damping=_matrix_from_config(item.get("damping"), dof_count, "damping"),
            pretension=_vector_from_config(item.get("pretension"), dof_count, "pretension"),
            dof_count=dof_count,
            metadata={
                "name": name,
                "node_one_based": node,
                "source": "mooring_config",
            },
        )
        attachments.append(
            NodalMooringAttachment(
                node_one_based=node,
                matrix=matrix,
                name=name,
            )
        )
    return tuple(attachments)


def build_reduced_mooring_terms_from_config(
    config: Mapping[str, Any],
    case: RodmFrequencyCase,
    structural: StructuralReductionResult,
) -> ReducedMooringTerms | None:
    """Build reduced mooring terms for one case/structural reduction pair."""

    if not is_mooring_enabled(config):
        return None
    section = mooring_section(config)
    dof_count = int(section.get("dof_count", case.full_dofs_per_node))
    attachments = build_mooring_attachments_from_config(section, dof_count=dof_count)
    retained = retained_full_dofs_from_config(section, case)
    reduced = build_nodal_mooring_reduced_terms(
        attachments,
        total_nodes=case.total_nodes,
        retained_full_dofs_zero_based=retained,
        transformation=structural.transformation,
        master_dofs=structural.master_dofs,
        slave_dofs=structural.slave_dofs,
        reverse_master_order=structural.reverse_master_order_for_reconstruction,
    )
    return ReducedMooringTerms(
        stiffness=reduced.stiffness,
        damping=reduced.damping,
        pretension=reduced.pretension,
        metadata={
            **dict(reduced.metadata),
            "enabled": True,
            "model": "linear_matrix",
            "attachment_count": len(attachments),
            "attachment_nodes_one_based": tuple(item.node_one_based for item in attachments),
            "attachment_names": tuple(item.name for item in attachments),
            "retained_full_dofs_zero_based": retained,
            "dof_count": dof_count,
        },
    )


def build_mooring_provider_from_config(config: Mapping[str, Any]) -> MooringProvider | None:
    """Return a provider callback for adapter/time-domain solvers."""

    if not is_mooring_enabled(config):
        return None

    def provider(
        case: RodmFrequencyCase,
        structural: StructuralReductionResult,
    ) -> ReducedMooringTerms | None:
        return build_reduced_mooring_terms_from_config(config, case, structural)

    return provider


def summarize_reduced_mooring_terms(terms: ReducedMooringTerms | None) -> dict[str, object]:
    """Return JSON-friendly summary values for reduced mooring terms."""

    if terms is None:
        return {"enabled": False}
    return {
        "enabled": bool(terms.enabled),
        "metadata": dict(terms.metadata),
        "stiffness_frobenius_norm": float(np.linalg.norm(terms.stiffness)),
        "stiffness_trace": float(np.trace(terms.stiffness)),
        "damping_frobenius_norm": float(np.linalg.norm(terms.damping)),
        "damping_trace": float(np.trace(terms.damping)),
        "pretension_norm": float(np.linalg.norm(terms.pretension)),
    }


def _matrix_from_config(section: object, dof_count: int, name: str) -> np.ndarray:
    if section is None:
        return np.zeros((dof_count, dof_count), dtype=float)
    if isinstance(section, Mapping):
        if "diagonal" in section:
            diagonal = np.asarray(section["diagonal"], dtype=float).reshape(-1)
            if diagonal.size != dof_count:
                raise ValueError(f"mooring {name}.diagonal must have length {dof_count}")
            return np.diag(diagonal)
        if "matrix" in section:
            matrix = np.asarray(section["matrix"], dtype=float)
        else:
            raise ValueError(f"mooring {name} must define 'diagonal' or 'matrix'")
    else:
        matrix = np.asarray(section, dtype=float)
    if matrix.shape != (dof_count, dof_count):
        raise ValueError(f"mooring {name}.matrix must have shape ({dof_count}, {dof_count})")
    return matrix


def _vector_from_config(section: object, dof_count: int, name: str) -> np.ndarray:
    if section is None:
        return np.zeros(dof_count, dtype=float)
    values = section["values"] if isinstance(section, Mapping) and "values" in section else section
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != dof_count:
        raise ValueError(f"mooring {name} must have length {dof_count}")
    return vector
