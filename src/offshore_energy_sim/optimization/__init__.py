"""Optimization workflows and design studies."""

from offshore_energy_sim.optimization.boundary18_doe import (
    BOUNDARY18_GROUP_NAMES,
    Boundary18Sample,
    generate_boundary18_doe_samples,
    generate_boundary18_refined_samples,
)
from offshore_energy_sim.optimization.connectors import (
    ConnectorDesignVariable,
    ConnectorObjectiveSpec,
    ConnectorOptimizationProblem,
    uniform_hinge_stiffness_variables,
)
from offshore_energy_sim.optimization.design_evaluator import (
    BoundaryStiffnessDesign,
    PitchStiffnessDesign,
    SingleFrequencyEvaluation,
    SingleFrequencyScenario,
    connector_envelope_rows,
    evaluate_complex_hinge_boundary_design,
    evaluate_complex_hinge_pitch_design,
    evaluate_design,
    evaluate_design_response,
    heave_amplitude_grid,
    heave_metrics,
)
from offshore_energy_sim.optimization.hinge_design_space import (
    HingeDesignGroup,
    HingeDesignSpaceSummary,
    apply_grouped_hinge_stiffness,
    build_hinge_design_groups,
    summarize_hinge_design_space,
)
from offshore_energy_sim.optimization.pareto import (
    MetricConstraint,
    MetricObjective,
    constraint_margins,
    constraints_satisfied,
    mark_pareto_rows,
    objective_matrix,
    pareto_mask_from_values,
)

__all__ = [
    "BOUNDARY18_GROUP_NAMES",
    "Boundary18Sample",
    "ConnectorDesignVariable",
    "ConnectorObjectiveSpec",
    "ConnectorOptimizationProblem",
    "BoundaryStiffnessDesign",
    "PitchStiffnessDesign",
    "SingleFrequencyEvaluation",
    "SingleFrequencyScenario",
    "connector_envelope_rows",
    "evaluate_complex_hinge_boundary_design",
    "evaluate_complex_hinge_pitch_design",
    "evaluate_design",
    "evaluate_design_response",
    "heave_amplitude_grid",
    "heave_metrics",
    "HingeDesignGroup",
    "HingeDesignSpaceSummary",
    "apply_grouped_hinge_stiffness",
    "build_hinge_design_groups",
    "summarize_hinge_design_space",
    "MetricConstraint",
    "MetricObjective",
    "constraint_margins",
    "constraints_satisfied",
    "mark_pareto_rows",
    "objective_matrix",
    "pareto_mask_from_values",
    "generate_boundary18_doe_samples",
    "generate_boundary18_refined_samples",
    "uniform_hinge_stiffness_variables",
]
