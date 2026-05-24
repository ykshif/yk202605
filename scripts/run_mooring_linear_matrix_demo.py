"""Run a small WEC-Sim-style linear mooring matrix demo.

The demo does not require external hydrodynamic or structural files.  It builds
one-node, four-corner, or YAML-defined linear mooring attachments, assembles the
natural retained global matrices, projects them through an identity reduction,
and writes reduced K/C/F0 arrays plus simple static/dynamic response metrics.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.core import load_case_config, write_metrics_json  # noqa: E402
from offshore_energy_sim.mooring import (  # noqa: E402
    LinearMooringMatrix,
    NodalMooringAttachment,
    assemble_nodal_mooring_terms,
    build_mooring_attachments_from_config,
    mooring_section,
    project_global_mooring_terms_to_reduced,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "mooring" / "linear_matrix_demo"
DEFAULT_RETAINED_DOFS = (0, 1, 2, 3, 4)
DOF_LABELS_5 = ("surge", "sway", "heave", "roll", "pitch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("one_node", "four_corner", "yaml"),
        default="one_node",
        help="Built-in demo scenario. Use 'yaml' together with --config.",
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML file with a mooring section.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--total-nodes", type=int, default=None)
    parser.add_argument("--grid-nodes-x", type=int, default=2)
    parser.add_argument("--grid-nodes-y", type=int, default=2)
    parser.add_argument("--retained-dofs", default="0,1,2,3,4")
    parser.add_argument("--surge-stiffness", type=float, default=1.0e7)
    parser.add_argument("--sway-stiffness", type=float, default=1.0e7)
    parser.add_argument("--heave-stiffness", type=float, default=0.0)
    parser.add_argument("--surge-damping", type=float, default=1.0e5)
    parser.add_argument("--sway-damping", type=float, default=1.0e5)
    parser.add_argument("--heave-damping", type=float, default=0.0)
    parser.add_argument("--pretension-surge", type=float, default=0.0)
    parser.add_argument("--pretension-sway", type=float, default=0.0)
    parser.add_argument("--pretension-heave", type=float, default=0.0)
    parser.add_argument("--structure-stiffness", type=float, default=2.0e7)
    parser.add_argument("--structure-damping", type=float, default=2.0e5)
    parser.add_argument("--mass", type=float, default=1.0e6)
    parser.add_argument("--omega", type=float, default=0.5)
    parser.add_argument("--force-dof", type=int, default=0)
    parser.add_argument("--force-amplitude", type=float, default=1.0e5)
    return parser.parse_args()


def parse_retained_dofs(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("--retained-dofs must contain at least one zero-based DOF index")
    return values


def built_in_matrix(args: argparse.Namespace) -> LinearMooringMatrix:
    stiffness = np.diag(
        [
            args.surge_stiffness,
            args.sway_stiffness,
            args.heave_stiffness,
            0.0,
            0.0,
            0.0,
        ]
    )
    damping = np.diag(
        [
            args.surge_damping,
            args.sway_damping,
            args.heave_damping,
            0.0,
            0.0,
            0.0,
        ]
    )
    pretension = np.array(
        [
            args.pretension_surge,
            args.pretension_sway,
            args.pretension_heave,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )
    return LinearMooringMatrix(
        stiffness=stiffness,
        damping=damping,
        pretension=pretension,
        metadata={"source": "run_mooring_linear_matrix_demo"},
    )


def corner_nodes(nx: int, ny: int) -> tuple[int, int, int, int]:
    if nx < 2 or ny < 2:
        raise ValueError("four_corner scenario requires --grid-nodes-x/y >= 2")
    return (1, nx, (ny - 1) * nx + 1, nx * ny)


def attachments_from_args(
    args: argparse.Namespace,
) -> tuple[tuple[NodalMooringAttachment, ...], int, tuple[int, ...], str]:
    retained = parse_retained_dofs(args.retained_dofs)
    if args.scenario == "yaml":
        if args.config is None:
            raise ValueError("--scenario yaml requires --config")
        config = load_case_config(args.config)
        section = mooring_section(config)
        attachments = build_mooring_attachments_from_config(config)
        if not attachments:
            raise ValueError("YAML mooring section must be enabled and contain attachments")
        retained = tuple(
            int(value)
            for value in section.get("retained_full_dofs_zero_based", retained)
        )
        max_node = max(item.node_one_based for item in attachments)
        total_nodes = int(section.get("demo_total_nodes", args.total_nodes or max_node))
        return attachments, total_nodes, retained, "yaml_config"

    matrix = built_in_matrix(args)
    if args.scenario == "one_node":
        total_nodes = int(args.total_nodes or 1)
        return (
            (
                NodalMooringAttachment(
                    node_one_based=1,
                    matrix=matrix,
                    name="single_demo_line",
                ),
            ),
            total_nodes,
            retained,
            "one_node",
        )

    total_nodes = args.grid_nodes_x * args.grid_nodes_y
    attachments = tuple(
        NodalMooringAttachment(
            node_one_based=node,
            matrix=matrix,
            name=f"corner_node_{node}",
        )
        for node in corner_nodes(args.grid_nodes_x, args.grid_nodes_y)
    )
    return attachments, total_nodes, retained, "four_corner"


def identity_reduction(ndof: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # T maps reduced/master coordinates to retained global coordinates.
    transformation = np.eye(ndof, dtype=float)
    master_dofs = np.arange(ndof, dtype=int)
    slave_dofs = np.array([], dtype=int)
    return transformation, master_dofs, slave_dofs


def response_metrics(
    stiffness: np.ndarray,
    damping: np.ndarray,
    pretension: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    ndof = stiffness.shape[0]
    if args.force_dof < 0 or args.force_dof >= ndof:
        raise ValueError("--force-dof is outside the reduced DOF range")

    # Diagonal surrogate structural terms keep the demo well conditioned.
    mass = args.mass * np.eye(ndof, dtype=float)
    structural_stiffness = args.structure_stiffness * np.eye(ndof, dtype=float)
    structural_damping = args.structure_damping * np.eye(ndof, dtype=float)
    effective_stiffness = structural_stiffness + stiffness
    effective_damping = structural_damping + damping

    harmonic_force = np.zeros(ndof, dtype=float)
    harmonic_force[args.force_dof] = args.force_amplitude
    dynamic_stiffness = (
        effective_stiffness
        - args.omega**2 * mass
        + 1j * args.omega * effective_damping
    )
    static_offset = np.linalg.solve(effective_stiffness, pretension)
    harmonic_amplitude = np.linalg.solve(dynamic_stiffness, harmonic_force)

    return {
        "static_offset_norm": float(np.linalg.norm(static_offset)),
        "harmonic_amplitude_norm": float(np.linalg.norm(harmonic_amplitude)),
        "force_dof": int(args.force_dof),
        "force_amplitude": float(args.force_amplitude),
        "omega_rad_s": float(args.omega),
        "static_offset": static_offset,
        "harmonic_amplitude_abs": np.abs(harmonic_amplitude),
        "harmonic_amplitude_phase_rad": np.angle(harmonic_amplitude),
    }


def plot_demo_outputs(output_root: Path, stiffness: np.ndarray, metrics: dict[str, object]) -> list[Path]:
    import matplotlib.pyplot as plt

    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    diagonal_path = figures_dir / "mooring_reduced_stiffness_diagonal.png"
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(np.arange(stiffness.shape[0]), np.diag(stiffness), color="#2b6cb0")
    ax.set_xlabel("reduced DOF index")
    ax.set_ylabel("stiffness")
    ax.set_title("Reduced mooring stiffness diagonal")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(diagonal_path, dpi=180)
    plt.close(fig)
    paths.append(diagonal_path)

    response_path = figures_dir / "demo_harmonic_amplitude.png"
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    amplitude = np.asarray(metrics["harmonic_amplitude_abs"], dtype=float)
    ax.bar(np.arange(amplitude.size), amplitude, color="#c05621")
    ax.set_xlabel("reduced DOF index")
    ax.set_ylabel("amplitude")
    ax.set_title("Demo harmonic response amplitude")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(response_path, dpi=180)
    plt.close(fig)
    paths.append(response_path)
    return paths


def main() -> int:
    args = parse_args()
    attachments, total_nodes, retained, scenario = attachments_from_args(args)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    global_terms = assemble_nodal_mooring_terms(
        attachments,
        total_nodes=total_nodes,
        retained_full_dofs_zero_based=retained,
    )
    transformation, master_dofs, slave_dofs = identity_reduction(global_terms.stiffness.shape[0])
    reduced = project_global_mooring_terms_to_reduced(
        global_terms,
        transformation,
        master_dofs,
        slave_dofs,
    )

    stiffness_path = output_root / "mooring_reduced_stiffness.npy"
    damping_path = output_root / "mooring_reduced_damping.npy"
    pretension_path = output_root / "mooring_reduced_pretension.npy"
    np.save(stiffness_path, reduced.stiffness)
    np.save(damping_path, reduced.damping)
    np.save(pretension_path, reduced.pretension)

    response = response_metrics(reduced.stiffness, reduced.damping, reduced.pretension, args)
    figures = plot_demo_outputs(output_root, reduced.stiffness, response)
    metrics = {
        "status": "completed",
        "scenario": scenario,
        "total_nodes": int(total_nodes),
        "retained_full_dofs_zero_based": retained,
        "attachment_count": len(attachments),
        "attachment_nodes_one_based": tuple(item.node_one_based for item in attachments),
        "attachment_names": tuple(item.name for item in attachments),
        "convention": "F_moor = F0 - K*q - C*qdot",
        "reduction": "identity demo reduction",
        "reduced_shape": reduced.stiffness.shape,
        "reduced_stiffness_path": stiffness_path,
        "reduced_damping_path": damping_path,
        "reduced_pretension_path": pretension_path,
        "reduced_stiffness_frobenius_norm": float(np.linalg.norm(reduced.stiffness)),
        "reduced_damping_frobenius_norm": float(np.linalg.norm(reduced.damping)),
        "reduced_pretension_norm": float(np.linalg.norm(reduced.pretension)),
        "response": response,
        "figures": figures,
    }
    metrics_path = write_metrics_json(output_root / "metrics.json", metrics)

    print("Linear mooring matrix demo completed.")
    print(f"scenario: {scenario}")
    print(f"attachments: {len(attachments)}")
    print(f"reduced_shape: {reduced.stiffness.shape}")
    print(f"metrics: {metrics_path}")
    for figure in figures:
        print(f"figure: {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
