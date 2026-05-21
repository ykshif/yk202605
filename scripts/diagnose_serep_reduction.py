"""Diagnose SEREP reduction conditioning for uniform-module RODM cases.

This script is intentionally separate from the production solver. It reuses the
already generated U5/U10/U15/U30 hydrodynamic data and compares the legacy
square SEREP inverse with SVD-based variants so that reduction-matrix failures
can be isolated from the hydrodynamic mesh/data generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from offshore_energy_sim.hydrodynamics import (  # noqa: E402
    open_hydrodynamic_dataset,
    prepare_hydrodynamic_terms,
)
from offshore_energy_sim.reduction import (  # noqa: E402
    reduce_matrix_dofs,
    separate_master_slave_dofs,
    transform_mass_matrix,
)
from offshore_energy_sim.reduction.modal import _reorder_master_slave_blocks  # noqa: E402
from offshore_energy_sim.response import reconstruct_global_response  # noqa: E402
from offshore_energy_sim.solver.frequency_domain import solve_frequency_domain  # noqa: E402
from offshore_energy_sim.structure.matrix_io import read_abaqus_matrix_dense  # noqa: E402

from run_uniform_reference_convergence import (  # noqa: E402
    FULL_DOFS_PER_NODE,
    HYDRO_NODE_REVERSE_BY_WAVELENGTH,
    HYDRO_DOF_TO_REMOVE_ZERO_BASED,
    MODULE_COUNTS,
    OUTPUT_ROOT,
    REMOVED_FULL_DOFS_ZERO_BASED,
    RESPONSE_DIR,
    RETAINED_DOFS_PER_NODE,
    STRUCTURAL_DX_M,
    STRUCTURAL_DY_M,
    STRUCTURAL_NODES_PER_X,
    STRUCTURE_DIR,
    TOTAL_NODES,
    WAVELENGTHS_M,
    build_hydro_config,
    extract_centerline_curve,
    module_geometry_rows,
    rodm_case,
    structural_paths,
)


DIAG_DIR = OUTPUT_ROOT / "serep_reduction_diagnostics"
FIGURE_DIR = DIAG_DIR / "figures"
DOF_LABELS = ("surge_x", "sway_y", "heave_z", "roll_rx", "pitch_ry")
VARIANTS = (
    ("legacy_inv", "inv", 1, None),
    ("square_pinv_r1e-14", "pinv", 1, 1e-14),
    ("square_pinv_r1e-12", "pinv", 1, 1e-12),
    ("rect_2p_pinv_r1e-12", "pinv", 2, 1e-12),
    ("rect_3p_pinv_r1e-12", "pinv", 3, 1e-12),
    ("guyan_static", "guyan", 0, None),
)


@dataclass(frozen=True)
class ModalData:
    module_count: int
    master_nodes: tuple[int, ...]
    master_dofs: np.ndarray
    slave_dofs: np.ndarray
    reordered_stiffness: np.ndarray
    reordered_mass: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    @property
    def master_size(self) -> int:
        return RETAINED_DOFS_PER_NODE * len(self.master_nodes)


def structural_node_xy(node_id: int) -> tuple[float, float]:
    node_index = node_id - 1
    grid_x_index = node_index % STRUCTURAL_NODES_PER_X
    grid_y_index = node_index // STRUCTURAL_NODES_PER_X
    x_index = STRUCTURAL_NODES_PER_X - 1 - grid_x_index
    return x_index * STRUCTURAL_DX_M, grid_y_index * STRUCTURAL_DY_M


def dof_metadata(global_dof: int) -> dict[str, object]:
    node_id = global_dof // RETAINED_DOFS_PER_NODE + 1
    local_dof = global_dof % RETAINED_DOFS_PER_NODE
    x_m, y_m = structural_node_xy(node_id)
    return {
        "node_id": node_id,
        "node_x_m": x_m,
        "node_y_m": y_m,
        "local_dof": local_dof,
        "dof_label": DOF_LABELS[local_dof],
    }


def load_retained_structural_matrices() -> tuple[np.ndarray, np.ndarray]:
    paths = structural_paths()
    mass_full = read_abaqus_matrix_dense(paths.mass, dofs_per_node=FULL_DOFS_PER_NODE)
    stiffness_full = read_abaqus_matrix_dense(
        paths.stiffness,
        dofs_per_node=FULL_DOFS_PER_NODE,
    )
    mass_retained = reduce_matrix_dofs(
        mass_full,
        TOTAL_NODES,
        REMOVED_FULL_DOFS_ZERO_BASED,
    )
    stiffness_retained = reduce_matrix_dofs(
        stiffness_full,
        TOTAL_NODES,
        REMOVED_FULL_DOFS_ZERO_BASED,
    )
    return transform_mass_matrix(mass_retained, beta=0.0), stiffness_retained


def build_modal_data(
    module_count: int,
    mass_retained: np.ndarray,
    stiffness_retained: np.ndarray,
) -> ModalData:
    from scipy.linalg import eigh

    rows = module_geometry_rows(module_count)
    master_nodes = tuple(row.selected_node_id for row in rows)
    master_dofs, slave_dofs = separate_master_slave_dofs(
        TOTAL_NODES,
        master_nodes,
        dofs_per_node=RETAINED_DOFS_PER_NODE,
    )
    reordered_stiffness, reordered_mass = _reorder_master_slave_blocks(
        stiffness_retained,
        mass_retained,
        slave_dofs,
    )
    eigenvalues, eigenvectors = eigh(reordered_stiffness, reordered_mass)
    for mode_index in range(eigenvectors.shape[1]):
        max_abs = np.max(np.abs(eigenvectors[:, mode_index]))
        if max_abs > 0.0:
            eigenvectors[:, mode_index] /= max_abs
    return ModalData(
        module_count=module_count,
        master_nodes=master_nodes,
        master_dofs=master_dofs,
        slave_dofs=slave_dofs,
        reordered_stiffness=reordered_stiffness,
        reordered_mass=reordered_mass,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
    )


def modal_block_svd(modal: ModalData, mode_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = modal.master_size
    master_modes = modal.eigenvectors[:p, :mode_count]
    return np.linalg.svd(master_modes, full_matrices=False)


def build_transform(
    modal: ModalData,
    *,
    inverse_kind: str,
    mode_multiplier: int,
    rcond: float | None,
) -> tuple[np.ndarray, dict[str, float]]:
    p = modal.master_size
    if inverse_kind == "guyan":
        stiffness_slave_slave = modal.reordered_stiffness[p:, p:]
        stiffness_slave_master = modal.reordered_stiffness[p:, :p]
        slave_transform = -np.linalg.solve(stiffness_slave_slave, stiffness_slave_master)
        transformation = np.vstack([np.eye(p), slave_transform])
        identity_error = np.linalg.norm(transformation[:p, :] - np.eye(p), ord="fro") / np.sqrt(p)
        return transformation, {
            "mode_count": 0.0,
            "modal_block_smax": "",
            "modal_block_smin": "",
            "modal_block_condition": "",
            "identity_error_fro_per_dof": float(identity_error),
            "transformation_fro_norm": float(np.linalg.norm(transformation, ord="fro")),
            "transformation_max_abs": float(np.max(np.abs(transformation))),
        }

    q = min(mode_multiplier * p, modal.eigenvectors.shape[1])
    modes = modal.eigenvectors[:, :q]
    master_modes = modes[:p, :]
    singular_values = np.linalg.svd(master_modes, compute_uv=False)

    if inverse_kind == "inv":
        if q != p:
            raise ValueError("Direct inverse is only valid for square SEREP blocks.")
        mapping = np.linalg.inv(master_modes)
    elif inverse_kind == "pinv":
        mapping = np.linalg.pinv(master_modes, rcond=1e-15 if rcond is None else rcond)
    else:
        raise ValueError(f"Unknown inverse kind: {inverse_kind}")

    transformation = modes @ mapping
    identity_error = np.linalg.norm(transformation[:p, :] - np.eye(p), ord="fro") / np.sqrt(p)
    stats = {
        "mode_count": float(q),
        "modal_block_smax": float(singular_values[0]),
        "modal_block_smin": float(singular_values[-1]),
        "modal_block_condition": float(singular_values[0] / singular_values[-1]),
        "identity_error_fro_per_dof": float(identity_error),
        "transformation_fro_norm": float(np.linalg.norm(transformation, ord="fro")),
        "transformation_max_abs": float(np.max(np.abs(transformation))),
    }
    return transformation, stats


def reduced_matrices(
    modal: ModalData,
    transformation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    reduced_mass = transformation.T @ modal.reordered_mass @ transformation
    reduced_stiffness = transformation.T @ modal.reordered_stiffness @ transformation
    stats = {
        "reduced_mass_fro_norm": float(np.linalg.norm(reduced_mass, ord="fro")),
        "reduced_stiffness_fro_norm": float(np.linalg.norm(reduced_stiffness, ord="fro")),
        "reduced_mass_condition": float(np.linalg.cond(reduced_mass)),
        "reduced_stiffness_condition": float(np.linalg.cond(reduced_stiffness)),
    }
    return reduced_mass, reduced_stiffness, stats


def solve_response_direct(
    modal: ModalData,
    transformation: np.ndarray,
    reduced_mass: np.ndarray,
    reduced_stiffness: np.ndarray,
    *,
    wavelength_index: int,
) -> np.ndarray:
    hydro_config = build_hydro_config(modal.module_count, n_jobs=1)
    case = rodm_case(
        module_count=modal.module_count,
        wavelength_index=wavelength_index,
        hydro_path=hydro_config.output_path,
        master_nodes_one_based=modal.master_nodes,
    )
    dataset = open_hydrodynamic_dataset(case.hydrodynamic_dataset, merge_complex=True)
    try:
        hydrodynamic = prepare_hydrodynamic_terms(case, dataset)
        master_displacement = solve_frequency_domain(
            hydrodynamic.added_mass + reduced_mass,
            hydrodynamic.radiation_damping,
            hydrodynamic.hydrostatic_stiffness + reduced_stiffness,
            hydrodynamic.wave_force,
            hydrodynamic.omega,
        )
    finally:
        dataset.close()
    return reconstruct_global_response(
        transformation,
        master_displacement,
        modal.master_dofs,
        modal.slave_dofs,
    )


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def max_abs_delta(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def roughness(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    return float(np.sum(np.abs(np.diff(values, n=2))))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_singular_values(singular_values_by_case: dict[int, np.ndarray], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5.5))
    for module_count, singular_values in singular_values_by_case.items():
        plt.semilogy(
            np.arange(1, singular_values.size + 1),
            singular_values,
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=f"U{module_count}",
        )
    plt.title("SEREP master modal block singular values")
    plt.xlabel("singular value index")
    plt.ylabel("singular value")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_variant_panel(curves: dict[tuple[int, str, int], np.ndarray], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(WAVELENGTHS_M), 1, figsize=(9, 13), sharex=True)
    fig.suptitle("Heave response: legacy U15 reference vs U30 SEREP variants", fontsize=14)
    for axis, wavelength_m in zip(axes, WAVELENGTHS_M):
        base_curve = curves[(15, "legacy_inv", wavelength_m)]
        x_over_l = np.linspace(0.0, 1.0, base_curve.size)
        axis.plot(
            x_over_l,
            base_curve,
            label="U15 legacy",
            linewidth=2.0,
        )
        for variant_name in (
            "legacy_inv",
            "square_pinv_r1e-12",
            "rect_2p_pinv_r1e-12",
            "guyan_static",
        ):
            variant_curve = curves[(30, variant_name, wavelength_m)]
            axis.plot(
                np.linspace(0.0, 1.0, variant_curve.size),
                variant_curve,
                label=f"U30 {variant_name}",
                linewidth=1.3,
                alpha=0.9,
            )
        axis.set_ylabel(f"{wavelength_m} m")
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("x/L")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    if not (STRUCTURE_DIR / "JobMesh5_5_MASS1.mtx").exists():
        raise FileNotFoundError(f"Missing structural directory: {STRUCTURE_DIR}")

    mass_retained, stiffness_retained = load_retained_structural_matrices()
    modal_by_count = {
        module_count: build_modal_data(module_count, mass_retained, stiffness_retained)
        for module_count in MODULE_COUNTS
    }

    block_rows: list[dict[str, object]] = []
    vector_rows: list[dict[str, object]] = []
    singular_values_by_case: dict[int, np.ndarray] = {}
    transform_cache: dict[tuple[int, str], np.ndarray] = {}
    reduced_cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}

    for module_count, modal in modal_by_count.items():
        p = modal.master_size
        row_order_dofs = np.setdiff1d(np.arange(modal.reordered_mass.shape[0]), modal.slave_dofs)
        u, singular_values, vh = modal_block_svd(modal, p)
        singular_values_by_case[module_count] = singular_values
        condition = singular_values[0] / singular_values[-1]
        block_rows.append(
            {
                "case": f"U{module_count}",
                "master_nodes": len(modal.master_nodes),
                "master_dofs": p,
                "mode_count": p,
                "smax": singular_values[0],
                "smin": singular_values[-1],
                "condition": condition,
                "count_s_rel_lt_1e-8": int(np.sum(singular_values < singular_values[0] * 1e-8)),
                "count_s_rel_lt_1e-10": int(np.sum(singular_values < singular_values[0] * 1e-10)),
                "count_s_rel_lt_1e-12": int(np.sum(singular_values < singular_values[0] * 1e-12)),
                "count_s_rel_lt_1e-14": int(np.sum(singular_values < singular_values[0] * 1e-14)),
                "inv_amplification_1_over_smin": 1.0 / singular_values[-1],
                "first_eigenvalue": modal.eigenvalues[0],
                "last_retained_eigenvalue": modal.eigenvalues[p - 1],
            }
        )

        left_vector = u[:, -1]
        right_vector = vh[-1, :]
        for rank, row_index in enumerate(np.argsort(np.abs(left_vector))[-20:][::-1], start=1):
            metadata = dof_metadata(int(row_order_dofs[row_index]))
            vector_rows.append(
                {
                    "case": f"U{module_count}",
                    "vector_type": "left_smallest_singular_master_dof",
                    "rank": rank,
                    "index": int(row_index),
                    "value": float(left_vector[row_index]),
                    "abs_value": float(abs(left_vector[row_index])),
                    **metadata,
                }
            )
        for rank, mode_index in enumerate(np.argsort(np.abs(right_vector))[-20:][::-1], start=1):
            vector_rows.append(
                {
                    "case": f"U{module_count}",
                    "vector_type": "right_smallest_singular_mode_combo",
                    "rank": rank,
                    "index": int(mode_index + 1),
                    "value": float(right_vector[mode_index]),
                    "abs_value": float(abs(right_vector[mode_index])),
                    "node_id": "",
                    "node_x_m": "",
                    "node_y_m": "",
                    "local_dof": "",
                    "dof_label": "",
                }
            )

        for variant_name, inverse_kind, mode_multiplier, rcond in VARIANTS:
            try:
                transformation, transform_stats = build_transform(
                    modal,
                    inverse_kind=inverse_kind,
                    mode_multiplier=mode_multiplier,
                    rcond=rcond,
                )
                reduced_mass, reduced_stiffness, matrix_stats = reduced_matrices(
                    modal,
                    transformation,
                )
            except np.linalg.LinAlgError as exc:
                block_rows.append(
                    {
                        "case": f"U{module_count}",
                        "master_nodes": len(modal.master_nodes),
                        "master_dofs": p,
                        "mode_count": p * mode_multiplier,
                        "variant": variant_name,
                        "failure": str(exc),
                    }
                )
                continue
            transform_cache[(module_count, variant_name)] = transformation
            reduced_cache[(module_count, variant_name)] = (reduced_mass, reduced_stiffness)
            block_rows.append(
                {
                    "case": f"U{module_count}",
                    "variant": variant_name,
                    "master_nodes": len(modal.master_nodes),
                    "master_dofs": p,
                    "inverse_kind": inverse_kind,
                    "rcond": "" if rcond is None else rcond,
                    **transform_stats,
                    **matrix_stats,
                }
            )

    write_csv(DIAG_DIR / "serep_modal_block_diagnostics.csv", block_rows)
    write_csv(DIAG_DIR / "serep_smallest_singular_vectors.csv", vector_rows)
    plot_singular_values(
        singular_values_by_case,
        FIGURE_DIR / "serep_master_modal_block_singular_values.png",
    )

    curves: dict[tuple[int, str, int], np.ndarray] = {}
    response_rows: list[dict[str, object]] = []
    for module_count in MODULE_COUNTS:
        modal = modal_by_count[module_count]
        for variant_name, *_ in VARIANTS:
            key = (module_count, variant_name)
            if key not in reduced_cache:
                continue
            reduced_mass, reduced_stiffness = reduced_cache[key]
            transformation = transform_cache[key]
            for wavelength_index, wavelength_m in enumerate(WAVELENGTHS_M):
                response = solve_response_direct(
                    modal,
                    transformation,
                    reduced_mass,
                    reduced_stiffness,
                    wavelength_index=wavelength_index,
                )
                curve = extract_centerline_curve(response, dof_index_zero_based=2).values
                curves[(module_count, variant_name, wavelength_m)] = curve
                response_rows.append(
                    {
                        "case": f"U{module_count}",
                        "variant": variant_name,
                        "wavelength_m": wavelength_m,
                        "heave_max": float(np.max(curve)),
                        "heave_mean": float(np.mean(curve)),
                        "heave_min": float(np.min(curve)),
                        "heave_roughness": roughness(curve),
                    }
                )

    comparison_rows: list[dict[str, object]] = []
    for variant_name, *_ in VARIANTS:
        for wavelength_m in WAVELENGTHS_M:
            if (15, "legacy_inv", wavelength_m) in curves and (30, variant_name, wavelength_m) in curves:
                comparison_rows.append(
                    {
                        "reference_case": "U15",
                        "reference_variant": "legacy_inv",
                        "comparison_case": "U30",
                        "comparison_variant": variant_name,
                        "wavelength_m": wavelength_m,
                        "rmse": rmse(
                            curves[(15, "legacy_inv", wavelength_m)],
                            curves[(30, variant_name, wavelength_m)],
                        ),
                        "max_abs": max_abs_delta(
                            curves[(15, "legacy_inv", wavelength_m)],
                            curves[(30, variant_name, wavelength_m)],
                        ),
                    }
                )
            if (10, "legacy_inv", wavelength_m) in curves and (15, variant_name, wavelength_m) in curves:
                comparison_rows.append(
                    {
                        "reference_case": "U10",
                        "reference_variant": "legacy_inv",
                        "comparison_case": "U15",
                        "comparison_variant": variant_name,
                        "wavelength_m": wavelength_m,
                        "rmse": rmse(
                            curves[(10, "legacy_inv", wavelength_m)],
                            curves[(15, variant_name, wavelength_m)],
                        ),
                        "max_abs": max_abs_delta(
                            curves[(10, "legacy_inv", wavelength_m)],
                            curves[(15, variant_name, wavelength_m)],
                        ),
                    }
                )

    write_csv(DIAG_DIR / "serep_variant_response_stats.csv", response_rows)
    write_csv(DIAG_DIR / "serep_variant_heave_comparisons.csv", comparison_rows)
    plot_variant_panel(curves, FIGURE_DIR / "u30_serep_variant_heave_panel.png")

    manifest = {
        "diagnostics_dir": str(DIAG_DIR),
        "modal_block_csv": str(DIAG_DIR / "serep_modal_block_diagnostics.csv"),
        "smallest_vectors_csv": str(DIAG_DIR / "serep_smallest_singular_vectors.csv"),
        "response_stats_csv": str(DIAG_DIR / "serep_variant_response_stats.csv"),
        "comparison_csv": str(DIAG_DIR / "serep_variant_heave_comparisons.csv"),
        "singular_value_figure": str(
            FIGURE_DIR / "serep_master_modal_block_singular_values.png"
        ),
        "variant_heave_figure": str(FIGURE_DIR / "u30_serep_variant_heave_panel.png"),
    }
    (DIAG_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
