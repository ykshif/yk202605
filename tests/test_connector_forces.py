from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from offshore_energy_sim.strength import compute_hinge_connector_forces  # noqa: E402
from offshore_energy_sim.structure import ExplicitHingeSpec  # noqa: E402


def test_ideal_hinge_released_rotations_carry_no_bending_moment() -> None:
    hinge = ExplicitHingeSpec(
        nodes_side_a_one_based=(1,),
        nodes_side_b_one_based=(2,),
        k_hinge=10.0,
        dofs_per_node=6,
        released_dofs_zero_based=(3, 4, 5),
        released_dof_stiffness=0.0,
        name="ideal hinge",
    )
    response = np.array(
        [
            1.0,
            2.0,
            3.0,
            0.10,
            0.20,
            0.30,
            0.5,
            1.0,
            1.0,
            0.05,
            0.15,
            0.25,
        ],
        dtype=np.complex128,
    ).reshape(-1, 1)

    result = compute_hinge_connector_forces(
        response,
        (hinge,),
        total_nodes=2,
        response_dofs_per_node=6,
        bending_moment_full_dofs_zero_based=(3, 4, 5),
    )[0]

    np.testing.assert_allclose(result.generalized_force, [5.0, 10.0, 20.0, 0.0, 0.0, 0.0])
    assert result.shear_force_abs == 20.0
    assert result.bending_moment_abs == 0.0
    assert result.released_moment_abs == 0.0


def test_elastic_hinge_released_pitch_stiffness_creates_moment() -> None:
    hinge = ExplicitHingeSpec(
        nodes_side_a_one_based=(1,),
        nodes_side_b_one_based=(2,),
        k_hinge=10.0,
        dofs_per_node=6,
        released_dofs_zero_based=(4,),
        released_dof_stiffness=100.0,
        name="elastic pitch hinge",
    )
    response = np.array(
        [
            0.0,
            0.0,
            1.0,
            0.20,
            0.30,
            0.0,
            0.0,
            0.7,
            0.10,
            0.10,
        ],
        dtype=np.complex128,
    ).reshape(-1, 1)

    result = compute_hinge_connector_forces(
        response,
        (hinge,),
        total_nodes=2,
        response_dofs_per_node=5,
        removed_full_dofs_zero_based=(5,),
        bending_moment_full_dofs_zero_based=(3, 4, 5),
    )[0]

    np.testing.assert_allclose(result.generalized_force, [0.0, 0.0, 3.0, 1.0, 20.0])
    np.testing.assert_allclose(result.shear_force_abs, 3.0)
    np.testing.assert_allclose(result.bending_moment_abs, np.sqrt(1.0**2 + 20.0**2))
    assert result.released_moment_abs == 20.0
