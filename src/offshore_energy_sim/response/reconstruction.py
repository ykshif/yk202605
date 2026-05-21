"""Response reconstruction helpers for reduced-order solutions."""

from __future__ import annotations

import numpy as np

from offshore_energy_sim.reduction import reorder_displacement_to_natural_order


def reconstruct_global_response(
    transformation: np.ndarray,
    master_displacement: np.ndarray,
    master_dofs: np.ndarray,
    slave_dofs: np.ndarray,
    *,
    reverse_master_order: bool = True,
) -> np.ndarray:
    """Reconstruct the retained global response from master DOF response.

    Parameters
    ----------
    transformation:
        SEREP transformation matrix mapping master DOFs to the disordered
        retained global DOF vector.
    master_displacement:
        Frequency-domain displacement at retained master DOFs.
    master_dofs, slave_dofs:
        Retained global DOF indices that define the disordered master/slave
        block ordering.

    Numerical-result expectation: unchanged relative to the previous inline
    ``transformation @ master_displacement`` plus natural-order reordering.
    """

    global_displacement_disordered = transformation @ master_displacement
    return reorder_displacement_to_natural_order(
        global_displacement_disordered,
        master_dofs,
        slave_dofs,
        reverse_master_order=reverse_master_order,
    )
