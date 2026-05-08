#import packages
import time
import numpy as np
import capytaine as cpt
import scipy
from capytaine.io.mesh_writers import write_STL
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.linalg import eigh
import vtk
import logging
import xarray as xr
from capytaine.io.xarray import merge_complex_values
from capytaine.post_pro import rao
logging.basicConfig(level=logging.INFO, format='%(levelname)-8s: %(message)s')
# user defined functions
import DM_ShowNodes as DMshow
import DM_Reading as dm_r
import DM_Assemble as DM_A
import SEREP as SEREP

def perform_RODM_reduce_order_model(num_nodes, node_position_params, hydrodynamic_data_path, structure_data_paths, use_hydrostatic=True):
    """
    Perform Reduced Order Dynamic Modeling (RODM) based on provided parameters.
    
    Args:
        num_nodes (int): Number of nodes in the model.
        node_position_params (tuple): Parameters for node position calculation.
        hydrodynamic_data_path (str): Path to the hydrodynamic data file.
        structure_data_paths (dict): Dictionary containing paths to structural data files.

    Returns:
        ndarray: The reordered global displacement after frequency domain analysis.
    """

    # Calculate node positions based on specified parameters
    master_nodes = DM_A.calculate_node_positions(*node_position_params)

    # Load hydrodynamic data and extract complex values and angular frequency
    dataset = merge_complex_values(xr.open_dataset(hydrodynamic_data_path))
    omega = dataset.omega.values  # Angular frequency from dataset

    # Read mass and stiffness matrices from specified file paths
    M = dm_r.get_stiffness_matrix(structure_data_paths['mass'])
    k = dm_r.get_stiffness_matrix(structure_data_paths['stiffness'])
    
    # Reduce the degrees of freedom for the mass and stiffness matrices
    M = SEREP.reduce_dofs(M, num_nodes, [5])
    k = SEREP.reduce_dofs(k, num_nodes, [5])

    # Transform the mass matrix to a consistent mass matrix using beta=0
    M = SEREP.transform_mass_matrix(M, beta=0)

    # Obtain the master and slave degrees of freedom
    MasterDofs, SlaveDofs = SEREP.separate_dofs(num_nodes, master_nodes)

    # Apply SEREP to reduce the system matrices
    MR, KR, T = SEREP.SEREP(k, M, SlaveDofs, master_nodes)

    # Extract and reduce hydrodynamic matrices for added mass, radiation damping, and hydrostatic stiffness
    added_mass = SEREP.reduce_dofs(dataset['added_mass'][0].values, 10, [5])
    radiation_damping = SEREP.reduce_dofs(dataset['radiation_damping'][0].values, 10, [5])

    if use_hydrostatic:
        hydrostatic_stiffness = SEREP.reduce_dofs(dataset['hydrostatic_stiffness'].values, 10, [5])
        stiffness = hydrostatic_stiffness + KR
    else:
        total_nodes = 793
        nodes_per_row = 61
        area = 5 * 5
        k_fem = SEREP.get_fem_spring_stiffness(total_nodes, nodes_per_row, area)
        k_fem = SEREP.reduce_dofs(k_fem, num_nodes, [5])
        stiffness = T.T@k_fem@T + KR

    # Prepare the force vector by reducing dimensions and reshaping
    F_w = SEREP.reduce_force_matrix_dofs(dataset['Froude_Krylov_force'][0].values + dataset['diffraction_force'][0].values, 10, 5).reshape(1, 50)
    # # 在计算内力的时候进行了修改
    # F_w = F_w.reshape(10,5)[::-1].reshape(1,50)
    # Combine mass matrices and solve in the frequency domain
    mass = added_mass + MR
    damping = radiation_damping

    master_displacement = DM_A.solve_frequency_domain(mass, damping, stiffness, F_w, omega)

    # Restore and reorder global displacement based on transformation matrix and master displacement
    global_displacement_disorder = T @ master_displacement
    global_displacement = SEREP.reorder_displacement_matrix(global_displacement_disorder, MasterDofs, SlaveDofs)
    # # find the error, which is the difference between the master_displacement and the global_displacement of master positions,
    # # so we use master_displacement to replace the global_displacement of master positions.
    # global_displacement_replace = replace_master_with_global(master_displacement, global_displacement, master_nodes)

    return global_displacement


import xarray as xr

def perform_expansion_and_solve(num_nodes, node_position_params, hydrodynamic_data_path, structure_data_paths, use_hydrostatic=True):
    """
    Perform expansion process and solve for global displacement in the frequency domain.

    Args:
        num_nodes (int): Total number of nodes in the model.
        node_position_params (tuple): Parameters used to calculate the positions of master nodes.
        hydrodynamic_data_path (str): File path to the hydrodynamic data stored in NetCDF format.
        structure_data_paths (dict): Dictionary with file paths to the mass and stiffness matrices.

    Returns:
        ndarray: The reordered global displacement array after performing dynamic analysis.
    """

    # Calculate node positions for the master nodes
    master_nodes = DM_A.calculate_node_positions(*node_position_params)

    # Load and process the hydrodynamic dataset
    dataset = merge_complex_values(xr.open_dataset(hydrodynamic_data_path))
    omega = dataset.omega.values  # Extract angular frequency from dataset

    # Load mass and stiffness matrices
    M = dm_r.get_stiffness_matrix(structure_data_paths['mass'])
    k = dm_r.get_stiffness_matrix(structure_data_paths['stiffness'])
    
    # Reduce the degrees of freedom in the mass and stiffness matrices
    M = SEREP.reduce_dofs(M, num_nodes, [5])
    k = SEREP.reduce_dofs(k, num_nodes, [5])

    # Transform the mass matrix to a consistent mass matrix
    M = SEREP.transform_mass_matrix(M, beta=0)

    # Determine master and slave degrees of freedom
    MasterDofs, SlaveDofs = SEREP.separate_dofs(num_nodes, master_nodes)

    # Perform SEREP to obtain reduced system matrices and the transformation matrix
    MR, KR, T = SEREP.SEREP_Expansion(k, M, SlaveDofs, master_nodes)

    # Read and reduce hydrodynamic data for system matrices
    added_mass = SEREP.reduce_dofs(dataset['added_mass'][0].values, 10, [5])
    radiation_damping = SEREP.reduce_dofs(dataset['radiation_damping'][0].values, 10, [5])
    hydrostatic_stiffness = SEREP.reduce_dofs(dataset['hydrostatic_stiffness'].values, 10, [5])

    if use_hydrostatic:
        hydrostatic_stiffness = SEREP.reduce_dofs(dataset['hydrostatic_stiffness'].values, 10, [5])
        stiffness = T.T @ hydrostatic_stiffness @ T + KR
    else:
        total_nodes = 793
        nodes_per_row = 61
        area = 5 * 5
        k_fem = SEREP.get_fem_spring_stiffness(total_nodes, nodes_per_row, area)
        k_fem = SEREP.reduce_dofs(k_fem, num_nodes, [5])
        stiffness = k_fem + KR


    F_w_hydro = dataset['Froude_Krylov_force'][0].values + dataset['diffraction_force'][0].values
    F_w_hydro_redu = SEREP.reduce_force_matrix_dofs(F_w_hydro, 10, 5).reshape(1, 50)

    # Combine reduced hydrodynamic matrices to form system matrices
    mass = T.T @ added_mass @ T + MR
    damping = T.T @ radiation_damping @ T
    F_w = T.T @ F_w_hydro_redu.T

    # Solve the system in the frequency domain
    master_displacement = DM_A.solve_frequency_domain(mass, damping, stiffness, F_w.T, omega)

    # Reorder the global displacement according to the master and slave DOFs
    global_displacement_disorder = master_displacement
    master_displacement = master_displacement[0:50,0]
    global_displacement = SEREP.reorder_displacement_matrix(global_displacement_disorder, MasterDofs, SlaveDofs)

    # 修复global_displacement中的master节点的值
    global_displacement_replace = replace_master_with_global(master_displacement, global_displacement, master_nodes)

    return global_displacement_replace


# 原方法
def calculate_initial_displacement(num_nodes, node_position_params, hydrodynamic_data_path, structure_data_paths):
    """
    Calculate the displacement for a given system of order method.

    Parameters:
    N (int): The number of nodes in the system.
    file_path (str): Path to the stiffness matrix file.
    dataset (DataFrame): A pandas DataFrame containing the necessary data.
    nodes (array): Array of node positions.

    Returns:
    ndarray: The displacement array.
    """
    # Calculate node positions for the master nodes
    nodes = DM_A.calculate_node_positions(*node_position_params)
    # Load and process the hydrodynamic dataset
    dataset = merge_complex_values(xr.open_dataset(hydrodynamic_data_path))
    omega = dataset.omega.values  # Extract angular frequency from dataset

    # Extract data from the dataset
    added_mass = dataset['added_mass'][0].values
    radiation_damping = dataset['radiation_damping'][0].values
    inertia_matrix = dataset['inertia_matrix'].values
    hydrostatic_stiffness = dataset['hydrostatic_stiffness'].values
    F_w = dataset['Froude_Krylov_force'][0].values + dataset['diffraction_force'][0].values

    # Construct the combined matrices
    M = added_mass + inertia_matrix  # Total mass
    C = radiation_damping  # Damping
    K = hydrostatic_stiffness  # Stiffness

    # Insert matrices into the system
    mass = DM_A.insert_matrix(num_nodes, M, nodes)
    damping = DM_A.insert_matrix(num_nodes, C, nodes)
    hy_stiffness = DM_A.insert_matrix(num_nodes, K, nodes)

    # Assemble the stiffness matrix
    stiffness = dm_r.get_stiffness_matrix(structure_data_paths['stiffness']) + hy_stiffness

    # Assemble the force matrix
    K_F_w = DM_A.extend_force_matrix(F_w, nodes, num_nodes)
    # Solve in the frequency domain
    X = DM_A.solve_frequency_domain(mass, damping, stiffness, K_F_w, omega)

    return X


def replace_master_with_global(master_displacement, global_displacement, control_point_indices, num_dofs=5):
    """
    Replace the displacement results of master control points with the corresponding positions in the global displacement results.

    Parameters:
    master_displacement (np.ndarray): Displacement results of master control points, shape (50, 1).
    global_displacement (np.ndarray): Restored global displacement results, shape (793*5, 1).
    control_point_indices (list): Indices of master control points in the global displacement, list of length 10.
    num_dofs (int): Number of degrees of freedom per node, default is 5.

    Returns:
    np.ndarray: Updated global displacement results with replaced values.
    """
    # Reverse the order of control point indices
    control_point_indices = control_point_indices[::-1]
    
    # Expand control point indices to include all degrees of freedom indices
    expanded_control_point_indices = []
    for idx in control_point_indices:
        for dof in range(num_dofs):
            expanded_control_point_indices.append((idx - 1) * num_dofs + dof)
    
    # Replace the corresponding values in the global displacement with master displacement values
    for i, idx in enumerate(expanded_control_point_indices):
        global_displacement[idx] = master_displacement[i]
    
    return global_displacement
