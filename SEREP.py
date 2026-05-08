import numpy as np

def reduce_dofs(matrix, num_nodes, dof_to_remove):
    """
    Reduce the degrees of freedom (DOFs) of a matrix by removing specified DOFs for each node.

    :param matrix: The original square matrix representing the system (2D array).
    :param num_nodes: The number of nodes in the system.
    :param dof_to_remove: A list of DOFs to remove for each node.
    :return: The reduced matrix.
    """
    # Total number of DOFs per node (assuming square matrix)
    total_dofs = matrix.shape[0]
    dofs_per_node = total_dofs // num_nodes

    # Generate the indices of DOFs to keep
    keep_dofs = [i for node in range(num_nodes) 
                 for i in range(node*dofs_per_node, (node+1)*dofs_per_node) 
                 if (i - node*dofs_per_node) not in dof_to_remove]

    # Reduce the matrix
    reduced_matrix = matrix[np.ix_(keep_dofs, keep_dofs)]
    return reduced_matrix


def transform_mass_matrix(consistant_mass_matrix, beta):
    """
    Transforms a consistent mass matrix into a modified mass matrix.

    A consistent mass matrix is typically used in finite element analysis and represents 
    mass distributed across the elements. A lumped mass matrix simplifies this by 
    concentrating the mass at the diagonal elements (nodes) of the matrix.

    This function converts a consistent mass matrix to a lumped mass matrix and then 
    blends it with the original consistent matrix based on the value of beta.

    Parameters:
    consistant_mass_matrix (np.array): The consistent mass matrix.
    beta (float): A coefficient used to adjust the mass matrix. It determines the blend 
                  between the consistent and lumped mass matrices.
                  - beta = 0: Results in the original consistent mass matrix.
                  - beta = 1: Results in a purely lumped mass matrix.
                  - 0 < beta < 1: Results in a weighted blend of both matrices.

    Returns:
    np.array: The modified mass matrix.
    """
    # Create a lumped mass matrix by summing each row's elements and placing the sum on the diagonal
    lumped_mass_matrix = np.diag(consistant_mass_matrix.sum(axis=1))

    # Compute the final mass matrix as a blend of lumped and consistent matrices
    M = beta * lumped_mass_matrix + (1 - beta) * consistant_mass_matrix

    return M


def separate_dofs(num_nodes, master_nodes, num_dofs_per_node=5):
    """
    Separates the DOFs into master and slave DOFs for a system with a specified number of nodes.

    Each node is assumed to have a specified number of DOFs (default is 5).
    Master DOFs correspond to the 'master_nodes', while the remaining DOFs are considered as slave DOFs.

    Parameters:
    num_nodes (int): Total number of nodes in the system.
    master_nodes (list): List of node IDs that are considered as master nodes.
    num_dofs_per_node (int, optional): Number of degrees of freedom per node. Default is 5.

    Returns:
    tuple: A tuple containing two numpy.ndarrays, the first one is the master DOFs and the second one is the slave DOFs.
    """
    # Total DOFs in the system
    total_dofs = num_nodes * num_dofs_per_node

    # Initialize an array to represent all DOFs
    dof_array = np.arange(total_dofs)

    # Adjust master_nodes to zero-based indexing
    master_nodes_zero_based = [node - 1 for node in master_nodes]

    # Calculate the DOF indices for the master nodes
    master_dofs = []
    for node_id in master_nodes_zero_based:
        start_dof = node_id * num_dofs_per_node
        master_dofs.extend(range(start_dof, start_dof + num_dofs_per_node))

    # Remove the master DOFs to find the slave DOFs
    slave_dofs = np.delete(dof_array, master_dofs)

    return np.array(master_dofs), slave_dofs

# Example usage
# num_nodes = 63
# master_nodes = [41, 39, 37, 35, 33, 31, 29, 27, 25, 23]
# MasterDofs, SlaveDofs = separate_dofs(num_nodes, master_nodes)
# print("Master DOFs:", MasterDofs)
# print("Slave DOFs:", SlaveDofs)
from scipy.linalg import eigh,eig

def SEREP(K, M, SlaveDofs, master_nodes):
    # Sort the SlaveDofs
    SlaveDofs = np.sort(SlaveDofs)
    #矩阵重组
    # Create index array for remaining DOFs
    index = np.setdiff1d(np.arange(K.shape[0]), SlaveDofs)
    M1 = M[index[:, np.newaxis], index]
    M2 = M[index[:, np.newaxis], SlaveDofs]
    M3 = M[SlaveDofs[:, np.newaxis], index]
    M4 = M[SlaveDofs[:, np.newaxis], SlaveDofs]
    k1 = K[index[:, np.newaxis], index]
    k2 = K[index[:, np.newaxis], SlaveDofs]
    k3 = K[SlaveDofs[:, np.newaxis], index]
    k4 = K[SlaveDofs[:, np.newaxis], SlaveDofs]
    M = np.vstack([np.hstack([M1, M2]), np.hstack([M3, M4])])
    K = np.vstack([np.hstack([k1, k2]), np.hstack([k3, k4])])
    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = eigh(K, M)
    # normalization vectors units or moralization mass.
    # # 对每个模态向量进行归一化
    for i in range(eigenvectors.shape[1]):  # 遍历所有列
        max_val = np.max(np.abs(eigenvectors[:, i]))  # 找到最大的绝对值
        eigenvectors[:, i] /= max_val  # 归一化S
    
    master_nodes_length = len(master_nodes)
    # #对每个模态向量进行归一化
    # for i in range(M.shape[1]):
    #     norm_val = np.sqrt(np.dot(eigenvectors[:, i].T, np.dot(M, eigenvectors[:, i])))
    #     eigenvectors[:, i] /= norm_val
    # Define transformation matrices
    T = eigenvectors[:,0:5*master_nodes_length] @ np.linalg.inv(eigenvectors[0:5*master_nodes_length,0:5*master_nodes_length])
    #T = eigenvectors[0:5*master_nodes_length,0:3*master_nodes_length]@np.linalg.pinv(eigenvectors[:,0:3*master_nodes_length])
    # T = T.T
    MR = T.T @ M @ T
    KR = T.T @ K @ T
    
    return MR,KR,T

def SEREP_Expansion(K, M, SlaveDofs, master_nodes):
    '''
    This function is used to implement the extension process.
    :param K: The stiffness matrix of the original system.
    :param M: The mass matrix of the original system.
    :param SlaveDofs: The slave DOFs of the original system.
    :return: The reorder mass, stiffness matrices and the transformation matrix.
    '''
    # Sort the SlaveDofs
    SlaveDofs = np.sort(SlaveDofs)
    #矩阵重组
    # Create index array for remaining DOFs
    index = np.setdiff1d(np.arange(K.shape[0]), SlaveDofs)
    M1 = M[index[:, np.newaxis], index]
    M2 = M[index[:, np.newaxis], SlaveDofs]
    M3 = M[SlaveDofs[:, np.newaxis], index]
    M4 = M[SlaveDofs[:, np.newaxis], SlaveDofs]
    k1 = K[index[:, np.newaxis], index]
    k2 = K[index[:, np.newaxis], SlaveDofs]
    k3 = K[SlaveDofs[:, np.newaxis], index]
    k4 = K[SlaveDofs[:, np.newaxis], SlaveDofs]
    M = np.vstack([np.hstack([M1, M2]), np.hstack([M3, M4])])
    K = np.vstack([np.hstack([k1, k2]), np.hstack([k3, k4])])
    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = eigh(K, M)

    # # 对每个模态向量进行归一化
    # for i in range(eigenvectors.shape[1]):  # 遍历所有列
    #     max_val = np.max(np.abs(eigenvectors[:, i]))  # 找到最大的绝对值
    #     eigenvectors[:, i] /= max_val  # 归一化
    master_nodes_length = len(master_nodes)
    # #对每个模态向量进行归一化
    for i in range(M.shape[1]):
        norm_val = np.sqrt(np.dot(eigenvectors[:, i].T, np.dot(M, eigenvectors[:, i])))
        eigenvectors[:, i] /= norm_val
    # Define transformation matrices
    T = eigenvectors[0:5*master_nodes_length,0:5*master_nodes_length]@np.linalg.pinv(eigenvectors[:,0:5*master_nodes_length])
    return M,K,T
    
# Example usage:
# MR,KR,T = SEREP(k, M, SlaveDofs)

import numpy as np

def reduce_force_matrix_dofs(force_matrix, num_nodes, dof_to_remove):
    """
    Reduces the degrees of freedom (DOFs) in a force matrix by removing the specified DOF from each node.

    This function is applicable to both 1D and 2D force matrices. It flattens a 2D matrix into 1D 
    before processing. The DOF to be removed is specified in a 0-indexed format, meaning that if 
    you want to remove the first DOF, you should pass 0 as the 'dof_to_remove' parameter.

    Parameters:
    force_matrix (numpy.ndarray): The original force matrix, either 1D or 2D.
    num_nodes (int): The number of nodes in the system.
    dof_to_remove (int): The index of the DOF to remove for each node (0-indexed).

    Returns:
    numpy.ndarray: The reduced force matrix with the specified DOFs removed.
    """
    # Determine the number of DOFs per node
    dofs_per_node = force_matrix.size // num_nodes

    # Identify which DOFs to keep
    keep_dofs = [i for node in range(num_nodes) 
                 for i in range(node * dofs_per_node, node * dofs_per_node + dofs_per_node) 
                 if i % dofs_per_node != dof_to_remove]

    # Flatten the force_matrix if it's 2D, then select the DOFs to keep
    return force_matrix.flatten()[keep_dofs]

# Example usage
# force_matrix = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # Example force matrix (can be 1D or 2D)
# num_nodes = 4
# dof_to_remove = 1  # Remove the second DOF of each node
# reduced_force_matrix = reduce_force_matrix_dofs(force_matrix, num_nodes, dof_to_remove)

# 恢复原始的节点顺序
def reorder_displacement_matrix(displacement_matrix, master_dofs, slave_dofs):
    """
    Reorder a displacement matrix from [master_dofs, slave_dofs] order to natural order (1, 2, 3, ...).

    :param displacement_matrix: The original displacement matrix (2D array).
    :param master_dofs: List or array of master degrees of freedom in current order.
    :param slave_dofs: List or array of slave degrees of freedom in current order.
    :return: The reordered displacement matrix.
    """
    # 创建从当前顺序到自然顺序的映射
    total_dofs = len(master_dofs) + len(slave_dofs)
    current_order = np.concatenate([master_dofs[::-1], slave_dofs])
    natural_order = np.empty(total_dofs, dtype=int)
    natural_order[current_order] = np.arange(total_dofs)

    # 根据自然顺序重排位移矩阵
    reordered_matrix = displacement_matrix[natural_order, :]
    return reordered_matrix

# Example usage:
# reordered_matrix = reorder_displacement_matrix(X_N, MasterDofs, SlaveDofs)
# The reordered_matrix is now reordered according to the specified node order

def get_fem_spring_stiffness(total_nodes, nodes_per_row, area):
    """
    Get the finite element spring stiffness matrix for a rectangular grid of nodes.

    Parameters:
    total_nodes (int): Total number of nodes in the grid.
    nodes_per_row (int): Number of nodes in each row of the grid.
    area (float): Area of each element in the grid.
    rho (float): Density of the material.
    g (float): Gravitational acceleration.

    Returns:
    tuple: A tuple containing lists of corner nodes, edge nodes, and interior nodes, 
    and the spring stiffness matrix.

    The function calculates the stiffness matrix for a finite element model.
    The nodes are assumed to be arranged in a rectangular grid.
    The stiffness is calculated differently for corner nodes, edge nodes, and interior nodes.
    The stiffness matrix is a diagonal matrix where the stiffness values are set at the
    third degree of freedom for each node.
    """
    # Calculate the number of nodes per column
    nodes_per_column = total_nodes // nodes_per_row

    # Initialize lists for different types of nodes
    corner_nodes = []
    edge_nodes = []
    interior_nodes = []

    # Initialize the stiffness matrix (a diagonal matrix)
    stiffness_matrix = np.zeros((total_nodes * 6, total_nodes * 6))
    length = 5
    width = 5
    draft = 0.5
    one_area_stiffness = calculate_hydrostatic_stiffness_matrix(length, width, draft)
    k33 = one_area_stiffness[2, 2]
    k44 = one_area_stiffness[3, 3]
    k55 = one_area_stiffness[4, 4]
    # Iterate over each node and classify them
    for row in range(nodes_per_column):
        for col in range(nodes_per_row):
            # Calculate the node number (starting from 1)
            node_number = row * nodes_per_row + col + 1

            # Determine the node type
            is_corner = (row in [0, nodes_per_column - 1]) and (col in [0, nodes_per_row - 1])
            is_edge = (row in [0, nodes_per_column - 1]) or (col in [0, nodes_per_row - 1])

            # Assign stiffness based on node type and add to the corresponding list
            if is_corner:
                corner_nodes.append(node_number)
                stiffness_33 = k33 / 4
                stiffness_44 = k44 / 4
                stiffness_55 = k55 / 4
            elif is_edge:
                edge_nodes.append(node_number)
                stiffness_33 = k33 / 2
                stiffness_44 = k44 / 2
                stiffness_55 = k55 / 2
            else:
                interior_nodes.append(node_number)
                stiffness_33 = k33
                stiffness_44 = k44
                stiffness_55 = k55

            # Set the stiffness values in the matrix (Python indexing starts from 0)
            stiffness_matrix[(node_number - 1) * 6 + 2, (node_number - 1) * 6 + 2] = stiffness_33  
            stiffness_matrix[(node_number - 1) * 6 + 3, (node_number - 1) * 6 + 3] = stiffness_44  
            stiffness_matrix[(node_number - 1) * 6 + 4, (node_number - 1) * 6 + 4] = stiffness_55 

    return stiffness_matrix


def calculate_hydrostatic_stiffness_matrix(length, width, draft, density=1025, gravity=9.81):
    """
    Calculate the hydrostatic stiffness matrix for a rectangular waterline area.
    """
    # 计算水线面积
    A_WL = length * width
    
    # 计算水线面的二次惯性矩
    I_xx = (length * width**3) / 12
    I_yy = (width * length**3) / 12
    
    # 计算浮体排水量
    delta = density * length * width * draft
    
    # 计算BM静水力矩臂
    BM_yy = I_xx / delta
    BM_xx = I_yy / delta
    
    # 计算净水刚度矩阵元素
    k33 = density * gravity * A_WL
    k44 = density * gravity * BM_yy * delta
    k55 = density * gravity * BM_xx * delta
    
    # 创建净水刚度矩阵
    K = np.zeros((6, 6))
    K[2, 2] = k33
    K[3, 3] = k44
    K[4, 4] = k55
    
    return K

# 引入静态凝聚和动态凝聚两种降维方法，三者模型的对比参考RODM研究报告
def static_condensation(K, M, master_dofs, slave_dofs):
    """
    Perform static condensation on a stiffness matrix and create a transformation matrix.
    
    Parameters:
    K (np.array): The full stiffness matrix.
    master_dofs (list): Indices of the master (primary) degrees of freedom.
    slave_dofs (list): Indices of the slave (secondary) degrees of freedom.
    
    Returns:
    np.array: The reduced stiffness matrix for the master DOFs.
    np.array: The transformation matrix to recover the full DOFs from the reduced DOFs.
    """
    # Extract submatrices
    K_mm = K[np.ix_(master_dofs, master_dofs)]
    K_ms = K[np.ix_(master_dofs, slave_dofs)]
    K_sm = K[np.ix_(slave_dofs, master_dofs)]
    K_ss = K[np.ix_(slave_dofs, slave_dofs)]
    # print(K_mm.shape, K_ms.shape, K_sm.shape, K_ss.shape)
    # Perform static condensation
    # epsilon = 1e-8
    # K_ss += np.eye(K_ss.shape[0])* epsilon
    K_reduced = K_mm - np.dot(np.dot(K_ms, np.linalg.inv(K_ss)), K_sm)

    # # Create the transformation matrix
    T_ms = -np.dot(np.linalg.inv(K_ss), K_sm)
    T_mm = np.eye(len(master_dofs))
    T = np.vstack((T_mm, T_ms))
    M_reduced = T.T@M@T

    return M_reduced,K_reduced,T


def dynamic_condensation(K, M, master_dofs, slave_dofs):
    """
    Perform dynamic condensation on a system and create a transformation matrix.

    Parameters:
    M (np.array): The mass matrix.
    K (np.array): The stiffness matrix.
    master_dofs (list): Indices of the master degrees of freedom.
    slave_dofs (list): Indices of the slave degrees of freedom.

    Returns:
    np.array: The reduced mass matrix.
    np.array: The reduced stiffness matrix.
    np.array: The transformation matrix to recover the full DOFs from the reduced DOFs.
    """
    # Extract submatrices
    M_mm = M[np.ix_(master_dofs, master_dofs)]
    M_ms = M[np.ix_(master_dofs, slave_dofs)]
    M_sm = M[np.ix_(slave_dofs, master_dofs)]
    M_ss = M[np.ix_(slave_dofs, slave_dofs)]
    
    K_mm = K[np.ix_(master_dofs, master_dofs)]
    K_ms = K[np.ix_(master_dofs, slave_dofs)]
    K_sm = K[np.ix_(slave_dofs, master_dofs)]
    K_ss = K[np.ix_(slave_dofs, slave_dofs)]

    # # Perform dynamic condensation
    M_reduced = M_mm - np.dot(np.dot(M_ms, np.linalg.inv(M_ss)), M_sm)
    K_reduced = K_mm - np.dot(np.dot(K_ms, np.linalg.inv(K_ss)), K_sm)

    # Create the transformation matrix
    T_ms = -np.dot(np.linalg.inv(K_ss), K_sm)
    T_mm = np.eye(len(master_dofs))

    # Stack to form the full transformation matrix
    T = np.vstack((T_mm, T_ms))
    # M_reduced = T.T@M@T
    # K_reduced = T.T@K@T
    return M_reduced, K_reduced, T

import numpy as np

def true_dynamic_condensation(K, M, master_dofs, slave_dofs, omega):
    """
    Perform *true* dynamic condensation on a system for a given single frequency omega,
    and create a frequency-dependent transformation matrix T(omega).

    Parameters:
    -----------
    K (np.array): The full stiffness matrix of shape (n, n).
    M (np.array): The full mass matrix of shape (n, n).
    master_dofs (list): Indices of the master degrees of freedom (kept DOFs).
    slave_dofs (list): Indices of the slave degrees of freedom (eliminated DOFs).
    omega (float): The reference frequency (rad/s) for the dynamic condensation.

    Returns:
    --------
    M_reduced (np.array): The reduced mass matrix in the master DOFs subspace.
    K_reduced (np.array): The reduced stiffness matrix in the master DOFs subspace.
    T_omega (np.array):  The transformation matrix T(omega),
                         dimension = [n, len(master_dofs)].
    """

    # 1. Extract submatrices
    K_mm = K[np.ix_(master_dofs, master_dofs)]
    K_ms = K[np.ix_(master_dofs, slave_dofs)]
    K_sm = K[np.ix_(slave_dofs, master_dofs)]
    K_ss = K[np.ix_(slave_dofs, slave_dofs)]

    M_mm = M[np.ix_(master_dofs, master_dofs)]
    M_ms = M[np.ix_(master_dofs, slave_dofs)]
    M_sm = M[np.ix_(slave_dofs, master_dofs)]
    M_ss = M[np.ix_(slave_dofs, slave_dofs)]

    # 2. Compute the frequency-dependent inverse block
    #    (K_ss - omega^2 M_ss)^(-1)
    # 需要保证 (K_ss - omega^2 M_ss) 非奇异（即不可逆），
    # 否则在这频率下会出现局部刚度与质量耦合的奇异情况。
    K_dyn_inv = np.linalg.inv(K_ss - (omega**2)*M_ss)

    # 3. Construct T_ms(omega)
    #    T_ms = - [ (K_ss - omega^2 M_ss)^(-1) (omega^2 M_sm - K_sm ) ]
    T_ms = - K_dyn_inv @ ((omega**2)*M_sm - K_sm)

    # 4. Stack to form the full transformation matrix T(omega)
    #    T(omega) = [ I
    #                 T_ms ]
    T_mm = np.eye(len(master_dofs))
    T_omega = np.vstack((T_mm, T_ms))

    # 5. Compute the reduced mass and stiffness matrices
    #    M^*(omega) = T(omega)^T M T(omega)
    #    K^*(omega) = T(omega)^T K T(omega)
    M_reduced = T_omega.T @ M @ T_omega
    K_reduced = T_omega.T @ K @ T_omega

    return M_reduced, K_reduced, T_omega



# # 示例输入
# length = 5
# width = 5
# draft = 0.5

# K = calculate_hydrostatic_stiffness_matrix(length, width, draft)

# # 示例参数
# total_nodes = 63  # 节点总数
# nodes_per_row = 21  # 每行节点数
# area = 1.0         # 单元面积
# rho = 1.0          # 密度
# g = 9.81           # 重力加速度

# # 获取节点位置和刚度矩阵
# stiffness_matrix = get_fem_spring_stiffness(total_nodes, nodes_per_row, area, rho, g)

# stiffness_matrix[:12, :12]  # 打印部分刚度矩阵进行验证
