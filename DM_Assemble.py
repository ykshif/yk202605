import numpy as np

def insert_matrix(N, K_additional_mass, node_ids, dof_num = 6):
    """
    在总刚度矩阵K_total的指定节点位置插入附加质量刚度矩阵K_additional_mass。

    参数：
    K_total (numpy.ndarray)：总刚度矩阵，形状为(N*6, N*6)，N为节点数。
    K_additional_mass (numpy.ndarray)：附加质量的刚度矩阵，形状为(M, M)，M为附加质量矩阵的维度。
    node_ids (list)：要插入附加质量刚度矩阵的节点ID列表。

    返回：
    numpy.ndarray：更新后的总刚度矩阵。
    """
    # 找出插入总刚度矩阵的位置
    # 初始化总刚度矩阵
    K_total = np.zeros((N*dof_num, N*dof_num))
    index = []
    for node_id in node_ids:
        for i in range(dof_num): # 因为每个节点有6个自由度
            index.append((node_id-1)*dof_num+i)

    # 插入附加质量刚度矩阵到总刚度矩阵
    for i, row in enumerate(index):
        for j, col in enumerate(index):
            K_total[row, col] += K_additional_mass[i, j]

    return K_total

from scipy.sparse import lil_matrix, csr_matrix
# 处理为稀疏矩阵
def sparse_insert_matrix(N, K_additional_mass, node_ids):
    """
    在总刚度矩阵K_total的指定节点位置插入附加质量刚度矩阵K_additional_mass。

    参数：
    N (int)：节点数。
    K_additional_mass (numpy.ndarray)：附加质量的刚度矩阵，形状为(M, M)，M为附加质量矩阵的维度。
    node_ids (list)：要插入附加质量刚度矩阵的节点ID列表。

    返回：
    scipy.sparse.lil_matrix：更新后的总刚度矩阵。
    """
    # 初始化总刚度矩阵（使用稀疏矩阵格式）
    K_total = lil_matrix((N*6, N*6), dtype='float64')
    
    index = []
    for node_id in node_ids:
        for i in range(6):  # 因为每个节点有6个自由度
            index.append((node_id-1)*6 + i)

    # 插入附加质量刚度矩阵到总刚度矩阵
    for i, row in enumerate(index):
        for j, col in enumerate(index):
            K_total[row, col] += K_additional_mass[i, j]

    return csr_matrix(K_total)


def extend_force_matrix(force_vector, node_ids, total_nodes, dof_num = 6):
    """
    根据指定的节点ID列表，将原始外力向量扩展到适用于所有节点的大小。

    参数：
    force_vector (numpy.ndarray)：原始外力向量，形状为(1, M)，M为节点数。
    node_ids (list)：需要应用外力的节点ID列表。
    total_nodes (int)：总的节点数。

    返回：
    numpy.ndarray：扩展后的外力矩阵，形状为(1, total_nodes*6)。
    """
    # 初始化一个形状为(1, total_nodes*6)的零向量
    extended_force_vector = np.zeros((1, total_nodes*dof_num), dtype=complex)

    # # 根据节点ID列表，将原始外力向量的值复制到扩展后的外力向量的适当位置
    # for node_id in node_ids:
    #     # 因为每个节点有6个自由度，所以外力向量的每个元素都需要复制到6个位置
    #     for i in range(6):
    #         extended_force_vector[0, (node_id-1)*6+i] = force_vector[0, i]
    #修正20230803
    for i, node_id in enumerate(node_ids):
        for j in range(dof_num):
            extended_force_vector[0, (node_id-1)*dof_num+j] = force_vector[0, i*dof_num+j]


    return extended_force_vector


def solve_frequency_domain(mass,damping,stiffness,F, omega):
    """
    解决频域的MCK方程，并返回位移。

    参数：
    inertia_matrix (numpy.ndarray)：惯性矩阵。
    added_mass (numpy.ndarray)：附加质量矩阵。
    radiation_damping (numpy.ndarray)：辐射阻尼矩阵。
    hydrostatic_stiffness (numpy.ndarray)：水静刚度矩阵。
    F (numpy.ndarray)：外力矩阵。
    omega (float)：角频率。

    返回：
    numpy.ndarray：位移。
    """
    # 根据给定的公式构建左侧矩阵H
    H = (-omega**2*(mass)-1j*omega*damping+stiffness)

    F = F.T

    # 解线性方程
    X = np.linalg.solve(H, F)
    #X = np.dot(np.linalg.pinv(H), F)

    return X

from scipy.sparse.linalg import spsolve

def sparse_solve_frequency_domain(mass, damping, stiffness, F, omega):
    """
    解决频域的MCK方程，并返回位移。

    参数：
    mass (scipy.sparse.spmatrix)：惯性矩阵。
    damping (scipy.sparse.spmatrix)：阻尼矩阵。
    stiffness (scipy.sparse.spmatrix)：刚度矩阵。
    F (numpy.ndarray)：外力矩阵。
    omega (float)：角频率。

    返回：
    numpy.ndarray：位移。
    """
    # 根据给定的公式构建左侧矩阵H
    H = (-omega**2 * mass - 1j * omega * damping + stiffness)

    # 转置外力向量
    F = F.T

    # 解线性方程
    X = spsolve(H, F)

    return X


def calculate_node_positions(first_node, node_interval, num_nodes):
    """
    计算作用力的节点位置。

    参数:
    first_node: 初始节点号
    node_interval: 节点间隔
    num_nodes: 节点数量

    返回:
    nodes: 一个列表，包含所有的节点位置
    """
    # 初始化节点列表
    nodes = []

    # 计算所有节点号
    for i in range(num_nodes):
        node = first_node - i * node_interval
        nodes.append(node)

    return nodes


def calculate_2d_node_positions_descending(first_node, col_interval, num_nodes_row, num_rows, num_cols):
    """
    Calculate the positions of nodes in a 2D structure in descending order.

    Parameters:
    - first_node: The ID of the first force node.  
    - col_interval: The interval between nodes in two module.
    - num_nodes_row: The number of nodes in a row.
    - num_rows: The number of modules.
    - num_cols: The number of modules.

    Returns:
    - nodes: A list of node IDs.
    """
    
    # Calculate the row interval based on the given first node.
    row_interval = num_nodes_row * col_interval

    nodes = []
    for i in range(num_rows):
        for j in range(num_cols):
            # Compute the current node's ID.
            node = first_node - i * row_interval - j * col_interval
            nodes.append(node)
    return nodes