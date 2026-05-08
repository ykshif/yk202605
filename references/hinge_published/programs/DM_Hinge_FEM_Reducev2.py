import numpy as np
import DM_Reading as dm_r

def main_hinge():
    # 定义网格参数
    nodes_per_row = 61
    rows_per_column = 13
    
    # 定义需要的列号
    column_number_1 = 31
    column_number_2 = 32
    
    # 生成节点列表
    node_list_1 = calculate_column_node_indices(column_number_1, nodes_per_row, rows_per_column)
    node_list_2 = calculate_column_node_indices(column_number_2, nodes_per_row, rows_per_column)

    # 文件路径
    file_path_total = 'E:\\phd\\Code\\DM-FEM2D\\StructureData\\Job-1_55_STIF1.mtx'
    file_path_element = 'E:\\phd\\Code\\DM-FEM2D\\StructureData\\Hinge\\Element_stiff.mtx'
    
    # 读取刚度矩阵
    K_element = dm_r.read_element_stiffness_matrix(file_path_element)
    K_total = dm_r.get_stiffness_matrix(file_path_total)
    
    # 生成单元信息
    elements = generate_elements(node_list_1, node_list_2)
    
    # 更新全局刚度矩阵，移除指定单元的刚度
    K_no_connect = update_global_stiffness_matrix(np.copy(K_total), K_element, elements)
    
    # 铰接刚度值
    k_hinge = 10e15
    
    # 添加铰接关系
    K_hinge_connect = add_hinge_connections(np.copy(K_total), node_list_1, node_list_2, k_hinge)
    
    # 返回两个刚度矩阵：一个不包含连接，一个包含铰接连接
    return K_hinge_connect

def calculate_column_node_indices(column_number, nodes_per_row, rows_per_column):
    """
    计算给定列的节点编号。

    :param column_number: int，指定要计算的列号（从1开始计数）
    :param nodes_per_row: int，每行的节点数
    :param rows_per_column: int，每列的节点数
    :return: list，该列的所有节点编号
    """
    # 验证输入
    if column_number < 1 or column_number > nodes_per_row:
        raise ValueError("列号超出范围")
    
    # 计算该列的所有节点编号
    column_indices = [(row_index - 1) * nodes_per_row + column_number
                      for row_index in range(1, rows_per_column + 1)]
    return column_indices

# 使用示例
# nodes_per_row = 61
# rows_per_column = 13
# column_number = 31  # 例如，计算第一列的节点编号
# column_node_indices = calculate_column_node_indices(column_number, nodes_per_row, rows_per_column)
# print("节点编号：", column_node_indices)


def generate_elements(node_list_1, node_list_2):
    """
    根据提供的两个节点列表生成单元信息。

    参数:
    node_list_1 : list of int
        第一个节点列表。
    node_list_2 : list of int
        第二个节点列表。

    返回:
    list of list of int
        生成的单元信息。
    """
    elements = []
    # 我们将循环通过节点，除了最后一个，以便将每组与下一组配对
    for i in range(len(node_list_1) - 1):  # 减1是因为我们在向前看一个
        n1 = node_list_1[i]
        n2 = node_list_2[i]
        n3 = node_list_1[i + 1]
        n4 = node_list_2[i + 1]
        elements.append([n1, n2, n3, n4])
    
    return elements


def update_global_stiffness_matrix(k_total, k_element, elements):
    """
    更新全局刚度矩阵，减去指定单元的刚度矩阵。

    参数:
    k_total : numpy.ndarray
        全局刚度矩阵。
    k_element : numpy.ndarray
        单元刚度矩阵。
    elements : list of list of int
        单元节点信息，每个列表项包含一个单元的节点编号。

    返回:
    numpy.ndarray
        更新后的全局刚度矩阵。
    """
    # 对每个单元进行遍历
    for elem in elements:
        # 计算全局索引
        global_indices = []
        for node in elem:
            global_indices.extend(range((node - 1) * 6, (node - 1) * 6 + 6))

        # 将 k_element 的值逐一减去到 k_total 的对应位置
        for i, gi in enumerate(global_indices):
            for j, gj in enumerate(global_indices):
                k_total[gi, gj] -= k_element[i, j]

    return k_total

import numpy as np

def add_hinge_connections(big_matrix, nodes_k1, nodes_k2, k_hinge):
    """
    在全局刚度矩阵中添加铰接关系。

    参数:
    big_matrix : numpy.ndarray
        全局刚度矩阵。
    nodes_k1 : list of int
        第一组节点编号。
    nodes_k2 : list of int
        第二组节点编号。
    k_hinge : float
        铰接自由度的刚度值。

    返回:
    None，直接修改 big_matrix 矩阵。
    """
    # 创建 KC 矩阵
    KC = np.diag([k_hinge, k_hinge, k_hinge, k_hinge, 0, k_hinge])
    negative_KC = -KC

    for node1, node2 in zip(nodes_k1, nodes_k2):
        # 计算在大矩阵中的索引位置
        index1 = (node1 - 1) * 6  # K_1 节点自由度起始位置
        index2 = (node2 - 1) * 6  # K_2 节点自由度起始位置

        # 在节点自身设置 KC
        big_matrix[index1:index1+6, index1:index1+6] += KC
        big_matrix[index2:index2+6, index2:index2+6] += KC

        # 设置两节点间的相互作用 -KC
        big_matrix[index1:index1+6, index2:index2+6] += negative_KC
        big_matrix[index2:index2+6, index1:index1+6] += negative_KC
    return big_matrix


# # 示例用法
# node_list_1 = [30, 91, 152, 213, 274, 335, 396, 457, 518, 579, 640, 701, 762]
# node_list_2 = [31, 92, 153, 214, 275, 336, 397, 458, 519, 580, 641, 702, 763]

# # 生成单元信息
# elements = generate_elements(node_list_1, node_list_2)
# file_path_total = 'E:\phd\Code\DM-FEM2D\StructureData\Job-1_55_STIF1.mtx'
# file_path_element = 'E:\\phd\\Code\\DM-FEM2D\\StructureData\\Hinge\\Element_stiff.mtx'
# K_element = dm_r.read_element_stiffness_matrix(file_path_element)
# K_total = dm_r.get_stiffness_matrix(file_path_total)
# K_no_connect = update_global_stiffness_matrix(K_total,K_element,elements)
# k_hinge = 10e15
# K_hinge_connect = add_hinge_connections(K_total,node_list_1,node_list_2,k_hinge)