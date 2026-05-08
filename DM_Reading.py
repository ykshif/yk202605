import numpy as np

def get_stiffness_matrix(file_path):
    """
    从指定的文件路径读取并返回刚度矩阵。

    参数：
    file_path (str)：包含刚度矩阵数据的文件路径。

    返回：
    numpy.ndarray：读取并恢复的刚度矩阵。
    """
    # 获取节点个数及自由度
    def get_size(file_path):
        max_node = 0
        with open(file_path, 'r') as file:
            for line in file:
                data = line.split(',')
                node1 = int(data[0])
                node2 = int(data[2])
                max_node = max(max_node, node1, node2)
        return max_node * 6  # 这里我们假设每个节点都有6个自由度

    # 从文件中读取稀疏矩阵数据
    def read_data(file_path, size):
        matrix = np.zeros((size, size))
        with open(file_path, 'r') as file:
            for line in file:
                data = line.split(',')
                node1 = int(data[0])
                dof1 = int(data[1])
                node2 = int(data[2])
                dof2 = int(data[3])
                value = float(data[4])
                # 计算在刚度矩阵中的位置，注意我们这里索引要减1，因为Python是0基索引
                index1 = (node1 - 1) * 6 + dof1 - 1
                index2 = (node2 - 1) * 6 + dof2 - 1
                matrix[index1, index2] = value
        return matrix

    size = get_size(file_path)
    matrix = read_data(file_path, size)

    # 将读取到的下三角部分复制到上三角部分，恢复完整的刚度矩阵
    for i in range(size):
        for j in range(i+1, size):
            matrix[i, j] = matrix[j, i]
    
    return matrix

from scipy.sparse import csr_matrix
import numpy as np

def get_stiffness_csr_matrix(file_path):
    """
    从指定的文件路径读取并返回刚度矩阵。

    参数：
    file_path (str)：包含刚度矩阵数据的文件路径。

    返回：
    scipy.sparse.csr_matrix：读取并恢复的刚度矩阵。
    """
    def get_size(file_path):
        max_node = 0
        with open(file_path, 'r') as file:
            for line in file:
                data = line.split(',')
                node1 = int(data[0])
                node2 = int(data[2])
                max_node = max(max_node, node1, node2)
        return max_node * 6  # 这里我们假设每个节点都有6个自由度

    # 从文件中读取稀疏矩阵数据
    def read_data(file_path, size):
        matrix = np.zeros((size, size))
        with open(file_path, 'r') as file:
            for line in file:
                data = line.split(',')
                node1 = int(data[0])
                dof1 = int(data[1])
                node2 = int(data[2])
                dof2 = int(data[3])
                value = float(data[4])
                # 计算在刚度矩阵中的位置，注意我们这里索引要减1，因为Python是0基索引
                index1 = (node1 - 1) * 6 + dof1 - 1
                index2 = (node2 - 1) * 6 + dof2 - 1
                matrix[index1, index2] = value
        return matrix

    size = get_size(file_path)
    matrix = read_data(file_path, size)

    # 将读取到的下三角部分复制到上三角部分，恢复完整的刚度矩阵
    for i in range(size):
        for j in range(i+1, size):
            matrix[i, j] = matrix[j, i]
    
    return csr_matrix(matrix)

#于2023年8月29日更新程序，由于出现密集矩阵消耗大量内存，故采用稀疏矩阵的方法
def get_stiffness_csr_matrix_optimized(file_path):
    def get_size(file_path):
        max_node = 0
        with open(file_path, 'r') as file:
            for line in file:
                data = line.split(',')
                node1 = int(data[0])
                node2 = int(data[2])
                max_node = max(max_node, node1, node2)
        return max_node * 6

    def read_data(file_path, size):
        row_indices = []
        col_indices = []
        values = []
        with open(file_path, 'r') as file:
            for line in file:
                data = line.split(',')
                node1 = int(data[0])
                dof1 = int(data[1])
                node2 = int(data[2])
                dof2 = int(data[3])
                value = float(data[4])
                
                index1 = (node1 - 1) * 6 + dof1 - 1
                index2 = (node2 - 1) * 6 + dof2 - 1
                
                row_indices.append(index1)
                col_indices.append(index2)
                values.append(value)
                
                if index1 != index2:
                    row_indices.append(index2)
                    col_indices.append(index1)
                    values.append(value)

        return csr_matrix((values, (row_indices, col_indices)), shape=(size, size))

    size = get_size(file_path)
    matrix = read_data(file_path, size)
    return matrix


def read_element_stiffness_matrix(file_path):
    """
    从指定的文件路径读取并返回单元刚度矩阵。
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    matrix_data = []
    found_element = False
    
    for i, line in enumerate(lines):
        if "ELEMENT NUMBER" in line:
            found_element = True
            # Skip to the MATRIX section
            while not lines[i].startswith("*MATRIX"):
                i += 1
            i += 1  # Move to the first line of the matrix
            
            # Read matrix data
            while i < len(lines) and not lines[i].startswith("** ELEMENT NUMBER"):
                for num in lines[i].split(','):
                    try:
                        matrix_data.append(float(num.strip().replace('E', 'e')))
                    except ValueError:
                        continue
                i += 1
            break
    
    if not found_element:
        raise ValueError("No element found in the file.")
    
    matrix_size = 24  # As per the given 24x24 symmetric matrix
    matrix = np.zeros((matrix_size, matrix_size))
    index = 0
    for i in range(matrix_size):
        for j in range(i, matrix_size):
            matrix[i, j] = matrix_data[index]
            matrix[j, i] = matrix_data[index]
            index += 1
    
    return matrix



