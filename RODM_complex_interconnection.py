import numpy as np
# This method creat in 2025年1月4日15:46:00

# define the function to apply hinge joints stiffness matrix to the global stiffness matrix
def apply_hinge_joints(N, k_hinge, hinges, direction = 0):
    """
    将铰链关节刚度矩阵应用到全局刚度矩阵中，适用于多个模块之间的铰接。

    参数:
    fem_kiffness (numpy.ndarray): 删除。修改为N节点数,建立约束的刚度矩阵,方便叠加。
    k_hinge (float): 铰链关节的刚度。
    hinges (list of tuple): 每个铰接节点对的列表，每个元组包含两个节点编号列表。
    direction (int): 铰链关节的方向，0表示沿着x轴布置铰接，1表示沿着y轴布置交接。

    返回:
    numpy.ndarray: 更新后的全局刚度矩阵。
    """
    fem_kiffness = np.zeros((6*N, 6*N))  
    # 定义铰链关节的刚度矩阵
    if direction == 0:
        KC = np.diag([k_hinge, k_hinge, k_hinge, k_hinge, 10, k_hinge])
    elif direction == 1:
        KC = np.diag([k_hinge, k_hinge, k_hinge, 10, k_hinge, k_hinge])
    else:
        raise ValueError("direction must be 0 or 1.")

    negative_KC = -KC

    # 处理所有铰接
    for nodes_k1, nodes_k2 in hinges:
        for node1, node2 in zip(nodes_k1, nodes_k2):
            # 计算在大矩阵中的索引位置
            index1 = (node1 - 1) * 6  # K_1 节点自由度起始位置
            index2 = (node2 - 1) * 6  # K_2 节点自由度起始位置

            # 在节点自身设置 KC
            fem_kiffness[index1:index1+6, index1:index1+6] += KC
            fem_kiffness[index2:index2+6, index2:index2+6] += KC

            # 设置两节点间的相互作用 -KC
            fem_kiffness[index1:index1+6, index2:index2+6] += negative_KC
            fem_kiffness[index2:index2+6, index1:index1+6] += negative_KC
    
    return fem_kiffness

# generate hinge pairs in x direction axis
def generate_hinge_x_pairs(grid_size=3, N=256, nodes_per_row=16, total_rows=16):  
    """  
    Generate hinge pairs with hinge constraints in the x direction for a square grid of modules.  
    For an n×n grid, there are (n-1)*n horizontal hinges.  

    Parameters:  
    grid_size (int): Size of the square grid (n×n modules)  
    N (int): The base value used for calculating module offsets  
    nodes_per_row (int): The number of nodes per row in each module  
    total_rows (int): The total number of rows in each module  

    Returns:  
    list: A list containing all hinge pairs for horizontal connections  
    """  
    hinges = []  # Initialize the list of all hinge pairs  

    # For each row in the grid  
    for row in range(grid_size):  
        # For each horizontal connection in the current row  
        # In each row, we have (grid_size-1) connections  
        for col in range(grid_size - 1):  
            # Calculate the base offset for the current module pair  
            # row * grid_size gives us the starting module number for each row  
            base_offset = (row * grid_size + col) * N  
            
            # Calculate nodes for the current module pair  
            # Last column of the first module  
            module_hinges_1 = [  
                base_offset + (r - 1) * nodes_per_row + nodes_per_row   
                for r in range(1, total_rows + 1)  
            ]  
            
            # First column of the second module  
            module_hinges_2 = [  
                base_offset + N + (r - 1) * nodes_per_row + 1   
                for r in range(1, total_rows + 1)  
            ]  

            hinge = [module_hinges_1, module_hinges_2]  
            hinges.append(hinge)  

    return hinges  


def print_hinge_x_pairs(grid_size=3, N=256, nodes_per_row=16, total_rows=16):  
    print(f"\nTesting {grid_size}×{grid_size} grid configuration:")  
    hinge_pairs = generate_hinge_x_pairs(grid_size=grid_size, N=N, nodes_per_row=nodes_per_row, total_rows=total_rows)  
    total_hinges = len(hinge_pairs)
    hinges = []  
    
    print(f"Total number of horizontal hinges: {total_hinges}")  
    print(f"Distribution: {grid_size} rows × {grid_size-1} connections per row")  
    
    for index, hinge in enumerate(hinge_pairs):  
        row_num = index // (grid_size-1)  
        col_num = index % (grid_size-1)  
        # print(f"\nHinge {index + 1} (Row {row_num + 1}, Connection {col_num + 1}):")  
        # print(f"Module 1 Hinges: {hinge[0]}")  
        # print(f"Module 2 Hinges: {hinge[1]}")  
        hinges.append((hinge[0], hinge[1]))  
    
    return hinges  

# Test with 3×3 and 4×4 configurations  
#hinges_3x3 = generate_hinge_x_pairs(3)  # Should generate 6 hinges (2 connections × 3 rows)  

# generate hinge pairs in y axis direction 
def generate_hinge_y_pairs(grid_size=3, N=256, nodes_per_row=16, total_rows=16):  
    """  
    Generate hinge pairs with hinge constraints in the y direction (vertical direction).  
    For 3×3 grid, vertical connections are:  
    1->4, 2->5, 3->6, 4->7, 5->8, 6->9  

    Parameters:  
    grid_size (int): Size of the square grid (n×n modules)  
    N (int): The base value used for calculating module offsets  
    nodes_per_row (int): The number of nodes per row in each module  
    total_rows (int): The total number of rows in each module  

    Returns:  
    list: A list containing all hinge pairs for vertical connections  
    """  
    hinges = []  
    
    # For modules in rows except the last row  
    for row in range(grid_size - 1):  
        # For each module in the current row  
        for col in range(grid_size):  
            # Calculate current module number and the module below it  
            current_module = row * grid_size + col + 1  
            below_module = current_module + grid_size  
            
            # Calculate base offsets for both modules  
            current_offset = (current_module - 1) * N  
            below_offset = (below_module - 1) * N  
            
            # Calculate nodes for the current module pair  
            # Bottom row of current module  
            module_hinges_1 = [  
                current_offset + total_rows * nodes_per_row - nodes_per_row + i   
                for i in range(1, nodes_per_row + 1)  
            ]  
            
            # Top row of module below  
            module_hinges_2 = [  
                below_offset + i   
                for i in range(1, nodes_per_row + 1)  
            ]  

            hinge = [module_hinges_1, module_hinges_2]  
            hinges.append(hinge)  

    return hinges  

def print_hinge_y_pairs(grid_size=3, N=256, nodes_per_row=16, total_rows=16):  
    """  
    Test function to demonstrate the results for y-direction hinges.  
    For 3×3 grid, displays connections as:  
    Hinge 1 (Module 1 to Module 4)  
    Hinge 2 (Module 2 to Module 5)  
    etc.  
    
    Parameters:  
    grid_size (int): Size of the square grid (n×n modules)  
    
    Returns:  
    list: List of hinge pairs  
    """  
    print(f"\nTesting {grid_size}×{grid_size} grid configuration:")  
    hinge_pairs = generate_hinge_y_pairs(grid_size=grid_size, N=N, nodes_per_row=nodes_per_row, total_rows=total_rows)  
    total_hinges = len(hinge_pairs)  
    hinges = []  
    
    print(f"Total number of vertical hinges: {total_hinges}")  
    print(f"Distribution: {grid_size} modules per row × {grid_size-1} rows")  
    
    for index, hinge in enumerate(hinge_pairs):  
        upper_module = index % grid_size + 1 + (index // grid_size) * grid_size  
        lower_module = upper_module + grid_size  
        
        # print(f"\nHinge {index + 1} (Module {upper_module} to Module {lower_module}):")  
        # print(f"Module {upper_module} Hinges: {hinge[0]}")  
        # print(f"Module {lower_module} Hinges: {hinge[1]}")  
        hinges.append((hinge[0], hinge[1]))  
    
    return hinges  

# hinges_3x3 = print_hinge_y_pairs(3)

# 可视化
import matplotlib.pyplot as plt  


def generate_hinge_pairs(num_modules=9, num_hinges=6, N=256, nodes_per_row=16, total_rows=16, modules_per_row=3):  
    """  
    Generate hinge pairs for the specified number of modules and hinges.  

    Parameters:  
    num_modules (int): The number of modules, default is 9.  
    num_hinges (int): The number of hinge pairs, default is 6.  
    N (int): The base value used for calculating hinge node offsets, default is 256.  
    nodes_per_row (int): The number of nodes per row, default is 16.  
    total_rows (int): The total number of rows, default is 16.  
    modules_per_row (int): The number of modules per row, default is 3.  

    Returns:  
    list: A list containing hinge pairs, where each hinge pair consists of hinge nodes from two modules.  
    """  
    hinges = []  # Initialize the list of hinge pairs  

    # Generate hinge pairs  
    for hinge_index in range(num_hinges):  
        hinge = []  # Initialize the current hinge pair  
        
        # Calculate hinge nodes for the two modules in the current hinge pair  
        for module_index in range(0, num_modules, 2):  # Process two modules at a time  
            if module_index < num_modules - 1:  # Ensure there are two modules available  
                # Select nodes from the last row of the first module (vertical direction)  
                module_hinges_1 = [  
                    (total_rows * nodes_per_row) - (nodes_per_row) + col  
                    for col in range(1, nodes_per_row + 1)  
                ]  
                
                # Select nodes from the first row of the next module (vertical direction)  
                module_hinges_2 = [  
                    N * modules_per_row + col  
                    for col in range(1, nodes_per_row + 1)  
                ]  
                
                # Add the hinge nodes to the hinge pair  
                hinge.append(module_hinges_1)  
                hinge.append(module_hinges_2)  

        # For subsequent hinge pairs, add an offset based on the hinge index  
        if hinge_index > 0:  
            # Ensure hinge is a list containing two lists  
            hinge = [  
                list(map(lambda x: x + hinge_index * N, hinge[0])),  
                list(map(lambda x: x + hinge_index * N, hinge[1]))  
            ]  
        
        hinges.append(hinge)  # Add the current hinge pair to the list of hinges  

    return hinges  # Return the generated list of hinge pairs  

def get_node_coordinates(node, num_modules, modules_per_row, nodes_per_row=16, total_rows=16):  
    """  
    Calculate the (x, y) coordinates of a given node based on its index.  

    Parameters:  
    node (int): The index of the node.  
    num_modules (int): The total number of modules.  
    modules_per_row (int): The number of modules per row.  
    nodes_per_row (int): The number of nodes per row, default is 16.  
    total_rows (int): The total number of rows, default is 16.  

    Returns:  
    tuple: The (x, y) coordinates of the node.  
    """  
    # Calculate the module index for the given node  
    module_index = (node - 1) // (nodes_per_row * nodes_per_row)  
    
    # Calculate the row and column of the module  
    row = module_index // modules_per_row  
    col = module_index % modules_per_row  
    
    # Calculate the position of the node within the module  
    node_in_module = (node - 1) % (nodes_per_row * nodes_per_row)  
    node_row = (nodes_per_row - 1) - (node_in_module // nodes_per_row)  # From top to bottom  
    node_col = node_in_module % nodes_per_row  # From left to right  
    
    # Calculate the (x, y) coordinates  
    x = col + node_col / (nodes_per_row - 1)  
    y = (num_modules // modules_per_row - row - 1) + node_row / (nodes_per_row - 1)  
    
    return x, y  # Return the coordinates  

def visualize_modules_and_hinges(num_modules=9, modules_per_row=3, num_hinges=6, N=256, nodes_per_row=16, total_rows=16):  
    """  
    Visualize the modules and hinge pairs in a grid format.  

    Parameters:  
    num_modules (int): The number of modules to visualize, default is 9.  
    modules_per_row (int): The number of modules per row, default is 3.  
    num_hinges (int): The number of hinge pairs to visualize, default is 6.  
    N (int): The base value used for calculating hinge node offsets, default is 256.  
    nodes_per_row (int): The number of nodes per row, default is 16.  
    total_rows (int): The total number of rows, default is 16.  
    """  
    # Create a figure and axis for the plot  
    fig, ax = plt.subplots(figsize=(15, 10))  
    
    # Generate hinge pairs  
    hinge_pairs = generate_hinge_pairs(num_modules, num_hinges, N, nodes_per_row, total_rows, modules_per_row)  
    
    # Calculate the total number of rows for the grid  
    total_rows = (num_modules + modules_per_row - 1) // modules_per_row  
    
    # Draw the module grid  
    for module_index in range(num_modules):  
        # Calculate the row and column of the module  
        row = module_index // modules_per_row  
        col = module_index % modules_per_row  
        
        # Draw a rectangle for the module  
        rect = plt.Rectangle((col, total_rows - row - 1), 1, 1, fill=False, edgecolor='blue', linewidth=2)  
        ax.add_patch(rect)  
        
        # Add the module number at the center of the module  
        ax.text(col + 0.5, total_rows - row - 0.5, str(module_index + 1), ha='center', va='center', fontweight='bold', fontsize=10)  
    
    # Draw the hinge points  
    colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink']  
    for hinge_index, hinge in enumerate(hinge_pairs):  
        color = colors[hinge_index % len(colors)]  
        
        # Draw the connection lines for the first list of nodes  
        list1_x = []  
        list1_y = []  
        for node in hinge[0]:  
            x, y = get_node_coordinates(node, num_modules, modules_per_row, nodes_per_row, total_rows)  
            list1_x.append(x)  
            list1_y.append(y)  
        ax.plot(list1_x, list1_y, color=color, linewidth=2, linestyle='--', label=f'Hinge {hinge_index + 1} - List 1')  
        
        # Draw the connection lines for the second list of nodes  
        list2_x = []  
        list2_y = []  
        for node in hinge[1]:  
            x, y = get_node_coordinates(node, num_modules, modules_per_row, nodes_per_row, total_rows)  
            list2_x.append(x)  
            list2_y.append(y)  
        ax.plot(list2_x, list2_y, color=color, linewidth=2, label=f'Hinge {hinge_index + 1} - List 2')  
    
    # Set plot properties  
    ax.set_xlim(-0.5, modules_per_row + 0.5)  
    ax.set_ylim(-0.5, total_rows + 0.5)  
    ax.set_aspect('equal')  
    ax.set_title(f'Module Grid with {num_hinges} Hinges\n{num_modules} Modules, {modules_per_row} Modules per Row')  
    ax.set_xlabel('Columns')  
    ax.set_ylabel('Rows')  
    ax.legend(loc='best')  
    ax.grid(True, linestyle='--', alpha=0.7)  
    
    plt.tight_layout()  # Adjust layout to prevent clipping  
    plt.show()  # Display the plot  

# Visualize 16 modules, 4 modules per row, and 12 hinge pairs  
#visualize_modules_and_hinges(num_modules=16, modules_per_row=4, num_hinges=12, N=256, nodes_per_row=16, total_rows=16)