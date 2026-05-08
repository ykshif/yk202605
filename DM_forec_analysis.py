import numpy as np
import matplotlib.pyplot as plt

class ForceAnalysis:
    def __init__(self, N, H, module_nodes, module_number, module_distribution="1D", module_rows=1, module_cols=1, dof=6, element_length=15, element_width=30):
        """
        Initialize the ForceAnalysis class.

        Parameters:
        - N (int): Total number of nodes.
        - H (int): Number of rows.
        - module_nodes (int): Number of nodes in one row of a module.
        - module_number (int): Number of modules.
        - dof (int, optional): Number of degrees of freedom for each node. Defaults to 6.
        - element_length (float, optional): Length of the element. Defaults to 15.
        - element_width (float, optional): Width of the element. Defaults to 30.
        - module_distribution (str, optional): Module distribution type, either "1D" or "2D". Defaults to "1D".
        - module_rows (int, optional): Number of rows for modules. Defaults to 1.
        - module_cols (int, optional): Number of columns for modules. Defaults to 1.
        """
        self.N = N
        self.H = H
        self.module_nodes = module_nodes
        self.module_number = module_number
        self.dof = dof
        self.element_length = element_length
        self.element_width = element_width
        self.module_distribution = module_distribution
        self.module_rows = module_rows
        self.module_cols = module_cols
        
        self.modules = self.generate_module_nodes()
        
        # Select the appropriate method to generate modules based on module_distribution
        if module_distribution == "1D":
            self.modules = self.generate_module_nodes()
            #每个模块上的节点总数

        elif module_distribution == "2D":
            if module_rows is None or module_cols is None:
                raise ValueError("For 2D distribution, both module_rows and module_cols must be provided.")
            self.modules = self.generate_module_nodes_2D(module_rows, module_cols)
#             self.nodes_per_module = self.modules[0].size
        else:
            raise ValueError("Invalid module_distribution value. It should be either '1D' or '2D'.")
        
        self.nodes_per_module = len(self.modules[0]) if self.modules else 0


    def generate_module_nodes(self):
        """
        Generate modules for the parameters provided during initialization.
        
        Returns:
        - List of lists: Each inner list contains the node numbers for a module.
        """
        G = self.N // self.H
        modules = []
        for i in range(self.module_number):
            module = []
            for j in range(self.H):
                base = j * G + i * self.module_nodes + 1
                module.extend(list(range(base, base + self.module_nodes)))
            modules.append(module)
 
        for i in range(1, self.module_number):  # start from the second module
            modules[i] = [node - i for node in modules[i]]
        
        return modules
    
    
    def generate_module_nodes_2D(self, module_rows, module_cols):
        """
        Generate 2D modules for the parameters provided during initialization.
        
        Returns:
        - List of lists: Each inner list contains the node numbers for a module.
        """
        nodes = np.arange(1, self.N+1).reshape(self.H, self.H)
        modules = []
        
        for row in range(module_rows):
            for col in range(module_cols):
                start_row = row * (self.module_nodes - 1)
                end_row = start_row + self.module_nodes
                start_col = col * (self.module_nodes - 1)
                end_col = start_col + self.module_nodes
                
                # Ensure indices do not exceed boundaries
                end_row = min(end_row, self.H)
                end_col = min(end_col, self.H)
                
                module = nodes[start_row:end_row, start_col:end_col].ravel().tolist() # Flatten the 2D array and convert to list
                modules.append(module)
                
        return modules
    

    def compute_module_displacements(self, displacement_matrix):
        """
        Extract module displacements from the provided global displacement matrix.
        
        Parameters:
        - displacement_matrix (numpy array): The global displacement matrix.

        Returns:
        - List of numpy arrays: Each array contains the displacement information for a module.
        """
        displacement_matrix = displacement_matrix.reshape(self.N, self.dof)
        module_displacements = []
        for module in self.modules:
            displacements = displacement_matrix[np.array(module) - 1]  # -1 because Python indexing starts from 0
            module_displacements.append(displacements)
        #
        return [displacement.reshape(self.nodes_per_module * self.dof, 1) for displacement in module_displacements]
    

    def compute_module_forces(self, K_element, module_displacements):
        """
        Compute the forces for each module using the provided stiffness matrix and module displacements.
        
        Parameters:
        - K_element (numpy array): The stiffness matrix.
        - module_displacements (list of numpy arrays): The displacements for each module.

        Returns:
        - List of numpy arrays: Each array contains the force information for a module.
        """
        return [np.dot(K_element, displacement) for displacement in module_displacements]

    def map_forces_to_global_nodes(self, module_forces):
        """
        Map the forces from each module back to the global nodes.
        
        Parameters:
        - module_forces (list of numpy arrays): The forces for each module.

        Returns:
        - numpy array: The global forces for all nodes.
        """
        global_forces = np.zeros((self.N, 5), dtype=np.complex128)
        processed_nodes = set()

        for module_idx, module in enumerate(self.modules):
            for node_idx, node in enumerate(module):
                if node not in processed_nodes:
                    node_forces = module_forces[module_idx].reshape(self.nodes_per_module, self.dof)
                    global_forces[node-1] = node_forces[node_idx]
                    processed_nodes.add(node)
        
        return global_forces

    def get_middle_interface_forces(self, global_forces):
        """
        Extract forces from the module interface nodes.
        
        Parameters:
        - global_forces (numpy array): The global forces for all nodes.
        
        Returns:
        - numpy array: The forces at the module interface nodes.
        """
        M = global_forces.reshape(self.N * self.dof, 1)[4::5]
        
        # Calculate the starting and ending indices for the interface nodes
        start_idx = int((self.H // 2) * (self.N/self.H))
        end_idx = int(start_idx + (self.N/self.H))
        #间隔
        interval = int(self.module_nodes - 1)
        return abs(M[start_idx:end_idx:interval] / self.element_width)
    

    def get_boundary_node_forces(self, global_forces):
        """
        获取边界节点上的力。

        参数:
        - global_forces (numpy array): 所有节点上的全局力。

        返回:
        - numpy array: 边界节点上的力。
        """
        boundary_nodes = self._get_boundary_nodes()
        boundary_forces = global_forces[np.array(boundary_nodes) - 1]  # -1 因为Python索引从0开始
        return np.array(boundary_forces)

    def _get_boundary_nodes(self):
        """
        获取边界节点。

        返回:
        - numpy array: 边界节点列表。
        """
        boundary_nodes = []
        node = 1
        while node <= self.N:
            boundary_nodes.append(node)
            if (node % (self.N/self.H)) == 0:
                node += 1  # 如果是模块的最后一个节点，则下一个节点也是边界节点
            else:
                node += self.module_nodes-1 # 每个一个模块的距离取点
        return np.array(boundary_nodes)
        

    def plot_forces(self, global_forces):
        """
        Plot the forces using scatter plot and cubic spline interpolation.
        
        Parameters:
        - global_forces (numpy array): The global forces for all nodes.
        """
        data = self.get_middle_interface_forces(global_forces)

        x = np.arange(len(data))
        y = data[:, 0]
        from scipy import interpolate
        spline = interpolate.CubicSpline(x, y)
        x_new = np.linspace(0, len(data)-1, 300)
        y_new = spline(x_new)

        plt.scatter(x, y, color='black', label='Interface')
        plt.plot(x_new, y_new,color = 'black', label='Present')
        plt.legend()
        # plt.title("Scatter Plot and Cubic Spline Interpolation")
        plt.xlabel(r"$x$/$L$")
        plt.ylabel(r"$M_y$($MN{\cdot}m$)")
        # plt.grid(True)
        plt.show()
        

#     def plot_2D_heatmap(self, global_forces):
#         """
#         Plot a 2D heatmap of the forces on the boundary nodes.

#         Parameters:
#         - global_forces (numpy array): The global forces for all nodes.
#         """
#         M_B = self.get_boundary_node_forces(global_forces) # M_B is a 2D array

#         M_B = M_B.reshape(M_B.shape[0]*self.dof, 1)[4::self.dof].reshape(self.H, self.module_number+1) 
#         M_B = M_B[1:-1]  # Remove the first and last rows
#         plt.figure(figsize=(10, 2))
#         plt.imshow(abs(M_B)/self.element_width, aspect='auto', cmap='coolwarm', origin='lower', interpolation='spline16')
#         plt.colorbar(label='MY Force')
#         plt.title('2D Heatmap of MY Forces')
#         plt.xlabel('Width')
#         plt.ylabel('Length')
#         plt.show()

    def plot_2D_heatmap(self, global_forces):
        """
        Plot a 2D heatmap of the forces on the boundary nodes for both 1D and 2D modules.

        Parameters:
        - global_forces (numpy array): The global forces for all nodes.
        """
        M_B = self.get_boundary_node_forces(global_forces) # M_B is a 2D array

        if self.module_distribution == "1D":
            M_B = M_B.reshape(M_B.shape[0]*self.dof, 1)[4::self.dof].reshape(self.H, self.module_number+1) 
            M_B = M_B[1:-1]  # Remove the first and last rows
            fig_size = (10, 2)
            aspect_ratio = 'auto'
            interpolation_type = 'spline16'
        elif self.module_distribution == "2D":
            M_B = M_B.reshape(M_B.shape[0] * self.dof, 1)[4::self.dof].reshape(self.H, self.module_cols + 1) 
            M_B = M_B[1:-1]  # Remove the first and last rows
            M_B = M_B[0:self.H-2:self.module_nodes-1] #由于图像存在不均匀的问题，对数据进行处理，只要不取到上下模块交界处即可
            fig_size = (10, 10)
            aspect_ratio = 'auto'
            interpolation_type = 'spline36'
        else:
            raise ValueError(f"Unsupported module_distribution value: {self.module_distribution}")

        plt.figure(figsize=fig_size)
        plt.imshow(abs(M_B)/self.element_width, aspect=aspect_ratio, cmap='coolwarm', origin='lower',      interpolation=interpolation_type)
        plt.colorbar(label='MY Force')
        plt.title('2D Heatmap of MY Forces')
        plt.xlabel('Width')
        plt.ylabel('Length')
        plt.show()


#     def plot_3D_surface(self, global_forces):
#         """
#         Plot a 3D surface of the forces on the boundary nodes.

#         Parameters:
#         - global_forces (numpy array): The global forces for all nodes.
#         """
#         M_B = self.get_boundary_node_forces(global_forces)
#         M_B = M_B.reshape(M_B.shape[0]*self.dof, 1)[4::self.dof].reshape(self.H, self.module_number+1)
#         M_B = M_B[1:-1]  # Remove the first and last rows
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')

#         # 生成网格数据
#         x = np.arange(M_B.shape[1])
#         y = np.arange(M_B.shape[0])
#         x, y = np.meshgrid(x, y)

#         # 绘制三维表面图
#         surf = ax.plot_surface(x, y, abs(M_B)/self.element_width, cmap='coolwarm', linewidth=0, antialiased=True)

#         # 添加颜色条和标题
#         fig.colorbar(surf, ax=ax, label='MY Force')
#         ax.set_title('3D Surface Plot of MY Forces')
#         ax.set_xlabel('Width')
#         ax.set_ylabel('Length')
#         ax.set_zlabel('Force')
#         plt.show()

    def plot_3D_surface(self, global_forces):
        """
        Plot a 3D surface of the forces on the boundary nodes for both 1D and 2D modules.

        Parameters:
        - global_forces (numpy array): The global forces for all nodes.
        """
        M_B = self.get_boundary_node_forces(global_forces)

        if self.module_distribution == "1D":
            M_B = M_B.reshape(M_B.shape[0]*self.dof, 1)[4::self.dof].reshape(self.H, self.module_number+1) 
            M_B = M_B[1:-1]  # Remove the first and last rows
        elif self.module_distribution == "2D":
            M_B = M_B.reshape(M_B.shape[0]*self.dof, 1)[4::self.dof].reshape(self.H, self.module_cols + 1) 
            M_B = M_B[1:-1]  # Remove the first and last rows
            M_B = M_B[0:self.H-2:self.module_nodes-1] # 由于图像存在不均匀的问题，对数据进行处理，只要不取到上下模块交界处即可
        else:
            raise ValueError(f"Unsupported module_distribution value: {self.module_distribution}")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 生成网格数据
        x = np.arange(M_B.shape[1])
        y = np.arange(M_B.shape[0])
        x, y = np.meshgrid(x, y)

        # 绘制三维表面图
        surf = ax.plot_surface(x, y, abs(M_B)/self.element_width, cmap='coolwarm', linewidth=0, antialiased=True)

        # 添加颜色条和标题
        fig.colorbar(surf, ax=ax, label='MY Force')
        ax.set_title('3D Surface Plot of MY Forces')
        ax.set_xlabel('Width')
        ax.set_ylabel('Length')
        ax.set_zlabel('Force')
        plt.show()

    


