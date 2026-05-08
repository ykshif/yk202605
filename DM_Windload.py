import numpy as np
import matplotlib.pyplot as plt

class WindLoad:
    def __init__(self, U10, z, file_path, wind_direction, A=2, total_rows=10, total_cols=50, rho=1.225, alpha=0.125):
        """
        Initialize the WindLoad class with the given parameters.
        
        Parameters:
        - U10: Wind speed at 10m above ground level.
        - z: Height above ground level.
        - file_path: Path to the scatter data file.
        - wind_dirction: define wind direction
        - A: Area (default is 1) projection.
        - total_rows: Number of rows for wind load coefficient matrix.
        - total_cols: Number of columns for wind load coefficient matrix.
        - rho: Air density (default is 1.225 kg/m^3).
        - alpha: Wind profile power law exponent (default is 0.125).
        """
        self.U10 = U10
        self.z = z
        self.file_path = file_path
        self.A = A
        self.total_rows = total_rows
        self.total_cols = total_cols
        self.rho = rho
        self.alpha = alpha
        self.wind_direction = wind_direction

    def adjust_wind_speed(self):
        """
        Compute and return the adjusted wind speed based on height.
        Adjusts the wind speed using the power law.
        """
        return abs(self.U10 * (self.z / 10)**self.alpha)

    def turbulence_intensity(self):
        """
        Compute and return the turbulence intensity based on height.
        Calculates turbulence intensity using a piecewise power law.
        """
        exponent = -0.125 if self.z <= 20 else -0.275
        return 0.15 * (self.z / 20)**exponent

    def api_spectrum(self):
        """
        Compute and return the API wind spectrum.
        Calculates the wind spectrum based on API standards.
        """
        f = np.arange(0.01, 2.0, 0.01) #np.linspace(0.01, 2.0, 40)
        U = self.adjust_wind_speed()
        Ti = self.turbulence_intensity()
        fp_value = 0.025 * U / self.z
        return U**2 * Ti**2 / fp_value * (1 + 1.5 * (f / fp_value))**(-5 / 3)

    def compute_amplitude_from_spectrum(self):
        """
        Compute and return the amplitude values from the wind spectrum.
        Calculates the amplitude using the square root of the spectrum.
        """
        spectrum = self.api_spectrum()
        delta_omega = 0.01
        return np.sqrt(2 * spectrum * delta_omega)
    
    def compute_amplitude_for_frequency(self, target_frequency):
        """
        Compute and return the amplitude for a given frequency.
        
        Parameters:
        - target_frequency: The desired frequency for which amplitude is computed.
        """
        amplitude = self.compute_amplitude_from_spectrum()
        f = np.arange(0.01, 2.0, 0.01)
        index = np.abs(f - target_frequency).argmin()
        return amplitude[index]

#该方法不适用多自由度问题，由于相位是随机产生的，相位之间应当具有一定关系。
    # def compute_amplitude_for_frequency(self, target_frequency):
    #     """
    #     Compute and return the amplitude for a given frequency.
        
    #     Parameters:
    #     - target_frequency: The desired frequency for which amplitude is computed.
    #     """
    #     amplitude = self.compute_amplitude_from_spectrum()
    #     f = np.linspace(0.01, 2.0, 40)
    #     index = np.abs(f - target_frequency).argmin()
    #     # # Set a random seed for reproducibility
    #     np.random.seed(0)
    #     # Generate random phase for each frequency
    #     phi = np.random.uniform(0, 2*np.pi, 40)
    #     # Convert amplitude to complex form using the generated phase
    #     amplitude = amplitude * (np.cos(phi) + 1j * np.sin(phi))
    #     return amplitude[index]

    def wind_load_coefficient(self):
        """
        Compute and return the wind load coefficient based on scatter data.
        Uses scatter data from a file to compute the wind load coefficient.
        """
        
        def read_scatter_data():
            """Helper function to read the scatter data from the class file_path."""
            data = np.loadtxt(self.file_path)
            return data[:, 0], data[:, 1]

        def extend_last_value(x_values, y_values):
            """Helper function to extend the last y value to the total_cols."""
            extended_x_values = np.arange(1, self.total_cols + 1)
            extended_y_values = np.concatenate([y_values, [y_values[-1]] * (self.total_cols - len(y_values))])
            return extended_x_values, extended_y_values

        x_data, y_data = read_scatter_data()
        if self.wind_direction == 0:
            _, extended_y = extend_last_value(x_data, y_data)
        else:
            _, extended_y = extend_last_value(x_data, y_data)
            extended_y = extended_y  #[::-1]  # 由于输入数据及方向问题反转该参数
        return np.tile(extended_y, (self.total_rows, 1))

    def compute_wind_force(self, target_frequency,dof=0):
        """
        Compute and return the wind force for a given frequency.
        
        Parameters:
        - target_frequency: The desired frequency for which wind force is computed.
        """
        amplitude = self.compute_amplitude_for_frequency(target_frequency)
        Cd = self.wind_load_coefficient()
        V_avg = self.adjust_wind_speed()
        #每个方向上的力
        wind_force_in_one_dof = 2 * Cd * V_avg * amplitude * self.A * self.rho
        #插入完整矩阵当中
        wind_force_in_one_dof = wind_force_in_one_dof.reshape(self.total_rows*self.total_cols)
        # Complete node force matrix
        force_matrix = np.zeros((1,self.total_rows*self.total_cols*6),dtype=complex)
        force_matrix[0,dof::6] = wind_force_in_one_dof

        return force_matrix

    def compute_wind_damping(self,dof=0):
        """
        Compute and return the wind damping.
        Calculates the wind damping based on wind speed and coefficient.
        """
        V_avg = self.adjust_wind_speed()
        Cd = self.wind_load_coefficient()
        wind_damping = 2 * Cd * V_avg * self.A * self.rho
        wind_damping = wind_damping.reshape(self.total_rows*self.total_cols)
        global_damping = np.zeros(self.total_rows*self.total_cols*6)
        # 切片，将矩阵插入其中
        global_damping[dof::6] = wind_damping
        # 将阻尼放在对角线位置
        damping = np.diag(global_damping)
        return damping

    # calucate the module wind load cofficients
    def compute_submodule_wind_load_coefficients(self, num_submodules=10, split_cols=16):
        """
        Compute submodule wind load coefficients.
        
        Parameters:
        - matrix: numpy.ndarray, load matrix.
        - num_submodules: int, number of submodules to split the matrix into.
        - split_cols: int, number of columns per submodule (excluding boundary columns).
        
        Returns:
        - submatrices: list, containing submodule load coefficient matrices.
        """
        matrix = self.wind_load_coefficient()
        split_cols = split_cols - 1 # 源代码具有16列，输入参数为15，由于python从0开始计数引起
        matrix[:, split_cols::split_cols] /= 2
        boundary_columns = matrix[:, split_cols::split_cols][:, :-1]
        matrix_new = matrix.copy()
        
        for i in range(boundary_columns.shape[1]):
            insert_position = split_cols + split_cols * i + i
            matrix_new = np.insert(matrix_new, insert_position, boundary_columns[:, i], axis=1)
        
        submatrices = np.array_split(matrix_new, num_submodules, axis=1)
        return submatrices
    # Lumped parameter wind load coefficient
    def wind_coefficient_lumped(self):
        """
        Compute and return the lumped parameter wind load coefficient.
        Calculates the lumped parameter wind load coefficient of the modules.
        """
        # Obtain the wind load coefficient matrix
        cc = self.wind_load_coefficient()
        # plt.imshow(cc)
        # plt.colorbar()
        # Only 0 and 180 degrees are considered
        if self.wind_direction != 0:
            cc = np.rot90(cc,2)
        
        c_submodules = self.compute_submodule_wind_load_coefficients() # 处理风载系数矩阵，分割为模块数

        # Sum of the wind load coefficients
        c_sums = np.array([np.sum(matrix) for matrix in c_submodules])

        # 处理风向的影响 
        if self.wind_direction != 0:
            c_sums = c_sums[::-1]
        return c_sums,c_submodules
    
    def wind_force_lumped(self,target_frequency, distance,cd_sums,cl_sums,cl_submodules,dist_matrix=7.5):
        """
        Compute and return the lumped parameter wind force.

        Parameters:
        - target_frequency: The desired frequency for which wind force is computed.
        - distance: The distance between the modules.
        - cd_sums: Array of drag coefficients for each submodule.
        - cl_sums: Array of lift coefficients for each submodule.
        - cl_submodules: List of lift coefficient matrices for each submodule.
        - dist_matrix: Distance matrix for calculating moments.
        """
        def compute_transform_force_my(submodule):
            b1 = np.zeros((31, 16))
            for i in range(8):
                b1[:, i] = submodule[:, i] - submodule[:, 15 - i]
            return 2 * self.adjust_wind_speed() * abs(np.sum(b1)) * dist_matrix * amplitude_complex[0] * self.A * self.rho

        # Calculate phases
        if self.wind_direction == 0:
            delta_theta = target_frequency * distance / self.adjust_wind_speed()
            phases = np.arange(10) * delta_theta
        else:
            delta_theta = target_frequency * distance / self.adjust_wind_speed()
            phases = np.arange(10) * delta_theta * -1

        # Compute amplitude and its complex representation
        amplitude = self.compute_amplitude_for_frequency(target_frequency)
        amplitude_complex = amplitude * (np.cos(phases) + 1j * np.sin(phases))

        # Compute the wind force for each submodule
        wind_force_x = 2 * abs(cd_sums) * self.adjust_wind_speed() * amplitude_complex * self.A * self.rho
        wind_force_z = 2 * abs(cl_sums) * self.adjust_wind_speed() * amplitude_complex * self.A * self.rho
        wind_force_my = 2 * abs(cd_sums) * self.adjust_wind_speed() * amplitude_complex * self.A * self.rho

        #Adjust wind force for My direction based on wind direction
        if self.wind_direction == 0:
            transfor_force_my = compute_transform_force_my(cl_submodules[0])
            wind_force_my[0] += transfor_force_my
        else:
            transfor_force_my = compute_transform_force_my(cl_submodules[0])
            wind_force_my[-1] += transfor_force_my

        # 加入符号
        # wind_force_x *= np.sign(cd_sums)
        # wind_force_z *= np.sign(cl_sums)
        # wind_force_my *= np.sign(cd_sums)
        # Form the fluctuating wind load matrix
        wind_load = np.zeros(50, dtype=complex)
        wind_load[0:50:5] = wind_force_x
        wind_load[2:50:5] = wind_force_z
        wind_load[4:50:5] = wind_force_my
        wind_load = wind_load.reshape(1, 50)

        return wind_load




# Example usage
# wind_load = WindLoad(U10=14.3, z=2, file_path="winddata/Ti0.1degree0.txt")
# y = wind_load.wind_load_coefficient()
# amplitude_spectrum = wind_load.compute_amplitude_from_spectrum()
# spectrum = wind_load.api_spectrum()
# plt.plot(spectrum)
# plt.show()