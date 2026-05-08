# imports
import numpy as np
import logging
import xarray as xr
import sys
from capytaine.io.xarray import merge_complex_values
import os
import matplotlib.pyplot as plt
import matplotlib

# import user-defined modules
import wave_spectrum as ws
import DM_Windload as DM_Wind
import DM_Reading as dm_r
import DM_Assemble as DM_A
import SEREP

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# # Set the working directory if necessary
# os.chdir(r'E:\phd\Code\DM-FEM2D')

# configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)-8s: %(message)s')

def main(Hs, Tp, wind_speed, wind_direction, num_nodes):
    logging.info("Starting simulation...")
    # setting initial parameter, reading mass and stiffness matrix
    master_nodes = DM_A.calculate_node_positions(424,6,10)
                                                            #33mesh : DM_A.calculate_node_positions(1106,10,10)
                                                            #55mesh : DM_A.calculate_node_positions(424,6,10)
    # Load hydrodynamic data
    dataset_path = r"E:\phd\Code\DM-FEM2D\HydrodynamicData\wind_stduy\BM10_direaction0_full200.nc" # 定义水动力文件
    # Define file paths for the structural data
    file_m = 'E:\phd\Code\DM-FEM2D\StructureData\Job-1_55_MASS1.mtx'
    file_k = 'E:\phd\Code\DM-FEM2D\StructureData\Job-1_55_STIF1.mtx'

    # Load the hydrodynamic data
    dataset = merge_complex_values(xr.open_dataset(dataset_path))
    omega = dataset.omega.values

    # Preprocess structural matrices
    M, K  = preprocess_structural_matrices(file_m,file_k,num_nodes, master_nodes)
    MasterDofs, SlaveDofs = SEREP.separate_dofs(num_nodes, master_nodes)
    K = K + create_mooring_stiffness_matrix(1e5)
    MR_serep,KR_serep,T_serep = SEREP.SEREP(K, M, SlaveDofs, master_nodes)
    MR,KR,T_static = SEREP.static_condensation(K, M, MasterDofs, SlaveDofs)
    # Load wave spectrum
    S_wave = ws.jonswap(Hs, Tp, omega)

    # Load wind loads
    Wind_Damping, wind_coefficients, windload_cd = load_wind_loads(wind_speed, wind_direction)
    # Simulation for each omega
    results = run_simulation(dataset, S_wave,
                             MR, KR*0.01, T_static, 
                             Wind_Damping, wind_coefficients, windload_cd,
                             master_nodes)

    # plot_motion_response_spectrum(omega, results)
    # plot_rms_contour_map(results)
    # print(results)
    # Save results
    save_results(results,hs,wind_speed)
    logging.info("Simulation completed.")

def preprocess_structural_matrices(file_m, file_k, num_nodes, master_nodes):

    # Read the structural mass and stiffness matrices from files
    M = dm_r.get_stiffness_matrix(file_m)
    k = dm_r.get_stiffness_matrix(file_k)

    # reduce dofs
    M_consistant= SEREP.reduce_dofs(M, num_nodes, [5])
    k = SEREP.reduce_dofs(k, num_nodes, [5])

    # transform mass matrix, beta=0 is consistant mass matrix
    M = SEREP.transform_mass_matrix(M_consistant,beta=0)

    # reduce matrix use SEREP
    # obtaine master dofs and slave dofs

    return M, k

def load_wind_loads(wind_speed, wind_direction):
    windload_cd_file = f"E:/phd/Code/DM-FEM2D/winddata/Ti0.1_cd_degree{wind_direction}.txt"
    windload_cl_file = f"E:/phd/Code/DM-FEM2D/winddata/Ti0.1_cl_degree{wind_direction}.txt"
    windload_cd = DM_Wind.WindLoad(U10=wind_speed, z=2, total_rows=31, total_cols=151, file_path=windload_cd_file, wind_direction=wind_direction)
    windload_cl = DM_Wind.WindLoad(U10=wind_speed, z=2, total_rows=31, total_cols=151, file_path=windload_cl_file, wind_direction=wind_direction)
    windload_cd_damping = DM_Wind.WindLoad(U10=wind_speed, z=2, total_rows=13, total_cols=61, file_path=windload_cd_file,wind_direction = wind_direction)
    windload_cl_damping = DM_Wind.WindLoad(U10=wind_speed, z=2, total_rows=13, total_cols=61, file_path=windload_cl_file,wind_direction = wind_direction)
    # 计算风产生的阻尼效应feng
    Wind_Damping_reduced = (windload_cd_damping.compute_wind_damping(dof=0) + 
                            windload_cl_damping.compute_wind_damping(dof=2) + 
                            windload_cd_damping.compute_wind_damping(dof=4))*5.9 # 5.9是转化系数多节点到少节点
    Wind_Damping = SEREP.reduce_dofs(Wind_Damping_reduced,num_nodes,[5])
    # 形成集中风载荷
    cd_sums,cd_submodules = windload_cd.wind_coefficient_lumped()
    cl_sums,cl_submodules = windload_cl.wind_coefficient_lumped()
    # Package the coefficients in a dictionary
    wind_coefficients = {
        'cd_sums': cd_sums,
        'cd_submodules': cd_submodules,
        'cl_sums': cl_sums,
        'cl_submodules': cl_submodules
    }
    # Return the reduced wind damping matrix and the dictionary of coefficients
    return Wind_Damping, wind_coefficients, windload_cd

def create_mooring_stiffness_matrix(stiffness_value):
    # 创建一个总的刚度矩阵，维度为 (total_nodes*6, total_nodes*6)
    total_nodes = 793
    total_dof = total_nodes * 5
    global_stiffness_matrix = np.zeros((total_dof, total_dof))
    
    # 确定四个角点的节点编号
    node1 = 1
    node2 = 61
    node3 = 733
    node4 = 793
    
    # 四个系泊点的节点列表
    mooring_nodes = [node1, node2, node3, node4]
    
    # 定义6x6的局部刚度矩阵
    local_stiffness_matrix = np.zeros((5, 5))
    local_stiffness_matrix[0, 0] = stiffness_value  # x方向刚度
    local_stiffness_matrix[1, 1] = stiffness_value  # y方向刚度
    
    # 将局部刚度矩阵插入到总的刚度矩阵中
    for node in mooring_nodes:
        start_index = (node - 1) * 5
        for i in range(5):  # 只处理前5个自由度
            for j in range(5):
                global_stiffness_matrix[start_index + i, start_index + j] = local_stiffness_matrix[i, j]
    
    return global_stiffness_matrix

def run_simulation(dataset, S_wave, M_reduce, K_reduce, T, Wind_Damping, wind_coefficients, windload_cd, master_nodes):
    # Initialize a list to store the results for each frequency1
    results = []
    omega = dataset.omega.values
    omega_number = np.arange(0, len(omega), 1)
    # obtaine master dofs and slave dofs
    MasterDofs, SlaveDofs = SEREP.separate_dofs(num_nodes, master_nodes)
    master_nodes_length = len(master_nodes)
    distance = 30 # Distance from the wind force application point to the center of the structure
    # Loop over each frequency in the omega_number list
    for i in omega_number:
        logging.info(f"Processing frequency {i+1}/{len(omega_number)} with omega = {dataset.omega.values[i]:.2f}")

        # Load hydrodynamic data for the current frequency
        added_mass = dataset['added_mass'][i].values
        radiation_damping = dataset['radiation_damping'][i].values
        Froude_Krylov_force = dataset['Froude_Krylov_force'][i].values
        diffraction_force = dataset['diffraction_force'][i].values
        hydrostatic_stiffness = dataset['hydrostatic_stiffness'].values

        # Apply the wave spectrum to scale the Froude-Krylov and diffraction forces
        F_wave = (Froude_Krylov_force + diffraction_force) * np.sqrt(S_wave[i]*0.01)

        # reduce dofs [5] means reduce 6th dof
        added_mass = SEREP.reduce_dofs(added_mass,master_nodes_length,[5])
        radiation_damping = SEREP.reduce_dofs(radiation_damping,master_nodes_length,[5])
        hydrostatic_stiffness = SEREP.reduce_dofs(hydrostatic_stiffness,master_nodes_length,[5])
        F_wave = SEREP.reduce_force_matrix_dofs(F_wave, master_nodes_length, 5).reshape(1,5*master_nodes_length)
        
        # Calculate wind forces for the current frequency
        wind_forces = windload_cd.wind_force_lumped(target_frequency=omega[i], distance=distance,
                                                    cd_sums=wind_coefficients['cd_sums'],
                                                    cl_sums=wind_coefficients['cl_sums'],
                                                    cl_submodules=wind_coefficients['cl_submodules'])
        # Combine wave and wind forces
        total_forces = F_wave + wind_forces


        # Calculate the effective mass, damping, and stiffness at this frequency
        effective_mass = added_mass + M_reduce
        effective_damping = radiation_damping + T.T@Wind_Damping@T
        effective_stiffness = hydrostatic_stiffness + K_reduce

        # Calculate the response for the current frequency
        master_displacement_wave = DM_A.solve_frequency_domain(effective_mass, effective_damping, 
                                                          effective_stiffness, F_wave, omega[i])
        master_displacement_wind = DM_A.solve_frequency_domain(effective_mass, effective_damping,
                                                            effective_stiffness, total_forces, omega[i])
    
        # Restore global displacement under disorder masterdofs and slavedofs
        global_displacement_disorder_wave = T @ master_displacement_wave
        global_displacement_disorder_wind = T @ master_displacement_wind
        # Reorder global displacement under order
        global_displacement_wave = SEREP.reorder_displacement_matrix(global_displacement_disorder_wave, MasterDofs, 
                                                                SlaveDofs)
        global_displacement_wind = SEREP.reorder_displacement_matrix(global_displacement_disorder_wind, MasterDofs,
                                                                SlaveDofs)
        # Append the results to the list
        # Store the computed displacement and other relevant data for this frequency
        # results.append({
        #     'frequency': omega[i],
        #     'displacement': global_displacement,
        #     'F_wave': F_wave,
        #     'wind_forces': wind_forces,
        # })
        results.append(abs(global_displacement_wave[0::5,:])+abs(global_displacement_wind[0::5,:]))
    displacement = np.array(results).reshape(199,793) # 793 2121
    print(displacement.shape)
    return displacement


def save_results(results,Hs,wind_speed):
    # Constructing the filename with wave height and wind speed
    filename = f"E:\phd\Code\DM-FEM2D\FEM_Reduce\windandwave2_paper\Surge_displacement_Hs{Hs}_Wind{wind_speed}.npy"
    # Assuming 'results' is a numpy array
    np.save(filename, results)
    print(f"Results saved to {filename}")

# Setting a style for scientific papers
matplotlib.rcParams.update({'font.size': 12, 'font.family': 'serif', 'figure.figsize': (8, 6)})

def plot_motion_response_spectrum(omega, displacement):
    mean_displacement = np.mean(np.abs(displacement), axis=1)
    response_spectrum = mean_displacement**2 / 0.01

    plt.figure()
    plt.plot(omega, response_spectrum, label='Motion Response Spectrum', linewidth=2)
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Response (units^2)')
    plt.title('Motion Response Spectrum')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Adjust layout to not cut off elements
    #plt.savefig('MotionResponseSpectrum.pdf')  # Save as PDF for high-quality publication
    plt.show()

# plot_motion_response_spectrum(omega, displacement)  # Uncomment for use
def plot_rms_contour_map(displacement, shape=(13, 61)):
    wind_displacement_df = np.abs(displacement) * 0.01
    wind_displacement_sum = np.sum(wind_displacement_df, axis=0)
    wind_displacement_rms = np.sqrt(wind_displacement_sum).reshape(shape)

    plt.figure()
    c = plt.imshow(wind_displacement_rms, cmap='viridis')
    plt.colorbar(c)
    plt.title('RMS Displacement Contour Map')
    plt.xlabel('Node Dimension 1')
    plt.ylabel('Node Dimension 2')
    plt.grid(False)
    plt.tight_layout()
    #plt.savefig('RMSContourMap.pdf')  # Save as PDF for high-quality publication
    plt.show()

if __name__ == "__main__":
    # Hs = 1.75  # Significant wave height
    # Tp = 6.59  # Peak period
    wind_speeds = np.arange(5, 45, 5)  # Wind speeds in m/s
    wind_speed = 30  # Wind speed in m/s
    wind_direction = 0  # Wind direction in degrees
    num_nodes = 793  # Number of nodes 2121 for 33mesh, 793 for 55mesh
    Hs = np.array([1., 2., 3., 4., 5., 6.])
    Tp = np.array([4.57, 6.46, 7.91, 9.13, 10.21, 11.18])
    for wind_speed in wind_speeds:
        for hs, tp in zip(Hs, Tp):
            main(hs, tp, wind_speed, wind_direction, num_nodes)
            
