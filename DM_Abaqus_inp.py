import numpy as np
"""
***This script generates an Abaqus INP file with boundary conditions for a given displacement matrix, including both real and imaginary parts.
It supports choosing the DOFs to apply the boundary conditions.
***For more information, please see main_applied_boundary_conditions, which processes the boundary conditions for both real and imaginary parts.
The function modify_inp_file is used to modify the DOFs information [1\2\4]. The results show that the idea is not useful in compute internal force.
"""


def generate_random_displacements(num_nodes, degrees_of_freedom):
    """
    Generate a random displacement matrix for specified number of nodes and degrees of freedom.
    
    Parameters:
    - num_nodes: int, number of nodes
    - degrees_of_freedom: int, degrees of freedom per node
    
    Returns:
    - numpy.ndarray, displacement matrix
    """
    return np.random.rand(num_nodes, degrees_of_freedom)

def create_node_set_definitions(node_ids):
    """
    Create node set definitions for each node.
    
    Parameters:
    - node_ids: list of int, node identifiers
    
    Returns:
    - list of str, lines defining node sets
    """
    node_set_definitions = ['** Node Sets Definitions\n']
    for node_id in node_ids:
        set_name = f'Set-{node_id}'
        node_set_definitions.append(f'*Nset, nset={set_name}, instance=Part-1-1\n')
        node_set_definitions.append(f' {node_id},\n')
    return node_set_definitions

def create_boundary_conditions(node_ids, displacement_matrix):
    """
    Create boundary conditions for each node set with all degrees of freedom in a single block.
    
    Parameters:
    - node_ids: list of int, node identifiers
    - displacement_matrix: numpy.ndarray, matrix of displacement values
    
    Returns:
    - list of str, lines defining boundary conditions
    """
    boundary_conditions = ['** Boundary Conditions\n']
    for i, node_id in enumerate(node_ids):
        set_name = f'Set-{node_id}'
        displacements = displacement_matrix[i]
        boundary_conditions.append(f'*Boundary\n')
        for dof, disp in enumerate(displacements, start=1):
            boundary_conditions.append(f' {set_name}, {dof}, {dof}, {disp}\n')
    return boundary_conditions

def create_boundary_conditions_real(node_ids, displacement_matrix):
    """
    Create boundary conditions for each node set with all degrees of freedom in a single block.
    
    Parameters:
    - node_ids: list of int, node identifiers
    - displacement_matrix: numpy.ndarray, matrix of displacement values
    
    Returns:
    - list of str, lines defining boundary conditions
    """
    boundary_conditions = ['** Boundary Conditions\n']
    for i, node_id in enumerate(node_ids):
        set_name = f'Set-{node_id}'
        displacements = displacement_matrix[i]
        boundary_conditions.append(f'*Boundary, real\n')
        for dof, disp in enumerate(displacements, start=1):
            boundary_conditions.append(f' {set_name}, {dof}, {dof}, {disp}\n')
    return boundary_conditions

def create_boundary_conditions_imag(node_ids, displacement_matrix):
    """
    Create boundary conditions for each node set with all degrees of freedom in a single block.
    
    Parameters:
    - node_ids: list of int, node identifiers
    - displacement_matrix: numpy.ndarray, matrix of displacement values
    
    Returns:
    - list of str, lines defining boundary conditions
    """
    boundary_conditions = ['** Boundary Conditions\n']
    for i, node_id in enumerate(node_ids):
        set_name = f'Set-{node_id}'
        displacements = displacement_matrix[i]
        boundary_conditions.append(f'*Boundary, imaginary\n')
        for dof, disp in enumerate(displacements, start=1):
            boundary_conditions.append(f' {set_name}, {dof}, {dof}, {disp}\n')
    return boundary_conditions

def write_inp_file(file_path, lines):
    """
    Write lines to a file.
    
    Parameters:
    - file_path: str, path to the output file
    - lines: list of str, content to write to the file
    """
    with open(file_path, 'w') as file:
        file.writelines(lines)

def main_appied_boundary_conditions(num_nodes, degrees_of_freedom, result_expanded, output_file_path):
    node_ids = list(range(1, num_nodes + 1))
    displacement_matrix = result_expanded.reshape(num_nodes, degrees_of_freedom)

    node_set_definitions = create_node_set_definitions(node_ids)
    boundary_conditions_real = create_boundary_conditions_real(node_ids, displacement_matrix.real)
    boundary_conditions_imag = create_boundary_conditions_imag(node_ids, displacement_matrix.imag)

    # Combine all parts into the final INP file content
    final_inp_lines = node_set_definitions + ['**\n'] + boundary_conditions_real + ['**imag\n'] + boundary_conditions_imag

    # Write to the output file
    write_inp_file(output_file_path, final_inp_lines)

    print(f"Boundary conditions INP file written to {output_file_path}")


def main_appied_boundary_conditions_static(num_nodes, degrees_of_freedom, result_expanded, output_file_path):
    node_ids = list(range(1, num_nodes + 1))
    displacement_matrix = result_expanded.reshape(num_nodes, degrees_of_freedom)

    node_set_definitions = create_node_set_definitions(node_ids)
    boundary_conditions = create_boundary_conditions(node_ids, displacement_matrix)

    # Combine all parts into the final INP file content
    final_inp_lines = node_set_definitions + ['**\n'] + boundary_conditions

    # Write to the output file
    write_inp_file(output_file_path, final_inp_lines)

    print(f"Boundary conditions INP file written to {output_file_path}")


# # Example usage
# if __name__ == "__main__":
#     num_nodes = 793
#     degrees_of_freedom = 5
#     result_expanded = your_result_expanded_variable  # replace this with your displacenmet matrix
#     output_file_path = 'Boundary_Conditions_Job-1.inp'
    
#     main(num_nodes, degrees_of_freedom, result_expanded, output_file_path)

# 删除部分自由度，保留3-5自由度
def modify_inp_file(input_file_path, output_file_path):
    import re

    # Load the content of the INP file
    with open(input_file_path, 'r') as file:
        inp_content = file.readlines()

    # Function to modify boundary conditions
    def modify_boundary_conditions(lines):
        modified_lines = []
        for line in lines:
            if line.startswith('*Boundary'):
                modified_lines.append(line)
                continue
            if re.match(r'^\s*\S+\s*,\s*[1245]\s*,', line):  # Match lines specifying DOF 1, 2, or 4 , if you want to keep 3, change to [1245]
                continue
            modified_lines.append(line)
        return modified_lines

    # Apply the function to modify the boundary conditions
    modified_inp_content = modify_boundary_conditions(inp_content)

    # Save the modified content to a new file
    with open(output_file_path, 'w') as file:
        file.writelines(modified_inp_content)

# # Example usage
# input_file_path = 'E:\phd\Code\DM-FEM2D\FEM_Reduce\Boundary_Conditions_Job-1.inp'
# output_file_path = 'Boundary_Conditions_Job-1-modify.inp'
# modify_inp_file(input_file_path, output_file_path)
