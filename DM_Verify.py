import matplotlib.pyplot as plt
import xarray as xr
from capytaine.io.xarray import merge_complex_values
from capytaine.post_pro import rao

def verify_hydrodynamic_data(filepath , omega = 0, wave_direction=0.0, dissipation=None, stiffness=None):
    """
    Verify the hydrodynamic data and calculate the rao for a given wave direction.

    Parameters:
    - filepath (str): Path to the .nc file containing the dataset.
    - wave_direction (float, default=0.0): Direction of the wave.
    - dissipation (type?, default=None): Dissipation parameter. Define its type if known.
    - stiffness (type?, default=None): Stiffness parameter. Define its type if known.

    Returns:
    - dataset (xarray.Dataset): Dataset after processing and addition of rao values.
    """

    # Load the dataset from the given file and merge its complex values
    dataset = merge_complex_values(xr.open_dataset(filepath))
    
    # Calculate the rao and add it to the dataset
    dataset['rao'] = rao(dataset, wave_direction=wave_direction, dissipation=dissipation, stiffness=stiffness)
    
    # Plot the rao
    plt.plot(abs(dataset['rao'][omega][0::6]),marker='o')
    plt.xlabel('Frequency Index')  # You can adjust this label if needed
    plt.ylabel('RAO Magnitude')  # You can adjust this label if needed
    plt.title('RAO for Wave Direction {}'.format(wave_direction))
    plt.show()

    return dataset

# # Example usage:
# dataset_result = verify_hydrodynamic_data("BM10_180_direction0.nc")

# 读取实验数据函数,读取由Matlab图像点数据，两列数据
def process_exp_data(filename):
    """
    Read the data from the given filename, process it, and plot the scatter plot.
    
    Parameters:
    - filename (str): The path to the file containing the data.
    
    Returns:
    - tuple: (x_values, y_values) where both are lists of processed data.
    """
    # Read the data from the file
    with open(filename, "r") as file:
        data = file.readlines()
    
    # Parse the data
    x_values = []
    y_values = []
    for line in data:
        x, y = map(float, line.split())
        x_values.append(x)
        y_values.append(y) 
    # Plot the scatter plot
    # plt.figure(figsize=(10, 6))
    # plt.scatter(x_values, y_values, color='blue', marker='o')
    # plt.title("Scatter Plot of Experimental Data")
    # plt.xlabel("X Values")
    # plt.ylabel("Y Values")
    # plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    # plt.show()
    # example
    # x, y = process_exp_data("data\Experiment_300_60\exp_300.txt")
    # y = np.array(y)
    return x_values, y_values
