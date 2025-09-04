import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_log_kspace_image(file_path):
    """
    Loads 2D k-space data from a .pt file, calculates its log magnitude,
    and plots it as an image with a colorbar, suitable for MRI k-space visualization.

    Args:
        file_path (str): The path to the .pt file containing the k-space tensor.
                         Assumes the tensor is complex and 2D (e.g., [rows, cols]).
    """
    try:
        # Load the tensor from the .pt file
        kspace_tensor = torch.load(file_path)

        # Ensure the tensor is complex (if it's not already, might need more specific handling)
        # MRI k-space data is typically complex.
        if not kspace_tensor.is_complex():
            print("Warning: Loaded tensor is not complex. Assuming it represents real k-space magnitude directly.")
            kspace_complex = kspace_tensor.to(torch.float32) # Ensure float type for magnitude
        else:
            kspace_complex = kspace_tensor.to(torch.complex64) # Ensure complex type

        # Convert to NumPy array
        kspace_array = kspace_complex.numpy()

        # Calculate the magnitude of the complex k-space data
        magnitude = np.abs(kspace_array)

        # Apply a logarithmic scale to the magnitude for better visualization
        # Adding a small epsilon to avoid log(0) if any magnitude is zero.
        log_magnitude = np.log1p(magnitude)

        # Shift the zero-frequency component (DC component) to the center for better visualization
        # This is a common step in k-space visualization if the data isn't already centered.
        #log_magnitude_shifted = np.fft.fftshift(log_magnitude)

        # Plot the log magnitude as an image
        plt.figure(figsize=(8, 8))
        # Use 'gray' colormap as in your example, or 'viridis', 'jet', etc., for different visual.
        # 'extent' can be used to label axes in physical units if known.
        plt.imshow(log_magnitude, cmap='gray', origin='lower')
        plt.title('Log K-space Magnitude Display')
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example Usage:
if __name__ == '__main__':
    plot_log_kspace_image('/home/amit.ja/AmitProject/sim_data/test_tensor/test_tensor_ksp/ksp_subject04_slice_49.pt')