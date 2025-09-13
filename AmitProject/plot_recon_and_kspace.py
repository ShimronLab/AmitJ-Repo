import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_log_kspace_and_recon(output_dir,seq_root,file_path,n_echo,pe_order_label,direction,shift=False):
    """
    Loads 2D k-space data from a .pt file, calculates its log magnitude,
    and plots it next to the reconstruction use IFFT.
    """
    try:
        kspace_tensor = torch.load(file_path)

        kspace_complex = kspace_tensor.to(torch.complex64)
        kspace_array = kspace_complex.numpy()
        magnitude = np.abs(kspace_array).astype(np.float32)

        # Apply a logarithmic scale to the magnitude for better visualization
        log_magnitude = np.log1p(magnitude)

        reco = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(kspace_complex)))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Make panels tight to each other
        fig.subplots_adjust(wspace=-0.7)  # top leaves room for titles

        # Left: recon image (swapped order)
        im0 = axes[0].imshow(reco.abs().numpy(), cmap="gray", origin="lower")
        axes[0].axis("off")

        # Right: k-space log magnitude (swapped order)
        im1 = axes[1].imshow(log_magnitude, cmap="gray", origin="lower")
        axes[1].axis("off")

        # Global title + smaller subtitle stacked (no overlap with axes titles)
        title = "Recon image and log magnitude of k-space"
        subtitle = f"ETL: {n_echo}   |   pe_order: {pe_order_label}   |   direction: {direction} {' |  shifted' if shift else ''}"

        fig.suptitle(title, fontsize=18, y=0.96)
        fig.text(0.5, 0.9, subtitle, ha="center", va="top", fontsize=14, color="0.2")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        filename = f'reconstruction_and_kspace_{seq_root}.png'
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path,dpi=300, bbox_inches='tight')

        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example Usage:
if __name__ == '__main__':
    plot_log_kspace_and_recon('/home/amit.ja/AmitProject/sim_data/test_tensor/test_tensor_ksp/ksp_subject04_slice_49.pt',n_echo=32,pe_order_label='CO',direction='horizontal')