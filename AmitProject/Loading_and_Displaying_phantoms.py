import MRzeroCore as mr0
import numpy as np
import matplotlib.pyplot as plt
# Load Phantom Maps
filepath = "output\\brainweb\\\subject04_3T.npz"
data = np.load(filepath)
print(data.files)

# Access each map
PD_map = data['PD_map']
print(PD_map.shape) #as an example

T1_map = data['T1_map']
T2_map = data['T2_map']
T2dash_map = data['T2dash_map']
D_map = data['D_map']
tissue_gm = data['tissue_gm']
tissue_wm = data['tissue_wm']
tissue_csf = data['tissue_csf']

# Select a slice along the Z-axis
slice_index = 59

# Create a figure with subplots (3 rows, 3 columns for the maps)
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

# Flatten the 3x3 grid to iterate through it
axs = axs.ravel()

# Plot PD_map
cax1 = axs[0].imshow(PD_map[:, :, slice_index], cmap='viridis')
axs[0].set_title('PD_map (Slice {})'.format(slice_index + 1), pad=5)
fig.colorbar(cax1, ax=axs[0])

# Plot T1_map
cax2 = axs[1].imshow(T1_map[:, :, slice_index], cmap='viridis')
axs[1].set_title('T1_map (Slice {})'.format(slice_index + 1), pad=5)
fig.colorbar(cax2, ax=axs[1])

# Plot T2_map
cax3 = axs[2].imshow(T2_map[:, :, slice_index], cmap='viridis')
axs[2].set_title('T2_map (Slice {})'.format(slice_index + 1), pad=5)
fig.colorbar(cax3, ax=axs[2])

# Plot T2dash_map
cax4 = axs[3].imshow(T2dash_map[:, :, slice_index], cmap='viridis')
axs[3].set_title('T2dash_map (Slice {})'.format(slice_index + 1), pad=5)
fig.colorbar(cax4, ax=axs[3])

# Plot D_map
cax5 = axs[4].imshow(D_map[:, :, slice_index], cmap='viridis')
axs[4].set_title('D_map (Slice {})'.format(slice_index + 1), pad=5)
fig.colorbar(cax5, ax=axs[4])

# Plot tissue_gm
cax6 = axs[6].imshow(tissue_gm[:, :, slice_index], cmap='viridis')
axs[6].set_title('Tissue GM (Slice {})'.format(slice_index + 1), pad=5)
fig.colorbar(cax6, ax=axs[6])

# Plot tissue_wm
cax7 = axs[7].imshow(tissue_wm[:, :, slice_index], cmap='viridis')
axs[7].set_title('Tissue WM (Slice {})'.format(slice_index + 1), pad=5)
fig.colorbar(cax7, ax=axs[7])

# Plot tissue_csf
cax8 = axs[8].imshow(tissue_csf[:, :, slice_index], cmap='viridis')
axs[8].set_title('Tissue CSF (Slice {})'.format(slice_index + 1), pad=5)
fig.colorbar(cax8, ax=axs[8])

# Hide the empty subplot (5th one)
axs[5].axis('off')

# Adjust layout
plt.tight_layout(pad=5.0)
plt.subplots_adjust(top=0.93, bottom=0.07, left=0.05, right=0.95)

plt.show()

# Save the figure
fig.savefig(f'Figure1-All Phantom Maps for slice {slice_index + 1}.png')

# Choose slice from PD_map
phantom_slice = PD_map[:, :, slice_index]  # Extract slice 60

# Perform FFT and shift the zero frequency to the center
fft_slice = np.fft.fft2(phantom_slice)        # 2D FFT
fft_shifted = np.fft.fftshift(fft_slice)      # Shift zero frequency to the center
magnitude_spectrum = np.abs(fft_shifted)     # Compute magnitude of the k-space

# Create figure with 2 subplots (phantom and k-space)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the Phantom Slice
cax1 = axs[0].imshow(phantom_slice, cmap='gray')
axs[0].set_title(f'Phantom PD map Slice {slice_index + 1} (Spatial Domain)')
fig.colorbar(cax1, ax=axs[0])

# Plot the K-space (magnitude spectrum)
cax2 = axs[1].imshow(np.log(1 + magnitude_spectrum), cmap='gray')  # Log scale for better visibility
axs[1].set_title(f'K-space of PD map Slice {slice_index + 1} (Frequency Domain)')
fig.colorbar(cax2, ax=axs[1])

# Adjust layout
plt.tight_layout()
plt.show()

# Save the figure
fig.savefig(f'Figure2-Phantom and its kspace based on PD Map for slice {slice_index + 1}.png')


