import numpy as np
import torch
import matplotlib.pyplot as plt
import MRzeroCore as mr0
import math
import matplotlib.animation as animation

# Load your phantom data
phantom = mr0.VoxelGridPhantom.brainweb("output\\brainweb\\subject05_3T.npz")
phantom = phantom.interpolate(128, 128, 32).slices([16])
# T2_map = phantom.T2
# num_unique = torch.unique(T2_map).numel()
# print(num_unique)
# data = phantom.build()
# Load the sequence
seq = mr0.Sequence.import_file("tse_seq_horizontal_necho16_128.seq")
seq.plot_kspace_trajectory(figsize=(10, 8))

# Compute and execute the simulation graph
graph = mr0.compute_graph(seq, data, 200, 1e-3)
signal = mr0.execute_graph(graph, seq, data)

fov = 256e-3
Ny, Nx = 128, 128
n_echo = 16
n_ex = math.floor(Ny / n_echo)

# signal = (4096,) = 64 (Nx) × 64 (Ny) total samples
signal = signal.view(n_ex, n_echo, Nx)

# Reconstruct pe_order exactly as in TSE_seq.py
pe_steps = np.arange(1, n_echo * n_ex + 1) - 0.5 * n_echo * n_ex - 1
if divmod(n_echo, 2)[1] == 0:
    pe_steps = np.roll(pe_steps, [0, int(-np.round(n_ex / 2))])
pe_order = pe_steps.reshape((n_ex, n_echo), order='F').T

delta_k = 1 / fov
phase_areas = pe_order * delta_k
#
# # Allocate k-space buffer
kspace = torch.zeros(Ny, Nx, dtype=signal.dtype)
#
# Fill in k-space line-by-line
for ex in range(n_ex):
    for e in range(n_echo):
        ky_f = phase_areas[e, ex] # Physical phase encoding value
        ky = int(round(ky_f / delta_k)) + Ny // 2 # Convert to array index
        if 0 <= ky < Ny:
            #  Each (ex, e) corresponds to one k-space line at a certain ky-position and by : we take all the values in that line
            kspace[ky, :] = signal[ex, e, :]

# acquisition_steps = []
# for ex in range(n_ex):
#     for e in range(n_echo):
#         ky_f = phase_areas[e, ex]
#         ky = int(round(ky_f / delta_k)) + Ny // 2  # ky index
#         kx = np.arange(Nx)  # kx indices (0 to 63) in order
#         acquisition_steps.append((ex, e, ky, kx))
#
# # Create a figure
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.set_xlim(0, Nx-1)
# ax.set_ylim(0, Ny-1)
# ax.invert_yaxis()  # Match image convention
# ax.set_title('K-space Acquisition Movement')
# ax.set_xlabel('kx')
# ax.set_ylabel('ky')
#
# colors = plt.cm.jet(np.linspace(0, 1, n_ex))  # Color per excitation
#
# # Store artists
# frames = []
#
# # Blank k-space
# kspace_mask = np.zeros((Ny, Nx))
#
# for idx, (ex, e, ky, kx_arr) in enumerate(acquisition_steps):
#     fig_frame, = ax.plot([], [], 'o', color=colors[ex], markersize=2)
#
#     # Fill one line
#     for kx in kx_arr:
#         kspace_mask[ky, kx] = 1
#
#     # Plot current status
#     ys, xs = np.where(kspace_mask > 0)
#     fig_frame.set_data(xs, ys)
#     frames.append([fig_frame])
#
# # Create animation
# ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True)
#
# # Save as gif
# ani.save('kspace_acquisition.gif', writer='pillow')
#
# plt.close()
# print("GIF saved as 'kspace_acquisition.gif'")
#
# Reconstruct image
reco = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(kspace)))

plt.figure()
plt.imshow(reco.abs(), cmap="gray")
plt.title("Image Reconstruction")
plt.axis("off")
plt.show()


# Track TE per ky line
# TE_init = 12e-3
# echo_spacing = 12e-3
# TEs = np.array([TE_init + i * echo_spacing for i in range(n_echo)])
# TE_per_ky = np.zeros(Ny)
# counts = np.zeros(Ny)

# TE_per_ky = TE_per_ky / np.maximum(counts, 1)
# T2_map = phantom.T2
#
# # Fill in k-space line-by-line
# for ex in range(n_ex):
#     for e in range(n_echo):
#         ky_f = phase_areas[e, ex]
#         ky = int(round(ky_f / delta_k)) + Ny // 2
#         if 0 <= ky < Ny:
#             kspace[ky, :] = signal[ex, e, :]
#             TE_per_ky[ky] += TEs[e]
#             counts[ky] += 1
#
# TE_per_ky = torch.tensor(TE_per_ky, dtype=T2_map.dtype, device=T2_map.device)
# T2_map = T2_map.squeeze(-1)
#
# decay = torch.exp(-TE_per_ky[:, None] / T2_map)
# decay = torch.clamp(decay, min=0.1, max=5)
# kspace = kspace / decay

# plt.imshow(decay.cpu().numpy(), cmap='hot')
# plt.title("Decay Compensation Map")
# plt.colorbar()
# plt.show()

# # Visualize acquisition pattern
# acquisition_steps = []
# for ex in range(n_ex):
#     for e in range(n_echo):
#         ky_f = phase_areas[e, ex]
#         ky = int(round(ky_f / delta_k)) + Ny // 2
#         kx = np.arange(Nx)
#         acquisition_steps.append((ex, e, ky, kx))
#
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.set_xlim(0, Nx-1)
# ax.set_ylim(0, Ny-1)
# ax.invert_yaxis()
# ax.set_title('K-space Acquisition Movement')
# ax.set_xlabel('kx')
# ax.set_ylabel('ky')
# colors = plt.cm.jet(np.linspace(0, 1, n_ex))
# frames = []
# kspace_mask = np.zeros((Ny, Nx))
#
# for idx, (ex, e, ky, kx_arr) in enumerate(acquisition_steps):
#     fig_frame, = ax.plot([], [], 'o', color=colors[ex], markersize=2)
#     for kx in kx_arr:
#         kspace_mask[ky, kx] = 1
#     ys, xs = np.where(kspace_mask > 0)
#     fig_frame.set_data(xs, ys)
#     frames.append([fig_frame])
#
# ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True)
# ani.save('kspace_acquisition.gif', writer='pillow')
# plt.close()
# print("GIF saved as 'kspace_acquisition.gif'")

# # Reconstruct image
# reco = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(kspace)))
#
# plt.figure()
# plt.imshow(reco.abs(), cmap="gray")
# plt.title("Image Reconstruction")
# plt.axis("off")
# plt.show()
