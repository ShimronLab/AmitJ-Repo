
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import MRzeroCore as mr0
import math
from TSE_seq_applied_to_PDG_sim import TSE_seq

from t2decay import T2_map

output_dir = "tse_sim_outputs"
os.makedirs(output_dir, exist_ok=True)

n_echo_values = [2, 4, 8, 16, 32, 64]
fov = 256e-3
Ny, Nx = 128, 128
# Load phantom
phantom = mr0.VoxelGridPhantom.brainweb("output/brainweb/subject05_3T.npz")
phantom = phantom.interpolate(128, 128, 120).slices([60])
data = phantom.build() #check
recon_images = []
T2_map = phantom.T2.squeeze()
TE_init = 12e-3
TE_step = 12e-3

for n_echo in n_echo_values:
    n_ex = Ny // n_echo

    # Load corresponding sequence
    seq_filename = f"TSE_seq_{n_echo}_128.seq"
    if not os.path.exists(seq_filename):
        TSE_seq(plot=False, write_seq=True, seq_filename=seq_filename, n_echo=n_echo, Ny = Ny, Nx = Nx)
    seq = mr0.Sequence.import_file(seq_filename)
    # seq.grad_m =

    # Plot and save k-space trajectory
    plt.figure(figsize=(6, 6))
    seq.plot_kspace_trajectory()
    traj_path = os.path.join(output_dir, f"kspace_traj_{n_echo}.png")
    plt.savefig(traj_path)
    plt.close()

    # Simulate
    graph = mr0.compute_graph(seq, data, 200, 1e-3)
    signal = mr0.execute_graph(graph, seq, data)

    # signal = signal.view(n_echo, n_ex, Nx)
    signal = signal.view(n_ex, n_echo, Nx)

    # Build pe_order
    pe_steps = np.arange(1, n_echo * n_ex + 1) - 0.5 * n_echo * n_ex - 1
    if divmod(n_echo, 2)[1] == 0:
        pe_steps = np.roll(pe_steps, [0, int(-np.round(n_ex / 2))])
    pe_order = pe_steps.reshape((n_ex, n_echo), order='F').T

    delta_k = 1 / fov
    phase_areas = pe_order * delta_k

    # Fill k-space
    kspace = torch.zeros(Ny, Nx, dtype=signal.dtype)

    for ex in range(n_ex):
        for e in range(n_echo):
            ky_f = phase_areas[e, ex]
            ky = int(round(ky_f / delta_k)) + Ny // 2
            if 0 <= ky < Ny:
                t = TE_init + e * TE_step
                decay_factor = torch.exp(-t / T2_map[ky, :])
                kspace[ky, :] = signal[ex, e, :] * decay_factor
                # kspace[ky, :] = signal[ex, e, :] # Earlier no decay

    # Reconstruct
    reco = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(kspace)))

    # Save image
    fig_img = plt.figure()
    plt.imshow(reco.abs().numpy(), cmap="gray")
    title = f"Reconstructed image for n_echo = {n_echo}"
    plt.title(title)
    plt.axis("off")
    img_path = os.path.join(output_dir, f"{title}.png")
    fig_img.savefig(img_path)
    plt.close(fig_img)

    recon_images.append((n_echo, reco.abs().numpy()))

# Plot all recon images
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for idx, (n_echo, img) in enumerate(recon_images):
    ax = axes[idx // 3, idx % 3]
    ax.imshow(img, cmap="gray")
    ax.set_title(f"n_echo = {n_echo}")
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "recon_grid_summary.png"))
plt.show()
