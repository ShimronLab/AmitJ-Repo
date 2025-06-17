import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import MRzeroCore as mr0
from TSE_seq_applied_to_PDG_sim import TSE_seq


def PDG_sim(filepath, Nx, Ny, TE, n_echo, fov, output_dir, seq_filename, plot_kspace_traj=False,pe_order_label='TD',is_horizontal_pe = False):
    # Inputs:
    # Brain Phantom simData type (a single slice) ready for PDG simulation: data
    # Phantom matrix shape: Nx,Ny
    # Echo time: TE
    # fov: field of view
    # output_dir - an output directory for the reconstruction image and kspace - make the name informative
    # seq_filename - sequence's filename - make it informative and uniqe if needed
    # plot_kspace_traj - True if want to see kspace trajectory and save it as an image: False by default

    # Outputs:
    # Reconstructed image using fft: recon
    # kspace: ksp

    os.makedirs(output_dir, exist_ok=True)

    phantom = mr0.VoxelGridPhantom.brainweb(filepath)
    phantom = phantom.interpolate(Nx, Ny, (phantom.PD).shape[2]).slices([(phantom.PD).shape[2]//2]) # Choosing the middle slice
    data = phantom.build()

    if not os.path.exists(seq_filename):
        TSE_seq(plot=True, write_seq=True, seq_filename=seq_filename, TE=TE, n_echo=n_echo, Ny=Ny, Nx=Nx,fov=fov,pe_order_label=pe_order_label,is_horizontal_pe = False)
    seq = mr0.Sequence.import_file(seq_filename)
    if plot_kspace_traj:
        plt.figure(figsize=(6, 6))
        seq.plot_kspace_trajectory()
        traj_path = os.path.join(output_dir, f"kspace_traj_TE{int(TE * 1e3)}ms.png")
        plt.savefig(traj_path)
        plt.close()
        
    n_ex = Ny // n_echo

    # Compute and execute the simulation graph
    graph = mr0.compute_graph(seq, data, 200, 1e-3)
    signal = mr0.execute_graph(graph, seq, data)

    signal = signal.view(n_ex, n_echo, Nx)

    # # Build pe_order exactly as in TSE_seq.py
    #old
    # pe_steps = np.arange(1, n_echo * n_ex + 1) - 0.5 * n_echo * n_ex - 1
    # if divmod(n_echo, 2)[1] == 0:
    #     pe_steps = np.roll(pe_steps, [0, int(-np.round(n_ex / 2))])
    # pe_order = pe_steps.reshape((n_ex, n_echo), order='F').T

    if pe_order_label=='TD': # Top-Down order
        pe_steps = np.linspace(-n_ex * n_echo // 2, n_ex * n_echo // 2 - 1, n_ex * n_echo, dtype=int)
        pe_order = pe_steps.reshape((n_echo, n_ex), order='C')

    elif pe_order_label=='CO': # Center-Out order
        center = n_echo * n_ex // 2
        offsets = np.arange(center)
        pe_steps = np.empty(n_echo * n_ex, dtype=int)
        pe_steps[::2] = center + offsets  # even indices: 0, +1, +2, ...
        pe_steps[1::2] = center - offsets - 1  # odd indices: -1, -2, -3, ...
        pe_steps = pe_steps - center  # shift to be centered around 0

        pe_order = pe_steps.reshape((n_echo, n_ex), order='C')

    delta_k = 1 / fov
    phase_areas = pe_order * delta_k
    #
    if is_horizontal_pe:
        ksp = torch.zeros(Nx, Ny, dtype=signal.dtype)
    else:
        ksp = torch.zeros(Ny, Nx, dtype=signal.dtype)

    # Fill in k-space line-by-line
    for ex in range(n_ex):
        for e in range(n_echo):
            ky_f = phase_areas[e, ex] # Physical phase encoding value
            if is_horizontal_pe:
                kx = int(round(phase_areas[e, ex] / delta_k)) + Nx // 2
                if 0 <= kx < Nx:
                    ksp[kx, :] = signal[ex, e, :]
            else: #original vertical
                ky = int(round(ky_f / delta_k)) + Ny // 2 # Convert to array index
                if 0 <= ky < Ny:
                    #  Each (ex, e) corresponds to one k-space line at a certain ky-position
                    ksp[ky, :] = signal[ex, e, :]

    # Reconstruct using ifft
    reco = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(ksp)))

    # Save image
    fig_img = plt.figure()
    plt.imshow(reco.abs().numpy(), cmap="gray")
    title = f"Reconstructed Image for Turbo Factor={n_echo}, {'Horizontal' if is_horizontal_pe else 'Vertical'}"
    plt.title(title)

    info = (
        f"TE = {TE * 1e3:.0f} ms\n"
        f"TR = 2 s\n"
        f"Turbo Factor = {n_echo}\n"
        #f"Orientation = {'Horizontal' if is_horizontal_pe else 'Vertical'}"
    )

    # Get image dimensions
    height, width = reco.abs().numpy().shape

    # Add text in top-right corner (adjust offset for padding)
    plt.text(
        2,  # X near right edge
        5,  # Y near top
        info,
        fontsize=12,
        color='white',
        ha='left',  # align text to the right edge
        va='top'  # align top of text block to y=5
    )

    plt.axis('off')
    plt.tight_layout()
    plt.show()


    img_path = os.path.join(output_dir, f"{title}.png")
    fig_img.savefig(img_path)
    plt.close(fig_img)

    return reco,ksp

recon, ksp = PDG_sim(filepath="output/brainweb/subject05_3T.npz", Nx=128, Ny=128, TE=12e-3, n_echo= 32, fov=256e-3, output_dir='TSE_horizontal_TD_n_echo_32_te12', seq_filename='TSE_vertical_seq_TE12ms_tf32_TD_fixed.seq', plot_kspace_traj=True,pe_order_label='TD',is_horizontal_pe = False)
