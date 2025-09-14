import os
import torch
import matplotlib.pyplot as plt
import MRzeroCore as mr0
from TSE_seq_applied_to_PDG_sim import TSE_seq
from plot_recon_and_kspace import plot_log_kspace_and_recon

def PDG_sim(filepath,output_dir, Nx, Ny, n_echo, fov, TEeff, TR, TE1, seq_filename, plot_kspace_traj=False,pe_order_label='TD',direction_label = 'vertical',shift=False):
    """
    Purpose:
    1. Create TSE sequence file
    2. Run PDG simulation on a selected slice from brainweb phantom
    3. Do recon using IFFT
    4. Save the recon image and its log magnitude k-space

    Inputs:
    filepath - Brain Phantom .npz file of one subject
    output_dir - directory to store recon and k-space
    Nx, Ny - resolution
    n_echo - echo train length (ETL)
    fov - field of view
    TEeff - effective TE for T2 contrast
    seq_filename - sequence's filename - make it informative and unique if needed - same name as an existing .seq file will change the former file
    plot_kspace_traj - True if you want to see k-space trajectory and save it as an image: False by default
    pe_order label - Top-Down (TD) or Center-Out (CO)
    direction_label - vertical or horizontal
    shift - to match center line acquisition at TEeff. False by default

    Outputs:
    Reconstructed image using IFFT
    k-space
    """

    # Extract center slice and create a sim data object for the simulation
    phantom = mr0.VoxelGridPhantom.brainweb(filepath)
    phantom = phantom.interpolate(Nx, Ny, (phantom.PD).shape[2]).slices([(phantom.PD).shape[2]//2]) # Choosing the middle slice
    data = phantom.build()

    # Generate TSE sequence file
    _,pe_order = TSE_seq(seq_path = os.path.join(base_path, 'sequences'),
    plot=False,write_seq=True,seq_filename=seq_filename,
    Nx=Nx, Ny=Ny, n_echo=n_echo, fov=fov,TE1=TE1, TR=TR, TEeff=TEeff,
    pe_order_label=pe_order_label,  # "CO" (CenterOut) or "TD" (TopDown)
    direction=direction_label,  # "horizontal" (RO=x, PE=y) or "vertical" (RO=y, PE=x)
    shift=shift # True to fit for TEeff for T2 contrast
)
    seq_full_path = os.path.join(base_path, 'sequences', seq_filename)
    seq = mr0.Sequence.import_file(seq_full_path)
    if plot_kspace_traj:
        plt.figure(figsize=(6, 6))
        seq.plot_kspace_trajectory()
        traj_path = os.path.join(output_dir, f"kspace_traj_TE{int(TE1 * 1e3)}ms.png")
        plt.savefig(traj_path)
        plt.show()

    # Compute and execute the simulation graph
    graph = mr0.compute_graph(seq, data, 200, 1e-5)
    signal = mr0.execute_graph(graph, seq, data)

    n_ex = Ny // n_echo #Calculate the number of shots

    signal = signal.view(n_ex, n_echo, Nx)
    ksp = torch.zeros(Ny, Nx, dtype=signal.dtype)

    # Fill in k-space line-by-line
    for ex in range(n_ex):
        for e in range(n_echo):
            idx = int(pe_order[e,ex])
            if direction_label=='horizontal':
                kx = idx + Nx // 2
                if 0 <= kx < Nx:
                    ksp[:, kx] = signal[ex, e, :]
            elif direction_label=='vertical':
                ky = idx + Ny//2
                if 0 <= ky < Ny:
                    ksp[ky, :] = signal[ex, e, :] #  Each (ex, e) corresponds to one k-space line at a certain ky-position

    # Reconstruct using IFFT
    reco = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(ksp)))

    # Plot and Save recon
    seq_root = os.path.splitext(os.path.basename(seq_filename))[0]
    img_filename = f"{seq_root}_recon.png"
    img_path = os.path.join(output_dir, img_filename)

    fig_img = plt.figure()
    plt.imshow(reco.abs().numpy(), cmap="gray")
    plt.title(f"ETL={n_echo}, {'Horiz' if direction_label=='horizontal' else 'Vert'}, PE order = {pe_order_label}")

    info = (
        f"TEeff = {TEeff * 1e3:.0f} ms\n"
        f"TR = {TR} s\n"
        f"Shifted: {'yes' if shift else 'no'}"
    )

    # Add text in top-left corner
    plt.text(2,3, info, fontsize=12, color='white', ha='left', va='top')
    plt.axis('off')
    plt.tight_layout()
    fig_img.savefig(img_path, dpi=300)
    plt.close(fig_img)

    # Save k-space
    ksp_path = os.path.join(ksp_filepath, f"ksp_{seq_root}.pt")
    torch.save(ksp, ksp_path)

    output_dir_image_and_kspace = os.path.join(base_path,'ksp_and_image_figs')
    # Plot k-space
    plot_log_kspace_and_recon(output_dir_image_and_kspace,seq_root,ksp_path,n_echo,pe_order_label,direction_label,shift)

    return reco,ksp

# Set Params and Folder Names
base_path = 'new_data'
outdir = os.path.join(base_path, 'recons')
ksp_and_image_path = os.path.join(base_path, 'ksp_and_image_figs')
ksp_filepath = os.path.join(base_path, 'ksp_tensors')
seq_path = os.path.join(base_path, 'sequences')
filepath = os.path.expanduser("~/AmitProject/data/brainweb/subject04_3T.npz")

Nx,Ny=256,256
TEeff=96e-3
fov = 220e-3
TR = 3
TE1=12e-3
n_echos = [16,32,64,128]
directions = ['vertical', 'horizontal']
pe_order_labels = ['CO','TD']
shifts = [True, False]

# Create Directories
os.makedirs(base_path, exist_ok=True)
os.makedirs(outdir, exist_ok=True)
os.makedirs(ksp_and_image_path, exist_ok=True)
os.makedirs(ksp_filepath, exist_ok=True)
os.makedirs(seq_path, exist_ok=True)

for ETL in n_echos:
    for pe_order_label in pe_order_labels:
        for direction in directions:
            for s in shifts:
                seq_filename = f'ETL{ETL}_TEeff{int(TEeff * 1e3)}ms_{pe_order_label}_{direction}_TR{TR}s_shift_{s}.seq'
                recon, ksp = PDG_sim(filepath=filepath, Nx=Nx, Ny=Ny, n_echo= ETL, fov=fov, TEeff=TEeff, TR=3,TE1=TE1, output_dir=outdir, seq_filename=seq_filename, plot_kspace_traj=False,pe_order_label=pe_order_label,direction_label = direction,shift=s)