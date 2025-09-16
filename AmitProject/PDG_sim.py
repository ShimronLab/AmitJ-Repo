import os
import torch
import matplotlib.pyplot as plt
import MRzeroCore as mr0
from pathlib import Path
from TSE_seq_applied_to_PDG_sim import TSE_seq
from PIL import Image
import numpy as np

def PDG_sim(filepath,
            Nx, Ny, n_echo, fov, TEeff, TR, TE1,
            seq_filename, ksp_tensor_dir, seq_dir,recon_dir, ksp_im_dir=None, slice_idx=None,
            plot_kspace_traj=False,pe_order_label='TD',direction_label = 'vertical',shift=False):
    """
    Purpose:
    1. Create TSE sequence file (.seq)
    2. Run PDG simulation on a selected slice from brainweb phantom
    3. IFFT recon
    4. Save recon (PNG) to output_dir and k-space (PT) to ksp_dir

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

    # Extract a single slice and create a sim data object for the simulation
    phantom = mr0.VoxelGridPhantom.brainweb(filepath)
    nz = phantom.PD.shape[2]
    if slice_idx is None:
        slice_idx = nz//2
    phantom = phantom.interpolate(Nx, Ny, (phantom.PD).shape[2]).slices([slice_idx])
    data = phantom.build()

    # Make sure dirs exist
    seq_dir = Path(seq_dir)
    seq_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = Path(recon_dir)
    recon_dir.mkdir(parents=True, exist_ok=True)
    ksp_tensor_dir = Path(ksp_tensor_dir)
    ksp_tensor_dir.mkdir(parents=True, exist_ok=True)
    if ksp_im_dir is not None:
        ksp_im_dir = Path(ksp_im_dir)
        ksp_im_dir.mkdir(parents=True, exist_ok=True)

    # Generate TSE sequence file
    _,pe_order = TSE_seq(seq_path = str(seq_dir),
    plot=False,write_seq=True,seq_filename=seq_filename,
    Nx=Nx, Ny=Ny, n_echo=n_echo, fov=fov,TE1=TE1, TR=TR, TEeff=TEeff,
    pe_order_label=pe_order_label,  # "CO" (CenterOut) or "TD" (TopDown)
    direction=direction_label,  # "horizontal" (RO=x, PE=y) or "vertical" (RO=y, PE=x)
    shift=shift # True to fit for TEeff for T2 contrast
    )

    # Load seq file
    seq_full_path = Path(seq_dir) / str(seq_filename)  # ensure it's a Path made from strs
    seq = mr0.Sequence.import_file(str(seq_full_path))

    # Plot k-space trajectory - Optional
    if plot_kspace_traj:
        plt.figure(figsize=(6, 6))
        seq.plot_kspace_trajectory()
        traj_path = os.path.join(recon_dir, f"kspace_traj_TE{int(TE1 * 1e3)}ms.png")
        plt.savefig(traj_path)
        plt.close()

    # Compute and execute the simulation
    graph = mr0.compute_graph(seq, data, 200, 1e-5)
    signal = mr0.execute_graph(graph, seq, data)

    n_ex = Ny // n_echo
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

    # Debug to make sure all lines were acquired
    if direction_label == 'vertical':
        filled_ky = int((ksp.abs().sum(dim=1) > 0).sum().item())
        print(f"[dbg] filled ky: {filled_ky}/{Ny}")
    elif direction_label == 'horizontal':
        filled_kx = int((ksp.abs().sum(dim=0) > 0).sum().item())
        print(f"[dbg] filled kx: {filled_kx}/{Nx}")

    # Reconstruct using IFFT
    reco = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(ksp)))

    # Save recon image (PNG)
    seq_root = os.path.splitext(os.path.basename(seq_filename))[0]
    img_filename = f"{seq_root}.png"
    img_path = os.path.join(recon_dir, img_filename)

    mag = reco.abs().cpu().numpy().astype(np.float32)
    den = float(mag.max())
    mag16 = np.zeros_like(mag, dtype=np.uint16) if den == 0 else (mag / den * 65535.0).astype(np.uint16)
    Image.fromarray(mag16, mode='I;16').save(img_path)

    # Save k-space
    ksp_path = os.path.join(ksp_tensor_dir, f"ksp_{seq_root}.pt")
    torch.save(ksp, ksp_path)

    # Save - log magnitude of k-space
    if ksp_im_dir is not None:
        log_kspace = torch.log1p(ksp.abs()).cpu().numpy().astype(np.float32)
        den = float(log_kspace.max())
        log16 = np.zeros_like(log_kspace, dtype=np.uint16) if den == 0 else (log_kspace / den * 65535).astype(np.uint16)
        klog_path = ksp_im_dir / f"log_ksp_{seq_root}.png"
        Image.fromarray(log16, mode="I;16").save(klog_path)

    return reco,ksp

if __name__ == "__main__":
    base = Path("new_data")
    test = True
    #simple test run
    if test:
        filepath = os.path.expanduser("~/AmitProject/data/brainweb/subject04_3T.npz")
        PDG_sim(filepath=filepath,Nx=256, Ny=256, n_echo=16, fov=220e-3, TEeff=96e-3, TR=3.0, TE1=12e-3,
        seq_filename="subject04_slice_050_ETL32_TEeff96ms_TD_vertical_TR3s.seq",
        seq_dir=base / "sequences",
        recon_dir=base / "recons",
        ksp_tensor_dir=base / "ksps_tensor",
        ksp_im_dir=base / "ksps_im",
        slice_idx=50,
        pe_order_label="TD",
        direction_label="vertical",
        shift=False
        )
    else:
        base_path = 'new_data_test'
        outdir = os.path.join(base_path, 'recons')
        figs_dir = os.path.join(base_path, 'ksp_and_image_figs')
        ksp_dir = os.path.join(base_path, 'ksp_tensors')
        seq_dir = os.path.join(base_path, 'sequences')
        filepath = os.path.expanduser("~/AmitProject/data/brainweb/subject04_3T.npz")

        Nx, Ny = 256, 256
        TEeff = 96e-3
        fov = 220e-3
        TR = 3
        TE1 = 12e-3
        n_echos = [16,32,64,128]
        directions = ['vertical', 'horizontal']
        pe_order_labels = ['TD','CO']
        shifts = [True, False]

        os.makedirs(base_path, exist_ok=True)
        for d in [outdir, figs_dir, ksp_dir, seq_dir]:
            os.makedirs(d, exist_ok=True)

        for ETL in n_echos:
            for peo in pe_order_labels:
                for direction in directions:
                    for s in shifts:
                        seq_filename = f'ETL{ETL}_TEeff{int(TEeff * 1e3)}ms_{peo}_{direction}_TR{TR}s_shift_{s}.seq'
                        PDG_sim(filepath=filepath, Nx=Nx, Ny=Ny, n_echo=ETL, fov=fov, TEeff=TEeff, TR=TR, TE1=TE1,
                                seq_dir=base / "sequences",
                                recon_dir=base / "recons",
                                ksp_tensor_dir=base / "ksps_tensor",
                                ksp_im_dir=base / "ksps_im",
                                pe_order_label=peo,
                                direction_label=direction,
                                shift=s)
