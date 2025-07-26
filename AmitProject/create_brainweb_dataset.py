import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import MRzeroCore as mr0
import sigpy.mri as mr
import time



# Selects the 70 slices with the most tissue (least zeros)
def get_best_slices(PD_volume, num_slices=70):
    non_zero_counts = [(i, (PD_volume[:, :, i] > 0).sum()) for i in range(PD_volume.shape[2])]
    sorted_by_tissue = sorted(non_zero_counts, key=lambda x: -x[1]) # sort by the second value in the tuple in descending order
    best_slices = [idx for idx, _ in sorted_by_tissue[:num_slices]] # save only index of the best slices by taking the first num_slices
    return best_slices


# Takes the signal, reshapes into k-space and reconstructs with IFFT
def recon_func(signal, Nx, Ny, Ny_acq):
    n_ex = signal.shape[0]
    n_echo = signal.shape[1]

    kspace = torch.zeros((Ny, Nx), dtype=signal.dtype)
    if Ny//Ny_acq == 2:
        center = Ny // 2
        acquired_lines = [center]
        offsets = np.arange(2, Ny, 2)
        for offset in offsets:
            up = center + offset
            down = center - offset
            if up < Ny:
                acquired_lines.append(up)
            if down >= 0:
                acquired_lines.append(down)
        acquired_lines = acquired_lines[:Ny_acq]

    elif Ny//Ny_acq == 1:
        shift = 'roll' #or roll
        esp = 12e-3
        TE = 96e-3
        target_echo_index = int(TE // esp)
        center = Ny // 2
        acquired_lines = [center]
        offsets = np.arange(1, Ny+1)
        for offset in offsets:
            up = center + offset
            down = center - offset
            if up < Ny:
                acquired_lines.append(up)
            if down >= 0:
                acquired_lines.append(down)
        acquired_lines = acquired_lines[:Ny_acq]
        acquired_lines_shifted = acquired_lines.copy()
        if shift == 'roll':
            acquired_lines_shifted = np.roll(acquired_lines_shifted, target_echo_index)
        elif shift == 'flip':
            acquired_lines_shifted[:target_echo_index] = acquired_lines_shifted[:target_echo_index][::-1]

    if len(acquired_lines_shifted) != n_ex * n_echo:
        raise ValueError(f"Mismatch: len(acquired_lines)={len(acquired_lines_shifted)} vs expected={n_ex * n_echo}")

    if shift == 'roll':
        signal = signal.view(-1, signal.shape[-1])  # (n_ex * n_echo, Nx)
        signal = torch.roll(signal, shifts=-target_echo_index, dims=0)  # Shift echo axis
        signal = signal.view(n_ex, n_echo, -1)
    elif shift == 'flip':
        signal = signal.view(-1, signal.shape[-1])
        signal[:target_echo_index] = torch.flip(signal[:target_echo_index], dims=[0])
        signal = signal.view(n_ex, n_echo, -1)

    for idx in range(n_ex * n_echo):
        ky = acquired_lines_shifted[idx]
        ex = idx // n_echo
        e = idx % n_echo
        if 0 <= ky < Ny:
            kspace[ky, :] = signal[ex, e, :]


    reco = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(kspace)))
    return reco, kspace

# Executes MRzero simulation for a slice using a .seq file
def PDG_sim(phantom_slice, seq, Nx, Ny, n_echo, Ny_acq):
    data = phantom_slice.build()
    #start_time = time.time()
    graph = mr0.compute_graph(seq, data, 200, 1e-4)
    signal, mag_adc = mr0.execute_graph(graph, seq, data,return_mag_adc=True,min_emitted_signal=1e-4, min_latent_signal=1e-4)
    #end_time = time.time()
    #print(f"Elapsed time: {end_time - start_time:.3f} seconds")

    # magnitudes = []
    # for rep in mag_adc:
    #    for echo in rep:
    #     mag = echo.abs().mean().item()  # mean magnitude across image
    #     magnitudes.append(mag)
    #
    # import matplotlib.pyplot as plt
    # plt.plot(magnitudes, 'o-')
    # plt.xlabel("Echo index")
    # plt.ylabel("Mean |Mxy| at ADC")
    # plt.title("Measured transverse magnetization per ADC, threshold = 1e-3 (default)")
    # plt.grid(True)
    # plt.show()

    # Calculate number of excitations n_ex
    total_adc = signal.numel() #lists all elements
    if total_adc % (n_echo * Nx) != 0:
        raise ValueError(f"Incompatible shape: expected multiple of n_echo*Nx={n_echo * Nx}, got {total_adc}")
    n_ex = total_adc // (n_echo * Nx)

    signal = signal.view(n_ex, n_echo, Nx)
    reco = recon_func(signal, Nx, Ny,Ny_acq)
    return reco

# Saves both .pt and .png for a single slice
def save_image_and_tensor(tensor, tensor_dir, image_dir, subject_id, slice_idx):
    base_name = f"{subject_id}_slice_{slice_idx:02d}"
    torch.save(tensor, os.path.join(tensor_dir, base_name + ".pt"))

    image_np = tensor.abs().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
    image_np = (image_np * 255).astype(np.uint8)
    plt.imsave(os.path.join(image_dir, base_name + ".png"), image_np, cmap='gray')


# Builds the nested folder structure required
def prepare_dirs(base_dir):
    root = os.path.join(base_dir, "sim_data")
    subsets = ["train", "val", "test"]
    for subset in subsets:
        for dtype in ["tensor", "im"]:
            for role in ["input", "target"]:
                path = os.path.join(root, f"{subset}_{dtype}", f"{subset}_{dtype}_{role}")
                os.makedirs(path, exist_ok=True)
        # Add k-space directory
        os.makedirs(os.path.join(root, f"{subset}_tensor", f"{subset}_tensor_ksp"), exist_ok=True)
    return root


# Helper to construct full paths for saving
def get_output_paths(base_dir, subset, dtype, role):
    return os.path.join(base_dir, f"{subset}_{dtype}", f"{subset}_{dtype}_{role}")


def generate_dataset(data_dir, seq_path, output_root, Nx=64, Ny=64, TE=60e-3, fov=200e-3,n_echo=32,R=1):
    base_dir = prepare_dirs(output_root)
    subjects = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])

    # Split subjects
    np.random.seed(42)
    np.random.shuffle(subjects)
    test_subj = subjects[:1]
    val_subj = subjects[1:4]
    train_subj = subjects[4:]

    subset_map = {s: "test" for s in test_subj}
    subset_map.update({s: "val" for s in val_subj})
    subset_map.update({s: "train" for s in train_subj})

    seq = mr0.Sequence.import_file(seq_path)
    Ny_acq = Ny // R

    for subj in subjects:
        subset = subset_map[subj]
        phantom = mr0.VoxelGridPhantom.brainweb(os.path.join(data_dir, subj))
        phantom = phantom.interpolate(Nx, Ny, phantom.PD.shape[2])
        best_slices = get_best_slices(phantom.PD)
        subj_id = subj.replace("_3T", "").replace(".npz", "")

        # for i, slice_idx in enumerate(best_slices):
        for slice_idx in best_slices:
            phantom_slice = phantom.slices([slice_idx])

            # Simulate RARE image and k-space
            reco, kspace = PDG_sim(phantom_slice, seq, Nx, Ny, n_echo, Ny_acq)

            # Save input tensor and image
            tensor_input_dir = get_output_paths(base_dir, subset, "tensor", "input")
            image_input_dir = get_output_paths(base_dir, subset, "im", "input")
            save_image_and_tensor(reco, tensor_input_dir, image_input_dir, subj_id, slice_idx)

            # Save k-space tensor
            ksp_dir = os.path.join(base_dir, f"{subset}_tensor", f"{subset}_tensor_ksp")
            torch.save(kspace, os.path.join(ksp_dir, f"ksp_{subj_id}_slice_{slice_idx:02d}.pt"))

            # Save target tensor and image (PD map)
            pd = phantom_slice.PD[0, :, :]
            t2 = phantom_slice.T2[0, :, :]
            gt = pd * np.exp(-TE / (t2 + 1e-8))
            tensor_target_dir = get_output_paths(base_dir, subset, "tensor", "target")
            image_target_dir = get_output_paths(base_dir, subset, "im", "target")
            save_image_and_tensor(torch.tensor(gt), tensor_target_dir, image_target_dir, subj_id, slice_idx)

        print(f"Done {subj_id}, saved to {subset} set")

    print("Dataset generation complete.")

# Usage example:
data_dir = os.path.expanduser("~/AmitProject/data/brainweb")
seq_path = os.path.expanduser("~/AmitProject/TSE_SEQ_CO_R1_TE96MS_TF128_ESP12ms_roll.seq")
output_root = os.path.expanduser("~/AmitProject")
generate_dataset(data_dir=data_dir, seq_path=seq_path, output_root=output_root,Nx=128, Ny=128,TE=96e-3,n_echo=128, fov= 200e-3, R=1)
