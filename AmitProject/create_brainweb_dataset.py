import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import MRzeroCore as mr0


# Selects the 70 slices with the most tissue (the least zeros)
def get_best_slices(PD_volume, num_slices=70):
    non_zero_counts = [(i, (PD_volume[:, :, i] > 0).sum()) for i in range(PD_volume.shape[2])]
    sorted_by_tissue = sorted(non_zero_counts, key=lambda x: -x[1]) # sort by the second value in the tuple in descending order
    best_slices = [idx for idx, _ in sorted_by_tissue[:num_slices]] # save only index of the best slices by taking the first num_slices
    return best_slices

def recon_func2(signal, Nx, Ny):
    # n_echo is a known constant from the sequence file
    n_echo = 32
    # Load the phase encoding order that was saved
    pe_order_array = np.load('pe_order_used.npy')

    # Check if the signal has the expected number of echoes for the number of shots (n_ex)
    if signal.numel() % (n_echo * Nx) != 0:
        raise ValueError(f"Signal size {signal.numel()} is not a multiple of (n_echo * Nx) = {n_echo * Nx}.")

    n_ex = signal.numel() // (n_echo * Nx)
    # Reshape the signal to separate shots and echoes
    signal = signal.view(n_ex, n_echo, Nx)

    # Initialize k-space with zeros
    kspace = torch.zeros((Ny, Nx), dtype=signal.dtype)

    # Map each echo from each shot to its correct k-space line using the PE order
    for ex_idx in range(n_ex):
        for echo_idx in range(n_echo):
            # Get the correct ky line from the PE order array
            ky = pe_order_array[echo_idx, ex_idx] + Ny // 2

            # Map the signal data to k-space
            kspace[ky, :] = signal[ex_idx, echo_idx, :]

    filled_lines = (kspace.abs().sum(dim=1) > 0).sum()
    print(f"Filled lines: {filled_lines}")

    reco = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(kspace)))
    return reco, kspace

# Takes the signal, reshapes into k-space and reconstructs with IFFT
def recon_func(signal, Nx, Ny):
    n_ex = signal.shape[0]
    n_echo = signal.shape[1]

    kspace = torch.zeros((Ny, Nx), dtype=signal.dtype)
    pe_order = np.load('pe_order_used.npy')  # shape (n_echo, n_ex)
    pe_order_flat = pe_order.flatten(order='F').astype(np.int64)
    if len(pe_order_flat) != n_ex * n_echo:
        raise ValueError("Mismatch between signal size and PE ordering")

    for idx in range(n_ex * n_echo):
        ky = pe_order_flat[idx]+Ny//2
        ex = idx // n_echo
        e = idx % n_echo
        if 0 <= ky < Ny:
            kspace[ky, :] = signal[ex, e, :]

    filled_lines = (kspace.abs().sum(dim=1) > 0).sum()
    print(f"Filled lines: {filled_lines}")

    reco = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(kspace)))
    return reco, kspace

# Executes MRzero simulation for a slice using a .seq file
def PDG_sim(phantom_slice, seq, Nx, Ny, n_echo, Ny_acq):
    data = phantom_slice.build()

    graph = mr0.compute_graph(seq, data, 200, 1e-5)
    signal, mag_adc_raw = mr0.execute_graph(graph, seq, data,return_mag_adc=True,min_emitted_signal=1e-5, min_latent_signal=1e-5)
    # mag_adc = [rep for rep in mag_adc_raw if len(rep) > 0]
    #
    # echoes = [echo for rep in mag_adc for echo in rep]  # length K (=128)
    # S = [e.abs().mean().item() for e in echoes]
    #
    # K = len(S)
    # n_shots = K // n_echo
    # t_ref = 0.012
    # TE1 = 0.012
    # TR = 0.8783899999999999
    # ESP = float(t_ref)  # 0.012 s
    # idx = np.arange(K)
    # shot_idx = idx // n_echo
    # echo_idx = idx % n_echo
    #
    # t_abs = shot_idx * TR + (TE1 + echo_idx * ESP)  # seconds
    #
    # plt.plot(t_abs * 1e3, S, 'o-')  # ms for readability
    # plt.xlabel('Echo time [ms]')
    # plt.ylabel("Mean |Mxy| at ADC")
    # plt.title("Measured transverse magnetization- mean amplitude per echo")
    # plt.show()

    # Calculate number of excitations n_ex
    total_adc = signal.numel() #lists all elements
    if total_adc % (n_echo * Nx) != 0:
        raise ValueError(f"Incompatible shape: expected multiple of n_echo*Nx={n_echo * Nx}, got {total_adc}")
    n_ex = total_adc // (n_echo * Nx)

    #signal = signal.view(n_ex, n_echo, Nx)
    reco,kspace = recon_func2(signal, Nx, Ny)
    return reco,kspace


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


# Construct full paths for saving
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

            # Simulate TSE image and k-space
            reco, kspace = PDG_sim(phantom_slice, seq, Nx, Ny, n_echo, Ny_acq)

            # Save input tensor and image
            tensor_input_dir = get_output_paths(base_dir, subset, "tensor", "input")
            image_input_dir = get_output_paths(base_dir, subset, "im", "input")
            save_image_and_tensor(reco, tensor_input_dir, image_input_dir, subj_id, slice_idx)

            # Save k-space tensor
            ksp_dir = os.path.join(base_dir, f"{subset}_tensor", f"{subset}_tensor_ksp")
            torch.save(kspace, os.path.join(ksp_dir, f"ksp_{subj_id}_slice_{slice_idx:02d}.pt"))

        print(f"Done {subj_id}, saved to {subset} set")

    print("Dataset generation complete.")

data_dir = os.path.expanduser("~/AmitProject/data/brainweb")
seq_path = os.path.expanduser("~/AmitProject/TSE_ETL32_TEeff96ms_ESP12ms_fov220_interleaved_co_TR4s.seq")
output_root = os.path.expanduser("~/AmitProject")
generate_dataset(data_dir=data_dir, seq_path=seq_path, output_root=output_root,Nx=128, Ny=128,TE=120e-3,n_echo=32, fov= 200e-3, R=1)
