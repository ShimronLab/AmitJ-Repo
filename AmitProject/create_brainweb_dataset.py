import os
import numpy as np
from pathlib import Path
import MRzeroCore as mr0
from PDG_sim import PDG_sim as run_pdg_sim

def get_best_slices(PD_volume, num_slices=70):
    """
    Pick slices with the most non-zero PD (most tissue)
    :param PD_volume: PD map per slice
    :param num_slices: number of slices to keep from the subject volume
    :return: indices of the best 'num_slices' slices
    """
    non_zero_counts = [(i, (PD_volume[:, :, i] > 0).sum()) for i in range(PD_volume.shape[2])]
    sorted_by_tissue = sorted(non_zero_counts, key=lambda x: -x[1]) # sort by the second value in the tuple in descending order
    best_slices = [idx for idx, _ in sorted_by_tissue[:num_slices]] # save only index of the best slices by taking the first num_slices
    return best_slices


def prepare_dirs(base_dir):
    """
    Create:
    sim_data/
        train/{recons, ksps_im, ksps_tensor}
        val/{recons, ksps_im, ksps_tensor}
        test/{recons, ksps_im, ksps_tensor}
        sequences/ (shared place for .seq files)
    :param base_dir:
    :return:
    """
    root = Path(base_dir) / "sim_data"
    for subset in ("train", "val", "test"):
        (root / subset / "recons").mkdir(parents=True, exist_ok=True)
        (root / subset / "ksps_im").mkdir(parents=True, exist_ok=True)
        (root / subset / "ksps_tensor").mkdir(parents=True, exist_ok=True)
    (root / "sequences").mkdir(parents=True, exist_ok=True)
    return root

def _make_seq_filename(subj_id, slice_idx, n_echo, peo, direction, TR, TEeff, shift, run_tag=None):
    desc = f"ETL{n_echo}_TEeff{int(TEeff*1e3)}ms_{peo}_{direction}_TR{int(TR)}s_shift_{shift}"
    if run_tag:
        desc += f"_tag-{run_tag}"
    return f"{subj_id}_slice_{slice_idx:03d}_{desc}.seq"

def _already_done(recon_dir: Path, ksp_tensor_dir: Path, seq_filename: str) -> bool:
    stem = Path(seq_filename).stem
    return (recon_dir / f"{stem}.png").exists() and (ksp_tensor_dir / f"ksp_{stem}.pt").exists()


def run_one_subject(
    subject_npz, subset, data_dir, output_root,
    Nx=256, Ny=256, fov=220e-3, TR=3.0, TE1=12e-3, TEeff=96e-3,
    n_echos=(16,32,64,128), directions=('vertical','horizontal'),
    pe_order_labels=('TD',), shift=True, run_tag=None
):
    """
    Minimal manual runner: process ONE subject and save into the given subset.
    """
    root = prepare_dirs(output_root)
    seq_dir = root / "sequences"

    subj_path = os.path.join(data_dir, subject_npz)
    subj_id = subject_npz.replace("_3T", "").replace(".npz", "")

    recon_dir = root / subset / "recons"
    ksp_im_dir = root / subset / "ksps_im"
    ksp_tensor_dir = root / subset / "ksps_tensor"

    # load + interpolate phantom once
    phantom = mr0.VoxelGridPhantom.brainweb(subj_path)
    phantom = phantom.interpolate(Nx, Ny, phantom.PD.shape[2])
    best_slices = get_best_slices(phantom.PD)

    total = len(best_slices) * len(n_echos) * len(directions) * len(pe_order_labels)
    done = skipped = 0

    for slice_idx in best_slices:
        for n_echo in n_echos:
            for direction in directions:
                seq_filename = _make_seq_filename(
                    subj_id=subj_id, slice_idx=slice_idx, n_echo=n_echo,
                    peo=pe_order_labels, direction=direction, TR=TR, TEeff=TEeff, shift=shift, run_tag=run_tag
                )
                if _already_done(recon_dir, ksp_tensor_dir, seq_filename):
                    skipped += 1
                    continue

                run_pdg_sim(
                        filepath=subj_path,
                        Nx=Nx, Ny=Ny, n_echo=n_echo,
                        fov=fov, TEeff=TEeff, TR=TR, TE1=TE1,
                        seq_filename=seq_filename,
                        seq_dir=str(seq_dir),
                        recon_dir=str(recon_dir),
                        ksp_tensor_dir=str(ksp_tensor_dir),
                        ksp_im_dir=str(ksp_im_dir),
                        slice_idx=slice_idx,
                        plot_kspace_traj=False,
                        pe_order_label=pe_order_labels,
                        direction_label=direction,
                        shift=shift
                )
                done += 1

    print(f"[{subj_id}] → subset={subset} | completed {done}, skipped {skipped}/{total}")

def generate_dataset(data_dir,output_root,
                     Nx=256, Ny=256, fov=220e-3, TR=3.0, TE1=12e-3, TEeff=96e-3,
                     n_echos=(16,32,64,128),directions=('vertical','horizontal'), pe_order_labels=('TD','CO'),
                     shift=False,run_tag=None):

    root = prepare_dirs(output_root)
    seq_dir = root / "sequences"

    subjects = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])

    # Split subjects (3 test, 3 val, 14 train)
    rng = np.random.default_rng(42)
    rng.shuffle(subjects)
    test_subj = subjects[:3]
    val_subj = subjects[3:6]
    train_subj = subjects[6:]

    subset_map = {s: "test" for s in test_subj}
    subset_map.update({s: "val" for s in val_subj})
    subset_map.update({s: "train" for s in train_subj})

    for subj_npz in subjects:
        subset = subset_map[subj_npz]
        subj_path = os.path.join(data_dir, subj_npz)

        # load and interpolate phantom once per subject
        phantom = mr0.VoxelGridPhantom.brainweb(subj_path)
        phantom = phantom.interpolate(Nx, Ny, phantom.PD.shape[2])

        best_slices = get_best_slices(phantom.PD)
        subj_id = subj_npz.replace("_3T", "").replace(".npz", "")

        # output dirs for this subset
        recon_dir = root / subset / "recons"
        ksp_im_dir = root / subset / "ksps_im"
        ksp_tensor_dir = root / subset / "ksps_tensor"

        for slice_idx in best_slices:
            for n_echo in n_echos:
                for peo in pe_order_labels:
                    for direction in directions:
                        desc = f"ETL{n_echo}_TEeff{int(TEeff * 1e3)}ms_{peo}_{direction}_TR{int(TR)}s_shift_{shift}"
                        if run_tag:
                            desc += f"_tag-{run_tag}"

                        seq_filename = f"{subj_id}_slice_{slice_idx:03d}_{desc}.seq"

                        run_pdg_sim(filepath=subj_path,
                            Nx=Nx, Ny=Ny,n_echo=n_echo,
                            fov=fov, TEeff=TEeff, TR=TR, TE1=TE1,
                            seq_filename=seq_filename,
                            seq_dir=str(seq_dir),
                            recon_dir=str(recon_dir),
                            ksp_tensor_dir=str(ksp_tensor_dir),
                            ksp_im_dir=str(ksp_im_dir),
                            slice_idx=slice_idx,
                            plot_kspace_traj=False,
                            pe_order_label=peo,direction_label=direction,shift=shift)

        print(f"Done {subj_id} → {subset}")

    print("Dataset generation complete.")


data_dir = os.path.expanduser("~/AmitProject/data/brainweb")
output_root = os.path.expanduser("~/AmitProject")

# here we run one subject at a time since it takes a lot of time.
# we assign it to a subset of choice, so we match the division of:
# 14 train | 3 val | 3 test
run_one_subject(
    subject_npz="subject04_3T.npz",
    subset="test",                      # <-- manually choose: "train"/"val"/"test"
    data_dir=data_dir,
    output_root=output_root,
    Nx=256, Ny=256,
    fov=220e-3, TR=3.0, TE1=12e-3, TEeff=96e-3,
    n_echos=(16,32,64,128),
    directions=('vertical','horizontal'),
    pe_order_labels='TD',
    shift=True,
    run_tag="first_try"
)
# All subjects run still exists under generate_dataset function
# generate_dataset(data_dir=data_dir, output_root=output_root,
#     Nx=256, Ny=256, fov=220e-3,TR=3.0,TE1=12e-3,TEeff=96e-3,n_echos=(16,32,64,128),
#     directions=('vertical','horizontal'), pe_order_labels=('TD','CO'), shift=False,run_tag=None)

