import numpy as np
# import sigpy as sp
# import sigpy.mri as mr
# import sigpy.plot as pl
import torch
from PDG_sim import PDG_sim
import matplotlib.pyplot as plt
import math

Nx = 128
Ny = 128
n_echo = 32
n_ex = math.floor(Ny/n_echo)
fov = 256e-3
delta_k = 1/fov
TE = 12e-3

_, ksp = PDG_sim(filepath="output/brainweb/subject05_3T.npz", Nx=Nx, Ny=Ny, TE=TE, n_echo= n_echo, fov=fov, output_dir='TSE_seq_n_echo=8', seq_filename='TSE_seq_TE12ms_tf8.seq', plot_ksapce_traj=False)
_, fully_sampled_ksp = PDG_sim(filepath="output/brainweb/subject05_3T.npz", Nx=Nx, Ny=Ny, TE=TE, n_echo= 1, fov=fov, output_dir='SE_seq_n_echo=1', seq_filename='SE_seq_TE12ms.seq', plot_ksapce_traj=False)
torch.save(fully_sampled_ksp,'fully_sampled_ksp.pt')
pe_steps = np.arange(1, n_echo * n_ex + 1) - 0.5 * n_echo * n_ex - 1
if divmod(n_echo, 2)[1] == 0:
    pe_steps = np.roll(pe_steps, [0, int(-np.round(n_ex / 2))])
pe_order = pe_steps.reshape((n_ex, n_echo), order='F').T
phase_areas = pe_order * delta_k

undersampled_ksp_opt = torch.zeros_like(ksp)
for ex in range(n_ex):
    ky_f = phase_areas[0, ex]
    ky = int(round(ky_f / delta_k)) + Ny // 2
    if 0 <= ky < Ny:
        undersampled_ksp_opt[ky, :] = ksp[ky, :]

torch.save(undersampled_ksp_opt, 'undersampled_ksp_opt.pt')

undersampled_ksp_nopt = torch.zeros_like(ksp)
for ex in range(n_ex):
    ky_f = phase_areas[n_echo-1, ex]
    ky = int(round(ky_f / delta_k)) + Ny // 2
    if 0 <= ky < Ny:
        undersampled_ksp_nopt[ky, :] = ksp[ky, :]

torch.save(undersampled_ksp_nopt, 'undersampled_ksp_nopt.pt')

undersampled_ksp_1st_shot = torch.zeros_like(ksp)
for e in range(n_echo):
    ky_f = phase_areas[e, 0]
    ky = int(round(ky_f / delta_k)) + Ny // 2
    if 0 <= ky < Ny:
        undersampled_ksp_1st_shot[ky, :] = ksp[ky, :]

mask = (undersampled_ksp_1st_shot.abs() > 0).int()

non_zero_rows = (undersampled_ksp_1st_shot.abs().sum(dim=1) > 0).nonzero(as_tuple=True)[0]

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(np.abs(undersampled_ksp_1st_shot.numpy()),cmap ='gray')
plt.title("Undersampled ksp - only first shot rows")

plt.subplot(1,2,2)
plt.imshow(mask, cmap='gray')
plt.title(f"Sampling mask\nRows {non_zero_rows.tolist()}")

plt.tight_layout()
plt.show()
torch.save(undersampled_ksp_1st_shot, 'undersampled_ksp_1st_shot.pt')