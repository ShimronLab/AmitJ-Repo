import numpy as np
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
pe_order_label="TD"
horizontal = True
vertical = False
R = 4 #accelation factor

_, ksp_h = PDG_sim(filepath="data/output/brainweb/subject05_3T.npz", Nx=Nx, Ny=Ny, TE=TE, n_echo= n_echo, fov=fov, output_dir='../data/TSE_TF32_TE12', seq_filename='TSE_Horiz_TopDown.seq', plot_kspace_traj=False, pe_order_label=pe_order_label, is_horizontal_pe=horizontal)
_, ksp_v = PDG_sim(filepath="data/output/brainweb/subject05_3T.npz", Nx=Nx, Ny=Ny, TE=TE, n_echo= n_echo, fov=fov, output_dir='../data/TSE_TF32_TE12', seq_filename='TSE_Vert_TopDown.seq', plot_kspace_traj=False, pe_order_label=pe_order_label, is_horizontal_pe=vertical)
# old
# pe_steps = np.arange(1, n_echo * n_ex + 1) - 0.5 * n_echo * n_ex - 1
# if divmod(n_echo, 2)[1] == 0:
#     pe_steps = np.roll(pe_steps, [0, int(-np.round(n_ex / 2))])
# pe_order = pe_steps.reshape((n_ex, n_echo), order='F').T
# phase_areas = pe_order * delta_k

if pe_order_label == 'TD':  # Top-Down order
    pe_steps = np.linspace(-n_ex * n_echo // 2, n_ex * n_echo // 2 - 1, n_ex * n_echo, dtype=int)
    pe_order = pe_steps.reshape((n_echo, n_ex), order='C')

elif pe_order_label == 'CO':  # Center-Out order
    center = n_echo * n_ex // 2
    offsets = np.arange(center)
    pe_steps = np.empty(n_echo * n_ex, dtype=int)
    pe_steps[::2] = center + offsets  # even indices: 0, +1, +2, ...
    pe_steps[1::2] = center - offsets - 1  # odd indices: -1, -2, -3, ...
    pe_steps = pe_steps - center  # shift to be centered around 0

    pe_order = pe_steps.reshape((n_echo, n_ex), order='C')

delta_k = 1 / fov
phase_areas = pe_order * delta_k

np.random.seed(1) #to reproduce

#Horizontal Sim - Horizontal Undersampling

n_keep_h = Nx // R
sampled_cols = np.random.choice(Nx, n_keep_h, replace=False)

mask_h = np.zeros_like(ksp_h, dtype=bool)
mask_h[:,sampled_cols] = True #mask each sampled col as 1

undersampled_ksp_h = ksp_h.clone()
undersampled_ksp_h[~mask_h] = 0 #when mask_h= 0 put 0 and otherwise keep the line

#Vertical Sim - Vertical Undersampling
n_keep_v = Ny//R
sampled_rows = np.random.choice(Ny, n_keep_v, replace=False)

mask_v = np.zeros_like(ksp_v, dtype=bool)
mask_v[sampled_rows,:] = True

undersampled_ksp_v = ksp_v.clone()
undersampled_ksp_v[~mask_v] = 0

pe_order_label_h = 'LeftRight' if pe_order_label == 'TD' and horizontal else 'CenterOut'

pe_order_label_v = 'TopDown' if pe_order_label == 'TD' else 'CenterOut'

# Plot Horizontal - ksp_h | undersampled_ksp_h | mask_h
plt.figure(figsize=(12, 4))
plt.suptitle(f"log1p(abs(ksp) {pe_order_label_h} with R={R} - (Horizontal PE)")

plt.subplot(1, 3, 1)
plt.imshow(np.log1p(ksp_h.abs().numpy()), cmap='gray')
plt.title("Full k-space")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(np.log1p(undersampled_ksp_h.abs().numpy()), cmap='gray')
plt.title("Undersampled k-space")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(mask_h, cmap='gray')
plt.title("Sampling Mask")
plt.axis('off')

plt.tight_layout()
plt.show()

# Plot Vertical - ksp_v | undersampled_ksp_v | mask_v
plt.figure(figsize=(12, 4))
plt.suptitle(f"log1p(abs(ksp) {pe_order_label_v} with R={R} - (Vertical PE)")

plt.subplot(1, 3, 1)
plt.imshow(np.log1p(ksp_v.abs().numpy()), cmap='gray')
plt.title("Full k-space")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(np.log1p(undersampled_ksp_v.abs().numpy()), cmap='gray')
plt.title("Undersampled k-space")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(mask_v, cmap='gray')
plt.title("Sampling Mask")
plt.axis('off')

plt.tight_layout()
plt.show()

np.savez('mask_h.npz', mask=mask_h)
np.savez('mask_v.npz', mask=mask_v)

torch.save(undersampled_ksp_h, 'undersampled_ksp_h.pt')
torch.save(undersampled_ksp_v, 'undersampled_ksp_v.pt')