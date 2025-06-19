
from bart import bart
import torch
import numpy as np
import matplotlib.pyplot as plt

# # bart pics command help
# !bart pics -h
# !bart pics -Rh
# !bart fft -h

undersampled_ksp_v_TD = r'/home/amit.ja/data/undersampled_ksp_v.pt'
undersampled_ksp_h_TD = r'/home/amit.ja/data/undersampled_ksp_h.pt' 

undersampled_ksp_v_TD = torch.load(undersampled_ksp_v_TD).numpy()
undersampled_ksp_h_TD = torch.load(undersampled_ksp_h_TD).numpy()

undersampled_ksp_v_TD = np.expand_dims(undersampled_ksp_v_TD, axis=0)   # shape: [1, Ny, Nx] for sens_map
undersampled_ksp_h_TD = np.expand_dims(undersampled_ksp_h_TD, axis=0)
sens_map = np.ones_like(undersampled_ksp_v_TD)

img_v = bart(1, 'pics -d1 -i100 -R W:7:0:0.1', undersampled_ksp_v_TD, sens_map) # use d4 for debugging
img_h = bart(1, 'pics -d1 -i100 -R W:7:0:0.1', undersampled_ksp_h_TD, sens_map)

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(np.abs(img_v[0]), cmap='gray')
plt.title('Reconstructed Vertical')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(np.abs(undersampled_ksp_v_TD[0]), cmap='gray')
plt.title('Undersampled k-Space - Vertical')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(np.abs(img_h[0].T), cmap='gray')
plt.title('Reconstructed Horizontal')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(np.abs(undersampled_ksp_h_TD[0]), cmap='gray')
plt.title('Undersampled k-Space - Horizontal')
plt.axis('off')

plt.tight_layout()
plt.savefig("/home/amit.ja/data/Vertical_vs_horitzontal_reconstruction.png")#, bbox_inches='tight', pad_inches=0)
plt.show()

np.save("/home/amit.ja/data/img_v.npy", np.abs(img_v[0]))
np.save("/home/amit.ja/data/img_h.npy", np.abs(img_h[0]))





