import MRzeroCore as mr0
import torch
import matplotlib.pyplot as plt
from TSE_seq_applied_to_PDG_sim import TSE_seq
import numpy as np
fov = 200e-3
slice_thickness = 8e-3

Nread = 128    # frequency encoding steps/samples
Nphase = 128    # phase encoding steps/samples
n_echo = 32
TE1 = 12e-3

obj_p = mr0.CustomVoxelPhantom(
        pos=[[0.0, 0.0, 0.0]],
        PD=[1.0],
        T1=1.0,
        T2=0.1,
        T2dash=0.1,
        D=0.0,
        B0=0,
        voxel_size=0.2,
        voxel_shape="box")
#PD = obj_p.generate_PD_map()
#B0 = torch.zeros_like(PD)

obj_p.plot()
obj_p.size=torch.tensor([fov, fov, slice_thickness])
# Convert Phantom into simulation data
obj_p = obj_p.build()



_, pe_order, t_ref, TR = TSE_seq(plot=True, write_seq=True, seq_filename="TSE_ETL32_TEeff96ms_ESP12ms.seq", n_echo=32, Ny=128, Nx=128,
                             TEeff=96e-3, fov=220e-3, pe_order_label='CO', is_horizontal_pe=True, R=1, TE1=12e-3)
# seq0.plot_kspace_trajectory()
seq0 = mr0.Sequence.import_file("TSE_ETL32_TEeff96ms_ESP12ms.seq")
# Simulate the sequence
graph = mr0.compute_graph(seq0, obj_p, 1, 1e-5)
signal,mag_adc_raw = mr0.execute_graph(graph, seq0, obj_p, return_mag_adc=True, min_latent_signal=1e-5,min_emitted_signal=1e-5)
mag_adc = [rep for rep in mag_adc_raw if len(rep) > 0]
# print([len(rep) for rep in mag_adc], 'total=', sum(len(rep) for rep in mag_adc))
#
# magnitudes = []
# t_abs= []
# for s, rep in enumerate(mag_adc):
#         t0 = s * TR #base of each TR
#         for e, echo in enumerate(rep):
#                 mag = echo.abs()[Nread//2].item()  # mean magnitude across image
#                 magnitudes.append(mag)
#                 t_abs.append((t0 + TE1 + e * t_ref)*1e3)  # ESP = t_ref, e is echo number in each shot
#
# plt.plot(t_abs, magnitudes, 'o-')
# plt.xlabel("Echo time [ms]")
# plt.ylabel("Mean |Mxy| at ADC")
# plt.title("Measured transverse magnetization")
# plt.grid(True)
# plt.show()

echoes = [echo for rep in mag_adc for echo in rep]   # length K (=128)
S = [e.abs().mean().item() for e in echoes]

K = len(S)
n_shots = K // n_echo

ESP = float(t_ref)     # 0.012 s

idx       = np.arange(K)
shot_idx  = idx // n_echo
echo_idx  = idx %  n_echo

t_abs = shot_idx*TR + (TE1 + echo_idx*ESP)   # seconds

plt.plot(t_abs*1e3, S, 'o-')                 # ms for readability
plt.xlabel('Echo time [ms]')
plt.show()