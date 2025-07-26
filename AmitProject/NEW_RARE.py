# RARE Sequence Generation for Single-Shot, R=2, Center-Out
import numpy as np
import MRzeroCore as mr0
import pypulseq as pp
import torch
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 5]
plt.rcParams['figure.dpi'] = 100

experiment_id = 'RARE_R2_CO'

# === System Setup ===
system = pp.Opts(
    max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
    rf_ringdown_time=20e-6, rf_dead_time=100e-6,
    adc_dead_time=20e-6, grad_raster_time=10e-6)

seq = pp.Sequence(system)

fov = 0.2  # 200 mm
slice_thickness = 5e-3

base_resolution = 128
Nread = base_resolution
Nphase = base_resolution
R = 1
Ny_acq = Nphase // R
TE = 96e-3
Excitation_FA = 90
Refocusing_FA = 180
r_spoil = 1

# === Events ===
rf1 = pp.make_block_pulse(
    flip_angle=Excitation_FA * np.pi / 180, phase_offset=np.pi/2,
    duration=1e-3, system=system)

rf2 = pp.make_block_pulse(
    flip_angle=Refocusing_FA * np.pi / 180, duration=1e-3, system=system)

sym_rf_block = pp.make_delay(pp.calc_duration(rf1) - rf1.ringdown_time + rf1.delay)

dwell = 50e-6
gx = pp.make_trapezoid(channel='x', flat_area=Nread / fov, flat_time=Nread*dwell, system=system)
adc = pp.make_adc(num_samples=Nread, duration=Nread*dwell, phase_offset=np.pi/2, delay=gx.rise_time, system=system)

gx_pre0 = pp.make_trapezoid(channel='x', area=(1.0 + r_spoil) * gx.area / 2, duration=1.5e-3, system=system)
gx_prewinder = pp.make_trapezoid(channel='x', area=r_spoil * gx.area / 2, duration=1e-3, system=system)

seq.add_block(rf1, sym_rf_block)
seq.add_block(gx_pre0)

# === TE Delay Calculation ===
minTE2 = (pp.calc_duration(sym_rf_block) + pp.calc_duration(gx) + 2*pp.calc_duration(pp.make_trapezoid('y', area=1/fov, duration=1e-3, system=system))) / 2
minTE2 = round(minTE2 / 10e-5) * 10e-5
TEd = round(max(0, (TE/2 - minTE2)) / 10e-5) * 10e-5
seq.add_block(pp.make_delay((minTE2 + TEd) - pp.calc_duration(sym_rf_block) - pp.calc_duration(gx_pre0)))

# === Center-Out PE Steps (R=2) ===
offsets = np.arange(1, Ny_acq // 2 + 1) * 2  # 2,4,...,Ny_acq
pe_steps = np.empty(Ny_acq, dtype=int)
pe_steps[0] = 0
pe_steps[1::2] = -offsets[:Ny_acq//2]
pe_steps[2::2] = offsets[:Ny_acq//2-1]

# === Sequence Loop ===
for ii in pe_steps:
    area = ii / fov
    gp = pp.make_trapezoid(channel='y', area=area, duration=1e-3, system=system)
    gp_ = pp.make_trapezoid(channel='y', area=-area, duration=1e-3, system=system)

    seq.add_block(rf2, sym_rf_block)
    seq.add_block(pp.make_delay(TEd))
    seq.add_block(gx_prewinder, gp)
    seq.add_block(adc, gx)
    seq.add_block(gx_prewinder, gp_)
    seq.add_block(pp.make_delay(TEd))

# === Write .seq ===
ok, error_report = seq.check_timing()
if ok:
    print('Timing check passed successfully')
else:
    print('Timing errors:', error_report)

seq.write("RARE_R1_CO_TE96.seq")
seq.plot()
