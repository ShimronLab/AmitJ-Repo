# Taken from https://github.com/MRsources/Article_MAGMA24_RMRI/blob/main/scripts/TSE_PyPulseq_mr0_mrpro_BART.ipynb
# Was provided in https://scholar.google.co.il/citations?view_op=view_citation&hl=en&user=CylZPggAAAAJ&sortby=pubdate&citation_for_view=CylZPggAAAAJ:dhFuZR0502QC

import numpy as np
import MRzeroCore as mr0
import pypulseq as pp
import torch
import matplotlib.pyplot as plt
import warnings

def TSE_2D_seq_gen(base_resolution=128, fov=200e-3, TE=96e-3, TR=5, PEtype='centric', shots=1, plot: bool = False, write: bool = False, seq_filename: str = 'TSE_CENTRIC.seq'):
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.rcParams['figure.dpi'] = 100

    experiment_id = 'TSE_2D'

    system = pp.Opts(
        max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
        rf_ringdown_time=20e-6, rf_dead_time=100e-6,
        adc_dead_time=20e-6, grad_raster_time=10e-6)

    seq = pp.Sequence(system)

    slice_thickness = 5e-3

    Nread = base_resolution
    Nphase = base_resolution
    if shots == 1:
        TR = max(TR, Nphase * TE + 1.0)
    TI_s = 0  # No FLAIR for pure TSE
    Excitation_FA = 90
    Refocusing_FA = 180
    r_spoil = 2
    PE_grad_on = True
    RO_grad_on = True
    dumshots = 1
    dumref = 0

    rf1, gz1, gzr1 = pp.make_sinc_pulse(
        flip_angle=Excitation_FA * np.pi / 180, phase_offset=90 * np.pi / 180, duration=1e-3,
        slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4, use='excitation',
        system=system, return_gz=True, delay=system.rf_dead_time)

    rf2, gz2, _ = pp.make_sinc_pulse(
        flip_angle=Refocusing_FA * np.pi / 180, duration=1e-3, use='refocusing',
        slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
        system=system, return_gz=True, delay=system.rf_dead_time)

    dwell = 50e-6 * 2
    G_flag = (int(RO_grad_on), int(PE_grad_on))

    gx = pp.make_trapezoid(channel='x', rise_time=0.5 * dwell, flat_area=Nread / fov * G_flag[0],
                            flat_time=Nread * dwell, system=system)
    adc = pp.make_adc(num_samples=Nread, duration=Nread * dwell, phase_offset=90 * np.pi / 180,
                      delay=system.adc_dead_time, system=system)
    gx_pre0 = pp.make_trapezoid(channel='x', area=+((1.0 + r_spoil) * gx.area / 2), duration=1.5e-3, system=system)
    gx_prewinder = pp.make_trapezoid(channel='x', area=+(r_spoil * gx.area / 2), duration=1e-3, system=system)
    gp = pp.make_trapezoid(channel='y', area=0 / fov, duration=1e-3, system=system)
    rf_prep = pp.make_block_pulse(flip_angle=180 * np.pi / 180, duration=1e-3, system=system, delay=system.rf_dead_time)

    if PE_grad_on:
        if PEtype == 'centric':
            phenc = np.asarray([i // 2 if i % 2 == 0 else -(i + 1) // 2 for i in range(Nphase)]) / fov
        else:
            phenc = np.arange(-Nphase // 2, Nphase // 2) / fov
    else:
        phenc = np.zeros((Nphase,))

    minTE2 = (pp.calc_duration(gz2) + pp.calc_duration(gx) + 2 * pp.calc_duration(gp)) / 2
    minTE2 = round(minTE2 / 10e-5) * 10e-5
    TEd = round(max(0, (TE / 2 - minTE2)) / 10e-5) * 10e-5


    if TEd == 0:
        print('echo time set to minTE [ms]', 2 * (minTE2 + TEd) * 1000)
    else:
        print(' Effective TE [ms] is ', 2 * (minTE2 + TEd) * 1000)

    if dumshots + shots > 1:
        TRd = TR - Nphase // shots * TE
    else:
        TRd = 1e-3  # safe minimum

    if TRd < 0:
        warnings.warn(f'TR too short, setting minimum delay to 1ms instead of {TRd:.2f} s')
        TRd = 1e-3

    for shot in range(-dumshots, shots):
        if TI_s > 0:
            seq.add_block(rf_prep)
            seq.add_block(pp.make_delay(TI_s))
            seq.add_block(gx_pre0)

        seq.add_block(rf1, gz1)
        seq.add_block(gx_pre0, gzr1)
        seq.add_block(pp.make_delay((minTE2 + TEd) - pp.calc_duration(gz1) - pp.calc_duration(gx_pre0)))

        if shot < 0:
            phenc_dum = np.zeros(Nphase // shots + dumref)
        else:
            phenc_dum = np.concatenate([np.repeat(np.nan, dumref), phenc[shot::shots]])

        for encoding in phenc_dum:
            dum_ref_flag = 0
            if np.isnan(encoding):
                encoding = 1e-8
                dum_ref_flag = 1

            gp = pp.make_trapezoid(channel='y', area=+encoding, duration=1e-3, system=system)
            gp_ = pp.make_trapezoid(channel='y', area=-encoding, duration=1e-3, system=system)

            seq.add_block(rf2, gz2)
            seq.add_block(pp.make_delay(TEd))
            seq.add_block(gx_prewinder, gp)

            if shot < 0 or dum_ref_flag:
                seq.add_block(gx)
            else:
                seq.add_block(adc, gx)

            seq.add_block(gx_prewinder, gp_)
            seq.add_block(pp.make_delay(TEd))

        seq.add_block(pp.make_delay(TRd))

    ok, error_report = seq.check_timing()
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]

    if plot:
        seq.plot(time_range=(TR-0.01, 2 * TR +0.01))

    if write:
        seq.write(seq_filename)
        print(f'Sequence written to {seq_filename}')

    return seq


seq = TSE_2D_seq_gen(base_resolution=128, fov=200e-3, TE=96e-3, TR=2, PEtype='centric', shots=1, plot=True, write=True, seq_filename='TSE_TF128_TE96ms.seq')
