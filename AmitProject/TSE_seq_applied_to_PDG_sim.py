import math
import warnings
import numpy as np
import pypulseq as pp

def TSE_seq(plot: bool = False, write_seq: bool = False, seq_filename: str = 'TSE_SEQ.seq',n_echo=16, Ny=64, Nx=6,TE = 12e-3,fov=256e-3,pe_order_label='TD',is_horizontal_pe = False):
#def main(plot: bool = False, write_seq: bool = False, seq_filename: str = 'tse_pypulseq.seq'):
    # ======
    # SETUP
    # ======
    dG = 250e-6

    # Set system limits
    system = pp.Opts(
        max_grad=32,
        grad_unit='mT/m',
        max_slew=130,
        slew_unit='T/m/s',
        rf_ringdown_time=100e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )

    if is_horizontal_pe:
        fe = 'y'
        pe = 'x'
    else:
        fe = 'x'
        pe = 'y'

    seq = pp.Sequence(system)  # Create a new sequence object
    n_slices = 1
    rf_flip = 180  # Flip angle
    if isinstance(rf_flip, int):
        rf_flip = np.zeros(n_echo) + rf_flip
    slice_thickness = 5e-3
    TR = 2000e-3  # Repetition time

    sampling_time = 6.4e-3
    readout_time = sampling_time + 2 * system.adc_dead_time
    t_ex = 2.5e-3
    t_exwd = t_ex + system.rf_ringdown_time + system.rf_dead_time
    t_ref = 2e-3
    t_refwd = t_ref + system.rf_ringdown_time + system.rf_dead_time
    t_sp = 0.5 * (TE - readout_time - t_refwd)
    t_spex = 0.5 * (TE - t_exwd - t_refwd)
    fsp_r = 1
    fsp_s = 0.5

    rf_ex_phase = np.pi / 2
    rf_ref_phase = 0

    # ======
    # CREATE EVENTS
    # ======
    flip_ex = 90 * np.pi / 180
    rf_ex, gz, _ = pp.make_sinc_pulse(
        flip_angle=flip_ex,
        system=system,
        duration=t_ex,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        phase_offset=rf_ex_phase,
        return_gz=True,
        delay=system.rf_dead_time,
    )
    gs_ex = pp.make_trapezoid( # rewinder gradient applied after the excitation RF pulse
        channel='z',
        system=system,
        amplitude=gz.amplitude,
        flat_time=t_exwd,
        rise_time=dG,
    )

    flip_ref = rf_flip[0] * np.pi / 180
    rf_ref, gz, _ = pp.make_sinc_pulse(
        flip_angle=flip_ref,
        system=system,
        duration=t_ref,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        phase_offset=rf_ref_phase,
        use='refocusing',
        return_gz=True,
        delay=system.rf_dead_time,
    )
    gs_ref = pp.make_trapezoid( # slice select during the refocusing pulse
        channel='z',
        system=system,
        amplitude=gs_ex.amplitude,
        flat_time=t_refwd,
        rise_time=dG,
    )

    ags_ex = gs_ex.area / 2
    gs_spr = pp.make_trapezoid(
        channel='z',
        system=system,
        area=ags_ex * (1 + fsp_s),
        duration=t_sp,
        rise_time=dG,
    )
    gs_spex = pp.make_trapezoid(channel='z', system=system, area=ags_ex * fsp_s, duration=t_spex, rise_time=dG)

    delta_k = 1 / fov
    k_width = Nx * delta_k

    gr_acq = pp.make_trapezoid(
        channel=fe,
        system=system,
        flat_area=k_width,
        flat_time=readout_time,
        rise_time=dG,
    )

    adc = pp.make_adc(num_samples=Nx, duration=sampling_time, delay=system.adc_dead_time, system=system)
    gr_spr = pp.make_trapezoid(
        channel=fe,
        system=system,
        area=gr_acq.area * fsp_r, #k_width
        duration=t_sp,
        rise_time=dG,
    )

    agr_spr = gr_spr.area
    agr_preph = gr_acq.area / 2 + agr_spr
    gr_preph = pp.make_trapezoid(channel=fe, system=system, area=agr_preph, duration=t_spex, rise_time=dG)

    # # Phase-encoding

    n_ex = math.floor(Ny / n_echo)

    # original
    # pe_steps = np.arange(1, n_echo * n_ex + 1) - 0.5 * n_echo * n_ex - 1
    # if divmod(n_echo, 2)[1] == 0:
    #     pe_steps = np.roll(pe_steps, [0, int(-np.round(n_ex / 2))])
    # pe_order = pe_steps.reshape((n_ex, n_echo), order='F').T

    if pe_order_label=='TD': # Top-Down order
        pe_steps = np.linspace(-n_ex * n_echo // 2, n_ex * n_echo // 2 - 1, n_ex * n_echo, dtype=int)
        pe_order = pe_steps.reshape((n_echo, n_ex), order='C')
    elif pe_order_label=='CO': # Center-Out order
        center = n_echo * n_ex // 2
        offsets = np.arange(center)
        pe_steps = np.empty(n_echo * n_ex, dtype=int)
        pe_steps[::2] = center + offsets  # even indices: 0, +1, +2, ...
        pe_steps[1::2] = center - offsets - 1  # odd indices: -1, -2, -3, ...
        pe_steps = pe_steps - center  # shift to be centered around 0

        pe_order = pe_steps.reshape((n_echo, n_ex), order='C')

    phase_areas = pe_order * delta_k

    # Split gradients and recombine into blocks
    gs1_times = np.array([0, gs_ex.rise_time])
    gs1_amp = np.array([0, gs_ex.amplitude])
    gs1 = pp.make_extended_trapezoid(channel='z', times=gs1_times, amplitudes=gs1_amp)

    gs2_times = np.array([0, gs_ex.flat_time])
    gs2_amp = np.array([gs_ex.amplitude, gs_ex.amplitude])
    gs2 = pp.make_extended_trapezoid(channel='z', times=gs2_times, amplitudes=gs2_amp)

    # gs1 - ramp-up segment - gradient that increases from 0 to full amplitude
    # gs2 - flat-top segment - gradient that holds constant amplitude
    # Together we get decomposed  gs_ex - this allows flexible reassembly and timing control

    gs3_times = np.array(
        [
            0,
            gs_spex.rise_time,
            gs_spex.rise_time + gs_spex.flat_time,
            gs_spex.rise_time + gs_spex.flat_time + gs_spex.fall_time,
        ]
    )
    gs3_amp = np.array([gs_ex.amplitude, gs_spex.amplitude, gs_spex.amplitude, gs_ref.amplitude])
    gs3 = pp.make_extended_trapezoid(channel='z', times=gs3_times, amplitudes=gs3_amp)

    gs4_times = np.array([0, gs_ref.flat_time])
    gs4_amp = np.array([gs_ref.amplitude, gs_ref.amplitude])
    gs4 = pp.make_extended_trapezoid(channel='z', times=gs4_times, amplitudes=gs4_amp)

    # Gradient | Duration                      | Function
    # gs3      | Rise + flat + fall of gs_spex | Smoothly transition from rewinder to gs_ref
    # gs4      | Flat - top duration of gs_ref | Slice - select gradient during 180° RF pulse

    gs5_times = np.array(
        [
            0,
            gs_spr.rise_time,
            gs_spr.rise_time + gs_spr.flat_time,
            gs_spr.rise_time + gs_spr.flat_time + gs_spr.fall_time,
        ]
    )
    gs5_amp = np.array([gs_ref.amplitude, gs_spr.amplitude, gs_spr.amplitude, 0])
    gs5 = pp.make_extended_trapezoid(channel='z', times=gs5_times, amplitudes=gs5_amp)

    gs7_times = np.array(
        [
            0,
            gs_spr.rise_time,
            gs_spr.rise_time + gs_spr.flat_time,
            gs_spr.rise_time + gs_spr.flat_time + gs_spr.fall_time,
        ]
    )
    gs7_amp = np.array([0, gs_spr.amplitude, gs_spr.amplitude, gs_ref.amplitude])
    gs7 = pp.make_extended_trapezoid(channel='z', times=gs7_times, amplitudes=gs7_amp)

    # Gradient | Role                                         | Connects From → To
    # gs5      | Smoothly exits 180° refocusing → spoiler → 0 | gs_ref → gs_spr → 0
    # gs7      | Prepares next RF: ramps 0 → spoiler → gs_ref | 0 → gs_spr → gs_ref

    # Readout gradient
    gr3 = gr_preph #readout pre-phasing gradient

    # Variable | Purpose | What it connects
    # gr3 | Readout pre-phaser (to center ADC) | Initial shift to -k/2

    gr5_times = np.array(
        [
            0,
            gr_spr.rise_time,
            gr_spr.rise_time + gr_spr.flat_time,
            gr_spr.rise_time + gr_spr.flat_time + gr_spr.fall_time,
        ]
    )
    gr5_amp = np.array([0, gr_spr.amplitude, gr_spr.amplitude, gr_acq.amplitude])
    gr5 = pp.make_extended_trapezoid(channel=fe, times=gr5_times, amplitudes=gr5_amp)

    # Variable | Purpose | What it connects
    # gr5 | Transition from spoiler → readout | 0 → gr_spr → gr_acq



    gr6_times = np.array([0, readout_time])
    gr6_amp = np.array([gr_acq.amplitude, gr_acq.amplitude])
    gr6 = pp.make_extended_trapezoid(channel=fe, times=gr6_times, amplitudes=gr6_amp)

    gr7_times = np.array(
        [
            0,
            gr_spr.rise_time,
            gr_spr.rise_time + gr_spr.flat_time,
            gr_spr.rise_time + gr_spr.flat_time + gr_spr.fall_time,
        ]
    )
    gr7_amp = np.array([gr_acq.amplitude, gr_spr.amplitude, gr_spr.amplitude, 0])
    gr7 = pp.make_extended_trapezoid(channel=fe, times=gr7_times, amplitudes=gr7_amp)

    # Gradient | Role | Connects From → To
    # gr6 | Main readout gradient | Applied during ADC
    # gr7 | Transition from readout → spoiler → rest | Smoothly decays readout into spoil/recover

    # Fill-times
    t_ex = pp.calc_duration(gs1) + pp.calc_duration(gs2) + pp.calc_duration(gs3)
    t_ref = pp.calc_duration(gs4) + pp.calc_duration(gs5) + pp.calc_duration(gs7) + readout_time
    t_end = pp.calc_duration(gs4) + pp.calc_duration(gs5)

    TE_train = t_ex + n_echo * t_ref + t_end
    TR_fill = (TR - n_slices * TE_train) / n_slices
    # Round to gradient raster
    TR_fill = system.grad_raster_time * np.round(TR_fill / system.grad_raster_time)
    if TR_fill < 0:
        TR_fill = 1e-3
        warnings.warn(f'TR too short, adapted to include all slices to: {1000 * n_slices * (TE_train + TR_fill)} ms')
    else:
        print(f'TR fill: {1000 * TR_fill} ms')
    delay_TR = pp.make_delay(TR_fill)

    # ======
    # CONSTRUCT SEQUENCE
    # ======
    for k_ex in range(n_ex+1):
        for s in range(n_slices):
            rf_ex.freq_offset = gs_ex.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
            rf_ref.freq_offset = gs_ref.amplitude * slice_thickness * (s - (n_slices - 1) / 2)
            rf_ex.phase_offset = rf_ex_phase - 2 * np.pi * rf_ex.freq_offset * pp.calc_rf_center(rf_ex)[0]
            rf_ref.phase_offset = rf_ref_phase - 2 * np.pi * rf_ref.freq_offset * pp.calc_rf_center(rf_ref)[0]

            seq.add_block(gs1)  # ramp-up gradient (beginning of excitation)
            seq.add_block(rf_ex, gs2)  # 90° RF pulse and flat-top gradient
            seq.add_block(gs3, gr3)  # transition to refocusing and readout prephaser

            for k_echo in range(n_echo):
                if k_ex > 0: # if k_ex==0 it is a dummy shot with no acquisition e.g T2 stabilization
                    phase_area = phase_areas[k_echo, k_ex - 1]
                else:
                    phase_area = 0.0  # 0.0 and not 0 because -phase_area should successfully result in negative zero

                gp_pre = pp.make_trapezoid( # Pre-phasing gradient (before readout)
                    channel=pe,
                    system=system,
                    area=phase_area,
                    duration=t_sp,
                    rise_time=dG,
                )
                gp_rew = pp.make_trapezoid( # Rewinder gradient (after readout) opposite sign because it cancels the phase shift
                    channel=pe,
                    system=system,
                    area=-phase_area,
                    duration=t_sp,
                    rise_time=dG,
                )
                seq.add_block(rf_ref, gs4) # 180° RF + slice-select flat-top
                seq.add_block(gr5, gp_pre, gs5) # transition into readout, phase encoding, ramp-down slice

                if k_ex > 0:
                    seq.add_block(gr6, adc) # real readout
                else:
                    seq.add_block(gr6) # dummy readout (no ADC)

                seq.add_block(gr7, gp_rew, gs7)  # rewind gradients, transition to next echo

            seq.add_block(gs4)  # one last 180° slice-select
            seq.add_block(gs5)  # ramp out

            seq.add_block(delay_TR)  # fill rest of TR

    (ok, error_report, ) = seq.check_timing()  # Check whether the timing of the sequence is correct
    # Confirms there are no timing overlaps or gaps
    # Verifies that all blocks are placed properly on the time grid
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]

    # ======
    # VISUALIZATION
    # ======
    if plot:
        # seq.plot(time_range=(1.9, 2.3))
        seq.plot()

    # =========
    # WRITE .SEQ
    # =========
    if write_seq:
        seq.write(seq_filename)

    return seq


if __name__ == '__main__':
    TSE_seq(plot=True, write_seq=False, seq_filename='TSE_SEQ_test_TD_new.seq',n_echo=32, Ny=128, Nx=128,TE = 12e-3,fov=256e-3,pe_order_label='TD',is_horizontal_pe = False)
