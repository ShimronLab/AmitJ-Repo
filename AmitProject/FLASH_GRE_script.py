import numpy as np
# newer numpy versions don't contain this, but pypulseq still relies on it
np.int = int
np.float = float
np.complex = complex

import pypulseq as pp
import MRzeroCore as mr0
import torch
import matplotlib.pyplot as plt

experiment_id = "flash"

# sys stores and define various system parameters that describe the MRI hardware and its capabilities
# This object helps to configure the MRI system parameters that are essential for defining the pulse sequence,
# including gradient strength, RF pulse properties, and timing characteristics.

# These parameters are used throughout the sequence to ensure that the generated sequence adheres
# to the physical limits of the MRI system.
sys = pp.Opts(
    max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
    rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=20e-6,
    grad_raster_time=50e-6
) #explantion about each system parameter is described in Terms.txt

n_read = 64 # Defines the number of samples to be collected in the readout direction
n_phase = 64 # Defines the number of phase encoding steps (samples in the phase encoding direction
fov = 192e-3 # Defines the field of view (FOV) for the scan (in meters). defines the physical area that is being imaged
slice_thickness = 8e-3 # Defines the thickness of the slice to be imaged. in meters

rf = pp.make_sinc_pulse( # Creating RF pulse
    flip_angle=5 * np.pi/180, duration=1e-3,
    slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
    system=sys, return_gz=False
)# Apodization is set to control the RF pulse's frequency characteristics. ranges between 0 and 1 so 0.5 is moderate smoothing
# time_bw_product defines the time-bandwidth product of the sinc pulse. It determines the trade-off between pulse duration and bandwidth.
# A value of 4 is commonly used in MRI sequences.
# return_gz - if set to true it will return the gradient along z-axis in this case it is false (according to the script)

# Readout gradient
gx = pp.make_trapezoid('x', flat_area=n_read / fov, flat_time=n_read*50e-6, system=sys)
# This line creates a trapezoidal gradient in the x-direction (readout direction)
# The flat area of the trapezoidal gradient, which is the area under the gradient waveform.
# It is calculated as the number of readout samples divided by the field of view
# flat_time is the duration of the flat part of the gradient. set to n_red times 50 microseconds.

adc = pp.make_adc( # Creates the Analog to Digital Converter which converts the analog signal during the readout into a digital signal
    num_samples=n_read, dwell=50e-6, delay=gx.rise_time,
    system=sys
)
# The ADC collects n_read samples and dwell is the time between every two samples which is set to 50 microseconds (sampling period)
# delay is set before the ADC starts sampling and it corresponds to the rise time of the readout gradient

# Rewinder before gx and spoiler afterwards
gx_pre = pp.make_trapezoid('x', area=-0.5*gx.area, duration=5e-3, system=sys)
gx_spoil = pp.make_trapezoid('x', area=1.5*gx.area, duration=2e-3, system=sys)

# gx_pre is a pre-readout gradient, which is applied before the main readout gradient g. it is used for preparation of the signal
# area of the gradient is set to half the area of the readout gradient but with opposite polarity meaning its an inverted gradient (upside-down trapezoid)
# the duration of gx_pre is 5 ms

# gx_spoil is a spoiler gradient applied after the readout gradient. It is used to dephase any remaining transverse magnetization that might interfere with future scans.
# The area is set to 1.5 times the area of the readout gradient but with the opposite polarity
# The duration of gx_spoil is 2 ms

# Construct the sequence

seq = pp.Sequence() # Creating the sequence by initializing an empty Sequence object (list of MRI events RF pulses, gradients, ADCs, delays, etc.)

for i in range(-n_phase//2, n_phase//2):
    # Apply phase encoding in y-direction (gy) for each phase encoding step starting from -32 to +32
    # The loop goes over each of the n_phase phase encoding steps and creates the necessary gradient and RF events

    # RF phase spoiling
    #  defines the phase angle by which the RF pulse and ADC will be shifted. The phase of an RF pulse or an ADC signal determines where along the signal's oscillation it starts.
    rf.phase_offset = (0.5 * (i**2+i+2) * 117) % 360 * np.pi / 180
    adc.phase_offset = rf.phase_offset
    seq.add_block(rf) # adds RF pulse to the sequence. The RF pulse will be applied at the current phase offset
    # Phase encoding
    gy = pp.make_trapezoid('y', area=i / fov, duration=5e-3, system=sys) # creates a trapezoidal gradient in the y-direction for phase encoding.
    # The area of the gradient is proportional to the current phase encoding index i divided by the FOV.
    seq.add_block(gx_pre, gy) # Allows simultaneous operations during the block but in different directions (x-y)
    seq.add_block(adc, gx)
    # Rewind phase and spoil
    gy = pp.make_trapezoid('y', area=-i / fov, duration=5e-3, system=sys)
    # After the readout, the phase encoding gradient (gy) is applied again with the opposite polarity (-i / fov),
    # followed by the spoiler gradient (gx_spoil), which dephases any residual magnetization.
    seq.add_block(gx_spoil, gy)
    seq.add_block(pp.make_delay(1e-3))
    # A delay of 1 millisecond is added to the sequence, allowing the system to relax before the next pulse sequence begins

ok, error_report = seq.check_timing() # This method checks the timing of the pulse sequence.
# It ensures that the sequence meets the timing constraints of the MRI system and doesn't violate any hardware limits
# (such as maximum gradient slew rate, RF pulse duration, or ADC sampling rates).
if ok:
    print("Timing check passed successfully")
else:
    print("Timing check failed:")
    [print(e, end="") for e in error_report]

seq.plot() # Generates a visual representation of the pulse sequence (seq) using matplotlib (or other plotting libraries).

seq.set_definition("FOV", [fov, fov, slice_thickness]) # Sets the Field of View (FOV) definition for the sequence
seq.set_definition("Name", experiment_id) # Sets a "Name" definition for the sequence. in this case flash
seq.write(experiment_id + ".seq") # Writes the pulse sequence to a .seq file.

#phantom = mr0.VoxelGridPhantom.brainweb("subject05.npz")
#filepath = "/mnt/c/Users/Amit/OneDrive - Technion/תואר שני/GitHubRepos/Project/AmitJ-Repo/output/brainweb/subject04_3T.npz"
filepath = "output/brainweb/subject04_3T.npz"

phantom = mr0.VoxelGridPhantom.load(filepath)
phantom = phantom.interpolate(64, 64, 32).slices([16]) #adjusts to the desired size of 64x64
# .slices extracts the 16th slice along z-direction
phantom.plot()
data = phantom.build() # This method builds the data for the phantom

seq = mr0.Sequence.import_file(experiment_id + ".seq") # Imports a pulse sequence from a .seq file into the mr0.Sequence object
seq.plot_kspace_trajectory() # plots the k-space trajectory of the pulse sequence



graph = mr0.compute_graph(seq, data, 200, 1e-3) #  Computes the graph for the MRI simulation process
# creates a computation graph based on the provided sequence (seq), phantom data (data), and certain simulation parameters.
# The computation graph describes the flow of data and operations required to simulate the MRI scan.

signal = mr0.execute_graph(graph, seq, data, print_progress=False) # executes the computation graph (graph) created in the previous step
# runs the simulation based on the graph, pulse sequence, and phantom data, producing the simulated MRI signal.

kspace = signal.view(n_phase, n_read) # reshapes the signal into a 2D array (matrix) with size: (n_phase,n_read)
reco = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(kspace)))

# plt.figure()
# plt.imshow(reco.abs(), origin="lower")
# plt.show()

plt.figure(figsize=(12, 6))

# Plot k-space in the first subplot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, plot in the first position
plt.imshow(kspace.abs(), origin="lower", cmap='gray')
plt.title('k-space')
plt.colorbar()

# Perform 2D Fourier Transform to reconstruct the image
reco = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(kspace)))

# Plot the reconstructed image in the second subplot
plt.subplot(1, 2, 2)  # 1 row, 2 columns, plot in the second position
plt.imshow(reco.abs(), origin="lower",cmap = 'gray')
plt.title('reco')
plt.title('Reconstructed Image')
plt.colorbar()

# Show the plot
plt.tight_layout()
plt.show()