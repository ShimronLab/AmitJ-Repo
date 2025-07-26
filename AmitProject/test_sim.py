# This code goal is to do the following:
# 1. Extract slice 49 from subject04 as test slice (centered slice so contains a lot of tissue)
# 2. Run simulation with two different sequences:
#    a.TSE_SEQ_CO_R1_TE96_TF128.seq - echo_spacing is same as TE = 96ms
#    b.TSE_SEQ_CO_R1_TE96MS_TF128_ESP16ms.seq - same parameters but echo_spacing is 16ms (6 times shorter)
# 3. Run simulation and measure the following:
#    a. Simulation time for the following threshold values: [1e-3, 1e-4,1e-5, 1e-6]
#    b. Lines acquired in k-space for the different thresholds from a.  + w & w/o shorter ESP
# 4. all results should be presented in a graph:
#    x - axis - threshold value
#    y- axis - lines acquired in kspace
#    values for both with ESP and without ESP seperated by color (connect the dot between points and sort by sequence label ESP or noESP
# Save figure of the graph in the end and also export a table as follows:
# Threshold Value | Lines acquired in k-space | Duration of simulation in seconds | ESP value|

import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import MRzeroCore as mr0
import pandas as pd

# Settings
data_dir = os.path.expanduser("~/AmitProject/data/brainweb")
subject_file = "subject04_3T.npz"
slice_idx = 49
Nx = Ny = 128
n_echo = 128
thresholds = [1e-3, 1e-4, 1e-5, 1e-6]
seq_files = {
    "ESP96ms": "TSE_SEQ_CO_R1_TE96_TF128.seq",
    "ESP12ms": "TSE_SEQ_CO_R1_TE96MS_TF128_ESP12ms_flipshift.seq"
}
output_dir = "test_subject04_slice49"
os.makedirs(output_dir, exist_ok=True)

# Load phantom and interpolate
phantom = mr0.VoxelGridPhantom.brainweb(os.path.join(data_dir, subject_file))
phantom = phantom.interpolate(Nx, Ny, phantom.PD.shape[2])
phantom_slice = phantom.slices([slice_idx])

# Simulation runner
def run_simulation(phantom_slice, seq_file, threshold):
    seq = mr0.Sequence.import_file(seq_file)
    data = phantom_slice.build()

    start = time.time()
    graph = mr0.compute_graph(seq, data, 200, threshold)
    signal, _ = mr0.execute_graph(graph, seq, data, return_mag_adc=True,
                                  min_emitted_signal=threshold, min_latent_signal=threshold)
    duration = time.time() - start

    total_adc = signal.numel()
    total_adc = signal.numel()
    if total_adc % Nx != 0:
        raise ValueError("Signal length is not divisible by Nx — unexpected format")

    # Reshape to (n_ex, n_echo, Nx)
    reshaped = signal.view(-1, Nx)

    # Count how many k-space lines have non-zero signal
    magnitudes = reshaped.abs().sum(dim=1)  # (n_lines,)
    lines_acquired = (magnitudes > 0).sum().item()  # actual line count
    return lines_acquired, duration

# Store results
results = []

for esp_label, seq_file in seq_files.items():
    full_seq_path = os.path.join("~/AmitProject", seq_file)
    full_seq_path = os.path.expanduser(full_seq_path)

    for thresh in thresholds:
        print(f"Running {esp_label} with threshold={thresh}")
        lines, duration = run_simulation(phantom_slice, full_seq_path, thresh)
        results.append({
            "Threshold Value": thresh,
            "Lines Acquired in k-space": lines,
            "Duration of simulation in seconds": duration,
            "ESP value": "96ms" if esp_label == "ESP96ms" else "16ms"
        })

# Save table
df = pd.DataFrame(results)
df["Threshold Value"] = df["Threshold Value"].apply(lambda x: f"{x:.0e}")
df["Duration of simulation in seconds"] = df["Duration of simulation in seconds"].apply(lambda x: f"{x:.3f}")
csv_path = os.path.join(output_dir, "simulation_results_table.csv")
df.to_csv(csv_path, index=False)

# Plot
plt.figure(figsize=(8, 6))
for esp in ["96ms", "16ms"]:
    sub_df = df[df["ESP value"] == esp]
    plt.plot(sub_df["Threshold Value"].astype(float), sub_df["Lines Acquired in k-space"], 'o-', label=f'ESP={esp}')

plt.xscale("log")
plt.xlabel("Threshold Value")
plt.ylabel("Lines Acquired in k-space")
plt.title("Lines Acquired vs Threshold (slice 49, subject04)")
plt.grid(True)
plt.xticks([1e-3, 1e-4, 1e-5, 1e-6], ['1e-3', '1e-4', '1e-5', '1e-6'])
plt.yticks(list(range(10, 129, 10)) + [128])  # 10 to 120 + 128
plt.legend()
plt.tight_layout()

plot_path = os.path.join(output_dir, "lines_vs_threshold.png")
plt.savefig(plot_path)
plt.close()

print(f"Saved results to {csv_path} and plot to {plot_path}")
