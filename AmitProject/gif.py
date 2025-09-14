import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_k_space_animation_subplot():
    k_space = np.tile(np.arange(1, 17).reshape(16, 1), (1, 16))
    pastel_colors = ['#AEC6CF', '#FFB347', '#F8C8DC', '#77DD77']  # gray, orange, pink, green

    def build_center_out_pe_order():
        center_rows = [7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2, 13, 1, 14, 0, 15]
        pe_order = []
        for i in range(4):
            pe_order.append(center_rows[i*4:(i+1)*4])
        return pe_order

    # Define all 4 combinations
    configs = [
        (False, False, "Vertical Top-Down"),
        (False, True, "Vertical Center-Out"),
        (True, False, "Horizontal Top-Down"),
        (True, True, "Horizontal Center-Out"),
    ]

    # Precompute reading sequences for each config
    sequences = []
    for horizontal, center_out, _ in configs:
        if center_out:
            pe_order = build_center_out_pe_order()
        else:
            pe_order = [np.arange(i, 16, 4) for i in range(4)]

        reading_sequence = []
        for color_idx, rows in enumerate(pe_order):
            for r in rows:
                for c in range(16):
                    if horizontal:
                        reading_sequence.append((c, r, color_idx))
                    else:
                        reading_sequence.append((r, c, color_idx))
        sequences.append(reading_sequence)

    # Set up figure with 1 row and 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    masks = [np.full((16, 16), -1) for _ in range(4)]

    def update(frame_idx):
        for ax, (reading_sequence, mask, (_, _, title)) in zip(axes, zip(sequences, masks, configs)):
            ax.clear()
            ax.axis('off')
            ax.set_title(title, fontsize=12)
            if frame_idx < len(reading_sequence):
                r, c, color_idx = reading_sequence[frame_idx]
                mask[r, c] = color_idx
            for (i, j), val in np.ndenumerate(k_space):
                if mask[i, j] >= 0:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         color=pastel_colors[mask[i, j]], alpha=0.4)
                    ax.add_patch(rect)
                ax.text(j, i, f"{val}", va='center', ha='center', fontsize=8)
            ax.set_xlim(-0.5, 15.5)
            ax.set_ylim(15.5, -0.5)

    # Max frames = longest sequence
    max_frames = max(len(seq) for seq in sequences)

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=100, repeat=True)

    ani.save("gifs.gif", writer="pillow", fps=10)
    print("Saved: gifs.gif")

generate_k_space_animation_subplot()
