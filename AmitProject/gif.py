import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_k_space_animation(horizontal=False, center_out=False):
    k_space = np.tile(np.arange(1, 17).reshape(16, 1), (1, 16))
    pastel_colors = ['#AEC6CF', '#FFB347', '#F8C8DC', '#77DD77'] # gray, orange, pink, green

    def build_center_out_pe_order():
        center_rows = [7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2, 13, 1, 14, 0, 15]  # manual center-out order
        pe_order = []
        for i in range(4):
            pe_order.append(center_rows[i*4:(i+1)*4])
        return pe_order

    # Define PE order
    if center_out:
        pe_order = build_center_out_pe_order()
    else:
        pe_order = [np.arange(i, 16, 4) for i in range(4)]

    # Reading order: assign a distinct color index to each shot
    reading_sequence = []
    for color_idx, rows in enumerate(pe_order):
        for r in rows:
            for c in range(16):
                if horizontal:
                    reading_sequence.append((c, r, color_idx))  # horizontal: read along rows
                else:
                    reading_sequence.append((r, c, color_idx))  # vertical: read along columns

    # Set up figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    mask = np.full((16, 16), -1)

    def update(frame_idx):
        ax.clear()
        ax.axis('off')
        r, c, color_idx = reading_sequence[frame_idx]
        mask[r, c] = color_idx
        for (i, j), val in np.ndenumerate(k_space):
            if mask[i, j] >= 0:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     color=pastel_colors[mask[i, j]], alpha=0.4)
                ax.add_patch(rect)
            ax.text(j, i, f"{val}", va='center', ha='center', fontsize=10)
        ax.set_xlim(-0.5, 15.5)
        ax.set_ylim(15.5, -0.5)

    # File name suffix
    suffix = f"{'horizontal' if horizontal else 'vertical'}_{'center_out' if center_out else 'top_down'}"
    gif_filename = f'k_space_pixel_reading_{suffix}.gif'

    # Animate
    ani = animation.FuncAnimation(fig, update, frames=len(reading_sequence),
                                  interval=100, repeat=True)
    ani.save(gif_filename, writer='pillow', fps=10)

    # Final static image
    fig_final, ax_final = plt.subplots(figsize=(6, 6))
    ax_final.axis('off')
    final_mask = np.full((16, 16), -1)
    for r, c, color_idx in reading_sequence:
        final_mask[r, c] = color_idx
    for (i, j), val in np.ndenumerate(k_space):
        if final_mask[i, j] >= 0:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 color=pastel_colors[final_mask[i, j]], alpha=0.4)
            ax_final.add_patch(rect)
        ax_final.text(j, i, f"{val}", va='center', ha='center', fontsize=10)
    ax_final.set_xlim(-0.5, 15.5)
    ax_final.set_ylim(15.5, -0.5)

    final_image_filename = f'k_space_final_pixel_reading_{suffix}.png'
    plt.savefig(final_image_filename, bbox_inches='tight')

    print(f"Saved:\n- {gif_filename}\n- {final_image_filename}")

# Generate all 4 combinations
for horizontal in [False, True]:
    for center_out in [False, True]:
        generate_k_space_animation(horizontal, center_out)