import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Generate a 16x16 matrix where each row has its own index (1 to 16)
k_space = np.tile(np.arange(1, 17).reshape(16, 1), (1, 16))

# 2. Pastel colors for each shot
pastel_colors = ['#AEC6CF', '#FFB347', '#B39EB5', '#77DD77']

# 3. Define PE order
pe_order = [np.arange(i, 16, 4) for i in range(4)]

# 4. Build reading order: (row, col, color_index)
reading_sequence = []
for color_idx, rows in enumerate(pe_order):
    for r in rows:
        for c in range(16):  # Read left to right
            reading_sequence.append((r, c, color_idx))

# 5. Set up figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.axis('off')

# 6. Create a mask to store which pixels were read
mask = np.full((16, 16), fill_value=-1)  # -1 means not read yet

def update(frame_idx):
    ax.clear()
    ax.axis('off')

    # Update mask
    r, c, color_idx = reading_sequence[frame_idx]
    mask[r, c] = color_idx

    # Plot numbers and background
    for (i, j), val in np.ndenumerate(k_space):
        if mask[i, j] >= 0:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 color=pastel_colors[mask[i, j]], alpha=0.4)
            ax.add_patch(rect)
        ax.text(j, i, f"{val}", va='center', ha='center', fontsize=10)

    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(15.5, -0.5)

# 7. Create animation
ani = animation.FuncAnimation(fig, update, frames=len(reading_sequence),
                              interval=100, repeat=True)

# 8. Save GIF
ani.save('k_space_pixel_reading.gif', writer='pillow', fps=10)

# 9. Save final frame
fig_final, ax_final = plt.subplots(figsize=(6, 6))
ax_final.axis('off')

# Fill the final mask fully
final_mask = np.full((16, 16), fill_value=-1)
for idx in range(len(reading_sequence)):
    r, c, color_idx = reading_sequence[idx]
    final_mask[r, c] = color_idx

for (i, j), val in np.ndenumerate(k_space):
    if final_mask[i, j] >= 0:
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             color=pastel_colors[final_mask[i, j]], alpha=0.4)
        ax_final.add_patch(rect)
    ax_final.text(j, i, f"{val}", va='center', ha='center', fontsize=10)

ax_final.set_xlim(-0.5, 15.5)
ax_final.set_ylim(15.5, -0.5)
plt.savefig('k_space_final_pixel_reading.png', bbox_inches='tight')

print("Saved:")
print("- GIF animation: k_space_pixel_reading.gif")
print("- Final static image: k_space_final_pixel_reading.png")
