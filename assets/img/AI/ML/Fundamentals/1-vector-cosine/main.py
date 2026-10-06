import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc
import os

# Data
data = {
    'cat': (0.9, 0.1),
    'kitten': (0.95, 0.05),
    'dog': (0.85, 0.15),
    'airplane': (0.05, 0.95),
    'helicopter': (0.1, 0.9)
}

fig, ax = plt.subplots(figsize=(8, 8))

# Plot vectors using quiver
origins_x = [0] * len(data)
origins_y = [0] * len(data)
x_coords = [pos[0] for pos in data.values()]
y_coords = [pos[1] for pos in data.values()]
labels = list(data.keys())

# Colors for animal vs machine
colors = ['#1f77b4' if pos[0] > pos[1] else '#ff7f0e' for pos in data.values()]

ax.quiver(origins_x, origins_y, x_coords, y_coords,
          angles='xy', scale_units='xy', scale=1,
          color=colors, width=0.006, headwidth=4, headlength=5)

# Annotations
for label, (x, y) in data.items():
    offset_x = 0.02
    offset_y = 0.02
    if label in ['airplane', 'helicopter']:
        offset_x = 0.02
        offset_y = -0.01
    ax.text(x + offset_x, y + offset_y, label, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"))

# Calculate angles for "cat" and "airplane"
angle_cat = np.degrees(np.arctan2(data['cat'][1], data['cat'][0]))
angle_airplane = np.degrees(np.arctan2(data['airplane'][1], data['airplane'][0]))

# Add Arc for angle between cat and airplane
arc_radius = 0.3
arc = Arc((0, 0), arc_radius * 2, arc_radius * 2, angle=0,
          theta1=angle_cat, theta2=angle_airplane,
          color='gray', linestyle='--', linewidth=1.5)
ax.add_patch(arc)

# Label angle theta
theta_mid = np.radians((angle_cat + angle_airplane) / 2)
ax.text(0.35 * np.cos(theta_mid), 0.35 * np.sin(theta_mid), r'$\theta$',
        fontsize=14, color='gray', ha='center', va='center', fontweight='bold')

# Configure axes and limits
ax.set_xlim(-0.05, 1.1)
ax.set_ylim(-0.05, 1.1)
ax.set_xlabel('Axis 1: Animal-like (x)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('Axis 2: Mechanical (y)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('2D Toy Word Embeddings Space', fontsize=14, fontweight='bold', pad=15)

# Grid and origin axes
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()
