#!/usr/bin/env python3
"""
Simple visualization of radiative heat transfer line-of-sight connections.
This version creates matplotlib figures.
"""

from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def visualize_radiation_connections():
  """Visualize the line-of-sight connections using matplotlib."""

  # Test data from case_23 (starting node at (2,3))
  result_23 = np.array([
      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
      [-1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1],
      [-1, -2, -3, -67, -34, -34, -34, -34, -34, -3, -2, -1],
      [-1, -2, -33, 0, 0, 0, 0, 0, 0, -33, -2, -1],
      [-1, -2, -33, 0, 0, 0, 0, 0, 0, -33, -2, -1],
      [-1, -2, -33, 0, 0, 0, 0, 0, 0, -33, -2, -1],
      [-1, -2, -33, 0, 0, -33, -33, -33, 0, -33, -2, -1],
      [-1, -2, -33, 0, 0, -33, 0, -34, 0, -33, -2, -1],
      [-1, -2, -33, 0, 0, -33, 0, -34, 0, -34, -2, -1],
      [-1, -2, -33, 0, 0, -33, 0, -34, 0, -34, -2, -1],
      [-1, -2, -33, 0, 0, -33, 0, 0, 0, -34, -2, -1],
      [-1, -2, -3, -33, -33, -3, -34, -34, -34, -3, -2, -1],
      [-1, -2, -3, 0, -3, 0, 0, 0, 0, -3, -2, -1],
      [-1, -2, -3, 0, -3, 0, 0, 0, 0, -3, -2, -1],
      [-1, -2, -3, -3, -3, 0, 0, 0, 0, -3, -2, -1],
      [-1, -2, -3, 0, 0, 0, 0, 0, 0, -3, -2, -1],
      [-1, -2, -3, 0, 0, 0, 0, 0, 0, -3, -2, -1],
      [-1, -2, -3, 0, 0, 0, 0, 0, 0, -3, -2, -1],
      [-1, -2, -3, 0, 0, 0, 0, 0, 0, -3, -2, -1],
      [-1, -2, -3, 0, 0, 0, 0, 0, 0, -3, -2, -1],
      [-1, -2, -3, -3, -3, -3, -3, -3, -3, -3, -2, -1],
      [-1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1],
      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  ])

  # Create figure
  fig, ax = plt.subplots(1, 1, figsize=(12, 10), layout='constrained')

  # Define colors for different cell types
  colors = {
      -1: '#000000',  # Exterior space (black)
      -2: '#8B4513',  # Exterior walls (brown)
      -3: '#A0A0A0',  # Interior walls (gray)
      0: '#FFFFFF',  # Air spaces (white)
      -33: '#00FF00',  # Directly seeing nodes (bright green)
      -34: '#FF6B6B',  # Blocked nodes (red)
      -67: '#FFD700',  # Starting node (gold)
  }

  # Create colormap
  unique_values = np.unique(result_23)
  color_list = [colors.get(val, '#CCCCCC') for val in unique_values]
  cmap = ListedColormap(color_list)

  # Plot the floor plan
  im = ax.imshow(result_23, cmap=cmap, aspect='equal')

  # Add grid
  ax.set_xticks(np.arange(-0.5, result_23.shape[1], 1), minor=True)
  ax.set_yticks(np.arange(-0.5, result_23.shape[0], 1), minor=True)
  ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

  # Add cell values as text
  for i in range(result_23.shape[0]):
    for j in range(result_23.shape[1]):
      value = result_23[i, j]
      if value != 0:  # Don't show text for air spaces
        ax.text(
            j,
            i,
            str(value),
            ha='center',
            va='center',
            fontsize=8,
            weight='bold' if value in [-67, -33, -34] else 'normal',
        )

  # Draw connections from starting node to seeing nodes
  start_pos = None
  seeing_nodes = []

  for i in range(result_23.shape[0]):
    for j in range(result_23.shape[1]):
      if result_23[i, j] == -67:
        start_pos = (i, j)
      elif result_23[i, j] == -33:
        seeing_nodes.append((i, j))

  if start_pos:
    start_row, start_col = start_pos
    for node_row, node_col in seeing_nodes:
      # Draw line with arrow
      ax.annotate(
          '',
          xy=(node_col, node_row),
          xytext=(start_col, start_row),
          arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.7),
      )

    # Highlight the starting node
    circle = patches.Circle(
        (start_col, start_row), 0.3, color='red', fill=False, linewidth=3
    )
    ax.add_patch(circle)

  ax.set_title(
      'Line-of-Sight Connections',
      fontsize=14,
      weight='bold',
  )
  ax.set_xlabel('Column')
  ax.set_ylabel('Row')

  # Create legend
  legend_elements = []
  for val, color in colors.items():
    if val in unique_values:
      if val == -67:
        legend_elements.append(
            mpatches.Patch(color=color, label='Starting Node (-67)')
        )
      elif val == -33:
        legend_elements.append(
            mpatches.Patch(color=color, label='Directly Seeing (-33)')
        )
      elif val == -34:
        legend_elements.append(
            mpatches.Patch(color=color, label='Blocked (-34)')
        )
      elif val == -3:
        legend_elements.append(
            mpatches.Patch(color=color, label='Interior Wall (-3)')
        )
      elif val == -2:
        legend_elements.append(
            mpatches.Patch(color=color, label='Exterior Wall (-2)')
        )
      elif val == 0:
        legend_elements.append(
            mpatches.Patch(color=color, label='Air Space (0)')
        )
      elif val == -1:
        legend_elements.append(
            mpatches.Patch(color=color, label='Exterior (-1)')
        )

  ax.legend(
      handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0)
  )

  # plt.tight_layout()

  # Save the figure
  plt.savefig(
      'radiation_connections_matplotlib.png', dpi=300, bbox_inches='tight'
  )
  print("Matplotlib figure saved as 'radiation_connections_matplotlib.png'")

  plt.show()


if __name__ == '__main__':
  visualize_radiation_connections()
