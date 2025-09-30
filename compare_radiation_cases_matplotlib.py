#!/usr/bin/env python3
"""
Compare all three test cases for radiative heat transfer line-of-sight using matplotlib.
Shows the differences between starting positions and their visibility patterns.
"""

from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def compare_all_cases():
  """Compare all three test cases side by side using matplotlib."""

  # Case 23: Starting at (2,3) - top-left corner
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

  # Case 27: Starting at (2,7) - top-right corner
  result_27 = np.array([
      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
      [-1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1],
      [-1, -2, -3, -34, -34, -34, -34, -67, -34, -3, -2, -1],
      [-1, -2, -33, 0, 0, 0, 0, 0, 0, -33, -2, -1],
      [-1, -2, -33, 0, 0, 0, 0, 0, 0, -33, -2, -1],
      [-1, -2, -33, 0, 0, 0, 0, 0, 0, -33, -2, -1],
      [-1, -2, -33, 0, 0, -33, -33, -33, 0, -33, -2, -1],
      [-1, -2, -33, 0, 0, -34, 0, -34, 0, -33, -2, -1],
      [-1, -2, -33, 0, 0, -34, 0, -34, 0, -33, -2, -1],
      [-1, -2, -33, 0, 0, -34, 0, -34, 0, -33, -2, -1],
      [-1, -2, -33, 0, 0, -34, 0, 0, 0, -33, -2, -1],
      [-1, -2, -3, -34, -34, -3, -34, -34, -33, -3, -2, -1],
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

  # Case 116: Starting at (11,6) - bottom-center
  result_116 = np.array([
      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
      [-1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1],
      [-1, -2, -3, -34, -34, -34, -34, -34, -34, -3, -2, -1],
      [-1, -2, -34, 0, 0, 0, 0, 0, 0, -34, -2, -1],
      [-1, -2, -34, 0, 0, 0, 0, 0, 0, -34, -2, -1],
      [-1, -2, -34, 0, 0, 0, 0, 0, 0, -34, -2, -1],
      [-1, -2, -34, 0, 0, -33, -33, -33, 0, -33, -2, -1],
      [-1, -2, -34, 0, 0, -33, 0, -33, 0, -33, -2, -1],
      [-1, -2, -34, 0, 0, -33, 0, -33, 0, -33, -2, -1],
      [-1, -2, -34, 0, 0, -33, 0, -33, 0, -33, -2, -1],
      [-1, -2, -34, 0, 0, -33, 0, 0, 0, -33, -2, -1],
      [-1, -2, -3, -34, -34, -34, -67, -34, -34, -3, -2, -1],
      [-1, -2, -3, 0, -33, 0, 0, 0, 0, -33, -2, -1],
      [-1, -2, -3, 0, -33, 0, 0, 0, 0, -33, -2, -1],
      [-1, -2, -3, -34, -33, 0, 0, 0, 0, -33, -2, -1],
      [-1, -2, -34, 0, 0, 0, 0, 0, 0, -33, -2, -1],
      [-1, -2, -34, 0, 0, 0, 0, 0, 0, -33, -2, -1],
      [-1, -2, -34, 0, 0, 0, 0, 0, 0, -33, -2, -1],
      [-1, -2, -33, 0, 0, 0, 0, 0, 0, -33, -2, -1],
      [-1, -2, -33, 0, 0, 0, 0, 0, 0, -33, -2, -1],
      [-1, -2, -3, -33, -33, -33, -33, -33, -33, -3, -2, -1],
      [-1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1],
      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
  ])

  # Create figure with subplots
  fig, axes = plt.subplots(1, 3, figsize=(18, 6))

  cases = [
      ('Case 23: Top-Left Corner (2,3)', result_23, axes[0]),
      ('Case 27: Top-Right Corner (2,7)', result_27, axes[1]),
      ('Case 116: Bottom-Center (11,6)', result_116, axes[2]),
  ]

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

  for case_name, result, ax in cases:
    # Create colormap
    unique_values = np.unique(result)
    color_list = [colors.get(val, '#CCCCCC') for val in unique_values]
    cmap = ListedColormap(color_list)

    # Plot the floor plan
    im = ax.imshow(result, cmap=cmap, aspect='equal')

    # Add grid
    ax.set_xticks(np.arange(-0.5, result.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, result.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

    # Add cell values as text
    for i in range(result.shape[0]):
      for j in range(result.shape[1]):
        value = result[i, j]
        if value != 0:  # Don't show text for air spaces
          ax.text(
              j,
              i,
              str(value),
              ha='center',
              va='center',
              fontsize=6,
              weight='bold' if value in [-67, -33, -34] else 'normal',
          )

    # Draw connections from starting node to seeing nodes
    start_pos = None
    seeing_nodes = []

    for i in range(result.shape[0]):
      for j in range(result.shape[1]):
        if result[i, j] == -67:
          start_pos = (i, j)
        elif result[i, j] == -33:
          seeing_nodes.append((i, j))

    if start_pos:
      start_row, start_col = start_pos
      for node_row, node_col in seeing_nodes:
        # Draw line with arrow
        ax.annotate(
            '',
            xy=(node_col, node_row),
            xytext=(start_col, start_row),
            arrowprops=dict(arrowstyle='->', color='red', lw=1, alpha=0.7),
        )

      # Highlight the starting node
      circle = patches.Circle(
          (start_col, start_row), 0.3, color='red', fill=False, linewidth=2
      )
      ax.add_patch(circle)

    ax.set_title(case_name, fontsize=12, weight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Count nodes for statistics
    start_nodes = np.sum(result == -67)
    seeing_nodes_count = np.sum(result == -33)
    blocked_nodes_count = np.sum(result == -34)

    # Add statistics text
    ax.text(
        0.02,
        0.98,
        f'Seeing: {seeing_nodes_count}\nBlocked: {blocked_nodes_count}',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

  # Create legend
  legend_elements = []
  for val, color in colors.items():
    if val == -67:
      legend_elements.append(
          mpatches.Patch(color=color, label='Starting Node (-67)')
      )
    elif val == -33:
      legend_elements.append(
          mpatches.Patch(color=color, label='Directly Seeing (-33)')
      )
    elif val == -34:
      legend_elements.append(mpatches.Patch(color=color, label='Blocked (-34)'))
    elif val == -3:
      legend_elements.append(
          mpatches.Patch(color=color, label='Interior Wall (-3)')
      )
    elif val == -2:
      legend_elements.append(
          mpatches.Patch(color=color, label='Exterior Wall (-2)')
      )
    elif val == 0:
      legend_elements.append(mpatches.Patch(color=color, label='Air Space (0)'))
    elif val == -1:
      legend_elements.append(mpatches.Patch(color=color, label='Exterior (-1)'))

  fig.legend(
      handles=legend_elements,
      loc='upper center',
      bbox_to_anchor=(0.5, 0.95),
      ncol=4,
  )

  plt.suptitle(
      'Radiative Heat Transfer - Comparison of All Three Test Cases',
      fontsize=16,
      weight='bold',
  )
  plt.tight_layout()

  # Save the figure
  plt.savefig(
      'radiation_connections_comparison_matplotlib.png',
      dpi=300,
      bbox_inches='tight',
  )
  print(
      'Comparison matplotlib figure saved as'
      " 'radiation_connections_comparison_matplotlib.png'"
  )

  plt.show()


if __name__ == '__main__':
  compare_all_cases()
