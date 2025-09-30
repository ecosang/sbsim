#!/usr/bin/env python3
"""
Visualization script for radiative heat transfer line-of-sight connections.

This script visualizes the connections between the starting node (-67) and
directly seeing nodes (-33) in the building radiation test cases.
"""

from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def visualize_radiation_connections():
  """Visualize the line-of-sight connections for radiative heat transfer."""

  # Test data from case_23 (starting node at (2,3))
  indexed_floor_plan = np.array([
      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
      [-1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1],
      [-1, -2, -3, -3, -3, -3, -3, -3, -3, -3, -2, -1],
      [-1, -2, -3, 0, 0, 0, 0, 0, 0, -3, -2, -1],
      [-1, -2, -3, 0, 0, 0, 0, 0, 0, -3, -2, -1],
      [-1, -2, -3, 0, 0, 0, 0, 0, 0, -3, -2, -1],
      [-1, -2, -3, 0, 0, -3, -3, -3, 0, -3, -2, -1],
      [-1, -2, -3, 0, 0, -3, 0, -3, 0, -3, -2, -1],
      [-1, -2, -3, 0, 0, -3, 0, -3, 0, -3, -2, -1],
      [-1, -2, -3, 0, 0, -3, 0, -3, 0, -3, -2, -1],
      [-1, -2, -3, 0, 0, -3, 0, 0, 0, -3, -2, -1],
      [-1, -2, -3, -3, -3, -3, -3, -3, -3, -3, -2, -1],
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

  # Result after marking air-connected interior walls and line-of-sight
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

  # Create figure with subplots
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

  # Plot 1: Original floor plan
  plot_floor_plan(
      ax1, indexed_floor_plan, 'Original Floor Plan', show_connections=False
  )

  # Plot 2: Line-of-sight connections
  plot_floor_plan(
      ax2,
      result_23,
      'Line-of-Sight Connections',
      show_connections=True,
      start_pos=(2, 3),
  )

  fig.tight_layout()
  # plt.subplots_adjust(wspace=0.05)  # Reduce horizontal gap
  # Save the figure
  plt.savefig(
      'radiation_connections_visualization.png', dpi=300, bbox_inches='tight'
  )
  print("Figure saved as 'radiation_connections_visualization.png'")

  plt.show()


def plot_floor_plan(
    ax, floor_plan, title, show_connections=False, start_pos=None
):
  """Plot the floor plan with appropriate colors and connections."""

  # Define colors for different cell types
  colors = {
      -1: '#87CEEB',  # Ambient air (light blue)
      -2: '#8B4513',  # Exterior walls (brown)
      -3: '#A0A0A0',  # Interior walls (gray)
      0: '#FFFFFF',  # Air spaces (white)
      -33: '#00FF00',  # Directly seeing nodes (bright green)
      -34: '#FF6B6B',  # Blocked nodes (red)
      -67: '#FFD700',  # Starting node (gold)
  }

  # Create a mapping array for proper color assignment
  unique_values = np.unique(floor_plan)
  color_mapping = np.zeros_like(floor_plan, dtype=float)

  # Map each unique value to a sequential index
  value_to_index = {val: idx for idx, val in enumerate(unique_values)}
  for i in range(floor_plan.shape[0]):
    for j in range(floor_plan.shape[1]):
      color_mapping[i, j] = value_to_index[floor_plan[i, j]]

  # Create colormap with proper color mapping
  color_list = [colors.get(val, '#CCCCCC') for val in unique_values]
  cmap = ListedColormap(color_list)

  # Plot the floor plan using the color mapping
  im = ax.imshow(
      color_mapping,
      cmap=cmap,
      aspect='equal',
      vmin=0,
      vmax=len(unique_values) - 1,
  )

  # Add grid
  ax.set_xticks(np.arange(-0.5, floor_plan.shape[1], 1), minor=True)
  ax.set_yticks(np.arange(-0.5, floor_plan.shape[0], 1), minor=True)
  ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

  # Add cell values as text
  for i in range(floor_plan.shape[0]):
    for j in range(floor_plan.shape[1]):
      value = floor_plan[i, j]
      # Show text for all non-zero values, including air spaces (0)
      if value != 0:
        ax.text(
            j,
            i,
            str(value),
            ha='center',
            va='center',
            fontsize=8,
            weight='bold' if value in [-67, -33, -34] else 'normal',
        )
      else:  # Show 0 for air spaces
        ax.text(
            j,
            i,
            '0',
            ha='center',
            va='center',
            fontsize=8,
            weight='normal',
            color='black',
        )

  # Draw connections if requested
  if show_connections and start_pos:
    draw_connections(ax, floor_plan, start_pos)

  ax.set_title(title, fontsize=14, weight='bold')
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
            mpatches.Patch(color=color, label='Ambient Air (-1)')
        )

  ax.legend(
      handles=legend_elements,
      loc='lower right',
      # bbox_to_anchor=(1.0, 1.0),
      framealpha=1.0,
      facecolor='white',
      edgecolor='black',
  )


def draw_connections(ax, floor_plan, start_pos):
  """Draw lines connecting the starting node to directly seeing nodes."""

  start_row, start_col = start_pos

  # Find all directly seeing nodes (-33)
  seeing_nodes = []
  for i in range(floor_plan.shape[0]):
    for j in range(floor_plan.shape[1]):
      if floor_plan[i, j] == -33:
        seeing_nodes.append((i, j))

  # Draw lines from starting node to each seeing node
  for node_row, node_col in seeing_nodes:
    # Convert to plot coordinates (x, y)
    start_x, start_y = start_col, start_row
    end_x, end_y = node_col, node_row

    # Draw line with arrow
    ax.annotate(
        '',
        xy=(end_x, end_y),
        xytext=(start_x, start_y),
        arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.7),
    )

  # Highlight the starting node
  circle = patches.Circle(
      (start_col, start_row), 0.3, color='red', fill=False, linewidth=3
  )
  ax.add_patch(circle)


if __name__ == '__main__':
  visualize_radiation_connections()
