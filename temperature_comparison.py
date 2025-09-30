#!/usr/bin/env python3
"""
Temperature distribution comparison between iterative and tensorized approaches.

This script creates publication-quality heat maps comparing temperature distributions
from two different computational methods, suitable for double-column papers.
"""

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_temperature_data():
  """Load temperature data from CSV files."""
  # Load iterative approach data
  iter_data = pd.read_csv('iter.csv', index_col=0)
  iter_array = iter_data.values

  # Load tensorized approach data
  tf_data = pd.read_csv('tf.csv', index_col=0)
  tf_array = tf_data.values

  return iter_array, tf_array


def add_floor_plan_overlay(ax, plan):
  """Add floor plan indices as text overlay on the heat map."""
  for i in range(plan.shape[0]):
    for j in range(plan.shape[1]):
      value = plan[i, j]
      # Choose text color based on background
      if value == 2:  # Ambient - use black text
        text_color = 'black'
        weight = 'bold'
      elif value == 1:  # Wall - use black text
        text_color = 'black'
        weight = 'bold'
      else:  # Air - use black text
        text_color = 'black'
        weight = 'normal'

      ax.text(
          j,
          i,
          str(value),
          ha='center',
          va='center',
          fontsize=8,
          weight=weight,
          color=text_color,
          bbox=dict(
              boxstyle='round,pad=0.1',
              facecolor='white',
              alpha=0.7,
              edgecolor='none',
          ),
      )


def create_temperature_comparison():
  """Create comprehensive temperature comparison visualization."""

  # Load data
  iter_data, tf_data = load_temperature_data()

  # Calculate difference
  diff_data = tf_data - iter_data

  # Define floor plan structure
  plan = np.array([
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
  ])

  # Set up the figure for double-column paper
  fig = plt.figure(figsize=(12, 8))

  # Create a 2x3 grid layout
  gs = fig.add_gridspec(
      2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.3
  )

  # Define common colorbar range for temperature maps
  temp_min = min(np.min(iter_data), np.min(tf_data))
  temp_max = max(np.max(iter_data), np.max(tf_data))

  # Define difference colorbar range
  diff_max = max(abs(np.min(diff_data)), abs(np.max(diff_data)))
  diff_min = -diff_max

  # Create custom colormap for temperature (blue to red)
  colors_temp = [
      '#000080',
      '#0000FF',
      '#00FFFF',
      '#FFFF00',
      '#FF8000',
      '#FF0000',
  ]
  n_bins = 256
  cmap_temp = LinearSegmentedColormap.from_list(
      'temperature', colors_temp, N=n_bins
  )

  # Create custom colormap for difference (blue-white-red)
  colors_diff = ['#0000FF', '#FFFFFF', '#FF0000']
  cmap_diff = LinearSegmentedColormap.from_list(
      'difference', colors_diff, N=n_bins
  )

  # Plot 1: Iterative approach
  ax1 = fig.add_subplot(gs[0, 0])
  im1 = ax1.imshow(
      iter_data,
      cmap=cmap_temp,
      vmin=temp_min,
      vmax=temp_max,
      aspect='equal',
      origin='upper',
  )
  ax1.set_title('Iterative Approach', fontsize=12, fontweight='bold')
  ax1.set_xlabel('Column', fontsize=10)
  ax1.set_ylabel('Row', fontsize=10)

  # Add grid
  ax1.set_xticks(np.arange(-0.5, iter_data.shape[1], 1), minor=True)
  ax1.set_yticks(np.arange(-0.5, iter_data.shape[0], 1), minor=True)
  ax1.grid(
      which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3
  )

  # Add floor plan overlay
  add_floor_plan_overlay(ax1, plan)

  # Plot 2: Tensorized approach
  ax2 = fig.add_subplot(gs[0, 1])
  im2 = ax2.imshow(
      tf_data,
      cmap=cmap_temp,
      vmin=temp_min,
      vmax=temp_max,
      aspect='equal',
      origin='upper',
  )
  ax2.set_title('Tensorized Approach', fontsize=12, fontweight='bold')
  ax2.set_xlabel('Column', fontsize=10)
  ax2.set_ylabel('Row', fontsize=10)

  # Add grid
  ax2.set_xticks(np.arange(-0.5, tf_data.shape[1], 1), minor=True)
  ax2.set_yticks(np.arange(-0.5, tf_data.shape[0], 1), minor=True)
  ax2.grid(
      which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3
  )

  # Add floor plan overlay
  add_floor_plan_overlay(ax2, plan)

  # Plot 3: Difference (Tensorized - Iterative)
  ax3 = fig.add_subplot(gs[0, 2])
  im3 = ax3.imshow(
      diff_data,
      cmap=cmap_diff,
      vmin=diff_min,
      vmax=diff_max,
      aspect='equal',
      origin='upper',
  )
  ax3.set_title(
      'Difference\n(Tensorized - Iterative)', fontsize=12, fontweight='bold'
  )
  ax3.set_xlabel('Column', fontsize=10)
  ax3.set_ylabel('Row', fontsize=10)

  # Add grid
  ax3.set_xticks(np.arange(-0.5, diff_data.shape[1], 1), minor=True)
  ax3.set_yticks(np.arange(-0.5, diff_data.shape[0], 1), minor=True)
  ax3.grid(
      which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3
  )

  # Add floor plan overlay
  add_floor_plan_overlay(ax3, plan)

  # Add colorbars
  cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
  cbar1.set_label('Temperature (K)', fontsize=10)
  cbar1.ax.tick_params(labelsize=8)

  cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
  cbar2.set_label('Temperature (K)', fontsize=10)
  cbar2.ax.tick_params(labelsize=8)

  cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
  cbar3.set_label('Temperature Difference (K)', fontsize=10)
  cbar3.ax.tick_params(labelsize=8)

  # Statistics panel
  ax_stats = fig.add_subplot(gs[1, :])
  ax_stats.axis('off')

  # Calculate statistics
  iter_mean = np.mean(iter_data)
  tf_mean = np.mean(tf_data)
  iter_std = np.std(iter_data)
  tf_std = np.std(tf_data)
  max_diff = np.max(np.abs(diff_data))
  mean_diff = np.mean(diff_data)
  std_diff = np.std(diff_data)

  # Create statistics text
  stats_text = f"""
  Statistical Comparison:

  Iterative Approach:     Mean = {iter_mean:.6f} K,  Std = {iter_std:.6f} K
  Tensorized Approach:    Mean = {tf_mean:.6f} K,  Std = {tf_std:.6f} K

  Difference (Tensorized - Iterative):
  Mean Difference:        {mean_diff:.2e} K
  Std of Difference:     {std_diff:.2e} K
  Maximum |Difference|:  {max_diff:.2e} K

  Floor Plan Structure:
  2 = Ambient Air, 1 = Wall, 0 = Air Space
  """

  ax_stats.text(
      0.05,
      0.95,
      stats_text,
      transform=ax_stats.transAxes,
      fontsize=11,
      verticalalignment='top',
      fontfamily='monospace',
      bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
  )

  # Add method comparison legend
  legend_elements = [
      mpatches.Patch(color='#0000FF', label='Iterative Method'),
      mpatches.Patch(color='#FF0000', label='Tensorized Method'),
      mpatches.Patch(color='#FFFFFF', label='Difference Map'),
  ]

  ax_stats.legend(
      handles=legend_elements,
      loc='upper right',
      bbox_to_anchor=(0.95, 0.95),
      fontsize=10,
  )

  # Overall title
  fig.suptitle(
      'Temperature Distribution Comparison: Iterative vs Tensorized Approaches',
      fontsize=14,
      fontweight='bold',
      y=0.95,
  )

  # Save the figure
  plt.savefig('temperature_comparison.png', dpi=300, bbox_inches='tight')
  print("Temperature comparison saved as 'temperature_comparison.png'")

  plt.show()


def create_side_by_side_comparison():
  """Create a simpler side-by-side comparison for quick viewing."""

  # Load data
  iter_data, tf_data = load_temperature_data()

  # Define floor plan structure
  plan = np.array([
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
  ])

  # Create figure
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

  # Define common colorbar range
  temp_min = min(np.min(iter_data), np.min(tf_data))
  temp_max = max(np.max(iter_data), np.max(tf_data))

  # Plot iterative approach
  im1 = ax1.imshow(
      iter_data,
      cmap='viridis',
      vmin=temp_min,
      vmax=temp_max,
      aspect='equal',
      origin='upper',
  )
  ax1.set_title('Iterative Approach', fontsize=14, fontweight='bold')
  ax1.set_xlabel('Column', fontsize=12)
  ax1.set_ylabel('Row', fontsize=12)

  # Add grid
  ax1.set_xticks(np.arange(-0.5, iter_data.shape[1], 1), minor=True)
  ax1.set_yticks(np.arange(-0.5, iter_data.shape[0], 1), minor=True)
  ax1.grid(
      which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.7
  )

  # Add floor plan overlay
  add_floor_plan_overlay(ax1, plan)

  # Plot tensorized approach
  im2 = ax2.imshow(
      tf_data,
      cmap='viridis',
      vmin=temp_min,
      vmax=temp_max,
      aspect='equal',
      origin='upper',
  )
  ax2.set_title('Tensorized Approach', fontsize=14, fontweight='bold')
  ax2.set_xlabel('Column', fontsize=12)
  ax2.set_ylabel('Row', fontsize=12)

  # Add grid
  ax2.set_xticks(np.arange(-0.5, tf_data.shape[1], 1), minor=True)
  ax2.set_yticks(np.arange(-0.5, tf_data.shape[0], 1), minor=True)
  ax2.grid(
      which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.7
  )

  # Add floor plan overlay
  add_floor_plan_overlay(ax2, plan)

  # Add colorbars
  cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
  cbar1.set_label('Temperature (K)', fontsize=12)

  cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
  cbar2.set_label('Temperature (K)', fontsize=12)

  plt.tight_layout()

  # Save the figure
  plt.savefig('temperature_side_by_side.png', dpi=300, bbox_inches='tight')
  print("Side-by-side comparison saved as 'temperature_side_by_side.png'")

  plt.show()


if __name__ == '__main__':
  print('Creating comprehensive temperature comparison...')
  create_temperature_comparison()

  print('\nCreating side-by-side comparison...')
  create_side_by_side_comparison()
