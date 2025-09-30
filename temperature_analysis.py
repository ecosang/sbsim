#!/usr/bin/env python3
"""
Detailed temperature analysis and comparison between iterative and tensorized approaches.

This script provides comprehensive statistical analysis and creates publication-ready
visualizations for comparing temperature distributions from different computational methods.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


def load_and_analyze_data():
  """Load temperature data and perform statistical analysis."""

  # Load data
  iter_data = pd.read_csv('iter.csv', index_col=0)
  tf_data = pd.read_csv('tf.csv', index_col=0)

  iter_array = iter_data.values
  tf_array = tf_data.values

  # Calculate difference
  diff_array = tf_array - iter_array

  # Statistical analysis
  stats_dict = {
      'iterative': {
          'mean': np.mean(iter_array),
          'std': np.std(iter_array),
          'min': np.min(iter_array),
          'max': np.max(iter_array),
          'range': np.max(iter_array) - np.min(iter_array),
      },
      'tensorized': {
          'mean': np.mean(tf_array),
          'std': np.std(tf_array),
          'min': np.min(tf_array),
          'max': np.max(tf_array),
          'range': np.max(tf_array) - np.min(tf_array),
      },
      'difference': {
          'mean': np.mean(diff_array),
          'std': np.std(diff_array),
          'min': np.min(diff_array),
          'max': np.max(diff_array),
          'max_abs': np.max(np.abs(diff_array)),
          'rmse': np.sqrt(np.mean(diff_array**2)),
      },
  }

  # Correlation analysis
  correlation = np.corrcoef(iter_array.flatten(), tf_array.flatten())[0, 1]

  # Relative error analysis
  relative_error = np.abs(diff_array) / np.abs(iter_array) * 100
  stats_dict['difference']['mean_relative_error'] = np.mean(relative_error)
  stats_dict['difference']['max_relative_error'] = np.max(relative_error)

  return iter_array, tf_array, diff_array, stats_dict, correlation


def create_detailed_comparison():
  """Create detailed comparison with statistical analysis."""

  # Load and analyze data
  iter_data, tf_data, diff_data, stats_dict, correlation = (
      load_and_analyze_data()
  )

  # Create figure with subplots
  fig = plt.figure(figsize=(16, 12))

  # Create grid layout
  gs = fig.add_gridspec(
      3,
      3,
      width_ratios=[1, 1, 1],
      height_ratios=[1, 1, 1],
      hspace=0.3,
      wspace=0.3,
  )

  # Define colorbar ranges
  temp_min = min(np.min(iter_data), np.min(tf_data))
  temp_max = max(np.max(iter_data), np.max(tf_data))
  diff_max = max(abs(np.min(diff_data)), abs(np.max(diff_data)))
  diff_min = -diff_max

  # Plot 1: Iterative approach
  ax1 = fig.add_subplot(gs[0, 0])
  im1 = ax1.imshow(
      iter_data,
      cmap='plasma',
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
      which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.7
  )

  # Plot 2: Tensorized approach
  ax2 = fig.add_subplot(gs[0, 1])
  im2 = ax2.imshow(
      tf_data,
      cmap='plasma',
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
      which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.7
  )

  # Plot 3: Difference
  ax3 = fig.add_subplot(gs[0, 2])
  im3 = ax3.imshow(
      diff_data,
      cmap='RdBu_r',
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

  # Plot 4: Scatter plot comparison
  ax4 = fig.add_subplot(gs[1, 0])
  ax4.scatter(iter_data.flatten(), tf_data.flatten(), alpha=0.6, s=20)
  ax4.plot([temp_min, temp_max], [temp_min, temp_max], 'r--', alpha=0.8)
  ax4.set_xlabel('Iterative Temperature (K)', fontsize=10)
  ax4.set_ylabel('Tensorized Temperature (K)', fontsize=10)
  ax4.set_title(
      f'Scatter Plot\n(r = {correlation:.6f})', fontsize=12, fontweight='bold'
  )
  ax4.grid(True, alpha=0.3)

  # Plot 5: Histogram of differences
  ax5 = fig.add_subplot(gs[1, 1])
  ax5.hist(
      diff_data.flatten(),
      bins=50,
      alpha=0.7,
      color='skyblue',
      edgecolor='black',
  )
  ax5.axvline(0, color='red', linestyle='--', alpha=0.8)
  ax5.set_xlabel('Temperature Difference (K)', fontsize=10)
  ax5.set_ylabel('Frequency', fontsize=10)
  ax5.set_title('Distribution of Differences', fontsize=12, fontweight='bold')
  ax5.grid(True, alpha=0.3)

  # Plot 6: Spatial error distribution
  ax6 = fig.add_subplot(gs[1, 2])
  abs_diff = np.abs(diff_data)
  im6 = ax6.imshow(abs_diff, cmap='Reds', aspect='equal', origin='upper')
  ax6.set_title(
      'Absolute Difference\n|Tensorized - Iterative|',
      fontsize=12,
      fontweight='bold',
  )
  ax6.set_xlabel('Column', fontsize=10)
  ax6.set_ylabel('Row', fontsize=10)

  # Add grid
  ax6.set_xticks(np.arange(-0.5, abs_diff.shape[1], 1), minor=True)
  ax6.set_yticks(np.arange(-0.5, abs_diff.shape[0], 1), minor=True)
  ax6.grid(
      which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.7
  )

  cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
  cbar6.set_label('|Temperature Difference| (K)', fontsize=10)
  cbar6.ax.tick_params(labelsize=8)

  # Statistics panel
  ax_stats = fig.add_subplot(gs[2, :])
  ax_stats.axis('off')

  # Create detailed statistics text
  stats_text = f"""
    COMPREHENSIVE STATISTICAL ANALYSIS

    Iterative Approach:           Mean = {stats_dict['iterative']['mean']:.6f} K,  Std = {stats_dict['iterative']['std']:.6f} K,  Range = {stats_dict['iterative']['range']:.6f} K
    Tensorized Approach:          Mean = {stats_dict['tensorized']['mean']:.6f} K,  Std = {stats_dict['tensorized']['std']:.6f} K,  Range = {stats_dict['tensorized']['range']:.6f} K

    Difference Analysis:
    Mean Difference:              {stats_dict['difference']['mean']:.2e} K
    Std of Difference:           {stats_dict['difference']['std']:.2e} K
    Maximum |Difference|:        {stats_dict['difference']['max_abs']:.2e} K
    RMSE:                        {stats_dict['difference']['rmse']:.2e} K
    Mean Relative Error:         {stats_dict['difference']['mean_relative_error']:.2e} %
    Maximum Relative Error:     {stats_dict['difference']['max_relative_error']:.2e} %

    Correlation Coefficient:     {correlation:.8f}
    """

  ax_stats.text(
      0.05,
      0.95,
      stats_text,
      transform=ax_stats.transAxes,
      fontsize=10,
      verticalalignment='top',
      fontfamily='monospace',
      bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
  )

  # Overall title
  fig.suptitle(
      'Comprehensive Temperature Analysis: Iterative vs Tensorized Approaches',
      fontsize=16,
      fontweight='bold',
      y=0.95,
  )

  # Save the figure
  plt.savefig('temperature_detailed_analysis.png', dpi=300, bbox_inches='tight')
  print("Detailed analysis saved as 'temperature_detailed_analysis.png'")

  plt.show()

  return stats_dict, correlation


def print_summary_statistics(stats_dict, correlation):
  """Print summary statistics to console."""

  print('\n' + '=' * 80)
  print('TEMPERATURE DISTRIBUTION COMPARISON SUMMARY')
  print('=' * 80)

  print(f'\nITERATIVE APPROACH:')
  print(f"  Mean Temperature:     {stats_dict['iterative']['mean']:.6f} K")
  print(f"  Standard Deviation:   {stats_dict['iterative']['std']:.6f} K")
  print(f"  Temperature Range:    {stats_dict['iterative']['range']:.6f} K")

  print(f'\nTENSORIZED APPROACH:')
  print(f"  Mean Temperature:     {stats_dict['tensorized']['mean']:.6f} K")
  print(f"  Standard Deviation:   {stats_dict['tensorized']['std']:.6f} K")
  print(f"  Temperature Range:    {stats_dict['tensorized']['range']:.6f} K")

  print(f'\nDIFFERENCE ANALYSIS (Tensorized - Iterative):')
  print(f"  Mean Difference:      {stats_dict['difference']['mean']:.2e} K")
  print(f"  Std of Difference:    {stats_dict['difference']['std']:.2e} K")
  print(f"  Maximum |Difference|: {stats_dict['difference']['max_abs']:.2e} K")
  print(f"  RMSE:                {stats_dict['difference']['rmse']:.2e} K")
  print(
      '  Mean Relative Error: '
      f' {stats_dict["difference"]["mean_relative_error"]:.2e} %'
  )
  print(
      '  Max Relative Error:  '
      f' {stats_dict["difference"]["max_relative_error"]:.2e} %'
  )

  print(f'\nCORRELATION:')
  print(f'  Correlation Coefficient: {correlation:.8f}')

  print('\n' + '=' * 80)


if __name__ == '__main__':
  print('Performing comprehensive temperature analysis...')
  stats_dict, correlation = create_detailed_comparison()
  print_summary_statistics(stats_dict, correlation)
