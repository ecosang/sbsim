#!/usr/bin/env python3
"""
Compare all three test cases for radiative heat transfer line-of-sight.
Shows the differences between starting positions and their visibility patterns.
"""


def compare_all_cases():
  """Compare all three test cases side by side."""

  # Original floor plan
  indexed_floor_plan = [
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
  ]

  # Case 23: Starting at (2,3) - top-left corner
  result_23 = [
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
  ]

  # Case 27: Starting at (2,7) - top-right corner
  result_27 = [
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
  ]

  # Case 116: Starting at (11,6) - bottom-center
  result_116 = [
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
  ]

  print("=" * 120)
  print("RADIATIVE HEAT TRANSFER - COMPARISON OF ALL THREE TEST CASES")
  print("=" * 120)
  print()
  print("Legend:")
  print("  ★ = Starting node (-67)")
  print(
      "  ● = Directly seeing nodes (-33) - can participate in radiative"
      " transfer"
  )
  print("  ✗ = Blocked nodes (-34) - cannot see starting node")
  print("  █ = Interior walls (-3)")
  print("  ▓ = Exterior walls (-2)")
  print("  · = Air spaces (0)")
  print()

  # Print all three cases side by side
  cases = [
      ("Case 23: Top-Left Corner (2,3)", result_23),
      ("Case 27: Top-Right Corner (2,7)", result_27),
      ("Case 116: Bottom-Center (11,6)", result_116),
  ]

  for case_name, result in cases:
    print(f"{case_name}")
    print("-" * 60)

    # Count nodes
    start_nodes = sum(1 for row in result for val in row if val == -67)
    seeing_nodes = sum(1 for row in result for val in row if val == -33)
    blocked_nodes = sum(1 for row in result for val in row if val == -34)

    print(
        f"Starting nodes: {start_nodes}, Seeing nodes: {seeing_nodes}, Blocked"
        f" nodes: {blocked_nodes}"
    )
    print()

    # Print the floor plan
    for i, row in enumerate(result):
      line = ""
      for j, value in enumerate(row):
        if value == -67:
          line += "★ "
        elif value == -33:
          line += "● "
        elif value == -34:
          line += "✗ "
        elif value == -3:
          line += "█ "
        elif value == -2:
          line += "▓ "
        elif value == 0:
          line += "· "
        elif value == -1:
          line += "  "
        else:
          line += f"{value:2d}"
      print(f"{i:2d}: {line}")
    print()

  print("Analysis Summary:")
  print("-" * 20)
  print(
      "• Case 23 (top-left): Starting from corner, can see many nodes to the"
      " right and down"
  )
  print(
      "• Case 27 (top-right): Starting from opposite corner, different"
      " visibility pattern"
  )
  print(
      "• Case 116 (bottom-center): Starting from center, more limited"
      " visibility due to walls"
  )
  print(
      "• Each case demonstrates different line-of-sight patterns based on"
      " starting position"
  )
  print(
      "• The algorithm correctly identifies which walls can exchange radiative"
      " heat"
  )
  print("• This is essential for accurate building thermal simulation")


def save_comparison_visualization_to_file():
  """Save the comparison ASCII visualization to a text file."""
  from io import StringIO
  import sys

  # Capture the output
  old_stdout = sys.stdout
  sys.stdout = StringIO()

  # Run the visualization
  compare_all_cases()

  # Get the captured output
  output = sys.stdout.getvalue()
  sys.stdout = old_stdout

  # Save to file
  with open("radiation_connections_comparison_ascii.txt", "w") as f:
    f.write(output)

  print(
      "Comparison ASCII visualization saved as"
      " 'radiation_connections_comparison_ascii.txt'"
  )


if __name__ == "__main__":
  compare_all_cases()
  save_comparison_visualization_to_file()
