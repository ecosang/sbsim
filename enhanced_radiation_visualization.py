#!/usr/bin/env python3
"""
Enhanced visualization of radiative heat transfer line-of-sight connections.
This version shows the actual line connections between nodes.
"""


def draw_line_connections():
  """Draw the floor plan with actual line connections between nodes."""

  # Test data from case_23 (starting node at (2,3))
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

  # Find starting node and directly seeing nodes
  start_pos = None
  seeing_nodes = []

  for i, row in enumerate(result_23):
    for j, value in enumerate(row):
      if value == -67:
        start_pos = (i, j)
      elif value == -33:
        seeing_nodes.append((i, j))

  print("=" * 100)
  print("RADIATIVE HEAT TRANSFER - LINE-OF-SIGHT CONNECTIONS WITH VISUAL LINES")
  print("=" * 100)
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
  print("  ─│╱╲ = Connection lines")
  print()

  # Create a larger grid to show connections
  rows, cols = len(result_23), len(result_23[0])
  display_grid = [[" " for _ in range(cols * 3)] for _ in range(rows * 3)]

  # Fill in the basic floor plan
  for i in range(rows):
    for j in range(cols):
      value = result_23[i][j]
      display_i, display_j = i * 3 + 1, j * 3 + 1

      if value == -67:
        display_grid[display_i][display_j] = "★"
      elif value == -33:
        display_grid[display_i][display_j] = "●"
      elif value == -34:
        display_grid[display_i][display_j] = "✗"
      elif value == -3:
        display_grid[display_i][display_j] = "█"
      elif value == -2:
        display_grid[display_i][display_j] = "▓"
      elif value == 0:
        display_grid[display_i][display_j] = "·"

  # Draw connections from starting node to seeing nodes
  if start_pos:
    start_row, start_col = start_pos
    start_display_i, start_display_j = start_row * 3 + 1, start_col * 3 + 1

    for node_row, node_col in seeing_nodes:
      node_display_i, node_display_j = node_row * 3 + 1, node_col * 3 + 1

      # Draw line from start to node
      draw_line_on_grid(
          display_grid,
          start_display_i,
          start_display_j,
          node_display_i,
          node_display_j,
      )

  # Print the grid
  print("Floor Plan with Line Connections:")
  print("-" * 60)

  for i, row in enumerate(display_grid):
    if i % 3 == 0:  # Print row numbers
      print(f"{i//3:2d}: ", end="")
    else:
      print("    ", end="")

    line = "".join(row)
    print(line)

  print()
  print("Connection Details:")
  print("-" * 20)

  if start_pos:
    start_row, start_col = start_pos
    print(f"Starting node at ({start_row}, {start_col})")
    print(f"Total directly seeing nodes: {len(seeing_nodes)}")
    print()

    # Group connections by direction for better analysis
    connections_by_direction = {}
    for node_row, node_col in seeing_nodes:
      dr = node_row - start_row
      dc = node_col - start_col

      if dr == 0 and dc > 0:
        direction = "right"
      elif dr == 0 and dc < 0:
        direction = "left"
      elif dr > 0 and dc == 0:
        direction = "down"
      elif dr < 0 and dc == 0:
        direction = "up"
      elif dr > 0 and dc > 0:
        direction = "down-right"
      elif dr > 0 and dc < 0:
        direction = "down-left"
      elif dr < 0 and dc > 0:
        direction = "up-right"
      elif dr < 0 and dc < 0:
        direction = "up-left"
      else:
        direction = "same"

      if direction not in connections_by_direction:
        connections_by_direction[direction] = []
      connections_by_direction[direction].append((node_row, node_col))

    for direction, nodes in connections_by_direction.items():
      print(f"{direction:12s}: {len(nodes)} nodes {nodes}")

  print()
  print("Physical Interpretation:")
  print("-" * 25)
  print(
      "• The starting node represents a wall surface that can emit/absorb"
      " radiation"
  )
  print(
      "• Directly seeing nodes are other wall surfaces that have a clear line"
      " of sight"
  )
  print("• These connections represent potential radiative heat transfer paths")
  print(
      "• The line-of-sight algorithm determines which walls can 'see' each"
      " other"
  )
  print(
      "• This is crucial for calculating radiative heat exchange in building"
      " simulation"
  )
  print("• Air spaces allow radiation to pass through, walls can block it")


def draw_line_on_grid(grid, start_i, start_j, end_i, end_j):
  """Draw a line on the grid from start to end position."""

  # Simple line drawing algorithm
  di = end_i - start_i
  dj = end_j - start_j

  steps = max(abs(di), abs(dj))
  if steps == 0:
    return

  step_i = di / steps
  step_j = dj / steps

  for k in range(1, steps):
    i = int(start_i + k * step_i)
    j = int(start_j + k * step_j)

    if 0 <= i < len(grid) and 0 <= j < len(grid[0]):
      if grid[i][j] == " ":
        # Choose line character based on direction
        if abs(step_i) > abs(step_j):
          grid[i][j] = "│"  # vertical
        elif abs(step_j) > abs(step_i):
          grid[i][j] = "─"  # horizontal
        elif step_i * step_j > 0:
          grid[i][j] = "╲"  # diagonal down-right
        else:
          grid[i][j] = "╱"  # diagonal down-left


def save_enhanced_visualization_to_file():
  """Save the enhanced ASCII visualization to a text file."""
  from io import StringIO
  import sys

  # Capture the output
  old_stdout = sys.stdout
  sys.stdout = StringIO()

  # Run the visualization
  draw_line_connections()

  # Get the captured output
  output = sys.stdout.getvalue()
  sys.stdout = old_stdout

  # Save to file
  with open("radiation_connections_enhanced_ascii.txt", "w") as f:
    f.write(output)

  print(
      "Enhanced ASCII visualization saved as"
      " 'radiation_connections_enhanced_ascii.txt'"
  )


if __name__ == "__main__":
  draw_line_connections()
  save_enhanced_visualization_to_file()
