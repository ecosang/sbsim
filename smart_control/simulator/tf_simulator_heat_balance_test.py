"""Tests for the face based heat balance of the TF finite difference solver.

Every control volume exchanges heat with a neighbor through the face the two
of them share, so the conductance of that face is the series combination of the
half cell on either side of it,

  Gamma_f = [l_P / (2 k_P) + l_N / (2 k_N)]^-1,

and a face that opens onto the ambient puts the convective film in series with
the outer half cell of the boundary CV,

  Gamma_inf,f = [1 / h_ext + l_P / (2 k_P)]^-1.

Taking the owner CV's own conductivity across its own full width instead makes
the two CVs sharing a face disagree about the heat crossing it, and using
h_ext on its own drops the conduction resistance of the outer half cell. These
tests pin down both conductances.
"""

from absl.testing import absltest
import numpy as np

from smart_control.simulator import tf_simulator as tf_simulator_py


def _mask_neighbors(mask: np.ndarray) -> list[list[list[tuple[int, int]]]]:
  """Returns the neighbor table of the CVs marked in a boolean mask.

  Args:
    mask: True on the CVs that make up the building.

  Returns:
    For every CV of the grid, the coordinates of its marked neighbors. CVs
    that are not marked have none, which makes them exterior CVs.
  """
  rows, cols = mask.shape

  def _inside(i: int, j: int) -> bool:
    return 0 <= i < rows and 0 <= j < cols and bool(mask[i][j])

  return [
      [
          [
              (a, b)
              for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
              if _inside(a, b)
          ]
          if _inside(i, j)
          else []
          for j in range(cols)
      ]
      for i in range(rows)
  ]


def _solid_block_neighbors(
    shape: tuple[int, int],
    rows: tuple[int, int],
    cols: tuple[int, int],
) -> list[list[list[tuple[int, int]]]]:
  """Returns a neighbor table whose CVs fill one solid rectangular block.

  Args:
    shape: shape of the grid.
    rows: inclusive first and last row of the block.
    cols: inclusive first and last column of the block.

  Returns:
    For every CV of the grid, the coordinates of its in-block neighbors. CVs
    outside the block have none, which makes them exterior CVs.
  """
  mask = np.full(shape, False)
  mask[rows[0] : rows[1] + 1, cols[0] : cols[1] + 1] = True
  return _mask_neighbors(mask)


class FaceConductanceTest(absltest.TestCase):
  """Tests the conductances that carry heat across a face."""

  # A 4 x 5 block of CVs inside a 6 x 7 grid, so that the block has a ring of
  # boundary CVs around a 2 x 3 patch of interior CVs.
  SHAPE = (6, 7)
  BLOCK_ROWS = (1, 4)
  BLOCK_COLS = (1, 5)
  CV_SIZE_M = 0.2
  CONVECTION_COEFFICIENT = 12.0

  def _two_material_conductivity(self) -> np.ndarray:
    """Returns a conductivity field with a vertical material interface."""
    conductivity = np.full(self.SHAPE, 0.05, dtype=np.float32)
    conductivity[:, 3:] = 50.0
    return conductivity

  def _conductances(self, conductivity: np.ndarray):
    """Returns the geometry, the face and the ambient conductance tensors."""
    boundary_cv_mapping = tf_simulator_py.get_cv_mapping(
        _solid_block_neighbors(self.SHAPE, self.BLOCK_ROWS, self.BLOCK_COLS)
    )
    t_u, t_v = tf_simulator_py.get_cv_dimension_tensors(
        self.CV_SIZE_M, boundary_cv_mapping, self.SHAPE
    )
    face = tf_simulator_py.get_oriented_face_conductance_tensors(
        conductivity, t_u, t_v, boundary_cv_mapping
    )
    ambient = tf_simulator_py.get_oriented_ambient_conductance_tensors(
        self.CONVECTION_COEFFICIENT,
        conductivity,
        t_u,
        t_v,
        boundary_cv_mapping,
    )
    return (
        t_u.numpy(),
        t_v.numpy(),
        [tensor.numpy() for tensor in face],
        [tensor.numpy() for tensor in ambient],
    )

  def test_half_cell_resistance_is_length_over_twice_the_conductivity(self):
    conductivity = self._two_material_conductivity()
    boundary_cv_mapping = tf_simulator_py.get_cv_mapping(
        _solid_block_neighbors(self.SHAPE, self.BLOCK_ROWS, self.BLOCK_COLS)
    )
    t_u, t_v = tf_simulator_py.get_cv_dimension_tensors(
        self.CV_SIZE_M, boundary_cv_mapping, self.SHAPE
    )

    t_r_u, t_r_v = tf_simulator_py.get_half_cell_resistance_tensors(
        conductivity, t_u, t_v
    )

    np.testing.assert_allclose(
        t_r_u.numpy(), t_u.numpy() / (2.0 * conductivity), rtol=1e-6
    )
    np.testing.assert_allclose(
        t_r_v.numpy(), t_v.numpy() / (2.0 * conductivity), rtol=1e-6
    )

  def test_interior_face_conductance_is_the_two_half_cells_in_series(self):
    conductivity = self._two_material_conductivity()
    u, v, (left, right, top, bottom), _ = self._conductances(conductivity)

    # (2, 2) and (2, 3) are interior CVs on either side of the material
    # interface: insulating exterior wall on the left, well mixed air on the
    # right.
    expected = 1.0 / (
        u[2][2] / (2.0 * conductivity[2][2])
        + u[2][3] / (2.0 * conductivity[2][3])
    )
    self.assertAlmostEqual(right[2][2], expected, places=5)
    self.assertAlmostEqual(left[2][3], expected, places=5)

    # Vertically the material is the same on both sides of the face.
    expected_vertical = 1.0 / (
        v[2][3] / (2.0 * conductivity[2][3])
        + v[3][3] / (2.0 * conductivity[3][3])
    )
    np.testing.assert_allclose(bottom[2][3], expected_vertical, rtol=1e-5)
    np.testing.assert_allclose(top[3][3], expected_vertical, rtol=1e-5)

  def test_interior_face_conductance_is_dominated_by_the_insulator(self):
    """The series form is bounded by twice the smaller half-cell conductance."""
    conductivity = self._two_material_conductivity()
    u, _, (_, right, _, _), _ = self._conductances(conductivity)

    insulating_side = 2.0 * conductivity[2][2] / u[2][2]
    owner_cell_form = conductivity[2][3] / u[2][3]

    self.assertLess(right[2][2], insulating_side)
    # The shipped form took the owner CV's own conductivity, which for air
    # against an exterior wall is off by three orders of magnitude.
    self.assertLess(right[2][2] * 100.0, owner_cell_form)

  def test_interior_face_conductance_is_symmetric_across_every_face(self):
    """Both CVs sharing a face have to see the same conductance."""
    conductivity = self._two_material_conductivity()
    u, v, (left, right, top, bottom), _ = self._conductances(conductivity)

    # Only the CVs of the block take part in the balance; the ones around it
    # are exterior CVs that are held at the ambient temperature.
    inside = np.full(self.SHAPE, False)
    inside[
        self.BLOCK_ROWS[0] : self.BLOCK_ROWS[1] + 1,
        self.BLOCK_COLS[0] : self.BLOCK_COLS[1] + 1,
    ] = True
    horizontal = inside[:, :-1] & inside[:, 1:]
    vertical = inside[:-1, :] & inside[1:, :]

    # The balance uses the product of the face area and the conductance, and
    # a left or right face has area v * z while a top or bottom face has u * z.
    np.testing.assert_allclose(
        (v * right)[:, :-1][horizontal],
        (v * left)[:, 1:][horizontal],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        (u * bottom)[:-1, :][vertical], (u * top)[1:, :][vertical], rtol=1e-5
    )

  def test_shared_face_area_is_the_narrower_of_the_two_cvs(self):
    """Where a half width CV meets a full width one, both see one area."""
    # Cutting the top right corner off the block leaves (2, 4) without a top
    # neighbor, so it is half as tall as the interior CV (2, 3) beside it and
    # the two of them only touch over the shorter of their two faces.
    mask = np.full(self.SHAPE, False)
    mask[1:5, 1:6] = True
    mask[1, 4:6] = False
    boundary_cv_mapping = tf_simulator_py.get_cv_mapping(_mask_neighbors(mask))
    conductivity = np.full(self.SHAPE, 2.0, dtype=np.float32)
    t_u, t_v = tf_simulator_py.get_cv_dimension_tensors(
        self.CV_SIZE_M, boundary_cv_mapping, self.SHAPE
    )
    u, v = t_u.numpy(), t_v.numpy()
    left, right, _, _ = (
        tensor.numpy()
        for tensor in tf_simulator_py.get_oriented_face_conductance_tensors(
            conductivity, t_u, t_v, boundary_cv_mapping
        )
    )

    self.assertAlmostEqual(v[2][3], self.CV_SIZE_M, places=6)
    self.assertAlmostEqual(v[2][4], self.CV_SIZE_M / 2.0, places=6)
    series = 1.0 / (
        u[2][3] / (2.0 * conductivity[2][3])
        + u[2][4] / (2.0 * conductivity[2][4])
    )
    # The balance multiplies by the owner's own face area, so the CV that
    # overhangs the shared face carries the halved conductance.
    self.assertAlmostEqual(right[2][3], series / 2.0, places=4)
    self.assertAlmostEqual(left[2][4], series, places=4)
    self.assertAlmostEqual(
        v[2][3] * right[2][3], v[2][4] * left[2][4], places=5
    )

  def test_uniform_material_keeps_the_owner_cell_conductance(self):
    """On a uniform grid of a single material both forms agree."""
    conductivity = np.full(self.SHAPE, 2.0, dtype=np.float32)
    u, v, (left, right, top, bottom), _ = self._conductances(conductivity)

    # (2, 2), (2, 3) and (3, 3) are interior CVs, so all of these faces are
    # between two full width CVs.
    self.assertAlmostEqual(right[2][2], conductivity[2][2] / u[2][2], places=4)
    self.assertAlmostEqual(left[2][3], conductivity[2][3] / u[2][3], places=4)
    self.assertAlmostEqual(bottom[2][3], conductivity[2][3] / v[2][3], places=4)
    self.assertAlmostEqual(top[3][3], conductivity[3][3] / v[3][3], places=4)

  def test_boundary_cv_face_conductance_accounts_for_the_half_width(self):
    """A boundary CV is half as wide, so its half cell is half as resistive."""
    conductivity = np.full(self.SHAPE, 2.0, dtype=np.float32)
    u, _, (_, right, _, _), _ = self._conductances(conductivity)

    # (2, 1) is a left edge boundary CV, half as wide as the interior CV
    # (2, 2) next to it.
    self.assertAlmostEqual(u[2][1], self.CV_SIZE_M / 2.0, places=6)
    expected = 1.0 / (
        u[2][1] / (2.0 * conductivity[2][1])
        + u[2][2] / (2.0 * conductivity[2][2])
    )
    self.assertAlmostEqual(right[2][1], expected, places=4)

  def test_no_interior_conductance_on_faces_that_open_onto_the_ambient(self):
    conductivity = self._two_material_conductivity()
    (
        _,
        _,
        (left, right, top, bottom),
        (
            ambient_left,
            ambient_right,
            ambient_top,
            ambient_bottom,
        ),
    ) = self._conductances(conductivity)

    # (2, 1) is a left edge CV and (1, 1) is a top left corner CV.
    self.assertEqual(left[2][1], 0.0)
    self.assertGreater(ambient_left[2][1], 0.0)
    self.assertEqual(left[1][1], 0.0)
    self.assertEqual(top[1][1], 0.0)
    self.assertGreater(ambient_left[1][1], 0.0)
    self.assertGreater(ambient_top[1][1], 0.0)

    # A face is either an interior face or an ambient face, never both.
    for interior, ambient in (
        (left, ambient_left),
        (right, ambient_right),
        (top, ambient_top),
        (bottom, ambient_bottom),
    ):
      np.testing.assert_array_equal(interior * ambient, np.zeros(self.SHAPE))

  def test_ambient_conductance_adds_the_outer_half_cell(self):
    conductivity = self._two_material_conductivity()
    (
        u,
        v,
        _,
        (
            ambient_left,
            _,
            ambient_top,
            _,
        ),
    ) = self._conductances(conductivity)

    expected_left = 1.0 / (
        1.0 / self.CONVECTION_COEFFICIENT + u[2][1] / (2.0 * conductivity[2][1])
    )
    self.assertAlmostEqual(ambient_left[2][1], expected_left, places=5)

    expected_top = 1.0 / (
        1.0 / self.CONVECTION_COEFFICIENT + v[1][2] / (2.0 * conductivity[1][2])
    )
    self.assertAlmostEqual(ambient_top[1][2], expected_top, places=5)

  def test_ambient_conductance_stays_below_the_film_coefficient(self):
    """Two resistances in series conduct less than either one alone."""
    conductivity = self._two_material_conductivity()
    _, _, _, ambient = self._conductances(conductivity)

    for tensor in ambient:
      active = tensor > 0.0
      self.assertTrue(np.any(active))
      self.assertTrue(np.all(tensor[active] < self.CONVECTION_COEFFICIENT))

    # On the insulating side the half cell dominates the film by a wide
    # margin, so the exchange with the ambient is nothing like h_ext.
    ambient_left = ambient[0]
    self.assertLess(ambient_left[2][1], self.CONVECTION_COEFFICIENT / 10.0)

  def test_ambient_conductance_approaches_the_film_coefficient(self):
    """A conductive boundary CV leaves the film as the only resistance."""
    conductivity = np.full(self.SHAPE, 5.0e5, dtype=np.float32)
    _, _, _, (ambient_left, _, _, _) = self._conductances(conductivity)

    self.assertAlmostEqual(
        ambient_left[2][1], self.CONVECTION_COEFFICIENT, places=3
    )


if __name__ == "__main__":
  absltest.main()
