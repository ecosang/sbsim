"""Tests checking the numpy and TF radiation solvers against each other."""

from unittest import mock

from absl.testing import absltest
import numpy as np
import pandas as pd

from smart_control.simulator import simulator_flexible_floor_plan as flexible_py
from smart_control.simulator import simulator_flexible_floor_plan_test
from smart_control.simulator import tf_simulator as tf_simulator_py
from smart_control.simulator import weather_controller as weather_controller_py


class RadiationLinearizationTest(absltest.TestCase):
  r"""Checks the exact split of the interior longwave exchange.

  Both solvers rewrite $\sum_j M_{ij} \sigma (T_j^4 - T_i^4)$ as
  $\sum_j M_{ij} h_{ij} (T_j - T_i)$ with $h_{ij}$ the exact secant of
  $\sigma T^4$, and move the part proportional to the CV's own temperature onto
  the denominator. That is an identity, so the fixed point is whatever the
  explicit form had; what changes is that the sweep can no longer run away. Left
  fully explicit the same balance reaches NaN at a day-long step, which is what
  the large time step case below guards against.
  """

  AMBIENT_TEMPERATURE = 315.0
  CONVECTION_COEFFICIENT = 12.0

  def _create_simulator(self, simulator_class, time_step_sec, iteration_limit):
    """Returns a simulator of the given class over a radiating building."""
    fixture = (
        simulator_flexible_floor_plan_test.FlexibleFloorplanSimulatorTest()
    )
    _, building = fixture._create_simulator_and_building(  # pylint: disable=protected-access
        initial_temp=292.0,
        include_interior_mass=True,
        include_radiative_heat_transfer=True,
        convergence_threshold=0.001,
        iteration_limit=100,
    )
    return simulator_class(
        building,
        fixture._create_small_hvac(),  # pylint: disable=protected-access
        mock.create_autospec(weather_controller_py.WeatherController),
        time_step_sec,
        0.0001,
        iteration_limit,
        iteration_limit,
        pd.Timestamp("2012-12-21"),
    )

  def _relax(self, simulator, sweeps):
    """Runs a fixed number of sweeps and returns the field and the movement."""
    temperatures = simulator.building.temp.copy()
    delta = np.inf
    for _ in range(sweeps):
      temperatures, delta = simulator.update_temperature_estimates(
          temperatures,
          ambient_temperature=self.AMBIENT_TEMPERATURE,
          convection_coefficient=self.CONVECTION_COEFFICIENT,
      )
    return np.asarray(temperatures, dtype=np.float64), float(delta)

  def test_a_huge_time_step_stays_finite_in_both_solvers(self):
    """A step far past the explicit stability limit does not run away.

    With the radiative self term left in the numerator this case reached NaN
    within a handful of sweeps, because a CV's update could overshoot its
    neighbors with nothing pulling it back. Moving that term onto the
    denominator makes the update a convex combination of temperatures, so the
    field can only stay inside the envelope it started in. Convergence at this
    step size is not the point and is not asserted - staying finite is.
    """
    for simulator_class in (
        flexible_py.SimulatorFlexibleGeometries,
        tf_simulator_py.TFSimulator,
    ):
      with self.subTest(simulator_class.__name__):
        simulator = self._create_simulator(simulator_class, 864000.0, 60)
        temperatures, _ = self._relax(simulator, 60)

        self.assertTrue(np.all(np.isfinite(temperatures)))
        # The zone still has a heat source in it, so the field is not bounded
        # by the ambient. What the split does guarantee is that radiation
        # cannot be what pushes it out of a physical range.
        self.assertGreaterEqual(temperatures.min(), 250.0)
        self.assertLessEqual(temperatures.max(), 400.0)

  def test_the_converged_field_is_a_fixed_point_of_the_sweep(self):
    """Sweeping the converged field again leaves it where it is.

    The split only moves terms between the two sides of the update, so the
    field it settles on has to satisfy the balance the explicit form was
    written for. If the diagonal and the source did not come from the same
    identity, this is where the mismatch would show up as a residual movement.
    """
    for simulator_class in (
        flexible_py.SimulatorFlexibleGeometries,
        tf_simulator_py.TFSimulator,
    ):
      with self.subTest(simulator_class.__name__):
        simulator = self._create_simulator(simulator_class, 300.0, 200)
        _, delta = self._relax(simulator, 60)
        self.assertLess(delta, 1e-4)

  def test_both_solvers_agree_with_radiation_switched_on(self):
    """The tensor and iterative radiation paths reach the same field.

    The two assemble the same split in different places - one CV at a time
    against the whole grid at once - so agreeing here covers the scatter back
    onto the grid, the area weighting, and the wall/mass branch all at once.
    """
    converged = []
    for simulator_class in (
        flexible_py.SimulatorFlexibleGeometries,
        tf_simulator_py.TFSimulator,
    ):
      simulator = self._create_simulator(simulator_class, 300.0, 200)
      temperatures, _ = self._relax(simulator, 40)
      converged.append(temperatures)

    # Same tolerance as the non-radiating comparison: wide enough for the
    # float32 the tensor path assembles its coefficients in.
    np.testing.assert_allclose(converged[0], converged[1], atol=5e-2)


if __name__ == "__main__":
  absltest.main()
