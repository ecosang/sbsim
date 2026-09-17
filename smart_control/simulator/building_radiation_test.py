"""Building radiation tests."""

import os

from absl.testing import absltest
import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd

from smart_control.simulator import conftest

TEST_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "building_radiation_test_data"
)

INDEXED_FLOOR_PLAN = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -1],
    [-1, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, 0, 0, 0, 0, 0, 0, -3, -3, -1],
    [-1, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -1],
    [-1, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
])


def assert_data_file_values_equal(
    data: np.array, csv_filename: str, precision: int = 3
):
  data_filepath = os.path.join(TEST_DATA_DIR, csv_filename)
  # temporarily uncomment to update the test data file, when applicable:
  # df = pd.DataFrame(data)
  # df.to_csv(data_filepath, index=False, header=False)

  df = pd.read_csv(data_filepath, header=None)
  expected = df.to_numpy()
  assert_array_almost_equal(
      np.round(data, precision),
      np.round(expected, precision),
  )


class BuildingRadiationScriptFTest(absltest.TestCase):

  def setUp(self):
    self.building = conftest.create_building_with_radiative_properties(
        view_factor_method="ScriptF"
    )
    self.building_with_interior_mass = (
        conftest.create_building_with_radiative_properties(
            view_factor_method="ScriptF",
            include_interior_mass=True,
        )
    )

  def test_interior_radiative_heat_transfer(self):
    with self.subTest("indexed_floor_plan"):
      assert_array_almost_equal(
          np.round(self.building.indexed_floor_plan, 3),
          np.round(INDEXED_FLOOR_PLAN, 3),
      )

    with self.subTest("view factor"):
      props = self.building.interior_wall_vf
      assert_data_file_values_equal(props, "expected_interior_wall_vf.csv")

    with self.subTest("alpha"):
      props = self.building._alpha
      self.assertEqual(props.shape, (23, 12))
      assert_data_file_values_equal(props, "alpha.csv")

    with self.subTest("epsilon"):
      props = self.building._epsilon
      self.assertEqual(props.shape, (23, 12))
      assert_data_file_values_equal(props, "epsilon.csv")

    with self.subTest("tau"):
      props = self.building._tau
      self.assertEqual(props.shape, (23, 12))
      assert_data_file_values_equal(props, "tau.csv")

    with self.subTest("ifa_inv"):
      results = self.building.ifa_inv
      self.assertEqual(results.shape, (50, 50))
      assert_data_file_values_equal(results, "ifa_inv.csv", precision=5)

    with self.subTest("view factor with interior mass"):
      props = self.building_with_interior_mass.interior_wall_vf
      assert_data_file_values_equal(
          props, "expected_interior_wall_vf_interior_mass.csv"
      )


class PartitionedRadiativeEnclosureTest(absltest.TestCase):

  def test_partitioned_and_full_area_ratio_operators_match(self):
    """Enclosure blocks reproduce the full matrix without allocating it."""
    for include_interior_mass in (False, True):
      with self.subTest(include_interior_mass=include_interior_mass):
        full = conftest.create_building_with_radiative_properties(
            floor_plan=conftest.TWO_ENCLOSURE_FLOOR_PLAN,
            include_interior_mass=include_interior_mass,
        )
        partitioned = conftest.create_building_with_radiative_properties(
            floor_plan=conftest.TWO_ENCLOSURE_FLOOR_PLAN,
            include_interior_mass=include_interior_mass,
            partition_radiative_enclosures=True,
        )

        self.assertLen(partitioned.radiative_enclosure_indices, 2)
        self.assertIsNone(partitioned.interior_wall_vf)
        self.assertIsNone(partitioned.ifa_inv)

        node_count = full.ifa_inv.shape[0]
        assembled_ifa = np.zeros_like(full.ifa_inv)
        assembled_vf = np.zeros_like(full.interior_wall_vf)
        for indices, view_factors, ifa_inv in zip(
            partitioned.radiative_enclosure_indices,
            partitioned.interior_wall_vf_by_enclosure,
            partitioned.ifa_inv_by_enclosure,
        ):
          assembled_ifa[np.ix_(indices, indices)] = ifa_inv
          assembled_vf[np.ix_(indices, indices)] = view_factors

        np.testing.assert_allclose(assembled_vf, full.interior_wall_vf)
        np.testing.assert_allclose(assembled_ifa, full.ifa_inv, atol=1e-12)
        self.assertLess(
            sum(block.size for block in partitioned.ifa_inv_by_enclosure),
            node_count**2,
        )

        temperatures = 285.0 + np.arange(full.temp.size).reshape(
            full.temp.shape
        )
        if include_interior_mass:
          mass_temperatures = 290.0 + np.arange(full.temp.size).reshape(
              full.temp.shape
          )
          full.interior_mass_temp[:] = mass_temperatures
          partitioned.interior_mass_temp[:] = mass_temperatures

        np.testing.assert_allclose(
            partitioned.apply_longwave_interior_radiative_heat_transfer(
                temperatures
            ),
            full.apply_longwave_interior_radiative_heat_transfer(temperatures),
            atol=1e-10,
        )
        partitioned_terms = (
            partitioned.get_longwave_interior_radiative_exchange_terms(
                temperatures
            )
        )
        full_terms = full.get_longwave_interior_radiative_exchange_terms(
            temperatures
        )
        for partitioned_term, full_term in zip(partitioned_terms, full_terms):
          np.testing.assert_allclose(partitioned_term, full_term, atol=1e-10)


if __name__ == "__main__":
  absltest.main()
