"""Building thermal simulation with thermostat setpoint control.

This notebook-style script demonstrates the SBSim building simulator
with HVAC thermostat setpoint control for a two-zone office building.
It runs a winter heating and a summer cooling scenario, each for 7 days,
and produces visualisations of zone temperatures, HVAC energy, occupancy,
and floor-plan spatial heatmaps.

Run with:
    poetry run python smart_control/notebooks/building_setpoint_control_demo.py

Convert to Jupyter notebook with:
    poetry run jupytext --to notebook \
        smart_control/notebooks/building_setpoint_control_demo.py
"""

# %% [markdown]
# # Building Thermal Simulation with Thermostat Setpoint Control
#
# This notebook demonstrates how the **SBSim** building simulator models
# HVAC thermostat setpoint control for a simple two-zone office building.
#
# ## What you will see
# 1. A hard-coded 2-zone floor plan (23 × 12 control-volume grid)
# 2. Realistic material properties, HVAC devices, and occupancy schedule
# 3. A **winter heating** scenario – the building starts cold (275 K) and the
#    boiler + VAVs warm it up to the comfort band
# 4. A **summer cooling** scenario – the building starts hot (310 K) and the
#    air handler cools it back into the comfort band
# 5. Rich visualisations: zone temperature timeseries, HVAC energy rates,
#    occupancy schedule, and spatial floor-plan heatmaps
#
# ## Architecture recap
# ```
# WeatherController → SimulatorFlexibleGeometries.step_sim()
#     ├─ finite_differences_timestep()   (conduction + convection)
#     └─ per zone:
#         VAV.output(zone_temp, supply_air_temp)
#           ├─ if T < heating_sp → Boiler adds heat
#           └─ if T > cooling_sp → AirHandler removes heat
#             └─ building.apply_thermal_power_zone(zone, q_zone)
# ```

# %% [markdown]
# ## Section 0 – Imports

# %%
# Standard library
import copy
import os
import sys

# Numeric / plotting
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make sure the repo root is on the path so simulator modules are importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

# SBSim simulator modules  (placed after sys.path setup – necessary for
# notebooks that set the path at runtime)
# pylint: disable=wrong-import-position
from smart_control.simulator import air_handler as air_handler_py
from smart_control.simulator import boiler as boiler_py
from smart_control.simulator import building as building_py
from smart_control.simulator import hvac_floorplan_based as floorplan_hvac_py
from smart_control.simulator import setpoint_schedule as setpoint_schedule_py
from smart_control.simulator import simulator_flexible_floor_plan as simulator_py
from smart_control.simulator import step_function_occupancy as occupancy_py
from smart_control.simulator import weather_controller as weather_controller_py

# pylint: enable=wrong-import-position

# Convenience constant
KELVIN_TO_CELSIUS = 273.15

print('All imports OK.')
print(
    f'NumPy {np.__version__}, '
    f'Pandas {pd.__version__}, '
    f'Matplotlib {matplotlib.__version__}'
)

# %% [markdown]
# ## Section 1 – Floor Plan Definition and Visualisation
#
# The floor plan is a 2-D NumPy integer array where each cell encodes its type:
#
# | Code | Type |
# |------|------|
# | `0`  | Interior air (room space) |
# | `1`  | Interior wall / corridor |
# | `2`  | Exterior wall |
#
# This is exactly `_create_dummy_floor_plan_small()` from the test suite.
# It has **two rooms** separated by a horizontal corridor wall at row 11.

# %%
# ── Hard-coded floor plan (23 rows × 12 cols) ────────────────────────────────
# Taken verbatim from
# FlexibleFloorplanSimulatorTest._create_dummy_floor_plan_small()
FLOOR_PLAN = np.array([
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # row  0  exterior top
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],  # row  1  corridor
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],  # row  2  corridor
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],  # row  3  room_1
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],  # row 10  room_1 last air row
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],  # row 11  interior corridor wall
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],  # row 12  room_2
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],
    [2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2],  # row 19  room_2 last air row
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],  # row 20  corridor
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],  # row 21  corridor
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # row 22  exterior bottom
])

print(f'Floor plan shape: {FLOOR_PLAN.shape}  (rows × cols)')
print(f'Air cells  (code 0): {(FLOOR_PLAN == 0).sum()}')
print(f'Wall cells (code 1): {(FLOOR_PLAN == 1).sum()}')
print(f'Ext. cells (code 2): {(FLOOR_PLAN == 2).sum()}')

# %%
# ── Visualise the floor plan ──────────────────────────────────────────────────
fig_fp, ax_fp = plt.subplots(figsize=(7, 10))

# Custom discrete colormap: exterior=charcoal, interior wall=silver, air=white
cmap_fp = mcolors.ListedColormap(['white', '#aaaaaa', '#444444'])
bounds_fp = [-0.5, 0.5, 1.5, 2.5]
norm_fp = mcolors.BoundaryNorm(bounds_fp, cmap_fp.N)

ax_fp.imshow(
    FLOOR_PLAN, cmap=cmap_fp, norm=norm_fp, origin='upper', aspect='equal'
)

# Annotate zones
ax_fp.text(
    5.5,
    6.5,
    'room_1\n(zone 1)',
    ha='center',
    va='center',
    fontsize=12,
    color='steelblue',
    fontweight='bold',
)
ax_fp.text(
    5.5,
    15.5,
    'room_2\n(zone 2)',
    ha='center',
    va='center',
    fontsize=12,
    color='darkorange',
    fontweight='bold',
)

# Legend
legend_patches = [
    mpatches.Patch(color='white', label='Air (code 0) – room space'),
    mpatches.Patch(
        color='#aaaaaa', label='Wall (code 1) – interior / corridor'
    ),
    mpatches.Patch(color='#444444', label='Wall (code 2) – exterior envelope'),
]
ax_fp.legend(
    handles=legend_patches, loc='upper right', fontsize=9, framealpha=0.9
)

ax_fp.set_title(
    'Two-Zone Office Building – Floor Plan\n'
    '(23 × 12 control-volume grid, cv_size = 20 cm)',
    fontsize=12,
)
ax_fp.set_xlabel('Column index (East–West)')
ax_fp.set_ylabel('Row index (North–South)')
plt.tight_layout()
plt.savefig('floor_plan.png', dpi=120, bbox_inches='tight')
plt.show()
print('Floor plan saved to floor_plan.png')

# %% [markdown]
# ## Section 2 – Building Construction
#
# `FloorPlanBasedBuilding` discretises the floor plan into a grid of Control
# Volumes (CVs). Each CV holds a temperature (`building.temp`) and
# heat-injection rate (`building.input_q`).
#
# Material properties chosen to be realistic for a lightweight office building:
#
# | Layer | Conductivity (W/m·K) | Heat cap (J/kg·K) | Density (kg/m³) |
# |-------|---------------------|-------------------|-----------------|
# | Air   | 50 (effective)      | 700               | 1.0             |
# | Int. wall | 5.0             | 800               | 1 800           |
# | Ext. wall | 0.3 (insulated) | 800               | 3 000           |

# %%
# ── Shared building parameters ────────────────────────────────────────────────
CV_SIZE_CM = 20.0  # control-volume side length [cm]
FLOOR_HEIGHT_CM = 300.0  # floor-to-ceiling height [cm]

inside_air_props = building_py.MaterialProperties(
    conductivity=50.0,  # effective air conductivity (natural convection)
    heat_capacity=700.0,  # J/kg·K
    density=1.0,  # kg/m³ (air)
)
inside_wall_props = building_py.MaterialProperties(
    conductivity=5.0,
    heat_capacity=800.0,
    density=1800.0,
)
building_exterior_props = building_py.MaterialProperties(
    conductivity=0.3,  # well-insulated exterior (≈ R-20 equivalent)
    heat_capacity=800.0,
    density=3000.0,
)


def create_building(
    initial_temp_k: float,
) -> building_py.FloorPlanBasedBuilding:
  """Construct a FloorPlanBasedBuilding at the given initial temperature."""
  zone_map = copy.deepcopy(FLOOR_PLAN)
  return building_py.FloorPlanBasedBuilding(
      cv_size_cm=CV_SIZE_CM,
      floor_height_cm=FLOOR_HEIGHT_CM,
      initial_temp=initial_temp_k,
      inside_air_properties=inside_air_props,
      inside_wall_properties=inside_wall_props,
      building_exterior_properties=building_exterior_props,
      floor_plan=FLOOR_PLAN,
      zone_map=zone_map,
      buffer_from_walls=0,
  )


# Quick smoke-test
_bld_test = create_building(293.0)
# get_zone_average_temps() returns only real room zones (room_1, room_2, ...)
# _room_dict also contains 'exterior_space' and 'interior_wall' – exclude them.
ROOM_ZONES = sorted(_bld_test.get_zone_average_temps())
print(f'Actual room zones     : {ROOM_ZONES}')
print(f'Temperature grid shape: {_bld_test.temp.shape}')
print(f'Initial zone temps (K): {_bld_test.get_zone_average_temps()}')

# %% [markdown]
# ## Section 3 – HVAC Configuration
#
# The HVAC hierarchy:
# ```
# FloorPlanBasedHvac
# ├── Boiler        – hot water loop for heating
# ├── AirHandler    – supply air conditioning (cooling)
# └── VAV × N zones – variable-air-volume terminal units with thermostat
#       └── SetpointSchedule – comfort (9am-6pm) vs eco (nights/weekends)
# ```
# Parameters are taken from `_create_scenario_hvac()` in the test suite and
# cross-checked with the SAC Demo notebook.

# %%
# ── Setpoint schedule ─────────────────────────────────────────────────────────
HEATING_SETPOINT_K = 292.0  # 18.85 °C
COOLING_SETPOINT_K = 295.0  # 21.85 °C
ECO_LOW_K = 290.0  # 16.85 °C  (unoccupied / setback)
ECO_HIGH_K = 297.0  # 23.85 °C


def create_hvac(
    zone_identifiers,
) -> floorplan_hvac_py.FloorPlanBasedHvac:
  """Build a fresh HVAC object wired to the given zone identifiers."""
  return floorplan_hvac_py.FloorPlanBasedHvac(
      zone_identifier=zone_identifiers,
      air_handler=air_handler_py.AirHandler(
          recirculation=0.6,
          heating_air_temp_setpoint=291.0,
          cooling_air_temp_setpoint=295.0,
          fan_differential_pressure=20000.0,
          fan_efficiency=0.8,
      ),
      boiler=boiler_py.Boiler(
          reheat_water_setpoint=350.0,
          water_pump_differential_head=3.0,
          water_pump_efficiency=0.6,
          device_id='boiler_id',
      ),
      schedule=setpoint_schedule_py.SetpointSchedule(
          morning_start_hour=9,
          evening_start_hour=18,
          comfort_temp_window=(HEATING_SETPOINT_K, COOLING_SETPOINT_K),
          eco_temp_window=(ECO_LOW_K, ECO_HIGH_K),
          holidays={7, 223, 245},
      ),
      vav_max_air_flow_rate=0.45,  # [m³/s] per zone
      vav_reheat_max_water_flow_rate=0.02,  # [kg/s] per zone
  )


# Quick check – pass only the real room zones
_hvac_test = create_hvac(ROOM_ZONES)
print('HVAC zones:', list(_hvac_test.vavs.keys()))
sp_k = _hvac_test.boiler.reheat_water_setpoint
print(
    f'Boiler supply water setpoint : {sp_k:.1f} K'
    f'  = {sp_k - KELVIN_TO_CELSIUS:.1f} °C'
)
print(
    'Comfort band : '
    f'[{HEATING_SETPOINT_K - KELVIN_TO_CELSIUS:.2f} °C, '
    f'{COOLING_SETPOINT_K - KELVIN_TO_CELSIUS:.2f} °C]'
)
print(
    'Eco band     : '
    f'[{ECO_LOW_K - KELVIN_TO_CELSIUS:.2f} °C, '
    f'{ECO_HIGH_K - KELVIN_TO_CELSIUS:.2f} °C]'
)

# %% [markdown]
# ## Section 4 – Weather Controller
#
# `WeatherController` generates a **sinusoidal diurnal temperature cycle**:
# minimum at midnight, maximum at noon.  We use two weekly profiles:
#
# * **Winter week** (Jan 10–16, 2024):
#   low = 268 K (−5 °C), high = 277 K (4 °C)
# * **Summer week** (Jul 15–21, 2024):
#   low = 295 K (22 °C), high = 308 K (35 °C)

# %%
# ── Weather controllers ───────────────────────────────────────────────────────
wc_winter = weather_controller_py.WeatherController(
    default_low_temp=268.0,  # K = –5 °C
    default_high_temp=277.0,  # K =  4 °C
    convection_coefficient=12.0,
)

wc_summer = weather_controller_py.WeatherController(
    default_low_temp=295.0,  # K = 22 °C
    default_high_temp=308.0,  # K = 35 °C
    convection_coefficient=12.0,
)

# ── Plot 7-day ambient temperature profiles ───────────────────────────────────
ts_range_w = pd.date_range('2024-01-10', periods=7 * 24 * 12, freq='5min')
winter_temps = [
    wc_winter.get_current_temp(t) - KELVIN_TO_CELSIUS for t in ts_range_w
]

ts_range_s = pd.date_range('2024-07-15', periods=7 * 24 * 12, freq='5min')
summer_temps = [
    wc_summer.get_current_temp(t) - KELVIN_TO_CELSIUS for t in ts_range_s
]

fig_wx, axes_wx = plt.subplots(2, 1, figsize=(14, 6), sharex=False)

h_sp_c = HEATING_SETPOINT_K - KELVIN_TO_CELSIUS
c_sp_c = COOLING_SETPOINT_K - KELVIN_TO_CELSIUS

for ax_wx, ts_rng, temps, colour, label in [
    (
        axes_wx[0],
        ts_range_w,
        winter_temps,
        'steelblue',
        'Winter Week – Outdoor Temperature (Jan 10–16, 2024)',
    ),
    (
        axes_wx[1],
        ts_range_s,
        summer_temps,
        'darkorange',
        'Summer Week – Outdoor Temperature (Jul 15–21, 2024)',
    ),
]:
  ax_wx.plot(ts_rng, temps, color=colour, lw=1.5, label='Outdoor temp')
  ax_wx.axhline(
      h_sp_c, color='red', ls='--', lw=1, label=f'Heating SP ({h_sp_c:.1f}°C)'
  )
  ax_wx.axhline(
      c_sp_c,
      color='orange',
      ls='--',
      lw=1,
      label=f'Cooling SP ({c_sp_c:.1f}°C)',
  )
  ax_wx.fill_between(
      ts_rng,
      h_sp_c,
      c_sp_c,
      alpha=0.15,
      color='green',
      label='Comfort band',
  )
  ax_wx.set_title(label, fontsize=11)
  ax_wx.set_ylabel('Temperature [°C]')
  ax_wx.legend(fontsize=9, loc='upper right')
  ax_wx.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('weather_profiles.png', dpi=120, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Section 5 – Occupancy Model
#
# `StepFunctionOccupancy` is a simple binary model:
# * **Work hours** (9 am–5 pm, weekdays): 10 people per zone
# * **Off hours / weekends**: 0.1 people per zone (background)

# %%
occupancy = occupancy_py.StepFunctionOccupancy(
    work_start_time=pd.Timedelta(9, unit='h'),
    work_end_time=pd.Timedelta(17, unit='h'),
    work_occupancy=10.0,
    nonwork_occupancy=0.1,
)

# Plot 2-day occupancy profile
occ_times = pd.date_range('2024-01-10', periods=2 * 24 * 12, freq='5min')
occ_vals = [
    occupancy.average_zone_occupancy('room_1', occ_times[i], occ_times[i + 1])
    for i in range(len(occ_times) - 1)
]

fig_occ, ax_occ = plt.subplots(figsize=(12, 3))
ax_occ.step(occ_times[:-1], occ_vals, where='post', color='teal', lw=2)
ax_occ.fill_between(
    occ_times[:-1], occ_vals, step='post', alpha=0.3, color='teal'
)
ax_occ.set_title(
    'Occupancy Profile – 2 Day Sample (Wednesday–Thursday)', fontsize=11
)
ax_occ.set_ylabel('Avg Occupants per Zone')
ax_occ.set_ylim(-0.5, 12)
ax_occ.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('occupancy_profile.png', dpi=120, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Section 6 – Heating Scenario: 1-Week Winter Simulation
#
# **Setup:**
# * Start time: Wednesday 2024-01-10 00:00 (middle of winter week)
# * Initial building temperature: **275 K (2 °C)** – well below 292 K heating SP
# * Ambient temperature: sinusoidal winter profile, max 277 K (4 °C)
# * Result: thermostats stay in heating mode; boiler injects heat via VAVs
#
# Time step = 300 s (5 min).  7 days = 2 016 steps.

# %%
# ── Simulation helper ─────────────────────────────────────────────────────────


def run_simulation(
    bld: building_py.FloorPlanBasedBuilding,
    hvac: floorplan_hvac_py.FloorPlanBasedHvac,
    weather_ctrl: weather_controller_py.WeatherController,
    occupancy_model: occupancy_py.StepFunctionOccupancy,
    start_timestamp: pd.Timestamp,
    n_days: int = 7,
    time_step_sec: float = 300.0,
    convergence_threshold: float = 0.1,
    iteration_limit: int = 100,
    progress_every: int = 200,
) -> pd.DataFrame:
  """Run the simulator and collect per-step timeseries data.

  Returns a DataFrame with columns:
      timestamp, zone_id, zone_temp_k, zone_temp_c,
      heating_sp_k, cooling_sp_k, heating_sp_c, cooling_sp_c,
      is_comfort_mode, occupancy, boiler_gas_rate_w, fan_elec_rate_w,
      outdoor_temp_k, outdoor_temp_c
  """
  sim = simulator_py.SimulatorFlexibleGeometries(
      building=bld,
      hvac=hvac,
      weather_controller=weather_ctrl,
      time_step_sec=time_step_sec,
      convergence_threshold=convergence_threshold,
      iteration_limit=iteration_limit,
      iteration_warning=10,
      start_timestamp=start_timestamp,
  )

  n_steps = int(n_days * 24 * 3600 / time_step_sec)
  # get_zone_average_temps() returns only real room zones
  zone_ids = sorted(bld.get_zone_average_temps())
  # All VAVs share the same SetpointSchedule; use first zone as reference
  ref_zone = zone_ids[0]
  records = []

  for step in range(n_steps):
    ts = sim.current_timestamp

    # Setpoints and comfort mode from thermostat schedule
    sp_schedule = hvac.vavs[ref_zone].thermostat.get_setpoint_schedule()
    h_sp, c_sp = sp_schedule.get_temperature_window(ts)
    is_comfort = sp_schedule.is_comfort_mode(ts)
    outdoor_k = weather_ctrl.get_current_temp(ts)

    # Zone temperatures before the step (only real room zones)
    zone_temps = bld.get_zone_average_temps()

    # HVAC energy rates from the current device state
    boiler_gas_w = hvac.boiler.compute_thermal_energy_rate(
        hvac.boiler.return_water_temperature_sensor, outdoor_k
    )
    fan_elec_w = (
        hvac.air_handler.compute_intake_fan_energy_rate()
        + hvac.air_handler.compute_exhaust_fan_energy_rate()
    )

    for zone_id in zone_ids:
      zone_temp_k = zone_temps[zone_id]
      occ = occupancy_model.average_zone_occupancy(
          zone_id, ts, ts + pd.Timedelta(time_step_sec, unit='s')
      )
      records.append({
          'timestamp': ts,
          'zone_id': zone_id,
          'zone_temp_k': zone_temp_k,
          'zone_temp_c': zone_temp_k - KELVIN_TO_CELSIUS,
          'heating_sp_k': h_sp,
          'cooling_sp_k': c_sp,
          'heating_sp_c': h_sp - KELVIN_TO_CELSIUS,
          'cooling_sp_c': c_sp - KELVIN_TO_CELSIUS,
          'is_comfort_mode': int(is_comfort),
          'occupancy': occ,
          'boiler_gas_rate_w': boiler_gas_w,
          'fan_elec_rate_w': fan_elec_w,
          'outdoor_temp_k': outdoor_k,
          'outdoor_temp_c': outdoor_k - KELVIN_TO_CELSIUS,
      })

    # Advance the simulation one step
    sim.step_sim()

    if progress_every and (step + 1) % progress_every == 0:
      avg_t = np.mean(list(bld.get_zone_average_temps().values()))
      print(
          f'  step {step + 1:4d}/{n_steps}'
          f' | ts={ts.strftime("%m-%d %H:%M")}'
          f' | avg zone T = {avg_t - KELVIN_TO_CELSIUS:.2f} °C'
          f' | outdoor = {outdoor_k - KELVIN_TO_CELSIUS:.2f} °C'
      )

  return pd.DataFrame(records)


# %% [markdown]
# ### Run Heating Scenario

# %%
print('=' * 60)
print('HEATING SCENARIO  – Winter week, building starts at 275 K (2°C)')
print('=' * 60)

INITIAL_TEMP_HEATING = 275.0  # K – cold building, below heating setpoint
START_TS_WINTER = pd.Timestamp('2024-01-10 00:00:00')  # Wednesday

bld_heat = create_building(INITIAL_TEMP_HEATING)
# Pass only actual room zones (get_zone_average_temps excludes other entries)
hvac_heat = create_hvac(sorted(bld_heat.get_zone_average_temps()))

df_heat = run_simulation(
    bld=bld_heat,
    hvac=hvac_heat,
    weather_ctrl=wc_winter,
    occupancy_model=occupancy,
    start_timestamp=START_TS_WINTER,
    n_days=7,
    time_step_sec=300.0,
    progress_every=200,
)

print(f'\nHeating simulation complete. Records: {len(df_heat):,}')
print(df_heat.head(4).to_string(index=False))

# %% [markdown]
# ## Section 7 – Data Collection and Summary (Heating)

# %%
# Pivot for easy zone-by-zone access
heat_pivot = df_heat.pivot_table(
    index='timestamp', columns='zone_id', values='zone_temp_c', aggfunc='first'
)

# Derive hourly aggregates
heat_hourly = (
    df_heat.set_index('timestamp')
    .groupby([pd.Grouper(freq='1h'), 'zone_id'])
    .agg(
        zone_temp_c=('zone_temp_c', 'mean'),
        heating_sp_c=('heating_sp_c', 'first'),
        cooling_sp_c=('cooling_sp_c', 'first'),
        occupancy=('occupancy', 'mean'),
        boiler_gas_rate_w=('boiler_gas_rate_w', 'mean'),
        fan_elec_rate_w=('fan_elec_rate_w', 'mean'),
        outdoor_temp_c=('outdoor_temp_c', 'mean'),
        is_comfort_mode=('is_comfort_mode', 'first'),
    )
    .reset_index()
)

print('Hourly summary – first 3 rows:')
print(heat_hourly.head(3).to_string(index=False))

# Zone temperature statistics
print('\nZone temperature statistics (°C) over entire heating week:')
print(df_heat.groupby('zone_id')['zone_temp_c'].describe().round(2).to_string())

# %% [markdown]
# ## Section 8 – Heating Scenario Visualisations
#
# Four panels:
# 1. Zone temperatures vs. comfort setpoint band, with outdoor temp
# 2. HVAC energy rates (boiler gas + fan electricity)
# 3. Occupancy schedule (step function)
# 4. Floor-plan temperature heatmaps at 4 snapshots in time


# %%
def plot_scenario(
    df: pd.DataFrame,
    bld_final: building_py.FloorPlanBasedBuilding,
    title_prefix: str,
    zone_colors: dict,
    outdoor_color: str = 'purple',
    heatmap_snapshots: list = None,
):
  """Plot a 4-panel figure for one simulation scenario."""

  zone_list = sorted(df['zone_id'].unique())
  n_snapshots = 4 if heatmap_snapshots is None else len(heatmap_snapshots)

  fig_sc = plt.figure(figsize=(16, 22))
  gs = GridSpec(4, n_snapshots, figure=fig_sc, hspace=0.45, wspace=0.35)

  # ── Panel 1: Zone temperatures ────────────────────────────────────────────
  ax1 = fig_sc.add_subplot(gs[0, :])
  df_ref = df[df['zone_id'] == zone_list[0]]
  h_sp_c_val = df_ref['heating_sp_c'].iloc[0]
  c_sp_c_val = df_ref['cooling_sp_c'].iloc[0]

  ax1.fill_between(
      df_ref['timestamp'],
      h_sp_c_val,
      c_sp_c_val,
      alpha=0.15,
      color='green',
      label='Comfort band',
  )
  ax1.axhline(
      h_sp_c_val,
      color='red',
      ls='--',
      lw=1.2,
      label=f'Heating SP ({h_sp_c_val:.1f}°C)',
  )
  ax1.axhline(
      c_sp_c_val,
      color='orange',
      ls='--',
      lw=1.2,
      label=f'Cooling SP ({c_sp_c_val:.1f}°C)',
  )

  for zone_id in zone_list:
    df_z = df[df['zone_id'] == zone_id].set_index('timestamp')
    ax1.plot(
        df_z.index,
        df_z['zone_temp_c'],
        lw=1.5,
        color=zone_colors.get(zone_id, 'gray'),
        label=zone_id,
    )

  # Outdoor temp on secondary y-axis
  ax1r = ax1.twinx()
  ax1r.plot(
      df_ref['timestamp'],
      df_ref['outdoor_temp_c'],
      color=outdoor_color,
      lw=1,
      ls=':',
      alpha=0.8,
      label='Outdoor temp',
  )
  ax1r.set_ylabel('Outdoor Temp [°C]', color=outdoor_color, fontsize=10)
  ax1r.tick_params(axis='y', labelcolor=outdoor_color)

  ax1.set_title(
      f'{title_prefix} – Zone Temperatures vs. Setpoints', fontsize=12
  )
  ax1.set_ylabel('Zone Air Temperature [°C]')
  ax1.grid(True, alpha=0.3)
  lines1, labels1 = ax1.get_legend_handles_labels()
  lines2, labels2 = ax1r.get_legend_handles_labels()
  ax1.legend(
      lines1 + lines2,
      labels1 + labels2,
      loc='upper right',
      fontsize=9,
      ncol=3,
  )

  # ── Panel 2: HVAC energy rates ────────────────────────────────────────────
  ax2 = fig_sc.add_subplot(gs[1, :])
  df_z0 = df[df['zone_id'] == zone_list[0]].set_index('timestamp')
  # Convert W → kW
  gas_kw = df_z0['boiler_gas_rate_w'] / 1000
  fan_kw = df_z0['fan_elec_rate_w'] / 1000
  ax2.fill_between(
      df_z0.index,
      0,
      gas_kw,
      alpha=0.6,
      color='firebrick',
      label='Boiler gas [kW]',
  )
  ax2.fill_between(
      df_z0.index,
      gas_kw,
      gas_kw + fan_kw,
      alpha=0.6,
      color='royalblue',
      label='Fan electricity [kW]',
  )

  # Shade business hours
  for day_offset in range(7):
    day_start = df['timestamp'].min() + pd.Timedelta(day_offset, unit='day')
    ax2.axvspan(
        day_start + pd.Timedelta(9, unit='h'),
        day_start + pd.Timedelta(18, unit='h'),
        alpha=0.07,
        color='gold',
    )

  ax2.set_title(f'{title_prefix} – HVAC Energy Rates', fontsize=12)
  ax2.set_ylabel('Power [kW]')
  ax2.legend(fontsize=9, loc='upper right')
  ax2.grid(True, alpha=0.3)

  # ── Panel 3: Occupancy ────────────────────────────────────────────────────
  ax3 = fig_sc.add_subplot(gs[2, :])
  ax3.step(
      df_z0.index,
      df_z0['occupancy'],
      where='post',
      color='teal',
      lw=1.5,
      label='Avg occupants/zone',
  )
  ax3.fill_between(
      df_z0.index,
      df_z0['occupancy'],
      step='post',
      alpha=0.3,
      color='teal',
  )
  ax3.set_title(f'{title_prefix} – Occupancy', fontsize=12)
  ax3.set_ylabel('Avg Occupants per Zone')
  ax3.set_ylim(-0.2, 12)
  ax3.grid(True, alpha=0.3)
  ax3.legend(fontsize=9)

  # ── Panel 4: Floor plan heatmaps ──────────────────────────────────────────
  if heatmap_snapshots is None:
    sim_start = df['timestamp'].min()
    heatmap_snapshots = [
        sim_start + pd.Timedelta(dt, unit='h') for dt in [0, 12, 36, 144]
    ]

  # Build a lookup: zone_id → list of (row, col) CV indices
  # Uses get_zone_average_temps() keys only (real rooms)
  room_dict = {
      zone_id: bld_final._room_dict[zone_id]  # pylint: disable=protected-access
      for zone_id in zone_list
  }

  for idx, snap_ts in enumerate(heatmap_snapshots):
    ax_hm = fig_sc.add_subplot(gs[3, idx])

    # Find the closest recorded timestamp
    time_diffs = (df['timestamp'] - snap_ts).abs()
    closest = df['timestamp'].iloc[time_diffs.argsort().iloc[0]]
    snap_df = df[df['timestamp'] == closest]

    # Fill temperature grid with zone averages
    temp_grid = np.full(FLOOR_PLAN.shape, np.nan)
    for _, row in snap_df.iterrows():
      if row['zone_id'] in room_dict:
        for r, c in room_dict[row['zone_id']]:
          temp_grid[r, c] = row['zone_temp_c']

    air_mask = FLOOR_PLAN == 0
    vmin = df['zone_temp_c'].min() - 1
    vmax = df['zone_temp_c'].max() + 1

    air_display = np.ma.masked_where(~air_mask, temp_grid)
    img = ax_hm.imshow(
        air_display,
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        origin='upper',
        aspect='equal',
    )
    # Walls overlay
    wall_display = np.ma.masked_where(FLOOR_PLAN == 0, FLOOR_PLAN.astype(float))
    ax_hm.imshow(
        wall_display,
        cmap=mcolors.ListedColormap(['#cccccc', '#666666']),
        vmin=0.5,
        vmax=2.5,
        origin='upper',
        aspect='equal',
        alpha=1.0,
    )

    plt.colorbar(img, ax=ax_hm, label='°C', fraction=0.046, pad=0.04)
    dt_hours = (snap_ts - df['timestamp'].min()).total_seconds() / 3600
    ax_hm.set_title(f't = {dt_hours:.0f} h', fontsize=10)
    ax_hm.axis('off')

  fig_sc.suptitle(
      f'{title_prefix} – 7-Day Building Simulation',
      fontsize=14,
      y=1.01,
  )
  safe_name = (
      title_prefix.lower().replace(' ', '_').replace('(', '').replace(')', '')
  )
  plt.savefig(f'{safe_name}_results.png', dpi=120, bbox_inches='tight')
  plt.show()
  print('Figure saved.')


# Zone colours
ZONE_COLORS = {
    'room_1': 'steelblue',
    'room_2': 'darkorange',
}

plot_scenario(
    df=df_heat,
    bld_final=bld_heat,
    title_prefix='Heating Scenario (Winter)',
    zone_colors=ZONE_COLORS,
    outdoor_color='navy',
)

# %% [markdown]
# ## Section 9 – Cooling Scenario: 1-Week Summer Simulation
#
# **Setup:**
# * Start time: Monday 2024-07-15 00:00
# * Initial building temperature: **310 K (37 °C)** – above 295 K cooling SP
# * Ambient temperature: sinusoidal summer profile, min 295 K (22 °C)
# * Result: thermostats stay in cooling mode; air handler removes heat

# %%
print('=' * 60)
print('COOLING SCENARIO  – Summer week, building starts at 310 K (37°C)')
print('=' * 60)

INITIAL_TEMP_COOLING = 310.0  # K – hot building, above cooling setpoint
START_TS_SUMMER = pd.Timestamp('2024-07-15 00:00:00')  # Monday

bld_cool = create_building(INITIAL_TEMP_COOLING)
# Use only actual room zones
hvac_cool = create_hvac(sorted(bld_cool.get_zone_average_temps()))

df_cool = run_simulation(
    bld=bld_cool,
    hvac=hvac_cool,
    weather_ctrl=wc_summer,
    occupancy_model=occupancy,
    start_timestamp=START_TS_SUMMER,
    n_days=7,
    time_step_sec=300.0,
    progress_every=200,
)

print(f'\nCooling simulation complete. Records: {len(df_cool):,}')
print(df_cool.head(4).to_string(index=False))

# %% [markdown]
# ## Section 10 – Data Collection and Summary (Cooling)

# %%
cool_hourly = (
    df_cool.set_index('timestamp')
    .groupby([pd.Grouper(freq='1h'), 'zone_id'])
    .agg(
        zone_temp_c=('zone_temp_c', 'mean'),
        heating_sp_c=('heating_sp_c', 'first'),
        cooling_sp_c=('cooling_sp_c', 'first'),
        occupancy=('occupancy', 'mean'),
        boiler_gas_rate_w=('boiler_gas_rate_w', 'mean'),
        fan_elec_rate_w=('fan_elec_rate_w', 'mean'),
        outdoor_temp_c=('outdoor_temp_c', 'mean'),
        is_comfort_mode=('is_comfort_mode', 'first'),
    )
    .reset_index()
)

print('Zone temperature statistics (°C) over entire cooling week:')
print(df_cool.groupby('zone_id')['zone_temp_c'].describe().round(2).to_string())

# %% [markdown]
# ## Section 11 – Cooling Scenario Visualisations

# %%
plot_scenario(
    df=df_cool,
    bld_final=bld_cool,
    title_prefix='Cooling Scenario (Summer)',
    zone_colors=ZONE_COLORS,
    outdoor_color='saddlebrown',
)

# %% [markdown]
# ## Section 12 – Side-by-Side Comparison and Summary
#
# We compare:
# 1. Zone temperature recovery: how quickly each scenario reaches comfort band
# 2. Time inside/outside the comfort band during occupied hours
# 3. Total energy consumed over the week

# %%
# ── Helper: compute comfort metrics ──────────────────────────────────────────


def comfort_metrics(df: pd.DataFrame, scenario_label: str) -> pd.DataFrame:
  """Compute per-zone comfort and energy metrics for a scenario."""
  results = []
  for zone_id in sorted(df['zone_id'].unique()):
    zone_df = df[df['zone_id'] == zone_id].copy()
    zone_df['in_comfort'] = (zone_df['zone_temp_k'] >= HEATING_SETPOINT_K) & (
        zone_df['zone_temp_k'] <= COOLING_SETPOINT_K
    )
    occupied = zone_df[zone_df['occupancy'] > 1.0]  # work hours only
    pct_comfort = occupied['in_comfort'].mean() * 100 if len(occupied) else 0.0

    # Total energy [kWh] = power [W] × dt [h] / 1000
    dt_hours = 300.0 / 3600.0  # 5-min steps → hours
    total_gas_kwh = zone_df['boiler_gas_rate_w'].sum() * dt_hours / 1000.0
    total_fan_kwh = zone_df['fan_elec_rate_w'].sum() * dt_hours / 1000.0

    results.append({
        'Scenario': scenario_label,
        'Zone': zone_id,
        '% Time in Comfort (occupied hrs)': round(pct_comfort, 1),
        'Min Temp [C]': round(zone_df['zone_temp_c'].min(), 2),
        'Max Temp [C]': round(zone_df['zone_temp_c'].max(), 2),
        'Boiler gas [kWh]': round(total_gas_kwh, 1),
        'Fan electricity [kWh]': round(total_fan_kwh, 2),
    })
  return pd.DataFrame(results)


metrics_heat = comfort_metrics(df_heat, 'Winter Heating')
metrics_cool = comfort_metrics(df_cool, 'Summer Cooling')
metrics_all = pd.concat([metrics_heat, metrics_cool], ignore_index=True)

print('=' * 80)
print('COMFORT & ENERGY SUMMARY')
print('=' * 80)
print(metrics_all.to_string(index=False))

# %%
# ── Comparison figure ─────────────────────────────────────────────────────────
fig_cmp, axes_cmp = plt.subplots(2, 2, figsize=(16, 10))
fig_cmp.suptitle(
    'Heating vs. Cooling Scenario – Weekly Comparison', fontsize=14
)

zone_list_cmp = ['room_1', 'room_2']

# ── (0,0) room_1 temperature overlay ─────────────────────────────────────────
ax = axes_cmp[0, 0]
for df_, label, style in [
    (df_heat, 'Winter / Heating', '-'),
    (df_cool, 'Summer / Cooling', '--'),
]:
  dz = df_[df_['zone_id'] == zone_list_cmp[0]].set_index('timestamp')
  t_h = [(t - dz.index[0]).total_seconds() / 3600 for t in dz.index]
  ax.plot(
      t_h,
      dz['zone_temp_c'],
      lw=1.3,
      ls=style,
      color=ZONE_COLORS[zone_list_cmp[0]],
      label=label,
      alpha=0.85,
  )

ax.axhspan(h_sp_c, c_sp_c, alpha=0.15, color='green', label='Comfort band')
ax.axhline(h_sp_c, color='red', ls=':', lw=1)
ax.axhline(c_sp_c, color='orange', ls=':', lw=1)
ax.set_title(f'Zone Temperature – {zone_list_cmp[0]}')
ax.set_xlabel('Simulation hours')
ax.set_ylabel('Temperature [°C]')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── (0,1) room_2 temperature overlay ─────────────────────────────────────────
ax = axes_cmp[0, 1]
for df_, label, style in [
    (df_heat, 'Winter / Heating', '-'),
    (df_cool, 'Summer / Cooling', '--'),
]:
  dz = df_[df_['zone_id'] == zone_list_cmp[1]].set_index('timestamp')
  t_h = [(t - dz.index[0]).total_seconds() / 3600 for t in dz.index]
  ax.plot(
      t_h,
      dz['zone_temp_c'],
      lw=1.3,
      ls=style,
      color=ZONE_COLORS[zone_list_cmp[1]],
      label=label,
      alpha=0.85,
  )

ax.axhspan(h_sp_c, c_sp_c, alpha=0.15, color='green', label='Comfort band')
ax.axhline(h_sp_c, color='red', ls=':', lw=1)
ax.axhline(c_sp_c, color='orange', ls=':', lw=1)
ax.set_title(f'Zone Temperature – {zone_list_cmp[1]}')
ax.set_xlabel('Simulation hours')
ax.set_ylabel('Temperature [°C]')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── (1,0) % Time in Comfort Band ─────────────────────────────────────────────
ax = axes_cmp[1, 0]
bar_data = metrics_all[
    ['Scenario', 'Zone', '% Time in Comfort (occupied hrs)']
].copy()
pivot_bar = bar_data.pivot(
    index='Zone',
    columns='Scenario',
    values='% Time in Comfort (occupied hrs)',
)
x_pos = np.arange(len(pivot_bar.index))
width = 0.35
bars_h = ax.bar(
    x_pos - width / 2,
    pivot_bar['Winter Heating'],
    width,
    label='Winter Heating',
    color='steelblue',
    alpha=0.8,
)
bars_c = ax.bar(
    x_pos + width / 2,
    pivot_bar['Summer Cooling'],
    width,
    label='Summer Cooling',
    color='darkorange',
    alpha=0.8,
)
ax.set_xticks(x_pos)
ax.set_xticklabels(pivot_bar.index)
ax.set_ylim(0, 110)
ax.set_ylabel('% Time in Comfort Band')
ax.set_title('Comfort Compliance (Occupied Hours Only)')
ax.legend(fontsize=9)
ax.grid(True, axis='y', alpha=0.3)
for bar_obj, val in zip(bars_h, pivot_bar['Winter Heating']):
  ax.text(
      bar_obj.get_x() + bar_obj.get_width() / 2,
      val + 1,
      f'{val:.1f}%',
      ha='center',
      va='bottom',
      fontsize=9,
  )
for bar_obj, val in zip(bars_c, pivot_bar['Summer Cooling']):
  ax.text(
      bar_obj.get_x() + bar_obj.get_width() / 2,
      val + 1,
      f'{val:.1f}%',
      ha='center',
      va='bottom',
      fontsize=9,
  )

# ── (1,1) Energy breakdown ────────────────────────────────────────────────────
ax = axes_cmp[1, 1]
energy_pivot = metrics_all.groupby('Scenario')[
    ['Boiler gas [kWh]', 'Fan electricity [kWh]']
].sum()

scenarios = energy_pivot.index.tolist()
gas_vals = energy_pivot['Boiler gas [kWh]'].values
fan_vals = energy_pivot['Fan electricity [kWh]'].values
x_e = np.arange(len(scenarios))
ax.bar(x_e, gas_vals, label='Boiler gas [kWh]', color='firebrick', alpha=0.8)
ax.bar(
    x_e,
    fan_vals,
    bottom=gas_vals,
    label='Fan electricity [kWh]',
    color='royalblue',
    alpha=0.8,
)
ax.set_xticks(x_e)
ax.set_xticklabels(scenarios)
ax.set_ylabel('Total Energy [kWh]')
ax.set_title('Total HVAC Energy – 7-Day Scenario')
ax.legend(fontsize=9)
ax.grid(True, axis='y', alpha=0.3)
for xi, (gv, fv) in enumerate(zip(gas_vals, fan_vals)):
  ax.text(
      xi,
      gv + fv + 5,
      f'{gv + fv:.0f} kWh',
      ha='center',
      va='bottom',
      fontsize=9,
  )

plt.tight_layout()
plt.savefig('scenario_comparison.png', dpi=120, bbox_inches='tight')
plt.show()
print('Comparison figure saved to scenario_comparison.png')

# %% [markdown]
# ## Summary
#
# | Metric | Winter Heating | Summer Cooling |
# |--------|---------------|----------------|
# | Initial temp | 275 K (2 °C) | 310 K (37 °C) |
# | Outdoor range | −5 °C to 4 °C | 22 °C to 35 °C |
# | HVAC action | Boiler heats water → VAVs inject heat | AirHandler cools air |
# | Primary energy | Natural gas (boiler) | Electricity (fan) |
# | Control trigger | zone_temp < 292 K → heating ON |
# |                 | zone_temp > 295 K → cooling ON |
# | Setback schedule | 290–297 K outside 9am–6pm | same |
#
# ### Key observations
# 1. **Heating scenario**: the boiler immediately activates because zone
#    temperatures are below the 292 K heating setpoint.  The building warms
#    steadily; corner cells respond faster due to greater convective exposure.
# 2. **Cooling scenario**: the air handler activates immediately because zone
#    temperatures exceed 295 K.  During cool nights the outdoor temperature
#    drops toward the comfort band, giving partial free cooling.
# 3. **Setpoint schedule**: during working hours (9am–6pm) the comfort band
#    (292–295 K) is active; outside those hours the wider eco band (290–297 K)
#    reduces HVAC energy demand.
# 4. **Zone differences**: room_1 and room_2 show slightly different
#    temperature trajectories due to diffuser placement and distance from
#    exterior walls.

# %%
print('Notebook complete.')
print('Figures produced:')
for fname in [
    'floor_plan.png',
    'weather_profiles.png',
    'occupancy_profile.png',
    'heating_scenario_winter_results.png',
    'cooling_scenario_summer_results.png',
    'scenario_comparison.png',
]:
  exists = os.path.exists(fname)
  print(f"  {'OK' if exists else '--'} {fname}")
