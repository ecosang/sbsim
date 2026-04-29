"""Standalone building simulation with schedule-based HVAC control (no RL).

Simple 2-zone building with hard-coded floor plan. Demonstrates the same
simulation loop that the RL environment uses, but driven by a schedule
instead of an RL agent.

Run with:
    poetry run python smart_control/notebooks/standalone_sim_demo.py
"""

# %% [markdown]
# # Standalone Building Simulation – 2-Zone with Schedule-Based HVAC
#
# This notebook bypasses the RL `Environment` wrapper and drives the
# **SimulatorFlexibleGeometries** directly, using the same two-phase step
# that the RL environment uses internally:
#
# 1. `setup_step_sim()` – thermostat-driven VAV settings
# 2. `set_action()` – apply boiler & air handler setpoints (RL-equivalent)
# 3. `execute_step_sim()` – finite-difference thermal simulation

# %% [markdown]
# ## Section 0 – Imports

# %%
import copy
import os
import sys
import time as time_module

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make sure the repo root is on the path before importing local packages.
try:
  _nb_dir = os.path.dirname(os.path.abspath(__file__))
  REPO_ROOT = os.path.abspath(os.path.join(_nb_dir, "..", ".."))
except NameError:
  REPO_ROOT = os.path.abspath(os.getcwd())

if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

# pylint: disable=wrong-import-position
from smart_control.simulator import air_handler as air_handler_py
from smart_control.simulator import boiler as boiler_py
from smart_control.simulator import building as building_py
from smart_control.simulator import hvac_floorplan_based as floorplan_hvac_py
from smart_control.simulator import setpoint_schedule as setpoint_schedule_py
from smart_control.simulator import simulator_flexible_floor_plan as sim_py
from smart_control.simulator import step_function_occupancy as occupancy_py
from smart_control.simulator import weather_controller as weather_controller_py

# pylint: enable=wrong-import-position

KELVIN_TO_CELSIUS = 273.15

print("All imports OK.")
print(
    f"NumPy {np.__version__}, Pandas {pd.__version__}, Matplotlib"
    f" {matplotlib.__version__}"
)

# %% [markdown]
# ## Section 1 – Hard-coded 2-Zone Floor Plan
#
# A 23×12 control-volume grid taken from the test suite
# (`simulator_flexible_floor_plan_test.py`).
#
# | Code | Type |
# |------|------|
# | `0`  | Interior air (room space) |
# | `1`  | Interior wall / corridor |
# | `2`  | Exterior wall |

# %%
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

print(f"Floor plan shape: {FLOOR_PLAN.shape}  (rows × cols)")
print(f"Air cells  (code 0): {(FLOOR_PLAN == 0).sum()}")
print(f"Wall cells (code 1): {(FLOOR_PLAN == 1).sum()}")
print(f"Ext. cells (code 2): {(FLOOR_PLAN == 2).sum()}")

# %%
# ── Visualise the floor plan ──────────────────────────────────────────────────
fig_fp, ax_fp = plt.subplots(figsize=(7, 10))
cmap_fp = mcolors.ListedColormap(["white", "#aaaaaa", "#444444"])
bounds_fp = [-0.5, 0.5, 1.5, 2.5]
norm_fp = mcolors.BoundaryNorm(bounds_fp, cmap_fp.N)
ax_fp.imshow(
    FLOOR_PLAN, cmap=cmap_fp, norm=norm_fp, origin="upper", aspect="equal"
)
ax_fp.text(
    5.5,
    6.5,
    "room_1\n(zone 1)",
    ha="center",
    va="center",
    fontsize=12,
    color="steelblue",
    fontweight="bold",
)
ax_fp.text(
    5.5,
    15.5,
    "room_2\n(zone 2)",
    ha="center",
    va="center",
    fontsize=12,
    color="darkorange",
    fontweight="bold",
)
legend_patches = [
    mpatches.Patch(color="white", label="Air (code 0)"),
    mpatches.Patch(color="#aaaaaa", label="Wall (code 1)"),
    mpatches.Patch(color="#444444", label="Exterior (code 2)"),
]
ax_fp.legend(handles=legend_patches, loc="upper right", fontsize=9)
ax_fp.set_title(
    "Two-Zone Office Building – Floor Plan (23×12 CV grid)", fontsize=12
)
ax_fp.set_xlabel("Column index")
ax_fp.set_ylabel("Row index")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Section 2 – Configuration & Building Construction
#
# Parameters from `sim_config.gin` adapted for the small building.

# %%
# ── Building geometry ─────────────────────────────────────────────────────────
CV_SIZE_CM = 20.0
FLOOR_HEIGHT_CM = 300.0
INITIAL_TEMP_K = 294.0  # ~21°C

# ── Material properties (from sim_config.gin) ─────────────────────────────────
inside_air_props = building_py.MaterialProperties(
    conductivity=50.0, heat_capacity=700.0, density=1.0
)
inside_wall_props = building_py.MaterialProperties(
    conductivity=50.0, heat_capacity=1.0, density=700.0
)
exterior_wall_props = building_py.MaterialProperties(
    conductivity=0.05, heat_capacity=700.0, density=1.0
)

# ── Build the building ────────────────────────────────────────────────────────
zone_map = copy.deepcopy(FLOOR_PLAN)
bld = building_py.FloorPlanBasedBuilding(
    cv_size_cm=CV_SIZE_CM,
    floor_height_cm=FLOOR_HEIGHT_CM,
    initial_temp=INITIAL_TEMP_K,
    inside_air_properties=inside_air_props,
    inside_wall_properties=inside_wall_props,
    building_exterior_properties=exterior_wall_props,
    floor_plan=FLOOR_PLAN,
    zone_map=zone_map,
    buffer_from_walls=0,
)

ZONE_IDS = sorted(bld.get_zone_average_temps().keys())
print(f"Building created with {len(ZONE_IDS)} zones: {ZONE_IDS}")
print(f"Temperature grid shape: {bld.temp.shape}")
print(f"Initial mean temp: {bld.temp.mean() - KELVIN_TO_CELSIUS:.2f} °C")

# %% [markdown]
# ## Section 3 – HVAC Configuration
#
# ```
# FloorPlanBasedHvac
# ├── Boiler        – hot water loop for VAV reheat
# ├── AirHandler    – mixed / conditioned supply air
# └── VAV × 2 zones – variable-air-volume with thermostat
#       └── SetpointSchedule – comfort (6am–7pm) vs eco
# ```

# %%
# ── HVAC schedule ─────────────────────────────────────────────────────────────
MORNING_START_HOUR = 6
EVENING_START_HOUR = 19
HEATING_SETPOINT_DAY_K = 273.15 + 20.0  # 20°C comfort heating
COOLING_SETPOINT_DAY_K = 273.15 + 24.0  # 24°C comfort cooling
HEATING_SETPOINT_NIGHT_K = 273.15 + 15.0  # 15°C eco/setback
COOLING_SETPOINT_NIGHT_K = 273.15 + 28.0  # 28°C eco/setback
TIME_ZONE = "US/Pacific"

# ── HVAC device parameters (from sim_config.gin, sized for small building) ────
BOILER_REHEAT_WATER_SP_K = 350.0
BOILER_PUMP_HEAD = 3.0
BOILER_PUMP_EFFICIENCY = 0.6
AHU_RECIRCULATION = 0.3
AHU_HEATING_SP_K = 285.0
AHU_COOLING_SP_K = 298.0
AHU_FAN_PRESSURE = 500.0
AHU_FAN_EFFICIENCY = 0.8
AHU_MAX_AIR_FLOW = 2.0
VAV_MAX_AIR_FLOW = 0.50
VAV_MAX_WATER_FLOW = 0.10

# ── Build HVAC ────────────────────────────────────────────────────────────────
boiler = boiler_py.Boiler(
    reheat_water_setpoint=BOILER_REHEAT_WATER_SP_K,
    water_pump_differential_head=BOILER_PUMP_HEAD,
    water_pump_efficiency=BOILER_PUMP_EFFICIENCY,
    device_id="boiler_id",
)
# Initialize return water sensor to avoid NoneType error at step 0
boiler.return_water_temperature_sensor = BOILER_REHEAT_WATER_SP_K

air_handler = air_handler_py.AirHandler(
    recirculation=AHU_RECIRCULATION,
    heating_air_temp_setpoint=AHU_HEATING_SP_K,
    cooling_air_temp_setpoint=AHU_COOLING_SP_K,
    fan_differential_pressure=AHU_FAN_PRESSURE,
    fan_efficiency=AHU_FAN_EFFICIENCY,
    max_air_flow_rate=AHU_MAX_AIR_FLOW,
)

schedule = setpoint_schedule_py.SetpointSchedule(
    morning_start_hour=MORNING_START_HOUR,
    evening_start_hour=EVENING_START_HOUR,
    comfort_temp_window=(HEATING_SETPOINT_DAY_K, COOLING_SETPOINT_DAY_K),
    eco_temp_window=(HEATING_SETPOINT_NIGHT_K, COOLING_SETPOINT_NIGHT_K),
    time_zone=TIME_ZONE,
)

hvac = floorplan_hvac_py.FloorPlanBasedHvac(
    zone_identifier=ZONE_IDS,
    air_handler=air_handler,
    boiler=boiler,
    schedule=schedule,
    vav_max_air_flow_rate=VAV_MAX_AIR_FLOW,
    vav_reheat_max_water_flow_rate=VAV_MAX_WATER_FLOW,
)

print(f"HVAC created. {len(hvac.vavs)} VAVs")
print(f"  Boiler SP: {boiler.reheat_water_setpoint - KELVIN_TO_CELSIUS:.1f} °C")
print(
    "  AHU heating SP:"
    f" {air_handler.heating_air_temp_setpoint - KELVIN_TO_CELSIUS:.1f} °C"
)
print(
    "  AHU cooling SP:"
    f" {air_handler.cooling_air_temp_setpoint - KELVIN_TO_CELSIUS:.1f} °C"
)

# %% [markdown]
# ## Section 4 – Weather & Occupancy

# %%
# ── Weather (sinusoidal, summer-like) ─────────────────────────────────────────
weather = weather_controller_py.WeatherController(
    default_low_temp=288.0,  # 15°C at night
    default_high_temp=303.0,  # 30°C at noon
    convection_coefficient=12.0,
)

# ── Occupancy (step function: 9am-5pm = 10 people, else 0.1) ────────────────
occupancy = occupancy_py.StepFunctionOccupancy(
    work_start_time=pd.Timedelta(9, unit="h"),
    work_end_time=pd.Timedelta(17, unit="h"),
    work_occupancy=10.0,
    nonwork_occupancy=0.1,
)

print("Weather: sinusoidal 15°C–30°C")
print("Occupancy: 10 people 9am-5pm, 0.1 otherwise")

# %% [markdown]
# ## Section 5 – Build Simulator

# %%
TIME_STEP_SEC = 300.0
START_TIMESTAMP = pd.Timestamp("2024-01-15 00:00:00")  # winter Monday
NUM_DAYS = 7
N_STEPS = int(NUM_DAYS * 24 * 3600 / TIME_STEP_SEC)

sim = sim_py.SimulatorFlexibleGeometries(
    building=bld,
    hvac=hvac,
    weather_controller=weather,
    time_step_sec=TIME_STEP_SEC,
    convergence_threshold=0.05,
    iteration_limit=100,
    iteration_warning=30,
    start_timestamp=START_TIMESTAMP,
)

print("Simulator created.")
print(f"  Start: {sim.current_timestamp}")
print(f"  Steps: {N_STEPS} ({NUM_DAYS} days)")
print(f"  Time step: {TIME_STEP_SEC}s")

# %% [markdown]
# ## Section 6 – Hard-coded Action Schedule
#
# The RL agent controls two setpoints. We use weekday/weekend schedule:
#
# | Time Period | Boiler Supply Water SP | AHU Heating Temp SP |
# |-------------|----------------------|---------------------|
# | Weekday 6am–7pm | 350 K (77°C) | 292 K (19°C) |
# | Night / Weekend  | 315 K (42°C) | 285 K (12°C) |

# %%
OCCUPIED_SUPPLY_WATER_SP_K = 350.0
OCCUPIED_AHU_HEATING_SP_K = 292.0
SETBACK_SUPPLY_WATER_SP_K = 315.0
SETBACK_AHU_HEATING_SP_K = 285.0

print("Action schedule configured.")
print(
    f"  Occupied: boiler={OCCUPIED_SUPPLY_WATER_SP_K-KELVIN_TO_CELSIUS:.0f}°C, "
    f"AHU={OCCUPIED_AHU_HEATING_SP_K-KELVIN_TO_CELSIUS:.0f}°C"
)
print(
    f"  Setback:  boiler={SETBACK_SUPPLY_WATER_SP_K-KELVIN_TO_CELSIUS:.0f}°C, "
    f"AHU={SETBACK_AHU_HEATING_SP_K-KELVIN_TO_CELSIUS:.0f}°C"
)

# %% [markdown]
# ## Section 7 – Simulation Loop
#
# Core loop replicating `Environment._step()`:
# 1. `setup_step_sim()` – thermostat reads zone temps → VAV damper/valve
# 2. `set_action()` on boiler & air handler
# 3. `execute_step_sim()` – FDM thermal sim + VAV heat delivery

# %%
# ── Data storage ──────────────────────────────────────────────────────────────
timestamps = []
zone_temps = {z: [] for z in ZONE_IDS}
heating_sps = []
cooling_sps = []
outdoor_temps = []
thermostat_modes = {z: [] for z in ZONE_IDS}

# HVAC data
ahu_supply_temps = []
boiler_sp_applied = []
ahu_heating_sp_applied = []

# Per-zone VAV data
damper_settings = {z: [] for z in ZONE_IDS}
reheat_valve_settings = {z: [] for z in ZONE_IDS}
q_zone_vals = {z: [] for z in ZONE_IDS}
zone_supply_temps = {z: [] for z in ZONE_IDS}

# Energy
boiler_gas_rates = []
fan_elec_rates = []
ac_elec_rates = []

# Occupancy
occupancy_vals = []

ZONE_COLORS = {"room_1": "steelblue", "room_2": "darkorange"}

print(f"Starting simulation: {N_STEPS} steps ({NUM_DAYS} days)")
print(f"  Start: {START_TIMESTAMP}")
print()

t_start = time_module.time()

for step in range(N_STEPS):
  ts = sim.current_timestamp

  # ── Phase 1: Setup (thermostat-driven VAV settings) ──────────────────
  sim.setup_step_sim()

  # ── Phase 2: Determine schedule action ───────────────────────────────
  # (Check local time for weekday/occupied logic)
  # Note: START_TIMESTAMP is timezone-naive, so use UTC offset manually
  # For simplicity, treat the timestamp as local-ish (winter Monday start)
  hour = ts.hour
  dow = ts.dayofweek  # 0=Mon
  is_weekday = dow < 5
  is_occupied = MORNING_START_HOUR <= hour < EVENING_START_HOUR

  if is_weekday and is_occupied:
    sw_sp = OCCUPIED_SUPPLY_WATER_SP_K
    ah_sp = OCCUPIED_AHU_HEATING_SP_K
  else:
    sw_sp = SETBACK_SUPPLY_WATER_SP_K
    ah_sp = SETBACK_AHU_HEATING_SP_K

  # Apply actions to devices (same as SimulatorBuilding.request_action)
  hvac.boiler.set_action("supply_water_setpoint", sw_sp, ts)
  hvac.air_handler.set_action(
      "supply_air_heating_temperature_setpoint", ah_sp, ts
  )

  # ── Phase 3: Execute thermal simulation ──────────────────────────────
  sim.execute_step_sim()

  # ── Phase 4: Collect data ────────────────────────────────────────────
  timestamps.append(ts)

  # Zone temperatures
  zone_avgs = bld.get_zone_average_temps()
  for zid in ZONE_IDS:
    zone_temps[zid].append(zone_avgs[zid])

  # Thermostat setpoints and mode
  ref_zone = ZONE_IDS[0]
  sched = hvac.vavs[ref_zone].thermostat.get_setpoint_schedule()
  h_sp, c_sp = sched.get_temperature_window(ts)
  heating_sps.append(h_sp)
  cooling_sps.append(c_sp)

  for zid in ZONE_IDS:
    thermostat_modes[zid].append(
        hvac.vavs[zid].thermostat._current_mode.value  # pylint: disable=protected-access
    )
    damper_settings[zid].append(hvac.vavs[zid].damper_setting)
    reheat_valve_settings[zid].append(hvac.vavs[zid].reheat_valve_setting)

  # Outdoor temperature
  outdoor_k = weather.get_current_temp(ts)
  outdoor_temps.append(outdoor_k)

  # AHU supply temperature
  rec_temp = bld.temp.mean()
  ahu_supply_k = hvac.air_handler.get_supply_air_temp(rec_temp, outdoor_k)
  ahu_supply_temps.append(ahu_supply_k)

  # Actions applied
  boiler_sp_applied.append(sw_sp)
  ahu_heating_sp_applied.append(ah_sp)

  # Per-zone VAV output
  for zid in ZONE_IDS:
    vav = hvac.vavs[zid]
    zst_k = vav.compute_zone_supply_temp(
        ahu_supply_k, hvac.boiler.reheat_water_setpoint
    )
    zone_supply_temps[zid].append(zst_k)
    q_w = vav.compute_energy_applied_to_zone(
        zone_avgs[zid], ahu_supply_k, hvac.boiler.reheat_water_setpoint
    )
    q_zone_vals[zid].append(q_w)

  # Energy rates
  fan_rate = (
      hvac.air_handler.compute_intake_fan_energy_rate()
      + hvac.air_handler.compute_exhaust_fan_energy_rate()
  )
  fan_elec_rates.append(fan_rate)
  # Note: compute_thermal_energy_rate is negative when AHU is cooling
  # (supply_air_temp < mixed_air_temp). We keep the raw signed value so
  # the plot can show heating (>0) and cooling (<0) separately.
  ac_rate = hvac.air_handler.compute_thermal_energy_rate(rec_temp, outdoor_k)
  ac_elec_rates.append(ac_rate)
  rw_k = hvac.boiler.return_water_temperature_sensor
  boiler_gas_rates.append(
      hvac.boiler.compute_thermal_energy_rate(rw_k, outdoor_k)
  )

  # Occupancy
  ts_end = ts + pd.Timedelta(TIME_STEP_SEC, unit="s")
  occ_val = occupancy.average_zone_occupancy("room_1", ts, ts_end)
  occupancy_vals.append(occ_val)

  # Progress
  if (step + 1) % 288 == 0:
    day_num = (step + 1) // 288
    elapsed = time_module.time() - t_start
    mean_t = np.mean([zone_avgs[z] for z in ZONE_IDS])
    print(
        f"  Day {day_num:2d}/{NUM_DAYS} | "
        f"{ts.strftime('%a %Y-%m-%d %H:%M')} | "
        f"mean T={mean_t - KELVIN_TO_CELSIUS:.2f}°C | "
        f"outdoor={outdoor_k - KELVIN_TO_CELSIUS:.1f}°C | "
        f"elapsed={elapsed:.1f}s"
    )

elapsed_total = time_module.time() - t_start
print(
    f"\nSimulation complete in {elapsed_total:.1f}s"
    f" ({elapsed_total/N_STEPS*1000:.1f} ms/step)"
)

# %% [markdown]
# ## Section 8 – Combined Visualization (all signals in one figure)
#
# 10-row subplot: zone temps, thermostat mode, damper, reheat valve,
# zone supply temp, q_zone, actions, energy, cumulative energy, occupancy.

# %%
TS = timestamps

# Pre-compute cumulative energy for the energy subplot
dt_hours = TIME_STEP_SEC / 3600
cum_gas = np.cumsum([r / 1000 * dt_hours for r in boiler_gas_rates])
cum_fan = np.cumsum([r / 1000 * dt_hours for r in fan_elec_rates])
# AHU thermal is signed: positive=heating, negative=cooling
cum_ac_signed = np.cumsum([r / 1000 * dt_hours for r in ac_elec_rates])

N_ROWS = 10
fig, axes = plt.subplots(N_ROWS, 1, figsize=(20, N_ROWS * 3.2), sharex=True)
fig.suptitle(
    "SB1-style 2-Zone Simulation – Schedule-Based HVAC Control"
    f" ({NUM_DAYS} Days)\nStart: {START_TIMESTAMP.strftime('%Y-%m-%d %H:%M')}",
    fontsize=14,
    fontweight="bold",
    y=0.995,
)

# ── Row 0: Zone temperature vs. setpoints ────────────────────────────────────
ax = axes[0]
for zid in ZONE_IDS:
  ax.plot(
      TS,
      [t - KELVIN_TO_CELSIUS for t in zone_temps[zid]],
      color=ZONE_COLORS[zid],
      lw=1.5,
      label=zid,
  )
ax.plot(
    TS,
    [t - KELVIN_TO_CELSIUS for t in heating_sps],
    color="red",
    lw=1.2,
    ls="--",
    drawstyle="steps-post",
    label="Heating SP",
)
ax.plot(
    TS,
    [t - KELVIN_TO_CELSIUS for t in cooling_sps],
    color="orange",
    lw=1.2,
    ls="--",
    drawstyle="steps-post",
    label="Cooling SP",
)
ax.plot(
    TS,
    [t - KELVIN_TO_CELSIUS for t in outdoor_temps],
    color="cyan",
    lw=1,
    alpha=0.6,
    label="Outdoor T",
)
ax.set_ylabel("Temp [°C]")
ax.set_title("Zone Air Temperature vs. Comfort Setpoints")
ax.legend(fontsize=8, ncol=5, loc="upper right")
ax.grid(True, alpha=0.3)

# ── Row 1: Thermostat mode ───────────────────────────────────────────────────
ax = axes[1]
for zid in ZONE_IDS:
  ax.plot(
      TS,
      thermostat_modes[zid],
      color=ZONE_COLORS[zid],
      lw=1.2,
      drawstyle="steps-post",
      label=f"{zid}",
      alpha=0.8,
  )
ax.set_ylabel("Mode")
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["OFF", "HEAT", "COOL", "P.COOL"])
ax.set_title("Thermostat Mode")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Row 2: VAV Damper Position ────────────────────────────────────────────────
ax = axes[2]
for zid in ZONE_IDS:
  ax.plot(
      TS,
      damper_settings[zid],
      color=ZONE_COLORS[zid],
      lw=1.2,
      drawstyle="steps-post",
      label=f"{zid}",
  )
ax.set_ylabel("Damper [0–1]")
ax.set_ylim(-0.05, 1.05)
ax.set_title("VAV Damper Position")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Row 3: VAV Reheat Valve Position ─────────────────────────────────────────
ax = axes[3]
for zid in ZONE_IDS:
  ax.plot(
      TS,
      reheat_valve_settings[zid],
      color=ZONE_COLORS[zid],
      lw=1.2,
      drawstyle="steps-post",
      label=f"{zid}",
  )
ax.set_ylabel("Reheat [0–1]")
ax.set_ylim(-0.05, 1.05)
ax.set_title("VAV Reheat Valve Position")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Row 4: VAV Zone Supply Temperature ───────────────────────────────────────
ax = axes[4]
for zid in ZONE_IDS:
  ax.plot(
      TS,
      [t - KELVIN_TO_CELSIUS for t in zone_supply_temps[zid]],
      color=ZONE_COLORS[zid],
      lw=1.5,
      label=f"{zid} supply T",
  )
ax.plot(
    TS,
    [t - KELVIN_TO_CELSIUS for t in ahu_supply_temps],
    color="purple",
    lw=1,
    ls="--",
    label="AHU supply air T",
)
ax.set_ylabel("Temp [°C]")
ax.set_title("VAV Zone Supply Temperature (after reheat coil)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Row 5: VAV Thermal Power to Zone ─────────────────────────────────────────
ax = axes[5]
for zid in ZONE_IDS:
  ax.plot(
      TS,
      [q / 1000 for q in q_zone_vals[zid]],
      color=ZONE_COLORS[zid],
      lw=1.5,
      label=f"{zid}",
  )
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_ylabel("Power [kW]")
ax.set_title("VAV Thermal Power to Zone (+heat / −cool)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Row 6: RL-equivalent Actions ─────────────────────────────────────────────
ax = axes[6]
ax.plot(
    TS,
    [t - KELVIN_TO_CELSIUS for t in boiler_sp_applied],
    color="red",
    lw=2,
    drawstyle="steps-post",
    label="Boiler Supply Water SP",
)
ax.plot(
    TS,
    [t - KELVIN_TO_CELSIUS for t in ahu_heating_sp_applied],
    color="blue",
    lw=2,
    drawstyle="steps-post",
    label="AHU Heating Temp SP",
)
ax.set_ylabel("Setpoint [°C]")
ax.set_title("RL-Equivalent Actions (Schedule-Based)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Row 7: HVAC Power Consumption ────────────────────────────────────────────
# Note: ac_elec_rates is signed:
#   positive → AHU heating (mixed air < heating_sp, AHU adds heat)
#   negative → AHU cooling (mixed air > cooling_sp, AHU removes heat)
ax = axes[7]
ax.plot(
    TS,
    [r / 1000 for r in boiler_gas_rates],
    color="orange",
    lw=1.5,
    label="Boiler gas [kW]",
)
ax.plot(
    TS,
    [r / 1000 for r in fan_elec_rates],
    color="green",
    lw=1.5,
    label="Fan elec. [kW]",
)
ax.plot(
    TS,
    [r / 1000 for r in ac_elec_rates],
    color="magenta",
    lw=1.5,
    label="AHU thermal [kW]\n(+heat / −cool)",
)
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_ylabel("Power [kW]")
ax.set_title("HVAC Power (AHU thermal: positive=heating, negative=cooling)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Row 8: Cumulative HVAC Energy ────────────────────────────────────────────
# For cumulative, separate heating and cooling energy for AHU
ac_heat_kwh = np.cumsum([max(r, 0) / 1000 * dt_hours for r in ac_elec_rates])
ac_cool_kwh = np.cumsum(
    [abs(min(r, 0)) / 1000 * dt_hours for r in ac_elec_rates]
)

ax = axes[8]
ax.plot(
    TS,
    cum_gas,
    color="orange",
    lw=2,
    label=f"Boiler gas ({cum_gas[-1]:.0f} kWh)",
)
ax.plot(
    TS, cum_fan, color="green", lw=2, label=f"Fan elec. ({cum_fan[-1]:.0f} kWh)"
)
ax.plot(
    TS,
    ac_heat_kwh,
    color="magenta",
    lw=2,
    ls="-",
    label=f"AHU heating ({ac_heat_kwh[-1]:.0f} kWh)",
)
ax.plot(
    TS,
    ac_cool_kwh,
    color="cyan",
    lw=2,
    ls="--",
    label=f"AHU cooling ({ac_cool_kwh[-1]:.0f} kWh)",
)
ax.set_ylabel("Energy [kWh]")
ax.set_title("Cumulative HVAC Energy")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Row 9: Occupancy ─────────────────────────────────────────────────────────
ax = axes[9]
ax.plot(
    TS,
    occupancy_vals,
    color="teal",
    lw=1.5,
    drawstyle="steps-post",
    label="Occupancy (per zone)",
)
ax.set_ylabel("Occupants")
ax.set_ylim(-0.5, 12)
ax.set_title("Occupancy Schedule")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlabel("Time")

plt.tight_layout()
plt.savefig(
    os.path.join(
        REPO_ROOT, "smart_control", "notebooks", "standalone_sim_results.png"
    ),
    dpi=150,
    bbox_inches="tight",
)
plt.show()
print(
    "Combined plot saved to smart_control/notebooks/standalone_sim_results.png"
)

# %% [markdown]
# ## Section 9 – Spatial Floor-Plan Heatmap (final state)

# %%
fig_hm, ax_hm = plt.subplots(figsize=(7, 10))
temps_display = bld.temp.copy()
temps_c = temps_display - KELVIN_TO_CELSIUS
temps_c[FLOOR_PLAN == 2] = np.nan

im = ax_hm.imshow(
    temps_c, cmap="bwr", vmin=15, vmax=30, origin="upper", aspect="equal"
)
plt.colorbar(im, ax=ax_hm, label="Temperature [°C]", shrink=0.6)
ax_hm.set_title(
    "Floor-Plan Temperature at End of Simulation\n"
    f"({timestamps[-1].strftime('%Y-%m-%d %H:%M')})",
    fontsize=12,
)
ax_hm.set_xlabel("Column index")
ax_hm.set_ylabel("Row index")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Section 10 – Summary Statistics

# %%
print("=" * 60)
print("SIMULATION SUMMARY")
print("=" * 60)

all_temps_c = np.array([
    [zone_temps[z][i] - KELVIN_TO_CELSIUS for z in ZONE_IDS]
    for i in range(N_STEPS)
])
print(f"\nZone temperatures ({len(ZONE_IDS)} zones, {N_STEPS} steps):")
print(f"  Min:  {all_temps_c.min():.2f} °C")
print(f"  Max:  {all_temps_c.max():.2f} °C")
print(f"  Mean: {all_temps_c.mean():.2f} °C")

# Comfort violations
h_arr = np.array(heating_sps)
c_arr = np.array(cooling_sps)
mean_t_k = all_temps_c.mean(axis=1) + KELVIN_TO_CELSIUS
too_cold = mean_t_k < h_arr
too_hot = mean_t_k > c_arr
n_cold = too_cold.sum()
n_hot = too_hot.sum()
print("\nComfort violations:")
print(f"  Steps too cold: {n_cold} ({100*n_cold/N_STEPS:.1f}%)")
print(f"  Steps too hot:  {n_hot} ({100*n_hot/N_STEPS:.1f}%)")
print(
    "  Comfortable:   "
    f" {N_STEPS - n_cold - n_hot} ({100*(N_STEPS-n_cold-n_hot)/N_STEPS:.1f}%)"
)

total_gas = cum_gas[-1]
total_fan = cum_fan[-1]
total_ac_heat = ac_heat_kwh[-1]
total_ac_cool = ac_cool_kwh[-1]
total_energy = total_gas + total_fan + total_ac_heat + total_ac_cool
print(f"\nEnergy consumption ({NUM_DAYS} days):")
print(f"  Boiler gas:       {total_gas:.1f} kWh")
print(f"  Fan electricity:   {total_fan:.1f} kWh")
print(f"  AHU heating:      {total_ac_heat:.1f} kWh")
print(f"  AHU cooling:      {total_ac_cool:.1f} kWh  (removed from zone)")
print(f"  Total HVAC:       {total_energy:.1f} kWh")

print(
    f"\nExecution: {elapsed_total:.1f}s total,"
    f" {elapsed_total/N_STEPS*1000:.1f} ms/step"
)
print("=" * 60)
