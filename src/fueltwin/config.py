"""Paths, physical constants and rig metadata used across the project.

The raw data is not stored in this repository. By default it is expected in a
sibling folder called ``Digital twin`` (the coursework resource pack); set the
``FUELTWIN_DATA`` environment variable to point somewhere else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("FUELTWIN_DATA", REPO_ROOT.parent / "Digital twin"))
INTERIM_DIR = REPO_ROOT / "data" / "interim"
FIG_DIR = REPO_ROOT / "results" / "figures"
TABLE_DIR = REPO_ROOT / "results" / "tables"

RIG_DIR = DATA_ROOT / "Task 2 & 3"
PUMP_DIR = DATA_ROOT / "Task 4"

# --- Physical constants and unit conversions -------------------------------
# The 2014 rig logs P3-P5 as gauge pressure and P1-P2 as absolute pressure.
# Atmospheric pressure was not recorded, so the standard atmosphere is assumed.
# A +/-20 mbar weather swing is the main systematic uncertainty this adds.
P_ATM_BAR = 1.01325
GPM_TO_LPM = 3.785411784
PSI_TO_BAR = 0.0689475729

# Burkert-style direct proportional valve command range in the PLC logs
# (Task 4 files): 20000 = fully open, 4000 = fully closed.
DPV_CMD_OPEN = 20000.0
DPV_CMD_CLOSED = 4000.0

# Pump datasets are logged by the PLC every 200 ms (Pump Datasets Specification).
PUMP_SAMPLE_PERIOD_S = 0.2

# Oberdorfer N999R built-in relief valve: factory setting 50 psig.
RELIEF_VALVE_SETTING_BAR = 50 * PSI_TO_BAR  # 3.447 bar


@dataclass(frozen=True)
class FaultMode:
    code: str
    name: str
    component: str
    valve: str  # column holding the emulating valve opening (%)
    healthy_opening: float  # valve opening (%) in the healthy configuration
    folder: str  # raw-data folder name


# Fault emulation on the 2014 UAV fuel-system demonstrator (File format.pdf and
# Lab 1 schematic). Severity is reported as valve travel away from the healthy
# position, in percent. It is a control setting, not a physical severity: the
# hydraulic effect of a proportional valve is strongly non-linear in its travel.
FAULTS: dict[str, FaultMode] = {
    "F1": FaultMode("F1", "Clogged filter", "Suction filter", "dpv1", 100.0,
                    "Task 3 - Fuel System Datasets - F1 - Clogged FilterTask 3 - Fuel System Datasets - F1 - Clogged Filter"),
    "F2": FaultMode("F2", "Degraded pump", "External gear pump", "dpv2", 0.0,
                    "Task3 -Fuel System Datasets - F2 - Degraded Pump"),
    "F3": FaultMode("F3", "Stuck valve", "Shut-off valve (mid-range)", "dpv3", 100.0,
                    "Task 3 - Fuel System Datasets - F3 - Stuck Valve"),
    "F4": FaultMode("F4", "Leaking pipe", "Pipe downstream of the shut-off valve", "dpv4", 0.0,
                    "Task 3 - Fuel System Datasets - F4 - Leaking Pipe"),
    "F5": FaultMode("F5", "Clogged injector", "Injector / nozzle", "dpv5", 100.0,
                    "Task 3 -Fuel System Datasets - F5 - Clogged InjectorTask 3 -Fuel System Datasets - F5 - Clogged Injector"),
}
HEALTHY_FOLDER = "Task 2 - Fuel System Datasets - Healthy Conditions"
CONDITIONS = ["HC", *FAULTS]  # display order
CONDITION_NAMES = {"HC": "Healthy", **{k: v.name for k, v in FAULTS.items()}}

# Canonical sensor names on the 2014 rig (Lab 1 schematic, File format.pdf).
SENSORS = {
    "P1": "Pressure before filter",
    "P2": "Pressure after filter (pump suction)",
    "P3": "Pressure after pump (before shut-off valve)",
    "P4": "Pressure after shut-off valve",
    "P5": "Pressure after nozzle",
    "Q": "Main-line flow at the end of the line",
}
PRESSURES = ["P1", "P2", "P3", "P4", "P5"]
MEASURED = [*PRESSURES, "Q"]
