"""Loading the Oberdorfer N999R pump test campaigns.

Two campaigns are provided, both logged by the new PLC-based testbed every
200 ms while the two main-line valves (DPV3, DPV4) close together and load the
pump:

* initial (files dated 2021-07-28): 400-1140 rpm, valves closed to ~10 % open
* final   (files dated 2023-02-02): 700-950 rpm, valves closed to 25-35 % open

Column notes from the Pump Datasets Specification:
* the time column is all zeros, so time is rebuilt from the 200 ms period;
* the flow column is labelled ``F1 (bar)`` but is in L/min (logger bug);
* P2 is pump suction, P3 is pump discharge (absolute, bar).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DPV_CMD_CLOSED, DPV_CMD_OPEN, PUMP_DIR, PUMP_SAMPLE_PERIOD_S
from .lvm import read_lvm

CAMPAIGN_DIRS = {
    "initial": "Task 4 - Initial Pump Test Campaign",
    "final": "Task 4 - Final Pump Test Campaign",
}
RENAME = {
    "P1 (bar)": "P1", "P2 (bar)": "P2", "P3 (bar)": "P3", "P4 (bar)": "P4",
    "P5 (bar)": "P5", "F1 (bar)": "Q", "F2 (bar)": "Q2", "RPM": "rpm",
    "DP_Pump (bar)": "dP_logged", "Tmep (C)": "temp_C", "visc (Pa)": "visc_Pa_s",
    "density kg/m3": "rho", "Patm (bar)": "patm", "level (mm)": "level_mm",
    "F1_model": "Q_rig_model",
}


def cmd_to_opening(cmd: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """PLC valve command (4000 closed .. 20000 open) to percent open."""
    return 100.0 * (cmd - DPV_CMD_CLOSED) / (DPV_CMD_OPEN - DPV_CMD_CLOSED)


def _nominal_rpm(name: str) -> int:
    m = re.search(r"(\d+)rpm", name)
    if not m:
        raise ValueError(f"Cannot read pump speed from {name}")
    return int(m.group(1))


def load_test(path: str | Path, campaign: str) -> pd.DataFrame:
    path = Path(path)
    df = read_lvm(path).data.rename(columns=RENAME)
    df = df.drop(columns=["X_Value"])
    df.insert(0, "t", np.arange(len(df)) * PUMP_SAMPLE_PERIOD_S)
    df.insert(0, "sample", np.arange(len(df)))
    df.insert(0, "nominal_rpm", _nominal_rpm(path.name))
    df.insert(0, "campaign", campaign)
    df.insert(0, "test", f"{campaign}-{_nominal_rpm(path.name)}")
    df["dP_pump"] = df["P3"] - df["P2"]
    df["dpv3_open"] = cmd_to_opening(df["DPV3_CMD"])
    df["dpv4_open"] = cmd_to_opening(df["DPV4_CMD"])
    df["fpr"] = 1000.0 * df["Q"] / df["rpm"]
    return df


def load_campaign(campaign: str, pump_dir: Path = PUMP_DIR) -> pd.DataFrame:
    files = sorted((pump_dir / CAMPAIGN_DIRS[campaign]).glob("*.lvm"))
    if not files:
        raise FileNotFoundError(f"No pump files for campaign {campaign!r}")
    df = pd.concat([load_test(f, campaign) for f in files], ignore_index=True)
    return df.sort_values(["nominal_rpm", "sample"]).reset_index(drop=True)


def load_all() -> pd.DataFrame:
    return pd.concat([load_campaign(c) for c in CAMPAIGN_DIRS], ignore_index=True)


def test_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-test facts: speed, duration, valve end point, pressure and flow range."""
    g = df.groupby("test", sort=False)
    out = g.agg(
        campaign=("campaign", "first"),
        nominal_rpm=("nominal_rpm", "first"),
        rpm_measured=("rpm", "median"),
        n=("sample", "size"),
        duration_s=("t", "max"),
        valve_start_pct=("dpv3_open", "max"),
        valve_end_pct=("dpv3_open", "min"),
        valves_match=("DPV3_CMD", lambda s: bool((s == df.loc[s.index, "DPV4_CMD"]).all())),
        dP_start=("dP_pump", lambda s: s.iloc[:25].mean()),
        dP_max=("dP_pump", "max"),
        Q_start=("Q", lambda s: s.iloc[:25].mean()),
        Q_end=("Q", lambda s: s.iloc[-25:].mean()),
    )
    return out.sort_values(["campaign", "nominal_rpm"]).reset_index()
