"""Loading the 2014 fuel-system rig recordings (Tasks 2 and 3).

Each file is 60 s at 1 kHz with one fixed fault setting. Column meaning comes
from ``File format.pdf``; sensor positions come from the Lab 1 schematic:

    main tank -> P1 -> filter (DPV1) -> P2 -> gear pump -> P3 -> shut-off valve
    + DPV3 -> P4 -> nozzle (DPV5) -> P5 -> flow meter Q -> sump tank

    DPV2 bleeds pump discharge back to the main tank (degraded pump: lost
    delivery). DPV4 opens a leak path between P4 and the nozzle.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FAULTS, HEALTHY_FOLDER, INTERIM_DIR, P_ATM_BAR, RIG_DIR
from .lvm import read_lvm

# Position-based mapping (File format.pdf). Column 6 is to be ignored.
RAW_COLUMNS = ["t", "P1", "P2", "P3g", "P4g", "ch6_ignored", "Q", "P5g", "rpm",
               "dpv1", "dpv2", "dpv3", "dpv4", "dpv5"]
FS_HZ = 1000.0
CACHE_FILE = INTERIM_DIR / "rig_1khz.parquet"


def _start_time(header: dict[str, str]) -> datetime:
    # The file-level Time is when the logging program started (shared by many
    # files); the per-channel block holds the start of this recording.
    date = header.get("channel_Date", header["Date"])
    clock = header.get("channel_Time", header["Time"]).split(".")[0]
    return datetime.strptime(f"{date} {clock}", "%Y/%m/%d %H:%M:%S")


def catalogue(rig_dir: Path = RIG_DIR) -> pd.DataFrame:
    """One row per recording: condition, valve setting, severity, start time."""
    folders = {"HC": HEALTHY_FOLDER, **{k: f.folder for k, f in FAULTS.items()}}
    rows = []
    for cond, folder in folders.items():
        files = sorted((rig_dir / folder).glob("*.lvm"))
        if not files:
            raise FileNotFoundError(f"No .lvm files in {rig_dir / folder}")
        for f in files:
            lvm = read_lvm(f)
            df = lvm.data
            df.columns = RAW_COLUMNS
            valves = df[["dpv1", "dpv2", "dpv3", "dpv4", "dpv5"]]
            if (valves.nunique() > 1).any():
                raise ValueError(f"Valve setting changes within {f.name}")
            if cond == "HC":
                severity = 0.0
            else:
                fm = FAULTS[cond]
                severity = abs(valves[fm.valve].iloc[0] - fm.healthy_opening)
            rows.append({
                "file": f.name, "path": str(f), "condition": cond,
                "severity": float(severity), "start": _start_time(lvm.header),
                "n_samples": len(df),
                **{f"{c}_open": float(valves[c].iloc[0]) for c in valves},
            })
    cat = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    cat["minutes_from_first"] = (cat["start"] - cat["start"].min()).dt.total_seconds() / 60
    cat["run_id"] = cat.apply(lambda r: f"{r.condition}-{int(round(r.severity)):03d}"
                              if r.condition != "HC" else None, axis=1)
    hc = cat["condition"] == "HC"
    cat.loc[hc, "run_id"] = [f"HC-{i + 1}" for i in range(hc.sum())]
    return cat


def to_physical(df: pd.DataFrame, p_atm: float = P_ATM_BAR) -> pd.DataFrame:
    """Add absolute P3-P5 and derived hydraulic quantities.

    Derived columns:
      dP_filter = P1 - P2   pressure drop across the filter path
      dP_pump   = P3 - P2   pressure rise across the pump
      dP_valve  = P3 - P4   drop across shut-off valve + DPV3
      dP_nozzle = P4 - P5   drop across the nozzle path (DPV5)
      fpr       = Q / rpm   delivered flow per revolution [mL/rev]
    """
    out = df.copy()
    for k in ("P3", "P4", "P5"):
        out[k] = out[f"{k}g"] + p_atm
    out["dP_filter"] = out["P1"] - out["P2"]
    out["dP_pump"] = out["P3"] - out["P2"]
    out["dP_valve"] = out["P3"] - out["P4"]
    out["dP_nozzle"] = out["P4"] - out["P5"]
    out["fpr"] = 1000.0 * out["Q"] / out["rpm"]
    return out


def load_file(path: str | Path) -> pd.DataFrame:
    df = read_lvm(path).data
    df.columns = RAW_COLUMNS
    return to_physical(df)


def build_cache(force: bool = False) -> Path:
    """Parse all 47 recordings once into a float32 parquet file (about 2.8 M rows)."""
    if CACHE_FILE.exists() and not force:
        return CACHE_FILE
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    cat = catalogue()
    parts = []
    for r in cat.itertuples():
        df = load_file(r.path)
        df = df.astype("float32")
        df.insert(0, "run_id", r.run_id)
        df.insert(1, "condition", r.condition)
        df.insert(2, "severity", np.float32(r.severity))
        parts.append(df)
    allruns = pd.concat(parts, ignore_index=True)
    for c in ("run_id", "condition"):
        allruns[c] = allruns[c].astype("category")
    allruns.to_parquet(CACHE_FILE, index=False)
    cat.to_csv(INTERIM_DIR / "rig_catalogue.csv", index=False)
    return CACHE_FILE


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (1 kHz samples for every run, catalogue). Builds the cache if needed."""
    build_cache()
    data = pd.read_parquet(CACHE_FILE)
    cat = pd.read_csv(INTERIM_DIR / "rig_catalogue.csv", parse_dates=["start"])
    return data, cat
