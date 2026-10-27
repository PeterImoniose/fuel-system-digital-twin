"""Turn 1 kHz recordings into fixed-length windows of features.

A window is the unit the digital shadow reasons about (default 1 s = 1000
samples, so each 60 s recording gives 60 windows). Means carry the operating
point; standard deviations carry the fast dynamics (gear-pump flow ripple,
turbulence, cavitation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rig import FS_HZ

MEAN_COLS = ["P1", "P2", "P3", "P4", "P5", "P3g", "P4g", "P5g", "Q", "rpm",
             "dP_filter", "dP_pump", "dP_valve", "dP_nozzle", "fpr"]
STD_COLS = ["P1", "P2", "P3", "P4", "P5", "Q"]
META_COLS = ["run_id", "condition", "severity"]


def windows(data: pd.DataFrame, cat: pd.DataFrame, seconds: float = 1.0) -> pd.DataFrame:
    """Non-overlapping windows per run with mean and std features.

    Also records the fraction of P5 samples at the sensor floor (0 bar gauge),
    a data-quality flag used by the digital shadow.
    """
    n = int(round(seconds * FS_HZ))
    out = []
    starts = cat.set_index("run_id")["start"]
    groups = dict(tuple(data.groupby("run_id", observed=True, sort=False)))
    for run_id in cat["run_id"]:  # chronological order
        g = groups[run_id]
        k = len(g) // n
        if k == 0:
            continue
        block = {c: g[c].to_numpy("float64")[: k * n].reshape(k, n) for c in set(MEAN_COLS) | set(STD_COLS)}
        rec = {f"{c}": block[c].mean(axis=1) for c in MEAN_COLS}
        rec.update({f"{c}_std": block[c].std(axis=1) for c in STD_COLS})
        rec["P5_floor_frac"] = (g["P5g"].to_numpy()[: k * n].reshape(k, n) <= 1e-6).mean(axis=1)
        df = pd.DataFrame(rec)
        df.insert(0, "window", np.arange(k))
        df.insert(0, "severity", float(g["severity"].iloc[0]))
        df.insert(0, "condition", str(g["condition"].iloc[0]))
        df.insert(0, "run_id", str(run_id))
        df["t_start_s"] = df["window"] * seconds
        df["clock"] = starts.loc[run_id] + pd.to_timedelta(df["t_start_s"], unit="s")
        out.append(df)
    return pd.concat(out, ignore_index=True)


def load_windows(seconds: float = 1.0, force: bool = False) -> pd.DataFrame:
    """Window features for every recording, cached in data/interim."""
    from .config import INTERIM_DIR
    from . import rig

    path = INTERIM_DIR / f"windows_{seconds:g}s.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    data, cat = rig.load_all()
    w = windows(data, cat, seconds)
    w["clock_h"] = w["clock"].dt.hour + w["clock"].dt.minute / 60 + w["clock"].dt.second / 3600
    w.to_parquet(path, index=False)
    return w


def run_means(w: pd.DataFrame) -> pd.DataFrame:
    """Collapse windows to one row per run (mean of window means)."""
    num = w.select_dtypes("number").columns.drop(["window", "t_start_s"], errors="ignore")
    m = w.groupby("run_id", sort=False)[list(num)].mean()
    meta = w.groupby("run_id", sort=False)[["condition", "clock"]].first()
    return meta.join(m).reset_index()


def psd(x: np.ndarray, fs: float = FS_HZ, nperseg: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density of a detrended signal."""
    from scipy.signal import welch
    return welch(x - np.mean(x), fs=fs, nperseg=nperseg, detrend="linear")
