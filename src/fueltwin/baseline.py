"""Healthy-condition baseline (Task 2).

The healthy reference is built from window means, not from raw 1 kHz samples.
Raw samples mostly measure sensor noise and pump ripple; a fault shifts the
operating point, which is what window means track. Two sources of healthy
variation are kept apart:

* within-run: window-to-window scatter inside one 60 s recording;
* between-run: how much the run averages move from one recording to the next.

Only six healthy runs exist, all within five minutes of each other, so the
between-run term certainly underestimates day-scale variation. That limitation
is measured later (Task 3) and is the main reason the twin needs drift checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HealthyBaseline:
    mean: pd.Series
    sd_within: pd.Series
    sd_between: pd.Series
    n_runs: int
    n_windows: int

    @property
    def sd_total(self) -> pd.Series:
        return np.sqrt(self.sd_within**2 + self.sd_between**2)

    @classmethod
    def fit(cls, w: pd.DataFrame, cols: list[str]) -> "HealthyBaseline":
        hc = w[w["condition"] == "HC"]
        run_means = hc.groupby("run_id")[cols].mean()
        within = hc.groupby("run_id")[cols].std().pow(2).mean().pow(0.5)
        return cls(mean=run_means.mean(), sd_within=within, sd_between=run_means.std(),
                   n_runs=len(run_means), n_windows=len(hc))

    def zscores(self, w: pd.DataFrame) -> pd.DataFrame:
        cols = list(self.mean.index)
        return (w[cols] - self.mean) / self.sd_total

    def envelope(self, k: float = 4.0) -> pd.DataFrame:
        """Healthy limits mean +/- k * total sd, per variable."""
        return pd.DataFrame({
            "mean": self.mean, "sd_within": self.sd_within, "sd_between": self.sd_between,
            "sd_total": self.sd_total, "lower": self.mean - k * self.sd_total,
            "upper": self.mean + k * self.sd_total,
        })


def variance_table(w: pd.DataFrame, cols: list[str], raw_sd: pd.Series | None = None) -> pd.DataFrame:
    """How big is each source of healthy variation, per variable."""
    hc = w[w["condition"] == "HC"]
    run_means = hc.groupby("run_id")[cols].mean()
    tab = pd.DataFrame({
        "mean": run_means.mean(),
        "sd_between_runs": run_means.std(),
        "sd_within_run_1s": hc.groupby("run_id")[cols].std().pow(2).mean().pow(0.5),
        "range_between_runs": run_means.max() - run_means.min(),
    })
    if raw_sd is not None:
        tab.insert(1, "sd_raw_1kHz", raw_sd.reindex(cols))
    tab["cv_between_pct"] = 100 * tab["sd_between_runs"] / tab["mean"].abs()
    return tab
