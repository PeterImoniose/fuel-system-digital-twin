"""Physics-informed health indicators for the fuel rig (Tasks 3 and 5).

Each pipe section obeys a pressure-flow law dP = k * Q^n while its component
is healthy. Comparing the measured drop with the law gives a residual that is
near zero when that section is healthy and grows when it degrades:

    r_filter = dP_filter / (k_f Q^n_f) - 1    (P1 -> P2)
    r_valve  = dP_valve  / (k_v Q^n_v) - 1    (P3 -> P4)
    r_nozzle = dP_nozzle / (k_n Q^n_n) - 1    (P4 -> P5)
    r_pump   = (Q_pump(dP_pump) - Q) / Q_pump  (share of pump delivery that
                                               never reaches the flow meter)

All residuals use the flow measured at the end of the line, Q. Where a fault
diverts flow before the meter (pump bypass F2, leak F4), sections upstream of
the diversion carry more than Q, so their residual rises too. That is what
makes every fault's pattern different:

           r_filter  r_pump  r_valve  r_nozzle
    F1        +        0       0        0      (filter resistance up)
    F2        +        +       0        0      (delivery lost at the pump)
    F3        0        0       +        0      (valve resistance up)
    F4        +        +       +        0      (leak after the valve: extra
                                                flow through filter, pump and
                                                valve, lost before the meter)
    F5        0        0       0        +      (nozzle resistance up)

The tutor's diagnostic tree leaves F1/F2/F4 as one ambiguity group because it
reads raw pressures; the residuals separate them.

Calibration rule (no peeking): a section law is fitted only on windows where
that section is known to be healthy and to carry exactly Q. When the indicators
are evaluated on fault Fx, every law is refitted with all Fx windows removed
("leave-one-fault-out"), so no fault ever sees its own data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SECTIONS = {
    # name: (pressure-drop column, conditions where the section is healthy and carries Q)
    "filter": ("dP_filter", ["HC", "F3", "F5", "F1_mild"]),
    "valve": ("dP_valve", ["HC", "F1", "F2", "F5"]),
    "nozzle": ("dP_nozzle", ["HC", "F1", "F2", "F3", "F4"]),
}
# The pump law is fitted where all pump delivery reaches the meter and the pump is healthy.
PUMP_CAL = ["HC", "F1_mild", "F3", "F5"]
RESIDUALS = ["r_filter", "r_pump", "r_valve", "r_nozzle"]
SIGNATURES = pd.DataFrame(
    [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1]],
    index=["F1", "F2", "F3", "F4", "F5"], columns=RESIDUALS)
F1_MILD_MAX = 40.0  # filter-clog settings above this cavitate the pump suction
P5_FLOOR_MAX = 0.01  # windows with P5 clipped at 0 bar gauge are not used for calibration


def _select(w: pd.DataFrame, conds: list[str], exclude: str | None) -> pd.DataFrame:
    keep = np.zeros(len(w), bool)
    for c in conds:
        if c == "F1_mild":
            m = (w["condition"] == "F1") & (w["severity"] <= F1_MILD_MAX)
        else:
            m = w["condition"] == c
        keep |= m.to_numpy()
    if exclude is not None:
        keep &= (w["condition"] != exclude).to_numpy()
    keep &= (w["P5_floor_frac"] <= P5_FLOOR_MAX).to_numpy()
    return w[keep]


@dataclass
class SectionLaw:
    k: float
    n: float

    def dp(self, q):
        return self.k * np.asarray(q, float) ** self.n

    @classmethod
    def fit(cls, q, dp) -> "SectionLaw":
        n, logk = np.polyfit(np.log(q), np.log(dp), 1)
        return cls(k=float(np.exp(logk)), n=float(n))


@dataclass
class PumpLaw:
    """Straight pump line at the rig's fixed speed: Q = q0 + slope * dP."""
    q0: float
    slope: float
    rpm: float

    def q(self, dp):
        return self.q0 + self.slope * np.asarray(dp, float)

    @property
    def displacement_mL_rev(self) -> float:
        return 1000.0 * self.q0 / self.rpm

    @classmethod
    def fit(cls, dp, q, rpm) -> "PumpLaw":
        slope, q0 = np.polyfit(dp, q, 1)
        return cls(q0=float(q0), slope=float(slope), rpm=float(np.mean(rpm)))


@dataclass
class ResidualModel:
    sections: dict[str, SectionLaw]
    pump: PumpLaw
    thresholds: pd.Series = field(default_factory=pd.Series)
    excluded: str | None = None

    @classmethod
    def fit(cls, w: pd.DataFrame, exclude: str | None = None, k_sigma: float = 4.0) -> "ResidualModel":
        laws = {}
        for name, (col, conds) in SECTIONS.items():
            cal = _select(w, conds, exclude)
            laws[name] = SectionLaw.fit(cal["Q"], cal[col])
        pc = _select(w, PUMP_CAL, exclude)
        model = cls(sections=laws, pump=PumpLaw.fit(pc["dP_pump"], pc["Q"], pc["rpm"]), excluded=exclude)
        model.thresholds = model._thresholds(w, exclude, k_sigma)
        return model

    def residuals(self, w: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=w.index)
        for name, (col, _) in SECTIONS.items():
            out[f"r_{name}"] = w[col] / self.sections[name].dp(w["Q"]) - 1.0
        qp = self.pump.q(w["dP_pump"])
        out["r_pump"] = (qp - w["Q"]) / qp
        return out[RESIDUALS]

    def _thresholds(self, w: pd.DataFrame, exclude: str | None, k_sigma: float) -> pd.Series:
        """k-sigma limits from each residual's scatter on its own calibration windows.

        The scatter includes model error, drift through the day and noise, so the
        limits are wider than limits built from the healthy runs alone.
        """
        thr = {}
        for name, (_, conds) in SECTIONS.items():
            cal = _select(w, conds, exclude)
            r = self.residuals(cal)[f"r_{name}"]
            thr[f"r_{name}"] = k_sigma * _robust_sd(r)
        cal = _select(w, PUMP_CAL, exclude)
        thr["r_pump"] = k_sigma * _robust_sd(self.residuals(cal)["r_pump"])
        return pd.Series(thr)[RESIDUALS]

    def normalised(self, w: pd.DataFrame) -> pd.DataFrame:
        return self.residuals(w) / self.thresholds


def _robust_sd(x: pd.Series) -> float:
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def isolate(e: pd.DataFrame) -> pd.Series:
    """Rule-based fault isolation on normalised residuals (|e| > 1 = beyond limit).

    1. nothing beyond its limit                     -> HC
    2. nozzle residual is the largest               -> F5
    3. valve residual largest and no lost delivery  -> F3
    4. lost delivery (pump residual beyond limit):
         valve residual also raised                 -> F4 (leak after the valve)
         otherwise                                  -> F2 (pump delivery loss)
    5. filter residual beyond limit                 -> F1
    """
    e = e[RESIDUALS]
    pos = e.clip(lower=0)
    largest = pos.idxmax(axis=1)
    active = e > 1.0
    out = pd.Series("HC", index=e.index, dtype=object)
    any_active = active.any(axis=1)
    rule5 = any_active & active["r_filter"]
    out[rule5] = "F1"
    lost = any_active & active["r_pump"]
    out[lost & active["r_valve"]] = "F4"
    out[lost & ~active["r_valve"]] = "F2"
    out[any_active & (largest == "r_valve") & ~active["r_pump"]] = "F3"
    out[any_active & (largest == "r_nozzle") & active["r_nozzle"]] = "F5"
    # something is beyond a limit but matches no rule
    out[any_active & (out == "HC")] = "unknown"
    return out


def physical_severity(r: pd.DataFrame, fault: pd.Series) -> pd.Series:
    """Severity in physical units for the isolated fault.

    F1/F3/F5: resistance ratio of the faulty section (1 = healthy).
    F2/F4:    share of pump delivery lost before the meter (0 = healthy).
    """
    out = pd.Series(np.nan, index=r.index)
    for f, col in {"F1": "r_filter", "F3": "r_valve", "F5": "r_nozzle"}.items():
        m = fault == f
        out[m] = 1.0 + r.loc[m, col]
    for f in ("F2", "F4"):
        m = fault == f
        out[m] = r.loc[m, "r_pump"]
    return out


def lofo_residuals(w: pd.DataFrame, k_sigma: float = 4.0) -> tuple[pd.DataFrame, dict[str, ResidualModel]]:
    """Residuals for every window, each fault scored with laws that never saw it.

    Healthy windows are scored with the model fitted on everything (they are in
    every calibration set by construction), which is reported as a limitation.
    """
    models = {f: ResidualModel.fit(w, exclude=f, k_sigma=k_sigma) for f in ["F1", "F2", "F3", "F4", "F5"]}
    models["all"] = ResidualModel.fit(w, exclude=None, k_sigma=k_sigma)
    parts = []
    for cond, g in w.groupby("condition", sort=False):
        m = models.get(cond, models["all"])
        r = m.residuals(g)
        e = r / m.thresholds
        e.columns = [c.replace("r_", "e_") for c in e.columns]
        parts.append(pd.concat([r, e], axis=1))
    return pd.concat(parts).loc[w.index], models
