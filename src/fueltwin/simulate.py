"""A steady-state hydraulic network model of the fuel rig, used to simulate faults.

The model joins the component laws identified on the rig (pump line, filter,
valve and nozzle sections, and the final pipe + flow meter to the sump):

    P2 = P1 - s_f * k_f * Qp^n_f                      filter (clog factor s_f)
    Qp = q0 * (1 + drift) + slope * (P3 - P2)         gear pump at 400 rpm
    Qb = g_b * sqrt(P3 - P_atm)                       pump bypass to tank (F2)
    Qv = Qp - Qb
    P4 = P3 - s_v * k_v * Qv^n_v                      shut-off valve (F3)
    Ql = g_l * sqrt(P4 - P_atm)                       leak to tank (F4)
    Qn = Qv - Ql                                      flow reaching the meter
    P5 = P4 - s_n * k_n * Qn^n_n                      nozzle (F5)
    P5 - P_atm = k_d * Qn^n_d                         pipe + meter to the sump

Two unknowns (Qp, P3) are solved so that the pump law and the downstream law
both hold. Faults are injected by changing s_f, g_b, s_v, g_l, s_n; slow drift
(pump delivery, tank level) and measurement noise can be added, so the model
can generate labelled training data for faults that were never recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from .config import P_ATM_BAR
from .hydraulics import ResidualModel, SectionLaw, _select

FAULT_PARAM = {"F1": "s_f", "F2": "g_b", "F3": "s_v", "F4": "g_l", "F5": "s_n"}


@dataclass
class RigModel:
    p1: float
    filt: SectionLaw
    valve: SectionLaw
    nozzle: SectionLaw
    down: SectionLaw
    q0: float
    slope: float

    @classmethod
    def from_windows(cls, w: pd.DataFrame, exclude: str | None = None) -> "RigModel":
        """Calibrate from data, using the same leave-one-fault-out rule as the residuals."""
        rm = ResidualModel.fit(w, exclude=exclude)
        # downstream pipe + meter: no fault acts on it; calibrate where P5 is not clipped
        cal = _select(w, ["HC", "F1", "F3", "F5"], exclude)
        down = SectionLaw.fit(cal["Q"], cal["P5g"])
        hc = w[w["condition"] == "HC"]
        return cls(p1=float(hc["P1"].mean()), filt=rm.sections["filter"], valve=rm.sections["valve"],
                   nozzle=rm.sections["nozzle"], down=down, q0=rm.pump.q0, slope=rm.pump.slope)

    def solve(self, s_f=1.0, g_b=0.0, s_v=1.0, g_l=0.0, s_n=1.0, drift=0.0, p1_shift=0.0) -> dict:
        p1 = self.p1 + p1_shift

        def branch(x):
            qp, p3 = x
            qp = max(qp, 1e-4)
            p2 = p1 - s_f * self.filt.k * qp ** self.filt.n
            qb = g_b * np.sqrt(max(p3 - P_ATM_BAR, 0.0))
            qv = max(qp - qb, 1e-5)
            p4 = p3 - s_v * self.valve.k * qv ** self.valve.n
            ql = g_l * np.sqrt(max(p4 - P_ATM_BAR, 0.0))
            qn = max(qv - ql, 1e-5)
            p5 = p4 - s_n * self.nozzle.k * qn ** self.nozzle.n
            return p2, qb, qv, p4, ql, qn, p5

        def resid(x):
            qp, p3 = x
            p2, qb, qv, p4, ql, qn, p5 = branch(x)
            r1 = qp - (self.q0 * (1 + drift) + self.slope * (p3 - p2))
            r2 = (p5 - P_ATM_BAR) - self.down.k * qn ** self.down.n
            return [r1, r2]

        sol, info, ok, _ = fsolve(resid, [0.7, 1.33], full_output=True)
        p2, qb, qv, p4, ql, qn, p5 = branch(sol)
        return {"P1": p1, "P2": p2, "P3": sol[1], "P4": p4, "P5": p5, "Q": qn, "Qp": sol[0], "Qb": qb, "Ql": ql,
                "converged": ok == 1 and max(abs(np.asarray(resid(sol)))) < 1e-7}


def sample_faults(model: RigModel, n_per_class: int, noise_sd: pd.Series, rng: np.random.Generator,
                  drift_sd: float = 0.015, p1_sd: float = 0.004) -> pd.DataFrame:
    """Labelled synthetic windows: healthy plus each fault over a broad severity range.

    Severity ranges are set by physics, not by the recorded settings: blockages
    up to 25x resistance, bypass/leak up to ~60 % lost delivery. Drift and noise
    are randomised so the classifier cannot rely on one exact healthy point.
    """
    ranges = {"F1": ("s_f", 1.0, 25.0, "log"), "F2": ("g_b", 0.0, 2.5, "lin"), "F3": ("s_v", 1.0, 15.0, "log"),
              "F4": ("g_l", 0.0, 2.5, "lin"), "F5": ("s_n", 1.0, 12.0, "log")}
    rows = []
    for cond in ["HC", *ranges]:
        for _ in range(n_per_class):
            kw = {}
            if cond != "HC":
                p, lo, hi, scale = ranges[cond]
                u = rng.uniform()
                kw[p] = lo * (hi / lo) ** u if scale == "log" else lo + (hi - lo) * u
            kw["drift"] = rng.normal(0, drift_sd)
            kw["p1_shift"] = rng.normal(0, p1_sd)
            out = model.solve(**kw)
            if not out["converged"]:
                continue
            for c in ["P1", "P2", "P3", "P4", "P5", "Q"]:
                out[c] += rng.normal(0, noise_sd[c])
            out["P5g"] = max(out["P5"] - P_ATM_BAR, 0.0)  # the real sensor clips at 0 bar gauge
            out["P5"] = out["P5g"] + P_ATM_BAR
            out.update({"condition": cond, **{k: v for k, v in kw.items() if k in FAULT_PARAM.values()}})
            rows.append(out)
    df = pd.DataFrame(rows)
    df["dP_filter"] = df["P1"] - df["P2"]
    df["dP_pump"] = df["P3"] - df["P2"]
    df["dP_valve"] = df["P3"] - df["P4"]
    df["dP_nozzle"] = df["P4"] - df["P5"]
    df["P5_floor_frac"] = (df["P5g"] <= 1e-6).astype(float)
    df["rpm"] = 400.0
    return df
