"""The digital shadow: a running health assessment of the fuel rig (Task 5).

For every 1 s window the shadow produces:

* health indicators: four physics residuals (filter, pump, valve, nozzle);
* a diagnosis from the physics rules, and a second opinion from a classifier
  trained only on simulated faults (``simulate.py``);
* a severity in physical units (resistance ratio or lost delivery);
* a quality indicator (green / yellow / red, after DNV-RP-A204 section 2.5)
  saying how far the result can be trusted, with the reasons;
* an alarm state that only changes when a diagnosis persists, so single noisy
  windows do not raise or clear alarms.

In Kritzinger et al.'s (2018) terms this is a digital shadow: data flows
automatically from the asset to the model, but nothing flows back.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import hydraulics as hy

RAW = ["P1", "P2", "P3", "P4", "P5", "Q"]


@dataclass
class QualityPolicy:
    rpm_nominal: float = 400.0
    rpm_tolerance: float = 5.0  # component laws were identified at 400 rpm
    p5_floor_max: float = 0.01  # share of P5 samples at 0 bar gauge before the nozzle residual is unreliable
    pump_drift_limit: float = 0.5  # pump residual (in limits) above which drift and fault are hard to separate


@dataclass
class DigitalShadow:
    residual_model: hy.ResidualModel
    classifier: object | None = None  # trained on simulated windows, raw features
    policy: QualityPolicy = field(default_factory=QualityPolicy)
    persistence: int = 5  # windows that must agree before the alarm state changes
    _history: deque = field(default_factory=lambda: deque(maxlen=60))
    _alarm: str = "HC"

    def assess(self, w: pd.DataFrame) -> pd.DataFrame:
        """Assess a batch of windows independently (no alarm memory)."""
        r = self.residual_model.residuals(w)
        e = r / self.residual_model.thresholds
        rule = hy.isolate(e)
        out = pd.DataFrame(index=w.index)
        for c in hy.RESIDUALS:
            out[c] = r[c]
            out[c.replace("r_", "e_")] = e[c]
        out["rules"] = rule
        out["severity"] = hy.physical_severity(r, rule)
        if self.classifier is not None:
            out["classifier"] = self.classifier.predict(w[RAW])
        q, reasons = self._quality(w, e, out)
        out["quality"] = q
        out["quality_reasons"] = reasons
        return out

    def _quality(self, w: pd.DataFrame, e: pd.DataFrame, out: pd.DataFrame):
        pol = self.policy
        level = pd.Series(0, index=w.index)
        reasons = pd.Series("", index=w.index, dtype=object)

        def flag(mask, lvl, text):
            nonlocal level
            mask = mask.fillna(False)
            level = level.where(~mask | (level >= lvl), lvl)
            reasons.loc[mask] = reasons.loc[mask] + text + "; "

        flag((w["rpm"] - pol.rpm_nominal).abs() > pol.rpm_tolerance, 2, "speed outside the calibrated range")
        floor = w["P5_floor_frac"] > pol.p5_floor_max
        flag(floor & (out["rules"] == "F5"), 2, "P5 clipped at 0 bar: nozzle residual invalid")
        flag(floor & (out["rules"] != "F5"), 1, "P5 clipped at 0 bar")
        flag((out["rules"] == "unknown"), 1, "pattern matches no known fault")
        if "classifier" in out:
            flag(out["rules"] != out["classifier"], 1, "rules and classifier disagree")
        mild_pump = (e["r_pump"] > pol.pump_drift_limit) & (e["r_pump"] < 1.5) & out["rules"].isin(["F2", "F4", "HC"])
        flag(mild_pump, 1, "small pump shortfall: drift or early pump/leak fault")
        names = {0: "green", 1: "yellow", 2: "red"}
        return level.map(names), reasons.str.rstrip("; ")

    def stream(self, w: pd.DataFrame) -> pd.DataFrame:
        """Assess windows in time order and track a debounced alarm state."""
        out = self.assess(w)
        alarms = []
        for diag in out["rules"]:
            self._history.append(diag)
            recent = list(self._history)[-self.persistence:]
            if len(recent) == self.persistence and all(d == recent[0] for d in recent) and recent[0] != self._alarm:
                self._alarm = recent[0]
            alarms.append(self._alarm)
        out["alarm"] = alarms
        return out

    def reset(self) -> None:
        self._history.clear()
        self._alarm = "HC"
