"""Models of the Oberdorfer N999R external gear pump.

Three model families are compared, all mapping (speed, pressure rise) -> flow:

* ``PhysicsPump``: positive-displacement law with internal slip,
  Q = D * n / 1000 - k_s * dP  (optionally - c * n/1000 * dP).
  D is the displacement [mL/rev], k_s the slip coefficient [L/min/bar].
  Two or three numbers, each with a physical meaning, so it extrapolates.
* ``PolyPump``: full quadratic polynomial in (dP, n), the usual curve fit.
* ``LookupPump``: binned 2-D lookup table with linear interpolation, the same
  idea as a MATLAB Curve Fitting / Simulink 2-D lookup model. It
  returns NaN outside the tested envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

from .config import GPM_TO_LPM, PSI_TO_BAR, RELIEF_VALVE_SETTING_BAR

# --- OEM data (N999 performance curve, water at 70 F) ----------------------
# Read off the datasheet chart: flow (GPM) falls linearly with pressure (psi).
OEM_CURVES_GPM_PSI = {
    1140: {"q0_gpm": 0.635, "slope_gpm_per_psi": -0.635 / 139.0},
    1750: {"q0_gpm": 1.01, "slope_gpm_per_psi": (0.39 - 1.01) / 150.0},
}
# Shaft power (dashed lines on the datasheet), hp, linear in psi.
OEM_POWER_HP_PSI = {1140: (0.0, 0.197 / 150.0), 1750: (0.012, (0.242 - 0.012) / 150.0)}
HP_TO_W = 745.7


def oem_flow(rpm: int, dp_bar: np.ndarray) -> np.ndarray:
    c = OEM_CURVES_GPM_PSI[rpm]
    dp_psi = np.asarray(dp_bar) / PSI_TO_BAR
    return GPM_TO_LPM * (c["q0_gpm"] + c["slope_gpm_per_psi"] * dp_psi)


def oem_power_w(rpm: int, dp_bar: np.ndarray) -> np.ndarray:
    a, b = OEM_POWER_HP_PSI[rpm]
    return HP_TO_W * (a + b * np.asarray(dp_bar) / PSI_TO_BAR)


def oem_si_table() -> pd.DataFrame:
    rows = []
    for rpm, c in OEM_CURVES_GPM_PSI.items():
        q0 = GPM_TO_LPM * c["q0_gpm"]
        slope = GPM_TO_LPM * c["slope_gpm_per_psi"] / PSI_TO_BAR
        rows.append({"rpm": rpm, "Q_at_0bar_Lpm": q0, "slope_Lpm_per_bar": slope,
                     "displacement_mL_rev": 1000 * q0 / rpm,
                     "shutoff_dP_bar": -q0 / slope})
    return pd.DataFrame(rows)


# --- Models ----------------------------------------------------------------
class PumpModel:
    name = "base"

    def fit(self, rpm, dp, q):  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, rpm, dp) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class PhysicsPump(PumpModel):
    speed_dependent_slip: bool = False
    params: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "physics (D, k_s, c)" if self.speed_dependent_slip else "physics (D, k_s)"

    def _design(self, rpm, dp):
        rpm, dp = np.asarray(rpm, float), np.asarray(dp, float)
        cols = [rpm / 1000.0, -dp]
        if self.speed_dependent_slip:
            cols.append(-dp * rpm / 1000.0)
        return np.column_stack(cols)

    def fit(self, rpm, dp, q):
        beta, *_ = np.linalg.lstsq(self._design(rpm, dp), np.asarray(q, float), rcond=None)
        self.params = {"D_mL_rev": beta[0], "k_s_Lpm_bar": beta[1]}
        if self.speed_dependent_slip:
            self.params["c_Lpm_bar_krpm"] = beta[2]
        return self

    def predict(self, rpm, dp):
        beta = [self.params["D_mL_rev"], self.params["k_s_Lpm_bar"]]
        if self.speed_dependent_slip:
            beta.append(self.params["c_Lpm_bar_krpm"])
        return self._design(rpm, dp) @ np.asarray(beta)

    def volumetric_efficiency(self, rpm, dp):
        return self.predict(rpm, dp) / (self.params["D_mL_rev"] * np.asarray(rpm) / 1000.0)


@dataclass
class PolyPump(PumpModel):
    degree: int = 2
    name: str = "quadratic polynomial"
    coef_: np.ndarray | None = None

    def _design(self, rpm, dp):
        x, y = np.asarray(dp, float), np.asarray(rpm, float) / 1000.0
        terms = [np.ones_like(x)]
        for total in range(1, self.degree + 1):
            for i in range(total + 1):
                terms.append(x ** (total - i) * y ** i)
        return np.column_stack(terms)

    def fit(self, rpm, dp, q):
        self.coef_, *_ = np.linalg.lstsq(self._design(rpm, dp), np.asarray(q, float), rcond=None)
        return self

    def predict(self, rpm, dp):
        return self._design(rpm, dp) @ self.coef_


@dataclass
class LookupPump(PumpModel):
    dp_bin: float = 0.05
    name: str = "2-D lookup table"
    interp_: LinearNDInterpolator | None = None
    table_: pd.DataFrame | None = None

    def fit(self, rpm, dp, q):
        df = pd.DataFrame({"rpm": np.asarray(rpm, float), "dp": np.asarray(dp, float), "q": np.asarray(q, float)})
        df["rpm_r"] = df["rpm"].round(-1)
        df["dp_b"] = (df["dp"] / self.dp_bin).round() * self.dp_bin
        tab = df.groupby(["rpm_r", "dp_b"], as_index=False).agg(q=("q", "mean"), n=("q", "size"))
        tab = tab[tab["n"] >= 3]
        self.table_ = tab
        # scale speed so a 50 rpm step is comparable to a 0.5 bar step
        self.interp_ = LinearNDInterpolator(np.c_[tab["rpm_r"] / 100.0, tab["dp_b"]], tab["q"])
        return self

    def predict(self, rpm, dp):
        return self.interp_(np.asarray(rpm, float) / 100.0, np.asarray(dp, float))


def default_models() -> list[PumpModel]:
    return [PhysicsPump(), PhysicsPump(speed_dependent_slip=True), PolyPump(), LookupPump()]


# --- Evaluation --------------------------------------------------------------
def metrics(q_true, q_pred) -> dict:
    q_true, q_pred = np.asarray(q_true, float), np.asarray(q_pred, float)
    ok = np.isfinite(q_pred)
    e = q_pred[ok] - q_true[ok]
    return {
        "coverage_pct": 100 * ok.mean(),
        "rmse_Lpm": float(np.sqrt(np.mean(e**2))) if ok.any() else np.nan,
        "bias_Lpm": float(np.mean(e)) if ok.any() else np.nan,
        "mape_pct": float(100 * np.mean(np.abs(e) / q_true[ok])) if ok.any() else np.nan,
        "max_abs_pct": float(100 * np.max(np.abs(e) / q_true[ok])) if ok.any() else np.nan,
    }


def leave_one_speed_out(df: pd.DataFrame, models_factory=default_models) -> pd.DataFrame:
    """Hold out each test speed in turn; fit on the others; score on the held-out one."""
    rows = []
    for rpm in sorted(df["nominal_rpm"].unique()):
        train, test = df[df["nominal_rpm"] != rpm], df[df["nominal_rpm"] == rpm]
        for m in models_factory():
            m.fit(train["rpm"], train["dP_pump"], train["Q"])
            rows.append({"held_out_rpm": rpm, "model": m.name,
                         **metrics(test["Q"], m.predict(test["rpm"], test["dP_pump"]))})
    return pd.DataFrame(rows)


def random_split_score(df: pd.DataFrame, models_factory=default_models, seed: int = 0) -> pd.DataFrame:
    """The optimistic score you get from a shuffled 80/20 row split (for contrast only)."""
    rng = np.random.default_rng(seed)
    test_mask = rng.random(len(df)) < 0.2
    train, test = df[~test_mask], df[test_mask]
    rows = []
    for m in models_factory():
        m.fit(train["rpm"], train["dP_pump"], train["Q"])
        rows.append({"model": m.name, **metrics(test["Q"], m.predict(test["rpm"], test["dP_pump"]))})
    return pd.DataFrame(rows)


def relief_onset(test: pd.DataFrame, fit_below: float = 2.8, window: int = 15,
                 min_drop: float = 0.05) -> dict:
    """Where does flow leave the straight pump line? (relief valve cracking)

    A straight line is fitted to the part of the test well below the relief
    setting; onset is the first pressure rise at which the smoothed flow falls
    more than ``max(min_drop, 4 sd)`` below that line.
    """
    g = test.sort_values("sample")
    lo = g[g["dP_pump"] < min(fit_below, 0.75 * g["dP_pump"].max())]
    slope, icpt = np.polyfit(lo["dP_pump"], lo["Q"], 1)
    resid = (g["Q"] - (icpt + slope * g["dP_pump"])).rolling(window, center=True).mean()
    thr = max(min_drop, 4 * float(np.std(lo["Q"] - (icpt + slope * lo["dP_pump"]))))
    hit = np.flatnonzero(resid.to_numpy() < -thr)
    return {
        "onset_dP_bar": float(g["dP_pump"].iloc[hit[0]]) if len(hit) else np.nan,
        "max_dP_bar": float(g["dP_pump"].max()),
        "flow_shortfall_at_end_Lpm": float(-resid.dropna().iloc[-1]),
        "line_Q0_Lpm": float(icpt), "line_slope_Lpm_bar": float(slope),
        "relief_setting_bar": RELIEF_VALVE_SETTING_BAR,
    }
