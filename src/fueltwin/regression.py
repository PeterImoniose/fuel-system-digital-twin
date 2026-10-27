"""Regression models for notebook 07.

Three regression problems, each validated so that the test data was never
used for fitting:

* pump flow from (speed, pressure rise): polynomial degree selection, machine
  learning regressors, a Gaussian process and a hybrid "physics + GP on the
  physics residual" model with prediction intervals;
* virtual flow meters for the rig (flow from pressures), scored on a fault the
  model never saw;
* severity regression (valve travel from sensor readings), scored on a
  recording the model never saw.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from . import pumpmodel as pm
from .hydraulics import SectionLaw


def _x(rpm, dp) -> np.ndarray:
    return np.column_stack([np.asarray(rpm, float) / 1000.0, np.asarray(dp, float)])


def binned(df: pd.DataFrame, dp_bin: float = 0.05) -> pd.DataFrame:
    """Average the 200 ms pump samples into 0.05 bar bins per test (keeps GP fits small)."""
    d = df.assign(dpb=(df["dP_pump"] / dp_bin).round() * dp_bin)
    g = d.groupby(["nominal_rpm", "dpb"]).agg(rpm=("rpm", "mean"), dP_pump=("dP_pump", "mean"),
                                              Q=("Q", "mean"), n=("Q", "size")).reset_index()
    return g[g["n"] >= 3]


# --- pump regressors -------------------------------------------------------------
@dataclass
class SklearnPump(pm.PumpModel):
    name: str
    factory: object = None
    model_: object = None

    def fit(self, rpm, dp, q):
        self.model_ = self.factory().fit(_x(rpm, dp), np.asarray(q, float))
        return self

    def predict(self, rpm, dp):
        return self.model_.predict(_x(rpm, dp))


def poly_pump(degree: int) -> SklearnPump:
    return SklearnPump(f"polynomial degree {degree}",
                       lambda: make_pipeline(PolynomialFeatures(degree), LinearRegression()))


def _gp(length_scales=(0.2, 1.0)) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0) * RBF(length_scale=list(length_scales), length_scale_bounds=(1e-2, 1e2)) \
        + WhiteKernel(1e-4, (1e-7, 1e-1))
    return GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0)


@dataclass
class GPPump(pm.PumpModel):
    """Gaussian process on binned data; optionally on the residual of a physics model (hybrid).

    ``predict_dist`` returns the mean and the predictive standard deviation,
    which includes the fitted measurement noise (WhiteKernel).
    """
    hybrid: bool = False
    gp_: GaussianProcessRegressor | None = None
    base_: pm.PhysicsPump | None = None

    @property
    def name(self) -> str:
        return "physics + GP (hybrid)" if self.hybrid else "Gaussian process"

    def fit(self, rpm, dp, q):
        df = pd.DataFrame({"rpm": rpm, "dP_pump": dp, "Q": q})
        df["nominal_rpm"] = (df["rpm"] / 10).round() * 10
        b = binned(df)
        y = b["Q"].to_numpy()
        if self.hybrid:
            self.base_ = pm.PhysicsPump().fit(rpm, dp, q)
            y = y - self.base_.predict(b["rpm"], b["dP_pump"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp_ = _gp().fit(_x(b["rpm"], b["dP_pump"]), y)
        return self

    def predict_dist(self, rpm, dp):
        mu, sd = self.gp_.predict(_x(rpm, dp), return_std=True)
        if self.hybrid:
            mu = mu + self.base_.predict(rpm, dp)
        return mu, sd

    def predict(self, rpm, dp):
        return self.predict_dist(rpm, dp)[0]


def ml_pump_models() -> list[pm.PumpModel]:
    return [
        pm.PhysicsPump(),
        poly_pump(2),
        poly_pump(3),
        SklearnPump("random forest", lambda: RandomForestRegressor(300, min_samples_leaf=5, random_state=0, n_jobs=-1)),
        SklearnPump("gradient boosting", lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                                                random_state=0)),
        SklearnPump("neural network (MLP)", lambda: make_pipeline(
            StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=3000, early_stopping=True,
                                           random_state=0))),
        GPPump(hybrid=False),
        GPPump(hybrid=True),
    ]


def interval_coverage(model: GPPump, rpm, dp, q, z: float = 1.96) -> float:
    mu, sd = model.predict_dist(rpm, dp)
    return float(np.mean(np.abs(np.asarray(q) - mu) <= z * sd))


# --- virtual flow meters --------------------------------------------------------
PRESSURES = ["P1", "P2", "P3", "P4", "P5"]


@dataclass
class PhysicsFlowMeter:
    """Flow from P5 through the final pipe and meter to the sump (no fault acts on it)."""
    law: SectionLaw | None = None
    name: str = "physics: P5 -> pipe law"

    def fit(self, w: pd.DataFrame):
        cal = w[(w["P5_floor_frac"] <= 0.01) & (w["P5g"] > 0.02)]
        self.law = SectionLaw.fit(cal["Q"], cal["P5g"])
        return self

    def predict(self, w: pd.DataFrame) -> np.ndarray:
        return (w["P5g"].clip(lower=1e-6) / self.law.k) ** (1 / self.law.n)


@dataclass
class SklearnFlowMeter:
    name: str
    cols: list[str]
    factory: object
    model_: object = None

    def fit(self, w: pd.DataFrame):
        self.model_ = self.factory().fit(w[self.cols], w["Q"])
        return self

    def predict(self, w: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(w[self.cols])


def flow_meters() -> list:
    return [
        PhysicsFlowMeter(),
        SklearnFlowMeter("linear: P4, P5", ["P4", "P5"], lambda: make_pipeline(StandardScaler(), LinearRegression())),
        SklearnFlowMeter("linear: all pressures", PRESSURES, lambda: make_pipeline(StandardScaler(), LinearRegression())),
        SklearnFlowMeter("polynomial 2: all pressures", PRESSURES,
                         lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(1e-3))),
        SklearnFlowMeter("random forest: all pressures", PRESSURES,
                         lambda: RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1)),
        SklearnFlowMeter("gradient boosting: all pressures", PRESSURES,
                         lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)),
    ]


# --- severity regression -------------------------------------------------------
SEVERITY_FEATURES = ["P1", "P2", "P3", "P4", "P5", "Q"]


def severity_models() -> dict:
    return {
        "linear": lambda: make_pipeline(StandardScaler(), LinearRegression()),
        "polynomial 2": lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(1e-3)),
        "random forest": lambda: RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1),
        "gradient boosting": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0),
    }
