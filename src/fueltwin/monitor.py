"""Live-monitoring methods for notebook 08.

* Healthy-only anomaly detectors (no fault labels at all).
* Change detection on a stream: Hotelling T^2 against a reference window,
  followed by a CUSUM or by Bayesian online change-point detection
  (Gaussian model with unknown mean and variance).
* Kalman tracking of health indicators: each indicator is a random walk
  observed with noise; the filter gives a smoothed estimate with an
  uncertainty band, and grows its uncertainty over gaps between recordings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# --- healthy-only anomaly detection -----------------------------------------------
DETECTORS = ["Mahalanobis (shrinkage)", "Isolation Forest", "One-class SVM", "Autoencoder"]


def anomaly_scores(name: str, x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fit on healthy windows only; return (train scores, test scores), higher = more anomalous."""
    sc = StandardScaler().fit(x_train)
    a, b = sc.transform(x_train), sc.transform(x_test)
    if name.startswith("Mahalanobis"):
        m = LedoitWolf().fit(a)
        return m.mahalanobis(a), m.mahalanobis(b)
    if name == "Isolation Forest":
        m = IsolationForest(n_estimators=300, random_state=0).fit(a)
        return -m.score_samples(a), -m.score_samples(b)
    if name == "One-class SVM":
        m = OneClassSVM(nu=0.01, gamma="scale").fit(a)
        return -m.score_samples(a), -m.score_samples(b)
    if name == "Autoencoder":
        m = MLPRegressor(hidden_layer_sizes=(8, 2, 8), max_iter=5000, random_state=0).fit(a, a)
        return ((m.predict(a) - a) ** 2).sum(1), ((m.predict(b) - b) ** 2).sum(1)
    raise ValueError(name)


# --- change detection ----------------------------------------------------------------
def hotelling_t2(x: np.ndarray, ref: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    d = x - ref.mean(axis=0)
    return np.einsum("ij,jk,ik->i", d, inv_cov, d)


def cusum(stat: np.ndarray, k: float, h: float, start: int = 0) -> int | None:
    """One-sided CUSUM: S = max(0, S + stat - k); alarm index when S > h."""
    s = 0.0
    for i in range(start, len(stat)):
        s = max(0.0, s + stat[i] - k)
        if s > h:
            return i
    return None


def bocpd(x: np.ndarray, hazard: float = 1 / 120, mu0: float = 0.0, kappa0: float = 1.0,
          alpha0: float = 1.0, beta0: float = 1.0) -> np.ndarray:
    """Bayesian online change-point detection, Gaussian data with unknown mean and variance.

    Returns the run-length posterior as a (T, T+1) array (row t: P(run length | x_1..t)).
    """
    T = len(x)
    post = np.zeros((T, T + 1))
    r = np.array([1.0])
    mu, ka, al, be = (np.array([v]) for v in (mu0, kappa0, alpha0, beta0))
    for t in range(T):
        df = 2 * al
        scale = np.sqrt(be * (ka + 1) / (al * ka))
        z = (x[t] - mu) / scale
        logp = (gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(np.pi * df) - np.log(scale)
                - (df + 1) / 2 * np.log1p(z ** 2 / df))
        p = np.exp(logp)
        growth = r * p * (1 - hazard)
        cp = (r * p * hazard).sum()
        r = np.r_[cp, growth]
        r /= r.sum()
        post[t, : len(r)] = r
        mu_n = (ka * mu + x[t]) / (ka + 1)
        be_n = be + ka * (x[t] - mu) ** 2 / (2 * (ka + 1))
        mu, ka, al, be = np.r_[mu0, mu_n], np.r_[kappa0, ka + 1], np.r_[alpha0, al + 0.5], np.r_[beta0, be_n]
    return post


def bocpd_alarm(post: np.ndarray, window: int = 5, threshold: float = 0.9, start: int = 0) -> int | None:
    """Alarm when the posterior probability that a change happened within the last ``window``
    steps exceeds ``threshold``."""
    for t in range(max(start, window), post.shape[0]):
        if post[t, :window].sum() > threshold:
            return t
    return None


# --- Kalman tracking -------------------------------------------------------------------
def kalman_track(y: np.ndarray, t: np.ndarray, r: float, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Random-walk Kalman filter for one indicator.

    ``r`` is the measurement-noise variance (from healthy windows). The process
    noise is set from a chosen response time ``tau`` (in samples): in steady
    state the filter then moves a fraction 1 - exp(-1/tau) of the way to each
    new reading. Over gaps longer than one sample the uncertainty grows in
    proportion to the gap, so the filter re-converges quickly after a pause.
    """
    k = 1 - np.exp(-1 / tau)
    q = k * k * r / (1 - k)
    x, p, last = float(y[0]), r, float(t[0])
    xs, sds = np.empty(len(y)), np.empty(len(y))
    for i, (yi, ti) in enumerate(zip(y, t)):
        p += q * max(ti - last, 1.0)
        last = ti
        gain = p / (p + r)
        x += gain * (yi - x)
        p *= 1 - gain
        xs[i], sds[i] = x, np.sqrt(p)
    return xs, sds
