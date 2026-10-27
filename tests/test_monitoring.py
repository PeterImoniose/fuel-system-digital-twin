"""Tests for the regression (notebook 07) and monitoring (notebook 08) modules."""

import numpy as np
import pandas as pd
import pytest

from fueltwin import monitor as mo, regression as rg


def _pump_data(rng):
    rpm = np.repeat([700.0, 750, 800, 850, 900, 950], 120)
    dp = np.tile(np.linspace(1.2, 3.2, 120), 6)
    q = 2.1 * rpm / 1000 - 0.2 * dp + 0.01 * np.sin(3 * dp) + rng.normal(0, 0.003, len(dp))
    return rpm, dp, q


def test_hybrid_gp_tracks_physics_and_gives_intervals():
    rng = np.random.default_rng(0)
    rpm, dp, q = _pump_data(rng)
    m = rg.GPPump(hybrid=True).fit(rpm, dp, q)
    mu, sd = m.predict_dist(np.array([825.0]), np.array([2.0]))
    truth = 2.1 * 0.825 - 0.2 * 2.0 + 0.01 * np.sin(6.0)
    assert mu[0] == pytest.approx(truth, abs=0.01)
    assert 0 < sd[0] < 0.05
    # outside the data the GP correction fades and the physics part remains
    far = m.predict(np.array([1400.0]), np.array([1.0]))[0]
    assert far == pytest.approx(2.1 * 1.4 - 0.2, abs=0.05)


def test_polynomial_degree_pipeline_runs():
    rng = np.random.default_rng(1)
    rpm, dp, q = _pump_data(rng)
    for deg in (1, 2, 3):
        p = rg.poly_pump(deg).fit(rpm, dp, q).predict(rpm[:5], dp[:5])
        assert np.all(np.isfinite(p))


def test_physics_flow_meter_inverts_its_law():
    q = np.linspace(0.4, 0.75, 40)
    w = pd.DataFrame({"Q": q, "P5g": 0.3 * q ** 3, "P5_floor_frac": 0.0})
    fm = rg.PhysicsFlowMeter().fit(w)
    assert np.allclose(fm.predict(w), q, rtol=1e-6)


def test_cusum_detects_step_not_noise():
    rng = np.random.default_rng(2)
    x = rng.chisquare(6, 200)
    assert mo.cusum(x, k=12.6, h=50) is None
    x[100:] += 40
    al = mo.cusum(x, k=12.6, h=50)
    assert al is not None and 100 <= al <= 103


def test_bocpd_finds_mean_shift():
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(0, 1, 80), rng.normal(5, 1, 80)]
    post = mo.bocpd(x, hazard=1 / 100)
    al = mo.bocpd_alarm(post, window=5, threshold=0.9, start=10)
    assert al is not None and 80 <= al <= 85


def test_kalman_track_smooths_and_follows_step():
    rng = np.random.default_rng(4)
    y = np.r_[np.zeros(100), np.ones(100)] + rng.normal(0, 0.1, 200)
    t = np.arange(200.0)
    x, sd = mo.kalman_track(y, t, r=0.01, tau=3)
    assert np.std(x[50:100]) < np.std(y[50:100])
    assert abs(x[-1] - 1) < 0.1
    assert np.all(sd > 0)
    # a long gap inflates the uncertainty
    t_gap = t.copy()
    t_gap[150:] += 3600
    _, sd_gap = mo.kalman_track(y, t_gap, r=0.01, tau=3)
    assert sd_gap[150] > sd[150]


def test_anomaly_detectors_rank_outliers_higher():
    rng = np.random.default_rng(5)
    tr = pd.DataFrame(rng.normal(0, 1, (300, 4)))
    te = pd.DataFrame(np.r_[rng.normal(0, 1, (20, 4)), rng.normal(6, 1, (20, 4))])
    for det in mo.DETECTORS:
        _, s = mo.anomaly_scores(det, tr, te)
        assert s[20:].mean() > s[:20].mean(), det
