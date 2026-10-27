"""Unit tests for the parts of the pipeline that decide the results.

Most tests use synthetic inputs so they run without the raw data; the last
group checks the real dataset and is skipped when it is not available.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fueltwin import config, hydraulics as hy, pump, pumpmodel as pm, simulate as sm
from fueltwin.lvm import read_lvm

LVM_SAMPLE = """LabVIEW Measurement\t
Writer_Version\t2
Date\t2014/05/30
Time\t10:25:38.11
***End_of_Header***\t
\t
Channels\t3\t\t\t
Date\t2014/05/30\t2014/05/30\t2014/05/30\t
Time\t14:56:17.61\t14:56:17.62\t14:56:17.63\t
***End_of_Header***\t\t\t\t
X_Value\tA\tB\tC\tComment
0.000000\t1.0\t2.0\t3.0
0.001000\t1.5\t2.5\t3.5
"""


def test_lvm_reader_finds_header_and_recording_time(tmp_path: Path):
    f = tmp_path / "x.lvm"
    f.write_text(LVM_SAMPLE)
    lvm = read_lvm(f)
    assert list(lvm.data.columns) == ["X_Value", "A", "B", "C"]
    assert lvm.data.shape == (2, 4)
    # the program-start time must not be mistaken for the recording start
    assert lvm.header["Time"].startswith("10:25")
    assert lvm.header["channel_Time"].startswith("14:56")


def test_valve_command_scale():
    assert pump.cmd_to_opening(np.array([20000.0]))[0] == pytest.approx(100.0)
    assert pump.cmd_to_opening(np.array([4000.0]))[0] == pytest.approx(0.0)
    assert pump.cmd_to_opening(np.array([9584.0]))[0] == pytest.approx(34.9, abs=0.05)


def test_oem_conversion():
    assert config.RELIEF_VALVE_SETTING_BAR == pytest.approx(3.447, abs=1e-3)
    assert pm.oem_flow(1140, 0.0) == pytest.approx(0.635 * 3.785411784)
    tab = pm.oem_si_table().set_index("rpm")
    assert tab.loc[1140, "displacement_mL_rev"] == pytest.approx(2.109, abs=0.005)


def test_physics_pump_recovers_parameters():
    rng = np.random.default_rng(0)
    rpm = rng.choice([700, 800, 900], 600)
    dp = rng.uniform(1, 3, 600)
    q = 2.1 * rpm / 1000 - 0.2 * dp + rng.normal(0, 0.002, 600)
    m = pm.PhysicsPump().fit(rpm, dp, q)
    assert m.params["D_mL_rev"] == pytest.approx(2.1, abs=0.01)
    assert m.params["k_s_Lpm_bar"] == pytest.approx(0.2, abs=0.01)


def test_lookup_table_does_not_extrapolate():
    rpm = np.repeat([700, 800, 900], 400)
    dp = np.tile(np.linspace(1, 3, 400), 3)
    m = pm.LookupPump().fit(rpm, dp, 2.1 * rpm / 1000 - 0.2 * dp)
    assert np.isfinite(m.predict([800], [2.0]))[0]
    assert np.isnan(m.predict([1100], [2.0]))[0]


def test_section_law_fit():
    q = np.linspace(0.3, 0.8, 50)
    law = hy.SectionLaw.fit(q, 0.2 * q ** 1.8)
    assert law.k == pytest.approx(0.2, rel=1e-6)
    assert law.n == pytest.approx(1.8, rel=1e-6)


@pytest.mark.parametrize("e, expected", [
    ([0.2, 0.1, -0.3, 0.4], "HC"),
    ([5.0, 0.2, 0.1, 0.0], "F1"),
    ([5.0, 3.0, -2.0, -1.0], "F2"),
    ([0.0, -0.5, 9.0, 0.3], "F3"),
    ([4.0, 3.0, 9.0, 2.0], "F4"),
    ([0.5, 0.8, 0.9, 6.0], "F5"),
])
def test_isolation_rules_follow_signature_table(e, expected):
    df = pd.DataFrame([e], columns=hy.RESIDUALS)
    assert hy.isolate(df).iloc[0] == expected


def _toy_rig() -> sm.RigModel:
    return sm.RigModel(p1=0.91, filt=hy.SectionLaw(0.1, 2.0), valve=hy.SectionLaw(0.2, 2.0),
                       nozzle=hy.SectionLaw(0.16, 1.4), down=hy.SectionLaw(0.32, 3.0), q0=0.9, slope=-0.43)


def test_network_model_physics():
    rig = _toy_rig()
    h = rig.solve()
    assert h["converged"]
    # mass and pressure consistency
    assert h["Qp"] == pytest.approx(h["Q"], abs=1e-6)
    assert h["P1"] > h["P2"] and h["P3"] > h["P4"] > h["P5"] > config.P_ATM_BAR
    clog = rig.solve(s_f=10)
    assert clog["P2"] < h["P2"] and clog["Q"] < h["Q"]
    leak = rig.solve(g_l=1.0)
    assert leak["Q"] < h["Q"] and leak["Qp"] > leak["Q"]
    # a leak after the valve raises the valve's apparent resistance (flow through it > Q)
    assert (leak["P3"] - leak["P4"]) / leak["Q"] ** 2 > (h["P3"] - h["P4"]) / h["Q"] ** 2


DATA_PRESENT = (config.RIG_DIR / config.HEALTHY_FOLDER).exists()


@pytest.mark.skipif(not DATA_PRESENT, reason="raw rig data not available")
def test_real_catalogue():
    from fueltwin import rig
    cat = rig.catalogue()
    assert len(cat) == 47
    assert cat["condition"].value_counts().to_dict() == {"F2": 9, "F1": 8, "F3": 8, "F4": 8, "F5": 8, "HC": 6}
    assert (cat["n_samples"] == 60000).all()
    # recordings run in clock order from 09:58 to 16:11
    assert cat["start"].is_monotonic_increasing


@pytest.mark.skipif(not DATA_PRESENT, reason="raw rig data not available")
def test_real_pump_files():
    df = pump.load_all()
    summary = pump.test_summary(df)
    assert len(summary) == 22
    assert summary["valves_match"].all()
    final = summary[summary["campaign"] == "final"]
    assert (final["dP_max"] < config.RELIEF_VALVE_SETTING_BAR).all()
