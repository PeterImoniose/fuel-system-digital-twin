"""Interactive digital shadow of the fuel-system testbed.

Two views:
* Replay: step through any recorded 60 s run window by window and see what the
  shadow reports (health indicators, diagnosis, severity, quality indicator).
* What-if: set fault severities on the calibrated network model and see the
  sensor readings it predicts and what the shadow would conclude.

Run from the repository root:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    st = import_module("streamlit")
except ModuleNotFoundError as exc:
    if exc.name != "streamlit":
        raise
    raise RuntimeError("Streamlit is required to run this app. Install it with `python -m pip install streamlit`.") from exc

# Let the app run without `pip install -e .`: the package lives in <repo>/src.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fueltwin import config, features, hydraulics as hy, plotting as pl, simulate as sm, twin

st.set_page_config(page_title="Fuel rig digital shadow", layout="wide")
pl.use_style()

STATUS_TEXT = {"HC": "Healthy", **{k: f"{k} {v.name}" for k, v in config.FAULTS.items()}, "unknown": "Unknown pattern"}
QUALITY_ICON = {"green": "OK", "yellow": "CAUTION", "red": "DO NOT TRUST"}


@st.cache_data(show_spinner="Loading windows...")
def load_windows() -> pd.DataFrame:
    return features.load_windows(1.0)


@st.cache_resource(show_spinner="Calibrating the shadow...")
def build(held_out: str | None):
    w = load_windows()
    return twin.DigitalShadow(hy.ResidualModel.fit(w, exclude=held_out)), sm.RigModel.from_windows(w, exclude=held_out)


w = load_windows()
st.title("Fuel-system rig: digital shadow")
st.caption("2014 UAV fuel-system demonstrator, 400 rpm. Physics residuals, rule-based diagnosis and a "
           "quality indicator; laws are calibrated without any data from the fault being shown.")
mode = st.sidebar.radio("View", ["Replay a recording", "What-if simulator"])


def residual_bars(e: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(5, 2.6))
    names = ["filter", "pump (lost delivery)", "valve", "nozzle"]
    vals = e[[c.replace("r_", "e_") for c in hy.RESIDUALS]].to_numpy(float)
    colors = [pl.STATUS["critical"] if v > 1 else pl.SERIES[0] for v in vals]
    ax.barh(names, vals, color=colors, height=0.55)
    ax.axvline(1, color=pl.INK_2, lw=1, ls=(0, (4, 2)))
    ax.axvline(-1, color=pl.INK_2, lw=1, ls=(0, (4, 2)))
    ax.set_xscale("symlog", linthresh=1)
    ax.invert_yaxis()
    ax.set_xlabel("residual / limit (beyond +/-1 = outside healthy)")
    ax.set_title(title, fontsize=9.5)
    return fig


def status_block(row: pd.Series, alarm: str | None = None):
    lines = [f"**Diagnosis (this window):** {STATUS_TEXT.get(row['rules'], row['rules'])}"]
    if alarm is not None:
        lines.append(f"**Alarm (5-window persistence):** {STATUS_TEXT.get(alarm, alarm)}")
    sev = row.get("severity")
    if row["rules"] in ("F1", "F3", "F5") and pd.notna(sev):
        lines.append(f"**Severity:** {sev:.2f} x healthy resistance")
    elif row["rules"] in ("F2", "F4") and pd.notna(sev):
        lines.append(f"**Severity:** {100 * sev:.0f} % of pump flow lost before the meter")
    st.markdown("  \n".join(lines))
    q = row["quality"]
    msg = f"Quality: **{q.upper()}** ({QUALITY_ICON[q]})" + (f": {row['quality_reasons']}" if row["quality_reasons"] else "")
    {"green": st.success, "yellow": st.warning, "red": st.error}[q](msg)


if mode == "Replay a recording":
    runs = w.groupby("run_id", sort=False)[["condition", "severity"]].first()
    labels = {r: f"{r}  ({STATUS_TEXT[c.condition]}, {c.severity:g} % valve travel)" if c.condition != "HC"
              else f"{r}  (Healthy)" for r, c in runs.iterrows()}
    run_id = st.sidebar.selectbox("Recording", list(labels), format_func=labels.get, index=list(labels).index("F5-010"))
    cond = runs.loc[run_id, "condition"]
    shadow, _ = build(cond if cond != "HC" else None)
    g = w[w["run_id"] == run_id]
    shadow.reset()
    out = shadow.stream(g)
    k = st.sidebar.slider("Window (s)", 0, len(g) - 1, len(g) - 1)
    row = out.iloc[k]
    st.subheader(f"{labels[run_id]} - window {k} s")
    status_block(row, out["alarm"].iloc[k])
    left, right = st.columns(2)
    left.pyplot(residual_bars(row, "Health indicators"), clear_figure=True)
    fig, ax = plt.subplots(figsize=(5, 2.6))
    for c, name in zip(["e_filter", "e_pump", "e_valve", "e_nozzle"], ["filter", "pump", "valve", "nozzle"]):
        ax.plot(out[c].to_numpy(), lw=1.4, label=name)
    ax.axhline(1, color=pl.INK_2, lw=1, ls=(0, (4, 2)))
    ax.axvline(k, color=pl.MUTED, lw=1)
    ax.set_xlabel("window (s)")
    ax.set_ylabel("residual / limit")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.set_title("Through the recording", fontsize=9.5)
    right.pyplot(fig, clear_figure=True)
    st.markdown("**Measurements in this window**")
    show = g.iloc[[k]][["P1", "P2", "P3", "P4", "P5", "Q", "rpm", "dP_pump"]].rename(index=lambda _: run_id)
    try:
        st.dataframe(show.style.format("{:.4f}"), width="stretch")  # Streamlit >= 1.45
    except TypeError:
        st.dataframe(show.style.format("{:.4f}"), use_container_width=True)  # older Streamlit
    st.caption("P1-P5 in bar absolute (P3-P5 converted from gauge with the standard atmosphere), Q in L/min.")

else:
    st.sidebar.markdown("**Fault settings** (physical, not valve travel)")
    s_f = st.sidebar.slider("Filter resistance (x clean)", 1.0, 30.0, 1.0, 0.5)
    g_b = st.sidebar.slider("Pump bypass conductance (F2)", 0.0, 2.5, 0.0, 0.05)
    s_v = st.sidebar.slider("Shut-off valve resistance (x healthy)", 1.0, 15.0, 1.0, 0.5)
    g_l = st.sidebar.slider("Leak conductance (F4)", 0.0, 2.5, 0.0, 0.05)
    s_n = st.sidebar.slider("Nozzle resistance (x healthy)", 1.0, 12.0, 1.0, 0.5)
    drift = st.sidebar.slider("Pump delivery drift (%)", -5.0, 5.0, 0.0, 0.5) / 100
    shadow, rig = build(None)
    sim = rig.solve(s_f=s_f, g_b=g_b, s_v=s_v, g_l=g_l, s_n=s_n, drift=drift)
    healthy = rig.solve()
    if not sim["converged"]:
        st.error("No physical operating point for this combination (for example, suction pressure would fall below "
                 "vacuum). Reduce the filter resistance.")
        st.stop()
    win = pd.DataFrame([{**{c: sim[c] for c in ["P1", "P2", "P3", "P4", "P5", "Q"]}, "rpm": 400.0}])
    win["P5g"] = max(win.loc[0, "P5"] - config.P_ATM_BAR, 0.0)
    win["P5"] = win["P5g"] + config.P_ATM_BAR
    win["P5_floor_frac"] = float(win.loc[0, "P5g"] <= 1e-6)
    win["dP_filter"], win["dP_pump"] = win["P1"] - win["P2"], win["P3"] - win["P2"]
    win["dP_valve"], win["dP_nozzle"] = win["P3"] - win["P4"], win["P4"] - win["P5"]
    out = shadow.assess(win).iloc[0]
    st.subheader("What the shadow would report")
    status_block(out)
    c1, c2, c3 = st.columns(3)
    c1.metric("Delivered flow (L/min)", f"{sim['Q']:.3f}", f"{100 * (sim['Q'] / healthy['Q'] - 1):+.1f} %")
    c2.metric("Pump output (L/min)", f"{sim['Qp']:.3f}", f"{100 * (sim['Qp'] / healthy['Qp'] - 1):+.1f} %")
    c3.metric("Pump suction P2 (bar abs)", f"{sim['P2']:.3f}", f"{sim['P2'] - healthy['P2']:+.3f}")
    left, right = st.columns(2)
    left.pyplot(residual_bars(out, "Health indicators"), clear_figure=True)
    fig, ax = plt.subplots(figsize=(5, 2.6))
    x = np.arange(5)
    ax.plot(x, [healthy[c] for c in ["P1", "P2", "P3", "P4", "P5"]], marker="o", color=pl.CONDITION_COLORS["HC"], label="healthy")
    ax.plot(x, win[["P1", "P2", "P3", "P4", "P5"]].iloc[0].to_numpy(), marker="o", color=pl.SERIES[1], label="with faults")
    ax.set_xticks(x, ["P1", "P2", "P3", "P4", "P5"])
    ax.set_ylabel("bar abs")
    ax.legend(fontsize=8)
    ax.set_title("Pressure along the line", fontsize=9.5)
    right.pyplot(fig, clear_figure=True)
    st.caption("Steady-state network model calibrated on the 2014 data. Several faults can be combined, which the "
               "recorded data never did; the rules were designed for single faults, so combinations may come out "
               "as 'unknown' or as the dominant fault.")
