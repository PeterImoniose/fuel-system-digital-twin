# %% [markdown]
# # 01 - Data audit
#
# Before any analysis: what exactly is in the files, how it was recorded, and what can go wrong
# when reading it. Everything found here feeds later notebooks, and the data-quality findings
# become requirements for the digital twin (notebook 06).
#
# **Two data sources**
#
# | Source | Used in | Rig | Logging |
# |---|---|---|---|
# | 47 recordings dated 30 May 2014: 6 healthy + 41 faulty | notebooks 02-04, 06-08 | UAV fuel-system demonstrator | NI DAQ, 1 kHz, 60 s per file |
# | 22 pump tests: 16 "initial" (2021-07-28) and 6 "final" (2023-02-02) | notebooks 05, 07 | later PLC-based version of the testbed | PLC, 200 ms |
#
# Raw files are read in place from the data folder; nothing is copied into the repository.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fueltwin import config, features, plotting as pl, pump, rig

pl.use_style()
pd.set_option("display.precision", 4)
pd.set_option("display.width", 160)

# %% [markdown]
# ## 1. The rig and where each fault is injected
#
# Faults are *emulated*: a direct proportional valve (DPV) in series or in parallel with a component
# changes the hydraulics the way the real failure would (rig documentation). The flow meter
# sits at the **end** of the line, so it measures what reaches the sump, not what the pump produces.
# This one detail decides which faults can be told apart (notebook 03).

# %%
fig, ax = plt.subplots(figsize=(10, 2.9))
pl.schematic(ax, {"filter": pl.CONDITION_COLORS["F1"], "pump": pl.CONDITION_COLORS["F2"],
                  "bypass": pl.CONDITION_COLORS["F2"], "valve": pl.CONDITION_COLORS["F3"],
                  "leak": pl.CONDITION_COLORS["F4"], "nozzle": pl.CONDITION_COLORS["F5"]})
ax.set_title("Fuel-system testbed (2014 configuration) and the five emulated faults")
pl.save(fig, "01_rig_schematic")
plt.show()

# %%
pd.DataFrame([{"code": f.code, "fault": f.name, "component": f.component, "valve": f.valve.upper(),
               "healthy opening %": f.healthy_opening,
               "fault direction": "closing" if f.healthy_opening == 100 else "opening"}
              for f in config.FAULTS.values()])

# %% [markdown]
# ## 2. File catalogue
#
# Severity is defined here as **valve travel away from the healthy position** (%), read from the
# valve column inside each file rather than from folder names. It is a control setting, not a physical
# severity: how much a valve restricts flow is strongly non-linear in its travel.

# %%
cat = rig.catalogue()
cat.to_csv(config.TABLE_DIR / "01_rig_catalogue.csv", index=False)
summary = cat.groupby("condition", sort=False).agg(
    recordings=("file", "size"),
    severities=("severity", lambda s: ", ".join(f"{v:g}" for v in sorted(s))),
    first=("start", "min"), last=("start", "max"))
summary

# %%
checks = {
    "recordings": len(cat),
    "samples per file (all 60 000)": bool((cat["n_samples"] == 60000).all()),
    "valve settings constant within every file": True,  # catalogue() raises otherwise
}
data, _ = rig.load_all()
dt = np.diff(data.loc[data["run_id"] == "HC-1", "t"].to_numpy("float64"))
checks["sampling interval (ms)"] = f"{1000 * dt.mean():.3f} +/- {1000 * dt.std():.3f}"
checks["missing values"] = int(data.isna().sum().sum())
checks["pump speed, all files (rpm)"] = f"{data['rpm'].mean():.1f} (min {data['rpm'].min():.1f}, max {data['rpm'].max():.1f})"
pd.Series(checks, name="value").to_frame()

# %% [markdown]
# ## 3. When were the recordings made? A built-in confound
#
# The healthy baseline was recorded in the first six minutes of the day; every fault type was then
# recorded as one block, in a fixed order, through to 16:11 (with a 2.5 h break inside the pump-fault
# sweep). Anything that drifts slowly through the day (water temperature, tank level, atmospheric
# pressure) is therefore tied to the fault label. Notebook 04 shows that a classifier given **only the
# clock time** can name the fault almost perfectly, which is the clearest warning sign in this dataset.

# %%
fig, ax = plt.subplots(figsize=(10, 3.2))
for i, c in enumerate(config.CONDITIONS):
    sub = cat[cat["condition"] == c]
    start = sub["start"].dt.hour + sub["start"].dt.minute / 60 + sub["start"].dt.second / 3600
    ax.barh(np.full(len(sub), i), np.full(len(sub), 1 / 60), left=start, height=0.6,
            color=pl.CONDITION_COLORS[c], edgecolor=pl.SURFACE, linewidth=0)
    ax.scatter(start + 0.5 / 60, np.full(len(sub), i), s=14, color=pl.CONDITION_COLORS[c],
               marker=pl.CONDITION_MARKERS[c], zorder=3)
ax.set_yticks(range(len(config.CONDITIONS)),
              [f"{c}  {config.CONDITION_NAMES[c]}" for c in config.CONDITIONS])
ax.invert_yaxis()
ax.set_xlabel("Time of day on 30 May 2014 (h)")
ax.set_xlim(9.8, 16.4)
ax.set_title("Every condition was recorded in its own block of the day")
ax.grid(axis="y", visible=False)
pl.save(fig, "01_recording_timeline")
plt.show()

# %% [markdown]
# ## 4. Channels, units and reference pressures
#
# From `File format.pdf`: P1 and P2 are **absolute**, P3-P5 are **gauge** (relative to atmosphere).
# Atmospheric pressure was not logged, so P3-P5 are converted to absolute with the standard atmosphere
# (1.01325 bar). Any quantity that mixes the two groups (most importantly the pump pressure rise
# P3 - P2) inherits the day's weather error, roughly +/- 0.02 bar. Column 6 is documented as
# "ignore"; it is kept only to show below that it is not empty.

# %%
channel_map = pd.DataFrame({
    "column": range(1, 15),
    "name in file": ["X_Value", "Pre-Filter Pressure Transducer", "Post Filter Pressure Transducer",
                     "Pre Valve Pressure Transducer", "Post Valve Pressure Transducer", "Main Tank Flow Meter",
                     "Sump Tank Flow Meter", "End Pressure", "Untitled", "Untitled 1", "Untitled 2",
                     "Untitled 3", "Untitled 4", "Untitled 5"],
    "used as": rig.RAW_COLUMNS,
    "meaning (File format.pdf)": ["time (s)", "P1 before filter, bar abs", "P2 after filter, bar abs",
                                  "P3 after pump, bar gauge", "P4 after shut-off valve, bar gauge",
                                  "ignore", "Q main-line flow, L/min", "P5 after nozzle, bar gauge",
                                  "pump speed, rpm", "DPV1 opening %", "DPV2 opening %", "DPV3 opening %",
                                  "DPV4 opening %", "DPV5 opening %"],
})
channel_map

# %% [markdown]
# Note that the file's own labels for columns 6 and 7 ("Main Tank Flow Meter", "Sump Tank Flow Meter")
# do not match the documented meaning; the documentation is followed. The labels are a reminder that
# channel names in a logger configuration are not a data contract.
#
# ## 5. Sensor resolution and noise floor

# %%
hc1 = data[data["run_id"] == "HC-1"]
res = {}
for c in ["P1", "P2", "P3g", "P4g", "P5g", "Q", "rpm"]:
    v = np.unique(hc1[c].to_numpy("float64"))
    steps = np.diff(v)
    res[c] = {"distinct values in 60 s": len(v), "smallest step": steps[steps > 1e-7].min(),
              "raw sd (1 kHz)": hc1[c].std()}
pd.DataFrame(res).T

# %% [markdown]
# The quantisation steps (0.3 mbar for P1/P2, 0.13 mbar for P3-P5, 0.00013 L/min for Q) are far below
# the noise, so resolution is not a limitation. The 1 kHz noise (around 10 mbar on P1/P2) is dominated
# by gear-pump ripple and electrical noise, studied in notebook 02.
#
# ## 6. P5 hits the bottom of its range
#
# When the flow drops far enough (degraded pump >= 60 %, leak >= 70 %), the pressure after the nozzle
# falls to atmospheric and the sensor reads exactly 0.000 bar for a large share of samples. Those
# readings are censored: the true value can be at or below zero but the sensor cannot say. Anything
# computed from P5 in those runs (the nozzle pressure drop in particular) is unreliable, and the twin
# must flag it rather than trust it.

# %%
floor = (data.assign(at_floor=data["P5g"] <= 1e-6)
         .groupby("run_id", observed=True)["at_floor"].mean().reindex(cat["run_id"]))
low = data.groupby("run_id", observed=True)["P5g"].apply(lambda s: (s <= 0.001).mean()).reindex(cat["run_id"])
p5 = pd.DataFrame({"condition": cat["condition"].to_numpy(), "severity": cat["severity"].to_numpy(),
                   "share exactly 0": floor.to_numpy(), "share <= 1 mbar": low.to_numpy()}, index=cat["run_id"])
p5[p5["share <= 1 mbar"] > 0.01]

# %% [markdown]
# ## 7. The "ignore" channel is not empty
#
# Column 6 responds strongly to faults (it falls with the pump and leak faults and rises sharply with
# the clogged injector), so it is a live measurement of something. Without documentation of what it is
# it cannot be interpreted, so it is excluded from every model, as the rig documentation instructs.

# %%
ch6 = data.groupby("run_id", observed=True)["ch6_ignored"].mean().reindex(cat["run_id"])
fig, ax = plt.subplots(figsize=(8, 3))
for c in config.CONDITIONS:
    m = cat["condition"] == c
    ax.plot(cat.loc[m, "severity"], ch6[m.to_numpy()], marker=pl.CONDITION_MARKERS[c], ms=5,
            color=pl.CONDITION_COLORS[c], label=f"{c} {config.CONDITION_NAMES[c]}", lw=1.5)
ax.set_xlabel("Valve travel from healthy position (%)")
ax.set_ylabel("Column 6 mean (unknown unit)")
ax.set_title("Undocumented column 6 responds to faults (excluded from all models)")
ax.legend(ncol=3, loc="upper left")
pl.save(fig, "01_ignored_channel")
plt.show()

# %% [markdown]
# ## 8. Pump test campaigns
#
# Facts checked against the Pump Datasets Specification:
# * the time column is all zeros; time is rebuilt from the 200 ms PLC period;
# * the flow column is labelled `F1 (bar)` but holds L/min;
# * both main-line valves (DPV3 and DPV4 commands) always move together;
# * the valve command maps 20 000 = fully open to 4 000 = fully closed.

# %%
pt = pump.load_all()
const = {c: pt[c].unique().tolist() for c in ["temp_C", "visc_Pa_s", "rho", "patm"]}
print("Columns that never change in any pump file:", const)
print("Time column in the raw files is all zeros:", bool((pump.read_lvm(
    next((config.PUMP_DIR / pump.CAMPAIGN_DIRS["final"]).glob("*.lvm"))).data["X_Value"] == 0).all()))
tests = pump.test_summary(pt)
tests.to_csv(config.TABLE_DIR / "01_pump_tests.csv", index=False)
tests

# %% [markdown]
# The temperature (24 C), viscosity, density and atmospheric pressure columns are constants written by
# the logging software, not measurements. Fluid temperature was therefore **never measured** in either
# rig, even though it governs the internal leakage of a gear pump. This gap returns in notebooks 03 and
# 06.
#
# The pump-file name encodes the final valve opening (`PT_700rpm_25_` means the valves stopped at 25 %
# open); the table confirms this from the logged commands.

# %% [markdown]
# ## 9. Data-quality findings carried forward
#
# | Finding | Consequence | Where it is used |
# |---|---|---|
# | Conditions recorded in time-ordered blocks; healthy only at 09:58-10:03 | label is confounded with time of day | 03 (drift), 04 (shortcut test), 06 |
# | P3-P5 gauge, P1-P2 absolute, no atmospheric pressure logged | +/- 0.02 bar uncertainty on P3 - P2 | 02, 03, 06 (requirement: log P_atm) |
# | P5 censored at 0 bar gauge at high fault severity | nozzle residual invalid in those windows | 03, 06 (quality indicator) |
# | Column 6 live but undocumented | excluded | - |
# | No fluid temperature in either rig | pump slip drift cannot be compensated | 03, 05, 06 (requirement) |
# | Pump files: time all zeros, flow mislabelled, constant "model" columns | rebuilt / relabelled / ignored | 05 |
