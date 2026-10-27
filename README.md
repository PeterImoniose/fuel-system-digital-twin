# Fuel-System Digital Twin: fault detection and diagnosis for an aircraft fuel rig

Condition monitoring for a laboratory aircraft fuel system: a gear pump feeding a filter, a shut-off
valve and an injector nozzle, run healthy and with five faults at increasing severity. The project
builds the pieces of a digital twin (a healthy baseline, physics-based health indicators, a pump
model, fault diagnosis and live monitoring) and checks each against data it was not fitted to.

**Presentation:** [presentation/fuel_system_digital_twin.pptx](presentation/fuel_system_digital_twin.pptx),
a short deck of the problem, approach and results.

![Fault residuals](results/figures/03_residual_heatmap.png)

## The problem

Fuel systems fail gradually: filters clog, pumps wear, valves stick, pipes leak and injectors foul.
Given five pressure sensors, one flow meter and the pump speed, can we tell:

1. whether the system is healthy,
2. which component is failing, and how badly,
3. how early, and with how much confidence?

## Data

* **47 rig recordings** (60 s each at 1 kHz): 6 healthy and 41 with a fault emulated by a proportional
  valve at 10-100 % travel.
* **22 pump tests** from two campaigns (2021 and 2023), mapping flow against pressure at 400-1140 rpm.

The raw data is not included; see [DATASET.md](DATASET.md) for the source, channels and known issues.

## Key results

| Area | Result |
|---|---|
| Data quality | Every fault was recorded in its own block of the day: **a model given only the time of day names the fault 96 % of the time**. Accuracy alone proves nothing here, so every result is tested in ways this shortcut cannot help. |
| Healthy baseline | Very repeatable over minutes (flow varies 0.06 % between runs). Most of the high-frequency "noise" is the pump itself, at shaft and gear-mesh frequencies. |
| Fault isolation | Physics-based pressure-drop indicators, fitted **without any data from the fault being scored**, tell all five faults apart, including the filter/pump/leak group that the rig's expert rules leave ambiguous. |
| Sim-to-real | A classifier trained **only on faults simulated by a calibrated model of the rig** scores macro F1 0.87 +/- 0.03 on the real recordings. |
| Pump model | The pump matches its datasheet within about 2 %. A two-parameter physics model predicts unseen speeds to about 1 %. Day-to-day shifts (3-10 %) live in the pump's internal leakage, not its displacement. |
| Regression | Training R^2 kept rising with polynomial degree while accuracy on unseen speeds fell from degree 4. Random forest and gradient boosting cannot predict an unseen pump speed at all. A physics + Gaussian-process hybrid adds honest prediction intervals. |
| Live monitoring | Change detection catches every real step of degradation within about a second, with no false alarms from the Bayesian detector. Anomaly detectors trained on five minutes of healthy data flag everything that follows, a data-coverage limit rather than an algorithm one. |
| Digital shadow | Replaying the whole test day: **no false alarms**, the right condition for 41 of 47 recordings, and the one confident-looking mistake (caused by drift) flagged as low confidence. |

The limit throughout is the data, not the algorithms: the healthy reference covers only five
minutes, and fluid temperature (which drives the pump's drift) was never measured.

![Degradation paths](results/figures/03_degradation_paths.png)

## Notebooks

| Notebook | Question |
|---|---|
| [01 Data audit](notebooks/01_data_audit.ipynb) | What is in the files, and what can go wrong reading them? |
| [02 Healthy baseline](notebooks/02_healthy_baseline.ipynb) | What does normal look like, and how repeatable is it? |
| [03 Fault characterisation](notebooks/03_fault_characterisation.ipynb) | What does each fault do to the system, and why? |
| [04 Fault diagnosis](notebooks/04_fault_diagnosis_ml.ipynb) | Can the faults be told apart, and how much of a good score is real? |
| [05 Pump model](notebooks/05_pump_map.ipynb) | Does the pump meet its datasheet, and which model holds up? |
| [06 Digital shadow](notebooks/06_digital_shadow.ipynb) | Does it all work together on a live stream, and can simulation replace fault data? |
| [07 Regression models](notebooks/07_regression_models.ipynb) | Polynomial degree, ML and Gaussian-process regressors, virtual sensors, severity |
| [08 Live monitoring](notebooks/08_live_monitoring.ipynb) | Anomaly detection, change points and Kalman tracking without fault labels |

An interactive Streamlit app ([app/streamlit_app.py](app/streamlit_app.py)) replays any recording
through the shadow, or simulates fault combinations and shows what the shadow would conclude.

## Approach and validation

* **Features:** 1 s windows, with physical pressure-drop indicators for each section of the line.
* **Models:** rules on physics indicators; logistic regression, random forest, gradient boosting; a
  steady-state network model of the rig for simulation; Gaussian processes; CUSUM and Bayesian
  change-point detection; Kalman filters.
* **Validation:**
  * diagnosis is scored leaving out whole recordings, and training on severe faults while testing on
    mild ones; shuffled splits appear only as a contrast;
  * physics indicators are calibrated leaving out the whole fault being scored;
  * the pump model is scored on held-out speeds, independent verification points and a different test
    campaign.

## Repository

```
src/fueltwin/    Python package
  lvm.py, rig.py, pump.py   data loading (LabVIEW files, units, derived quantities)
  features.py, baseline.py  1 s windows and the healthy envelope
  hydraulics.py             physics indicators, isolation rules, physical severity
  pumpmodel.py              datasheet curve, pump models, relief-valve detection, validation
  diagnosis.py              ML classifiers, validation splits, expert rule baseline
  simulate.py               network model of the rig (fault simulation, what-if)
  twin.py                   the digital shadow
  regression.py, monitor.py regression models; anomaly, change-point and Kalman methods
notebooks/       executed notebooks (sources in notebooks/source/)
results/         figures, tables and the fitted pump model
presentation/    project slides
app/             Streamlit dashboard
tests/           22 unit tests
```

## Running it

```bash
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -e .
.venv\Scripts\python -m pytest
.venv\Scripts\python scripts\build_notebooks.py
.venv\Scripts\streamlit run app\streamlit_app.py
```

Put the data folder `Digital twin` next to this repository or set `FUELTWIN_DATA`. Rebuilding all
notebooks takes about 10 minutes.

## Limitations

* One day of rig data at a single pump speed (400 rpm); single faults only, emulated by valves.
* Healthy recordings cover only 09:58-10:03, so day-scale drift cannot be fully separated from faults.
* Models are steady-state; leak transients and multiple simultaneous faults are out of scope.
* The simulation-trained classifier's healthy boundary is not yet reliable (0-53 % of healthy windows
  flagged depending on the random seed), so inside the shadow it is only a second opinion.

## Earlier version

This is a full rebuild of an earlier notebook
([Analysis-of-the-condition-of-a-fuel-system](https://github.com/PeterImoniose/Analysis-of-the-condition-of-a-fuel-system)).
It fixes these problems in the earlier version:
* a reversed severity axis;
* a pump model validated on its own training data and then applied outside its speed range;
* a sign error in the app;
* healthy limits set on raw 1 kHz samples.
