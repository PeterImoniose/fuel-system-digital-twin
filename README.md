# Fuel-system digital twin: from healthy baseline to a physics-based digital shadow

Condition monitoring and fault diagnosis for a laboratory aircraft fuel system (gear pump, filter,
shut-off valve, nozzle, five emulated faults), plus a model of its Oberdorfer N999R gear pump. The
pieces are assembled into a working **digital shadow**, and its failures are turned into a design
specification for a full **digital twin**.

It is a complete solution to the five tasks of the Digital Twins coursework (MMH126833, MSc Applied
Data Science in Engineering, GCU) and a data-science project in its own right, with the emphasis on
what the data can and cannot prove.

![Fault residuals](results/figures/03_residual_heatmap.png)

## Headline findings

1. **The labels are tied to the clock.** Each fault was recorded in its own block of the day. A
   classifier given *only the time of day* names the fault in 96 % of windows with every recording held
   out, which matches or beats every sensor-based model. Accuracy on this dataset is cheap; evidence has
   to come from methods that cannot use time.
2. **Physics separates all five faults** with the existing sensors. Section pressure-drop laws and a pump
   residual are fitted *without any data from the fault being scored*. They split the filter/pump/leak
   group the module's diagnostic tree leaves ambiguous, because leak flow still passes the shut-off
   valve while bypass flow does not.
3. **A classifier trained only on simulated faults works on the real rig.** A steady-state network model
   built from the identified laws generates the training data. Tested on the 47 real recordings it scores
   macro F1 0.87 +/- 0.03 (0.93 without three settings that have no physical effect). Its healthy
   boundary still needs qualifying.
4. **The healthy system drifts.** By the afternoon the pump delivered 1-3.5 % less at the same settings,
   about the size of the mildest faults. This is most likely water warming and increasing gear slip;
   temperature was never measured. The same slip shift appears between the 2021 and 2023 pump tests,
   while displacement stays at 2.1 mL/rev.
5. **The pump matches its datasheet within about 2 %.** The initial campaign's flow collapse above
   600 rpm is the N999R's built-in 3.45 bar relief valve. A two-parameter physics pump model fits the
   final campaign to about 1 % on held-out speeds. The module's published verification errors are
   mis-computed: the true figure is a 3.6 % mean over-prediction, not under 1.52 %.
6. **The digital shadow**, replayed over the whole test day, raised no false alarms and ended on the
   right condition for 41 of 47 recordings. Its one confident-looking mistake (a mild nozzle clog that
   drift made look like a pump fault) was flagged by the quality indicator in every window.
7. **Regression done properly.** The coursework's polynomial pump model scored R^2 = 0.997 on a shuffled
   split. Tested on unseen speeds and test days:
   * degree 2 is the best compromise;
   * tree models and a neural network fail between or beyond the tested speeds;
   * a physics + Gaussian-process hybrid adds prediction intervals, but they only hold within one test
     day.

   Virtual flow meters each fail on the fault that breaks their assumption, so the twin needs several
   that vote.
8. **Live monitoring without labels.**
   * Detectors trained only on healthy data flag every afternoon recording, including settings with no
     physical fault: the healthy reference covers five minutes.
   * Change detection relative to the recent past works: every real step is caught within about a
     second, with no false alarms from the Bayesian detector.
   * On simulated gradual degradation, alarms fire at 1-3 % extra resistance or under 1 % lost flow.
   * Kalman-smoothed indicators improve the borderline recordings.

## What is where

| Task (brief) | Notebook / document | Answer in one line |
|---|---|---|
| Data audit | [01_data_audit](notebooks/01_data_audit.ipynb) | channels, units, references, sensor clipping, the recording schedule |
| 1 Literature review of the five failure modes | [docs/task1_literature_review.md](docs/task1_literature_review.md) | mechanism, system effect, detection and emulation fidelity, tested against the data |
| 2 Healthy conditions | [02_healthy_baseline](notebooks/02_healthy_baseline.ipynb) | very repeatable over minutes; pump ripple at shaft and mesh orders; P1 tracks tank level |
| 3 Faulty conditions | [03_fault_characterisation](notebooks/03_fault_characterisation.ipynb) | signatures, physics residuals, degradation paths, afternoon drift, lecture tree tested |
| Fault diagnosis (data science) | [04_fault_diagnosis_ml](notebooks/04_fault_diagnosis_ml.ipynb) | rules vs ML under three validation designs; the clock shortcut; severity |
| 4 Pump vs datasheet, pump model | [05_pump_map](notebooks/05_pump_map.ipynb) | relief valve, valve end points, four model forms, independent verification, pump map |
| 5 Shadow to twin | [06_digital_shadow](notebooks/06_digital_shadow.ipynb), [docs/task5_digital_twin_design.md](docs/task5_digital_twin_design.md) | day replay, sim-to-real, what-if, requirements and roadmap |
| Regression models | [07_regression_models](notebooks/07_regression_models.ipynb) | polynomial degree, ML and Gaussian-process pump models with intervals, virtual flow meters, severity regression |
| Live monitoring | [08_live_monitoring](notebooks/08_live_monitoring.ipynb) | healthy-only anomaly detection, CUSUM and Bayesian change points, gradual-degradation detection, Kalman tracking |
| Report | [REPORT.md](REPORT.md) | all five tasks in one write-up |
| Interactive app | [app/streamlit_app.py](app/streamlit_app.py) | replay any recording, or set fault severities and see the shadow's verdict |

![Degradation paths](results/figures/03_degradation_paths.png)

## How it works

```
src/fueltwin/
  lvm.py         LabVIEW .lvm reader (finds the header row; recording start time from the channel block)
  rig.py         2014 rig recordings: catalogue, units (gauge -> absolute), derived pressure drops
  pump.py        pump campaigns: 200 ms time base, relabelled flow, valve command -> % open
  features.py    1 s windows (means, ripple sd, sensor-floor flag)
  baseline.py    healthy envelope with within- and between-run variation
  hydraulics.py  section laws, pump line, physics residuals, isolation rules, physical severity
  pumpmodel.py   OEM curve, physics / polynomial / lookup pump models, relief-valve onset, validation
  diagnosis.py   ML and rule baselines, validation splits, lecture diagnostic tree
  simulate.py    steady-state network model of the rig for fault simulation and what-if
  twin.py        DigitalShadow: residuals + rules + second opinion + persistence + quality indicator
  regression.py  pump regressors (polynomial, trees, MLP, GP, physics + GP), virtual flow meters, severity
  monitor.py     healthy-only anomaly detectors, CUSUM, Bayesian online change points, Kalman tracking
  plotting.py    one visual system for every figure
```

Validation rules used throughout:
* section laws and pump line are fitted leave-one-fault-out;
* classifiers are scored leave-one-recording-out and train-severe/test-mild, never on shuffled windows
  alone;
* the pump model is scored leave-one-speed-out, on independent verification points, and across
  campaigns.

## Running it

```bash
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -e .
.venv\Scripts\python -m pytest
.venv\Scripts\python scripts\build_notebooks.py
.venv\Scripts\streamlit run app\streamlit_app.py
```

The raw data is not included (see [DATASET.md](DATASET.md)). Put the coursework folder `Digital twin`
next to this repository or set `FUELTWIN_DATA`. The notebooks are generated from the percent-format
sources in `notebooks/source/` and executed by `scripts/build_notebooks.py` (about 10 minutes, most of it
the leave-one-recording-out grid in notebook 04).

## Limitations

* One day of rig data at a single pump speed (400 rpm); single faults only; faults are emulated by
  valves, so mechanisms (wear rates, clogging dynamics) come from the literature, not the rig.
* Healthy recordings exist only from 09:58-10:03; healthy windows are scored by models whose
  calibration included them.
* Models are steady-state; leak transients and multiple simultaneous faults are out of scope.
* The simulation-trained classifier's healthy boundary is not yet qualified (0-53 % of healthy windows
  flagged depending on the random seed); inside the shadow it is only a second opinion.

## Earlier version

This is a fresh build. An earlier notebook on the same coursework
([Analysis-of-the-condition-of-a-fuel-system](https://github.com/PeterImoniose/Analysis-of-the-condition-of-a-fuel-system))
had the following problems, which this version fixes:
* the degraded-pump severity axis was reversed;
* its pump model was validated on its own training data and with shuffled rows;
* the model was applied at 400 rpm after fitting on 700-950 rpm;
* the app computed the pressure difference with the wrong sign;
* healthy limits were set on raw 1 kHz samples.
