# Data card

The data comes from a laboratory fuel-system testbed at Glasgow Caledonian University, shared with
its documentation (file formats, rig schematic, pump datasheet). It is **not redistributed** in this
repository. The code reads it in place from a folder called `Digital twin` next to the repository, or
from the path in the `FUELTWIN_DATA` environment variable.

```
Digital twin/
  Task 2 & 3/   rig recordings: one folder of .lvm files per condition, plus file-format notes
  Task 4/       pump test campaigns: two folders of .lvm files, plus the N999R datasheet
```

## 1. Fuel-system recordings

| Item | Value |
|---|---|
| Rig | UAV fuel-system demonstrator (laboratory testbed) |
| Date | 30 May 2014, 09:58-16:11 |
| Recordings | 47: 6 healthy, 8 clogged filter, 9 degraded pump, 8 stuck valve, 8 leaking pipe, 8 clogged injector |
| Length | 60 s each, 60 000 samples at 1 kHz |
| Pump speed | 400 rpm (397.6-405.0) |
| Format | LabVIEW Measurement (.lvm), tab separated, 23-line header |

Channels (from the rig's file-format notes; sensor positions from the rig schematic):

| Col | Used as | Meaning | Reference |
|---|---|---|---|
| 1 | t | time, s | |
| 2 | P1 | before filter, bar | absolute |
| 3 | P2 | after filter (pump suction), bar | absolute |
| 4 | P3g | after pump, bar | gauge |
| 5 | P4g | after shut-off valve, bar | gauge |
| 6 | ch6_ignored | documented "ignore" (it does respond to faults) | |
| 7 | Q | main-line flow at the end of the line, L/min | |
| 8 | P5g | after nozzle, bar | gauge |
| 9 | rpm | pump speed | |
| 10-14 | dpv1-dpv5 | valve openings, % | |

Fault emulation (valve; healthy position; direction): F1 DPV1, 100 %, closing; F2 DPV2, 0 %, opening;
F3 DPV3, 100 %, closing; F4 DPV4, 0 %, opening; F5 DPV5, 100 %, closing. Severity in this project =
valve travel from the healthy position, read from the file.

**Known issues** (notebook 01):

* Each condition was recorded in its own block of time; healthy only at 09:58-10:03. Labels are
  therefore confounded with time of day.
* P3-P5 gauge vs P1-P2 absolute; atmospheric pressure not logged (standard atmosphere assumed).
* P5 clips at 0 bar gauge for F2 >= 60 % and F4 >= 70 %.
* The file's header labels for columns 6 and 7 do not match the documentation.
* The first header block's time is the logging program's start time; the recording start is in the
  per-channel block.
* F2 at 20 % and F4 at 20-30 % produce no measurable hydraulic change (valve dead band).

## 2. Pump test campaigns

| Item | Initial campaign | Final campaign |
|---|---|---|
| Files dated | 2021-07-28 | 2023-02-02 |
| Speeds | 400-1140 rpm (16 tests) | 700-950 rpm (6 tests) |
| Valve end point | 9.9 % open for every test | 25 % (700 rpm), 35 % (others) |
| Max pump pressure | 2.6-3.86 bar (past the 3.45 bar relief setting from 550 rpm) | 2.8-3.37 bar |
| Sampling | PLC, 200 ms | PLC, 200 ms |

**Known issues:** the time column is all zeros (rebuilt from 200 ms); the flow column `F1 (bar)` is
in L/min; temperature, viscosity, density and P_atm columns are constants written by the software,
not measurements; the `F1_model` column is the rig software's own model and is not used.

## 3. Licence and use

The data belongs to Glasgow Caledonian University and is used here for a personal portfolio
analysis. It is not included in this repository.
