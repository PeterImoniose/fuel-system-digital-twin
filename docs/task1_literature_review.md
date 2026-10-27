# Task 1 - The five failure modes: a literature review

**Brief:** articulate a literature review of the five failure modes affecting five different
components whose degradation can be emulated on the fuel-system testbed: clogged filter, degraded
gear pump, shut-off valve stuck mid-range, leaking pipe and clogged injector/nozzle.

Each section covers the component's job, how it fails in service, how the failure shows up in the
system's pressures and flows, how it is detected or predicted in the literature, and how faithfully a
proportional valve can emulate it. The last part of each section links the literature to what the
2014 testbed data shows (notebooks 03-06), so the review is tested against measurements rather than
left as background.

![Rig schematic](../results/figures/01_rig_schematic.png)

## The system

The testbed is a laboratory emulation of an aircraft (UAV) fuel system (Niculita, Skaf and Jennions,
2014; Lab 1 notes). A gear pump draws fuel (water on the rig) from a main tank through a suction
filter and delivers it through a shut-off valve and an injector nozzle to a sump tank. Five pressure
transducers (P1-P5), a flow meter at the end of the line and a speed sensor monitor it. Five direct
proportional valves (DPV1-DPV5) emulate degradation: three in series with a component (filter, shut-off
valve, nozzle) emulate restriction, and two in parallel (a bypass at the pump discharge and a leak
after the shut-off valve) emulate lost flow.

Fuel systems are a good case for health management because their failure modes are few, well
understood and hydraulically coupled. A restriction anywhere changes pressures everywhere, so a small
set of sensors can, in principle, both detect and locate a fault (Bardakis, 2019, ch. 4-5). The
coupling is also the difficulty: different faults move the same sensors, and the diagnostic logic in
the module's own lecture leaves the filter, pump and leak faults in one ambiguity group.

Three ways of generating fault data are compared in the literature (Bardakis, 2019, sec. 2.1):
accelerated run-to-failure testing (realistic, slow and expensive), seeded faults (machined or
damaged parts, realistic but only at discrete severities) and **emulated faults** (an actuator imitates
the hydraulic effect of the degradation). The testbed uses emulation, which makes severity continuously
adjustable and repeatable but reproduces only the hydraulic *effect* of a failure, not its mechanism or
its dynamics. That limitation recurs below.

---

## 1. Clogged suction filter (F1, DPV1 closing)

**Function.** The suction filter protects the pump and the precision components downstream (gear
clearances, valves, injectors) from particles. In fuel systems contamination is a leading cause of
wear and sticking, so the filter is deliberately sacrificial.

**How it fails.** Particles accumulate in and on the filter medium. Eker, Camci and Jennions (2016)
model the process physically: in the early **depth-filtration** phase particles are captured inside
the medium and the pressure drop rises slowly; once the pores are bridged, a **cake** forms on the
surface and the pressure drop rises much faster, often exponentially with time. Their experiments on a
filter rig show the shape depends strongly on particle size and concentration. Skaf, Eker and Jennions
(2015) use the same physics to build a simple state-based prognostic model for remaining useful life.

**System effect.** A suction filter sits on the low-pressure side of the pump, so clogging lowers the
pump inlet pressure. A positive-displacement pump keeps delivering almost the same flow until the
inlet pressure approaches the point where the fluid cannot fill the gear chambers, after which flow
falls quickly and cavitation sets in (Bardakis, 2019, sec. 5.2). The fault is therefore easy to miss if
only flow is monitored.

**Detection in the literature.** Differential pressure across the filter, normalised for flow, is the
standard indicator; prognostic work trends it against the two-phase clogging model (Eker, Camci and
Jennions, 2016; Skaf, Eker and Jennions, 2015). Bardakis (2019) proposes "suction pressure falling
while flow stays inside its healthy band" as a signature unique to this fault.

**Emulation fidelity.** A closing valve reproduces the rising pressure drop well. It does not
reproduce the time evolution (a valve setting is a step, real clogging a slow exponential) and at
extreme settings it produces effects a filter would not, such as the "line stripping" Bardakis (2019)
observed when a valve almost isolates the tank.

**What the testbed data shows.** P2 fell from 0.87 to 0.35 bar abs between 10 % and 80 % valve travel,
while delivered flow fell by only 3 % up to 40 % travel. The flow-normalised filter resistance rose
about 20-fold, roughly exponentially with valve travel, and was the first indicator to move (about
1.4x at 20 % travel). A what-if analysis with the calibrated network model shows delivery drops 5 % at
about 4x resistance, while suction pressure only reaches 0.3 bar abs near 26x, so a twin can schedule a
filter change on resistance ratio well before flow suffers.

---

## 2. Degraded gear pump (F2, DPV2 opening)

**Function.** The external gear pump sets the system's flow. Two meshing gears carry fluid around the
casing from inlet to outlet; delivery is fixed by displacement and speed, minus internal leakage (slip)
back across the gear tips and side plates. The N999R on the rig has bronze gears, carbon-graphite
bearings and a built-in relief valve set at 50 psig (Oberdorfer Pumps, datasheet).

**How it fails.** Wear of gear tips, side faces and bushings enlarges the clearances. Guo et al. (2020)
show analytically that end-face leakage rises with the cube of the gap and radial leakage with the
radial gap; in their accelerated life test volumetric efficiency fell to about 70 % as radial and end
clearances grew to about 0.1 and 0.37 mm. Abrasive contamination (linking back to the filter),
cavitation erosion when inlet pressure is too low, and running on low-viscosity fluid accelerate it.
Frosina, Senatore and Rigosi (2017) use CFD to show how leakage paths and pressure distribution
govern volumetric efficiency in an external gear pump.

**System effect.** Worn clearances mean more slip, so less flow at every pressure: the pump curve
(flow against pressure rise) moves down and becomes steeper. Because slip through a gap scales with
1/viscosity, the same pump also loses delivery as the fluid warms.

**Detection in the literature.** Flow per revolution compared with the healthy value (Roemer,
Byington and Kacprzynski, 2007, on a lubrication-system testbed; adopted by Bardakis, 2019, as a
dimensionless performance metric), pump-curve residuals, and the flow-ripple signature at the
gear-mesh frequency (Manring and Kasaragadda, 2003; Zongbin et al., 2018), which changes as teeth
wear.

**Emulation fidelity.** DPV2 bleeds pump discharge back to the tank. That reproduces **lost
delivery**, the system-level symptom of internal leakage, but the leak is external to the pump and
grows with discharge pressure through an orifice rather than through gear clearances. Bardakis (2019)
argues valve emulation cannot reproduce internal pump leakage dynamics and recommends seeded faults for
true pump wear. On this rig the bypass valve also passes nothing below about 30 % travel.

**What the testbed data shows.** With the bypass open, up to 57 % of the pump's output never reached
the flow meter, and the operating point fell *below* the pump curve (lower pressure and lower flow),
unlike any blockage. The data also contains a real, unplanned example of the mechanism: the 2014 pump
delivered 1-3.5 % less flow in the afternoon than in the morning at the same settings, and the
2021/2023 pump tests show slip varying by about 30 % between test days with unchanged displacement. Both
are consistent with temperature-driven slip, which is exactly what makes early pump wear hard to
separate from normal operation without a temperature measurement.

---

## 3. Shut-off valve stuck mid-range (F3, DPV3 closing)

**Function.** The shut-off valve isolates the engine feed. In normal operation it is fully open and
contributes little pressure loss.

**How it fails.** Valves stick because of friction and contamination: deposits, corrosion or
particles in the guide raise static friction until the actuator cannot move the stem, leaving the
valve part-open ("stuck mid-range"). Choudhury, Thornhill and Shah (2005) model valve stiction as
deadband plus stick-slip and show how it distorts the behaviour of the loop around the valve. Liu
et al. (2015) model degradation of an electro-hydraulic valve under contamination as wear and
clearance growth with time. Daigle, Kulkarni and Gorospe (2014) apply model-based prognostics to
pneumatic valves on a NASA testbed, estimating friction and leakage parameters from the valve's
response.

**System effect.** A valve stuck part-open is a fixed restriction downstream of the pump: discharge
pressure rises, flow falls, and the system's operating point slides **along** the pump curve (Bardakis,
2019, sec. 5.4). Pressure after the valve falls.

**Detection in the literature.** Pressure drop across the valve normalised by flow (a resistance
estimate), and for actuated valves, the relation between command and response (stiction signatures).

**Emulation fidelity.** A proportional valve in series reproduces the restriction exactly; it does
not reproduce stick-slip dynamics (the test valve moves smoothly to its set position). The emulated
fault is therefore a "stuck at a fixed opening" fault, which is the right steady-state model for a
valve that has seized.

**What the testbed data shows.** P3 rose and P4 fell together; the operating point followed the
pump curve as the literature predicts; the valve-section resistance rose about 11-fold by 80 % travel
and was detected from 20 % travel (16 % resistance increase).

---

## 4. Leaking pipe (F4, DPV4 opening)

**Function.** The pipework carries fuel between components. Leaks are a safety hazard in a fuel
system (fire), not only a performance loss (Bardakis, 2019).

**How it fails.** Fatigue cracking from vibration and pressure pulsation (the gear pump's ripple is a
continuous cyclic load), corrosion, loosened fittings and damaged seals.

**System effect.** A leak diverts flow before it reaches the consumer. Pressure downstream of the leak
falls, the pump sees a lower load, and the operating point moves below the pump curve, like the pump
fault. The leak's position matters: flow that leaks after a component still passed through it, so that
component's pressure drop *rises* even though delivered flow falls.

**Detection in the literature.** Mass balance (flow in minus flow out), pressure-profile and
negative-pressure-wave methods, and transient-based methods that read leak reflections in pressure
waves; Colombo, Lee and Karney (2009) review the transient methods and note how little field
validation they had. Bardakis (2019) uses the sudden pressure and flow dip at leak onset, and the
degree of recovery, to separate leaks at different positions, and notes that this needs a sampling
rate fast enough to catch the transient.

**Emulation fidelity.** A proportional valve to the tank is a good emulation of a leak orifice and,
unlike real cracks, its size can be set and reversed exactly. Real leaks may start intermittently and
grow; the emulation starts at a fixed size.

**What the testbed data shows.** The leak valve passes no flow at 20-30 % travel, so those two
recordings are physically healthy despite their label. From 40 % travel the leak is clear, and its
position is visible: the shut-off valve section's apparent resistance rose (leak flow still passes
through it), while for the pump bypass it fell. That one difference resolves the lecture tree's
pump/leak ambiguity with the existing sensors. Only one flow meter exists and it sits at the end of the
line, so a direct mass balance is not possible; a second flow meter at the pump outlet would make leak
detection and location much simpler.

---

## 5. Clogged injector / nozzle (F5, DPV5 closing)

**Function.** The injector atomises fuel into the combustor or cylinder. Its small orifices make it
the most restrictive point in the line by design.

**How it fails.** Deposits build up in and around the nozzle holes. Birgel et al. (2008) review deposit
formation in diesel injector nozzles: deposits form through thermal condensation and cracking of fuel
near 300 C at the nozzle tip, are influenced by the nozzle hole design (through cavitation), and are
promoted by trace metals such as zinc and by some biodiesel blends. Partial blockage reduces flow and
distorts the spray, degrading combustion long before the flow loss is large.

**System effect.** A downstream restriction, like the stuck valve: pump pressure rises and flow falls
along the pump curve. Unlike the valve, the pressure *upstream* of the nozzle (P4) rises.

**Detection in the literature.** Injection-pressure and flow residuals, and in engines, combustion
indicators (exhaust temperature spread between cylinders); on the testbed, the nozzle pressure drop
normalised by flow.

**Emulation fidelity.** Good for the steady hydraulic effect; spray quality, which is what actually
harms combustion, is outside the emulation.

**What the testbed data shows.** P3 and P4 both rose and the nozzle resistance rose about 8.5-fold by
80 % travel. The valve and nozzle faults are separated only by P4, confirming Bardakis's (2019)
observation that the two downstream blockages are indistinguishable with pump-side sensors alone and
need the pressure between them. Because the clogged-injector series was recorded last in the day, its
mildest settings are masked by the afternoon drift in pump delivery (notebook 03).

---

## Summary: signatures and how separable they are

| Fault | Mechanism (literature) | Main system effect | Indicator that isolates it (this work) | Emulation limits |
|---|---|---|---|---|
| F1 filter | depth then cake filtration, exponential late growth | suction pressure falls; flow held until late | filter resistance up, pump on its curve | no time evolution; line stripping at extreme settings |
| F2 pump | clearance wear, slip up (cube of gap) | flow lost at the pump; curve moves down | lost delivery without a valve-section rise | external bleed, not internal slip; dead band below 30 % |
| F3 valve | stiction, deposits, corrosion | restriction after pump; along pump curve | valve-section resistance up, P4 down | no stick-slip dynamics |
| F4 leak | fatigue, corrosion, fittings | flow lost after the valve | lost delivery *and* valve-section rise | fixed orifice; dead band below 35 % |
| F5 nozzle | thermal deposits in holes | restriction at the end; along pump curve | nozzle resistance up, P4 up | no spray-quality effect |

Two themes from the literature are confirmed by the data and shape the twin design (Task 5):

1. **Physics locates faults better than raw thresholds.** Every fault's signature follows from where
   it sits in the hydraulic circuit and whether it restricts or diverts flow (Fox, McDonald and
   Pritchard, 2012). Encoding that structure (section pressure-drop laws and a pump curve) separates all
   five faults, including the filter/pump/leak group the lecture tree cannot split.
2. **Normal variation competes with incipient faults.** Viscosity and temperature, tank level and
   ambient pressure shift the healthy operating point by about as much as the mildest emulated faults.
   Prognostic studies on filters and pumps (Eker, Camci and Jennions, 2016; Guo et al., 2020) control
   or measure these conditions; the 2014 testbed did not record them, which sets the practical
   detection limit in this dataset.

## References

Bardakis, I. (2019) *A System-Level, Performance-Oriented Investigation of Degradation*. MSc
dissertation, Glasgow Caledonian University.

Birgel, A., Ladommatos, N., Aleiferis, P., Zülch, S., Milovanovic, N., Lafon, V., Orlovic, A.,
Lacey, P. and Richards, P. (2008) 'Deposit formation in the holes of diesel injector nozzles: a
critical review', SAE Technical Paper 2008-01-2383. doi:10.4271/2008-01-2383.

Choudhury, M.A.A.S., Thornhill, N.F. and Shah, S.L. (2005) 'Modelling valve stiction', *Control
Engineering Practice*, 13(5), pp. 641-658.

Colombo, A.F., Lee, P. and Karney, B.W. (2009) 'A selective literature review of transient-based leak
detection methods', *Journal of Hydro-environment Research*, 2(4), pp. 212-227.
doi:10.1016/j.jher.2009.02.003.

Daigle, M., Kulkarni, C. and Gorospe, G. (2014) 'Application of model-based prognostics to a
pneumatic valves testbed', *2014 IEEE Aerospace Conference*, pp. 1-8. doi:10.1109/AERO.2014.6836319.

Eker, O.F., Camci, F. and Jennions, I.K. (2016) 'Physics-based prognostic modelling of filter clogging
phenomena', *Mechanical Systems and Signal Processing*, 75, pp. 395-412.

Fox, R., McDonald, A. and Pritchard, P. (2012) *Fluid Mechanics*, SI version, 8th international edn.
Hoboken, NJ: Wiley.

Frosina, E., Senatore, A. and Rigosi, M. (2017) 'Study of a high-pressure external gear pump with a
computational fluid dynamic modeling approach', *Energies*, 10(8), 1113. doi:10.3390/en10081113.

Guo, R., Li, Y., Shi, Y., Li, H., Zhao, J. and Gao, D. (2020) 'Research on identification method of
wear degradation of external gear pump based on flow field analysis', *Sensors*, 20(14), 4058.
doi:10.3390/s20144058.

Liu, Y., Ma, B., Zheng, C. and Zhang, S. (2015) 'Degradation modelling and experiment of
electro-hydraulic shift valve in contamination circumstances', *Advances in Mechanical Engineering*,
7(5), pp. 1-9. doi:10.1177/1687814015585925.

Manring, N. and Kasaragadda, S. (2003) 'The theoretical flow ripple of an external gear pump',
*Journal of Dynamic Systems, Measurement, and Control*, 125(3). doi:10.1115/1.1592193.

Niculita, O., Skaf, Z. and Jennions, I. (2014) 'The application of Bayesian change point detection in
UAV fuel systems', *Procedia CIRP*, 22, pp. 115-121. doi:10.1016/j.procir.2014.07.119.

Oberdorfer Pumps (n.d.) *Gear Pumps N999 Series: Bronze Close Coupled Rotary Gear Pumps*
(specification sheet).

Roemer, M., Byington, C. and Kacprzynski, G. (2007) *Prognosis Algorithm Design and Examples*.
Technical workshop, PHM/CBM Workshop and User's Forum, Florida, 13-15 November 2007.

Skaf, Z., Eker, O. and Jennions, I. (2015) 'A simple state-based prognostic model for filter
clogging', *Procedia CIRP*, 38, pp. 177-182. doi:10.1016/j.procir.2015.08.094.

Zongbin, C., Zhiqiang, L., Rongwu, X. and Jian, L. (2018) 'Simulation and test of gear pump flow
pulsation', *International Journal of Fluid Machinery and Systems*, 11(3), pp. 265-272.
doi:10.5293/IJFMS.2018.11.3.265.
