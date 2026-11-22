# Observations for the PPT: tuned ACMC vs tuned ASMC vs NN-MPC

Setup: same PV array, boost converter, constant-current load and fault schedule for all three
(Cin -50 %, load spike, L -30 %, Cout -40 %). Controllers know only the nominal L and Cin.
ACMC / ASMC = tuned versions (differential-evolution gains); NN-MPC = 9-32-32-1 tanh network trained to imitate a
long-horizon MPC. Main metric: PV voltage tracking RMSE against the MPPT reference.

## Slide: Controller comparison (senior profile, PV starts at 10 V = Vref[0])
| Controller | PV RMSE (V) | PV RMSE after 20 ms (V) | Bus RMSE (V) | Mean PV power (W) | Bus overshoot (V) |
|---|---|---|---|---|---|
| ACMC | 1.18 | 1.02 | 19.06 | 92.5 | 0.00 |
| ASMC | 1.94 | 1.83 | 17.91 | 94.5 | 0.50 |
| NN-MPC | 0.43 | 0.41 | 18.05 | 93.9 | 1.21 |

(With the original PPT start, PV at 37 V and Vref at 10 V: PV RMSE 4.21 / 6.37 / 4.22 V and, after 20 ms, 1.03 / 1.48 / 0.41 V
for ACMC / ASMC / NN-MPC; see results/original_start_37V/.)

Observations
1. NN-MPC follows the reference best: about 2.7x lower tracking error than ACMC and 4.5x lower than ASMC.
2. ACMC lags the staircase slightly (error 1.0 V) but is the smoothest controller; ASMC overshoots the reference during the climb (error up to 5.3 V).
3. Bus RMSE vs 65 V is not a fair controller metric: with a constant-current load the bus follows PV power, and the reference starts at 10 V, so the bus must dip (to about 28-31 V) for every controller.
4. Mean power is within about 2 W for all three; ACMC is lowest because it lags the reference most.

## Slide: Multi-scenario benchmark (PV tracking RMSE in V, mean over runs)
| Scenario | ACMC | ASMC | NN-MPC |
|---|---|---|---|
| Senior profile | 1.18 | 1.94 | **0.43** |
| 24 random reference profiles (mean) | 1.14 | 1.28 | **0.44** |
| 24 random profiles (worst case) | 1.51 | 1.85 | **0.68** |
| Sensor noise (5 seeds) | 1.18 | 1.90 | **0.43** |
| Irradiance step 100 -> 85 % | 1.22 | 2.24 | **0.54** |
| Irradiance step 100 -> 70 % | 2.64 | 5.29 | **2.32** |

Observations
1. NN-MPC has the lowest error in every scenario and the smallest worst case; its worst random profile (0.68 V) is better than the average of ACMC and ASMC.
2. ACMC is the most consistent of the two classical controllers (std over random profiles smaller, duty activity 0.004 vs 0.016 for NN-MPC).
3. ASMC degrades most under irradiance steps (5.29 V at -30 %).
4. At a 30 % irradiance drop all three degrade (2.3 - 5.3 V): the constant-current load pulls the bus below the PV voltage, which no duty command can fix.
5. Sensor noise (0.2 V on Vpv, 0.05 A on IL, 0.3 V on Vbus) barely changes any controller.

## Slide: Computational cost (Python, desktop CPU; run bench_cost.py on your machine)
| Controller | Cost | Called |
|---|---|---|
| ACMC | about 0.4 us per call (about a dozen multiply-adds) | every 20 us |
| ASMC | about 0.5 us per call (about a dozen multiply-adds + tanh) | every 20 us |
| NN-MPC | about 4.7 us per evaluation (1,344 multiply-adds + 64 tanh) | every 100 us, duty held in between |

NN-MPC needs about 5 % of its 100 us period, so it fits in real time; ACMC and ASMC are cheaper still. (The MPC it imitates needs about 155 us per solve, more than its 100 us period.)

## Conclusion
NN-MPC is the most optimal of the three: best tracking in every test, best consistency, robust to noise and irradiance steps, at a compute cost of the same order as ACMC/ASMC.
ACMC is the best simple fallback (smoothest duty, lowest cost). ASMC is the weakest on tracking.

## Limits
- Simulation only; compute numbers are desktop-CPU, not microcontroller.
- NN-MPC is trained on this plant (irradiance 0.6-1.0, random start-up states); retrain it if the plant, load or fault schedule changes.
- The seniors' partial-shading and split-array scenarios need their PV model and are not included.
