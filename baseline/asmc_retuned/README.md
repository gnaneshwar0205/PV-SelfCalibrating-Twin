# ASMC v2 - re-tuned adaptive sliding-mode controller

Drop-in replacement for `tuned_asmc.ASMC`:  `from tuned_asmc_v2 import ASMC`  (same `duty(ref, vp, vo, il, dt, ipv)` interface).

## What changed vs the teammate's tuned ASMC
- Same structure: sliding surface on the inductor-current error, tanh boundary layer, switching gain that adapts with |s|.
- One new term: reference-slope feed-forward `i_ref -= KFF*Cin*dVref/dt` (the slope is estimated from successive reference values; the NN-MPC receives the same information).
- All 8 parameters re-tuned by differential evolution on randomised reference profiles, irradiance steps and sensor noise (`tune_asmc.py 0 1`; two seeds give the same optimum).
  Parameters: `outputs/asmc_v2_params.json` (order K1, K2, KP_I, G0, GSLOPE, GMAX, PHI, KFF).
- Also tried: a voltage-surface ASMC with equivalent control (variant 1, worse: 0.39 V on the senior profile) and wider bounds (no gain on held-out tests).

## Results (held-out scenarios, PV starts at Vref[0] = 10 V) - `results/asmc_v2_benchmark.csv`
PV voltage tracking RMSE: senior 0.19 V (teammate ASMC 1.94, ACMC 1.18, NN-MPC 0.43); 24 unseen random profiles 0.21 V mean / 0.47 V worst
(1.28 / 1.85, 1.14 / 1.51, 0.44 / 0.68); sensor noise 0.19 V; irradiance step to 85 % 0.36 V; to 70 % 2.24 V.

## Run
```
pip install numpy pandas matplotlib scipy numba
python eval_asmc.py            # harness comparison (about 2 min) -> eval_asmc_runs.csv
python tune_asmc.py 0 1        # re-tune (about 3 min single run; stored result in outputs/)
```

## Limits
Simulation only. Duty-cycle activity is higher than the old ASMC/ACMC (0.017 vs 0.004 on the senior profile; 0.09 with sensor noise), i.e. more switching effort.
Gains sit at the upper bounds of the original search range (a wider search changed nothing on held-out tests).
ACMC was NOT re-tuned with the same feed-forward term, so the comparison favours the new ASMC.
