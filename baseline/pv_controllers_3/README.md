# PV boost converter: tuned ACMC vs tuned ASMC vs NN-MPC

Minor project: Comparative Analysis and Predictive Health Management of a Digital-Twin-Assisted DC-DC Boost Converter
(JNU School of Engineering). Python simulation of the PV + boost plant from the seniors' `02_compare_controllers.py`,
same load and fault schedule for every controller. Simulation only. MATLAB is a separate step (see below).

## The three controllers
| Name | File | Interface |
|---|---|---|
| ACMC (tuned) | `tuned_acmc.py` -> `ACMC` | `duty(ref, vp, vo, il, dt, ipv)` |
| ASMC (tuned) | `tuned_asmc.py` -> `ASMC` | same |
| NN-MPC | `nn_mpc.py` -> `NNMPC` + `nnmpc_weights_H32.npz` (keep them in the same folder) | same |

ACMC and ASMC: gains found by differential evolution. NN-MPC: a 9-32-32-1 tanh network trained (DAgger imitation) to copy a
long-horizon MPC ("MPC 2", `mpc_2.py`), so it needs about 1.3k multiply-adds per decision instead of an online optimisation.

## Result in one paragraph
NN-MPC gives the lowest tracking error in every scenario (0.43 V RMSE on the senior profile vs 1.18 V ACMC and 1.94 V ASMC),
the best worst-case over 24 unseen profiles (0.68 V vs 1.51 / 1.85 V), and stays robust to sensor noise and irradiance steps, at about 4.7 us per
decision. ACMC is the smoothest and cheapest fallback. Full tables, figures and slide text: `results/`.

## Folder map
- `tuned_acmc.py`, `tuned_asmc.py`, `nn_mpc.py`, `nnmpc_weights_H32.npz` the three controllers
- `final3.py` runs all three on the senior profile, 24 random profiles, sensor noise and irradiance steps (about 1 min)
- `plots3.py` makes the four figures; `bench_cost.py` measures compute cost
- `compare_orig.py` plant model (the seniors' script), `original/` the unchanged original, `data/mppt_vref_profile.csv` the MPPT reference
- `results/start_at_10V/` main results (PV starts at 10 V = Vref[0], then follows the path); `results/original_start_37V/` same with the PPT's start (PV at 37 V)
- `results/observations_for_ppt.md` slide-ready tables and bullet text
- Only needed to rebuild the NN-MPC: `core.py` (fast simulator), `mpc_2.py` (the MPC the network copies), `tune_mpc_2.py`, `train_nnmpc.py`, `outputs/mpc_2_params.json`

## How to run
```
pip install -r requirements.txt
python final3.py                 # PV starts at 10 V  -> rerun_start10V/
python plots3.py
python final3.py --orig-start    # PV starts at 37 V as in the PPT -> rerun_orig_start/
python plots3.py --orig-start
python bench_cost.py
```
Run from this folder (relative paths).

## Rebuilding the NN-MPC (only if the plant, load or faults change)
```
python tune_mpc_2.py      # about 8 min, writes outputs/mpc_2_params.json (update constants in mpc_2.py)
python train_nnmpc.py 32  # 1-2 min, writes outputs/nnmpc_H32.npz -> copy to nnmpc_weights_H32.npz
```

## Comparison set-up
- Plant: 4S2P PV surrogate, boost converter at 20 us step, constant-current load, faults at 40/55/70/82 % of the run.
- Controllers know only nominal L, Cin. ACMC and ASMC run every 20 us; NN-MPC evaluates every 100 us and holds the duty.
- Main metric: PV voltage tracking RMSE vs the MPPT reference. Bus RMSE vs 65 V is reported but is not a fair controller metric
  (with a constant-current load the bus follows PV power, and the reference starts at 10 V, so the bus must dip).

## Limits
Simulation only; compute numbers are desktop-CPU. The NN-MPC must be retrained if the plant changes. Partial-shading / split-array
scenarios from the seniors' benchmark are not included.
