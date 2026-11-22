"""Tuned ACMC vs tuned ASMC vs NN-MPC on the common plant.
    python final3.py                # PV starts at 10 V = Vref[0] and follows the path   -> rerun_start10V/
    python final3.py --orig-start   # PV starts at 37 V as in the original PPT           -> rerun_orig_start/
"""
import os, sys, time, pickle
import numpy as np, pandas as pd
ORIG = '--orig-start' in sys.argv
OUT = 'rerun_orig_start' if ORIG else 'rerun_start10V'
os.makedirs(OUT, exist_ok=True)
import compare_orig as O, tuned_acmc as TACMC, tuned_asmc as TASMC, nn_mpc as NN
import core as C
class NNc:
    def __init__(s): s.m = NN.NNMPC("nnmpc_weights_H32.npz")
    def duty(s, *a): return s.m.duty(*a)
NAMES = ["ACMC", "ASMC", "NN-MPC"]
D = {"ACMC": TACMC.ACMC, "ASMC": TASMC.ASMC, "NN-MPC": NNc}

def sim(ctrl, t_ref, vref_ref, vp0, noise=0.0, gstep=None, seed=0):
    dt = O.SIM_DT; T_end = float(t_ref[-1]); t = np.arange(0, T_end + dt*.5, dt); vref = np.interp(t, t_ref, vref_ref)
    rng = np.random.default_rng(seed); vp = vp0; il = O.pv_current(vp); vo = O.VBUS_REF; rows = []; tc = 0.0; t_all = time.perf_counter()
    for k, tk in enumerate(t):
        ref = float(vref[k]); Lr, Cir, Cor, sp = O.fault_schedule(tk, T_end)
        L, Cin, Cout = O.L0*Lr, O.CIN0*Cir, O.COUT0*Cor; load = O.VBUS_REF/O.RLOAD + sp
        G = 1.0 if (gstep is None or tk < gstep[0]) else gstep[1]
        ipv = O.pv_current(vp, G); mvp, mil, mvo = vp, il, vo
        if noise > 0: mvp += rng.normal(0, noise*0.4); mil += rng.normal(0, noise*0.1); mvo += rng.normal(0, noise*0.6)
        t0 = time.perf_counter(); d = ctrl.duty(ref, mvp, mvo, mil, dt, ipv); tc += time.perf_counter() - t0
        il += dt*(vp - (1-d)*vo)/L; vp += dt*(ipv - il)/Cin; vo += dt*((1-d)*il - load)/Cout
        il = float(np.clip(il, 0, O.IMAX)); vp = float(np.clip(vp, 0, O.PV_VOC)); vo = float(np.clip(vo, 20, 90))
        rows.append((tk, ref, vp, il, vo, ipv*vp, d))
    return np.array(rows), tc, time.perf_counter() - t_all

def met(r):
    e = r[:, 2] - r[:, 1]; w = 50; lp = np.convolve(e, np.ones(w)/w, 'same'); k = int(20e-3/O.SIM_DT)
    return dict(PV_RMSE_V=np.sqrt(np.mean(e**2)), PV_RMSE_after20ms_V=np.sqrt(np.mean(e[k:]**2)), MaxErr_after20ms_V=np.abs(e[k:]).max(),
                Ripple_V=np.sqrt(np.mean((e - lp)[w:-w]**2)), Bus_RMSE_V=np.sqrt(np.mean((r[:, 4] - 65)**2)), Bus_std_V=r[:, 4].std(),
                Mean_PV_Power_W=r[:, 5].mean(), Bus_overshoot_V=max(0.0, r[:, 4].max() - 65), Duty_activity=np.mean(np.abs(np.diff(r[::5, 6]))))

tr, vr = O.load_profile()
v0 = min(45., float(np.median(vr))) if ORIG else float(vr[0])
rows = []; traces = {}
for n in NAMES:
    r, tc, tt = sim(D[n](), tr, vr, v0); m = met(r); steps = len(r)
    m.update(Controller=n, Runtime_s=tt, Runtime_us_per_sample=tt/steps*1e6, Controller_only_us_per_sample=tc/steps*1e6); rows.append(m); traces[n] = r
t1 = pd.DataFrame(rows)[["Controller", "PV_RMSE_V", "PV_RMSE_after20ms_V", "Bus_RMSE_V", "Bus_std_V", "Mean_PV_Power_W", "Bus_overshoot_V", "Runtime_s", "Runtime_us_per_sample", "Controller_only_us_per_sample"]]
t1.to_csv(OUT + "/table1_controller_comparison.csv", index=False); print(t1.round(3).to_string(index=False), flush=True)
pickle.dump(traces, open(OUT + "/traces.pkl", "wb"))
# ---- multi-scenario benchmark (all start at Vref[0])
sc = [("senior", "Senior profile (start at Vref[0])", tr, vr, {})]
for s in range(2000, 2024):
    v = C.random_scenario(s)['vref'][::50][:80]; sc.append((f"rand{s}", "24 random reference profiles", np.arange(80)*1e-3, v, {}))
for s in range(5): sc.append((f"noise{s}", "Sensor noise (5 seeds)", tr, vr, dict(noise=0.5, seed=s)))
sc.append(("g85", "Irradiance step 100->85%", tr, vr, dict(gstep=(0.040, 0.85))))
sc.append(("g70", "Irradiance step 100->70%", tr, vr, dict(gstep=(0.040, 0.70))))
rows = []
for pn, grp, t_, v_, kw in sc:
    for n in NAMES:
        r, tc, tt = sim(D[n](), t_, v_, float(v_[0]), **kw); m = met(r); m.update(Scenario=grp, Controller=n, Profile=pn); rows.append(m)
df = pd.DataFrame(rows); df.to_csv(OUT + "/table2_all_runs.csv", index=False)
order = ["Senior profile (start at Vref[0])", "24 random reference profiles", "Sensor noise (5 seeds)", "Irradiance step 100->85%", "Irradiance step 100->70%"]
g = df.groupby(["Scenario", "Controller"]).agg(PV_RMSE_V=("PV_RMSE_V", "mean"), PV_RMSE_worst_V=("PV_RMSE_V", "max"), PV_RMSE_after20ms_V=("PV_RMSE_after20ms_V", "mean"),
    MaxErr_after20ms_V=("MaxErr_after20ms_V", "mean"), Ripple_V=("Ripple_V", "mean"), Mean_PV_Power_W=("Mean_PV_Power_W", "mean"), Duty_activity=("Duty_activity", "mean")).reset_index()
g["o"] = g.Scenario.map({k: i for i, k in enumerate(order)}); g["c"] = g.Controller.map({k: i for i, k in enumerate(NAMES)})
g = g.sort_values(["o", "c"]).drop(columns=["o", "c"]); g.to_csv(OUT + "/table2_multi_scenario_benchmark.csv", index=False); print(g.round(3).to_string(index=False), flush=True)
pickle.dump(df, open(OUT + "/df.pkl", "wb"))
