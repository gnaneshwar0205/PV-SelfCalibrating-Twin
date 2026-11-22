import sys, json, time, numpy as np, pandas as pd
import compare_orig as O, core as C, tuned_acmc as TA, tuned_asmc as TS, nn_mpc as NN, asmc_new as AN
def sim(ctrl, t_ref, vref_ref, vp0, noise=0.0, gstep=None, seed=0):
    dt = O.SIM_DT; T_end = float(t_ref[-1]); t = np.arange(0, T_end + dt*.5, dt); vref = np.interp(t, t_ref, vref_ref)
    rng = np.random.default_rng(seed); vp = vp0; il = O.pv_current(vp); vo = O.VBUS_REF; rows = []
    for k, tk in enumerate(t):
        ref = float(vref[k]); Lr, Cir, Cor, sp = O.fault_schedule(tk, T_end)
        L, Cin, Cout = O.L0*Lr, O.CIN0*Cir, O.COUT0*Cor; load = O.VBUS_REF/O.RLOAD + sp
        G = 1.0 if (gstep is None or tk < gstep[0]) else gstep[1]
        ipv = O.pv_current(vp, G); mvp, mil, mvo = vp, il, vo
        if noise > 0: mvp += rng.normal(0, noise*0.4); mil += rng.normal(0, noise*0.1); mvo += rng.normal(0, noise*0.6)
        d = ctrl.duty(ref, mvp, mvo, mil, dt, ipv)
        il += dt*(vp-(1-d)*vo)/L; vp += dt*(ipv-il)/Cin; vo += dt*((1-d)*il-load)/Cout
        il = float(np.clip(il, 0, O.IMAX)); vp = float(np.clip(vp, 0, O.PV_VOC)); vo = float(np.clip(vo, 20, 90))
        rows.append((tk, ref, vp, il, vo, ipv*vp, d))
    return np.array(rows)
def met(r):
    e = r[:, 2]-r[:, 1]; w = 50; lp = np.convolve(e, np.ones(w)/w, 'same'); k = 1000
    return dict(RMSE=np.sqrt(np.mean(e**2)), RMSE20=np.sqrt(np.mean(e[k:]**2)), Max20=np.abs(e[k:]).max(), Ripple=np.sqrt(np.mean((e-lp)[w:-w]**2)),
                Power=r[:, 5].mean(), DutyAct=np.mean(np.abs(np.diff(r[::5, 6]))))
def build(files):
    D = {"ACMC": TA.ACMC, "ASMC (teammate)": TS.ASMC, "NN-MPC": lambda: NN.NNMPC("nnmpc_weights_H32.npz")}
    for name, (fn, cls) in files.items():
        th = json.load(open(fn))['theta']; c = type(cls.__name__ + name.replace(' ', ''), (cls,), {'TH': th}); D[name] = c
    return D
if __name__ == "__main__":
    files = {"ASMC v0 (re-tuned)": ("outputs/asmc_v2_params.json", AN.ASMC_V0), "ASMC v1 (voltage surface)": ("outputs/asmc_v1_s1.json", AN.ASMC_V1)}
    for a in sys.argv[1:]:
        files[a.split(':')[0]] = (a.split(':')[1], AN.ASMC_V0 if 'v0' in a.split(':')[1] else AN.ASMC_V1)
    D = build(files); tr, vr = O.load_profile()
    sc = [("Senior profile", "senior", tr, vr, {})]
    for s in range(2000, 2024):
        sc.append((f"r{s}", "24 random profiles", np.arange(80)*1e-3, C.random_scenario(s)['vref'][::50][:80], {}))
    for s in range(5): sc.append((f"n{s}", "Sensor noise", tr, vr, dict(noise=0.5, seed=s)))
    sc.append(("g85", "Irradiance 100->85%", tr, vr, dict(gstep=(0.040, 0.85)))); sc.append(("g70", "Irradiance 100->70%", tr, vr, dict(gstep=(0.040, 0.70))))
    rows = []
    for pn, grp, t_, v_, kw in sc:
        for n, cls in D.items():
            m = met(sim(cls(), t_, v_, float(v_[0]), **kw)); m.update(Scenario=grp, Controller=n, P=pn); rows.append(m)
    df = pd.DataFrame(rows); df.to_csv("eval_asmc_runs.csv", index=False)
    g = df.groupby(["Scenario", "Controller"]).agg(RMSE=("RMSE", "mean"), Worst=("RMSE", "max"), RMSE20=("RMSE20", "mean"), Max20=("Max20", "mean"), Ripple=("Ripple", "mean"), DutyAct=("DutyAct", "mean"), Power=("Power", "mean"))
    pd.set_option("display.width", 250); print(g.round(3).to_string())
