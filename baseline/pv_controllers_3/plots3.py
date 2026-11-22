import os, sys, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ORIG = '--orig-start' in sys.argv
OUT = 'rerun_orig_start' if ORIG else 'rerun_start10V'
traces = pickle.load(open(OUT + "/traces.pkl", "rb")); df = pickle.load(open(OUT + "/df.pkl", "rb")); t1 = pd.read_csv(OUT + "/table1_controller_comparison.csv").set_index("Controller")
N = ["ACMC", "ASMC", "NN-MPC"]; col = dict(zip(N, ["tab:blue", "tab:orange", "tab:purple"]))
start = "original start, PV at 37 V as in the PPT" if ORIG else "PV starts at 10 V = Vref[0]"
fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for n in N:
    r = traces[n]; ax[0].plot(r[:, 0], r[:, 2], color=col[n], label=n, lw=1.5); ax[1].plot(r[:, 0], r[:, 4], color=col[n], label=n, lw=1.5); ax[2].plot(r[:, 0], r[:, 5], color=col[n], label=n, lw=1.5)
ax[0].plot(r[:, 0], r[:, 1], "k--", label="MPPT Vref"); ax[1].axhline(65, color="k", ls="--", label="65 V reference")
for a, l in zip(ax, ("PV Voltage (V)", "Bus Voltage (V)", "PV Power (W)")): a.set_ylabel(l); a.grid(alpha=.25); a.legend(fontsize=9, ncol=2)
ax[2].set_xlabel("Time (s)"); ax[0].set_title(f"Tuned ACMC vs tuned ASMC vs NN-MPC — same plant, load and fault schedule ({start})")
fig.tight_layout(); fig.savefig(OUT + "/fig1_three_controller_comparison.png", dpi=170); plt.close(fig)
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for n in N:
    r = traces[n]; ax[0].plot(r[:, 0], r[:, 2], color=col[n], label=n); ax[1].plot(r[:, 0], r[:, 2] - r[:, 1], color=col[n], label=n)
ax[0].plot(r[:, 0], r[:, 1], "k--", label="MPPT Vref"); ax[1].axhline(0, color="k", lw=.6)
ax[0].set_ylabel("PV Voltage (V)"); ax[1].set_ylabel("Tracking error Vpv − Vref (V)"); ax[1].set_xlabel("Time (s)")
for a in ax: a.grid(alpha=.25); a.legend(fontsize=9, ncol=2)
ax[0].set_title(f"Reference tracking and error ({start})"); fig.tight_layout(); fig.savefig(OUT + "/fig2_tracking_error.png", dpi=170); plt.close(fig)
rnd = df[df.Scenario == "24 random reference profiles"]; worst = rnd.groupby("Controller").PV_RMSE_V.max().reindex(N)
panels = [("PV voltage RMSE, full run (V)", t1.PV_RMSE_V.reindex(N)), ("PV voltage RMSE after 20 ms (V)", t1.PV_RMSE_after20ms_V.reindex(N)),
          ("Worst-case PV RMSE, 24 random profiles (V)", worst), ("DC bus RMSE vs 65 V (V)", t1.Bus_RMSE_V.reindex(N)),
          ("Mean PV power (W)", t1.Mean_PV_Power_W.reindex(N)), ("Controller compute, µs per 20 µs step (log)", t1.Controller_only_us_per_sample.reindex(N))]
fig, ax = plt.subplots(2, 3, figsize=(14, 8))
for a, (ttl, s) in zip(ax.ravel(), panels):
    a.bar(range(3), s.values, color=[col[n] for n in N]); a.set_title(ttl, fontsize=10); a.set_xticks(range(3)); a.set_xticklabels(N); a.grid(alpha=.25, axis="y")
    if "log" in ttl: a.set_yscale("log")
fig.tight_layout(); fig.savefig(OUT + "/fig3_metric_dashboard.png", dpi=170); plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 4.8)); ax.boxplot([rnd[rnd.Controller == n].PV_RMSE_V for n in N], tick_labels=N); ax.set_ylabel("PV voltage tracking RMSE (V)"); ax.grid(alpha=.25)
ax.set_title("Consistency over 24 unseen random reference profiles"); fig.tight_layout(); fig.savefig(OUT + "/fig4_consistency_boxplot.png", dpi=170)
