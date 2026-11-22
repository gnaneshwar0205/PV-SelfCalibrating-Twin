from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Controller comparison for the PV Digital Twin project
#
# IMPORTANT:
# This is a corrected research prototype, not a replacement for the senior
# converter model. All three controllers use the SAME PV plant, boost plant,
# sampling grid, load and fault schedule. Only the controller changes.
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "mppt_vref_profile.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# Senior-project nominal values
VBUS_REF = 65.0
L0 = 5e-3
CIN0 = 470e-6
COUT0 = 220e-6
DMIN, DMAX = 0.02, 0.90
IMAX = 5.0

# Two parallel strings, four 36-cell modules in series per string.
# This gives a practical nominal PV operating voltage in the ~40-50 V range.
PV_VOC = 4 * 14.4
PV_ISC = 2 * 1.45
PV_SHAPE = 8.0

# Output load chosen so that nominal MPP power can sustain a 65 V bus.
RLOAD = VBUS_REF / 1.73

# The original CSV can be relatively coarse (often milliseconds). The power
# stage has microsecond-scale electrical dynamics, so we integrate internally
# at a finer fixed step and interpolate the MPPT reference onto that grid.
SIM_DT = 20e-6


def load_profile():
    if not DATA.exists():
        raise FileNotFoundError(
            "Copy senior mppt_vref_profile.csv into the project's data/ folder."
        )

    d = pd.read_csv(DATA)
    col = next(
        (
            c
            for c in d.columns
            if c.lower().replace(" ", "_")
            in {"v_ref", "vref", "vpv_ref", "vpvref"}
        ),
        None,
    )
    if col is None:
        raise ValueError(f"Need a V_ref column. Found: {list(d.columns)}")

    v = pd.to_numeric(d[col], errors="coerce").interpolate().bfill().ffill().to_numpy()

    tc = next(
        (c for c in d.columns if c.lower().replace(" ", "_") in {"time", "time_s", "t"}),
        None,
    )
    if tc is None:
        t = np.arange(len(v), dtype=float) * 1e-3
    else:
        t = pd.to_numeric(d[tc], errors="coerce").interpolate().bfill().ffill().to_numpy()

    t = t - t[0]
    keep = np.r_[True, np.diff(t) > 0]
    t, v = t[keep], v[keep]

    if len(t) < 2:
        raise ValueError("MPPT profile must contain at least two time samples.")

    # The senior MPPT profile may contain startup/transient values. We do not
    # overwrite them; we only keep them inside a physically meaningful PV range.
    v = np.clip(v, 5.0, PV_VOC * 0.98)
    return t, v


def pv_current(vpv, irradiance_scale=1.0):
    """Smooth surrogate for the senior 4S x 2P PV array I-V curve."""
    x = np.clip(vpv / PV_VOC, 0.0, 1.0)
    return max(0.0, PV_ISC * irradiance_scale * (1.0 - x**PV_SHAPE))


class ACMC:
    """Senior-inspired cascade voltage/current controller."""

    def __init__(self):
        self.iv = 0.0
        self.ii = 0.0
        self.vf = None

    def duty(self, ref, vp, vo, il, dt, ipv):
        self.vf = vp if self.vf is None else 0.985 * self.vf + 0.015 * vp
        e_v = ref - self.vf
        self.iv = np.clip(self.iv + e_v * dt, -0.5, 0.5)

        # Outer voltage loop generates an inductor-current reference.
        i_ref = np.clip(ipv - 0.80 * e_v - 1.2 * self.iv, 0.0, IMAX)
        e_i = i_ref - il
        self.ii = np.clip(self.ii + e_i * dt, -0.5, 0.5)

        d_ff = 1.0 - vp / max(vo, 1.0)
        d = d_ff + 0.035 * e_i + 0.045 * self.ii
        return float(np.clip(d, DMIN, DMAX))


class ASMC:
    """Adaptive sliding-mode controller with a boundary layer."""

    def __init__(self):
        self.ie = 0.0
        self.gain = 0.035
        self.lam = 10.0

    def duty(self, ref, vp, vo, il, dt, ipv):
        e = ref - vp
        self.ie = np.clip(self.ie + e * dt, -0.5, 0.5)
        s = e + self.lam * self.ie

        # Adaptive switching gain; bounded to avoid duty chatter.
        self.gain = float(np.clip(0.025 + 0.0025 * abs(s), 0.025, 0.09))
        i_ref = np.clip(ipv - 0.25 * s, 0.0, IMAX)
        e_i = i_ref - il

        d_ff = 1.0 - vp / max(vo, 1.0)
        boundary = 0.25
        d = d_ff + 0.028 * e_i + self.gain * np.tanh(s / boundary)
        return float(np.clip(d, DMIN, DMAX))


class MPC:
    """Low-cost finite-candidate MPC using the same plant model."""

    def __init__(self):
        self.d = 0.30

    @staticmethod
    def predict(x, d, dt, params, ref, ipv_fun):
        vp, il, vo = x
        L, cin, cout, load = params
        ipv = ipv_fun(vp)

        dil = (vp - (1.0 - d) * vo) / L
        dvp = (ipv - il) / cin
        dvo = ((1.0 - d) * il - load) / cout
        return np.array([vp + dt * dvp, il + dt * dil, vo + dt * dvo])

    def duty(self, ref, vp, vo, il, dt, params, ipv_fun):
        candidates = np.linspace(
            max(DMIN, self.d - 0.06), min(DMAX, self.d + 0.06), 9
        )
        best_cost = float("inf")
        best_d = self.d

        for d in candidates:
            x = np.array([vp, il, vo], dtype=float)
            cost = 0.0
            for _ in range(4):
                x = self.predict(x, d, dt, params, ref, ipv_fun)
                vp_p, il_p, vo_p = x
                cost += (
                    20.0 * (vp_p - ref) ** 2
                    + 0.15 * (vo_p - VBUS_REF) ** 2
                    + 0.35 * max(0.0, il_p - IMAX) ** 2
                    + 0.08 * (d - self.d) ** 2
                )

                # Hard penalties keep candidates from driving the plant outside
                # a physically meaningful operating region.
                if vp_p < 0 or vp_p > PV_VOC:
                    cost += 1e5
                if vo_p < 25 or vo_p > 90:
                    cost += 1e4

            if cost < best_cost:
                best_cost, best_d = cost, d

        self.d = float(np.clip(best_d, DMIN, DMAX))
        return self.d


def fault_schedule(t, t_end):
    """Common deterministic fault/degradation schedule scaled to the run length.

    This keeps the fault events inside both short MPPT profiles (for example
    an ~80 ms CSV) and longer profiles. All controllers receive exactly the
    same events.
    """
    t_end = max(float(t_end), SIM_DT)
    f1 = 0.40 * t_end
    f2 = 0.55 * t_end
    f3 = 0.70 * t_end
    f4 = 0.82 * t_end

    L_ratio = 1.0 if t < f3 else 0.70
    Cin_ratio = 1.0 if t < f1 else 0.50
    Cout_ratio = 1.0 if t < f4 else 0.60
    load_spike = 0.0 if not (f2 <= t <= f2 + 0.06 * t_end) else 0.45
    return L_ratio, Cin_ratio, Cout_ratio, load_spike


def run(name, controller, t_ref, vref_ref):
    t_end = float(t_ref[-1])
    t = np.arange(0.0, t_end + SIM_DT * 0.5, SIM_DT)
    vref = np.interp(t, t_ref, vref_ref)

    # Start near nominal MPP operating point.
    vp = min(45.0, float(np.median(vref)))
    il = pv_current(vp)
    vo = VBUS_REF

    rows = []
    start = time.perf_counter()

    for k, tk in enumerate(t):
        ref = float(vref[k])
        Lr, Cir, Cor, load_spike = fault_schedule(tk, t_end)
        L = L0 * Lr
        Cin = CIN0 * Cir
        Cout = COUT0 * Cor
        load = VBUS_REF / RLOAD + load_spike

        # Common PV plant. Irradiance is kept nominal so that all differences
        # come from the controller rather than an artificial plant mismatch.
        ipv = pv_current(vp)

        if name == "MPC":
            params = (L, Cin, Cout, load)
            d = controller.duty(ref, vp, vo, il, SIM_DT, params, pv_current)
        elif name == "ASMC":
            d = controller.duty(ref, vp, vo, il, SIM_DT, ipv)
        else:
            d = controller.duty(ref, vp, vo, il, SIM_DT, ipv)

        # Averaged continuous-conduction boost-converter model.
        dil = (vp - (1.0 - d) * vo) / L
        dvp = (ipv - il) / Cin
        dvo = ((1.0 - d) * il - load) / Cout

        il += SIM_DT * dil
        vp += SIM_DT * dvp
        vo += SIM_DT * dvo

        il = float(np.clip(il, 0.0, IMAX))
        vp = float(np.clip(vp, 0.0, PV_VOC))
        vo = float(np.clip(vo, 20.0, 90.0))

        rows.append(
            [
                tk,
                ref,
                vp,
                il,
                vo,
                ipv * vp,
                d,
                Lr,
                Cir,
                Cor,
            ]
        )

    runtime = time.perf_counter() - start
    return np.asarray(rows), runtime


def summarize(name, r, runtime):
    pv_rmse = np.sqrt(np.mean((r[:, 2] - r[:, 1]) ** 2))
    bus_rmse = np.sqrt(np.mean((r[:, 4] - VBUS_REF) ** 2))
    bus_std = float(np.std(r[:, 4]))
    mean_power = float(np.mean(r[:, 5]))
    max_overshoot = float(max(0.0, np.max(r[:, 4]) - VBUS_REF))
    return [
        name,
        pv_rmse,
        bus_rmse,
        bus_std,
        mean_power,
        max_overshoot,
        runtime,
        runtime / len(r) * 1e6,
    ]


def main():
    t_ref, vref = load_profile()

    controllers = {
        "ACMC": ACMC(),
        "ASMC": ASMC(),
        "MPC": MPC(),
    }

    all_results = []
    metrics = []
    for name, controller in controllers.items():
        r, runtime = run(name, controller, t_ref, vref)
        all_results.append((name, r))
        metrics.append(summarize(name, r, runtime))

    metric_cols = [
        "Controller",
        "PV_RMSE_V",
        "Bus_RMSE_V",
        "Bus_std_V",
        "Mean_PV_Power_W",
        "Bus_overshoot_V",
        "Runtime_s",
        "Runtime_us_per_sample",
    ]
    metric_df = pd.DataFrame(metrics, columns=metric_cols)
    metric_df.to_csv(OUT / "controller_metrics.csv", index=False)

    rows = []
    for name, r in all_results:
        for q in r:
            rows.append([*q, name])
    ts_cols = [
        "time_s",
        "V_ref",
        "V_pv",
        "I_L",
        "V_out",
        "PV_power_W",
        "Duty",
        "L_ratio",
        "Cin_ratio",
        "Cout_ratio",
        "Controller",
    ]
    pd.DataFrame(rows, columns=ts_cols).to_csv(
        OUT / "controller_timeseries.csv", index=False
    )

    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for name, r in all_results:
        ax[0].plot(r[:, 0], r[:, 2], label=name)
        ax[1].plot(r[:, 0], r[:, 4], label=name)
        ax[2].plot(r[:, 0], r[:, 5], label=name)

    ax[0].plot(t_ref, vref, "k--", label="MPPT Vref")
    ax[1].axhline(VBUS_REF, color="k", linestyle="--", label="65 V reference")

    ax[0].set_ylabel("PV Voltage (V)")
    ax[1].set_ylabel("Bus Voltage (V)")
    ax[2].set_ylabel("PV Power (W)")
    ax[2].set_xlabel("Time (s)")
    ax[0].set_title("ACMC vs ASMC vs MPC — Corrected Common-Plant Simulation")

    for a in ax:
        a.grid(alpha=0.25)
        a.legend()

    fig.tight_layout()
    fig.savefig(OUT / "controller_comparison.png", dpi=180)
    plt.close(fig)

    print(metric_df.to_string(index=False))
    print("\nSaved:")
    print(OUT / "controller_metrics.csv")
    print(OUT / "controller_timeseries.csv")
    print(OUT / "controller_comparison.png")


if __name__ == "__main__":
    main()
