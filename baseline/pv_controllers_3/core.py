"""
core.py  -  common-plant simulation library (numba-accelerated)

Same PV + boost plant, load and fault schedule as the original
02_compare_controllers.py.  What is different here:

  * every controller runs at the SAME sample time (TS = 100 us), the plant is
    integrated at 20 us;
  * every controller only knows NOMINAL L, Cin (faults are unknown to it, so
    robustness is actually tested);
  * scenarios are randomised (reference shape, irradiance, irradiance steps) so
    that controllers can be trained/tuned on one set and judged on another.
"""
import numpy as np
import numba as nb
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------- constants
VBUS = 65.0
L0, CIN0, COUT0 = 5e-3, 470e-6, 220e-6
DMIN, DMAX, IMAX = 0.02, 0.90, 5.0
VOC, ISC, SHAPE = 4 * 14.4, 2 * 1.45, 8.0
ILOAD = 1.73                       # = VBUS / RLOAD in the original script
DT = 20e-6                         # plant integration step
HOLD = 5                           # controller sample = HOLD*DT = 100 us
TS = HOLD * DT
T_END = 0.079
N_STEPS = int(round(T_END / DT)) + 1

ACMC, ASMC, MPC, NNMPC = 0, 1, 2, 3
NIN = 9                            # NN input features


@nb.njit(cache=True, inline="always")
def pv_i(vp, g):
    x = min(max(vp / VOC, 0.0), 1.0)
    return max(0.0, ISC * g * (1.0 - x ** SHAPE))


@nb.njit(cache=True)
def faults(t, T):
    f1, f2, f3, f4 = 0.40 * T, 0.55 * T, 0.70 * T, 0.82 * T
    Lr = 1.0 if t < f3 else 0.70
    Cir = 1.0 if t < f1 else 0.50
    Cor = 1.0 if t < f4 else 0.60
    spike = 0.45 if (t >= f2 and t <= f2 + 0.06 * T) else 0.0
    return Lr, Cir, Cor, spike


# ------------------------------------------------------------ NN features
@nb.njit(cache=True)
def features(vp, il, vo, dprev, vref, dref, ipv, f):
    iss = ipv - CIN0 * dref
    f[0] = (vp - vref) / 5.0
    f[1] = (il - iss) / 2.0
    f[2] = (vo - VBUS) / 15.0
    f[3] = dprev
    f[4] = dref / 1000.0
    f[5] = ipv / 3.0
    f[6] = vp / 50.0
    f[7] = il / 3.0
    f[8] = (1.0 - vp / max(vo, 1.0))          # feed-forward duty


@nb.njit(cache=True)
def nn_forward(f, W1, b1, W2, b2, W3, b3):
    h1 = np.tanh(f @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    return (h2 @ W3)[0] + b3[0]


# ----------------------------------------------------------- MPC expert
@nb.njit(cache=True)
def mpc_expert(vp0, il0, vo0, dprev, vref, dref, g, mp):
    """Receding-horizon finite-set MPC with 2 moves over a 1.2 ms horizon,
    reference preview by its current slope, terminal current cost."""
    q, r1, r2, wterm, wvo = mp[0], mp[1], mp[2], mp[3], mp[4]
    NH = int(mp[5])
    NSW = int(mp[6])
    n1, n2 = 13, 9
    lo = max(DMIN, dprev - 0.15)
    hi = min(DMAX, dprev + 0.15)
    best = 1e30
    bd = dprev
    for i in range(n1):
        d1 = lo + (hi - lo) * i / (n1 - 1)
        for j in range(n2):
            d2 = min(max(d1 - 0.10 + 0.20 * j / (n2 - 1), DMIN), DMAX)
            vp, il, vo = vp0, il0, vo0
            cost = r1 * (d1 - dprev) ** 2 + r2 * (d2 - d1) ** 2
            for k in range(NH):
                d = d1 if k < NSW else d2
                ipv = pv_i(vp, g)
                dil = (vp - (1.0 - d) * vo) / L0
                dvp = (ipv - il) / CIN0
                dvo = ((1.0 - d) * il - ILOAD) / COUT0
                il = il + TS * dil
                vp = vp + TS * dvp
                vo = vo + TS * dvo
                vr = vref + dref * TS * (k + 1)
                cost += q * (vp - vr) ** 2
                if il > IMAX:
                    cost += 50.0 * (il - IMAX) ** 2
                if il < 0.0:
                    cost += 50.0 * il ** 2
                if vo < 30.0:
                    cost += wvo * (30.0 - vo) ** 2
            ipvN = pv_i(vp, g)
            vrN = vref + dref * TS * NH
            cost += wterm * (il - (ipvN - CIN0 * dref)) ** 2
            cost += wterm * 0.5 * (vp - vrN) ** 2
            if cost < best:
                best = cost
                bd = d1
    return min(max(bd, DMIN), DMAX)


# ------------------------------------------------------------- simulator
@nb.njit(cache=True)
def simulate(ctrl, th, W1, b1, W2, b2, W3, b3,
             vref, dref, G, vp0, record, beta, seed):
    """Returns out[N,5] = vp, il, vo, duty, pv_power  (+ recorded DAgger data)."""
    N = vref.shape[0]
    T = (N - 1) * DT
    np.random.seed(seed)
    out = np.zeros((N, 5))
    maxrec = N // HOLD + 2
    Frec = np.zeros((maxrec if record else 1, NIN))
    Yrec = np.zeros(maxrec if record else 1)
    nrec = 0
    f = np.zeros(NIN)

    vp = vp0
    il = pv_i(vp0, G[0])
    vo = VBUS
    d = 1.0 - vp / vo
    # controller states
    vf = vp
    ui = 0.0
    uii = 0.0
    ie = 0.0
    kad = 0.0
    mp = np.array([20.0, 5.0, 1.0, 5.0, 0.2, 12.0, 3.0])
    if th.shape[0] >= 7:
        mp = th.copy()

    for k in range(N):
        t = k * DT
        g = G[k]
        Lr, Cir, Cor, spike = faults(t, T)
        L, Cin, Cout = L0 * Lr, CIN0 * Cir, COUT0 * Cor
        load = ILOAD + spike
        ipv = pv_i(vp, g)

        if k % HOLD == 0:                       # ---- controller tick
            ref = vref[k]
            dr = dref[k]
            if ctrl == ACMC:
                # th: kpv, kiv, kpi, kii, kff, a_f
                vf = vf + th[5] * (vp - vf)
                ev = ref - vf
                ui = min(max(ui + th[1] * ev * TS, -2.0), 2.0)
                iref = ipv - th[4] * CIN0 * dr - th[0] * ev - ui
                iref = min(max(iref, 0.0), IMAX)
                ei = iref - il
                uii = min(max(uii + th[3] * ei * TS, -0.3), 0.3)
                d = (1.0 - vp / max(vo, 1.0)) + th[2] * ei + uii
            elif ctrl == ASMC:
                # th: c, ci, k0, g1, g2, phi, kmax
                ev = ref - vp
                vdot = (ipv - il) / CIN0
                edot = dr - vdot
                ie = min(max(ie + ev * TS, -0.02), 0.02)
                s = edot + th[0] * ev + th[1] * ie
                b = max(vo, 1.0) / (L0 * CIN0)
                sn = s / b
                kad = kad + TS * (th[3] * abs(sn) - th[4] * kad)
                kad = min(max(kad, 0.0), th[6])
                deq = 1.0 - (vp + L0 * CIN0 * (th[0] * edot + th[1] * ev)) / max(vo, 1.0)
                d = deq - (th[2] + kad) * np.tanh(sn / th[5])
            else:
                if ctrl == MPC:
                    d = mpc_expert(vp, il, vo, d, ref, dr, g, mp)
                else:
                    features(vp, il, vo, d, ref, dr, ipv, f)
                    dn = (1.0 - vp / max(vo, 1.0)) + nn_forward(f, W1, b1, W2, b2, W3, b3) / 5.0
                    if record:
                        de = mpc_expert(vp, il, vo, d, ref, dr, g, mp)
                        Frec[nrec, :] = f
                        Yrec[nrec] = (de - f[8]) * 5.0
                        nrec += 1
                        if np.random.random() < beta:
                            dn = de
                    d = dn
            d = min(max(d, DMIN), DMAX)

        # averaged CCM boost model (Euler, 20 us)
        il = min(max(il + DT * (vp - (1.0 - d) * vo) / L, 0.0), IMAX)
        vp_n = vp + DT * (ipv - il) / Cin
        vo = min(max(vo + DT * ((1.0 - d) * il - load) / Cout, 20.0), 90.0)
        out[k, 0] = vp
        out[k, 1] = il
        out[k, 2] = vo
        out[k, 3] = d
        out[k, 4] = ipv * vp
        vp = min(max(vp_n, 0.0), VOC)
    return out, Frec[:nrec], Yrec[:nrec]


# -------------------------------------------------------------- scenarios
def senior_scenario():
    p = Path(__file__).resolve().parent / "data" / "mppt_vref_profile.csv"
    d = pd.read_csv(p)
    v = d["V_ref"].to_numpy(float)
    return _from_knots(v, 1e-3, np.ones(N_STEPS))


def _from_knots(v, dt_knot, G):
    t = np.arange(N_STEPS) * DT
    tk = np.arange(len(v)) * dt_knot
    vref = np.interp(t, tk, v)
    dref = np.gradient(vref, DT)
    # piecewise-constant slope of the linear interpolation
    idx = np.minimum((t / dt_knot).astype(int), len(v) - 2)
    dref = (v[idx + 1] - v[idx]) / dt_knot
    return dict(vref=vref, dref=dref, G=G, vp0=float(vref[0]))


def random_scenario(seed):
    """Random P&O-style reference: ramp up from a low start, then dither around
    the MPP; irradiance level randomised with an optional step."""
    r = np.random.default_rng(seed)
    n = 80
    step = r.uniform(0.8, 1.6)                 # V per knot (1 ms)
    v = np.empty(n)
    v[0] = r.uniform(8, 25)
    vmpp = r.uniform(42, 48)
    direction = 1
    hold_rev = 0
    for k in range(1, n):
        if v[k - 1] < vmpp - 2 * step:
            direction = -1 if (hold_rev > 0 or r.random() < 0.12) else 1
            hold_rev = hold_rev - 1 if hold_rev > 0 else (r.integers(1, 3) if direction == -1 else 0)
            if hold_rev == 0 and direction == -1:
                direction = 1
        else:
            direction = 1 if v[k - 1] < vmpp else -1
            if r.random() < 0.2:
                direction = -direction
        v[k] = v[k - 1] + direction * step
    v = np.clip(v, 5, 50)
    g0 = r.uniform(0.90, 1.0)
    G = np.full(N_STEPS, g0)
    if r.random() < 0.5:                       # irradiance step
        k0 = int(r.uniform(0.3, 0.7) * N_STEPS)
        G[k0:] = np.clip(g0 * r.uniform(0.9, 1.05), 0.88, 1.0)
    return _from_knots(v, 1e-3, G)


NN0 = (np.zeros((NIN, 1)), np.zeros(1), np.zeros((1, 1)), np.zeros(1), np.zeros((1, 1)), np.zeros(1))


def run(ctrl, th, sc, nn=None, record=False, beta=0.0, seed=0):
    W1, b1, W2, b2, W3, b3 = nn if nn is not None else dummy_nn()
    return simulate(ctrl, np.asarray(th, float), W1, b1, W2, b2, W3, b3,
                    sc["vref"], sc["dref"], sc["G"], sc["vp0"], record, beta, seed)


def dummy_nn():
    return (np.zeros((NIN, 2)), np.zeros(2), np.zeros((2, 2)), np.zeros(2), np.zeros((2, 1)), np.zeros(1))


# ---------------------------------------------------------------- metrics
def metrics(out, sc):
    vp, il, vo, d, p = out.T
    vref, G = sc["vref"], sc["G"]
    e = vp - vref
    n = len(e)
    k5 = int(5e-3 / DT)
    # high-frequency content of the error (oscillation / chatter):
    w = int(1e-3 / DT)
    ker = np.ones(w) / w
    e_lp = np.convolve(e, ker, mode="same")
    ripple = float(np.sqrt(np.mean((e - e_lp)[w:-w] ** 2)))
    vo_lp = np.convolve(vo, np.ones(5 * w) / (5 * w), mode="same")
    xm = 9.0 ** (-1.0 / SHAPE) * VOC
    pmax = xm * ISC * G * (1 - (xm / VOC) ** SHAPE)
    return dict(
        RMSE_V=float(np.sqrt(np.mean(e ** 2))),
        RMSE_post5ms_V=float(np.sqrt(np.mean(e[k5:] ** 2))),
        MaxErr_V=float(np.max(np.abs(e))),
        Ripple_V=ripple,
        Bus_ripple_V=float(np.sqrt(np.mean((vo - vo_lp)[5 * w:-5 * w] ** 2))),
        Bus_std_V=float(np.std(vo)),
        MPPT_eff_pct=float(100 * np.mean(p) / np.mean(pmax)),
        Mean_PV_W=float(np.mean(p)),
        Duty_activity=float(np.mean(np.abs(np.diff(d[::HOLD])))),
        Unstable=bool(np.max(np.abs(e)) > 30),
    )
