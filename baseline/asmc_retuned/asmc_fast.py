"""Fast (numba) closed-loop simulator replicating the Python harness plant exactly, with two ASMC structures."""
import numpy as np, numba as nb
import core as C
from core import pv_i, faults, VBUS, L0, CIN0, COUT0, DMIN, DMAX, IMAX, VOC, ILOAD, DT


@nb.njit(cache=True)
def sim_asmc(variant, th, vref, G, vp0, Z, nscale):
    N = vref.shape[0]
    T = (N - 1) * DT
    vp = vp0
    il = pv_i(vp0, 1.0)
    vo = VBUS
    vp_out = np.zeros(N)
    d_out = np.zeros(N)
    vo_out = np.zeros(N)
    p_out = np.zeros(N)
    ie = 0.0
    kad = 0.0
    prev = vref[0]
    first = True
    for k in range(N):
        t = k * DT
        Lr, Cir, Cor, sp = faults(t, T)
        L = L0 * Lr
        Cin = CIN0 * Cir
        Cout = COUT0 * Cor
        load = ILOAD + sp
        g = G[k]
        ipv = pv_i(vp, g)
        mvp = vp + nscale * 0.4 * Z[k, 0]
        mil = il + nscale * 0.1 * Z[k, 1]
        mvo = vo + nscale * 0.6 * Z[k, 2]
        ref = vref[k]
        dref = 0.0 if first else (ref - prev) / DT
        first = False
        prev = ref
        if variant == 0:     # current-surface ASMC (teammate structure) + reference-slope feed-forward
            e = ref - mvp
            ie = min(max(ie + e * DT, -0.5), 0.5)
            iref = min(max(ipv - th[0] * e - th[1] * ie - th[7] * CIN0 * dref, 0.0), IMAX)
            s = mil - iref
            gain = min(max(th[3] + th[4] * abs(s), th[3]), th[5])
            d = (1.0 - mvp / max(mvo, 1.0)) - th[2] * s - gain * np.tanh(s / th[6])
        else:                # voltage-surface ASMC with equivalent control + adaptive switching gain
            ev = ref - mvp
            vdot = (ipv - mil) / CIN0
            edot = dref - vdot
            ie = min(max(ie + ev * DT, -0.02), 0.02)
            s = edot + th[0] * ev + th[1] * ie
            b = max(mvo, 1.0) / (L0 * CIN0)
            sn = s / b
            kad = kad + DT * (th[3] * abs(sn) - th[4] * kad)
            kad = min(max(kad, 0.0), th[6])
            deq = 1.0 - (mvp + L0 * CIN0 * (th[0] * edot + th[1] * ev)) / max(mvo, 1.0)
            d = deq - (th[2] + kad) * np.tanh(sn / th[5])
        d = min(max(d, DMIN), DMAX)
        il_n = il + DT * (vp - (1.0 - d) * vo) / L
        vp_n = vp + DT * (ipv - il_n) / Cin
        vo_n = vo + DT * ((1.0 - d) * il_n - load) / Cout
        il = min(max(il_n, 0.0), IMAX)
        vp = min(max(vp_n, 0.0), VOC)
        vo = min(max(vo_n, 20.0), 90.0)
        vp_out[k] = vp
        d_out[k] = d
        vo_out[k] = vo
        p_out[k] = ipv * vp
    return vp_out, d_out, vo_out, p_out


def metrics_fast(vp, d, vref):
    e = vp - vref
    w = 50
    lp = np.convolve(e, np.ones(w) / w, 'same')
    return (np.sqrt(np.mean(e ** 2)), np.abs(e).max(), np.sqrt(np.mean((e - lp)[w:-w] ** 2)), np.mean(np.abs(np.diff(d[::5]))))
