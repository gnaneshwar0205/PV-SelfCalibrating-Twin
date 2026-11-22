"""MPC 2 (tuned MPC) for 02_compare_controllers.py (drop-in replacement, same interface as tuned_acmc.py / tuned_asmc.py).

    duty(ref, vp, vo, il, dt, ipv) -> duty in [DMIN, DMAX]

Receding-horizon finite-set MPC on the averaged boost model:
  * prediction step 100 us, horizon NH = 24 steps (2.4 ms); the duty is held for the first NSW = 3 steps (move 1),
    then a second constant duty (move 2, within +-0.10 of move 1) for the rest;
  * the controller only knows the NOMINAL L, Cin, Cout (not the injected faults);
  * reference preview = current reference slope; terminal cost on inductor current and PV voltage;
  * the optimiser runs once per 100 us and the duty is held for the 20 us plant steps in between.
Weights were found by differential evolution on randomised reference profiles (see mpc_2_params.json).
Uses numba if installed (about 100x faster); falls back to plain Python otherwise.
"""
import math
try:
    import numba as _nb
    _jit = _nb.njit(cache=True)
except Exception:                       # plain-Python fallback
    def _jit(f):
        return f

DMIN, DMAX, IMAX = 0.02, 0.90, 5.0
L0, CIN0, COUT0 = 5e-3, 470e-6, 220e-6
VOC, ISC, SHAPE, ILOAD = 4 * 14.4, 2 * 1.45, 8.0, 1.73
TP = 100e-6

Q, R1, R2, WTERM = 4.736871359041204, 4.334037967412131, 1.4182432850973667, 113.71339622346659
NH, NSW = 24, 3


@_jit
def _pv_i(vp, g):
    x = min(max(vp / VOC, 0.0), 1.0)
    return max(0.0, ISC * g * (1.0 - x ** SHAPE))


@_jit
def _solve(vp0, il0, vo0, dprev, vref, dref, g, q, r1, r2, wterm, nh, nsw):
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
            for k in range(nh):
                d = d1 if k < nsw else d2
                ipv = _pv_i(vp, g)
                dil = (vp - (1.0 - d) * vo) / L0
                dvp = (ipv - il) / CIN0
                dvo = ((1.0 - d) * il - ILOAD) / COUT0
                il = il + TP * dil
                vp = vp + TP * dvp
                vo = vo + TP * dvo
                cost += q * (vp - (vref + dref * TP * (k + 1))) ** 2
                if il > IMAX:
                    cost += 50.0 * (il - IMAX) ** 2
                if il < 0.0:
                    cost += 50.0 * il ** 2
                if vo < 30.0:
                    cost += 0.2 * (30.0 - vo) ** 2
            vrN = vref + dref * TP * nh
            cost += wterm * (il - (_pv_i(vp, g) - CIN0 * dref)) ** 2
            cost += wterm * 0.5 * (vp - vrN) ** 2
            if cost < best:
                best = cost
                bd = d1
    return min(max(bd, DMIN), DMAX)


class MPC:
    def __init__(self):
        self.d = 0.30
        self.k = 0
        self.prev = None

    def duty(self, ref, vp, vo, il, dt, ipv):
        hold = max(1, int(round(TP / dt)))
        if self.k % hold == 0:
            dref = 0.0 if self.prev is None else (ref - self.prev) / dt
            x = min(max(vp / VOC, 0.0), 1.0)
            den = ISC * (1.0 - x ** SHAPE)
            g = ipv / den if den > 1e-6 else 1.0        # irradiance estimate from measured PV current
            self.d = float(_solve(vp, il, vo, self.d, ref, dref, g, Q, R1, R2, WTERM, NH, NSW))
        self.prev = ref
        self.k += 1
        return self.d
