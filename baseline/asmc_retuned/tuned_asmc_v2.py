"""ASMC v2: re-tuned adaptive sliding-mode controller (drop-in replacement for tuned_asmc.ASMC).

Same interface: duty(ref, vp, vo, il, dt, ipv) -> duty in [DMIN, DMAX].

Structure = the teammate's current-surface ASMC (sliding surface s = il - i_ref, i_ref = ipv - K1*e - K2*int(e),
adaptive switching gain g = G0 + GSLOPE*|s| bounded by GMAX, tanh boundary layer PHI) plus ONE new term:
a feed-forward of the reference slope  i_ref -= KFF*Cin*dVref/dt  (dVref/dt is estimated from successive reference values).
All 8 parameters were re-tuned by differential evolution on randomised reference profiles, irradiance steps and sensor noise
(training seeds never used for testing). Tuning script: tune_asmc.py (variant 0).
"""
import math

DMIN, DMAX = 0.02, 0.90
IMAX = 5.0
CIN0 = 470e-6


def _clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


class ASMC:
    K1, K2 = 1.0078583283524658, 12.602587937850016
    KP_I = 0.04999985292055483
    G0, GSLOPE, GMAX = 0.0999999852487289, 0.19999987090197324, 0.3452853873554632
    PHI = 0.39208126550373956
    KFF = 0.556774501272047          # reference-slope feed-forward (0 = the teammate's ASMC)

    def __init__(self):
        self.ie = 0.0
        self.prev = None

    def duty(self, ref, vp, vo, il, dt, ipv):
        dref = 0.0 if self.prev is None else (ref - self.prev) / dt
        self.prev = ref
        e = ref - vp
        self.ie = _clip(self.ie + e * dt, -0.5, 0.5)
        i_ref = _clip(ipv - self.K1 * e - self.K2 * self.ie - self.KFF * CIN0 * dref, 0.0, IMAX)
        s = il - i_ref
        gain = _clip(self.G0 + self.GSLOPE * abs(s), self.G0, self.GMAX)
        d_ff = 1.0 - vp / max(vo, 1.0)
        d = d_ff - self.KP_I * s - gain * math.tanh(s / self.PHI)
        return _clip(d, DMIN, DMAX)
