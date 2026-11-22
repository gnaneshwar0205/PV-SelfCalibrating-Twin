import math
DMIN, DMAX, IMAX = 0.02, 0.90, 5.0
CIN0, L0 = 470e-6, 5e-3
def _clip(x, lo, hi): return lo if x < lo else (hi if x > hi else x)
class ASMC_V0:
    """Current-surface ASMC (same structure as the teammate's) + reference-slope feed-forward; gains re-tuned."""
    TH = None
    def __init__(self): self.ie = 0.0; self.prev = None
    def duty(self, ref, vp, vo, il, dt, ipv):
        K1, K2, KP_I, G0, GS, GMAX, PHI, KFF = self.TH
        dref = 0.0 if self.prev is None else (ref - self.prev) / dt; self.prev = ref
        e = ref - vp; self.ie = _clip(self.ie + e * dt, -0.5, 0.5)
        iref = _clip(ipv - K1 * e - K2 * self.ie - KFF * CIN0 * dref, 0.0, IMAX)
        s = il - iref; gain = _clip(G0 + GS * abs(s), G0, GMAX)
        return _clip((1.0 - vp / max(vo, 1.0)) - KP_I * s - gain * math.tanh(s / PHI), DMIN, DMAX)
class ASMC_V1:
    """Voltage-surface ASMC with equivalent control + adaptive switching gain."""
    TH = None
    def __init__(self): self.ie = 0.0; self.kad = 0.0; self.prev = None
    def duty(self, ref, vp, vo, il, dt, ipv):
        c, ci, k0, g1, g2, phi, kmax = self.TH
        dref = 0.0 if self.prev is None else (ref - self.prev) / dt; self.prev = ref
        ev = ref - vp; edot = dref - (ipv - il) / CIN0
        self.ie = _clip(self.ie + ev * dt, -0.02, 0.02)
        s = edot + c * ev + ci * self.ie; b = max(vo, 1.0) / (L0 * CIN0); sn = s / b
        self.kad = _clip(self.kad + dt * (g1 * abs(sn) - g2 * self.kad), 0.0, kmax)
        deq = 1.0 - (vp + L0 * CIN0 * (c * edot + ci * ev)) / max(vo, 1.0)
        return _clip(deq - (k0 + self.kad) * math.tanh(sn / phi), DMIN, DMAX)
