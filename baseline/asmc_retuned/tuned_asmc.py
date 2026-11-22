"""Tuned ASMC (adaptive sliding-mode control) for 02_compare_controllers.py (drop-in replacement).

Same interface: duty(ref, vp, vo, il, dt, ipv) -> duty in [DMIN, DMAX].
Gains were found by differential evolution on the project's common plant and fault schedule.

USAGE:  from tuned_asmc import ASMC
"""
import math

DMIN, DMAX = 0.02, 0.90
IMAX = 5.0


def _clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


class ASMC:
    """Adaptive sliding-mode control on the current surface s = il - i_ref.

    i_ref = ipv - K1*e - K2*integral(e).  Switching term is  -g * tanh(s/PHI),  g adapting with |s|.

    IMPORTANT FIX vs original: the original added +gain*tanh(s_voltage) to the duty, which has the
    wrong sign for a boost converter with the PV on the input (voltage too low -> duty up -> current
    up -> voltage falls further). Here the switching term acts on the current error with the correct
    sign: il above reference -> duty down.
    """
    # v2 gains (wider search, 3.87 composite cost). Earlier v1 gains are kept in ASMC_V1 below.
    K1, K2 = 5.139795127835239, 0.06576269117570638
    KP_I = 4.289012320489765e-05
    G0, GSLOPE, GMAX = 0.005010457221801006, 0.01532848120397138, 0.46516242712423617
    PHI = 2.974395004324842   # wide boundary layer => essentially no chatter

    def __init__(self):
        self.ie = 0.0
        self.gain = self.G0

    def duty(self, ref, vp, vo, il, dt, ipv):
        e = ref - vp
        self.ie = _clip(self.ie + e * dt, -0.5, 0.5)
        i_ref = _clip(ipv - self.K1 * e - self.K2 * self.ie, 0.0, IMAX)
        s = il - i_ref

        self.gain = _clip(self.G0 + self.GSLOPE * abs(s), self.G0, self.GMAX)
        d_ff = 1.0 - vp / max(vo, 1.0)
        d = d_ff - self.KP_I * s - self.gain * math.tanh(s / self.PHI)
        return _clip(d, DMIN, DMAX)


class ASMC_V1(ASMC):
    """First-round tuned ASMC (composite cost 4.11), kept for reference."""
    K1, K2 = 2.993164930889173, 2.194201512766682
    KP_I = 0.00209214263806648
    G0, GSLOPE, GMAX = 0.017079839236454557, 0.012629014480952647, 0.08881505890905333
    PHI = 1.9549983764045153
