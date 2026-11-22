"""Tuned ACMC (average current-mode control) for 02_compare_controllers.py (drop-in replacement).

Same interface: duty(ref, vp, vo, il, dt, ipv) -> duty in [DMIN, DMAX].
Gains were found by differential evolution on the project's common plant and fault schedule.

USAGE:  from tuned_acmc import ACMC
"""
DMIN, DMAX = 0.02, 0.90
IMAX = 5.0


def _clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


class ACMC:
    """Average current-mode control: voltage PI -> inductor-current reference -> current PI + duty feed-forward.

    Changes vs original: almost no voltage filtering (alpha 0.015 -> 0.959), much stronger inner
    current loop (kp 0.035 -> 0.429, ki 0.045 -> 255.7), softer/faster outer loop. This damps the
    ~100 Hz L-Cin resonance that made the original ring.
    """
    ALPHA = 0.9588962055401444   # voltage measurement filter coefficient
    KV    = 0.2660537954975266   # outer loop: proportional
    KIV   = 0.0839989276639983   # outer loop: integral
    KP_I  = 0.035 + 0.39441267047693274   # inner loop: proportional (= 0.4294)
    KI_I  = 255.65888754284805   # inner loop: integral

    def __init__(self):
        self.iv = 0.0
        self.ii = 0.0
        self.vf = None

    def duty(self, ref, vp, vo, il, dt, ipv):
        self.vf = vp if self.vf is None else (1 - self.ALPHA) * self.vf + self.ALPHA * vp
        e_v = ref - self.vf
        self.iv = _clip(self.iv + e_v * dt, -0.5, 0.5)

        i_ref = _clip(ipv - self.KV * e_v - self.KIV * self.iv, 0.0, IMAX)
        e_i = i_ref - il
        self.ii = _clip(self.ii + e_i * dt, -0.5, 0.5)

        d_ff = 1.0 - vp / max(vo, 1.0)
        return _clip(d_ff + self.KP_I * e_i + self.KI_I * self.ii, DMIN, DMAX)
