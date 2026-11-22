"""Neural-network MPC (drop-in, same interface as the other tuned controllers).

    NNMPC(weights="nnmpc_weights_H32.npz").duty(ref, vp, vo, il, dt, ipv) -> duty

A 9-input -> 32 -> 32 -> 1 tanh network that imitates the tuned receding-horizon MPC ("MPC 2").
It outputs a correction to the feed-forward duty  d_ff = 1 - vp/vo.  Only numpy is needed at run time;
the forward pass is ~1.3k multiply-adds and can be ported to C/DSP directly.
The network is evaluated once per 100 us (duty held in between), like MPC 2.
"""
import os
import numpy as np

DMIN, DMAX = 0.02, 0.90
CIN0, VBUS = 470e-6, 65.0
TP = 100e-6


class NNMPC:
    def __init__(self, weights=None):
        weights = weights or os.path.join(os.path.dirname(os.path.abspath(__file__)), "nnmpc_weights_H32.npz")
        w = np.load(weights)
        self.W1, self.b1, self.W2, self.b2 = w["W1"], w["b1"], w["W2"], w["b2"]
        self.W3, self.b3 = w["W3"].ravel(), float(np.ravel(w["b3"])[0])
        self.d, self.k, self.prev = 0.30, 0, None

    def duty(self, ref, vp, vo, il, dt, ipv):
        hold = max(1, int(round(TP / dt)))
        if self.k % hold == 0:
            dref = 0.0 if self.prev is None else (ref - self.prev) / dt
            dff = 1.0 - vp / max(vo, 1.0)
            f = np.array([(vp - ref) / 5.0, (il - (ipv - CIN0 * dref)) / 2.0, (vo - VBUS) / 15.0, self.d,
                          dref / 1000.0, ipv / 3.0, vp / 50.0, il / 3.0, dff])
            h = np.tanh(np.tanh(f @ self.W1 + self.b1) @ self.W2 + self.b2)
            self.d = float(min(max(dff + (h @ self.W3 + self.b3) / 5.0, DMIN), DMAX))
        self.prev = ref
        self.k += 1
        return self.d
