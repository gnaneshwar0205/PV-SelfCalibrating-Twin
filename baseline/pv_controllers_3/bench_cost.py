"""Compute cost per decision of the three controllers (desktop-CPU numbers; re-run on your own machine)."""
import time
import tuned_acmc as TA, tuned_asmc as TS, nn_mpc as NN
def bench(f, n):
    f(); t = time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter() - t) / n * 1e6
a, s, nn = TA.ACMC(), TS.ASMC(), NN.NNMPC()
def fn(): nn.k = 0; return nn.duty(41., 40., 62., 2.4, 2e-5, 2.5)
print("ACMC   us per call (called every 20 us)    ", round(bench(lambda: a.duty(41., 40., 62., 2.4, 2e-5, 2.5), 100000), 2))
print("ASMC   us per call (called every 20 us)    ", round(bench(lambda: s.duty(41., 40., 62., 2.4, 2e-5, 2.5), 100000), 2))
print("NN-MPC us per evaluation (every 100 us)    ", round(bench(fn, 20000), 2), " | multiply-adds: 1344")
