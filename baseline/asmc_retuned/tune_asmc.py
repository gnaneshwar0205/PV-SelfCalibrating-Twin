import sys, json, time, numpy as np
from scipy.optimize import differential_evolution
import core as C
from asmc_fast import *
variant = int(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
WIDE = len(sys.argv) > 3 and sys.argv[3] == 'wide'
def scen(seed, noisy):
    sc = C.random_scenario(seed); r = np.random.default_rng(777 + seed)
    Z = r.standard_normal((C.N_STEPS, 3)) if noisy else np.zeros((C.N_STEPS, 3))
    return sc['vref'].copy(), sc['G'].copy(), float(sc['vref'][0]), Z, (0.5 if noisy else 0.0)
TRAIN = [scen(s, s % 3 == 0) for s in range(24)]
def cost_th(th):
    r = []; mx = []; rp = []; da = []
    for vref, G, vp0, Z, ns in TRAIN:
        vp, d, vo, p = sim_asmc(variant, th, vref, G, vp0, Z, ns)
        m = metrics_fast(vp, d, vref)
        if not np.isfinite(m[0]) or m[1] > 30: return 1e3
        r.append(m[0]); mx.append(m[1]); rp.append(m[2]); da.append(m[3])
    r = np.array(r)
    return float(r.mean() + 0.5 * r.std() + 0.1 * np.mean(mx) + 0.5 * np.mean(rp) + 2.0 * np.mean(da))
if variant == 0:
    b = [(0.1, 30), (0, 100), (0, 1.0), (0.001, 0.3), (0, 1.0), (0.02, 1.0), (0.02, 5), (0, 1.3)] if WIDE else [(0.1, 20), (0, 50), (0, 0.05), (0.001, 0.1), (0, 0.2), (0.02, 0.6), (0.02, 5), (0, 1.3)]
    f = lambda x: cost_th(np.array(x)); conv = lambda x: list(x)
else:
    b = [(2, 3.8), (3, 6.5), (0.002, 0.1), (0, 500), (1, 100), (0.002, 0.1), (0.02, 0.3)]
    conv = lambda x: [10 ** x[0], 10 ** x[1]] + list(x[2:]); f = lambda x: cost_th(np.array(conv(x)))
cost_th(np.array(conv([ (lo+hi)/2 for lo, hi in b])))   # compile
t0 = time.time()
r = differential_evolution(f, b, popsize=15, maxiter=120, seed=seed, tol=1e-8, polish=False, updating='immediate')
th = conv(r.x)
json.dump(dict(variant=variant, theta=th, J=r.fun, seed=seed), open(f'outputs/asmc_v{variant}_s{seed}{'_wide' if WIDE else ''}.json', 'w'))
print('variant', variant, 'seed', seed, 'J=%.4f' % r.fun, 'time %.0fs' % (time.time() - t0), np.round(th, 5).tolist(), flush=True)
