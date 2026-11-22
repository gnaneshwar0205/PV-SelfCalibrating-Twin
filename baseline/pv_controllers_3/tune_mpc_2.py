import json, time, numpy as np
from scipy.optimize import differential_evolution
from core import *
TR=[]
for s in range(6):
    sc=random_scenario(s); sc['G']=np.ones(N_STEPS); TR.append(sc)
def th_of(x): return [10**x[0],10**x[1],10**x[2],10**x[3],0.2,float(int(round(x[4]))),float(min(int(round(x[5])),int(round(x[4]))))]
def cost(x):
    th=th_of(x); r=[];mx=[];rp=[];da=[]
    for sc in TR:
        out,_,_=run(MPC,th,sc); m=metrics(out,sc)
        if m['Unstable']: return 1e3
        r.append(m['RMSE_V']);mx.append(m['MaxErr_V']);rp.append(m['Ripple_V']);da.append(m['Duty_activity'])
    r=np.array(r); return float(r.mean()+0.5*r.std()+0.1*np.mean(mx)+0.5*np.mean(rp)+2.0*np.mean(da))
t0=time.time(); it=[0]
def cb(xk,convergence=None):
    it[0]+=1; open('tune_mpc_2.log','a').write(f"gen {it[0]} J={cost(xk):.4f} th={np.round(th_of(xk),3).tolist()} t={time.time()-t0:.0f}s\n")
b=[(0,3),(-1,2),(-1,2),(-1,3),(6,24),(1,8)]
x0=np.array([np.log10(20),np.log10(5),0,np.log10(5),12,3.])
r=differential_evolution(cost,b,popsize=8,maxiter=16,seed=3,tol=1e-7,polish=False,callback=cb,x0=x0,updating='immediate')
th=th_of(r.x); json.dump(dict(order="q,r1,r2,wterm,wvo,NH,NSW",theta=th,J=r.fun),open('outputs/mpc_2_params.json','w'))
open('tune_mpc_2.log','a').write(f"DONE J={r.fun:.4f} th={th}\n")
