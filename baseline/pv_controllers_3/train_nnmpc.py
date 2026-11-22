"""Train a neural-network MPC (imitation of the tuned MPC 'MPC 2') with DAgger."""
import json, time, numpy as np
from sklearn.neural_network import MLPRegressor
from core import *
EXP=json.load(open('outputs/mpc_2_params.json'))['theta']          # q,r1,r2,wterm,wvo,NH,NSW
H=(int(__import__('sys').argv[1]) if len(__import__('sys').argv)>1 else 32)
def scen(seed):
    sc=random_scenario(seed); r=np.random.default_rng(10_000+seed)
    g0=r.uniform(0.85,1.0); G=np.full(N_STEPS,g0)
    if r.random()<0.5:
        k0=int(r.uniform(0.3,0.7)*N_STEPS); G[k0:]=np.clip(g0*r.uniform(0.7,1.05),0.6,1.0)
    if r.random()<0.3: G[:]=1.0
    if r.random()<0.5: sc['vp0']=float(r.uniform(15,45))   # start-up mismatch states (e.g. 37 V start vs 10 V reference)
    sc['G']=G; return sc
TRAIN=[scen(s) for s in range(40)]; VAL=[scen(500+s) for s in range(10)]
def unpack(m): return (m.coefs_[0],m.intercepts_[0],m.coefs_[1],m.intercepts_[1],m.coefs_[2],m.intercepts_[2])
def rmse(sc,nn,ctrl=NNMPC):
    o,_,_=run(ctrl,EXP if ctrl==MPC else [0],sc,nn=nn); return metrics(o,sc)['RMSE_V']
ex_val=np.mean([rmse(sc,None,MPC) for sc in VAL]); print("expert val RMSE",round(ex_val,4),flush=True)
t0=time.time(); X=[];Y=[]; nn=None
m=MLPRegressor(hidden_layer_sizes=(H,H),activation='tanh',solver='adam',learning_rate_init=2e-3,batch_size=512,max_iter=150,tol=1e-8,n_iter_no_change=40,warm_start=True,random_state=0)
for it,beta in enumerate([1.0,0.6,0.3,0.15,0.0,0.0,0.0,0.0]):
    for i,sc in enumerate(TRAIN):
        _,F,Yl=run(NNMPC,EXP,sc,nn=(nn if nn is not None else dummy_nn()),record=True,beta=beta,seed=it*1000+i); X.append(F);Y.append(Yl)
    Xa,Ya=np.vstack(X),np.concatenate(Y)
    m.max_iter=150 if it<7 else 500
    import warnings; warnings.filterwarnings('ignore')
    m.fit(Xa,Ya); nn=unpack(m)
    v=np.mean([rmse(sc,nn) for sc in VAL])
    print(f"it{it} beta={beta} data={len(Ya)} R2={m.score(Xa,Ya):.4f} val RMSE NN={v:.4f} (expert {ex_val:.4f}) {time.time()-t0:.0f}s",flush=True)
    np.savez(f'outputs/nnmpc_H{H}.npz',W1=nn[0],b1=nn[1],W2=nn[2],b2=nn[3],W3=nn[4],b3=nn[5])
print("DONE",flush=True)
