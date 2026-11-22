import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'baseline'/'asmc_retuned'))
sys.path.insert(0,str(ROOT/'baseline'/'pv_controllers_3'))
from tuned_asmc_v2 import ASMC
from nn_mpc import NNMPC
from common.estimator import OnlineEstimator
from common.twin import BoostTwin
DT=20e-6; HOLD=5; TS=DT*HOLD; T_END=.079; N=int(round(T_END/DT))+1
VOC=57.6; ISC=2.9; SHAPE=8.; VBUS=65.; L0=5e-3; CI0=470e-6; CO0=220e-6; ILOAD=1.73

def pv_i(v,g):
    x=np.clip(v/VOC,0,1); return max(0.,ISC*g*(1-x**SHAPE))
def reference():
    p=ROOT/'baseline'/'pv_controllers_3'/'data'/'mppt_vref_profile.csv'
    d=pd.read_csv(p); knots=d['V_ref'].to_numpy(float)
    t=np.arange(N)*DT; tk=np.arange(len(knots))*1e-3
    v=np.interp(t,tk,knots); idx=np.minimum((t/1e-3).astype(int),len(knots)-2)
    slope=(knots[idx+1]-knots[idx])/1e-3
    return v,slope

def run(name,seed=7,noise_std=0.0):
    rng=np.random.default_rng(seed); vref,dref=reference(); G=np.ones(N)
    # Keep supplied fault timing and magnitudes from core.py exactly.
    out=[]; controller=ASMC() if name=='asmc' else NNMPC(str(ROOT/'baseline'/'pv_controllers_3'/'nnmpc_weights_H32.npz'))
    vp=float(vref[0]); il=pv_i(vp,1.); vo=VBUS; duty=float(np.clip(1-vp/vo,.02,.9))
    estimator=OnlineEstimator(DT); twin=BoostTwin(DT,vp,il,vo)
    params={'L_H':L0,'Cin_F':CI0,'Cout_F':CO0,'Iload_A':ILOAD}
    for k in range(N):
        t=k*DT; g=G[k]
        L=L0*(.70 if t>=.70*T_END else 1.0)
        Cin=CI0*(.50 if t>=.40*T_END else 1.0)
        Cout=CO0*(.60 if t>=.82*T_END else 1.0)
        load=ILOAD+(0.45 if .55*T_END<=t<=.61*T_END else 0.0)
        ipv=pv_i(vp,g)
        if k%HOLD==0:
            duty=controller.duty(vref[k],vp,vo,il,TS,ipv)
            duty=float(np.clip(duty,.02,.90))
        # plant integration with the actual hidden fault parameters
        il=float(np.clip(il+DT*(vp-(1-duty)*vo)/L,0,5))
        vp=float(np.clip(vp+DT*(ipv-il)/Cin,0,VOC))
        vo=float(np.clip(vo+DT*((1-duty)*il-load)/Cout,20,90))
        # Estimator receives measured/noisy signals only; never receives L/C/load truth.
        meas=np.array([vp,il,vo,duty,ipv])+rng.normal(0,noise_std,5)
        params=estimator.update(*meas)
        twin.calibrate(params); pred=twin.step(duty,g)
        if k%HOLD==0:
            out.append({'time_s':t,'Vpv':vp,'IL':il,'Vout':vo,'Duty':duty,'Ipv':ipv,
              **pred,'L_true_H':L,'Cin_true_F':Cin,'Cout_true_F':Cout,'Iload_true_A':load,
              'L_est_H':params['L_H'],'Cin_est_F':params['Cin_F'],'Cout_est_F':params['Cout_F'],
              'Iload_est_A':params['Iload_A'],'estimator_confidence':estimator.confidence,
              'PV_residual':vp-pred['Vpv_twin'],'IL_residual':il-pred['IL_twin'],'Vout_residual':vo-pred['Vout_twin']})
    df=pd.DataFrame(out)
    folder=ROOT/'results'/name; folder.mkdir(parents=True,exist_ok=True)
    df.to_csv(folder/'twin_log.csv',index=False)
    summary={'controller':name.upper(),'samples':len(df),'PV_tracking_RMSE_V':float(np.sqrt(np.mean((df.Vpv-np.interp(df.time_s,np.arange(N)*DT,vref))**2))),
      'twin_PV_RMSE_V':float(np.sqrt(np.mean(df.PV_residual**2))),'twin_IL_RMSE_A':float(np.sqrt(np.mean(df.IL_residual**2))),
      'twin_bus_RMSE_V':float(np.sqrt(np.mean(df.Vout_residual**2))),'final_L_est_mH':float(df.L_est_H.iloc[-1]*1e3),
      'final_Cin_est_uF':float(df.Cin_est_F.iloc[-1]*1e6),'final_Cout_est_uF':float(df.Cout_est_F.iloc[-1]*1e6)}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2)); return df
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser(); p.add_argument('--controller',choices=['asmc','nnmpc','both'],default='both'); p.add_argument('--seed',type=int,default=7); p.add_argument('--noise',type=float,default=0.0); a=p.parse_args()
 if a.controller in ('asmc','both'): run('asmc',a.seed,a.noise)
 if a.controller in ('nnmpc','both'): run('nnmpc',a.seed,a.noise)
