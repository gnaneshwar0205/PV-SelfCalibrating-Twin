import numpy as np

class RLS:
    def __init__(self, theta0, P0=1e8, forgetting=0.9995):
        self.theta=np.asarray(theta0,dtype=float).reshape(-1,1)
        self.P=np.eye(len(self.theta))*P0
        self.lam=forgetting
        self.updates=0
    def update(self, phi, y):
        phi=np.asarray(phi,dtype=float).reshape(-1,1)
        if not np.all(np.isfinite(phi)) or not np.isfinite(y): return self.theta.ravel()
        pphi=self.P@phi; den=self.lam+float(phi.T@pphi)
        if den<=1e-15: return self.theta.ravel()
        K=pphi/den; err=float(y)-float(phi.T@self.theta)
        self.theta=self.theta+K*err
        self.P=(self.P-K@phi.T@self.P)/self.lam
        self.P=(self.P+self.P.T)/2
        self.updates+=1
        return self.theta.ravel()

class OnlineEstimator:
    """Estimates inverse L, inverse Cin and [1/Cout, Iload/Cout]."""
    def __init__(self, dt=20e-6, forgetting=0.9995):
        self.dt=dt
        self.L=RLS([1/5e-3],forgetting=forgetting)
        self.Ci=RLS([1/470e-6],forgetting=forgetting)
        self.Co=RLS([1/220e-6,1.73/220e-6],forgetting=forgetting)
        self.prev=None; self.n=0
        self.low={'L_H':1e-3,'Cin_F':100e-6,'Cout_F':50e-6,'Iload_A':0.1}
        self.high={'L_H':10e-3,'Cin_F':1000e-6,'Cout_F':600e-6,'Iload_A':5.0}
        self.accepted={'L_H':5e-3,'Cin_F':470e-6,'Cout_F':220e-6,'Iload_A':1.73}
        self.last_raw=self.accepted.copy(); self.confidence=0.0
    def update(self,vp,il,vo,d,ipv):
        x=np.array([vp,il,vo,d,ipv],float)
        if not np.all(np.isfinite(x)): return self.accepted.copy()
        if self.prev is None: self.prev=x; return self.accepted.copy()
        a=self.prev; dt=self.dt
        dvp=(x[0]-a[0])/dt; dil=(x[1]-a[1])/dt; dvo=(x[2]-a[2])/dt
        vp_m=(x[0]+a[0])/2; il_m=(x[1]+a[1])/2; vo_m=(x[2]+a[2])/2
        d_m=(x[3]+a[3])/2; ipv_m=(x[4]+a[4])/2
        phL=vp_m-(1-d_m)*vo_m; phCi=ipv_m-il_m; phCo=(1-d_m)*il_m
        # Excitation gates prevent updates around near-zero regressors.
        if abs(phL)>0.05: self.L.update([phL],dil)
        if abs(phCi)>0.05: self.Ci.update([phCi],dvp)
        if abs(phCo)>0.05: self.Co.update([phCo,-1.0],dvo)
        self.prev=x; self.n+=1
        invL=self.L.theta[0,0]; invCi=self.Ci.theta[0,0]
        invCo,loadOver=self.Co.theta.ravel()
        raw=self.accepted.copy()
        if invL>0: raw['L_H']=1/invL
        if invCi>0: raw['Cin_F']=1/invCi
        if invCo>0:
            raw['Cout_F']=1/invCo; raw['Iload_A']=loadOver/invCo
        self.last_raw=raw.copy()
        # Conservative plausibility gate: max 5% change per sample, physical bounds.
        for k,v in raw.items():
            if np.isfinite(v) and self.low[k]<=v<=self.high[k]:
                old=self.accepted[k]; maxstep=max(abs(old)*0.05, self.low[k]*0.01)
                self.accepted[k]=float(np.clip(v,old-maxstep,old+maxstep))
        self.confidence=float(min(1.0,(self.n/2000.0)))
        return self.accepted.copy()
