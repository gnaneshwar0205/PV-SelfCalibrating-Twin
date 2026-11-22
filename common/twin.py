import numpy as np
class BoostTwin:
    def __init__(self,dt=20e-6,vp0=20.0,il0=0.0,vo0=65.0):
        self.dt=dt; self.vp=float(vp0); self.il=float(il0); self.vo=float(vo0)
        self.L=5e-3; self.Cin=470e-6; self.Cout=220e-6; self.Iload=1.73
        self.VOC=57.6; self.ISC=2.9; self.SHAPE=8.0
    def calibrate(self,p):
        self.L=p['L_H']; self.Cin=p['Cin_F']; self.Cout=p['Cout_F']; self.Iload=p['Iload_A']
    def step(self,duty,g):
        x=np.clip(self.vp/self.VOC,0,1); ipv=max(0.,self.ISC*g*(1-x**self.SHAPE))
        old=(self.vp,self.il,self.vo)
        dil=(self.vp-(1-duty)*self.vo)/self.L
        dvp=(ipv-self.il)/self.Cin
        dvo=((1-duty)*self.il-self.Iload)/self.Cout
        self.il=np.clip(self.il+self.dt*dil,0,5)
        self.vp=np.clip(self.vp+self.dt*dvp,0,self.VOC)
        self.vo=np.clip(self.vo+self.dt*dvo,20,90)
        return {'Vpv_twin':self.vp,'IL_twin':self.il,'Vout_twin':self.vo}
