import numpy as np
from gekko import GEKKO
import json
from scipy.integrate import odeint, solve_ivp


CB_INDEX = 1
CC_INDEX = 2
CD_INDEX = 3
V_INDEX = 4

class SemiBatchReactor:
    def __init__(self, y0 = [0.72, 0.05, 0.08, 0.01, 1.0], k = [0.053, 0.128, 0.028, 0.001, 5]):
        self.y0 = y0
        self.k1, self.k2, self.k3, self.k4, self.Cb_in = k
        self.stoptime = 250
        self.numpoints = 100

    def fobj(self, x):
        sim_results = self.solveIVP(x)
        Cc_tf = sim_results.y[CC_INDEX][-1]
        V_tf = sim_results.y[V_INDEX][-1]
        Cb_tf = sim_results.y[CB_INDEX][-1]
        Cd_tf = sim_results.y[CD_INDEX][-1]

        f = Cc_tf * V_tf
        g = [Cb_tf, Cd_tf]

        return f, g

    def sim(self, t, w, x):
        Ca, Cb, Cc, Cd, V = w
        F0, tm, Fm, ts, Fs = x 
        F = F0
        if(t > tm):
            F = Fm
        
        if(t > ts):
            F = Fs

        # variable
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4
        Cb_in = 5

        # Process model
        df = [
        -k1*Ca*Cb - F*Ca/V,
        -k1*Ca*Cb -2*k2*Cb*Cb -k3*Cb -k4*Cb*Cc + F*(Cb_in - Cb)/V,
         k1*Ca*Cb - k4*Cb*Cc - F*Cc/V,
         k2*Cb*Cb - F*Cd/V,
         F]

        return df

    def solveIVP(self, x):
        t = [self.stoptime * float(i) / (self.numpoints - 1) for i in range(self.numpoints)]
        return solve_ivp(fun=self.sim, t_span=[0, self.stoptime], t_eval=t, y0=self.y0, args=(x,))
    
    def solveODE(self, abserr=1.0e-8, relerr=1.0e-6, stoptime=250, numpoints=250):
        t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
        # Pack up the parameters and initial conditions:
        w0 = [0.72, 0.05, 0.08, 0.01, 1.0]
        wsol = odeint(self.sim, w0, t, atol=abserr, rtol=relerr)

        return wsol

    def getModel(self):
        # initialize model
        model = GEKKO()
       
        # manipulated variable
        F = model.MV(value=0.002, lb=0.0, ub=0.002, name='F')
        F.STATUS = 1

        # controlled variables
        Ca = model.CV(value=0.72, name='Ca', lb=0, ub=1)
        Cb = model.CV(value=0.05, name='Cb', lb=0.02, ub=0.025)
        Cc = model.CV(value=0.08, name='Cc', lb=0, ub=1)
        Cd = model.CV(value=0.01, name='Cd', lb=0.001, ub=0.015)
        V = model.CV(value=1.0, name='V', lb=0, ub=1)

        Cb.STATUS = 1
        Cd.STATUS = 1

        # variable
        k1 = model.Const(value=0.053, name='k1')
        k2 = model.Const(value=0.128, name='k2')
        k3 = model.Const(value=0.028, name='k3') 
        k4 = model.Const(value=0.001, name='k4')
        Cb_in = model.Const(value=5, name='Cb_in')

        # Process model
        model.Equation(Ca.dt() == -k1*Ca*Cb - F*Ca/V)
        model.Equation(Cb.dt() == -k1*Ca*Cb -2*k2*Cb*Cb -k3*Cb -k4*Cb*Cc + F*(Cb_in - Cb)/V)
        model.Equation(Cc.dt() == k1*Ca*Cb - k4*Cb*Cc - F*Cc/V)
        model.Equation(Cd.dt() == k2*Cb*Cb - F*Cd/V)
        model.Equation(V.dt() == F)

        # Objective
        #model.Obj(-Cc*V)
        
        return model
    
    def solveMPC(self, model):
        model.time = np.linspace(0,100,100)
        model.options.CV_TYPE = 2 # squared error
        model.options.IMODE = 6 # control

        model.solve()

        with open(model.path+'//results.json') as f:
            results = json.load(f)
        
        return results

    def solveRTO(self, model):
        model.options.IMODE = 3 # control
        model.solve()

        with open(model.path+'//results.json') as f:
            results = json.load(f)
        
        return results
