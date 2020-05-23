import numpy as np
from gekko import GEKKO
import json

class SemiBatchReactor:
    def __init__(self):
        pass

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
