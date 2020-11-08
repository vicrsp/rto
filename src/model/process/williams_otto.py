import numpy as np
from gekko import GEKKO
import json

## Forbes, J. F., & Marlin, T. E. (1996). 
# Design cost: a systematic approach to technology selection for model-based real-time optimization systems.
# Computers & Chemical Engineering, 20(6-7), 717–734. doi:10.1016/0098-1354(95)00205-7 
# x (manipulated variables): Fb, Tr
# u (dependent variables): Xp, Xe, Xg
# Fr = Fa + Fb (mass balance)
# Fa and Vr are not manipulated
# Known optimum: Tr:89.647ºC, Fb:4.7836kg/s

class Reactor:
    # Initialize the reactor model
    def __init__(self):
        #self.a = np.array([1.6599*10^6, 7.2117*10^6, 2,6745*10^6]) # frequency factors
        #self.b = np.array([6666.7, 8333.3, 11111]) # activation energies
        #self.P = np.array([5554.10, 125.91, -370.30, -555.42])
        #self.Vr = 1 # tank volume
        #self.Fa = 1.8725
        pass

    def r_to_c(self, r):
        return (r - 491.67) * 5 / 9 

    def c_to_r(self,c):
        return c * 9/5 + 491.67
    
    def k_to_r(self,k):
        return self.c_to_r(k - 273.15)

    ## Gets the model in GEKKO format
    def getModel(self):
        # initialize model
        model = GEKKO()
       
        # manipulated variable
        Fa = model.MV(value=1.8725, lb=1.5, ub=3, name='Fa')
        Fb = model.MV(value=6.9, lb=4, ub=7, name='Fb')
        Tr = model.MV(value=self.c_to_r(86), lb=self.c_to_r(80), ub=self.c_to_r(95), name='Tr')

        # MV allowed to change
        Fa.STATUS = 0
        Fb.STATUS = 1
        Tr.STATUS = 1

        #Fa.DCOST = 0.01 # smooth out gas pedal movement
        Fa.DMAX = 0.1   # slow down change of gas pedal

        #Fb.DCOST = 0.01 # smooth out gas pedal movement
        Fb.DMAX = 0.1   # slow down change of gas pedal

        Tr.DCOST = 0.01 # smooth out gas pedal movement
        Tr.DMAX = 10   # slow down change of gas pedal

        # controlled variables
        Xa = model.CV(value=0.1, name='Xa', lb=0, ub=1)
        Xb = model.CV(value=0.9, name='Xb', lb=0, ub=1)
        Xc = model.CV(value=0.0, name='Xc', lb=0, ub=1)
        Xe = model.CV(value=0.0, name='Xe', lb=0, ub=1)
        Xp = model.CV(value=0.0, name='Xp', lb=0, ub=1)
        Xg = model.CV(value=0.0, name='Xg', lb=0, ub=1)

        Xa.STATUS = 1  # add the SP to the objective
        Xa.SP = 0.2     # set point
        Xa.TR_INIT = 1 # set point trajectory
        Xa.TAU = 1     # time constant of trajectory

        Xb.STATUS = 1  # add the SP to the objective
        Xb.SP = 0.8     # set point
        #Xb.TR_INIT = 1 # set point trajectory
        #Xb.TAU = 1     # time constant of trajectory

        #Xg.STATUS = 1  # add the SP to the objective
        #Xg.SP = 0.05     # set point
        #Xg.TR_INIT = 1 # set point trajectory
        #Xg.TAU = 10     # time constant of trajectory

        # parameters
        # fixed
        split = model.Const(value=0.5)
        Vr = model.Const(value=90)

        # variable
        k1 = model.Intermediate(1.6599*(10**6)*model.exp(-12000/Tr), name='k1') #m3/kg*h
        k2 = model.Intermediate(7.2117*(10**8)*model.exp(-15000/Tr), name='k2') #m3/kg*h
        k3 = model.Intermediate(2.6745*(10**12)*model.exp(-20000/Tr), name='k3') #m3/kg*h
        Fr = model.Intermediate(Fa + Fb, name='Fr') #Kg/s

        # Process model
        model.Equation(Xa.dt() == Fa - Fr * Xa - k1 * Xa * Xb * Vr)
        model.Equation(Xb.dt() == Fb - Fr * Xb - (k1 * Xa * Xb * Vr) - (k2 * Xb * Xc * Vr))
        model.Equation(Xc.dt() == (2*k1 * Xa * Xb * Vr) - (Fr * Xc) - (2*k2 * Xb* Xc * Vr) - (k3 * Xc* Xp *Vr))
        model.Equation(Xe.dt() == (2*k2 * Xb* Xc * Vr) - (Fr * Xe))
        model.Equation(Xg.dt() ==  ((1.5) * k3 * Xc * Xp * Vr) - (Fr * Xg))
        model.Equation(Xp.dt() == (k2* Xb * Xc * Vr) -(Fr* Xp) -(0.5 * k3 * Xc *Xp * Vr))
        
        # Objective
        model.Obj(25.2*Fr + 0.5712*Fr*Xp*Xe -1.68*Fa - 2.52*Fb)
        
        return model

    def solveMPC(self, model):
        model.time = np.linspace(0,10,200)
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
