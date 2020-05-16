import numpy as np
from gekko import GEKKO

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

        
    ## Gets the reaction rate
    ## k = a*e^(-b/Tr)
    def getK(self, Tr):
        return np.multiply(self.a, np.exp(-self.b/Tr))

    ## Gets the parameter vector
    def getBeta(self):
        return np.array([self.a.reshape(-1,1), self.b.reshape(-1,1)])
    
    ## Gets the cost value
    def getCost(self, Fb, Xp, Xe, Fr = 0, Fa = 1.8725):
        return np.dot(self.P , np.array([Xp*Fr, Xe*Fr, Fa, Fb]))

    
    def getApproximateModelConstraints(self, Tr, Fb):
        Fr = self.Fa + Fb
        k = self.getK(Tr)
        
        step = 1.0 #s
        Xa = self.Fa * step / self.Vr
        Xb = Fb * step / self.Vr

        k1 = k[0]
        k2 = k[1]
        k3 = k[2]

        Xp = (self.Fa - Fr*Xa -k1*self.Vr*Xa*(Xb^2)) / (k2*self.Vr*Xa*Xb)
        Xe = (2*k1*self.Vr*Xa*(Xb^2)) / Fr
        Xg = (3*k2*self.Vr*Xa*Xb*Xp) / Fr



    
    def getExactModelConstraints(self, Tr, Fb):
        Fr = self.Fa + Fb
        k = self.getK(Tr)

        step = 1.0 #s
        Xa = self.Fa * step / self.Vr
        Xb = Fb * step / self.Vr

        k1 = k[0]
        k2 = k[1]
        k3 = k[2]
 

    ## Gets the model in GEKKO format
    def getModel(self):
        # initialize model
        model = GEKKO()
       
        # manipulated variable
        Fa = model.MV(value=10, lb=5, ub=15, name='Fa')
        Fb = model.MV(value=20, lb=15, ub=25, name='Fb')
        Tr = model.MV(value=580, lb=560, ub=600, name='Tr')

        # MV allowed to change
        Fa.STATUS = 1
        Fb.STATUS = 1
        Tr.STATUS = 1

        #Fb.DCOST = 0.1 # smooth out gas pedal movement
        #Fb.DMAX = 20   # slow down change of gas pedal


        #Tr.DCOST = 0.001 # smooth out gas pedal movement
        #Tr.DMAX = 1   # slow down change of gas pedal

        # controlled variables
        ma = model.CV(value=10.0, name='ma')
        mb = model.CV(value=1.0, name='mb')
        mc = model.CV(value=0.0, name='mc')
        me = model.CV(value=0.0, name='me')
        mp = model.CV(value=0.0, name='mp')
        mg = model.CV(value=0.0, name='mg')
        eta = model.CV(value=0.2, lb=0.0, ub=1.0, name='eta')
        mu = model.CV(value=129.5, lb=100, ub=150, name='mu')

        ma.STATUS = 1  # add the SP to the objective
        ma.SP = 8     # set point
        ma.TR_INIT = 1 # set point trajectory
        ma.TAU = 10     # time constant of trajectory

        mg.STATUS = 1  # add the SP to the objective
        mg.SP = 2     # set point
        #mg.TR_INIT = 1 # set point trajectory
        #mg.TAU = 10     # time constant of trajectory

        # parameters
        # fixed
        eff = model.Const(value=0.1)
        split = model.Const(value=0.5)
        pho = model.Const(value=40) #800.923kg/m3 = 50lb/ft3

        # variable
        k1 = model.Intermediate(5.9755*(10^9)*model.exp(-12000/Tr)/pho, name='k1') #m3/kg*h
        k2 = model.Intermediate(2.5962*(10^12)*model.exp(-15000/Tr)/pho, name='k2') #m3/kg*h
        k3 = model.Intermediate(9.6283*(10^15)*model.exp(-20000/Tr)/pho, name='k3') #m3/kg*h
        m = model.Intermediate(ma + mb + mc + me + mp + mg, name='m')
        V = model.Intermediate(m / pho, name='V')

        # Process model
        model.Equation(ma.dt() == Fa + ((1-eta)*mu - mu) * ma / m - k1 * ma * mb / V)
        model.Equation(mb.dt() == Fb + ((1-eta)*mu - mu) * mb / m - (k1 * ma * mb / V) - (k2 * mb* mc / V))
        model.Equation(mc.dt() == ((1-eta)*mu - mu) * mc / m + (2*k1 * ma * mb / V) - (2*k2 * mb* mc / V) - (k3*mc*mp/V))
        model.Equation(me.dt() == ((1-eta)*mu - mu) * me / m + (2*k2 * mb* mc / V))
        model.Equation(mp.dt() == (eff * (1-eta)* mu * me / m) - mu*mp/m + k2*mb*mc / V - split * k3*mc*mp/V)
        model.Equation(mg.dt() == -mu*mg/m + (1+split)*k3*mc*mp/V)
        #m.Equation(m == ma + mb + mc + me + mp + mg)        

        return model


    def getModelODE(self, w, t, p):        
        
        eff = 0.1
        split = 0.5
        pho = 40 #800.923kg/m3 = 50lb/ft3
        
        ma, mb, mc, me, mp, mg, Fa, Fb, Tr  = w
        m = ma + mb + mc + me + mp + mg
        V = m / pho
        eta = 0.2
        mu = 129.5

        k1 = 5.9755*(10^9)*np.exp(-12000/Tr)/pho
        k2 = 2.5962*(10^12)*np.exp(-15000/Tr)/pho
        k3 = 9.6283*(10^15)*np.exp(-20000/Tr)/pho

        f = [Fa + ((1-eta)*mu - mu) * ma / m - k1 * ma * mb / V,
            Fb + ((1-eta)*mu - mu) * mb / m - (k1 * ma * mb / V) - (k2 * mb* mc / V),
            ((1-eta)*mu - mu) * mc / m + (2*k1 * ma * mb / V) - (2*k2 * mb* mc / V) - (k3*mc*mp/V),
            (eff * (1-eta)* mu * me / m) - mu*mp/m + k2*mb*mc / V - split * k3*mc*mp/V,
            -mu*mg/m + (1+split)*k3*mc*mp/V]

        return f


