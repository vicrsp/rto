import numpy as np
#from src.model.williams_otto import Reactor 
#from gekko import GEKKO
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def process_equations(w, t, x):            
        eff = 0.1
        split = 0.5
        pho = 50 
        
        ma, mb, mc, me, mp, mg = w
        Fa, Fb, Tr = x
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
            ((1-eta)*mu - mu) * me / m + (2*k2 * mb* mc / V),
            (eff * (1-eta)* mu * me / m) - mu*mp/m + k2*mb*mc / V - split * k3*mc*mp/V,
            -mu*mg/m + (1+split)*k3*mc*mp/V]

        return f

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 20.0
numpoints = 250

# Create the time samples for the output of the ODE solver.
# I use a large number of points, only because I want to make
# a plot of the solution that looks nice.
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

# Pack up the parameters and initial conditions:
x = [10, 20, 580]
w0 = [10, 1, 0, 0, 0, 0]

# Call the ODE solver.
wsol = odeint(process_equations, w0, t, args=(x,),
              atol=abserr, rtol=relerr)

