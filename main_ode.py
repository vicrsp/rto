import numpy as np
#from src.model.williams_otto import Reactor 
#from gekko import GEKKO
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def process_equations(w, t, x, k):            
        #eff = 0.1
        split = 0.5
        pho = 1
        
        ma, mb, mc, me, mg, mp = w
        Fa, Fb, Tr = x
        m = ma + mb + mc + me + mg + mp
        
        if(t > 3):
           Tr = 500

        if(t > 6):
           Tr = 600

        Fr = Fa + Fb

        V = m / pho
        #eta = 0.0
        #mu = (ma + mb)/ V
        #mu = 129.5

        
        k1 = 1.6599*(10**6)*np.exp(-6666.7/Tr)
        k2 = 7.2117*(10**8)*np.exp(-8333.3/Tr)
        k3 = 2.6745*(10**12)*np.exp(-11111/Tr)

        # df = [Fa + ((1-eta)*mu - mu) * ma / m - k1 * ma * mb / V,
        #     Fb + ((1-eta)*mu - mu) * mb / m - (k1 * ma * mb / V) - (k2 * mb* mc / V),
        #     ((1-eta)*mu - mu) * mc / m + (2*k1 * ma * mb / V) - (2*k2 * mb* mc / V) - (k3*mc*mp/V),
        #     ((1-eta)*mu - mu) * me / m + (2*k2 * mb* mc / V),
        #     (eff * (1-eta)* mu * me / m) - mu*mp/m + k2*mb*mc / V - split * k3*mc*mp/V,
        #     -mu*mg/m + (1+split)*k3*mc*mp/V]

        df = [Fa - Fr * ma / m - k1 * ma * mb / V,
            Fb - Fr * mb / m - (k1 * ma * mb / V) - (k2 * mb * mc / V),
            (2*k1 * ma * mb / V) - (Fr*mc/m) - (2*k2 * mb* mc / V) - (k3*mc*mp/V),
            (2*k2 * mb* mc / V) - (Fr * me / m),
             (1.5 * k3 * mc * mp / V) - (Fr * mg / m),
            (k2*mb*mc / V) -(Fr*mp/m) -(0.5 * k3 * mc *mp / V)
           ]

        return df

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 10.0
numpoints = 100

# Create the time samples for the output of the ODE solver.
# I use a large number of points, only because I want to make
# a plot of the solution that looks nice.
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

# Pack up the parameters and initial conditions:
x = [2, 4, 550]
w0 = [10, 10, 0, 0, 0, 0]
k = np.zeros_like(t)

# Call the ODE solver.
wsol = odeint(process_equations, w0, t, args=(x,k),
              atol=abserr, rtol=relerr)


plt.plot(wsol[:,0])
plt.plot(wsol[:,1])
plt.plot(wsol[:,2])
plt.plot(wsol[:,3])
plt.plot(wsol[:,4])
plt.plot(wsol[:,5])
plt.legend(['ma','mb','mc','me','mg','mp'])