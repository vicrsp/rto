import numpy as np
#from src.model.williams_otto import Reactor 
#from gekko import GEKKO
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

def r_to_c(r):
   return (r - 491.67) * 5 / 9 

def c_to_r(c):
   return c * 9 /5 + 491.67

def process_equations_conc(w, t, controller, s):            
   Xa, Xb, Xc, Xe, Xg, Xp = w
   Fa, Fb, Tr = c.control(w, s.last_rates())
   Fr = Fa + Fb #kg/s

   Vr = 92.8 #ft*3

   k1 = 1.6599*(10**6)*np.exp(-12000/Tr) #s^-1
   k2 = 7.2117*(10**8)*np.exp(-15000/Tr) #s^-1
   k3 = 2.6745*(10**12)*np.exp(-20000/Tr) #s^-1

   df = [Fa - Fr * Xa - k1 * Xa * Xb * Vr,
      Fb - Fr * Xb - (k1 * Xa * Xb * Vr) - (k2 * Xb * Xc * Vr),
      (2*k1 * Xa * Xb * Vr) - (Fr * Xc) - (2*k2 * Xb* Xc * Vr) - (k3 * Xc* Xp *Vr),
      (2*k2 * Xb* Xc * Vr) - (Fr * Xe),
         (1.5 * k3 * Xc * Xp * Vr) - (Fr * Xg),
      (k2* Xb * Xc * Vr) -(Fr* Xp) -(0.5 * k3 * Xc *Xp * Vr)
      ]

   s.save(df)

   return df

class Saver:
   def __init__(self):
      self.rates = []

   def save(self, d):
       self.rates.append(d)
   
   def last_rates(self):
      if(len(self.rates) > 0):
         return self.rates[-1]
      return None

class Controller:
   def __init__(self, setpoints, Fa, Fb, Tr):
      self.setpoints = setpoints
      self.Fa = Fa
      self.Fb = Fb
      self.Tr = Tr
      self.ub = []
      self.lb = []
      self.results = []

   def control(self, X, dX):
      if dX is not None:
         dXa, dXb, dXc, dXe, dXg, dXp = dX
         Xa, Xb, Xc, Xe, Xg, Xp = X
         spa, spb, spc, spe, spg, spp = self.setpoints

         if(spa > 0):
            if(Xa > spa):
               if(dXa > 0):
                  self.Fb = self.Fb * 1.05
               if(np.abs(dXa) < 0.001):
                  self.Fb = min(self.Fb + 0.1, 7)
                  self.Tr = max(self.Tr - 1, 70)
            else:
               if(dXa < 0):
                  self.Fb = self.Fb * 0.95
               if(np.abs(dXa) < 0.001):
                  self.Fb = max(self.Fb - 0.1, 3)
                  self.Tr = min(self.Tr + 1, 110)
            
         if(spg > 0):
            if(Xg > spg):
               if(dXg > 0):
                  self.Tr = max(0.95 * self.Tr, 70)
            else:
               if(dXg < 0):
                  self.Tr = min(1.05 * self.Tr, 100)
      self.results.append([self.Fa, self.Fb, c_to_r(self.Tr)])
      return [self.Fa, self.Fb, c_to_r(self.Tr)]

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 10.0
numpoints = 200

# Create the time samples for the output of the ODE solver.
# I use a large number of points, only because I want to make
# a plot of the solution that looks nice.
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

# Pack up the parameters and initial conditions:
w0 = [0.5, 0.5, 0, 0, 0, 0]
# Call the ODE solver.


s = Saver()
c = Controller([0.12, -1, -1, -1, 0.08, -1], 2, 4, 90)
wsol = odeint(process_equations_conc, w0, t, args=(c, s),
              atol=abserr, rtol=relerr)

plt.figure()
plt.plot(t,wsol[:,0])
plt.plot(t,wsol[:,1])
#plt.legend(['Xa','Xb'])

#plt.figure()
plt.plot(t, wsol[:,2])
plt.plot(t, wsol[:,3])
plt.plot(t, wsol[:,4])
plt.plot(t, wsol[:,5])
#plt.legend(['Xc','Xe','Xg','Xp'])
plt.legend(['Xa','Xb','Xc','Xe','Xg','Xp','Sum'])