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

        df = [Fa - Fr * ma / m - k1 * ma * mb / V,
            Fb - Fr * mb / m - (k1 * ma * mb / V) - (k2 * mb * mc / V),
            (2*k1 * ma * mb / V) - (Fr*mc/m) - (2*k2 * mb* mc / V) - (k3*mc*mp/V),
            (2*k2 * mb* mc / V) - (Fr * me / m),
             (1.5 * k3 * mc * mp / V) - (Fr * mg / m),
            (k2*mb*mc / V) -(Fr*mp/m) -(0.5 * k3 * mc *mp / V)
           ]

        return df

def r_to_c(r):
   return (r - 491.67) * 5 / 9 

def c_to_r(c):
   return c * 9 /5 + 491.67

def process_equations_conc(w, t, controller):            
   Xa, Xb, Xc, Xe, Xg, Xp = w
   Fa, Fb, Tr = controller(w, t)

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

   return df

def process_equations_aprox(w, t, controller):            
   Xa, Xb, Xc, Xe, Xg, Xp = w
   Fa, Fb, Tr = controller(w, t)

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

   return df

def controller(X, t):
   x = [2, 4, c_to_r(89)] #Fa [kg/s], Fb [kg/s], Tr [Rankine degrees]
   if(t > 2):
      x = [1, 4, c_to_r(70)]

   if(t > 3):
      x = [2, 3, c_to_r(100)]

   if(t > 4):
      x = [0.5, 4, c_to_r(50)]


   return x

# ODE solver parameters
abserr = 1.0e-12
relerr = 1.0e-8
stoptime = 5.0
numpoints = 200

# Create the time samples for the output of the ODE solver.
# I use a large number of points, only because I want to make
# a plot of the solution that looks nice.
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

# Pack up the parameters and initial conditions:
w0 = [0.9, 0.1, 0, 0, 0, 0]
k = np.zeros_like(t)

# Call the ODE solver.
wsol = odeint(process_equations_conc, w0, t, args=(controller,),
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