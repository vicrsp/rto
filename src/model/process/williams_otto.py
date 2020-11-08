import numpy as np
from scipy.integrate import odeint, solve_ivp

## Forbes, J. F., & Marlin, T. E. (1996). 
# Design cost: a systematic approach to technology selection for model-based real-time optimization systems.
# Computers & Chemical Engineering, 20(6-7), 717–734. doi:10.1016/0098-1354(95)00205-7 
# x (manipulated variables): Fb, Tr
# u (dependent variables): Xp, Xe, Xg
# Fr = Fa + Fb (mass balance)
# Fa and Vr are not manipulated
# Known optimum: Tr:89.647ºC, Fb:4.7836kg/s
class WilliamsOttoReactor:
    def __init__(self, y0, k = [1.6599*(10**6), 7.2117*(10**8), 2.6745*(10**12)]):
        self.stoptime = 250
        self.numpoints = 100
        self.y0 = y0
        self.k = k

    def r_to_c(self, r):
        return (r - 491.67) * 5 / 9 

    def c_to_r(self, c):
        return c * 9/5 + 491.67

    def k_to_r(self, k):
        return self.c_to_r(k - 273.15)

    def odecallback(self, t, w, x):
        Xa, Xb, Xc, Xe, Xg, Xp = w
        Fa, Fb, Tr = x
        Fr = Fa + Fb #kg/s

        Vr = 2.0 #ft*3

        k1, k2, k3 = self.k

        k1 = k1*np.exp(-12000/Tr) #s^-1
        k2 = k2*np.exp(-15000/Tr) #s^-1
        k3 = k3*np.exp(-20000/Tr) #s^-1

        df = [Fa - Fr * Xa - k1 * Xa * Xb * Vr,
            Fb - Fr * Xb - (k1 * Xa * Xb * Vr) - (k2 * Xb * Xc * Vr),
            (2*k1 * Xa * Xb * Vr) - (Fr * Xc) - (2*k2 * Xb* Xc * Vr) - (k3 * Xc* Xp *Vr),
            (2*k2 * Xb* Xc * Vr) - (Fr * Xe),
                (1.5 * k3 * Xc * Xp * Vr) - (Fr * Xg),
            (k2* Xb * Xc * Vr) -(Fr* Xp) -(0.5 * k3 * Xc *Xp * Vr)
            ]

        return df

    def simulate(self, x):
        t = [self.stoptime * float(i) / (self.numpoints - 1)
             for i in range(self.numpoints)]
        xnew = [0.002, x[0], x[1], x[2], 0]
        return solve_ivp(fun=self.odecallback, t_span=[0, self.stoptime], t_eval=t, y0=self.y0, args=(xnew,))