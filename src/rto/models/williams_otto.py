import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize.minpack import fsolve
from .base import ProcessModel
# Forbes, J. F., & Marlin, T. E. (1996).
# Design cost: a systematic approach to technology selection for model-based real-time optimization systems.
# Computers & Chemical Engineering, 20(6-7), 717–734. doi:10.1016/0098-1354(95)00205-7
# x (manipulated variables): Fb, Tr
# u (dependent variables): Xp, Xe, Xg
# Fr = Fa + Fb (mass balance)
# Fa and Vr are not manipulated
# Known optimum: Tr:89.647ºC, Fb:4.7836kg/s


class WilliamsOttoReactor(ProcessModel):
    def __init__(self, y0=[1, 1, 1, 1, 1, 1], k=[1.6599e6, 7.2117e8, 2.6745e12]):
        super().__init__('Williams-Otto', y0, k)
        self.stoptime = 250
        self.numpoints = 100
        self.eta = [6666.7, 8333.3, 11111]

    def odecallback(self, t, w, x):
        xa, xb, xc, xp, xe, xg = w
        Fa = 1.8275
        Fb, Tr = x
        Fr = Fa + Fb

        Vr = 2105.2  # % 2.63; % Reactor mass holdup - constant
        # Tref = 110.0 + 273.15
        # phi1 = - 3.
        # psi1 = -17.
        # phi2 = - 4.
        # psi2 = -29.

        k1, k2, k3 = self.k
        eta1, eta2, eta3 = self.eta

        k1 = k1*np.exp(-eta1/(Tr + 273))  # s^-1
        k2 = k2*np.exp(-eta2/(Tr + 273))  # s^-1
        k3 = k3*np.exp(-eta3/(Tr + 273))  # s^-1

        # tau1 = 150
        # tauC = 150
        # Kp = tau1/(0.00517*tauC)
        # r1 = k1 * xa * xb * Vr
        # r2 = k2 * xb * xc * Vr
        # r3 = k3 * xc * xp * Vr

        df = [(Fa - (Fr)*xa - Vr*xa*xb*k1)/Vr,
              (Fb - (Fr)*xb - Vr*xa*xb*k1 - Vr*xb*xc*k2)/Vr,
              -(Fr)*xc/Vr + 2*xa*xb*k1 - 2*xb*xc*k2 - xc*xp*k3,
              -(Fr)*xp/Vr + xb*xc*k2 - 0.5*xp*xc*k3,
              -(Fr)*xe/Vr + 2*xb*xc*k2,
              -(Fr)*xg/Vr + 1.5*xp*xc*k3,
              ]

        return df

    def simulate(self, x):
        t = [self.stoptime * float(i) / (self.numpoints - 1)
             for i in range(self.numpoints)]
        xnew = x
        return solve_ivp(fun=self.odecallback, t_span=[0, self.stoptime], t_eval=t, y0=self.y0, args=(xnew,))

    def get_objective(self, u, noise=None):
        Fa = 1.8275
        Fb, _ = u
        _, _, _, xp, xe, _ = self.solve_steady_state(u)
        fx = -(1043.38*xp*(Fa+Fb)+20.92*xe*(Fa+Fb) - 79.23*Fa - 118.34*Fb)

        return fx if noise == None else fx * (1 + np.random.normal(scale=noise))

    def get_constraints(self, u, noise=None):
        xa, _, _, _, _, xg = self.solve_steady_state(u)
        g = np.array([xa, xg])
        if(noise != None):
            g = g * (1 + np.random.normal(scale=noise))

        return g
