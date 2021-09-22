import os
import sys
import numpy as np
from scipy.integrate import odeint, solve_ivp
import math
from .base import ProcessModel

CX_INDEX = 0
CN_INDEX = 1
CP_INDEX = 2

class BioReactor(ProcessModel):
    def __init__(self, y0=[1., 150., 0.], k=[0.0923 * 0.62, 178.85, 447.12, 393.10, 0.001, 504.49, 2.544 * 0.62 * 1e-4, 23.51, 800.0, 0.281, 16.89]):
        super().__init__('Bio-Reactor', y0, k)
        self.stoptime = 240
        self.numpoints = 6
        self.t = np.array([self.stoptime * float(i) / (self.numpoints - 1) for i in range(self.numpoints)])

    def set_parameters(self, k):
        super().set_parameters(k)

    def odecallback(self, t, w, x):
        Cx, Cn, Cp, = w
        step_number = np.argmin(np.abs(self.t - t))
        L, Fn = x[step_number], x[step_number+1]

        # variable
        u_m, k_s, k_i, K_N, u_d, Y_nx, k_m, k_sq, k_iq, k_d, K_Np = self.k

        # Process model
        dx = u_m * L / (L + k_s + L ** 2. / k_i) * Cx * Cn / (Cn + K_N) - u_d * Cx
        dn = - Y_nx * u_m * L / (L + k_s + L ** 2. / k_i) * Cx * Cn / (Cn + K_N) + Fn
        dp = k_m * L / (L + k_sq + L ** 2. / k_iq) * Cx - k_d * Cp / (Cn + K_Np)

        df = [dx, dn, dp]
        return df

    def simulate(self, x):
        xnew = x
        return solve_ivp(fun=self.odecallback, t_span=[0, self.stoptime], t_eval=self.t, y0=self.y0, args=(xnew,))

    def get_objective(self, x, noise=None):
        sim_results = self.simulate(x)
        Cp_tf = sim_results.y[CP_INDEX][-1]
        fx = Cp_tf
        return fx if noise == None else fx * (1 + np.random.normal(scale=noise))

    def get_constraints(self, x, noise=None):
        sim_results = self.simulate(x)
        g0_trajectory = []
        g1_trajectory = []

        for index in range(self.numpoints):
            Cx_t = sim_results.y[CX_INDEX][index]
            Cn_t = sim_results.y[CN_INDEX][index]
            Cp_t = sim_results.y[CP_INDEX][index]
            g0_trajectory.append(Cp_t - 0.011* Cx_t)
            g1_trajectory.append(Cn_t)
        
        Cn_tf = sim_results.y[CN_INDEX][-1]
        g = np.array([*g0_trajectory, *g1_trajectory, Cn_tf] )
    
        if(noise != None):
            g = g * (1 + np.random.normal(scale=noise))
        return g