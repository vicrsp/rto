import os
import sys
import numpy as np
from scipy.integrate import odeint, solve_ivp

from .base import ProcessModel

CA_INDEX = 0
CB_INDEX = 1
CC_INDEX = 2
CD_INDEX = 3
V_INDEX = 4


class SemiBatchReactor(ProcessModel):
    def __init__(self, y0=[0.72, 0.05, 0.08, 0.01, 1.0], k=[0.053, 0.128, 0.028, 0.001, 5]):
        super().__init__('Semi-Batch', y0, k)
        self.stoptime = 250
        self.numpoints = 100
        self.g = [0.025, 0.15]

    def set_parameters(self, k):
        super().set_parameters(k)

    def odecallback(self, t, w, x):
        Ca, Cb, Cc, Cd, V = w
        F0, tm, Fs, ts, Fmin = x
        F = F0
        if(t > tm):
            F = Fs

        if(t > ts):
            F = Fmin

        # variable
        k1, k2, k3, k4, Cb_in = self.k
        Cb_in = 5

        # Process model
        df = [
            -k1*Ca*Cb - F*Ca/V,
            -k1*Ca*Cb - 2*k2*Cb*Cb - k3*Cb - k4*Cb*Cc + F*(Cb_in - Cb)/V,
            k1*Ca*Cb - k4*Cb*Cc - F*Cc/V,
            k2*Cb*Cb - F*Cd/V,
            F]

        return df

    def simulate(self, x):
        t = [self.stoptime * float(i) / (self.numpoints - 1)
             for i in range(self.numpoints)]
        xnew = [0.002, x[0], x[1], x[2], 0]
        return solve_ivp(fun=self.odecallback, t_span=[0, self.stoptime], t_eval=t, y0=self.y0, args=(xnew,))

    def get_objective(self, x, noise=None):
        sim_results = self.simulate(x)
        Cc_tf = sim_results.y[2][-1]
        V_tf = sim_results.y[4][-1]
        fx = Cc_tf * V_tf
        return -fx if noise == None else -fx * (1 + np.random.normal(scale=noise))

    def get_constraints(self, x, noise=None):
        sim_results = self.simulate(x)
        Cb_tf = sim_results.y[1][-1]
        Cd_tf = sim_results.y[3][-1]

        if(noise != None):
            Cb_tf = Cb_tf * (1 + np.random.normal(scale=noise))
            Cd_tf = Cd_tf * (1 + np.random.normal(scale=noise))

        return np.asarray([Cb_tf, Cd_tf])
