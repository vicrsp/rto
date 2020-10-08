import numpy as np
from optimization.utils import build_F
from optimization.de import DifferentialEvolution


class ProfileOptimizer:
    # F0, tm, Fm, ts, Fs = x
    def __init__(self, ub=[0.002, 250, 0.002, 250, 0.002], lb=[0.0, 0, 0, 0, 0], g=[0.025, 0.15]):
        self.g = g
        self.lb = lb
        self.ub = ub
        self.xk = []
        self.fxk = []
        self.gk = []

    # x0=[0.002, 60, 0.001, 100, 0.0]
    def run(self, model):
        self.model = model

        best_fobj, sol = DifferentialEvolution(
            func=self.eval_objective, lb=self.lb, ub=self.ub, callback=self.save_results).run()
        return best_fobj, sol, self.xk, self.fxk, self.gk

    def save_results(self, x, fx, gx):
        self.xk.append(x)
        self.fxk.append(fx)
        self.gk.append(gx)

    def eval_objective(self, x):
        sim_results = self.model.simulate(x)
        Cb_tf = sim_results.y[1][-1]
        Cc_tf = sim_results.y[2][-1]
        Cd_tf = sim_results.y[3][-1]
        V_tf = sim_results.y[4][-1]

        fx = -Cc_tf * V_tf
        g = [Cb_tf - self.g[0], Cd_tf - self.g[1]]

        return fx, g

class ModelParameterOptimizer:
    def __init__(self, lb=[0.0011, 0.0026], ub=[0.212, 0.5120]):
        self.lb = lb
        self.ub = ub
        self.xk = []
        self.fxk = []

    #  x0=[0.053, 0.128]
    def run(self, model, input, samples):

        self.model = model
        self.samples = samples
        self.input = input

        best_fobj, sol = DifferentialEvolution(
            func=self.eval_objective, lb=self.lb, ub=self.ub, callback=self.save_results).run()
        return best_fobj, sol, self.xk, self.fxk

    def save_results(self, x, fx, gx):
        self.xk.append(x)
        self.fxk.append(fx)

    def eval_objective(self, x):
        sim_values = self.model.get_simulated_samples(
            self.input, x, self.samples)
        # Weight vector
        w = np.ones_like(self.input)

        # SSE
        error = 0
        for time, sim_value in sim_values.items():
            meas_value = self.samples[time]
            for i in range(len(meas_value)):
                if(i > 0 & i < 4):
                    error = error + \
                        w[i]*((meas_value[i] - sim_value[i])/meas_value[i])**2
        return error, []
