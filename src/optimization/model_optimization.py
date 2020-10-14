import numpy as np
from optimization.utils import calculate_SSE
from optimization.de import DifferentialEvolution

CA_INDEX = 0
CB_INDEX = 1
CC_INDEX = 2
CD_INDEX = 3
V_INDEX = 4


class ProfileOptimizer:
    # F0, tm, Fm, ts, Fs = x
    def __init__(self, ub=[250, 0.002, 250], lb=[0, 0, 0], g=[0.025, 0.15]):
        self.g = g
        self.lb = lb
        self.ub = ub


    # x0=[0.002, 60, 0.001, 100, 0.0]
    def run(self, model, max_generations=100, pop_size=20, de_type='rand/1/bin'):
        self.model = model
        self.xk = []
        self.fxk = []
        self.gk = []
        
        best_fobj, sol = DifferentialEvolution(
            func=self.eval_objective, lb=self.lb, ub=self.ub,
            callback=self.save_results, max_generations=max_generations, pop_size=pop_size, de_type=de_type).run(debug=False)
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
        g = [Cb_tf - self.g[0], Cd_tf - self.g[1], x[0] - x[2]]

        return fx, g


class ModelParameterOptimizer:
    def __init__(self, lb=[0.0011, 0.0026], ub=[0.212, 0.5120]):
        self.lb = lb
        self.ub = ub
        self.xk = []
        self.fxk = []

    #  x0=[0.053, 0.128]
    def run(self, model, input, samples, max_generations=100, pop_size=20, de_type='rand/1/bin'):

        self.model = model
        self.samples = samples
        self.input = input

        best_fobj, sol = DifferentialEvolution(
            func=self.eval_objective, lb=self.lb, ub=self.ub, callback=self.save_results, 
            max_generations=max_generations, pop_size=pop_size, de_type=de_type).run(debug=False)
        return best_fobj, sol, self.xk, self.fxk

    def save_results(self, x, fx, gx):
        self.xk.append(x)
        self.fxk.append(fx)

    def eval_objective(self, x):
        sim_values = self.model.get_simulated_samples(
            self.input, x, self.samples)

        sim_values_to_use = {}
        samples_to_use = {}
        for key in self.samples.keys():
            sim_values_to_use[key] = [sim_values[key][CB_INDEX],
                                      sim_values[key][CC_INDEX], sim_values[key][CD_INDEX]]
            samples_to_use[key] = [self.samples[key][CB_INDEX],
                                   self.samples[key][CC_INDEX], self.samples[key][CD_INDEX]]

        # SSE
        error = calculate_SSE(sim_values_to_use, samples_to_use)
        return error, []
