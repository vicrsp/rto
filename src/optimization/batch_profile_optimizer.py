import numpy as np
from scipy.optimize import differential_evolution, minimize, Bounds, NonlinearConstraint


class BatchProfileOptimizer:
    def __init__(self, ub, lb, g, solver='de_scipy'):
        self.lb = lb
        self.ub = ub
        self.g = g
        self.solver = solver

    def optimize(self, ub, lb, process_model, ma_model, x0=[]):
        bounds = Bounds(lb, ub)

        def constraints(x):
            sim_results = process_model.simulate(x)
            modifiers = ma_model.get_modifiers(x)
            gm = modifiers[1:].reshape(-1,)
            g = process_model.get_constraints(x, sim_results).reshape(-1,) + gm
            return g

        def func(x):
            sim_results = process_model.simulate(x)
            modifiers = ma_model.get_modifiers(x)
            return process_model.get_objective(sim_results) + float(modifiers[0])

        nlc = NonlinearConstraint(constraints, -np.inf, self.g)
        if(self.solver == 'de_scipy'):
            result = differential_evolution(
                func, bounds, maxiter=50, popsize=20, polish=False, constraints=nlc)
            return result.fun, result.x
        elif(self.solver == 'slsqp_scipy'):
            x_start = (np.asarray(ub) - np.asarray(lb)) / 2 if len(x0) == 0 else x0
            result = minimize(func, x_start, method='SLSQP',
                              bounds=bounds, constraints=nlc, options={'disp': True})
            return result.fun, result.x

    def run(self, process_model, ma_model, lb, ub, x0=None):
        self.process_model = process_model
        self.xk = []
        self.fxk = []
        self.gk = []
        self.ma_model = ma_model
        best_fobj, sol = self.optimize(ub, lb, process_model, ma_model, x0)
        return best_fobj, sol

    def eval_objective(self, x):
        sim_results = self.process_model.simulate(x)
        modifiers = self.ma_model.get_modifiers(x)
        fm, gm = modifiers[0], modifiers[1:]
        fx = self.process_model.get_objective(sim_results) + fm
        g = self.process_model.get_constraints(x, sim_results) + gm - self.g
        return fx, g
