import numpy as np


class BatchProfileOptimizer:
    def __init__(self, ub, lb, g):
        self.lb = lb
        self.ub = ub
        self.g = np.asarray(g)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def run(self, process_model, ma_model):
        self.process_model = process_model
        self.xk = []
        self.fxk = []
        self.gk = []
        self.ma_model = ma_model

        best_fobj, sol = self.optimizer.run(func=self.eval_objective)
        return best_fobj, sol

    def save_results(self, x, fx, gx):
        self.xk.append(x)
        self.fxk.append(fx)
        self.gk.append(gx)

    def eval_objective(self, x):
        sim_results = self.process_model.simulate(x)
        modifiers = self.ma_model.get_modifiers(x) if self.ma_model.models != None else [0, np.zeros(len(self.g))]
        fm, gm = modifiers[0], modifiers[1:]

        fx = self.process_model.get_objective(sim_results) + fm
        g = self.process_model.get_constraints(x, sim_results) + gm - self.g
        return fx, g
