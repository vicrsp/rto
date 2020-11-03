import numpy as np
from .solvers.de import DifferentialEvolution

class BatchProfileOptimizer:
    def __init__(self, ub, lb, g, solver='de/rand/1/bin'):
        self.lb = lb
        self.ub = ub
        self.g = np.asarray(g)
        self.solver = solver

    def set_optimizer(self, ub, lb):
        if(self.solver == 'de/rand/1/bin'):
            optimizer = DifferentialEvolution(
                lb=lb, ub=ub, callback=self.save_results, max_generations=100, pop_size=20, de_type='rand/1/bin')
            self.optimizer = optimizer
        else:
            self.optimizer = None

    def run(self, process_model, ma_model, lb, ub):
        self.process_model = process_model
        self.xk = []
        self.fxk = []
        self.gk = []
        self.ma_model = ma_model
        self.set_optimizer(ub, lb)

        best_fobj, sol = self.optimizer.run(func=self.eval_objective)
        return float(best_fobj), sol

    def save_results(self, x, fx, gx):
        self.xk.append(x)
        self.fxk.append(fx)
        self.gk.append(gx)

    def eval_objective(self, x):
        sim_results = self.process_model.simulate(x)
        modifiers = self.ma_model.get_modifiers(x)
        fm, gm = modifiers[0], modifiers[1:]
        fx = self.process_model.get_objective(sim_results) + fm
        g = self.process_model.get_constraints(x, sim_results) + gm - self.g
        return fx, g
