import numpy as np
from scipy.optimize import differential_evolution, minimize, Bounds, NonlinearConstraint


class ModelBasedOptimizer:
    def __init__(self, ub, lb, g, solver={'name': 'de', 'params': {'strategy': 'best1bin'}}, backoff=0.00):
        self.lb = lb
        self.ub = ub
        self.g = g
        self.solver = solver['name']
        self.solver_params = solver['params'] if 'params' in solver else {}
        self.backoff = backoff  # %

    def run(self, process_model, adaptation_strategy=None, x0=[]):
        bounds = Bounds(self.lb, self.ub)
        x_start = (np.asarray(self.ub) - np.asarray(self.lb)) / \
            2 if len(x0) == 0 else x0
        best_fobj, sol, nfev = self.optimize(
            process_model, adaptation_strategy, bounds, x_start)
        return best_fobj, sol, nfev

    def get_nlc(self, constraints, adaptation_strategy, x0):
        g_backoff = self.g * (1 - self.backoff)
        if((adaptation_strategy is not None) & (adaptation_strategy.type == 'modifier_adaptation')):
            adaptation = adaptation_strategy.get_adaptation(x0).get()
            if(hasattr(adaptation, 'trust_region_radius')):
                ub = np.append(g_backoff, adaptation.trust_region_radius)
                return NonlinearConstraint(constraints, -np.inf, ub)
        return NonlinearConstraint(constraints, -np.inf, g_backoff)

    def optimize(self, process_model, adaptation_strategy, bounds, x0):
        def constraints(x):
            g_model = process_model.get_constraints(x).reshape(-1,)
            if(adaptation_strategy is None):
                return g_model
            if(adaptation_strategy.type != 'modifier_adaptation'):
                return g_model
            adaptation = adaptation_strategy.get_adaptation(x).get()
            g_modifiers = g_model + adaptation.modifiers[1:].reshape(-1,)
            if(hasattr(adaptation, 'trust_region_radius')):
                return np.append(g_modifiers, np.linalg.norm(x - x0))
            return g_modifiers

        def func(x):
            f_model = process_model.get_objective(x)
            if(adaptation_strategy is None):
                return f_model
            if(adaptation_strategy.type != 'modifier_adaptation'):
                return f_model
            adaptation = adaptation_strategy.get_adaptation(x).get()
            return f_model + float(adaptation.modifiers[0])

        nlc = self.get_nlc(constraints, adaptation_strategy, x0)
        if(self.solver == 'de'):
            result = differential_evolution(
                func, bounds, maxiter=500, atol=0.0001, polish=False, constraints=nlc, **self.solver_params)
        elif(self.solver == 'sqp'):
            result = minimize(func, x0, method='SLSQP',
                              bounds=bounds, constraints=nlc, options={'disp': False, 'ftol': 0.000001, 'maxiter': 1000})

        # check for feasibility
        isUnfeasible = np.any(constraints(result.x) > nlc.ub)
        if(result.success == False & isUnfeasible):
            return None, [], result.nfev
        else:
            return result.fun, result.x, result.nfev
