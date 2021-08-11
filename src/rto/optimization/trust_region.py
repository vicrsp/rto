import numpy as np
from scipy.optimize import differential_evolution, minimize, Bounds, NonlinearConstraint

from rto.optimization.optimizer import ModelBasedOptimizer


class ModifierAdaptationTrustRegionOptimizer(ModelBasedOptimizer):
    def __init__(self, ub, lb, g, solver, backoff):
        super().__init__(ub, lb, g, solver=solver, backoff=backoff)

    def normalize_input(self, u):
        return (u - self.lb)/(self.ub - self.lb)

    def denormalize_input(self, u):
        return u * (self.ub - self.lb) + self.lb

    def _get_nlc(self, constraints, radius):
        g_backoff = self.g * (1 - self.backoff)
        # return NonlinearConstraint(constraints, -np.inf, g_backoff)
        g_ub = np.append(g_backoff, radius)
        g_lb = np.append(np.array([-np.inf] * len(self.g)), 0)

        return NonlinearConstraint(constraints, g_lb, g_ub)

    def get_adjusted_bounds(self, xk, radius):
        # for each dimension, evalute if the current radius does not violate the box constraints
        # and adjust the upper/lower bound if necessary.
        lb_adjusted = np.maximum(np.zeros_like(self.lb), xk * (1-radius))
        ub_adjusted = np.minimum(np.ones_like(self.ub), xk * (1+radius))

        return Bounds(lb_adjusted, ub_adjusted)

    def optimize(self, process_model, adaptation_strategy, bounds, x0, f_best):
        def constraints(x):
            x_denormalized = self.denormalize_input(x)

            g_model = process_model.get_constraints(x_denormalized).reshape(-1,)
            adaptation = adaptation_strategy.get_adaptation(x_denormalized).get()
            modifiers = adaptation.modifiers[1:].reshape(-1,)

            g_modifiers = g_model + modifiers
            return np.append(g_modifiers, np.linalg.norm(x - x0))

        def func(x):
            x_denormalized = self.denormalize_input(x)
            f_model = process_model.get_objective(x_denormalized)
            adaptation = adaptation_strategy.get_adaptation(x_denormalized).get()
            return f_model + float(adaptation.modifiers[0])

        radius = adaptation_strategy.get_adaptation(x0).get().trust_region_radius
        adj_bounds = bounds # self.get_adjusted_bounds(x0, radius)
        nlc = self._get_nlc(constraints, radius)
        if(self.solver == 'de'):
            result = differential_evolution(
                func, bounds=adj_bounds, disp=False, maxiter=1000, atol=0.0001, polish=False, constraints=nlc, **self.solver_params)
        elif(self.solver == 'sqp'):
            result = minimize(func, x0, method='SLSQP',
                              bounds=adj_bounds, constraints=nlc, options={'disp': False, 'ftol': 0.000001, 'maxiter': 1000})

        # check for feasibility
        isUnfeasible = np.any(constraints(result.x) > nlc.ub)
        if(result.success == False & isUnfeasible):
            return None, [], result.nfev
        else:
            return result.fun, result.x, result.nfev