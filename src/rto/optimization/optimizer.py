import numpy as np
from scipy.optimize import differential_evolution, minimize, Bounds, NonlinearConstraint
from scipy.stats import norm

class ModelBasedOptimizer:
    def __init__(self, ub, lb, g, solver={'name': 'de', 'params': {'strategy': 'best1bin'}}, backoff=0.00):
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.g = g
        self.solver = solver['name']
        self.solver_params = solver['params'] if 'params' in solver else {}
        self.backoff = backoff  # %

    def run(self, process_model, adaptation_strategy=None, x0=[], f_best=[]):
        bounds = Bounds(np.zeros_like(self.lb), np.ones_like(self.ub))
        x_start = (np.asarray(self.ub) - np.asarray(self.lb)) / \
            2 if len(x0) == 0 else x0

        x_start_norm = self.normalize_input(x_start)
        best_fobj, sol, nfev = self.optimize(
            process_model, adaptation_strategy, bounds, x_start_norm, f_best)
        return best_fobj, self.denormalize_input(sol), nfev

    def normalize_input(self, u):
        return (u - self.lb)/(self.ub - self.lb)

    def denormalize_input(self, u):
        return u * (self.ub - self.lb) + self.lb

    def _get_nlc(self, constraints):
        g_backoff = self.g * (1 - self.backoff)
        return NonlinearConstraint(constraints, -np.inf, g_backoff)

    def optimize(self, process_model, adaptation_strategy, bounds, x0, f_best):
        def constraints(x):
            x_denormalized = self.denormalize_input(x)
            g_model = process_model.get_constraints(x_denormalized).reshape(-1,)
            return g_model

        def func(x):
            x_denormalized = self.denormalize_input(x)
            f_model = process_model.get_objective(x_denormalized)
            return f_model

        nlc = self._get_nlc(constraints)
        if(self.solver == 'de'):
            result = differential_evolution(
                func, bounds, maxiter=1000, atol=0.0001, polish=False, constraints=nlc, **self.solver_params)
        elif(self.solver == 'sqp'):
            result = minimize(func, x0, method='SLSQP',
                              bounds=bounds, constraints=nlc, options={'disp': False, 'ftol': 0.000001, 'maxiter': 1000})
        # check for feasibility
        isUnfeasible = np.any(constraints(result.x) > nlc.ub)
        if(result.success == False & isUnfeasible):
            return None, [], result.nfev
        else:
            return result.fun, result.x, result.nfev


class ModifierAdaptationOptimizer(ModelBasedOptimizer):
    def __init__(self, ub, lb, g, solver, backoff):
        super().__init__(ub, lb, g, solver=solver, backoff=backoff)

    def optimize(self, process_model, adaptation_strategy, bounds, x0, f_best):
        def constraints(x):
            x_denormalized = self.denormalize_input(x)
            g_model = process_model.get_constraints(x_denormalized).reshape(-1,)
            adaptation = adaptation_strategy.get_adaptation(x_denormalized).get()
            g_modifiers = g_model + adaptation.modifiers[1:].reshape(-1,)

            return g_modifiers

        def func(x):
            x_denormalized = self.denormalize_input(x)
            f_model = process_model.get_objective(x_denormalized)
            adaptation = adaptation_strategy.get_adaptation(x_denormalized).get()
            return f_model + float(adaptation.modifiers[0])

        nlc = self._get_nlc(constraints)
        if(self.solver == 'de'):
            result = differential_evolution(
                func, bounds, maxiter=1000, atol=0.0001, polish=False, constraints=nlc, **self.solver_params)
        elif(self.solver == 'sqp'):
            result = minimize(func, x0, method='SLSQP',
                              bounds=bounds, constraints=nlc, options={'disp': False, 'ftol': 0.000001, 'maxiter': 1000})
        # check for feasibility
        isUnfeasible = np.any(constraints(result.x) > nlc.ub)
        if(result.success == False & isUnfeasible):
            return None, [], result.nfev
        else:
            return result.fun, result.x, result.nfev



class ModifierAdaptationTrustRegionOptimizer(ModelBasedOptimizer):
    def __init__(self, ub, lb, g, solver, backoff):
        super().__init__(ub, lb, g, solver=solver, backoff=backoff)

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

class ModelBasedBayesianOptimizer(ModelBasedOptimizer):
    def __init__(self, ub, lb, g, solver={'name': 'de', 'params': {'strategy': 'best1bin'}}, backoff=0.00):
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.g = g
        self.solver = solver['name']
        self.solver_params = solver['params'] if 'params' in solver else {}
        self.backoff = backoff  # %

    def optimize(self, process_model, adaptation_strategy, bounds, x0, f_best):
        def func(x):
            x_denormalized = self.denormalize_input(x)
            adaptation = adaptation_strategy.get_adaptation(x_denormalized, return_std=True).get()
            g_model = process_model.get_constraints(x_denormalized).reshape(-1,)
            f_model = process_model.get_objective(x_denormalized)

            # calculate the expected improvement (EI) objective
            f_mod, std_mod = adaptation.modifiers[0]
            f_mod, std_mod = float(f_mod), float(std_mod)
            
            f_obj_mod = f_model + f_mod
            delta = f_best - f_obj_mod
            if std_mod == 0.0:
                z = 0.0
            else:
                z = (delta) / std_mod
            ei_fobj = -(std_mod * norm.pdf(z) + delta * norm.cdf(z))
            # calculate the expected improvement (EI) constraints
            # ei_constraints = np.prod([norm.cdf((g_model[i] + float(mean_g))/float(std_g)) for i, (mean_g, std_g) in enumerate(adaptation.modifiers[1:])])
            probs = np.array([norm.cdf((self.g[i] - (g_model[i] + float(mean_g)))/float(std_g)) for i, (mean_g, std_g) in enumerate(adaptation.modifiers[1:])])
            # ei_constraints = np.all(probs > 0.9) * 1
            ei_constraints = np.prod(probs)

            return ei_fobj * ei_constraints

        if(self.solver == 'de'):
            result = differential_evolution(
                func, bounds, maxiter=1000, atol=0.00001, polish=False, **self.solver_params)
        elif(self.solver == 'nm'):
            result = minimize(func, x0, method='Nelder-Mead',
                              bounds=bounds, options={'disp': False, 'fatol': 0.0001, 'maxiter': 1000})
        # check for feasibility
        if(result.success == False):
            return None, [], result.nfev
        else:
            return result.fun, result.x, result.nfev