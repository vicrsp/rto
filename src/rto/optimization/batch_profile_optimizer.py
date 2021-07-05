import numpy as np
from scipy.optimize import differential_evolution, minimize, Bounds, NonlinearConstraint

class BatchProfileOptimizer:
    def __init__(self, ub, lb, g, solver={'name': 'de_scipy_best1bin'}, backoff=0.00):
        self.lb = lb
        self.ub = ub
        self.g = g
        self.solver = solver['name']
        self.solver_params = solver['params'] if 'params' in solver else None
        self.backoff = backoff  # %

    def optimize(self, ub, lb, process_model, ma_model, x0=[]):
        bounds = Bounds(lb, ub)
        x_start = (np.asarray(ub) - np.asarray(lb)) / 2 if len(x0) == 0 else x0

        def constraints(x):
            modifiers = ma_model.get_modifiers(x)
            gm = modifiers[1:].reshape(-1,)
            g = process_model.get_constraints(x).reshape(-1,) + gm
            return g

        def func(x):
            modifiers = ma_model.get_modifiers(x)
            return process_model.get_objective(x) + float(modifiers[0])

        # add the backoff to constraints
        nlc = NonlinearConstraint(
            constraints, -np.inf, self.g * (1 - self.backoff))
        if(self.solver == 'de_scipy_best1bin'):
            result = differential_evolution(
                func, bounds, polish=False, constraints=nlc, atol=0.000001, strategy='best1bin')

            isUnfeasible = np.any(constraints(result.x) > self.g)
            if(result.success == False & isUnfeasible):
                return None, [], result.nfev
            else:
                return result.fun, result.x, result.nfev
        elif(self.solver == 'de_scipy_rand1bin'):
            result = differential_evolution(
                func, bounds, polish=False, constraints=nlc, atol=0.000001, strategy='rand1bin')

            isUnfeasible = np.any(constraints(result.x) > self.g)
            if(result.success == False & isUnfeasible):
                return None, [], result.nfev
            else:
                return result.fun, result.x, result.nfev
        elif(self.solver == 'slsqp_scipy'):
            result = minimize(func, x_start, method='SLSQP',
                              bounds=bounds, constraints=nlc, options={'disp': False, 'ftol': 0.000001, 'maxiter': 1000})

            # check for feasibility
            isUnfeasible = np.any(constraints(result.x) > self.g)
            if(result.success == False & isUnfeasible):
                return None, [], result.nfev 
            else:
                return result.fun, result.x, result.nfev
        elif(self.solver == 'de_scipy'):
            result = differential_evolution(
                func, bounds, maxiter=500, atol=0.0001, polish=False, constraints=nlc, **self.solver_params)

            isUnfeasible = np.any(constraints(result.x) > self.g)
            if(result.success == False & isUnfeasible):
                return None, [], result.nfev
            else:
                return result.fun, result.x, result.nfev
        elif(self.solver == 'de_sqp_hybrid'):
            result_de = differential_evolution(
                func, bounds, maxiter=50, atol=0.01, polish=False, constraints=nlc)

            result = minimize(func, result_de.x, method='SLSQP',
                              bounds=bounds, constraints=nlc, options={'disp': False, 'ftol': 0.000001})
            t_nfev = result_de.nfev + result.nfev
            isUnfeasible = np.any(constraints(result.x) > self.g)
            if(result.success == False & isUnfeasible):
                return None, [], t_nfev
            else:
                return result.fun, result.x, t_nfev

    def run(self, process_model, ma_model, x0=None):
        self.process_model = process_model
        self.ma_model = ma_model
        best_fobj, sol, nfev = self.optimize(
            self.ub, self.lb, process_model, ma_model, x0)
        return best_fobj, sol, nfev

    def eval_objective(self, x):
        sim_results = self.process_model.simulate(x)
        modifiers = self.ma_model.get_modifiers(x)
        fm, gm = modifiers[0], modifiers[1:]
        fx = self.process_model.get_objective(sim_results) + fm
        g = self.process_model.get_constraints(x, sim_results) + gm - self.g
        return fx, g