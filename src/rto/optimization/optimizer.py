import numpy as np
from scipy.optimize import differential_evolution, minimize, Bounds, NonlinearConstraint

class ModelBasedOptimizer:
    def __init__(self, ub, lb, g, solver={'name': 'de', 'params': {'strategy': 'best1bin'}}, backoff=0.00):
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.g = g
        self.solver = solver['name']
        self.solver_params = solver['params'] if 'params' in solver else {}
        self.backoff = backoff  # %

    def run(self, process_model, x0, **kwargs):
        bounds = Bounds(self.lb, self.ub)
        x_start = (np.asarray(self.ub) - np.asarray(self.lb)) / \
            2 if len(x0) == 0 else x0
        best_fobj, sol, nfev = self.optimize(process_model, bounds, x_start, **kwargs)
        return best_fobj, sol, nfev

    def _get_nlc(self, constraints):
        g_backoff = self.g * (1 - self.backoff)
        return NonlinearConstraint(constraints, -np.inf, g_backoff)

    def optimize(self, process_model, bounds, x0, **kwargs):
        def constraints(x):
            g_model = process_model.get_constraints(x).reshape(-1,)
            return g_model

        def func(x):
            f_model = process_model.get_objective(x)
            return f_model

        result = self._solve_nlp(bounds, x0, constraints, func)
        return self._get_solution(result, constraints)
        
    def _solve_nlp(self, bounds, x0, constraints, func):
        nlc = self._get_nlc(constraints)
        if(self.solver == 'de'):
            result = differential_evolution(
                func, bounds, maxiter=1000, atol=0.0001, polish=False, constraints=nlc, **self.solver_params)
        elif(self.solver == 'sqp'):
            result = minimize(func, x0, method='SLSQP',
                              bounds=bounds, constraints=nlc, options={'disp': False, 'ftol': 0.0001, 'maxiter': 1000})
                              
        return result
    
    def _get_solution(self, result, constraints):
        # check for feasibility
        isUnfeasible = np.any(constraints(result.x) > self.g)
        if(result.success == False & isUnfeasible):
            return None, [], result.nfev
        else:
            return result.fun, result.x, result.nfev


class ModifierAdaptationOptimizer(ModelBasedOptimizer):
    def __init__(self, ub, lb, g, solver, backoff):
        super().__init__(ub, lb, g, solver=solver, backoff=backoff)

    def optimize(self, process_model, bounds, x0, **kwargs):
        adaptation_strategy = kwargs.get('adaptation_strategy')
        if(adaptation_strategy is None):
            raise ValueError('Adaptation strategy must be informed for MA optimizer.')

        def constraints(x):
            g_model = process_model.get_constraints(x).reshape(-1,)
            adaptation = adaptation_strategy.get_adaptation(x.reshape(1,-1)).get()
            g_modifiers = g_model + adaptation.modifiers[1:].reshape(-1,)

            return g_modifiers

        def func(x):
            f_model = process_model.get_objective(x)
            adaptation = adaptation_strategy.get_adaptation(x.reshape(1,-1)).get()
            return f_model + float(adaptation.modifiers[0])

        result = self._solve_nlp(bounds, x0, constraints, func)
        return self._get_solution(result, constraints)
