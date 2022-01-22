import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm

from rto.optimization.optimizer import ModelBasedOptimizer

class ModelBasedBayesianOptimizer(ModelBasedOptimizer):
    def __init__(self, ub, lb, g, solver={'name': 'de', 'params': {'strategy': 'best1bin'}}, backoff=0.00):
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.g = g
        self.solver = solver['name']
        self.solver_params = solver['params'] if 'params' in solver else {}
        self.backoff = backoff  # %
        self.c_delta = 0.01

    @staticmethod
    def ei_acquisition(f_model, adaptation, f_best):
        # calculate the expected improvement (EI) objective
        f_mod, std_mod = adaptation.modifiers[0]
        f_mod, std_mod = float(f_mod), float(std_mod)
        
        f_obj_mod = f_model + f_mod
        delta = f_best - f_obj_mod
        if std_mod == 0.0:
            z = 0.0
        else:
            z = (delta) / std_mod
        ei_fobj = (std_mod * norm.pdf(z) + delta * norm.cdf(z))

        return ei_fobj
    
    @staticmethod
    def constraint_probability(g, g_model, adaptation):
        probs = np.array([norm.cdf((g[i] - (g_model[i] + float(mean_g)))/float(std_g)) for i, (mean_g, std_g) in enumerate(adaptation.modifiers[1:])])
        return probs

    def _get_eic_function(self, process_model, adaptation_strategy, f_best):
        def eic(x):
            adaptation = adaptation_strategy.get_adaptation(x.reshape(1,-1), return_std=True).get()
            g_model = process_model.get_constraints(x).reshape(-1,)
            f_model = process_model.get_objective(x)
            # calculate the expected improvement with constraints (EIC) 
            probs = ModelBasedBayesianOptimizer.constraint_probability(self.g, g_model, adaptation)
            ei_constraints = np.prod(probs)
            # when no feasible solution was found yet, guide the search using the constraints only
            if f_best is None:
                return -ei_constraints
            ei_fobj = ModelBasedBayesianOptimizer.ei_acquisition(f_model, adaptation, f_best)
            return -ei_constraints * ei_fobj
        return eic

    def optimize(self, process_model, bounds, x0, **kwargs):
        adaptation_strategy = kwargs.get('adaptation_strategy')
        f_best = kwargs.get('f_best')

        func = self._get_eic_function(process_model, adaptation_strategy, f_best)
        result = self._solve_nlp(bounds, x0, func)
        return self._get_solution(result)

    def _solve_nlp(self, bounds, x0, func):
        if(self.solver == 'de'):
            result = differential_evolution(
                func, bounds, maxiter=1000, atol=1e-6, polish=False, **self.solver_params)
        elif(self.solver == 'nm'):
            result = minimize(func, x0, method='Nelder-Mead',
                              bounds=bounds, options={'disp': False, 'fatol': 1e-6, 'maxiter': 1000})
        elif(self.solver == 'sqp'):
            result = minimize(func, x0, method='SLSQP',
                              bounds=bounds, options={'disp': False, 'ftol': 1e-6, 'maxiter': 1000})                          
        return result
    
    def _get_solution(self, result):
        # check for feasibility
        if(result.success == False):
            return None, [], result.nfev
        else:
            return result.fun, result.x, result.nfev