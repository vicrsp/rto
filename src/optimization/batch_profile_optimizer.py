import numpy as np
from scipy.optimize import differential_evolution, minimize, approx_fprime, Bounds, NonlinearConstraint
import ipopt


class BatchProfileOptimizer:
    def __init__(self, ub, lb, g, solver='de_scipy'):
        self.lb = lb
        self.ub = ub
        self.g = g
        self.solver = solver

    def optimize(self, ub, lb, process_model, ma_model, x0=[]):
        bounds = Bounds(lb, ub)
        x_start = (np.asarray(ub) - np.asarray(lb)) / 2 if len(x0) == 0 else x0

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
            result = minimize(func, x_start, method='SLSQP',
                              bounds=bounds, constraints=nlc, options={'disp': True})
            return result.fun, result.x
        elif(self.solver == 'ipopt'):
            nlp = ipopt.problem(
            n=len(x_start),
            m=len(self.g),
            problem_obj=IPOPTProblem(process_model, ma_model),
            lb=lb,
            ub=ub,
            cl=[-np.inf] * len(self.g),
            cu=self.g
            )

            # IMPORTANT: need to use limited-memory / lbfgs here as we didn't give a valid hessian-callback
            nlp.addOption(b'hessian_approximation', b'limited-memory')
            nlp.addOption(b'print_level', 0)
            nlp.addOption(b'sb', b'no')
            nlp.addOption(b'max_cpu_time', 30.0)
            x, info = nlp.solve(x_start)
            return info['obj_val'], np.asarray(x)

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


class IPOPTProblem(object):
    def __init__(self, process_model, ma_model):
        self.process_model = process_model
        self.ma_model = ma_model
        self.num_diff_eps = 1e-8  # maybe tuning needed!

    def objective(self, x):
        # callback for objective
        sim_results = self.process_model.simulate(x)
        modifiers = self.ma_model.get_modifiers(x)
        return self.process_model.get_objective(sim_results) + float(modifiers[0])

    def constraint_0(self, x):
        sim_results = self.process_model.simulate(x)
        modifiers = self.ma_model.get_modifiers(x)
        gm = modifiers[1]
        g = self.process_model.get_constraints(x, sim_results)[0] + gm
        return g

    def constraint_1(self, x):
        sim_results = self.process_model.simulate(x)
        modifiers = self.ma_model.get_modifiers(x)
        gm = modifiers[2]
        g = self.process_model.get_constraints(x, sim_results)[1] + gm
        return g

    def constraints(self, x):
        # callback for constraints
        sim_results = self.process_model.simulate(x)
        modifiers = self.ma_model.get_modifiers(x)
        gm = modifiers[1:].reshape(-1,)
        g = self.process_model.get_constraints(
            x, sim_results).reshape(-1,) + gm
        return g

    def gradient(self, x):
        # callback for gradient
        return approx_fprime(x, self.objective, self.num_diff_eps)

    def jacobian(self, x):
        # callback for jacobian
        return np.concatenate([
            approx_fprime(x, self.constraint_0, self.num_diff_eps), 
            approx_fprime(x, self.constraint_1, self.num_diff_eps)])

    def hessian(self, x, lagrange, obj_factor):
        return False  # we will use quasi-newton approaches to use hessian-info
    
    # progress callback
    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):
        pass
        # if(iter_count % 10 == 0):
        #     print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))